#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""Tests for CTable persistent and in-memory indexing."""

import dataclasses
import shutil
import tempfile
import weakref
from pathlib import Path

import numpy as np
import pytest

import blosc2
import blosc2.indexing

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Row:
    id: int = blosc2.field(blosc2.int32())
    value: float = blosc2.field(blosc2.float64())
    category: int = blosc2.field(blosc2.int32())


def _make_table(n=100, persistent_path=None):
    """Return a CTable with *n* rows, optionally persistent at *persistent_path*."""
    if persistent_path is not None:
        t = blosc2.CTable(Row, urlpath=persistent_path, mode="w")
    else:
        t = blosc2.CTable(Row)
    # Bulk extend instead of n single appends (same data, ~100x faster to build).
    ids = np.arange(n, dtype=np.int32)
    t.extend({"id": ids, "value": ids * 1.5, "category": (ids % 5).astype(np.int32)})
    return t


# ---------------------------------------------------------------------------
# In-memory table tests
# ---------------------------------------------------------------------------


def test_create_index_in_memory():
    t = _make_table(50)
    idx = t.create_index("id")
    assert idx is not None
    assert idx.col_name == "id"
    assert not idx.stale
    assert len(t.indexes) == 1
    assert t.indexes[0].col_name == "id"


def test_create_index_in_memory_duplicate_raises():
    t = _make_table(20)
    t.create_index("id")
    with pytest.raises(ValueError, match="Index already exists"):
        t.create_index("id")


def test_drop_index_in_memory():
    t = _make_table(20)
    t.create_index("id")
    t.drop_index("id")
    assert len(t.indexes) == 0
    with pytest.raises(KeyError):
        t.index("id")


def test_drop_nonexistent_index_raises():
    t = _make_table(20)
    with pytest.raises(KeyError, match="No index found"):
        t.drop_index("id")


def test_drop_indexed_column_clears_catalog():
    t = _make_table(20)
    t.create_index("id")
    t.drop_column("id")
    assert [idx.col_name for idx in t.indexes] == []
    with pytest.raises(KeyError, match="No index found"):
        t.index("id")


@pytest.mark.heavy
def test_where_with_index_matches_scan_in_memory():
    t = _make_table(200)
    t.create_index("id")
    result_idx = t.where(t["id"] > 100)
    # Drop index to force scan
    t.drop_index("id")
    result_scan = t.where(t["id"] > 100)
    ids_idx = sorted(int(v) for v in result_idx["id"][:])
    ids_scan = sorted(int(v) for v in result_scan["id"][:])
    assert ids_idx == ids_scan


@pytest.mark.heavy
def test_indexed_where_view_sort_by_reuses_cached_live_positions(monkeypatch):
    t = _make_table(200)
    t.create_index("id", kind=blosc2.IndexKind.FULL)

    view = t.where(t["id"] > 100, columns=["id", "value"])
    assert view._cached_live_positions is not None

    def fail_iter_live_positions_chunks():
        raise AssertionError("sort_by() should reuse cached live positions")

    monkeypatch.setattr(view, "_iter_live_positions_chunks", fail_iter_live_positions_chunks)
    sorted_view = view.sort_by("id")

    assert sorted_view["id"][:].tolist() == list(range(101, 200))


def test_create_expression_index_in_memory():
    t = _make_table(50)
    idx = t.create_index(expression="value * category", kind=blosc2.IndexKind.FULL, name="vc")
    assert idx.kind == "full"
    assert t.index(expression="value * category").name == "vc"
    assert t.index(name="vc").name == "vc"


@pytest.mark.heavy
def test_where_with_expression_index_matches_scan_in_memory():
    t = _make_table(200)
    t.create_index(expression="value * category", kind=blosc2.IndexKind.FULL, name="vc")
    result_idx = t.where((t._cols["value"] * t._cols["category"]) >= 150)
    t.drop_index(expression="value * category")
    result_scan = t.where((t._cols["value"] * t._cols["category"]) >= 150)
    ids_idx = sorted(int(v) for v in result_idx["id"][:])
    ids_scan = sorted(int(v) for v in result_scan["id"][:])
    assert ids_idx == ids_scan


def test_bool_column_composes_naturally_in_where():
    @dataclasses.dataclass
    class BoolRow:
        sensor_id: int = blosc2.field(blosc2.int32())
        region: str = blosc2.field(blosc2.string(max_length=8), default="")
        active: bool = blosc2.field(blosc2.bool(), default=True)

    t = blosc2.CTable(BoolRow)
    for i in range(20):
        t.append([i, "north" if i % 4 == 0 else "south", i % 2 == 0])

    result = t.where((t["sensor_id"] >= 8) & t["active"] & (t["region"] == "north"))
    assert sorted(int(v) for v in result["sensor_id"][:]) == [8, 12, 16]

    result_bare = t.where(t["active"])
    assert sorted(int(v) for v in result_bare["sensor_id"][:]) == list(range(0, 20, 2))


def test_rebuild_index_in_memory():
    t = _make_table(30)
    t.create_index("id")
    t.append([999, 999.0, 4])  # marks stale
    assert t.index("id").stale
    idx2 = t.rebuild_index("id")
    assert not idx2.stale
    result = t.where(t["id"] == 999)
    assert len(result) == 1


def test_stale_on_append_in_memory():
    t = _make_table(20)
    t.create_index("id")
    t.append([100, 100.0, 0])
    assert t.index("id").stale


def test_stale_on_extend_in_memory():
    t = _make_table(20)
    t.create_index("id")
    t.extend([[101, 101.0, 0], [102, 102.0, 1]])
    assert t.index("id").stale


def test_stale_on_column_setitem_in_memory():
    t = _make_table(20)
    t.create_index("id")
    t["id"][0] = 999
    assert t.index("id").stale


def test_stale_on_column_assign_in_memory():
    t = _make_table(20)
    t.create_index("id")
    t["id"].assign(np.arange(20, dtype=np.int32))
    assert t.index("id").stale


def test_delete_bumps_visibility_epoch_not_stale_in_memory():
    t = _make_table(20)
    t.create_index("id")
    t.delete(0)
    idx = t.index("id")
    # delete should NOT mark stale (only bumps visibility_epoch)
    assert not idx.stale
    _, vis_e = t._storage.get_epoch_counters()
    assert vis_e >= 1


def test_stale_fallback_to_scan_in_memory():
    t = _make_table(50)
    t.create_index("id")
    t.append([200, 200.0, 0])  # marks stale
    # Query should still work (falls back to scan)
    result = t.where(t["id"] > 40)
    ids = sorted(int(v) for v in result["id"][:])
    assert 200 in ids
    assert 41 in ids


def test_compact_index_in_memory():
    t = _make_table(50, persistent_path=None)
    t.create_index("id", kind=blosc2.IndexKind.FULL)
    # compact_index should not raise for full indexes
    t.compact_index("id")


@pytest.mark.heavy
def test_multi_column_conjunction_uses_multiple_indexes_in_memory():
    t = _make_table(200)
    t.create_index("id", kind=blosc2.IndexKind.FULL)
    t.create_index("category", kind=blosc2.IndexKind.FULL)
    expr = (t["id"] >= 50) & (t["id"] < 120) & (t["category"] == 3)
    result_idx = t.where(expr)
    t.drop_index("id")
    t.drop_index("category")
    result_scan = t.where(expr)
    ids_idx = sorted(int(v) for v in result_idx["id"][:])
    ids_scan = sorted(int(v) for v in result_scan["id"][:])
    assert ids_idx == ids_scan


def test_full_index_large_ctable_column_matches_scan_in_memory():
    @dataclasses.dataclass
    class SensorRow:
        sensor_id: int = blosc2.field(blosc2.int32())

    n = 300_000
    rng = np.random.default_rng(42)
    data = np.empty(n, dtype=np.dtype([("sensor_id", np.int32)]))
    data["sensor_id"] = rng.integers(0, n // 10, size=n, dtype=np.int32)

    t = blosc2.CTable(SensorRow, expected_size=n)
    t.extend(data)
    t.create_index("sensor_id", kind=blosc2.IndexKind.FULL)

    result_idx = t.where(t["sensor_id"] > 29_000)
    t.drop_index("sensor_id")
    result_scan = t.where(t["sensor_id"] > 29_000)

    ids_idx = np.sort(np.asarray(result_idx["sensor_id"][:]))
    ids_scan = np.sort(np.asarray(result_scan["sensor_id"][:]))
    assert np.array_equal(ids_idx, ids_scan)


# ---------------------------------------------------------------------------
# Persistent table tests
# ---------------------------------------------------------------------------


@pytest.fixture
def tmpdir():
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


def test_create_index_persistent(tmpdir):
    path = str(tmpdir / "table.b2d")
    t = _make_table(50, persistent_path=path)
    idx = t.create_index("id")
    assert not idx.stale
    # Sidecar directory must exist
    index_dir = Path(path) / "_indexes" / "id"
    assert index_dir.exists()
    # At least one .b2nd sidecar file
    sidecars = list(index_dir.glob("**/*.b2nd"))
    assert sidecars, "No sidecar .b2nd files found"


def test_create_index_persistent_does_not_cache_sidecar_handles(tmpdir):
    import blosc2.indexing as indexing

    path = str(tmpdir / "table.b2d")
    t = _make_table(50, persistent_path=path)
    t.create_index("id", kind=blosc2.IndexKind.FULL)

    cached = [
        key
        for key in indexing._SIDECAR_HANDLE_CACHE
        if key[0][0] == "persistent" and str(tmpdir) in key[0][1]
    ]
    assert cached == []


def test_persistent_ctable_releases_immediately_without_gc(tmpdir):
    path = str(tmpdir / "table.b2d")

    def build_table():
        t = _make_table(50, persistent_path=path)
        t.create_index("id", kind=blosc2.IndexKind.FULL)
        return weakref.ref(t)

    table_ref = build_table()
    assert table_ref() is None


def test_catalog_survives_reopen(tmpdir):
    path = str(tmpdir / "table.b2d")
    t = _make_table(30, persistent_path=path)
    t.create_index("id")
    del t  # close

    t2 = blosc2.open(path, mode="r")
    idxs = t2.indexes
    assert len(idxs) == 1
    assert idxs[0].col_name == "id"
    assert not idxs[0].stale


@pytest.mark.heavy
def test_index_catalog_cached_per_opened_ctable(tmpdir, monkeypatch):
    path = str(tmpdir / "table.b2d")
    t = _make_table(200, persistent_path=path)
    t.create_index("id", kind=blosc2.IndexKind.FULL)
    del t

    with blosc2.open(path, mode="r") as t2:
        calls = 0
        original = t2._storage.load_index_catalog

        def wrapped_load_index_catalog():
            nonlocal calls
            calls += 1
            return original()

        monkeypatch.setattr(t2._storage, "load_index_catalog", wrapped_load_index_catalog)

        first = t2.where(t2["id"] > 100, columns=["id", "value"]).sort_by("id")
        second = t2.where(t2["id"] > 150, columns=["id", "value"]).sort_by("id")
        idxs = t2.indexes

        assert first["id"][:].tolist() == list(range(101, 200))
        assert second["id"][:].tolist() == list(range(151, 200))
        assert len(idxs) == 1
        assert calls == 1


@pytest.mark.heavy
def test_where_with_index_matches_scan_persistent(tmpdir):
    path = str(tmpdir / "table.b2d")
    t = _make_table(200, persistent_path=path)
    t.create_index("id")
    result_idx = t.where(t["id"] > 150)

    t.drop_index("id")
    result_scan = t.where(t["id"] > 150)

    ids_idx = sorted(int(v) for v in result_idx["id"][:])
    ids_scan = sorted(int(v) for v in result_scan["id"][:])
    assert ids_idx == ids_scan


@pytest.mark.heavy
def test_relative_b2d_ctable_index_sidecars_survive_reopen(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    t = _make_table(200, persistent_path="table.b2d")
    t.create_index("id", kind=blosc2.IndexKind.BUCKET)
    t.close()

    reopened = blosc2.open("table.b2d", mode="r")
    result = reopened.where(reopened["id"] > 150)

    assert sorted(int(v) for v in result["id"][:]) == list(range(151, 200))


@pytest.mark.heavy
def test_persistent_index_drop_releases_sidecars_without_gc(tmpdir):
    import gc

    def run_query_and_drop():
        path = str(tmpdir / "table.b2d")
        t = _make_table(200, persistent_path=path)
        t.create_index("id")
        result = t.where(t["id"] > 150)
        ids = sorted(int(v) for v in result["id"][:])
        assert ids == list(range(151, 200))
        t.drop_index("id")

    run_query_and_drop()

    sidecars = [
        obj
        for obj in gc.get_objects()
        if isinstance(obj, blosc2.NDArray)
        and obj.urlpath
        and str(tmpdir) in obj.urlpath
        and "__index__" in obj.urlpath
    ]
    assert sidecars == []


def test_drop_index_persistent(tmpdir):
    path = str(tmpdir / "table.b2d")
    t = _make_table(30, persistent_path=path)
    t.create_index("id")
    t.drop_index("id")
    assert len(t.indexes) == 0
    index_dir = Path(path) / "_indexes" / "id"
    # After drop, index dir should be gone (or empty)
    sidecars = list(index_dir.glob("**/*.b2nd")) if index_dir.exists() else []
    assert sidecars == []


def test_expression_index_persistent_roundtrip(tmpdir):
    path = str(tmpdir / "table.b2d")
    t = _make_table(50, persistent_path=path)
    t.create_index(expression="value * category", kind=blosc2.IndexKind.FULL, name="vc")

    reopened = blosc2.CTable.open(path, mode="r")
    idx = reopened.index(expression="value * category")
    assert idx.kind == "full"
    result = reopened.where((reopened._cols["value"] * reopened._cols["category"]) >= 60)
    assert len(result) > 0


def test_sort_by_computed_column_with_expression_full_index():
    t = _make_table(40)
    t.add_computed_column("score", "value * category")
    t.create_index(expression="value * category", kind=blosc2.IndexKind.FULL, name="score_expr")

    sorted_t = t.sort_by("score")
    expected = np.sort(np.asarray(t._computed_cols["score"]["lazy"][:])[t._valid_rows[:]])
    np.testing.assert_allclose(sorted_t["score"][:], expected)


def test_sort_by_stored_column_with_full_index():
    t = _make_table(40)
    t.create_index("id", kind=blosc2.IndexKind.FULL)
    sorted_t = t.sort_by("id", ascending=False)
    np.testing.assert_array_equal(sorted_t["id"][:], np.arange(39, -1, -1, dtype=np.int32))


def test_drop_index_persistent_catalog_cleared(tmpdir):
    path = str(tmpdir / "table.b2d")
    t = _make_table(30, persistent_path=path)
    t.create_index("id")
    t.drop_index("id")
    del t

    t2 = blosc2.open(path, mode="r")
    assert len(t2.indexes) == 0


def test_drop_indexed_column_removes_persistent_sidecars(tmpdir):
    path = str(tmpdir / "table.b2d")
    t = _make_table(30, persistent_path=path)
    t.create_index("id")
    t.drop_column("id")
    assert len(t.indexes) == 0
    assert not (Path(path) / "_indexes" / "id").exists()


def test_rebuild_index_persistent(tmpdir):
    path = str(tmpdir / "table.b2d")
    t = _make_table(50, persistent_path=path)
    t.create_index("id")
    t.append([500, 750.0, 2])  # marks stale
    assert t.index("id").stale
    idx2 = t.rebuild_index("id")
    assert not idx2.stale
    result = t.where(t["id"] == 500)
    assert len(result) == 1


def test_compact_index_persistent(tmpdir):
    path = str(tmpdir / "table.b2d")
    t = _make_table(50, persistent_path=path)
    t.create_index("id", kind=blosc2.IndexKind.FULL)
    t.compact_index("id")
    # Query should still work after compact
    result = t.where(t["id"] > 40)
    ids = sorted(int(v) for v in result["id"][:])
    expected = list(range(41, 50))
    assert ids == expected


def test_stale_on_append_persistent(tmpdir):
    path = str(tmpdir / "table.b2d")
    t = _make_table(20, persistent_path=path)
    t.create_index("id")
    t.append([200, 300.0, 1])
    assert t.index("id").stale


def test_stale_persists_after_reopen(tmpdir):
    path = str(tmpdir / "table.b2d")
    t = _make_table(20, persistent_path=path)
    t.create_index("id")
    t.append([200, 300.0, 1])  # marks stale
    del t

    t2 = blosc2.open(path, mode="r")
    assert t2.index("id").stale


def test_delete_bumps_visibility_epoch_persistent(tmpdir):
    path = str(tmpdir / "table.b2d")
    t = _make_table(20, persistent_path=path)
    t.create_index("id")
    t.delete(0)
    idx = t.index("id")
    # delete should NOT mark index stale
    assert not idx.stale
    _, vis_e = t._storage.get_epoch_counters()
    assert vis_e >= 1


@pytest.mark.heavy
def test_query_after_reopen_persistent(tmpdir):
    path = str(tmpdir / "table.b2d")
    t = _make_table(100, persistent_path=path)
    t.create_index("id")
    del t

    t2 = blosc2.open(path, mode="r")
    result = t2.where(t2["id"] > 90)
    ids = sorted(int(v) for v in result["id"][:])
    assert ids == list(range(91, 100))


def test_rename_indexed_column_rebuilds_catalog_persistent(tmpdir):
    path = str(tmpdir / "table.b2d")
    t = _make_table(40, persistent_path=path)
    t.create_index("id")
    t.rename_column("id", "newid")
    assert [idx.col_name for idx in t.indexes] == ["newid"]
    assert not (Path(path) / "_indexes" / "id").exists()
    assert (Path(path) / "_indexes" / "newid").exists()
    result = t.where(t["newid"] > 35)
    assert sorted(int(v) for v in result["newid"][:]) == [36, 37, 38, 39]


# ---------------------------------------------------------------------------
# View tests
# ---------------------------------------------------------------------------


def test_view_cannot_create_index():
    t = _make_table(20)
    view = t.where(t["id"] > 5)
    with pytest.raises(ValueError, match="view"):
        view.create_index("id")


def test_view_cannot_drop_index():
    t = _make_table(20)
    t.create_index("id")
    view = t.where(t["id"] > 5)
    with pytest.raises(ValueError, match="view"):
        view.drop_index("id")


def test_view_cannot_rebuild_index():
    t = _make_table(20)
    t.create_index("id")
    view = t.where(t["id"] > 5)
    with pytest.raises(ValueError, match="view"):
        view.rebuild_index("id")


def test_view_cannot_compact_index():
    t = _make_table(20)
    t.create_index("id")
    view = t.where(t["id"] > 5)
    with pytest.raises(ValueError, match="view"):
        view.compact_index("id")


@pytest.mark.heavy
def test_view_query_uses_root_index():
    t = _make_table(200)
    t.create_index("id")
    # Query on the original table
    result_direct = t.where(t["id"] > 180)
    ids_direct = sorted(int(v) for v in result_direct["id"][:])
    assert ids_direct == list(range(181, 200))


def test_malformed_catalog_entry_raises_clear_error():
    t = _make_table(20)
    t._storage.save_index_catalog({"id": {"kind": "bucket"}})
    with pytest.raises(ValueError, match="Malformed index metadata"):
        t.where(t["id"] > 5)


# ---------------------------------------------------------------------------
# index() and indexes property
# ---------------------------------------------------------------------------


def test_index_lookup_missing_raises():
    t = _make_table(10)
    with pytest.raises(KeyError):
        t.index("nonexistent")


def test_indexes_empty_on_new_table():
    t = _make_table(10)
    assert t.indexes == []


def test_indexes_multiple_columns():
    t = _make_table(30)
    t.create_index("id")
    t.create_index("category")
    assert len(t.indexes) == 2
    col_names = {idx.col_name for idx in t.indexes}
    assert col_names == {"id", "category"}


def test_indexed_ctable_b2z_double_open_append_no_corruption(tmp_path):
    """Opening an indexed CTable .b2z in append mode twice must not corrupt it.

    Regression test: GC of a CTable opened from .b2z was calling close() →
    to_b2z() even when nothing was modified, overwriting the archive with a
    near-empty ZIP that broke subsequent opens.
    """
    path = str(tmp_path / "indexed.b2z")
    b2d_path = str(tmp_path / "indexed.b2d")

    t = _make_table(50, persistent_path=b2d_path)
    t.create_index("id")
    t._storage._store.to_b2z(filename=path, overwrite=True)
    t._storage._store.close()
    shutil.rmtree(b2d_path)

    # First open without explicit close — GC must not corrupt the archive
    t1 = blosc2.open(path, mode="a")
    assert t1.nrows == 50
    assert len(t1.indexes) == 1
    del t1  # triggers __del__; must NOT repack/corrupt

    # Second open must succeed and see correct data
    t2 = blosc2.open(path, mode="a")
    assert t2.nrows == 50
    assert len(t2.indexes) == 1
    del t2


def test_indexing_purges_stale_persistent_caches():
    import blosc2.indexing as indexing

    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "table.b2d")
        t = _make_table(50, persistent_path=path)
        t.create_index("id")
        _ = t.where(t["id"] > 10)
        t.close()

        persistent_scope = ("persistent", str(Path(path).resolve()))
        indexing._PERSISTENT_INDEXES[persistent_scope] = {"version": 1, "indexes": {}}
        indexing._DATA_CACHE[(persistent_scope, "token", "partial", "offsets")] = np.arange(
            3, dtype=np.int64
        )
        indexing._SIDECAR_HANDLE_CACHE[(persistent_scope, "token", "partial_handle", "offsets")] = object()
        indexing._QUERY_CACHE_STORE_HANDLES[str(Path(tmpdir) / "query-cache.b2frame")] = object()
        indexing._GATHER_MMAP_HANDLES[str(Path(tmpdir) / "gather-cache.b2nd")] = object()

    indexing._purge_stale_persistent_caches()

    assert all(tmpdir not in key[1] for key in indexing._PERSISTENT_INDEXES if key[0] == "persistent")
    assert all(tmpdir not in key[0][1] for key in indexing._DATA_CACHE if key[0][0] == "persistent")
    assert all(
        tmpdir not in key[0][1] for key in indexing._SIDECAR_HANDLE_CACHE if key[0][0] == "persistent"
    )
    assert all(tmpdir not in path for path in indexing._QUERY_CACHE_STORE_HANDLES)
    assert all(tmpdir not in path for path in indexing._GATHER_MMAP_HANDLES)


def test_indexing_purge_tolerates_reentrant_sidecar_handle_cache_mutation(monkeypatch):
    import blosc2.indexing as indexing

    stale_scope = ("persistent", "/tmp/stale-index.b2nd")
    stale_key = (stale_scope, "token", "partial_handle", "offsets")
    injected_key = (("memory", 12345), "token", "partial_handle", "offsets")
    sentinel = object()
    original_exists = indexing._persistent_cache_path_exists

    indexing._SIDECAR_HANDLE_CACHE[stale_key] = sentinel

    def mutating_exists(path):
        indexing._SIDECAR_HANDLE_CACHE[injected_key] = sentinel
        if path == stale_scope[1]:
            return False
        return original_exists(path)

    monkeypatch.setattr(indexing, "_persistent_cache_path_exists", mutating_exists)

    indexing._purge_stale_persistent_caches()

    assert stale_key not in indexing._SIDECAR_HANDLE_CACHE
    indexing._SIDECAR_HANDLE_CACHE.pop(injected_key, None)


def test_summary_index_compact_store_no_cross_column_confusion(tmp_path):
    """Regression: a SUMMARY index on one column of a compact (.b2z) store must
    not be applied to a *different* column's predicate.

    In compact stores every column shares one urlpath, so the urlpath-keyed
    index store could hand back the indexed column's descriptor for an unrelated
    column.  After column alignment siblings also share shape/chunks, defeating
    the only guard that previously distinguished them.  A predicate impossible
    for the indexed column's value range (``c < 0`` while the indexed column
    ``a`` is non-negative) was then applied to ``a``'s per-segment min/max,
    wrongly pruning every segment and silently returning 0 rows.
    """

    @dataclasses.dataclass
    class Aligned:
        # Force a small shared chunk grid so the SUMMARY index has several
        # segments and all columns are chunk-aligned (same shape and chunks).
        a: float = blosc2.field(blosc2.float32(), chunks=(1000,), blocks=(250,))
        b: float = blosc2.field(blosc2.float32(), chunks=(1000,), blocks=(250,))
        c: float = blosc2.field(blosc2.float64(), chunks=(1000,), blocks=(250,))

    n = 10_000
    rng = np.random.default_rng(0)
    a = (rng.random(n) * 100).astype(np.float32)  # always >= 0 (indexed column)
    b = (rng.random(n) + 1).astype(np.float32)  # always > 0
    c = rng.random(n) * 200 - 100  # spans negatives, so "c < 0" matches real rows

    t = blosc2.CTable(Aligned)
    t.extend(list(zip(a.tolist(), b.tolist(), c.tolist(), strict=True)))
    path = str(tmp_path / "aligned.b2z")
    t.to_b2z(path)

    with blosc2.open(path, mode="a") as w:
        w.create_index("a", kind=blosc2.IndexKind.SUMMARY)

    expected = int(((a > 90) & (b > 0) & (c < 0)).sum())
    assert expected > 0  # the predicate must actually match real rows

    with blosc2.open(path) as r:
        got = r.where((r.a > 90) & (r.b > 0) & (r.c < 0)).nrows
    assert got == expected, f"index returned {got}, expected {expected} (scan)"


def test_sidecar_handle_cache_no_cross_column_collision(tmp_path):
    """Regression: in a compact (.b2z) multi-column store, reading the SUMMARY
    block sidecar handle for each column must return *that* column's data, not
    a sibling's.

    The process-wide ``_SIDECAR_HANDLE_CACHE`` was keyed only by
    ``(_array_key, token, category, name)``.  In compact stores all columns
    share the same urlpath → same ``_array_key``, so sibling columns collided
    on a single cache entry and every column received the handle opened first.
    Including ``path`` in the key fixes this.
    """

    @dataclasses.dataclass
    class Distinct:
        a: float = blosc2.field(blosc2.float64(), chunks=(500,), blocks=(250,))
        b: float = blosc2.field(blosc2.float64(), chunks=(500,), blocks=(250,))

    n = 1000
    rng = np.random.default_rng(42)
    # Non-overlapping ranges so we can trivially tell columns apart by max.
    a = (rng.random(n) * 10).astype(np.float64)  # max ≈ 10
    b = (rng.random(n) * 100 + 100).astype(np.float64)  # max ≈ 200, min ≥ 100

    t = blosc2.CTable(Distinct)
    t.extend(list(zip(a.tolist(), b.tolist(), strict=True)))
    path = str(tmp_path / "distinct.b2z")
    t.to_b2z(path)

    # Build SUMMARY indexes on both columns.
    with blosc2.open(path, mode="a") as w:
        w.create_index("a", kind=blosc2.IndexKind.SUMMARY)
        w.create_index("b", kind=blosc2.IndexKind.SUMMARY)

    with blosc2.open(path) as r:
        catalog = dict(r._get_index_catalog())
        nd_a = r["a"]._raw_col
        nd_b = r["b"]._raw_col

        summary_a = blosc2.indexing._open_level_summary_handle(nd_a, catalog["a"], "block")
        summary_b = blosc2.indexing._open_level_summary_handle(nd_b, catalog["b"], "block")

    max_a = summary_a["max"].max()
    max_b = summary_b["max"].max()

    # Without the fix, both handles return the same (first-opened) column's data
    # because the cache key collides.  With the fix they must differ.
    assert max_a != max_b, f"cross-column collision: both max values are {max_a}"
    assert max_a < 11, f"column a max {max_a} outside expected range [0, 10]"
    assert 100 <= max_b <= 200, f"column b max {max_b} outside expected range [100, 200]"


@dataclasses.dataclass
class _GranRow:
    # Small explicit grid so the SUMMARY index spans several chunks/blocks.
    v: float = blosc2.field(blosc2.float64(), chunks=(1000,), blocks=(250,))


def _make_gran_table(n=5000):
    rng = np.random.default_rng(1)
    t = blosc2.CTable(_GranRow)
    t.extend([(x,) for x in (rng.random(n) * 100).tolist()])
    return t, rng


def test_summary_index_defaults_to_block_granularity():
    t, _ = _make_gran_table()
    t.create_index("v", kind=blosc2.IndexKind.SUMMARY)
    levels = list(dict(t._get_index_catalog())["v"]["levels"].keys())
    assert levels == ["block"]


@pytest.mark.heavy
@pytest.mark.parametrize("granularity", ["chunk", "block"])
def test_summary_index_granularity_override(granularity):
    t, rng = _make_gran_table()
    t.create_index("v", kind=blosc2.IndexKind.SUMMARY, granularity=granularity)
    levels = list(dict(t._get_index_catalog())["v"]["levels"].keys())
    assert levels == [granularity]
    # Correctness must hold regardless of granularity.
    v = t["v"][:]
    expected = int((v > 95).sum())
    assert t.where(t.v > 95).nrows == expected


@dataclasses.dataclass
class _IncrRow:
    # Several chunks/blocks so the summary spans many segments; mixed dtypes.
    f: float = blosc2.field(blosc2.float32(null_value=float("nan")), chunks=(2000,), blocks=(500,))
    i: int = blosc2.field(blosc2.int64(), chunks=(2000,), blocks=(500,))


def _build_incr_data(n=9000):
    rng = np.random.default_rng(7)
    f = (rng.standard_normal(n) * 50).astype(np.float32)
    f[rng.integers(0, n, n // 100)] = np.nan  # exercise NaN flags
    i = rng.integers(-1000, 1000, n).astype(np.int64)
    return f, i


def _summary_sidecars(table):
    """Return {col: structured summary array} for all SUMMARY block sidecars."""
    out = {}
    for name, desc in dict(table._get_index_catalog()).items():
        if desc.get("kind") != "summary":
            continue
        side = blosc2.indexing._open_sidecar_file(desc["levels"]["block"]["path"])
        out[name] = side[:]
    return out


def test_incremental_summary_matches_ooc_build(tmp_path):
    """The incremental per-block accumulator (folded during the write phase)
    must produce SUMMARY sidecars byte-identical to the out-of-core
    decompress-and-recompute path, including NaN flags."""
    f, i = _build_incr_data()

    # Accumulator path: extend in one shot, close (uses precomputed summaries).
    acc_path = str(tmp_path / "acc.b2z")
    with blosc2.CTable(_IncrRow, urlpath=acc_path, mode="w") as t:
        t.extend({"f": f, "i": i})
    acc = _summary_sidecars(blosc2.open(acc_path))

    # Reference path: force the OOC builder by disabling the precomputed hook.
    ooc_path = str(tmp_path / "ooc.b2z")
    import blosc2.ctable as _ct

    orig = _ct.CTable._precomputed_summary_for
    try:
        _ct.CTable._precomputed_summary_for = lambda self, name: None
        with blosc2.CTable(_IncrRow, urlpath=ooc_path, mode="w") as t:
            t.extend({"f": f, "i": i})
    finally:
        _ct.CTable._precomputed_summary_for = orig
    ooc = _summary_sidecars(blosc2.open(ooc_path))

    assert set(acc) == set(ooc) == {"f", "i"}
    for name in acc:
        a, b = acc[name], ooc[name]
        assert len(a) == len(b)
        assert np.array_equal(a["flags"], b["flags"])
        assert np.allclose(a["min"], b["min"], equal_nan=True)
        assert np.allclose(a["max"], b["max"], equal_nan=True)


def test_incremental_summary_invalidated_by_inplace_update(tmp_path):
    """An in-place column write before close must invalidate the accumulator so
    the builder falls back to a correct full rescan."""
    f, i = _build_incr_data(n=4000)
    path = str(tmp_path / "upd.b2z")
    with blosc2.CTable(_IncrRow, urlpath=path, mode="w") as t:
        t.extend({"f": f, "i": i})
        # Mutate a value through the column handle (the bypass path) and update
        # the local reference so the expected result reflects the change.
        t["i"][0] = 999_999
        i = i.copy()
        i[0] = 999_999
        acc = t.__dict__.get("_summary_accumulators", {}).get("i")
        assert acc is None or not acc.valid  # accumulator disabled for "i"

    with blosc2.open(path) as r:
        expected = int((i > 1000).sum())
        assert r.where(r.i > 1000).nrows == expected


def test_summary_granularity_rejects_invalid_value():
    t, _ = _make_gran_table(n=10)
    with pytest.raises(ValueError, match="granularity must be one of"):
        t.create_index("v", kind=blosc2.IndexKind.SUMMARY, granularity="bogus")


def test_granularity_only_valid_for_summary():
    t, _ = _make_gran_table(n=10)
    with pytest.raises(ValueError, match=r"only supported for kind=IndexKind\.SUMMARY"):
        t.create_index("v", kind=blosc2.IndexKind.BUCKET, granularity="block")


@pytest.mark.heavy
@pytest.mark.parametrize("threshold", [5.0, 50.0, 99.0, 99.99])
def test_summary_cost_gate_correctness_across_selectivity(threshold):
    """The SUMMARY cost gate may use the index (selective query) or fall back to
    a scan (broad query); both branches must return scan-correct results."""
    t, _ = _make_gran_table(n=6000)
    t.create_index("v", kind=blosc2.IndexKind.SUMMARY)  # block granularity
    v = t["v"][:]
    expected = int((v > threshold).sum())
    assert t.where(t.v > threshold).nrows == expected


@dataclasses.dataclass
class _TwoColRow:
    a: float = blosc2.field(blosc2.float64(), chunks=(50000,), blocks=(2048,))
    b: float = blosc2.field(blosc2.float64(), chunks=(50000,), blocks=(2048,))


def test_multi_column_summary_combined_block_pruning(tmp_path):
    """A conjunction over several SUMMARY-indexed columns must AND their per-block
    masks, pruning more than any single column — and stay scan-correct.

    ``a`` ascending and ``b`` descending give tight per-block ranges, so
    ``a > 0.8N`` and ``b > 0.8N`` prune *disjoint* blocks and their AND is empty.
    """
    import blosc2.ctable_indexing as cti

    n = 200_000
    a = np.arange(n, dtype="f8")
    b = (n - 1 - np.arange(n)).astype("f8")
    t = blosc2.CTable(_TwoColRow)
    t.extend(list(zip(a.tolist(), b.tolist(), strict=True)))
    path = str(tmp_path / "twocol.b2z")
    t.to_b2z(path)
    with blosc2.open(path, mode="a") as w:
        w.create_index("a", kind=blosc2.IndexKind.SUMMARY)
        w.create_index("b", kind=blosc2.IndexKind.SUMMARY)

    with blosc2.open(path) as r:
        # Per-column masks prune ~20% each; the combined (AND) mask is empty.
        cat = dict(r._get_index_catalog())
        expr = (r.a > 0.8 * n) & (r.b > 0.8 * n)
        idx = r._find_indexed_columns(r._cols, cat, expr.operands)
        combined = cti._CTableIndexingMixin._combined_summary_block_bitmap(r, expr, idx, idx[0][1])
        a_only = r.a > 0.8 * n
        a_idx = r._find_indexed_columns(r._cols, cat, a_only.operands)
        a_mask = cti._CTableIndexingMixin._combined_summary_block_bitmap(r, a_only, a_idx, a_idx[0][1])
        assert int(a_mask.sum()) > 0  # 'a' alone keeps some blocks
        assert int(combined.sum()) == 0  # the AND prunes everything

        # Correctness across several conjunctions (engaged and scan paths).
        for cond, expected in [
            ((r.a > 0.8 * n) & (r.b > 0.8 * n), int(((a > 0.8 * n) & (b > 0.8 * n)).sum())),
            ((r.a > 0.6 * n) & (r.b > 0.7 * n), int(((a > 0.6 * n) & (b > 0.7 * n)).sum())),
            ((r.a > 1000) & (r.b > 1000), int(((a > 1000) & (b > 1000)).sum())),
        ]:
            assert r.where(cond).nrows == expected


@dataclasses.dataclass
class _SortedRow:
    v: float = blosc2.field(blosc2.float64(), chunks=(2000,), blocks=(500,))


def test_summary_chunk_skip_scoped_extraction(tmp_path):
    """Sorted column so a high threshold matches only the last chunk(s): exercises
    the chunk-skip + scoped-extraction path (few candidate chunks) as well as the
    full-mask path (many candidate chunks).  Both must be scan-correct."""
    n = 20_000  # 10 chunks of 2000, 4 blocks each
    v = np.arange(n, dtype="f8")  # ascending
    t = blosc2.CTable(_SortedRow)
    t.extend([(x,) for x in v.tolist()])
    path = str(tmp_path / "sorted.b2z")
    t.to_b2z(path)
    with blosc2.open(path, mode="a") as w:
        w.create_index("v", kind=blosc2.IndexKind.SUMMARY)

    with blosc2.open(path) as r:
        for frac in (0.05, 0.5, 0.95, 0.99, 0.999, 1.5):
            thr = frac * n
            assert r.where(r.v > thr).nrows == int((v > thr).sum())
            # A negative threshold matches everything (full-mask path).
        assert r.where(r.v > -1).nrows == n


# ---------------------------------------------------------------------------
# Cross-column SUMMARY-index pruning on compact (.b2z) stores
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _CrossRow:
    # Small explicit grid so the SUMMARY index spans many block-segments and
    # both columns are chunk/block aligned (shared row→segment grid).
    a: float = blosc2.field(blosc2.float64(), chunks=(1000,), blocks=(250,))
    b: float = blosc2.field(blosc2.float64(), chunks=(1000,), blocks=(250,))


def _build_cross_b2z(tmp_path, a, b, name):
    """Compact .b2z with SUMMARY indexes on *both* columns."""
    t = blosc2.CTable(_CrossRow)
    t.extend(list(zip(a.tolist(), b.tolist(), strict=True)))
    path = str(tmp_path / name)
    t.to_b2z(path)
    with blosc2.open(path, mode="a") as w:
        w.create_index("a", kind=blosc2.IndexKind.SUMMARY)
        w.create_index("b", kind=blosc2.IndexKind.SUMMARY)
    return path


def _spy_on_plans(monkeypatch):
    """Patch ``plan_query`` to record every ``IndexPlan`` it returns.

    ``_try_index_where`` imports ``plan_query`` from ``blosc2.indexing`` at call
    time, so patching the module attribute is picked up by the next query.
    """
    plans = []
    orig = blosc2.indexing.plan_query

    def spy(*args, **kwargs):
        plan = orig(*args, **kwargs)
        plans.append(plan)
        return plan

    monkeypatch.setattr(blosc2.indexing, "plan_query", spy)
    return plans


def test_cross_column_and_prunes_segments_compact_b2z(tmp_path, monkeypatch):
    """Regression guard: an AND across two SUMMARY-indexed columns of a compact
    store must combine their per-segment candidate masks (intersection) instead
    of falling back to a full scan.  See todo/multiple-indexes-in-queries.md —
    the per-column-token change alone made the cross-column merge fail and
    silently disable pruning."""
    n = 10_000
    a = np.arange(n, dtype=np.float64)  # ascending → segment-selective
    b = np.random.default_rng(0).random(n) * 1000.0  # random → non-selective
    path = _build_cross_b2z(tmp_path, a, b, "and.b2z")

    expected = int(((a > n - 100) & (b > 500.0)).sum())
    assert expected > 0  # predicate must match real rows

    plans = _spy_on_plans(monkeypatch)
    with blosc2.open(path) as r:
        got = r.where(f"(a > {n - 100}) & (b > 500.0)").nrows
    assert got == expected

    pruned = [p for p in plans if p.usable and p.selected_units < p.total_units]
    assert pruned, "cross-column AND fell back to a full scan instead of pruning"


def test_cross_column_or_prunes_segments_compact_b2z(tmp_path, monkeypatch):
    """An OR across two SUMMARY-indexed columns must union their candidate masks
    (both sides segment-selective ⇒ still prunes), and stay correct."""
    n = 10_000
    a = np.arange(n, dtype=np.float64)  # ascending  → high values at high index
    b = np.arange(n, 0, -1, dtype=np.float64)  # descending → high values at low index
    path = _build_cross_b2z(tmp_path, a, b, "or.b2z")

    expected = int(((a > n - 100) | (b > n - 100)).sum())
    assert expected > 0

    plans = _spy_on_plans(monkeypatch)
    with blosc2.open(path) as r:
        got = r.where(f"(a > {n - 100}) | (b > {n - 100})").nrows
    assert got == expected

    pruned = [p for p in plans if p.usable and p.selected_units < p.total_units]
    assert pruned, "cross-column OR fell back to a full scan instead of pruning"


def test_cross_column_predicates_match_scan_compact_b2z(tmp_path):
    """Cross-column AND/OR over two SUMMARY-indexed columns must match the
    boolean-mask (no-index) result across selective, non-selective, empty, and
    mixed-direction predicates."""
    n = 8_000
    rng = np.random.default_rng(3)
    a = (rng.random(n) * 100).astype(np.float64)
    b = (rng.random(n) * 100).astype(np.float64)
    path = _build_cross_b2z(tmp_path, a, b, "corr.b2z")

    cases = [
        ("(a > 90) & (b > 10)", (a > 90) & (b > 10)),  # selective ∧ non-selective
        ("(a > 50) & (b > 50)", (a > 50) & (b > 50)),  # both moderately selective
        ("(a > 10) | (b > 90)", (a > 10) | (b > 90)),  # OR
        ("(a > 99.9) & (b > 99.9)", (a > 99.9) & (b > 99.9)),  # near-empty intersection
        ("(a < 5) & (b > 0)", (a < 5) & (b > 0)),  # low-end selective
    ]
    with blosc2.open(path) as r:
        for expr, mask in cases:
            assert r.where(expr).nrows == int(mask.sum()), expr


def _seg_plan(units, *, base_nrows=1000, segment_len=250, level="block"):
    import types

    from blosc2.indexing import SegmentPredicatePlan

    return SegmentPredicatePlan(
        base=types.SimpleNamespace(shape=(base_nrows,)),
        candidate_units=np.asarray(units, dtype=bool),
        descriptor={"token": "x"},
        target={},
        field=None,
        level=level,
        segment_len=segment_len,
    )


def test_merge_segment_plans_intersection_union_and_fallback():
    """Unit-level guard for the cross-column merge semantics."""
    from blosc2.indexing import _merge_segment_plans

    left = _seg_plan([1, 1, 1, 0])
    right = _seg_plan([0, 1, 1, 1])

    # Grid-compatible AND → intersection; OR → union.
    np.testing.assert_array_equal(_merge_segment_plans(left, right, "and").candidate_units, [0, 1, 1, 0])
    np.testing.assert_array_equal(_merge_segment_plans(left, right, "or").candidate_units, [1, 1, 1, 1])

    # Incompatible grid (different segment_len): AND keeps the more selective
    # side (fewer candidate rows = fewer nonzero units × segment_len), never a
    # full scan; OR cannot prune safely → None.
    coarse = _seg_plan([1, 1], segment_len=500)  # 2 units selected
    fine = _seg_plan([1, 0, 0, 0])  # 1 unit selected, finer grid
    assert _merge_segment_plans(coarse, fine, "and") is fine  # fine prunes more
    assert _merge_segment_plans(fine, coarse, "and") is fine
    assert _merge_segment_plans(coarse, fine, "or") is None
