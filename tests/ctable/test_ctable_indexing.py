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
    for i in range(n):
        t.append([i, float(i) * 1.5, i % 5])
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


def test_create_expression_index_in_memory():
    t = _make_table(50)
    idx = t.create_index(expression="value * category", kind=blosc2.IndexKind.FULL, name="vc")
    assert idx.kind == "full"
    assert t.index(expression="value * category").name == "vc"
    assert t.index(name="vc").name == "vc"


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
