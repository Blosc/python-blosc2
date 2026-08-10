#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""The ``.notnull`` validity sidecar, through every persistence path (Phase 3).

A mask-storage nullable column keeps its nullity in a bool NDArray beside the
values.  Nothing reads or writes one through the public API yet -- ``extend``,
``is_null`` and friends arrive in Phase 4 -- so these tests build the sidecar
by hand via ``_ensure_null_mask`` and check that every path that moves, grows,
shrinks or serializes a column carries it along unchanged.

The load-bearing invariants here:

* **An absent sidecar means all-valid.**  A nullable column that has never
  been given a null has no ``.notnull`` key at all, and must not acquire one
  just by being saved, copied or reopened.
* **Chunk pinning.**  The sidecar shares its column's row grid, so mask and
  values never need re-aligning on a paired read.
"""

from __future__ import annotations

import dataclasses
import os

import numpy as np
import pytest
from utf8_compat import needs_utf8

import blosc2
from blosc2.ctable_storage import _NOTNULL_SUFFIX, FileTableStorage

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

NULL_POSITIONS = (3, 7, 41)


def make_row_type():
    return dataclasses.make_dataclass(
        "MaskRow",
        [
            ("a", int, blosc2.field(blosc2.int64(null_storage="mask"))),
            ("b", float, blosc2.field(blosc2.float64())),
            ("c", str, blosc2.field(blosc2.string(max_length=8, null_storage="mask"))),
        ],
    )


def make_table(n_rows=50, capacity=100, *, with_nulls=True, urlpath=None):
    """A table with two mask columns; only ``a`` is given nulls."""
    kwargs = {"expected_size": capacity}
    if urlpath is not None:
        kwargs |= {"urlpath": urlpath, "mode": "w"}
    t = blosc2.CTable(make_row_type(), **kwargs)
    t.extend([(i, float(i), f"s{i}") for i in range(n_rows)])
    if with_nulls:
        mask = t._ensure_null_mask("a")
        for pos in NULL_POSITIONS:
            if pos < n_rows:
                mask[pos] = False
    return t


def null_positions(table, name="a", n=None):
    """Physical positions the sidecar marks null, or ``None`` if there is none."""
    mask = table._null_mask(name)
    if mask is None:
        return None
    n = table._n_rows if n is None else n
    return np.flatnonzero(~mask[:n]).tolist()


def sidecar_key_exists(table, name):
    storage = table._storage
    return storage.has_null_mask(name)


# ---------------------------------------------------------------------------
# Decision 9: an absent sidecar means all-valid
# ---------------------------------------------------------------------------


def test_a_fresh_mask_column_has_no_sidecar():
    t = make_table(with_nulls=False)
    assert t._null_mask_names == ["a", "c"]
    assert t._null_mask("a") is None
    assert t._null_mask("c") is None
    assert t._existing_null_masks() == {}


def test_non_nullable_columns_are_not_listed():
    t = make_table(with_nulls=False)
    assert "b" not in t._null_mask_names
    assert t._null_mask("b") is None


def test_sentinel_columns_are_not_listed():
    Row = dataclasses.make_dataclass("SentinelRow", [("v", int, blosc2.field(blosc2.int64(null_value=-1)))])
    t = blosc2.CTable(Row, expected_size=8)
    assert t._null_mask_names == []


def test_materializing_the_sidecar_starts_all_valid():
    t = make_table(with_nulls=False)
    mask = t._ensure_null_mask("a")
    assert mask.shape == (len(t._valid_rows),)
    assert mask.dtype == np.dtype(np.bool_)
    # An explicit all-True fill, not zeros: a fresh sidecar must say exactly
    # what its absence said.
    assert bool(np.asarray(mask[:]).all())


def test_ensure_null_mask_is_idempotent():
    t = make_table(with_nulls=False)
    first = t._ensure_null_mask("a")
    first[5] = False
    second = t._ensure_null_mask("a")
    assert second is first
    assert null_positions(t) == [5]


def test_a_null_free_column_writes_no_sidecar_to_disk(tmp_path):
    path = str(tmp_path / "clean.b2d")
    make_table(with_nulls=False).save(path)
    reopened = blosc2.CTable.open(path)
    try:
        assert not sidecar_key_exists(reopened, "a")
        assert reopened._null_mask("a") is None
    finally:
        reopened.close()


def test_only_the_column_given_nulls_gets_a_sidecar(tmp_path):
    path = str(tmp_path / "one.b2d")
    make_table().save(path)
    reopened = blosc2.CTable.open(path)
    try:
        assert sidecar_key_exists(reopened, "a")
        assert not sidecar_key_exists(reopened, "c")
    finally:
        reopened.close()


# ---------------------------------------------------------------------------
# Chunk pinning
# ---------------------------------------------------------------------------


def test_sidecar_shares_its_column_row_grid():
    t = make_table()
    col = t._cols["a"]
    mask = t._null_mask("a")
    assert mask.chunks[0] == col.chunks[0]
    assert mask.blocks[0] == col.blocks[0]


def test_sidecar_grid_survives_a_chunk_override(tmp_path):
    """``copy(chunks=…)`` reblocks the column; the sidecar must follow it."""
    t = make_table()
    copied = t.copy(urlpath=str(tmp_path / "ovr.b2d"), chunks=16, blocks=8)
    try:
        assert copied._cols["a"].chunks[0] == 16
        assert copied._null_mask("a").chunks[0] == 16
        assert null_positions(copied) == list(NULL_POSITIONS)
    finally:
        copied.close()


@needs_utf8
def test_utf8_sidecar_falls_back_to_the_table_grid():
    """utf8 offsets carry ``n + 1`` entries, so they are not the grid to pin to."""
    Row = dataclasses.make_dataclass("Utf8Row", [("u", str, blosc2.field(blosc2.utf8(null_storage="mask")))])
    t = blosc2.CTable(Row, expected_size=64)
    t.extend([(f"v{i}",) for i in range(64)])
    mask = t._ensure_null_mask("u")
    assert mask.chunks[0] == t._valid_rows.chunks[0]
    assert mask.blocks[0] == t._valid_rows.blocks[0]


# ---------------------------------------------------------------------------
# Persistence round-trips
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("suffix", [".b2d", ".b2z"])
def test_sidecar_survives_save_and_open(tmp_path, suffix):
    path = str(tmp_path / f"t{suffix}")
    make_table().save(path)
    reopened = blosc2.CTable.open(path)
    try:
        assert null_positions(reopened) == list(NULL_POSITIONS)
    finally:
        reopened.close()


def test_sidecar_survives_mmap_open(tmp_path):
    path = str(tmp_path / "m.b2z")
    make_table().save(path)
    reopened = blosc2.CTable.open(path, mmap_mode="r")
    try:
        assert null_positions(reopened) == list(NULL_POSITIONS)
    finally:
        reopened.close()


def test_sidecar_survives_ctable_load(tmp_path):
    path = str(tmp_path / "l.b2d")
    make_table().save(path)
    loaded = blosc2.CTable.load(path)
    assert null_positions(loaded) == list(NULL_POSITIONS)
    # load() rebuilds every array in RAM, so the pinning must be re-established
    # rather than inherited.
    assert loaded._null_mask("a").chunks[0] == loaded._cols["a"].chunks[0]


def test_sidecar_survives_cframe_round_trip():
    original = make_table()
    rebuilt = blosc2.ctable_from_cframe(original.to_cframe())
    assert null_positions(rebuilt) == list(NULL_POSITIONS)


def test_cframe_of_a_null_free_table_carries_no_sidecar():
    rebuilt = blosc2.ctable_from_cframe(make_table(with_nulls=False).to_cframe())
    assert rebuilt._null_mask("a") is None


def test_sidecar_survives_inline_treestore(tmp_path):
    path = str(tmp_path / "tree.b2d")
    with blosc2.TreeStore(path, mode="w") as store:
        store["/tbl"] = make_table()
    with blosc2.TreeStore(path, mode="r") as store:
        assert null_positions(store["/tbl"]) == list(NULL_POSITIONS)


def test_sidecar_survives_physical_pack_and_unpack(tmp_path):
    """``to_b2z``/``to_b2d`` zip the leaves as-is; the sidecar is one of them."""
    src_path = str(tmp_path / "src.b2d")
    make_table().save(src_path)

    packed = str(tmp_path / "packed.b2z")
    src = blosc2.CTable.open(src_path, mode="r")
    try:
        src.to_b2z(packed)
    finally:
        src.close()

    zipped = blosc2.CTable.open(packed, mode="r")
    try:
        assert null_positions(zipped) == list(NULL_POSITIONS)
        unpacked_path = str(tmp_path / "unpacked.b2d")
        zipped.to_b2d(unpacked_path)
    finally:
        zipped.close()

    unpacked = blosc2.CTable.open(unpacked_path)
    try:
        assert null_positions(unpacked) == list(NULL_POSITIONS)
    finally:
        unpacked.close()


@pytest.mark.parametrize("compact", [True, False])
def test_sidecar_survives_in_memory_copy(compact):
    copied = make_table().copy(compact=compact)
    assert null_positions(copied) == list(NULL_POSITIONS)


def test_copy_of_a_null_free_table_stays_sidecar_free():
    assert make_table(with_nulls=False).copy()._null_mask("a") is None


def test_view_materialization_remaps_the_sidecar():
    t = make_table()
    materialized = t.where("b >= 5").copy()
    # Physical rows 0-4 are filtered out, so physical nulls 7 and 41 land at
    # logical 2 and 36; physical 3 is not in the view at all.
    assert null_positions(materialized) == [2, 36]


def test_a_view_shares_its_base_sidecar_cache():
    t = make_table()
    view = t.where("b >= 5")
    assert view._null_masks is t._null_masks
    assert view._null_mask("a") is t._null_mask("a")


# ---------------------------------------------------------------------------
# Capacity management
# ---------------------------------------------------------------------------


def test_grow_extends_the_sidecar_as_all_valid():
    t = make_table(n_rows=100, capacity=100)
    before = len(t._valid_rows)
    t.extend([(i, float(i), f"s{i}") for i in range(100, 150)])
    mask = t._null_mask("a")
    assert mask.shape[0] == len(t._valid_rows) > before
    # resize() zero-fills, i.e. *invalid*; the new tail must be corrected.
    assert bool(np.asarray(mask[before:]).all())
    assert null_positions(t) == list(NULL_POSITIONS)


def test_repeated_grow_cycles_keep_the_sidecar_aligned():
    t = make_table(n_rows=20, capacity=20)
    for start in range(20, 200, 20):
        t.extend([(i, float(i), f"s{i}") for i in range(start, start + 20)])
    mask = t._null_mask("a")
    assert mask.shape[0] == len(t._valid_rows)
    assert null_positions(t) == [p for p in NULL_POSITIONS if p < 20]
    assert int((~np.asarray(mask[: t._n_rows])).sum()) == 2


def test_trim_capacity_shrinks_the_sidecar():
    t = make_table(n_rows=50, capacity=100)
    t.trim_capacity()
    mask = t._null_mask("a")
    assert mask.shape == (50,)
    assert null_positions(t) == list(NULL_POSITIONS)


def test_capacity_paths_find_an_unopened_sidecar(tmp_path):
    """A reopened table has opened no sidecar; grow and trim must still find it.

    Iterating only the *materialized* sidecars would silently leave this one
    at its old length, out of step with its column.
    """
    path = str(tmp_path / "trim.b2d")
    make_table(n_rows=50, capacity=100).save(path)
    reopened = blosc2.CTable.open(path, mode="a")
    try:
        assert reopened._null_masks.materialized() == {}
        reopened.extend([(i, float(i), f"s{i}") for i in range(50, 90)])
        assert reopened._null_mask("a").shape[0] == len(reopened._valid_rows) > 50
        reopened.trim_capacity()
        assert reopened._null_mask("a").shape == (90,)
        assert null_positions(reopened) == list(NULL_POSITIONS)
    finally:
        reopened.close()


def test_compact_gathers_the_sidecar_with_its_column():
    t = make_table()
    t.delete(0)
    t.compact()
    # Every surviving row shifts down one, so the nulls do too.
    assert null_positions(t) == [p - 1 for p in NULL_POSITIONS]


def test_compact_leaves_the_freed_tail_valid():
    t = make_table()
    t.delete(0)
    t.compact()
    mask = np.asarray(t._null_mask("a")[:])
    assert bool(mask[t._n_rows :].all())


# ---------------------------------------------------------------------------
# Column-level mutation
# ---------------------------------------------------------------------------


def test_drop_column_removes_the_sidecar_key(tmp_path):
    path = str(tmp_path / "drop.b2d")
    make_table().save(path)
    table = blosc2.CTable.open(path, mode="a")
    try:
        key = table._storage._col_key("a") + _NOTNULL_SUFFIX
        assert key in table._storage._open_store()
        table.drop_column("a")
        assert key not in table._storage._open_store()
        assert table._null_mask_names == ["c"]
    finally:
        table.close()


@pytest.mark.parametrize("persistent", [True, False])
def test_rename_column_carries_the_sidecar(tmp_path, persistent):
    if persistent:
        path = str(tmp_path / "ren.b2d")
        make_table().save(path)
        table = blosc2.CTable.open(path, mode="a")
    else:
        table = make_table()
    try:
        table.rename_column("a", "renamed")
        assert null_positions(table, "renamed") == list(NULL_POSITIONS)
        assert table._null_mask("a") is None
    finally:
        if persistent:
            table.close()


def test_renamed_sidecar_survives_reopen(tmp_path):
    path = str(tmp_path / "ren2.b2d")
    make_table().save(path)
    table = blosc2.CTable.open(path, mode="a")
    try:
        table.rename_column("a", "renamed")
    finally:
        table.close()
    reopened = blosc2.CTable.open(path)
    try:
        assert null_positions(reopened, "renamed") == list(NULL_POSITIONS)
    finally:
        reopened.close()


# ---------------------------------------------------------------------------
# Storage-backend surface
# ---------------------------------------------------------------------------


def test_storage_reports_absence_before_creation(tmp_path):
    path = str(tmp_path / "backend.b2d")
    table = make_table(with_nulls=False, urlpath=path)
    try:
        assert table._storage.has_null_mask("a") is False
        table._ensure_null_mask("a")
        assert table._storage.has_null_mask("a") is True
    finally:
        table.close()


def test_delete_null_mask_is_a_no_op_when_absent(tmp_path):
    path = str(tmp_path / "nodel.b2d")
    table = make_table(with_nulls=False, urlpath=path)
    try:
        table._storage.delete_null_mask("a")  # must not raise
        assert table._storage.has_null_mask("a") is False
    finally:
        table.close()


def test_in_memory_storage_never_reports_a_stored_sidecar():
    """In-memory sidecars are held by the CTable, mirroring ``open_column``."""
    t = make_table()
    assert t._storage.has_null_mask("a") is False
    assert t._null_mask("a") is not None


def test_sidecar_key_is_beside_the_column_key(tmp_path):
    path = str(tmp_path / "layout.b2d")
    make_table().save(path)
    storage = FileTableStorage(path, "r")
    try:
        keys = set(storage._open_store())
        assert "/_cols/a" in keys
        assert "/_cols/a" + _NOTNULL_SUFFIX in keys
        assert "/_cols/c" + _NOTNULL_SUFFIX not in keys
    finally:
        storage.close()


def test_sidecar_leaf_lands_on_disk(tmp_path):
    path = str(tmp_path / "leaf.b2d")
    make_table().save(path)
    assert os.path.exists(os.path.join(path, "_cols", "a.notnull.b2nd"))


def test_a_widened_bool_ndarray_column_remembers_that_it_was_widened(tmp_path):
    """The flag cannot be re-derived on load, so it has to be written down.

    A nullable-sentinel ``ndarray(bool)`` column is stored as ``uint8`` to make
    room for its sentinel, and ``dtype_str`` therefore serializes *already
    widened* -- byte-identical to what a **declared** ``uint8`` column writes.
    Without the flag the reopened column looks declared, so
    ``_unflip_mask_bool_dtype`` no longer fires and ``convert_nulls(to="mask")``
    leaves it ``uint8`` where the same conversion before the save gave
    ``np.bool_``.  The persistent dtype-change guard was misled the same way.
    """
    spec = blosc2.ndarray((2,), dtype=np.bool_, nullable=True, null_storage="sentinel")
    Row = dataclasses.make_dataclass("NdBoolRow", [("a", object, blosc2.field(spec))])
    path = tmp_path / "ndbool.b2t"
    t = blosc2.CTable(Row, expected_size=8, urlpath=str(path), mode="w")
    t.extend([(np.array([True, False]),), (np.array([False, False]),)])
    before = t.copy().convert_nulls("a", to="mask")._schema.columns_by_name["a"].spec.dtype
    del t

    reopened = blosc2.CTable.open(str(path), mode="a")
    assert reopened._schema.columns_by_name["a"].spec.bool_widened_to_uint8 is True
    after = reopened.convert_nulls("a", to="mask")._schema.columns_by_name["a"].spec.dtype
    assert after == before == np.dtype(np.bool_)


def test_a_declared_uint8_ndarray_column_is_not_treated_as_widened(tmp_path):
    """The distinction the flag exists for: real bytes must never be unflipped."""
    spec = blosc2.ndarray((2,), dtype=np.uint8, nullable=True)
    Row = dataclasses.make_dataclass("NdU8Row", [("a", object, blosc2.field(spec))])
    path = tmp_path / "ndu8.b2t"
    t = blosc2.CTable(Row, expected_size=8, urlpath=str(path), mode="w")
    t.extend([(np.array([200, 7], dtype=np.uint8),), (np.array([1, 2], dtype=np.uint8),)])
    del t

    reopened = blosc2.CTable.open(str(path))
    assert reopened._schema.columns_by_name["a"].spec.bool_widened_to_uint8 is False
    assert [x.tolist() for x in reopened["a"][:]] == [[200, 7], [1, 2]]
