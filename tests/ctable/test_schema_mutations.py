#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""Tests for add_column, drop_column, rename_column, Column.assign,
and the read-only view mutability model."""

import os
import pathlib
import shutil
from dataclasses import dataclass

import numpy as np
import pytest

import blosc2
from blosc2 import CTable

TABLE_ROOT = str(pathlib.Path(__file__).parent / "saved_ctable" / "test_schema_mutations")


@dataclass
class Row:
    id: int = blosc2.field(blosc2.int64(ge=0))
    score: float = blosc2.field(blosc2.float64(ge=0, le=100), default=0.0)
    active: bool = blosc2.field(blosc2.bool(), default=True)


DATA10 = [(i, float(i * 10), True) for i in range(10)]


@pytest.fixture(autouse=True)
def clean_dir():
    if os.path.exists(TABLE_ROOT):
        shutil.rmtree(TABLE_ROOT)
    os.makedirs(TABLE_ROOT, exist_ok=True)
    yield
    if os.path.exists(TABLE_ROOT):
        shutil.rmtree(TABLE_ROOT)


def table_path(name):
    return os.path.join(TABLE_ROOT, f"{name}.b2d")


# ===========================================================================
# View mutability — views are read-only: value writes and structural
# changes are both blocked.
# ===========================================================================


def test_view_blocks_column_setitem():
    """Writing values through a view raises; the base table is untouched."""
    t = CTable(Row, new_data=DATA10)
    view = t.where(t["id"] > 4)  # rows 5-9
    indices = list(range(len(view)))
    new_scores = view["score"][:] * 2
    with pytest.raises(ValueError, match="view"):
        view["score"][indices] = new_scores
    assert t["score"][5] == pytest.approx(50.0)  # unchanged


def test_view_blocks_column_setitem_slice():
    t = CTable(Row, new_data=DATA10)
    view = t[2:5]
    with pytest.raises(ValueError, match="view"):
        view["score"][0] = 99.0
    assert t["score"][2] == pytest.approx(20.0)


def test_view_blocks_column_setitem_gathered_rows():
    t = CTable(Row, new_data=DATA10)
    view = t[[1, 3, 5]]
    with pytest.raises(ValueError, match="view"):
        view["score"][0] = 99.0
    assert t["score"][1] == pytest.approx(10.0)


def test_view_blocks_column_setitem_sorted():
    t = CTable(Row, new_data=DATA10)
    view = t.sort_by("score", ascending=False, view=True)
    with pytest.raises(ValueError, match="view"):
        view["score"][0] = 99.0
    assert t["score"][9] == pytest.approx(90.0)


def test_view_blocks_column_setitem_projection():
    t = CTable(Row, new_data=DATA10)
    view = t[["id", "score"]]
    with pytest.raises(ValueError, match="view"):
        view["score"][0] = 99.0
    assert t["score"][0] == pytest.approx(0.0)


def test_view_blocks_column_setitem_boolean_mask():
    t = CTable(Row, new_data=DATA10)
    view = t.where(t["id"] > 4)
    mask = np.zeros(len(view), dtype=bool)
    mask[0] = True
    with pytest.raises(ValueError, match="view"):
        view["score"][mask] = np.array([99.0])
    assert t["score"][5] == pytest.approx(50.0)


def test_view_blocks_assign():
    """assign() through a view raises; the base table is untouched."""
    t = CTable(Row, new_data=DATA10)
    view = t.where(t["id"] > 4)
    with pytest.raises(ValueError, match="view"):
        view["score"].assign(np.zeros(len(view)))
    assert t["score"][5] == pytest.approx(50.0)


def test_take_from_view_yields_independent_writable_table():
    t = CTable(Row, new_data=DATA10)
    view = t.where(t["id"] > 4)
    independent = view.take([0, 1])
    independent["score"][0] = 99.0
    assert independent["score"][0] == pytest.approx(99.0)
    # base and view unaffected
    assert t["score"][5] == pytest.approx(50.0)


def test_writes_on_base_still_work_while_view_exists():
    t = CTable(Row, new_data=DATA10)
    view = t.where(t["id"] > 4)
    t["score"][0] = 999.0
    assert t["score"][0] == pytest.approx(999.0)


def test_view_blocks_append():
    t = CTable(Row, new_data=DATA10)
    view = t.where(t["id"] > 4)
    with pytest.raises(TypeError):
        view.append((99, 10.0, True))


def test_view_blocks_delete():
    t = CTable(Row, new_data=DATA10)
    view = t.where(t["id"] > 4)
    with pytest.raises(ValueError, match="view"):
        view.delete(0)


def test_view_blocks_compact():
    t = CTable(Row, new_data=DATA10)
    view = t.where(t["id"] > 4)
    with pytest.raises(ValueError, match="view"):
        view.compact()


def test_readonly_disk_table_blocks_assign():
    path = table_path("ro")
    t = CTable(Row, urlpath=path, mode="w", new_data=DATA10)
    t.close()
    t_ro = CTable.open(path, mode="r")
    with pytest.raises(ValueError, match="read-only"):
        t_ro["score"].assign(np.ones(len(t_ro)))


def test_readonly_disk_table_blocks_setitem():
    path = table_path("ro_setitem")
    t = CTable(Row, urlpath=path, mode="w", new_data=DATA10)
    t.close()
    t_ro = CTable.open(path, mode="r")
    with pytest.raises(ValueError, match="read-only"):
        t_ro["score"][0] = 99.0


def test_blosc2_open_materializes_ctable():
    path = table_path("open_ctable")
    t = CTable(Row, urlpath=path, mode="w", new_data=DATA10)
    t.close()
    opened = blosc2.open(path, mode="r")
    assert isinstance(opened, CTable)
    assert opened.col_names == ["id", "score", "active"]
    np.testing.assert_array_equal(opened["id"][:], np.arange(10))


def test_blosc2_open_raw_treestore_without_manifest():
    path = table_path("raw_store")
    with blosc2.TreeStore(path, mode="w", threshold=0) as tstore:
        tstore["/group/node"] = np.arange(5)

    opened = blosc2.open(path, mode="r")
    assert isinstance(opened, blosc2.TreeStore)
    assert np.array_equal(opened["/group/node"][:], np.arange(5))


def test_blosc2_open_raw_treestore_for_unknown_manifest_kind():
    path = table_path("unknown_manifest")
    with blosc2.TreeStore(path, mode="w", threshold=0) as tstore:
        meta = blosc2.SChunk()
        meta.vlmeta["kind"] = "mystery"
        meta.vlmeta["version"] = 1
        tstore["/_meta"] = meta
        tstore["/payload"] = np.arange(3)

    opened = blosc2.open(path, mode="r")
    assert isinstance(opened, blosc2.TreeStore)
    assert np.array_equal(opened["/payload"][:], np.arange(3))


def test_extensionless_ctable_path_uses_extensionless_store():
    path = os.path.join(TABLE_ROOT, "alias_ctable")
    t = CTable(Row, urlpath=path, mode="w", new_data=DATA10)
    t.close()
    assert os.path.isdir(path)
    opened = blosc2.open(path, mode="r")
    assert isinstance(opened, CTable)
    np.testing.assert_array_equal(opened["id"][:], np.arange(10))


# ===========================================================================
# Column.assign
# ===========================================================================


def test_assign_replaces_all_values():
    t = CTable(Row, new_data=DATA10)
    t["score"].assign([99.0] * 10)
    assert list(t["score"][:]) == [99.0] * 10


def test_assign_coerces_python_ints_to_float():
    t = CTable(Row, new_data=DATA10)
    t["score"].assign(list(range(10)))  # Python ints → float64
    np.testing.assert_array_equal(t["score"][:], np.arange(10, dtype=np.float64))


def test_assign_wrong_length_raises():
    t = CTable(Row, new_data=DATA10)
    with pytest.raises(ValueError, match="10"):
        t["score"].assign([1.0, 2.0])


def test_assign_through_view_raises():
    t = CTable(Row, new_data=DATA10)
    view = t.where(t["id"] < 5)  # rows 0-4
    with pytest.raises(ValueError, match="view"):
        view["score"].assign([0.0] * 5)
    # base table untouched
    np.testing.assert_array_equal(t["score"][:], np.arange(10, dtype=np.float64) * 10)


def test_assign_respects_deleted_rows():
    t = CTable(Row, new_data=DATA10)
    t.delete([0])  # delete id=0; 9 live rows remain
    t["score"].assign([1.0] * 9)
    assert len(t["score"][:]) == 9
    assert all(v == 1.0 for v in t["score"][:])


# ===========================================================================
# add_column
# ===========================================================================


def test_add_column_appears_in_col_names():
    t = CTable(Row, new_data=DATA10)
    t.add_column("weight", blosc2.field(blosc2.float64(), default=0.0))
    assert "weight" in t.col_names


def test_add_column_fills_default_for_existing_rows():
    t = CTable(Row, new_data=DATA10)
    t.add_column("weight", blosc2.field(blosc2.float64(), default=5.5))
    np.testing.assert_array_equal(t["weight"][:], np.full(10, 5.5))


def test_add_column_without_default_allowed_for_empty_table():
    t = CTable(Row)
    t.add_column("weight", blosc2.float64())
    t.append((1, 2.0, True, 3.0))
    assert t["weight"][0] == pytest.approx(3.0)


def test_add_column_without_default_on_non_empty_table_raises():
    t = CTable(Row, new_data=DATA10)
    with pytest.raises(ValueError, match="requires a default"):
        t.add_column("weight", blosc2.float64())


def test_add_column_new_rows_can_use_it():
    t = CTable(Row, new_data=DATA10)
    t.add_column("weight", blosc2.field(blosc2.float64(), default=0.0))
    # After adding, extend doesn't know about weight — add manually
    t["weight"].assign(np.ones(10) * 2.0)
    assert t["weight"].mean() == pytest.approx(2.0)


def test_add_column_schema_updated():
    t = CTable(Row, new_data=DATA10)
    t.add_column("weight", blosc2.field(blosc2.float64(), default=0.0))
    assert "weight" in t.schema.columns_by_name


def test_add_column_uses_field_storage_config():
    t = CTable(Row, new_data=DATA10)
    t.add_column("weight", blosc2.field(blosc2.float64(), default=0.0, cparams={"clevel": 9}))
    assert t.column_schema("weight").config.cparams == {"clevel": 9}


def test_add_column_persists_on_disk():
    path = table_path("add_col")
    t = CTable(Row, urlpath=path, mode="w", new_data=DATA10)
    t.add_column("weight", blosc2.field(blosc2.float64(), default=7.0))
    t.close()
    t2 = CTable.open(path)
    assert "weight" in t2.col_names
    np.testing.assert_array_equal(t2["weight"][:], np.full(10, 7.0))


def test_add_column_view_raises():
    t = CTable(Row, new_data=DATA10)
    view = t.where(t["id"] > 4)
    with pytest.raises(ValueError, match="view"):
        view.add_column("weight", blosc2.field(blosc2.float64(), default=0.0))


def test_add_column_duplicate_raises():
    t = CTable(Row, new_data=DATA10)
    with pytest.raises(ValueError, match="already exists"):
        t.add_column("score", blosc2.field(blosc2.float64(), default=0.0))


def test_add_column_bad_default_raises():
    t = CTable(Row, new_data=DATA10)
    with pytest.raises(TypeError):
        t.add_column("flag", blosc2.field(blosc2.int8(), default="not_a_number"))


def test_add_column_skips_deleted_rows():
    t = CTable(Row, new_data=DATA10)
    t.delete([0, 1])  # 8 live rows
    t.add_column("weight", blosc2.field(blosc2.float64(), default=3.0))
    vals = t["weight"][:]
    assert len(vals) == 8
    assert all(v == 3.0 for v in vals)


# ===========================================================================
# drop_column
# ===========================================================================


def test_drop_column_removes_from_col_names():
    t = CTable(Row, new_data=DATA10)
    t.drop_column("active")
    assert "active" not in t.col_names


def test_drop_column_schema_updated():
    t = CTable(Row, new_data=DATA10)
    t.drop_column("active")
    assert "active" not in t.schema.columns_by_name


def test_drop_column_last_raises():
    @dataclass
    class OneCol:
        id: int = blosc2.field(blosc2.int64())

    t = CTable(OneCol, new_data=[(i,) for i in range(5)])
    with pytest.raises(ValueError, match="last"):
        t.drop_column("id")


def test_drop_column_missing_raises():
    t = CTable(Row, new_data=DATA10)
    with pytest.raises(KeyError):
        t.drop_column("nonexistent")


def test_drop_column_view_raises():
    t = CTable(Row, new_data=DATA10)
    view = t.where(t["id"] > 4)
    with pytest.raises(ValueError, match="view"):
        view.drop_column("active")


def test_drop_column_deletes_file_on_disk():
    path = table_path("drop_col")
    t = CTable(Row, urlpath=path, mode="w", new_data=DATA10)
    t.drop_column("active")
    assert not os.path.exists(os.path.join(path, "_cols", "active.b2nd"))


def test_drop_column_persists_schema_on_disk():
    path = table_path("drop_schema")
    t = CTable(Row, urlpath=path, mode="w", new_data=DATA10)
    t.drop_column("active")
    t.close()
    t2 = CTable.open(path)
    assert "active" not in t2.col_names
    assert t2.ncols == 2


# ===========================================================================
# rename_column
# ===========================================================================


def test_rename_column_updates_col_names():
    t = CTable(Row, new_data=DATA10)
    t.rename_column("score", "points")
    assert "points" in t.col_names
    assert "score" not in t.col_names


def test_rename_column_data_intact():
    t = CTable(Row, new_data=DATA10)
    original = t["score"][:].copy()
    t.rename_column("score", "points")
    np.testing.assert_array_equal(t["points"][:], original)


def test_rename_column_schema_updated():
    t = CTable(Row, new_data=DATA10)
    t.rename_column("score", "points")
    assert "points" in t.schema.columns_by_name
    assert "score" not in t.schema.columns_by_name


def test_rename_column_order_preserved():
    t = CTable(Row, new_data=DATA10)
    t.rename_column("score", "points")
    assert t.col_names == ["id", "points", "active"]


def test_rename_column_missing_raises():
    t = CTable(Row, new_data=DATA10)
    with pytest.raises(KeyError):
        t.rename_column("nonexistent", "foo")


def test_rename_column_conflict_raises():
    t = CTable(Row, new_data=DATA10)
    with pytest.raises(ValueError, match="already exists"):
        t.rename_column("score", "active")


def test_rename_column_view_raises():
    t = CTable(Row, new_data=DATA10)
    view = t.where(t["id"] > 4)
    with pytest.raises(ValueError, match="view"):
        view.rename_column("score", "points")


def test_rename_column_persists_on_disk():
    path = table_path("rename_col")
    t = CTable(Row, urlpath=path, mode="w", new_data=DATA10)
    t.rename_column("score", "points")
    t.close()
    t2 = CTable.open(path)
    assert "points" in t2.col_names
    assert "score" not in t2.col_names
    assert os.path.exists(os.path.join(path, "_cols", "points.b2nd"))
    assert not os.path.exists(os.path.join(path, "_cols", "score.b2nd"))


# ===========================================================================
# Boolean mask indexing  (pandas-style)
# ===========================================================================


def test_bool_mask_getitem():
    t = CTable(Row, new_data=DATA10)
    mask = t["id"][:] % 2 == 0  # even ids
    result = t["score"][mask]
    np.testing.assert_array_equal(result, np.array([0.0, 20.0, 40.0, 60.0, 80.0]))


def test_bool_mask_setitem():
    t = CTable(Row, new_data=DATA10)
    mask = t["id"][:] % 2 == 0
    t["score"][mask] = 0.0
    scores = t["score"][:]
    np.testing.assert_array_equal(scores[0::2], np.zeros(5))  # evens zeroed
    np.testing.assert_array_equal(scores[1::2], np.array([10.0, 30.0, 50.0, 70.0, 90.0]))


def test_bool_mask_inplace_multiply():
    """The pandas idiom: col[mask] *= scalar."""
    t = CTable(Row, new_data=DATA10)
    mask = t["id"][:] % 2 == 0
    t["score"][mask] *= 2
    scores = t["score"][:]
    np.testing.assert_array_equal(scores[0::2], np.array([0.0, 40.0, 80.0, 120.0, 160.0]))
    np.testing.assert_array_equal(scores[1::2], np.array([10.0, 30.0, 50.0, 70.0, 90.0]))


def test_bool_mask_wrong_length_raises():
    t = CTable(Row, new_data=DATA10)
    bad_mask = np.array([True, False, True], dtype=np.bool_)
    with pytest.raises(IndexError, match="length"):
        _ = t["score"][bad_mask]


def test_bool_mask_setitem_through_view_raises():
    """Boolean-mask reads still work on views; writes through a view raise."""
    t = CTable(Row, new_data=DATA10)
    view = t.where(t["id"] < 6)  # rows 0-5
    mask = view["id"][:] % 2 == 0
    with pytest.raises(ValueError, match="view"):
        view["score"][mask] *= 10
    # base untouched
    assert t["score"][0] == pytest.approx(0.0)
    assert t["score"][2] == pytest.approx(20.0)
    assert t["score"][4] == pytest.approx(40.0)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
