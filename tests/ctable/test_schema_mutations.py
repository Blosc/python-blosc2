#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""Tests for add_column, drop_column, rename_column, Column.assign,
and the corrected view mutability model."""

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
    return os.path.join(TABLE_ROOT, name)


# ===========================================================================
# View mutability — value writes allowed, structural changes blocked
# ===========================================================================


def test_view_allows_column_setitem():
    """Writing values through a view modifies the parent table."""
    t = CTable(Row, new_data=DATA10)
    view = t.where(t["id"] > 4)  # rows 5-9
    # double scores of those rows using __setitem__
    indices = list(range(len(view)))
    new_scores = view["score"].to_numpy() * 2
    view["score"][indices] = new_scores
    # check parent was modified
    assert t["score"][5] == pytest.approx(100.0)  # was 50.0


def test_view_allows_assign():
    """assign() through a view modifies the parent table."""
    t = CTable(Row, new_data=DATA10)
    view = t.where(t["id"] > 4)
    view["score"].assign(np.zeros(len(view)))
    assert t["score"][5] == pytest.approx(0.0)
    assert t["score"][4] == pytest.approx(40.0)  # untouched


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
    del t
    t_ro = CTable.open(path, mode="r")
    with pytest.raises(ValueError, match="read-only"):
        t_ro["score"].assign(np.ones(len(t_ro)))


def test_readonly_disk_table_blocks_setitem():
    path = table_path("ro_setitem")
    t = CTable(Row, urlpath=path, mode="w", new_data=DATA10)
    del t
    t_ro = CTable.open(path, mode="r")
    with pytest.raises(ValueError, match="read-only"):
        t_ro["score"][0] = 99.0


# ===========================================================================
# Column.assign
# ===========================================================================


def test_assign_replaces_all_values():
    t = CTable(Row, new_data=DATA10)
    t["score"].assign([99.0] * 10)
    assert list(t["score"].to_numpy()) == [99.0] * 10


def test_assign_coerces_python_ints_to_float():
    t = CTable(Row, new_data=DATA10)
    t["score"].assign(list(range(10)))  # Python ints → float64
    np.testing.assert_array_equal(t["score"].to_numpy(), np.arange(10, dtype=np.float64))


def test_assign_wrong_length_raises():
    t = CTable(Row, new_data=DATA10)
    with pytest.raises(ValueError, match="10"):
        t["score"].assign([1.0, 2.0])


def test_assign_through_view_touches_only_matching_rows():
    t = CTable(Row, new_data=DATA10)
    view = t.where(t["id"] < 5)  # rows 0-4
    view["score"].assign([0.0] * 5)
    # rows 0-4 → 0, rows 5-9 unchanged
    scores = t["score"].to_numpy()
    np.testing.assert_array_equal(scores[:5], np.zeros(5))
    np.testing.assert_array_equal(scores[5:], np.arange(5, 10, dtype=np.float64) * 10)


def test_assign_respects_deleted_rows():
    t = CTable(Row, new_data=DATA10)
    t.delete([0])  # delete id=0; 9 live rows remain
    t["score"].assign([1.0] * 9)
    assert len(t["score"].to_numpy()) == 9
    assert all(v == 1.0 for v in t["score"].to_numpy())


# ===========================================================================
# add_column
# ===========================================================================


def test_add_column_appears_in_col_names():
    t = CTable(Row, new_data=DATA10)
    t.add_column("weight", blosc2.float64(), 0.0)
    assert "weight" in t.col_names


def test_add_column_fills_default_for_existing_rows():
    t = CTable(Row, new_data=DATA10)
    t.add_column("weight", blosc2.float64(), 5.5)
    np.testing.assert_array_equal(t["weight"].to_numpy(), np.full(10, 5.5))


def test_add_column_new_rows_can_use_it():
    t = CTable(Row, new_data=DATA10)
    t.add_column("weight", blosc2.float64(), 0.0)
    # After adding, extend doesn't know about weight — add manually
    t["weight"].assign(np.ones(10) * 2.0)
    assert t["weight"].mean() == pytest.approx(2.0)


def test_add_column_schema_updated():
    t = CTable(Row, new_data=DATA10)
    t.add_column("weight", blosc2.float64(), 0.0)
    assert "weight" in t.schema.columns_by_name


def test_add_column_persists_on_disk():
    path = table_path("add_col")
    t = CTable(Row, urlpath=path, mode="w", new_data=DATA10)
    t.add_column("weight", blosc2.float64(), 7.0)
    del t
    t2 = CTable.open(path)
    assert "weight" in t2.col_names
    np.testing.assert_array_equal(t2["weight"].to_numpy(), np.full(10, 7.0))


def test_add_column_view_raises():
    t = CTable(Row, new_data=DATA10)
    view = t.where(t["id"] > 4)
    with pytest.raises(ValueError, match="view"):
        view.add_column("weight", blosc2.float64(), 0.0)


def test_add_column_duplicate_raises():
    t = CTable(Row, new_data=DATA10)
    with pytest.raises(ValueError, match="already exists"):
        t.add_column("score", blosc2.float64(), 0.0)


def test_add_column_bad_default_raises():
    t = CTable(Row, new_data=DATA10)
    with pytest.raises(TypeError):
        t.add_column("flag", blosc2.int8(), "not_a_number")


def test_add_column_skips_deleted_rows():
    t = CTable(Row, new_data=DATA10)
    t.delete([0, 1])  # 8 live rows
    t.add_column("weight", blosc2.float64(), 3.0)
    vals = t["weight"].to_numpy()
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
    del t
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
    original = t["score"].to_numpy().copy()
    t.rename_column("score", "points")
    np.testing.assert_array_equal(t["points"].to_numpy(), original)


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
    del t
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
    mask = t["id"].to_numpy() % 2 == 0  # even ids
    result = t["score"][mask]
    np.testing.assert_array_equal(result, np.array([0.0, 20.0, 40.0, 60.0, 80.0]))


def test_bool_mask_setitem():
    t = CTable(Row, new_data=DATA10)
    mask = t["id"].to_numpy() % 2 == 0
    t["score"][mask] = 0.0
    scores = t["score"].to_numpy()
    np.testing.assert_array_equal(scores[0::2], np.zeros(5))  # evens zeroed
    np.testing.assert_array_equal(scores[1::2], np.array([10.0, 30.0, 50.0, 70.0, 90.0]))


def test_bool_mask_inplace_multiply():
    """The pandas idiom: col[mask] *= scalar."""
    t = CTable(Row, new_data=DATA10)
    mask = t["id"].to_numpy() % 2 == 0
    t["score"][mask] *= 2
    scores = t["score"].to_numpy()
    np.testing.assert_array_equal(scores[0::2], np.array([0.0, 40.0, 80.0, 120.0, 160.0]))
    np.testing.assert_array_equal(scores[1::2], np.array([10.0, 30.0, 50.0, 70.0, 90.0]))


def test_bool_mask_wrong_length_raises():
    t = CTable(Row, new_data=DATA10)
    bad_mask = np.array([True, False, True], dtype=np.bool_)
    with pytest.raises(IndexError, match="length"):
        _ = t["score"][bad_mask]


def test_bool_mask_through_view():
    """Boolean mask indexing works on views too."""
    t = CTable(Row, new_data=DATA10)
    view = t.where(t["id"] < 6)  # rows 0-5
    mask = view["id"].to_numpy() % 2 == 0
    view["score"][mask] *= 10
    # rows 0,2,4 in view → ids 0,2,4 in parent → scores 0,20,40 * 10
    assert t["score"][0] == pytest.approx(0.0)
    assert t["score"][2] == pytest.approx(200.0)
    assert t["score"][4] == pytest.approx(400.0)
    assert t["score"][1] == pytest.approx(10.0)  # untouched


if __name__ == "__main__":
    pytest.main(["-v", __file__])
