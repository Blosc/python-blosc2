#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

from dataclasses import dataclass

import numpy as np
import pytest

import blosc2
from blosc2 import CTable


@dataclass
class Row:
    id: int = blosc2.field(blosc2.int64(ge=0))
    score: float = blosc2.field(blosc2.float64(ge=0))
    active: bool = blosc2.field(blosc2.bool(), default=True)


@dataclass
class StrRow:
    label: str = blosc2.field(blosc2.string(max_length=16))


DATA20 = [(i, float(i * 10), True) for i in range(20)]


# -------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------


def test_column_metadata():
    """dtype correctness, internal reference consistency, and mask defaults."""
    tabla = CTable(Row, new_data=DATA20)

    assert tabla.id.dtype == np.int64
    assert tabla.score.dtype == np.float64
    assert tabla.active.dtype == np.bool_

    assert tabla.id._raw_col is tabla._cols["id"]
    assert tabla.id._valid_rows is tabla._valid_rows

    # mask is None by default
    assert tabla.id._mask is None
    assert tabla.score._mask is None


def test_column_getitem_no_holes():
    """int, slice, and list indexing on a full table."""
    tabla = CTable(Row, new_data=DATA20)
    col = tabla.id

    # int
    assert col[0] == 0
    assert col[5] == 5
    assert col[19] == 19
    assert col[-1] == 19
    assert col[-5] == 15

    # slice returns a Column view
    assert isinstance(col[0:5], blosc2.Column)
    assert isinstance(col[10:15], blosc2.Column)

    # list
    assert list(col[[0, 5, 10, 15]]) == [0, 5, 10, 15]
    assert list(col[[19, 0, 10]]) == [19, 0, 10]


def test_column_getitem_with_holes():
    """int, slice, and list indexing after deletions."""
    tabla = CTable(Row, new_data=DATA20)
    tabla.delete([1, 3, 5, 7, 9])
    col = tabla.id

    assert col[0] == 0
    assert col[1] == 2
    assert col[2] == 4
    assert col[3] == 6
    assert col[4] == 8
    assert col[-1] == 19
    assert col[-2] == 18

    assert list(col[[0, 2, 4]]) == [0, 4, 8]
    assert list(col[[5, 3, 1]]) == [10, 6, 2]

    tabla2 = CTable(Row, new_data=DATA20)
    tabla2.delete([1, 3, 5, 7, 9, 11, 13, 15, 17, 19])
    col2 = tabla2.id

    assert list(col2[0:5].to_numpy()) == [0, 2, 4, 6, 8]
    assert list(col2[5:10].to_numpy()) == [10, 12, 14, 16, 18]
    assert list(col2[::2].to_numpy()) == [0, 4, 8, 12, 16]


def test_column_getitem_out_of_range():
    """int and list indexing raise IndexError when out of bounds."""
    tabla = CTable(Row, new_data=DATA20)
    tabla.delete([1, 3, 5, 7, 9])
    col = tabla.id

    with pytest.raises(IndexError):
        _ = col[100]
    with pytest.raises(IndexError):
        _ = col[-100]
    with pytest.raises(IndexError):
        _ = col[[0, 1, 100]]


def test_column_setitem_no_holes():
    """int, slice, and list assignment on a full table."""
    tabla = CTable(Row, new_data=DATA20)
    col = tabla.id

    col[0] = 999
    assert col[0] == 999
    col[10] = 888
    assert col[10] == 888
    col[-1] = 777
    assert col[-1] == 777

    col[0:5] = [100, 101, 102, 103, 104]
    assert list(col[0:5].to_numpy()) == [100, 101, 102, 103, 104]

    col[[0, 5, 10]] = [10, 50, 100]
    assert col[0] == 10
    assert col[5] == 50
    assert col[10] == 100


def test_column_setitem_with_holes():
    """int, slice, and list assignment after deletions."""
    tabla = CTable(Row, new_data=DATA20)
    tabla.delete([1, 3, 5, 7, 9])
    col = tabla.id

    col[0] = 999
    assert col[0] == 999
    assert tabla._cols["id"][0] == 999

    col[2] = 888
    assert col[2] == 888
    assert tabla._cols["id"][4] == 888

    col[-1] = 777
    assert col[-1] == 777

    col[0:3] = [100, 200, 300]
    assert col[0] == 100
    assert col[1] == 200
    assert col[2] == 300

    col[[0, 2, 4]] = [11, 22, 33]
    assert col[0] == 11
    assert col[2] == 22
    assert col[4] == 33


def test_column_iter():
    """Iteration over full table, with odd-index holes, and on score column."""
    tabla = CTable(Row, new_data=DATA20)
    assert list(tabla.id) == list(range(20))

    tabla2 = CTable(Row, new_data=DATA20)
    tabla2.delete([1, 3, 5, 7, 9, 11, 13, 15, 17, 19])
    assert list(tabla2.id) == [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]

    tabla3 = CTable(Row, new_data=DATA20)
    tabla3.delete([0, 5, 10, 15])
    # fmt: off
    expected_score = [
        10.0, 20.0, 30.0, 40.0,
        60.0, 70.0, 80.0, 90.0,
        110.0, 120.0, 130.0, 140.0,
        160.0, 170.0, 180.0, 190.0,
    ]
    # fmt: on
    assert list(tabla3.score) == expected_score


def test_column_len():
    """len() after no deletions, partial deletions, cumulative deletions, and cross-column."""
    tabla = CTable(Row, new_data=DATA20)
    col = tabla.id
    assert len(col) == 20

    tabla.delete([1, 3, 5, 7, 9])
    assert len(col) == 15

    tabla2 = CTable(Row, new_data=DATA20)
    col2 = tabla2.id
    tabla2.delete([0, 1, 2])
    assert len(col2) == 17
    tabla2.delete([0, 1, 2, 3, 4])
    assert len(col2) == 12

    data = [(i, float(i * 10), i % 2 == 0) for i in range(10)]
    tabla3 = CTable(Row, new_data=data, expected_size=10)
    tabla3.delete([0, 1, 5, 6, 9])
    assert len(tabla3.id) == len(tabla3.score) == len(tabla3.active) == 5
    for i in range(len(tabla3.id)):
        assert tabla3.score[i] == float(tabla3.id[i] * 10)


def test_column_edge_cases():
    """Empty table and fully-deleted table both behave as zero-length columns."""
    tabla = CTable(Row)
    assert len(tabla.id) == 0
    assert list(tabla.id) == []

    data = [(i, float(i * 10), True) for i in range(10)]
    tabla2 = CTable(Row, new_data=data)
    tabla2.delete(list(range(10)))
    assert len(tabla2.id) == 0
    assert list(tabla2.id) == []


# -------------------------------------------------------------------
# New tests for Column view (mask) and to_array()
# -------------------------------------------------------------------


def test_column_slice_returns_view():
    """Column[slice] returns a Column instance with a non-None mask."""
    tabla = CTable(Row, new_data=DATA20)
    col = tabla.id

    view = col[0:5]
    assert isinstance(view, blosc2.Column)
    assert view._mask is not None
    assert view._table is tabla
    assert view._col_name == "id"


def test_to_array_slices():
    """to_array() on slice views: full table and with holes."""
    # No holes
    tabla = CTable(Row, new_data=DATA20)
    col = tabla.id
    np.testing.assert_array_equal(col[0:5].to_numpy(), np.array([0, 1, 2, 3, 4], dtype=np.int64))
    np.testing.assert_array_equal(col[5:10].to_numpy(), np.array([5, 6, 7, 8, 9], dtype=np.int64))
    np.testing.assert_array_equal(col[15:20].to_numpy(), np.array([15, 16, 17, 18, 19], dtype=np.int64))
    np.testing.assert_array_equal(col[0:20].to_numpy(), np.arange(20, dtype=np.int64))

    # With holes: delete odd indices → keep evens 0,2,4,...,18
    tabla.delete([1, 3, 5, 7, 9, 11, 13, 15, 17, 19])
    col = tabla.id
    np.testing.assert_array_equal(col[0:5].to_numpy(), np.array([0, 2, 4, 6, 8], dtype=np.int64))
    np.testing.assert_array_equal(col[5:10].to_numpy(), np.array([10, 12, 14, 16, 18], dtype=np.int64))


def test_to_array_full_column():
    """to_array() with no slice (full column) returns all valid rows."""
    tabla = CTable(Row, new_data=DATA20)
    tabla.delete([0, 10, 19])
    col = tabla.id

    expected = np.array([i for i in range(20) if i not in {0, 10, 19}], dtype=np.int64)
    np.testing.assert_array_equal(col[0 : len(col)].to_numpy(), expected)


def test_to_array_mask_does_not_include_deleted():
    """Mask & valid_rows intersection excludes deleted rows inside the slice range."""
    tabla = CTable(Row, new_data=DATA20)
    # delete rows 2 and 3, which fall inside slice [0:5]
    tabla.delete([2, 3])
    col = tabla.id

    # logical [0:5] should now map to physical rows 0,1,4,5,6
    result = col[0:5].to_numpy()
    np.testing.assert_array_equal(result, np.array([0, 1, 4, 5, 6], dtype=np.int64))


def test_column_view_mask_is_independent():
    """Two slice views on the same column have independent masks."""
    tabla = CTable(Row, new_data=DATA20)
    col = tabla.id

    view_a = col[0:5]

    np.testing.assert_array_equal(view_a.to_numpy(), np.arange(0, 5, dtype=np.int64))


# -------------------------------------------------------------------
# iter_chunks
# -------------------------------------------------------------------


def test_iter_chunks_full_table():
    """iter_chunks reassembles to the same values as to_numpy()."""
    tabla = CTable(Row, new_data=DATA20)
    expected = tabla["id"].to_numpy()
    got = np.concatenate(list(tabla["id"].iter_chunks(size=7)))
    np.testing.assert_array_equal(got, expected)


def test_iter_chunks_chunk_sizes():
    """Each yielded chunk has at most *size* elements; last may be smaller."""
    tabla = CTable(Row, new_data=DATA20)
    chunks = list(tabla["score"].iter_chunks(size=6))
    for c in chunks[:-1]:
        assert len(c) == 6
    assert len(chunks[-1]) <= 6
    assert sum(len(c) for c in chunks) == 20


def test_iter_chunks_skips_deleted_rows():
    """Deleted rows are not included in any chunk."""
    tabla = CTable(Row, new_data=DATA20)
    tabla.delete([0, 1, 2])  # delete id 0, 1, 2
    chunks = list(tabla["id"].iter_chunks(size=5))
    all_vals = np.concatenate(chunks)
    assert 0 not in all_vals
    assert 1 not in all_vals
    assert 2 not in all_vals
    assert len(all_vals) == 17


def test_iter_chunks_size_larger_than_table():
    """A size larger than the table yields a single chunk with all rows."""
    tabla = CTable(Row, new_data=DATA20)
    chunks = list(tabla["id"].iter_chunks(size=1000))
    assert len(chunks) == 1
    np.testing.assert_array_equal(chunks[0], np.arange(20, dtype=np.int64))


def test_iter_chunks_empty_table():
    """iter_chunks on an empty table yields nothing."""
    tabla = CTable(Row)
    chunks = list(tabla["id"].iter_chunks())
    assert chunks == []


# -------------------------------------------------------------------
# Aggregates: sum
# -------------------------------------------------------------------


def test_sum_int():
    t = CTable(Row, new_data=DATA20)
    assert t["id"].sum() == sum(range(20))


def test_sum_float():
    t = CTable(Row, new_data=DATA20)
    assert t["score"].sum() == pytest.approx(sum(i * 10.0 for i in range(20)))


def test_sum_bool_counts_trues():
    t = CTable(Row, new_data=DATA20)  # all active=True
    assert t["active"].sum() == 20


def test_sum_skips_deleted_rows():
    t = CTable(Row, new_data=DATA20)
    t.delete([0])  # remove id=0
    assert t["id"].sum() == sum(range(1, 20))


def test_sum_empty_raises():
    t = CTable(Row)
    with pytest.raises(ValueError, match="empty"):
        t["id"].sum()


def test_sum_wrong_type_raises():
    t = CTable(StrRow, new_data=[("hello",)])
    with pytest.raises(TypeError):
        t["label"].sum()


# -------------------------------------------------------------------
# Aggregates: min / max
# -------------------------------------------------------------------


def test_min_int():
    t = CTable(Row, new_data=DATA20)
    assert t["id"].min() == 0


def test_max_int():
    t = CTable(Row, new_data=DATA20)
    assert t["id"].max() == 19


def test_min_float():
    t = CTable(Row, new_data=DATA20)
    assert t["score"].min() == pytest.approx(0.0)


def test_max_float():
    t = CTable(Row, new_data=DATA20)
    assert t["score"].max() == pytest.approx(190.0)


def test_min_max_string():
    t = CTable(StrRow, new_data=[("banana",), ("apple",), ("cherry",)])
    assert t["label"].min() == "apple"
    assert t["label"].max() == "cherry"


def test_min_skips_deleted():
    t = CTable(Row, new_data=DATA20)
    t.delete([0])  # remove id=0, next min is 1
    assert t["id"].min() == 1


def test_min_empty_raises():
    t = CTable(Row)
    with pytest.raises(ValueError, match="empty"):
        t["id"].min()


def test_max_complex_raises():
    @dataclass
    class CRow:
        val: complex = blosc2.field(blosc2.complex128())

    t = CTable(CRow, new_data=[(1 + 2j,)])
    with pytest.raises(TypeError):
        t["val"].max()


# -------------------------------------------------------------------
# Aggregates: mean
# -------------------------------------------------------------------


def test_mean_int():
    t = CTable(Row, new_data=DATA20)
    assert t["id"].mean() == pytest.approx(9.5)


def test_mean_float():
    t = CTable(Row, new_data=DATA20)
    assert t["score"].mean() == pytest.approx(95.0)


def test_mean_skips_deleted():
    t = CTable(Row, new_data=[(0, 0.0, True), (10, 100.0, True)])
    t.delete([0])  # remove id=0; only id=10 remains
    assert t["id"].mean() == pytest.approx(10.0)


def test_mean_empty_raises():
    t = CTable(Row)
    with pytest.raises(ValueError, match="empty"):
        t["id"].mean()


# -------------------------------------------------------------------
# Aggregates: std
# -------------------------------------------------------------------


def test_std_population():
    t = CTable(Row, new_data=DATA20)
    ids = np.arange(20, dtype=np.float64)
    assert t["id"].std() == pytest.approx(float(ids.std(ddof=0)))


def test_std_sample():
    t = CTable(Row, new_data=DATA20)
    ids = np.arange(20, dtype=np.float64)
    assert t["id"].std(ddof=1) == pytest.approx(float(ids.std(ddof=1)))


def test_std_single_element():
    t = CTable(Row, new_data=[(5, 50.0, True)])
    assert t["id"].std() == pytest.approx(0.0)


def test_std_single_element_ddof1_is_nan():
    t = CTable(Row, new_data=[(5, 50.0, True)])
    assert np.isnan(t["id"].std(ddof=1))


def test_std_empty_raises():
    t = CTable(Row)
    with pytest.raises(ValueError, match="empty"):
        t["id"].std()


# -------------------------------------------------------------------
# Aggregates: any / all
# -------------------------------------------------------------------


def test_any_all_true():
    t = CTable(Row, new_data=DATA20)  # all active=True
    assert t["active"].any() is True
    assert t["active"].all() is True


def test_any_some_false():
    data = [(i, float(i), i % 2 == 0) for i in range(10)]
    t = CTable(Row, new_data=data)
    assert t["active"].any() is True
    assert t["active"].all() is False


def test_all_false():
    data = [(i, float(i), False) for i in range(5)]
    t = CTable(Row, new_data=data)
    assert t["active"].any() is False
    assert t["active"].all() is False


def test_any_empty_is_false():
    t = CTable(Row)
    assert t["active"].any() is False


def test_all_empty_is_true():
    # vacuous truth: all() over nothing is True (same as Python's built-in)
    t = CTable(Row)
    assert t["active"].all() is True


def test_any_wrong_type_raises():
    t = CTable(Row, new_data=DATA20)
    with pytest.raises(TypeError):
        t["id"].any()


# -------------------------------------------------------------------
# unique
# -------------------------------------------------------------------


def test_unique_int():
    t = CTable(Row, new_data=[(i % 5, float(i), True) for i in range(20)])
    result = t["id"].unique()
    np.testing.assert_array_equal(result, np.array([0, 1, 2, 3, 4], dtype=np.int64))


def test_unique_bool():
    data = [(i, float(i), i % 3 != 0) for i in range(10)]
    t = CTable(Row, new_data=data)
    result = t["active"].unique()
    assert set(result.tolist()) == {True, False}


def test_unique_skips_deleted():
    t = CTable(Row, new_data=[(i % 3, float(i), True) for i in range(9)])
    # ids are [0,1,2,0,1,2,0,1,2]; logical rows with id==0 are at positions 0,3,6
    t.delete([0, 3, 6])
    result = t["id"].unique()
    assert 0 not in result.tolist()
    assert set(result.tolist()) == {1, 2}


def test_unique_empty():
    t = CTable(Row)
    result = t["id"].unique()
    assert len(result) == 0


# -------------------------------------------------------------------
# value_counts
# -------------------------------------------------------------------


def test_value_counts_basic():
    data = [(i % 3, float(i), True) for i in range(9)]  # ids: 0,1,2,0,1,2,0,1,2
    t = CTable(Row, new_data=data)
    vc = t["id"].value_counts()
    assert vc[0] == 3
    assert vc[1] == 3
    assert vc[2] == 3


def test_value_counts_sorted_by_count():
    data = [(0, 0.0, True)] * 5 + [(1, 1.0, True)] * 2 + [(2, 2.0, True)] * 8
    t = CTable(Row, new_data=data)
    vc = t["id"].value_counts()
    counts = list(vc.values())
    assert counts == sorted(counts, reverse=True)


def test_value_counts_bool():
    data = [(i, float(i), i % 4 != 0) for i in range(20)]  # 5 False, 15 True
    t = CTable(Row, new_data=data)
    vc = t["active"].value_counts()
    assert vc[True] == 15
    assert vc[False] == 5
    assert list(vc.keys())[0] is True  # True comes first (higher count)


def test_value_counts_empty():
    t = CTable(Row)
    assert t["id"].value_counts() == {}


# -------------------------------------------------------------------
# sample (on CTable)
# -------------------------------------------------------------------


def test_sample_returns_correct_count():
    t = CTable(Row, new_data=DATA20)
    s = t.sample(5, seed=0)
    assert len(s) == 5


def test_sample_rows_are_subset():
    t = CTable(Row, new_data=DATA20)
    s = t.sample(7, seed=42)
    all_ids = set(t["id"].to_numpy().tolist())
    sample_ids = set(s["id"].to_numpy().tolist())
    assert sample_ids.issubset(all_ids)


def test_sample_is_read_only():
    t = CTable(Row, new_data=DATA20)
    s = t.sample(5, seed=0)
    with pytest.raises((ValueError, TypeError)):
        s.append((99, 9.0, True))


def test_sample_seed_reproducible():
    t = CTable(Row, new_data=DATA20)
    s1 = t.sample(5, seed=7)
    s2 = t.sample(5, seed=7)
    np.testing.assert_array_equal(s1["id"].to_numpy(), s2["id"].to_numpy())


def test_sample_n_larger_than_table():
    t = CTable(Row, new_data=DATA20)
    s = t.sample(1000, seed=0)
    assert len(s) == 20


def test_sample_zero():
    t = CTable(Row, new_data=DATA20)
    assert len(t.sample(0)) == 0


# -------------------------------------------------------------------
# cbytes / nbytes / __repr__
# -------------------------------------------------------------------


def test_cbytes_nbytes_positive():
    t = CTable(Row, new_data=DATA20)
    assert t.cbytes > 0
    assert t.nbytes > 0
    assert t.nbytes >= t.cbytes  # compressed is never larger than raw


def test_cbytes_nbytes_consistent_with_info():
    t = CTable(Row, new_data=DATA20)
    expected_cb = sum(col.cbytes for col in t._cols.values()) + t._valid_rows.cbytes
    expected_nb = sum(col.nbytes for col in t._cols.values()) + t._valid_rows.nbytes
    assert t.cbytes == expected_cb
    assert t.nbytes == expected_nb


def test_repr_contains_col_names_and_row_count():
    t = CTable(Row, new_data=DATA20)
    r = repr(t)
    assert "id" in r
    assert "score" in r
    assert "active" in r
    assert "20" in r


def test_repr_is_single_line():
    t = CTable(Row, new_data=DATA20)
    assert "\n" not in repr(t)


def test_column_repr_shows_preview_values():
    t = CTable(Row, new_data=DATA20)
    r = repr(t["id"][:])
    assert "Column('id'" in r
    assert "dtype=int64" in r
    assert "len=20" in r
    assert "values=[0, 1, 2" in r
    assert "..." in r


def test_info_omits_capacity_and_read_only_for_in_memory_table():
    t = CTable(Row, new_data=DATA20)
    info = repr(t.info)
    assert "capacity" not in info
    assert "read_only" not in info
    assert "open_mode" not in info


def test_info_shows_open_mode_for_persistent_table(tmp_path):
    path = str(tmp_path / "table.b2d")
    t = CTable(Row, new_data=DATA20, urlpath=path, mode="w")
    t.close()

    opened = CTable.open(path)
    info = repr(opened.info)
    assert "capacity" not in info
    assert "read_only" not in info
    assert "open_mode       : r" in info
    opened.close()


def test_info_schema_expands_unicode_dtype_labels():
    t = CTable(StrRow, new_data=[("alpha",), ("beta",)])
    info = repr(t.info)
    assert "U16 (Unicode, max 16 chars)" in info


def test_info_valid_rows_mask_only_reports_cbytes():
    t = CTable(Row, new_data=DATA20)
    info = repr(t.info)
    assert "valid_rows_mask : cbytes=" in info
    assert "valid_rows_mask : nbytes=" not in info


def test_info_indexes_only_report_cbytes(tmp_path):
    @dataclass
    class IndexedRow:
        id: int = blosc2.field(blosc2.int32())
        active: bool = blosc2.field(blosc2.bool(), default=True)

    data = [(i, i % 2 == 0) for i in range(32)]
    path = str(tmp_path / "indexed.b2d")
    t = CTable(IndexedRow, new_data=data, urlpath=path, mode="w")
    t.create_index("id", kind=blosc2.IndexKind.FULL)

    info = repr(t.info)
    index_block = info.split("indexes         :", 1)[1]
    assert "cbytes=" in index_block
    assert "nbytes=" not in index_block
    assert "cratio=" not in index_block


def test_info_cratio_uses_one_decimal_with_suffix():
    t = CTable(Row, new_data=DATA20)
    info = repr(t.info)
    assert "cratio          :" in info
    assert "x" in next(line for line in info.splitlines() if line.startswith("cratio"))


if __name__ == "__main__":
    pytest.main(["-v", __file__])
