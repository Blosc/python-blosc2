#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

from dataclasses import dataclass, make_dataclass

import numpy as np
import pytest

import blosc2
from blosc2 import CTable


@dataclass
class SalesRow:
    city: str = blosc2.field(blosc2.string(max_length=16))
    category: int = blosc2.field(blosc2.int32())
    sales: float = blosc2.field(blosc2.float64(nullable=True), default=0.0)
    qty: int = blosc2.field(blosc2.int32(), default=0)


DATA = [
    ("Paris", 1, 10.0, 1),
    ("Paris", 1, np.nan, 2),
    ("Rome", 1, 20.0, 3),
    ("Paris", 2, 30.0, 4),
    ("Rome", 1, 40.0, 5),
    ("Berlin", 2, np.nan, 6),
]


def col(table, name):
    return list(table._cols[name][: table.nrows])


def rows(table):
    return [tuple(table._cols[name][i] for name in table.col_names) for i in range(table.nrows)]


def test_groupby_size_counts_rows_per_group():
    t = CTable(SalesRow, new_data=DATA)

    out = t.group_by("city", sort=True).size()

    assert out.col_names == ["city", "size"]
    assert rows(out) == [("Berlin", 1), ("Paris", 3), ("Rome", 2)]


def test_groupby_count_counts_non_null_values():
    t = CTable(SalesRow, new_data=DATA)

    out = t.group_by("city", sort=True).count("sales")

    assert out.col_names == ["city", "sales_count"]
    assert rows(out) == [("Berlin", 0), ("Paris", 2), ("Rome", 2)]


def test_groupby_agg_numeric_reductions():
    t = CTable(SalesRow, new_data=DATA)

    out = t.group_by("city", sort=True).agg({"sales": ["sum", "mean", "min", "max", "count"]})

    assert out.col_names == ["city", "sales_sum", "sales_mean", "sales_min", "sales_max", "sales_count"]
    got = rows(out)
    assert got[0][0] == "Berlin"
    assert np.isnan(got[0][1])
    assert np.isnan(got[0][2])
    assert np.isnan(got[0][3])
    assert np.isnan(got[0][4])
    assert got[0][5] == 0
    assert got[1] == ("Paris", 40.0, 20.0, 10.0, 30.0, 2)
    assert got[2] == ("Rome", 60.0, 30.0, 20.0, 40.0, 2)


def test_groupby_argmin_argmax_return_logical_positions():
    t = CTable(SalesRow, new_data=DATA)

    out = t.group_by("city", sort=True).agg({"sales": ["argmin", "argmax"]})

    assert out.col_names == ["city", "sales_argmin", "sales_argmax"]
    assert rows(out) == [("Berlin", -1, -1), ("Paris", 0, 3), ("Rome", 2, 4)]


def test_groupby_argmin_argmax_convenience_methods_and_view_positions():
    t = CTable(SalesRow, new_data=DATA)
    view = t.where("qty >= 3")

    out = view.group_by("city", sort=True).argmax("sales")
    argmin = view.group_by(view.city, sort=True).argmin(view.sales)

    assert out.col_names == ["city", "sales_argmax"]
    assert rows(out) == [("Berlin", -1), ("Paris", 1), ("Rome", 2)]
    assert rows(argmin) == [("Berlin", -1), ("Paris", 1), ("Rome", 0)]


def test_groupby_argmin_argmax_ties_keep_first_row():
    # Two rows tie for the group's extreme; the earliest logical row wins, even
    # when the tie straddles a chunk boundary (locks the vectorized path).
    TieRow = make_dataclass(
        "TieRow",
        [("g", int, blosc2.field(blosc2.int32())), ("v", float, blosc2.field(blosc2.float64()))],
    )
    data = [(0, 5.0), (0, 9.0), (0, 9.0), (0, 5.0)]  # rows 0..3
    t = CTable(TieRow, new_data=data)

    out = t.group_by("g", sort=True, chunk_size=2).agg({"v": ["argmin", "argmax"]})

    # min 5.0 first at row 0; max 9.0 first at row 1 (tie with row 2 ignored).
    assert rows(out) == [(0, 0, 1)]


def test_groupby_multi_key_size():
    t = CTable(SalesRow, new_data=DATA)

    out = t.group_by(["city", "category"], sort=True).size()

    assert rows(out) == [("Berlin", 2, 1), ("Paris", 1, 2), ("Paris", 2, 1), ("Rome", 1, 2)]


def test_groupby_nested_column_name_result():
    t = CTable(SalesRow, new_data=DATA)
    t.rename_column("category", "trip.sec")
    t.rename_column("sales", "payment.tips")

    view = t.where((t["payment.tips"] > 10) & (t["trip.sec"] > 1))
    out = view.group_by("trip.sec", sort=True).size()
    out_from_column = view.group_by(t.trip.sec, sort=True).size()

    assert out.col_names == ["trip.sec", "size"]
    assert rows(out) == [(2, 1)]
    assert rows(out_from_column) == rows(out)
    assert out["trip.sec"][:].tolist() == [2]

    count = t.group_by(t.trip.sec, sort=True).count(t.payment.tips)
    assert count.col_names == ["trip.sec", "payment.tips_count"]
    assert rows(count) == [(1, 3), (2, 1)]

    agg = t.group_by(t.trip.sec, sort=True).agg({"payment.tips": "sum"})

    assert agg.col_names == ["trip.sec", "payment.tips_sum"]
    assert rows(agg) == [(1, 70.0), (2, 30.0)]


def test_groupby_respects_views_and_deleted_rows():
    t = CTable(SalesRow, new_data=DATA)
    t.delete(0)
    view = t.where("qty >= 3")

    out = view.group_by("city", sort=True).size()

    assert rows(out) == [("Berlin", 1), ("Paris", 1), ("Rome", 2)]


@dataclass
class DictRow:
    city: str = blosc2.field(blosc2.dictionary())
    sales: int = blosc2.field(blosc2.int32())


def test_groupby_dictionary_key_groups_by_decoded_value():
    t = CTable(DictRow, new_data=[("Paris", 10), ("Rome", 20), ("Paris", 30)])

    out = t.group_by("city", sort=True).agg({"sales": "sum"})

    assert out.col_names == ["city", "sales_sum"]
    assert rows(out) == [("Paris", 40), ("Rome", 20)]


def test_groupby_dictionary_key_sorted_by_string_not_code_order():
    """Dict groups come out alphabetical even when codes are assigned otherwise.

    Regression for the always-sorted contract: with "Rome" seen before "Paris"
    (codes Rome=0, Paris=1), the result must still be Paris-then-Rome.  The old
    code emitted code-assignment order unless ``sort=True``, and even then only
    happened to look right when first-seen order matched alphabetical.
    """
    t = CTable(DictRow, new_data=[("Rome", 20), ("Paris", 10), ("Rome", 5), ("Paris", 30)])

    # No sort= argument: ordering is now guaranteed.
    assert rows(t.group_by("city").agg({"sales": "sum"})) == [("Paris", 40), ("Rome", 25)]
    # argmin/argmax drive the dense-position path; it must sort the same way.
    out = t.group_by("city").agg({"sales": ["argmin", "argmax"]})
    # Paris rows 1,3 (10,30) -> argmin 1, argmax 3; Rome rows 0,2 (20,5).
    assert rows(out) == [("Paris", 1, 3), ("Rome", 2, 0)]


def test_groupby_dictionary_key_sorted_matches_python_sorted():
    """Vectorized dict-key ordering matches a Python sorted() reference."""
    rng = np.random.default_rng(0)
    labels = [f"city_{i:03d}" for i in range(200)]
    data = [(labels[int(rng.integers(len(labels)))], int(rng.integers(100))) for _ in range(5000)]
    t = CTable(DictRow, new_data=data)

    got = [r[0] for r in rows(t.group_by("city").size())]
    assert got == sorted({label for label, _ in data})


def test_groupby_string_key_sorted_without_sort_flag():
    """Fixed-width string keys (generic path) are also always sorted by key."""
    t = CTable(SalesRow, new_data=DATA)
    out = t.group_by("city").size()
    assert [r[0] for r in rows(out)] == ["Berlin", "Paris", "Rome"]


def test_groupby_dictionary_key_argmin_argmax_positions():
    # Dictionary key drives the dense-position fast path; verify it returns the
    # logical row positions of the extremes (chicago-taxi "company" shape).
    t = CTable(DictRow, new_data=[("Paris", 10), ("Rome", 50), ("Paris", 30), ("Rome", 20)])

    out = t.group_by("city", sort=True).agg({"sales": ["argmin", "argmax"]})

    assert out.col_names == ["city", "sales_argmin", "sales_argmax"]
    # Paris rows 0,2 (10,30) -> argmin row0, argmax row2; Rome rows 1,3 (50,20).
    assert rows(out) == [("Paris", 0, 2), ("Rome", 3, 1)]


def test_groupby_dictionary_key_beyond_default_code_capacity():
    data = [("Paris" if i % 2 == 0 else "Rome", 1) for i in range(5000)]
    t = CTable(DictRow, new_data=data)

    out = t.group_by("city", sort=True).size()

    assert rows(out) == [("Paris", 2500), ("Rome", 2500)]


def test_groupby_dropna_key_default_and_false():
    t = CTable(DictRow, new_data=[("Paris", 10), (None, 20), ("Paris", 30)])

    dropped = t.group_by("city", sort=True).size()
    kept = t.group_by("city", sort=True, dropna=False).size()

    assert rows(dropped) == [("Paris", 2)]
    assert rows(kept) == [(None, 1), ("Paris", 2)]


def test_groupby_agg_star_size():
    t = CTable(SalesRow, new_data=DATA)

    out = t.group_by("city", sort=True).agg({"*": "size"})

    assert rows(out) == [("Berlin", 1), ("Paris", 3), ("Rome", 2)]


def test_groupby_empty_table_returns_empty_result():
    t = CTable(SalesRow)

    out = t.group_by("city").size()

    assert out.nrows == 0
    assert out.col_names == ["city", "size"]


@dataclass
class Int32FloatRow:
    key: int = blosc2.field(blosc2.int32())
    value: float = blosc2.field(blosc2.float64())


@dataclass
class Float64KeyRow:
    key: float = blosc2.field(blosc2.float64())
    value: float = blosc2.field(blosc2.float64())


@dataclass
class Float32KeyRow:
    key: float = blosc2.field(blosc2.float32())
    value: float = blosc2.field(blosc2.float64())


@dataclass
class DictFloatRow:
    key: str = blosc2.field(blosc2.dictionary())
    value: float = blosc2.field(blosc2.float64())


@pytest.mark.parametrize(
    ("row_type", "data", "expected"),
    [
        (
            Int32FloatRow,
            [(0, 1.5), (2, 10.0), (1, 2.5), (2, 3.0), (0, 4.0)],
            [(0, 5.5), (1, 2.5), (2, 13.0)],
        ),
        (
            Float64KeyRow,
            [(0.0, 1.5), (2.0, 10.0), (1.0, 2.5), (2.0, 3.0), (0.0, 4.0)],
            [(0.0, 5.5), (1.0, 2.5), (2.0, 13.0)],
        ),
        (
            Float32KeyRow,
            [(0.0, 1.5), (2.0, 10.0), (1.0, 2.5), (2.0, 3.0), (0.0, 4.0)],
            [(0.0, 5.5), (1.0, 2.5), (2.0, 13.0)],
        ),
        (
            DictFloatRow,
            [("a", 1.5), ("c", 10.0), ("b", 2.5), ("c", 3.0), ("a", 4.0)],
            # Groups come out sorted by decoded string, not code-assignment order
            # ("c" was seen before "b").
            [("a", 5.5), ("b", 2.5), ("c", 13.0)],
        ),
    ],
)
def test_groupby_fast_path_sum_variants(row_type, data, expected):
    t = CTable(row_type, new_data=data)

    out = t.group_by("key").agg({"value": "sum"})

    assert rows(out) == expected


def test_groupby_float_integral_fast_path_falls_back_for_non_integral_keys():
    t = CTable(Float64KeyRow, new_data=[(0.5, 1.0), (1.5, 2.0), (0.5, 3.0)])

    # Float keys are not key-sorted by default (sort=None); request sort=True to
    # assert a specific order.
    out = t.group_by("key", sort=True).agg({"value": "sum"})

    assert rows(out) == [(0.5, 4.0), (1.5, 2.0)]


def test_groupby_float_integral_fast_path_falls_back_for_nan_group_when_kept():
    t = CTable(Float64KeyRow, new_data=[(0.0, 1.0), (np.nan, 2.0), (0.0, 3.0)])

    out = t.group_by("key", dropna=False).agg({"value": "sum"})

    got = rows(out)
    assert got[0] == (0.0, 4.0)
    assert np.isnan(got[1][0])
    assert got[1][1] == 2.0


@pytest.mark.parametrize("row_type", [Float64KeyRow, Float32KeyRow])
def test_groupby_integral_float_key_dense_min_max(row_type):
    # Non-negative integral float keys take the dense single-key path (the same
    # one used for integer keys), which also covers min/max -- not just sum.
    t = CTable(row_type, new_data=[(2.0, 5.0), (1.0, 9.0), (2.0, 3.0), (1.0, 4.0)])

    out_max = t.group_by("key", sort=True).max("value")
    out_min = t.group_by("key", sort=True).min("value")

    assert rows(out_max) == [(1.0, 9.0), (2.0, 5.0)]
    assert rows(out_min) == [(1.0, 4.0), (2.0, 3.0)]
    # The key column keeps its original float dtype in the result.
    assert out_max._cols["key"][:].dtype == t._cols["key"][:].dtype


def test_groupby_integral_float_key_falls_back_for_negative_keys():
    # Negative keys cannot use the dense (non-negative) mapping; the generic
    # path must still produce correct max results.
    t = CTable(Float64KeyRow, new_data=[(-1.0, 5.0), (-1.0, 8.0), (2.0, 3.0)])

    out = t.group_by("key", sort=True).max("value")

    assert rows(out) == [(-1.0, 8.0), (2.0, 3.0)]


def test_group_reduce_object_keys_sort_with_none():
    groups, sizes = blosc2.group_reduce(
        np.array([None, "b", "a", "b"], dtype=object), sort=True, dropna=False
    )

    assert groups.tolist() == [None, "a", "b"]
    assert sizes.tolist() == [1, 1, 2]


def test_group_reduce_object_numeric_keys_sort_with_none():
    groups, sizes = blosc2.group_reduce(np.array([None, 2, 1, 2], dtype=object), sort=True, dropna=False)

    assert groups.tolist() == [None, 1, 2]
    assert sizes.tolist() == [1, 1, 2]


@pytest.mark.parametrize(
    ("kernel_name", "value_dtype"),
    [
        ("groupby_dense_int_f64_min_checked", np.float64),
        ("groupby_dense_int_f64_max_checked", np.float64),
        ("groupby_dense_int_i64_min_checked", np.int64),
        ("groupby_dense_int_i64_max_checked", np.int64),
    ],
)
@pytest.mark.parametrize("bad_arg", ["values", "valid", "state"])
def test_groupby_ext_min_max_checked_validate_shapes(kernel_name, value_dtype, bad_arg):
    groupby_ext = pytest.importorskip("blosc2.groupby_ext")
    kernel = getattr(groupby_ext, kernel_name)

    keys = np.array([0, 1], dtype=np.int64)
    values = np.array([10, 20], dtype=value_dtype)
    valid = np.array([True, True], dtype=np.bool_)
    state = np.zeros(2, dtype=value_dtype)
    has_value = np.zeros(2, dtype=np.bool_)
    keys_present = np.zeros(2, dtype=np.bool_)

    if bad_arg == "values":
        values = values[:1]
        match = "keys, values and valid must have the same length"
    elif bad_arg == "valid":
        valid = valid[:1]
        match = "keys, values and valid must have the same length"
    else:
        has_value = has_value[:1]
        match = "state arrays must have the same length"

    with pytest.raises(ValueError, match=match):
        kernel(keys, values, valid, state, has_value, keys_present)


def test_groupby_rejects_bad_engine():
    t = CTable(SalesRow, new_data=DATA)

    with pytest.raises(ValueError):
        t.group_by("city", engine="cython")


def test_groupby_engine_numpy_matches_auto():
    t = CTable(SalesRow, new_data=DATA)
    auto_result = t.group_by("city", engine="auto").sum("sales")
    numpy_result = t.group_by("city", engine="numpy").sum("sales")
    assert col(auto_result, "city") == col(numpy_result, "city")
    np.testing.assert_array_equal(col(auto_result, "sales_sum"), col(numpy_result, "sales_sum"))


def test_groupby_engine_jit_not_implemented():
    t = CTable(SalesRow, new_data=DATA)
    with pytest.raises(NotImplementedError):
        t.group_by("city", engine="jit")


@pytest.mark.parametrize(
    ("schema_factory", "values"),
    [
        (blosc2.int8, [0, 2, 1, 2, 0]),
        (blosc2.uint8, [0, 2, 1, 2, 0]),
        (blosc2.int16, [0, 2, 1, 2, 0]),
        (blosc2.uint16, [0, 2, 1, 2, 0]),
        (blosc2.int32, [0, 2, 1, 2, 0]),
        (blosc2.uint32, [0, 2, 1, 2, 0]),
        (blosc2.int64, [0, 2, 1, 2, 0]),
        (blosc2.uint64, [0, 2, 1, 2, 0]),
    ],
)
def test_groupby_cython_fused_integer_key_dtypes(schema_factory, values):
    row_type = make_dataclass(
        f"FusedKey{schema_factory.__name__}Row",
        [
            ("key", int, blosc2.field(schema_factory())),
            ("value", int, blosc2.field(blosc2.int32())),
        ],
    )
    t = CTable(row_type, new_data=list(zip(values, [1, 10, 2, 3, 4], strict=True)))

    out = t.group_by("key", sort=True).agg({"value": "sum"})

    assert rows(out) == [(0, 5), (1, 2), (2, 13)]


def test_groupby_cython_integer_key_more_integer_aggs():
    row_type = make_dataclass(
        "IntKeyMoreIntegerAggsRow",
        [
            ("key", int, blosc2.field(blosc2.int16())),
            ("value", int, blosc2.field(blosc2.int32())),
        ],
    )
    t = CTable(row_type, new_data=[(0, 5), (1, 10), (0, -2), (1, 20), (2, 7)])

    out = t.group_by("key", sort=True).agg({"*": "size", "value": ["count", "sum", "mean", "min", "max"]})

    assert rows(out) == [(0, 2, 2, 3, 1.5, -2, 5), (1, 2, 2, 30, 15.0, 10, 20), (2, 1, 1, 7, 7.0, 7, 7)]


def test_groupby_cython_integer_key_nullable_float_aggs():
    row_type = make_dataclass(
        "IntKeyNullableFloatAggsRow",
        [
            ("key", int, blosc2.field(blosc2.uint16())),
            ("value", float, blosc2.field(blosc2.float64(nullable=True))),
        ],
    )
    t = CTable(row_type, new_data=[(0, 1.5), (1, np.nan), (0, 2.5), (1, np.nan), (2, 10.0)])

    out = t.group_by("key", sort=True).agg({"value": ["count", "sum", "mean", "min", "max"]})

    got = rows(out)
    assert got[0] == (0, 2, 4.0, 2.0, 1.5, 2.5)
    assert got[1][0] == 1
    assert got[1][1] == 0
    assert np.isnan(got[1][2])
    assert np.isnan(got[1][3])
    assert np.isnan(got[1][4])
    assert np.isnan(got[1][5])
    assert got[2] == (2, 1, 10.0, 10.0, 10.0, 10.0)


def test_groupby_cython_arbitrary_float_key_aggs():
    t = CTable(
        Float64KeyRow,
        new_data=[(0.5, 1.0), (1.25, 10.0), (0.5, 3.0), (-2.5, 4.0), (1.25, 2.0)],
    )

    # Float keys are not key-sorted by default (sort=None); request sort=True to
    # assert a specific order.
    out = t.group_by("key", sort=True).agg({"value": ["count", "sum", "mean", "min", "max"]})

    assert rows(out) == [
        (-2.5, 1, 4.0, 4.0, 4.0, 4.0),
        (0.5, 2, 4.0, 2.0, 1.0, 3.0),
        (1.25, 2, 12.0, 6.0, 2.0, 10.0),
    ]


def test_groupby_cython_arbitrary_float_key_nan_and_signed_zero():
    t = CTable(Float64KeyRow, new_data=[(-0.0, 1.0), (0.0, 2.0), (np.nan, 3.0), (np.nan, 4.0)])

    dropped = t.group_by("key").agg({"value": "sum"})
    kept = t.group_by("key", dropna=False).agg({"value": "sum"})

    assert rows(dropped) == [(0.0, 3.0)]
    got = rows(kept)
    assert got[0] == (0.0, 3.0)
    assert np.isnan(got[1][0])
    assert got[1][1] == 7.0


@dataclass
class TwoIntKeyFloatRow:
    key0: int = blosc2.field(blosc2.int16())
    key1: int = blosc2.field(blosc2.uint16())
    value: float = blosc2.field(blosc2.float64(nullable=True), default=0.0)


def test_groupby_cython_two_integer_key_hash_aggs():
    t = CTable(
        TwoIntKeyFloatRow,
        new_data=[(0, 1, 1.0), (0, 1, 3.0), (0, 2, 10.0), (1, 1, np.nan), (1, 1, 5.0)],
    )

    out = t.group_by(["key0", "key1"], sort=True).agg(
        {"*": "size", "value": ["count", "sum", "mean", "min", "max"]}
    )

    assert rows(out) == [
        (0, 1, 2, 2, 4.0, 2.0, 1.0, 3.0),
        (0, 2, 1, 1, 10.0, 10.0, 10.0, 10.0),
        (1, 1, 2, 1, 5.0, 5.0, 5.0, 5.0),
    ]


@dataclass
class DictIntKeyFloatRow:
    key0: str = blosc2.field(blosc2.dictionary())
    key1: int = blosc2.field(blosc2.int32())
    value: float = blosc2.field(blosc2.float64())


def test_groupby_cython_dictionary_integer_key_hash():
    t = CTable(DictIntKeyFloatRow, new_data=[("b", 2, 1.0), ("a", 1, 2.0), ("b", 2, 3.0)])

    out = t.group_by(["key0", "key1"], sort=True).agg({"value": "sum"})

    assert rows(out) == [("a", 1, 2.0), ("b", 2, 4.0)]


def test_groupby_convenience_numeric_methods():
    t = CTable(SalesRow, new_data=DATA)

    assert rows(t.group_by("city", sort=True).sum("qty")) == rows(
        t.group_by("city", sort=True).agg({"qty": "sum"})
    )
    assert rows(t.group_by("city", sort=True).mean("qty")) == rows(
        t.group_by("city", sort=True).agg({"qty": "mean"})
    )
    assert rows(t.group_by("city", sort=True).min("qty")) == rows(
        t.group_by("city", sort=True).agg({"qty": "min"})
    )
    assert rows(t.group_by("city", sort=True).max("qty")) == rows(
        t.group_by("city", sort=True).agg({"qty": "max"})
    )


def test_groupby_persistent_output_urlpath(tmp_path):
    t = CTable(SalesRow, new_data=DATA)
    path = tmp_path / "grouped.b2d"

    out = t.group_by("city", sort=True).agg({"qty": "sum"}, urlpath=path)
    out.close()

    reopened = CTable.open(str(path), mode="r")
    assert reopened.col_names == ["city", "qty_sum"]
    assert rows(reopened) == [("Berlin", 6), ("Paris", 7), ("Rome", 8)]


def test_groupby_persistent_output_urlpath_on_convenience_method(tmp_path):
    t = CTable(SalesRow, new_data=DATA)
    path = tmp_path / "grouped_mean.b2d"

    out = t.group_by("city", sort=True).mean("qty", urlpath=path)
    out.close()

    reopened = CTable.open(str(path), mode="r")
    assert rows(reopened) == [("Berlin", 6.0), ("Paris", 7 / 3), ("Rome", 4.0)]


# ----------------------------------------------------------------------
# Tri-state sort= (True / False / None-auto)
# ----------------------------------------------------------------------

# First-seen order is deliberately non-alphabetical / non-ascending so that a
# real reorder is observable.
_DICT_SORT_DATA = [("zeta", 1.0), ("alpha", 2.0), ("mike", 3.0), ("alpha", 4.0), ("zeta", 5.0)]
_INT_SORT_DATA = [(30, 1.0), (10, 2.0), (20, 3.0), (10, 4.0)]
_FLOAT_SORT_DATA = [(3.5, 1.0), (1.5, 2.0), (2.5, 3.0), (1.5, 4.0)]


def _keys(out):
    return [out._cols[out.col_names[0]][i] for i in range(out.nrows)]


def test_groupby_int_key_always_ascending_regardless_of_sort():
    # Integer/dense keys come out ascending under every sort= value -- nonzero
    # ordering is free and unavoidable.
    t = CTable(Int32FloatRow, new_data=_INT_SORT_DATA)
    for sort in (None, True, False):
        assert _keys(t.group_by("key", sort=sort).sum("value")) == [10, 20, 30]


def test_groupby_dict_key_sorted_under_auto_and_true():
    # Dictionary keys are cheap to sort, so None (auto) and True both sort by
    # string; False keeps first-seen code order.
    t = CTable(DictFloatRow, new_data=_DICT_SORT_DATA)
    assert _keys(t.group_by("key").sum("value")) == ["alpha", "mike", "zeta"]  # default None
    assert _keys(t.group_by("key", sort=True).sum("value")) == ["alpha", "mike", "zeta"]
    assert _keys(t.group_by("key", sort=False).sum("value")) == ["zeta", "alpha", "mike"]


def test_groupby_float_key_unsorted_under_auto_sorted_under_true():
    # Float keys only sort via a Python list.sort, so None (auto) leaves them
    # unsorted; True sorts. The unsorted order must be deterministic across runs.
    t = CTable(Float64KeyRow, new_data=_FLOAT_SORT_DATA)
    assert _keys(t.group_by("key", sort=True).sum("value")) == [1.5, 2.5, 3.5]
    auto1 = _keys(t.group_by("key").sum("value"))
    auto2 = _keys(t.group_by("key").sum("value"))
    assert auto1 == auto2  # deterministic
    assert sorted(auto1) == [1.5, 2.5, 3.5]  # same groups, order unspecified


def test_groupby_multikey_unsorted_under_auto_sorted_under_true():
    # Multi-key results only sort via a Python list.sort, so None (auto) leaves
    # them unsorted (deterministic but unspecified order); True sorts.
    data = [("z", 2, 1.0), ("a", 1, 2.0), ("z", 1, 3.0), ("a", 1, 4.0)]
    t = CTable(DictIntKeyFloatRow, new_data=data)

    def keypairs(out):
        return [(str(r[0]), int(r[1])) for r in rows(out)]

    expected = {("a", 1), ("z", 1), ("z", 2)}
    assert keypairs(t.group_by(["key0", "key1"], sort=True).sum("value")) == [
        ("a", 1),
        ("z", 1),
        ("z", 2),
    ]
    auto1 = keypairs(t.group_by(["key0", "key1"]).sum("value"))
    auto2 = keypairs(t.group_by(["key0", "key1"]).sum("value"))
    assert auto1 == auto2  # deterministic
    assert set(auto1) == expected  # same groups, order unspecified


def test_group_reduce_tristate_sort():
    # group_reduce mirrors the tri-state. Its float path is vectorized
    # (np.argsort), so unlike CTable's float-hash it *does* sort under None.
    keys = blosc2.array([3, 1, 2, 1])
    values = blosc2.array([1.0, 2.0, 3.0, 4.0])
    g_true, _ = blosc2.group_reduce(keys, values, op="sum", sort=True)
    assert list(g_true) == [1, 2, 3]
    g_auto, _ = blosc2.group_reduce(keys, values, op="sum")  # default None, cheap -> sorted
    assert list(g_auto) == [1, 2, 3]
    g_false, _ = blosc2.group_reduce(keys, values, op="sum", sort=False)
    assert sorted(g_false) == [1, 2, 3]  # order unspecified, same groups


# ----------------------------------------------------------------------
# agg() output column naming: auto suffix vs explicit named aggregation
# ----------------------------------------------------------------------


def test_agg_auto_suffix_names():
    t = CTable(SalesRow, new_data=DATA)
    out = t.group_by("city", sort=True).agg({"sales": ["sum", "mean"]})
    assert out.col_names == ["city", "sales_sum", "sales_mean"]


def test_agg_explicit_named_kwargs():
    t = CTable(SalesRow, new_data=DATA)
    out = t.group_by("city", sort=True).agg(revenue=("sales", "sum"), avg_sale=("sales", "mean"))
    assert out.col_names == ["city", "revenue", "avg_sale"]
    got = rows(out)
    assert got[1][0] == "Paris"
    assert got[1][1] == 40.0  # revenue == sales_sum
    assert got[1][2] == 20.0  # avg_sale == sales_mean


def test_agg_combines_mapping_and_named():
    t = CTable(SalesRow, new_data=DATA)
    out = t.group_by("city", sort=True).agg({"sales": "sum"}, n=("*", "size"))
    assert out.col_names == ["city", "sales_sum", "n"]
    got = rows(out)
    assert got[0][0] == "Berlin"
    assert np.isnan(got[0][1])
    assert got[0][2] == 1
    assert got[1] == ("Paris", 40.0, 3)
    assert got[2] == ("Rome", 60.0, 2)


def test_agg_list_of_pairs_auto_named():
    t = CTable(SalesRow, new_data=DATA)
    out = t.group_by("city", sort=True).agg([("sales", ["sum", "mean"])])
    assert out.col_names == ["city", "sales_sum", "sales_mean"]


def test_agg_list_of_pairs_accepts_column_objects():
    # The list form's whole point: use Column objects, which can't be dict keys.
    t = CTable(SalesRow, new_data=DATA)
    out = t.group_by("city", sort=True).agg([(t.sales, ["sum", "mean"])])
    assert out.col_names == ["city", "sales_sum", "sales_mean"]
    got = rows(out)
    assert got[1] == ("Paris", 40.0, 20.0)


def test_agg_list_of_pairs_combines_with_named():
    t = CTable(SalesRow, new_data=DATA)
    out = t.group_by("city", sort=True).agg([(t.sales, "sum")], n=("*", "size"))
    assert out.col_names == ["city", "sales_sum", "n"]


def test_agg_positional_must_be_mapping_or_pairs():
    t = CTable(SalesRow, new_data=DATA)
    with pytest.raises(ValueError, match="mapping or a list of"):
        t.group_by("city").agg("sales")
    with pytest.raises(ValueError, match=r"must contain \(column, ops\) pairs"):
        t.group_by("city").agg([("sales",)])


def test_agg_named_accepts_column_objects():
    # A Column object can't be a dict key (unhashable: __eq__ is overloaded for
    # expressions), but it works as a named-agg value, like group_by() args.
    t = CTable(SalesRow, new_data=DATA)
    g = t.group_by("city", sort=True)
    with pytest.raises(TypeError, match="unhashable"):
        _ = {t.sales: ["sum"]}  # fails at dict construction, before agg() runs
    out = g.agg(total=(t.sales, "sum"), avg=(t.sales, "mean"))
    assert out.col_names == ["city", "total", "avg"]


def test_agg_accepts_blosc2_reduction_functions():
    # blosc2 reduction functions are accepted as ops (matched by identity),
    # interchangeably with strings, in both the list and named forms.
    t = CTable(SalesRow, new_data=DATA)
    g = t.group_by("city", sort=True)
    by_func = g.agg([(t.sales, [blosc2.sum, blosc2.mean])])
    by_str = g.agg([(t.sales, ["sum", "mean"])])
    assert by_func.col_names == by_str.col_names == ["city", "sales_sum", "sales_mean"]
    assert col(by_func, "city") == col(by_str, "city")
    np.testing.assert_array_equal(  # NaN-safe (Berlin has no non-null sales)
        col(by_func, "sales_sum"), col(by_str, "sales_sum")
    )
    named = g.agg(revenue=(t.sales, blosc2.sum))
    assert named.col_names == ["city", "revenue"]


def test_agg_rejects_non_blosc2_callables_by_identity():
    # A UDF that merely shares a builtin op name must NOT be silently accepted;
    # np.sum / builtin sum are likewise rejected (only blosc2.* by identity).
    t = CTable(SalesRow, new_data=DATA)
    g = t.group_by("city")

    def sum(values):
        return -1

    for fn in (sum, np.sum):
        with pytest.raises(ValueError, match="Unsupported aggregation function"):
            g.agg([(t.sales, fn)])
    # blosc2.std is a real function but not a supported group-by op.
    with pytest.raises(ValueError, match="Unsupported aggregation function"):
        g.agg([(t.sales, blosc2.std)])


def test_agg_named_star_size():
    t = CTable(SalesRow, new_data=DATA)
    out = t.group_by("city", sort=True).agg(total=("*", "size"))
    assert out.col_names == ["city", "total"]
    assert rows(out) == [("Berlin", 1), ("Paris", 3), ("Rome", 2)]


def test_agg_requires_some_aggregation():
    t = CTable(SalesRow, new_data=DATA)
    with pytest.raises(ValueError, match="requires a mapping"):
        t.group_by("city").agg()


def test_agg_named_must_be_column_op_pair():
    t = CTable(SalesRow, new_data=DATA)
    with pytest.raises(ValueError, match=r"must be a \(column, op\) or \(column, op, dtype\) tuple"):
        t.group_by("city").agg(x=("sales",))


def test_agg_named_rejects_multiple_ops_with_guidance():
    # A named output maps to a single op; multiple ops need the mapping form or
    # one name each. The error should say so, not "unsupported aggregation".
    t = CTable(SalesRow, new_data=DATA)
    with pytest.raises(ValueError, match="takes a single op"):
        t.group_by("city").agg(total=("sales", ("sum", "mean")))


def test_agg_duplicate_output_names_rejected():
    t = CTable(SalesRow, new_data=DATA)
    with pytest.raises(ValueError, match="must be unique"):
        t.group_by("city").agg({"sales": "sum"}, sales_sum=("sales", "sum"))


# ===========================================================================
# UDF aggregations (Gap D2) -- named form only
# ===========================================================================


def test_agg_udf_matches_pandas_reference():
    pd = pytest.importorskip("pandas")
    t = CTable(SalesRow, new_data=DATA)
    result = t.group_by("city", sort=True).agg(rng=("sales", lambda a: a.max() - a.min()))

    df = pd.DataFrame(DATA, columns=["city", "category", "sales", "qty"])
    # Berlin's only row has NaN sales, so unlike df.dropna()-then-groupby (which
    # would drop the row and the whole group with it), blosc2 keeps Berlin as a
    # group -- its key is not null -- with a null (NaN) result, the same
    # convention built-in sum/min/max already use for an all-null group.
    expected = (
        df.groupby("city")["sales"]
        .apply(lambda a: (a.dropna().max() - a.dropna().min()) if a.notna().any() else float("nan"))
        .sort_index()
    )
    np.testing.assert_allclose(col(result, "rng"), expected.to_numpy())
    assert col(result, "city") == list(expected.index)


def test_agg_udf_receives_only_live_nonnull_values():
    seen = []

    def probe(values):
        seen.append(np.sort(values).tolist())
        return float(values.sum())

    t = CTable(SalesRow, new_data=DATA)
    t.group_by("city", sort=True).agg(total=("sales", probe))
    # Berlin's only row has NaN sales -> the UDF is never called for it (an
    # empty group produces a null result directly, like sum/min/max do);
    # Paris/Rome nulls are dropped before the UDF sees their values.
    assert seen == [[10.0, 30.0], [20.0, 40.0]]


def test_agg_udf_requires_named_form():
    t = CTable(SalesRow, new_data=DATA)
    g = t.group_by("city")
    with pytest.raises(ValueError, match="Unsupported aggregation function"):
        g.agg({"sales": lambda a: a.sum()})
    with pytest.raises(ValueError, match="Unsupported aggregation function"):
        g.agg([(t.sales, lambda a: a.sum())])


def test_agg_udf_infers_dtype_from_all_groups():
    t = CTable(SalesRow, new_data=DATA)
    result = t.group_by("city").agg(rng=("sales", lambda a: a.max() - a.min() if len(a) else 0.0))
    assert result["rng"].dtype == np.float64


def test_agg_udf_explicit_dtype():
    t = CTable(SalesRow, new_data=DATA)
    result = t.group_by("city").agg(
        rng=("sales", lambda a: a.max() - a.min() if len(a) else 0.0, blosc2.float32())
    )
    assert result["rng"].dtype == np.float32


def test_agg_udf_inconsistent_types_raise_clear_error():
    t = CTable(SalesRow, new_data=DATA)
    g = t.group_by("city")
    calls = {"n": 0}

    def inconsistent(values):
        # A shape-inconsistent return (list vs. scalar) forces NumPy to fall
        # back to an object array when collecting results across groups.
        calls["n"] += 1
        return [1, 2, 3] if calls["n"] == 1 else float(values.sum())

    with pytest.raises(ValueError, match="inconsistent or unsupported types"):
        g.agg(x=("sales", inconsistent))


def test_agg_udf_unsupported_result_dtype_raises_clear_error():
    t = CTable(SalesRow, new_data=DATA)
    g = t.group_by("city")

    with pytest.raises(ValueError, match="Cannot infer a CTable dtype"):
        g.agg(x=("sales", lambda a: "always-a-string"))


def test_agg_udf_never_called_raises_clear_error():
    # Every group has zero non-null "sales" values, so the UDF is never
    # called for any of them -- there is nothing to infer a dtype from.
    t = CTable(SalesRow, new_data=[("Paris", 1, np.nan, 0), ("Rome", 1, np.nan, 0)])
    g = t.group_by("city")

    with pytest.raises(ValueError, match="it was never called"):
        g.agg(x=("sales", lambda a: a.max() - a.min()))


def test_agg_udf_result_not_wrapped_in_zero_d_array(monkeypatch):
    # Regression test: the UDF result must reach _python_scalar() directly,
    # not pre-wrapped in np.asarray() -- a plain Python/NumPy scalar wrapped
    # in np.asarray() becomes a 0-D ndarray, which _python_scalar() (only
    # unwraps np.generic) then fails to turn back into a plain scalar.
    import blosc2.groupby as gb_module

    seen_types = []
    original = gb_module._python_scalar
    monkeypatch.setattr(gb_module, "_python_scalar", lambda v: (seen_types.append(type(v)), original(v))[1])

    t = CTable(SalesRow, new_data=[("Paris", 1, 10.0, 0), ("Rome", 1, 20.0, 0)])
    g = t.group_by("city")
    g.agg(x=("sales", lambda a: float(a.sum())))

    assert np.ndarray not in seen_types


def test_agg_udf_error_names_the_group_key():
    t = CTable(SalesRow, new_data=DATA)
    g = t.group_by("city")

    def boom(values):
        raise KeyError("nope")

    with pytest.raises(RuntimeError, match=r"raised for group \('(Paris|Rome|Berlin)',\)"):
        g.agg(x=("sales", boom))


def test_agg_udf_combines_with_builtin_ops():
    t = CTable(SalesRow, new_data=DATA)
    result = t.group_by("city", sort=True).agg(
        total=("sales", "sum"), rng=("sales", lambda a: a.max() - a.min())
    )
    assert result.col_names == ["city", "total", "rng"]
    # Berlin's sum is NaN too (all-null group) -- the existing sum() convention.
    np.testing.assert_allclose(col(result, "total"), [np.nan, 40.0, 60.0])
