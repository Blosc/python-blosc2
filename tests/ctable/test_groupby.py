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
            [("a", 5.5), ("c", 13.0), ("b", 2.5)],
        ),
    ],
)
def test_groupby_fast_path_sum_variants(row_type, data, expected):
    t = CTable(row_type, new_data=data)

    out = t.group_by("key").agg({"value": "sum"})

    assert rows(out) == expected


def test_groupby_float_integral_fast_path_falls_back_for_non_integral_keys():
    t = CTable(Float64KeyRow, new_data=[(0.5, 1.0), (1.5, 2.0), (0.5, 3.0)])

    out = t.group_by("key").agg({"value": "sum"})

    assert rows(out) == [(0.5, 4.0), (1.5, 2.0)]


def test_groupby_float_integral_fast_path_falls_back_for_nan_group_when_kept():
    t = CTable(Float64KeyRow, new_data=[(0.0, 1.0), (np.nan, 2.0), (0.0, 3.0)])

    out = t.group_by("key", dropna=False).agg({"value": "sum"})

    got = rows(out)
    assert got[0] == (0.0, 4.0)
    assert np.isnan(got[1][0])
    assert got[1][1] == 2.0


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

    out = t.group_by("key").agg({"value": ["count", "sum", "mean", "min", "max"]})

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
