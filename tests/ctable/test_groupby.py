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
