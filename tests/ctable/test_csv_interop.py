#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""Tests for CTable.to_csv() and CTable.from_csv()."""

import csv
import os
from dataclasses import dataclass

import numpy as np
import pytest

import blosc2
from blosc2 import CTable


@dataclass
class Row:
    id: int = blosc2.field(blosc2.int64(ge=0))
    score: float = blosc2.field(blosc2.float64(ge=0, le=100), default=0.0)
    active: bool = blosc2.field(blosc2.bool(), default=True)
    label: str = blosc2.field(blosc2.string(max_length=16), default="")


DATA10 = [(i, float(i * 10 % 100), i % 2 == 0, f"r{i}") for i in range(10)]


@pytest.fixture
def tmp_csv(tmp_path):
    return str(tmp_path / "table.csv")


@pytest.fixture
def table10():
    return CTable(Row, new_data=DATA10)


# ===========================================================================
# to_csv()
# ===========================================================================


def test_to_csv_creates_file(table10, tmp_csv):
    table10.to_csv(tmp_csv)
    assert os.path.exists(tmp_csv)


def test_to_csv_header_row(table10, tmp_csv):
    table10.to_csv(tmp_csv)
    with open(tmp_csv) as f:
        first = f.readline().strip()
    assert first == "id,score,active,label"


def test_to_csv_row_count(table10, tmp_csv):
    table10.to_csv(tmp_csv)
    with open(tmp_csv) as f:
        rows = list(csv.reader(f))
    assert len(rows) == 11  # 1 header + 10 data


def test_to_csv_no_header(table10, tmp_csv):
    table10.to_csv(tmp_csv, header=False)
    with open(tmp_csv) as f:
        rows = list(csv.reader(f))
    assert len(rows) == 10


def test_to_csv_int_values(table10, tmp_csv):
    table10.to_csv(tmp_csv)
    with open(tmp_csv) as f:
        reader = csv.DictReader(f)
        ids = [int(row["id"]) for row in reader]
    assert ids == list(range(10))


def test_to_csv_float_values(table10, tmp_csv):
    table10.to_csv(tmp_csv)
    with open(tmp_csv) as f:
        reader = csv.DictReader(f)
        scores = [float(row["score"]) for row in reader]
    assert scores == [r[1] for r in DATA10]


def test_to_csv_bool_values(table10, tmp_csv):
    table10.to_csv(tmp_csv)
    with open(tmp_csv) as f:
        reader = csv.DictReader(f)
        actives = [row["active"] for row in reader]
    # numpy bool serialises as "True"/"False"
    assert actives == [str(r[2]) for r in DATA10]


def test_to_csv_string_values(table10, tmp_csv):
    table10.to_csv(tmp_csv)
    with open(tmp_csv) as f:
        reader = csv.DictReader(f)
        labels = [row["label"] for row in reader]
    assert labels == [r[3] for r in DATA10]


def test_to_csv_custom_separator(table10, tmp_csv):
    table10.to_csv(tmp_csv, sep="\t")
    with open(tmp_csv) as f:
        first = f.readline().strip()
    assert "\t" in first
    assert "," not in first


def test_to_csv_skips_deleted_rows(table10, tmp_csv):
    table10.delete([0, 1])
    table10.to_csv(tmp_csv)
    with open(tmp_csv) as f:
        rows = list(csv.reader(f))
    assert len(rows) == 9  # 1 header + 8 live rows
    assert rows[1][0] == "2"  # first live id


def test_to_csv_empty_table(tmp_csv):
    t = CTable(Row)
    t.to_csv(tmp_csv)
    with open(tmp_csv) as f:
        rows = list(csv.reader(f))
    assert rows == [["id", "score", "active", "label"]]


def test_to_csv_select_view(table10, tmp_csv):
    table10.select(["id", "label"]).to_csv(tmp_csv)
    with open(tmp_csv) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert list(rows[0].keys()) == ["id", "label"]
    assert len(rows) == 10


# ===========================================================================
# from_csv()
# ===========================================================================


def test_from_csv_returns_ctable(table10, tmp_csv):
    table10.to_csv(tmp_csv)
    t2 = CTable.from_csv(tmp_csv, Row)
    assert isinstance(t2, CTable)


def test_from_csv_row_count(table10, tmp_csv):
    table10.to_csv(tmp_csv)
    t2 = CTable.from_csv(tmp_csv, Row)
    assert len(t2) == 10


def test_from_csv_column_names(table10, tmp_csv):
    table10.to_csv(tmp_csv)
    t2 = CTable.from_csv(tmp_csv, Row)
    assert t2.col_names == ["id", "score", "active", "label"]


def test_from_csv_int_values(table10, tmp_csv):
    table10.to_csv(tmp_csv)
    t2 = CTable.from_csv(tmp_csv, Row)
    np.testing.assert_array_equal(t2["id"][:], table10["id"][:])


def test_from_csv_float_values(table10, tmp_csv):
    table10.to_csv(tmp_csv)
    t2 = CTable.from_csv(tmp_csv, Row)
    np.testing.assert_allclose(t2["score"][:], table10["score"][:])


def test_from_csv_bool_values(table10, tmp_csv):
    table10.to_csv(tmp_csv)
    t2 = CTable.from_csv(tmp_csv, Row)
    # bool is serialised as "True"/"False"; np.array(..., dtype=bool) parses that
    np.testing.assert_array_equal(t2["active"][:], table10["active"][:])


def test_from_csv_string_values(table10, tmp_csv):
    table10.to_csv(tmp_csv)
    t2 = CTable.from_csv(tmp_csv, Row)
    assert t2["label"][:].tolist() == table10["label"][:].tolist()


def test_from_csv_no_header(table10, tmp_csv):
    table10.to_csv(tmp_csv, header=False)
    t2 = CTable.from_csv(tmp_csv, Row, header=False)
    assert len(t2) == 10
    np.testing.assert_array_equal(t2["id"][:], table10["id"][:])


def test_from_csv_custom_separator(table10, tmp_csv):
    table10.to_csv(tmp_csv, sep="\t")
    t2 = CTable.from_csv(tmp_csv, Row, sep="\t")
    assert len(t2) == 10


def test_from_csv_empty_file(tmp_csv):
    with open(tmp_csv, "w") as f:
        f.write("id,score,active,label\n")
    t = CTable.from_csv(tmp_csv, Row)
    assert len(t) == 0
    assert t.col_names == ["id", "score", "active", "label"]


def test_from_csv_roundtrip(table10, tmp_csv):
    """to_csv then from_csv preserves all values."""
    table10.to_csv(tmp_csv)
    t2 = CTable.from_csv(tmp_csv, Row)
    for name in ["id", "score"]:
        np.testing.assert_array_equal(t2[name][:], table10[name][:])
    np.testing.assert_array_equal(t2["active"][:], table10["active"][:])
    assert t2["label"][:].tolist() == table10["label"][:].tolist()


def test_from_csv_wrong_field_count_raises(tmp_csv):
    with open(tmp_csv, "w") as f:
        f.write("id,score,active,label\n")
        f.write("1,2.0\n")  # only 2 fields instead of 4
    with pytest.raises(ValueError, match="expected 4 fields"):
        CTable.from_csv(tmp_csv, Row)


def test_from_csv_not_dataclass_raises(tmp_csv):
    with open(tmp_csv, "w") as f:
        f.write("id\n1\n")
    with pytest.raises(TypeError):
        CTable.from_csv(tmp_csv, int)


# ===========================================================================
# from_csv / to_csv with fixed-shape ndarray columns
# ===========================================================================


@dataclass
class NdarrayRow:
    id: int = blosc2.field(blosc2.int64(ge=0))
    embedding: object = blosc2.field(blosc2.ndarray((3,), dtype=blosc2.float32()))


@dataclass
class NullableNdarrayRow:
    id: int = blosc2.field(blosc2.int32())
    codes: object = blosc2.field(blosc2.ndarray((2,), dtype=blosc2.int16(), nullable=True))
    image: object = blosc2.field(blosc2.ndarray((2, 2, 3), dtype=blosc2.float32(), nullable=True))


@pytest.fixture
def ndarray_table():
    return CTable(
        NdarrayRow,
        new_data=[
            (1, np.array([1.0, 2.0, 3.0], dtype=np.float32)),
            (2, np.array([4.0, 5.0, 6.0], dtype=np.float32)),
        ],
    )


@pytest.fixture
def nullable_ndarray_table():
    return CTable(
        NullableNdarrayRow,
        new_data=[
            (1, None, None),
            (2, np.array([10, 20], dtype=np.int16), np.ones((2, 2, 3), dtype=np.float32)),
            (3, np.array([1, 2], dtype=np.int16), None),
        ],
    )


def test_to_csv_ndarray_roundtrip(ndarray_table, tmp_csv):
    """1-D ndarray column values are serialised as JSON arrays and restored correctly."""
    ndarray_table.to_csv(tmp_csv)
    t2 = CTable.from_csv(tmp_csv, NdarrayRow)
    assert len(t2) == 2
    np.testing.assert_array_equal(t2["id"][:], ndarray_table["id"][:])
    np.testing.assert_array_equal(t2["embedding"][:], ndarray_table["embedding"][:])


def test_to_csv_ndarray_json_format(ndarray_table, tmp_csv):
    """CSV cells hold valid JSON arrays."""
    import csv
    import json

    ndarray_table.to_csv(tmp_csv)
    with open(tmp_csv) as f:
        reader = csv.reader(f)
        header = next(reader)
        row1 = next(reader)
    assert header == ["id", "embedding"]
    # CSV cell is a JSON array string; parse it
    row1_embedding = json.loads(row1[1])
    assert row1_embedding == [1.0, 2.0, 3.0]


def test_to_csv_nullable_ndarray_writes_empty_cells(nullable_ndarray_table, tmp_csv):
    """Null ndarray cells serialise as empty CSV fields."""
    nullable_ndarray_table.to_csv(tmp_csv)
    with open(tmp_csv) as f:
        lines = f.read().strip().split("\n")
    # Row 0: id=1, codes=null, image=null  → "1,,"
    assert lines[1].endswith(",")


def test_from_csv_nullable_ndarray_restores_nulls(nullable_ndarray_table, tmp_csv):
    """Empty CSV cells restore as null sentinel arrays."""
    nullable_ndarray_table.to_csv(tmp_csv)
    t2 = CTable.from_csv(tmp_csv, NullableNdarrayRow)
    assert t2["codes"].null_count() == 1
    assert t2["image"].null_count() == 2
    np.testing.assert_array_equal(t2["codes"].is_null(), np.array([True, False, False]))
    np.testing.assert_array_equal(t2["image"].is_null(), np.array([True, False, True]))


def test_to_csv_empty_ndarray_table(tmp_csv):
    """Empty table with ndarray columns writes header only."""
    t = CTable(NdarrayRow)
    t.to_csv(tmp_csv)
    with open(tmp_csv) as f:
        content = f.read().strip()
    assert content == "id,embedding"


def test_to_csv_ndarray_select_view(ndarray_table, tmp_csv):
    """Column-projection view with ndarray columns writes correctly."""
    ndarray_table.select(["id", "embedding"]).to_csv(tmp_csv)
    with open(tmp_csv) as f:
        lines = f.read().strip().split("\n")
    assert lines[0] == "id,embedding"
    assert len(lines) == 3  # header + 2 rows


def test_from_csv_ndarray_wrong_shape_raises(tmp_csv):
    """CSV cell with wrong-shaped JSON array raises ValueError."""
    with open(tmp_csv, "w") as f:
        f.write("id,embedding\n")
        f.write('1,"[1.0, 2.0]"\n')  # only 2 elements, expected 3
    with pytest.raises(ValueError, match="expected item shape"):
        CTable.from_csv(tmp_csv, NdarrayRow)


# ===========================================================================
# to_pandas / from_pandas with fixed-shape ndarray columns
# ===========================================================================


@pytest.fixture
def pandas_table():
    pytest.importorskip("pandas")
    return CTable(
        NdarrayRow,
        new_data=[
            (1, np.array([1.0, 2.0, 3.0], dtype=np.float32)),
            (2, np.array([4.0, 5.0, 6.0], dtype=np.float32)),
        ],
    )


def test_to_pandas_scalar_columns(pandas_table):
    """Scalar columns become regular DataFrame columns."""
    df = pandas_table.to_pandas()
    assert df["id"].tolist() == [1, 2]
    assert df["id"].dtype == np.int64


def test_to_pandas_ndarray_columns_are_object_dtype(pandas_table):
    """Ndarray columns become object-dtype with NumPy arrays in each cell."""
    df = pandas_table.to_pandas()
    assert df["embedding"].dtype == object
    np.testing.assert_array_equal(df["embedding"][0], np.array([1.0, 2.0, 3.0], dtype=np.float32))
    np.testing.assert_array_equal(df["embedding"][1], np.array([4.0, 5.0, 6.0], dtype=np.float32))


def test_from_pandas_roundtrip(pandas_table):
    """DataFrame roundtrip through from_pandas preserves all values."""
    df = pandas_table.to_pandas()
    t2 = CTable.from_pandas(df, NdarrayRow)
    np.testing.assert_array_equal(t2["id"][:], pandas_table["id"][:])
    np.testing.assert_array_equal(t2["embedding"][:], pandas_table["embedding"][:])


def test_from_pandas_missing_columns_raises(pandas_table):
    """DataFrame missing schema columns raises ValueError."""
    pytest.importorskip("pandas")
    import pandas as pd

    df = pd.DataFrame({"id": [1, 2]})
    with pytest.raises(ValueError, match="missing columns"):
        CTable.from_pandas(df, NdarrayRow)


def test_from_pandas_extra_columns_raises(pandas_table):
    """DataFrame with extra columns beyond schema raises ValueError."""
    pytest.importorskip("pandas")
    import pandas as pd

    df = pd.DataFrame(
        {"id": [1, 2], "embedding": [np.array([1, 2, 3], dtype=np.float32)] * 2, "extra": [99, 100]}
    )
    with pytest.raises(ValueError, match="has extra columns"):
        CTable.from_pandas(df, NdarrayRow)


def test_from_pandas_ndarray_not_object_dtype_raises():
    """Non-object scalar column mapped to ndarray schema raises."""
    pytest.importorskip("pandas")
    import pandas as pd

    df = pd.DataFrame({"id": [1, 2], "embedding": [1.0, 2.0]})  # float column, not object
    with pytest.raises(ValueError, match="expected object dtype"):
        CTable.from_pandas(df, NdarrayRow)


@dataclass
class AllScalarsRow:
    a_int: int = blosc2.field(blosc2.int64())
    a_float: float = blosc2.field(blosc2.float64())
    a_bool: bool = blosc2.field(blosc2.bool())
    a_str: str = blosc2.field(blosc2.string(max_length=32))


def test_from_pandas_all_scalars_roundtrip():
    """DataFrame with only scalar columns roundtrips correctly."""
    pytest.importorskip("pandas")
    import pandas as pd

    df = pd.DataFrame(
        {
            "a_int": [1, 2, 3],
            "a_float": [1.0, 2.0, 3.0],
            "a_bool": [True, False, True],
            "a_str": ["hello", "world", "!"],
        }
    )
    t = CTable.from_pandas(df, AllScalarsRow)
    assert len(t) == 3
    np.testing.assert_array_equal(t["a_int"][:], df["a_int"].to_numpy())
    np.testing.assert_array_equal(t["a_float"][:], df["a_float"].to_numpy())
    np.testing.assert_array_equal(t["a_bool"][:], df["a_bool"].to_numpy())
    assert t["a_str"][:].tolist() == df["a_str"].tolist()


@dataclass
class PandasSpecialRow:
    vendor: str = blosc2.field(blosc2.dictionary())
    note: str = blosc2.field(blosc2.vlstring(nullable=True))
    tags: list[str] = blosc2.field(blosc2.list(blosc2.string(max_length=16), nullable=True))  # noqa: RUF009


def test_from_pandas_special_columns_roundtrip():
    """from_pandas creates the specialized backing storage required by non-ndarray columns."""
    pytest.importorskip("pandas")
    import pandas as pd

    df = pd.DataFrame(
        {
            "vendor": ["Uber", "Lyft", "Uber"],
            "note": ["fast", None, "ok"],
            "tags": [["airport", "night"], None, []],
        }
    )

    t = CTable.from_pandas(df, PandasSpecialRow)

    assert t["vendor"][:] == ["Uber", "Lyft", "Uber"]
    assert t["note"][:] == ["fast", None, "ok"]
    assert t["tags"][:] == [["airport", "night"], None, []]


@dataclass
class NdarrayMixedRow:
    id: int = blosc2.field(blosc2.int64())
    image: object = blosc2.field(blosc2.ndarray((2, 2, 3), dtype=blosc2.float32()))
    label: str = blosc2.field(blosc2.string(max_length=16))


def test_to_pandas_multi_dim_ndarray():
    """Multi-dimensional ndarray columns export as object-dtype cells."""
    pytest.importorskip("pandas")
    t = CTable(
        NdarrayMixedRow,
        new_data=[
            (1, np.ones((2, 2, 3), dtype=np.float32), "a"),
            (2, np.full((2, 2, 3), 2.0, dtype=np.float32), "b"),
        ],
    )
    df = t.to_pandas()
    assert df["image"].dtype == object
    np.testing.assert_array_equal(df["image"][0], np.ones((2, 2, 3), dtype=np.float32))
    assert df["label"].tolist() == ["a", "b"]


def test_from_pandas_multi_dim_ndarray_roundtrip():
    """Multi-dimensional ndarray roundtrip from DataFrame works."""
    pytest.importorskip("pandas")
    t = CTable(
        NdarrayMixedRow,
        new_data=[
            (1, np.ones((2, 2, 3), dtype=np.float32), "a"),
        ],
    )
    df = t.to_pandas()
    t2 = CTable.from_pandas(df, NdarrayMixedRow)
    np.testing.assert_array_equal(t2["image"][:], t["image"][:])
    assert t2["label"][:].tolist() == t["label"][:].tolist()


if __name__ == "__main__":
    pytest.main(["-v", __file__])
