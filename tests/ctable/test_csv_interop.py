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
    np.testing.assert_array_equal(t2["id"].to_numpy(), table10["id"].to_numpy())


def test_from_csv_float_values(table10, tmp_csv):
    table10.to_csv(tmp_csv)
    t2 = CTable.from_csv(tmp_csv, Row)
    np.testing.assert_allclose(t2["score"].to_numpy(), table10["score"].to_numpy())


def test_from_csv_bool_values(table10, tmp_csv):
    table10.to_csv(tmp_csv)
    t2 = CTable.from_csv(tmp_csv, Row)
    # bool is serialised as "True"/"False"; np.array(..., dtype=bool) parses that
    np.testing.assert_array_equal(t2["active"].to_numpy(), table10["active"].to_numpy())


def test_from_csv_string_values(table10, tmp_csv):
    table10.to_csv(tmp_csv)
    t2 = CTable.from_csv(tmp_csv, Row)
    assert t2["label"].to_numpy().tolist() == table10["label"].to_numpy().tolist()


def test_from_csv_no_header(table10, tmp_csv):
    table10.to_csv(tmp_csv, header=False)
    t2 = CTable.from_csv(tmp_csv, Row, header=False)
    assert len(t2) == 10
    np.testing.assert_array_equal(t2["id"].to_numpy(), table10["id"].to_numpy())


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
        np.testing.assert_array_equal(t2[name].to_numpy(), table10[name].to_numpy())
    np.testing.assert_array_equal(t2["active"].to_numpy(), table10["active"].to_numpy())
    assert t2["label"].to_numpy().tolist() == table10["label"].to_numpy().tolist()


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


if __name__ == "__main__":
    pytest.main(["-v", __file__])
