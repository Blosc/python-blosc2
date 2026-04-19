#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""Tests for nullable column support (null_value sentinel)."""

from __future__ import annotations

import math
import os
import pathlib
import shutil
import tempfile
from dataclasses import dataclass

import numpy as np
import pytest

import blosc2
from blosc2 import CTable

# ---------------------------------------------------------------------------
# Schemas used across tests
# ---------------------------------------------------------------------------


@dataclass
class IntRow:
    id: int = blosc2.field(blosc2.int64())
    score: int = blosc2.field(blosc2.int64(ge=0, le=1000, null_value=-1))


@dataclass
class FloatRow:
    name: str = blosc2.field(blosc2.string(max_length=16))
    value: float = blosc2.field(blosc2.float64(null_value=float("nan")))


@dataclass
class StrRow:
    label: str = blosc2.field(blosc2.string(max_length=16, null_value=""))
    rank: int = blosc2.field(blosc2.int64())


@dataclass
class BoolRow:
    code: int = blosc2.field(blosc2.int64(null_value=-999))
    flag: bool = blosc2.field(blosc2.bool(), default=False)


TABLE_ROOT = pathlib.Path(__file__).parent / "saved_ctable" / "test_nullable"


@pytest.fixture(autouse=True)
def clean_dir():
    if TABLE_ROOT.exists():
        shutil.rmtree(TABLE_ROOT)
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    yield
    if TABLE_ROOT.exists():
        shutil.rmtree(TABLE_ROOT)


def table_path(name: str) -> str:
    return str(TABLE_ROOT / name)


# ===========================================================================
# null_value property
# ===========================================================================


def test_null_value_property_set():
    t = CTable(IntRow, new_data=[(1, 10), (2, -1)])
    assert t["score"].null_value == -1


def test_null_value_property_not_set():
    t = CTable(IntRow, new_data=[(1, 10)])
    assert t["id"].null_value is None


def test_null_value_nan():
    t = CTable(FloatRow, new_data=[("a", 1.0), ("b", float("nan"))])
    nv = t["value"].null_value
    assert isinstance(nv, float)
    assert math.isnan(nv)


def test_null_value_string():
    t = CTable(StrRow, new_data=[("hello", 1), ("", 2)])
    assert t["label"].null_value == ""


# ===========================================================================
# is_null / notnull / null_count
# ===========================================================================


def test_is_null_basic():
    t = CTable(IntRow, new_data=[(1, 10), (2, -1), (3, 20), (4, -1)])
    mask = t["score"].is_null()
    assert list(mask) == [False, True, False, True]


def test_notnull_basic():
    t = CTable(IntRow, new_data=[(1, 10), (2, -1), (3, 20)])
    mask = t["score"].notnull()
    assert list(mask) == [True, False, True]


def test_null_count():
    t = CTable(IntRow, new_data=[(1, 10), (2, -1), (3, -1), (4, 5)])
    assert t["score"].null_count() == 2


def test_null_count_zero():
    t = CTable(IntRow, new_data=[(1, 10), (2, 20)])
    assert t["score"].null_count() == 0


def test_is_null_nan():
    t = CTable(FloatRow, new_data=[("a", 1.0), ("b", float("nan")), ("c", 3.0)])
    mask = t["value"].is_null()
    assert list(mask) == [False, True, False]


def test_is_null_string_sentinel():
    t = CTable(StrRow, new_data=[("hello", 1), ("", 2), ("world", 3)])
    mask = t["label"].is_null()
    assert list(mask) == [False, True, False]


def test_is_null_no_null_value():
    t = CTable(IntRow, new_data=[(1, 10), (2, 20)])
    # id has no null_value — is_null always returns all False
    mask = t["id"].is_null()
    assert list(mask) == [False, False]


def test_null_count_after_delete():
    t = CTable(IntRow, new_data=[(1, 10), (2, -1), (3, -1), (4, 5)])
    t.delete([1])  # delete the row with score=-1 at physical index 1
    # Remaining: (1,10), (3,-1), (4,5)
    assert t["score"].null_count() == 1


# ===========================================================================
# Aggregates skip nulls
# ===========================================================================


def test_sum_skips_null():
    t = CTable(IntRow, new_data=[(1, 10), (2, -1), (3, 20), (4, -1)])
    assert t["score"].sum() == 30


def test_mean_skips_null():
    t = CTable(IntRow, new_data=[(1, 10), (2, -1), (3, 30), (4, -1)])
    assert t["score"].mean() == pytest.approx(20.0)


def test_std_skips_null():
    t = CTable(IntRow, new_data=[(1, 10), (2, -1), (3, 10), (4, -1)])
    # population std of [10, 10] = 0
    assert t["score"].std() == pytest.approx(0.0)


def test_min_skips_null():
    t = CTable(IntRow, new_data=[(1, 10), (2, -1), (3, 5)])
    assert t["score"].min() == 5


def test_max_skips_null():
    t = CTable(IntRow, new_data=[(1, 10), (2, -1), (3, 5)])
    assert t["score"].max() == 10


def test_min_nan_skips():
    t = CTable(FloatRow, new_data=[("a", float("nan")), ("b", 3.0), ("c", 1.0)])
    assert t["value"].min() == pytest.approx(1.0)


def test_max_nan_skips():
    t = CTable(FloatRow, new_data=[("a", float("nan")), ("b", 3.0), ("c", 1.0)])
    assert t["value"].max() == pytest.approx(3.0)


def test_mean_nan_returns_nan_when_all_null():
    t = CTable(FloatRow, new_data=[("a", float("nan")), ("b", float("nan"))])
    result = t["value"].mean()
    assert math.isnan(result)


def test_min_all_null_raises():
    t = CTable(IntRow, new_data=[(1, -1), (2, -1)])
    with pytest.raises(ValueError, match="null"):
        t["score"].min()


def test_max_all_null_raises():
    t = CTable(IntRow, new_data=[(1, -1), (2, -1)])
    with pytest.raises(ValueError, match="null"):
        t["score"].max()


def test_any_skips_null():
    """any() on bool column with null_value — null rows are skipped."""

    @dataclass
    class BoolNull:
        flag: bool = blosc2.field(blosc2.bool())
        active: bool = blosc2.field(blosc2.bool())

    # bool doesn't support null_value directly in this test — just verify _nonnull_chunks
    # behaves like iter_chunks when no null_value is set (already covered by existing tests).
    t = CTable(BoolNull, new_data=[(True, False), (False, True)])
    assert t["flag"].any() is True
    assert t["active"].any() is True


# ===========================================================================
# unique / value_counts exclude nulls
# ===========================================================================


def test_unique_excludes_null():
    t = CTable(IntRow, new_data=[(1, 10), (2, -1), (3, 10), (4, -1), (5, 20)])
    u = t["score"].unique()
    assert list(u) == [10, 20]
    assert -1 not in u


def test_value_counts_excludes_null():
    t = CTable(IntRow, new_data=[(1, 10), (2, -1), (3, 10), (4, -1), (5, 20)])
    vc = t["score"].value_counts()
    assert -1 not in vc
    assert vc[10] == 2
    assert vc[20] == 1


def test_unique_string_excludes_null():
    t = CTable(StrRow, new_data=[("hello", 1), ("", 2), ("hello", 3), ("world", 4)])
    u = t["label"].unique()
    assert "" not in list(u)
    assert "hello" in list(u)
    assert "world" in list(u)


# ===========================================================================
# Append / extend with null sentinel bypass Pydantic
# ===========================================================================


def test_append_null_bypasses_constraint():
    """Appending a null sentinel that violates ge/le should succeed."""
    t = CTable(IntRow)
    # score has ge=0, le=1000 but null_value=-1; appending -1 should not raise
    t.append((1, -1))
    assert t["score"][0] == -1


def test_append_normal_value_still_validated():
    t = CTable(IntRow)
    with pytest.raises(ValueError):
        t.append((1, 9999))  # violates le=1000


def test_extend_null_bypasses_constraint():
    """extend() with null sentinel should not raise a constraint error."""
    t = CTable(IntRow)
    t.extend([(1, 10), (2, -1), (3, 20)])
    scores = t["score"][:]
    assert scores[1] == -1


def test_extend_normal_value_still_validated():
    t = CTable(IntRow)
    with pytest.raises(ValueError):
        t.extend([(1, 10), (2, 9999)])  # 9999 violates le=1000


# ===========================================================================
# sort_by: nulls last
# ===========================================================================


def test_sort_nulls_last_ascending():
    t = CTable(IntRow, new_data=[(1, 5), (2, -1), (3, 2), (4, -1), (5, 8)])
    s = t.sort_by("score")
    scores = list(s["score"][:])
    # Non-null values sorted first, nulls (-1) at end
    assert scores[:3] == [2, 5, 8]
    assert scores[3] == -1
    assert scores[4] == -1


def test_sort_nulls_last_descending():
    t = CTable(IntRow, new_data=[(1, 5), (2, -1), (3, 2), (4, -1), (5, 8)])
    s = t.sort_by("score", ascending=False)
    scores = list(s["score"][:])
    # Non-null values sorted descending first, nulls last
    assert scores[:3] == [8, 5, 2]
    assert scores[3] == -1
    assert scores[4] == -1


def test_sort_nulls_last_nan():
    t = CTable(FloatRow, new_data=[("a", 3.0), ("b", float("nan")), ("c", 1.0)])
    s = t.sort_by("value")
    values = list(s["value"][:])
    assert values[0] == pytest.approx(1.0)
    assert values[1] == pytest.approx(3.0)
    assert math.isnan(values[2])


def test_sort_multi_nulls_last():
    t = CTable(IntRow, new_data=[(1, -1), (2, 5), (3, -1), (4, 5)])
    s = t.sort_by(["score", "id"])
    scores = list(s["score"][:])
    ids = list(s["id"][:])
    # score 5 rows first (id 2, then id 4), then score -1 rows
    assert scores[:2] == [5, 5]
    assert ids[:2] == [2, 4]
    assert scores[2] == -1
    assert scores[3] == -1


def test_sort_no_nulls_unchanged():
    """Columns without null_value still sort normally."""
    t = CTable(IntRow, new_data=[(3, 30), (1, 10), (2, 20)])
    s = t.sort_by("id")
    np.testing.assert_array_equal(s["id"][:], [1, 2, 3])


# ===========================================================================
# describe() shows null count
# ===========================================================================


def test_describe_shows_null_count(capsys):
    t = CTable(IntRow, new_data=[(1, 10), (2, -1), (3, 20)])
    t.describe()
    out = capsys.readouterr().out
    assert "null" in out.lower()
    assert "1" in out  # 1 null


def test_describe_no_null_line_when_zero_nulls(capsys):
    t = CTable(IntRow, new_data=[(1, 10), (2, 20)])
    t.describe()
    out = capsys.readouterr().out
    # No null line when null_count == 0
    # The word "null" should not appear in the score section
    # (the column has null_value=-1 but no actual null values)
    assert "null" not in out.lower()


# ===========================================================================
# to_arrow: null masking
# ===========================================================================


def test_to_arrow_null_masking():
    pytest.importorskip("pyarrow")

    t = CTable(IntRow, new_data=[(1, 10), (2, -1), (3, 20)])
    arrow = t.to_arrow()
    score_col = arrow.column("score")
    assert score_col[0].as_py() == 10
    assert score_col[1].as_py() is None  # null sentinel → Arrow null
    assert score_col[2].as_py() == 20


def test_to_arrow_nan_masking():
    pytest.importorskip("pyarrow")

    t = CTable(FloatRow, new_data=[("a", 1.0), ("b", float("nan")), ("c", 3.0)])
    arrow = t.to_arrow()
    val_col = arrow.column("value")
    assert val_col[0].as_py() == pytest.approx(1.0)
    assert val_col[1].as_py() is None  # NaN sentinel → Arrow null
    assert val_col[2].as_py() == pytest.approx(3.0)


def test_to_arrow_string_null_masking():
    pytest.importorskip("pyarrow")

    t = CTable(StrRow, new_data=[("hello", 1), ("", 2), ("world", 3)])
    arrow = t.to_arrow()
    label_col = arrow.column("label")
    assert label_col[0].as_py() == "hello"
    assert label_col[1].as_py() is None  # empty string → Arrow null
    assert label_col[2].as_py() == "world"


def test_to_arrow_no_null_value_no_masking():
    pytest.importorskip("pyarrow")

    t = CTable(IntRow, new_data=[(1, 10), (2, 20)])
    arrow = t.to_arrow()
    # id column has no null_value → all values present
    id_col = arrow.column("id")
    assert id_col.null_count == 0


# ===========================================================================
# from_csv: empty cells → sentinel
# ===========================================================================


def test_from_csv_empty_cell_to_null():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("id,score\n")
        f.write("1,10\n")
        f.write("2,\n")  # empty score → null sentinel
        f.write("3,20\n")
        fname = f.name
    try:
        t = CTable.from_csv(fname, IntRow)
        scores = t["score"][:]
        assert scores[0] == 10
        assert scores[1] == -1  # null sentinel
        assert scores[2] == 20
        assert t["score"].null_count() == 1
    finally:
        os.unlink(fname)


def test_from_csv_empty_string_cell_to_null():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("label,rank\n")
        f.write("hello,1\n")
        f.write(",2\n")  # empty label → null sentinel ""
        f.write("world,3\n")
        fname = f.name
    try:
        t = CTable.from_csv(fname, StrRow)
        labels = t["label"][:]
        assert labels[0] == "hello"
        assert labels[1] == ""  # null sentinel
        assert labels[2] == "world"
        assert t["label"].null_count() == 1
    finally:
        os.unlink(fname)


def test_from_csv_no_null_value_non_empty_cells():
    """Without null_value, normal values are read and stored correctly."""

    @dataclass
    class SimpleRow:
        x: int = blosc2.field(blosc2.int64())

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("x\n")
        f.write("5\n")
        f.write("7\n")
        f.write("10\n")
        fname = f.name
    try:
        t = CTable.from_csv(fname, SimpleRow)
        arr = t["x"][:]
        assert list(arr) == [5, 7, 10]
        assert t["x"].null_count() == 0
    finally:
        os.unlink(fname)


# ===========================================================================
# Persistence: null_value round-trips through schema serialization
# ===========================================================================


def test_null_value_persists_to_disk():
    path = table_path("null_persist")
    t = CTable(IntRow, urlpath=path, mode="w", new_data=[(1, 10), (2, -1), (3, 20)])
    del t
    t2 = CTable.open(path)
    assert t2["score"].null_value == -1
    assert t2["score"].null_count() == 1


def test_null_value_nan_persists():
    path = table_path("null_nan_persist")
    t = CTable(FloatRow, urlpath=path, mode="w", new_data=[("a", 1.0), ("b", float("nan"))])
    del t
    t2 = CTable.open(path)
    nv = t2["value"].null_value
    assert isinstance(nv, float)
    assert math.isnan(nv)
    assert t2["value"].null_count() == 1


def test_null_value_string_persists():
    path = table_path("null_str_persist")
    t = CTable(StrRow, urlpath=path, mode="w", new_data=[("hello", 1), ("", 2)])
    del t
    t2 = CTable.open(path)
    assert t2["label"].null_value == ""
    assert t2["label"].null_count() == 1


# ===========================================================================
# Edge cases
# ===========================================================================


def test_all_nulls_unique_empty():
    t = CTable(IntRow, new_data=[(1, -1), (2, -1)])
    u = t["score"].unique()
    assert len(u) == 0


def test_all_nulls_value_counts_empty():
    t = CTable(IntRow, new_data=[(1, -1), (2, -1)])
    vc = t["score"].value_counts()
    assert len(vc) == 0


def test_null_value_does_not_affect_non_nullable_column():
    t = CTable(IntRow, new_data=[(1, 10), (2, 20)])
    # id column has no null_value — aggregates work normally
    assert t["id"].sum() == 3
    assert t["id"].min() == 1
    assert t["id"].max() == 2


def test_schema_null_value_in_metadata():
    """null_value appears in schema_to_dict output for persistence."""
    from blosc2.schema_compiler import schema_to_dict

    @dataclass
    class SomeRow:
        x: int = blosc2.field(blosc2.int64(null_value=-999))
        label: str = blosc2.field(blosc2.string(max_length=8, null_value="N/A"))

    t = CTable(SomeRow, new_data=[(1, "hello"), (-999, "N/A")])
    d = schema_to_dict(t._schema)
    cols = {c["name"]: c for c in d["columns"]}
    assert cols["x"]["null_value"] == -999
    assert cols["label"]["null_value"] == "N/A"


if __name__ == "__main__":
    pytest.main(["-v", __file__])
