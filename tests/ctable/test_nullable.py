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


def test_numpy_nan_null_value_skips_scalar_validation_constraints():
    @dataclass
    class NumpyNaNFloatRow:
        value: float = blosc2.field(blosc2.float32(ge=0, null_value=np.float32(np.nan)))

    t = CTable(NumpyNaNFloatRow)
    t.append((np.float32(np.nan),))

    assert t.value.null_count() == 1


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


def test_nullable_true_uses_default_null_policy():
    @dataclass
    class Row:
        i: int = blosc2.field(blosc2.int32(nullable=True))
        u: int = blosc2.field(blosc2.uint32(nullable=True))
        f: float = blosc2.field(blosc2.float64(nullable=True))
        flag: bool = blosc2.field(blosc2.bool(nullable=True))
        s: str = blosc2.field(blosc2.string(max_length=4, nullable=True))
        b: bytes = blosc2.field(blosc2.bytes(max_length=4, nullable=True))

    t = CTable(Row)
    assert t["i"].null_value == np.iinfo(np.int32).min
    assert t["u"].null_value == np.iinfo(np.uint32).max
    assert math.isnan(t["f"].null_value)
    assert t["flag"].null_value == 255
    assert t["s"].null_value == "__BLOSC2_NULL__"
    assert t["b"].null_value == b"__BLOSC2_NULL__"
    assert t["s"].dtype.itemsize // 4 >= len("__BLOSC2_NULL__")
    assert t["b"].dtype.itemsize >= len(b"__BLOSC2_NULL__")


def test_nullable_true_uses_null_policy_context_and_column_null_values():
    @dataclass
    class Row:
        i: int = blosc2.field(blosc2.int32(nullable=True))
        s: str = blosc2.field(blosc2.string(max_length=4, nullable=True))

    policy = blosc2.NullPolicy(
        signed_int_strategy="max", string_value="<NULL>", column_null_values={"i": -1}
    )
    with blosc2.null_policy(policy):
        t = CTable(Row)

    assert t["i"].null_value == -1
    assert t["s"].null_value == "<NULL>"


def test_explicit_null_value_overrides_nullable_policy():
    @dataclass
    class Row:
        i: int = blosc2.field(blosc2.int32(nullable=True, null_value=-5))

    policy = blosc2.NullPolicy(signed_int_strategy="max")
    with blosc2.null_policy(policy):
        t = CTable(Row)

    assert t["i"].null_value == -5


def test_add_column_nullable_true_uses_null_policy():
    t = CTable(IntRow)
    with blosc2.null_policy(blosc2.NullPolicy(signed_int_strategy="max")):
        t.add_column("extra", blosc2.field(blosc2.int32(nullable=True), default=0))

    assert t["extra"].null_value == np.iinfo(np.int32).max


def test_nullable_policy_rejects_out_of_range_integer_sentinel():
    @dataclass
    class Row:
        x: int = blosc2.field(blosc2.int8(nullable=True))

    with blosc2.null_policy(blosc2.NullPolicy(column_null_values={"x": 1000})):
        with pytest.raises(ValueError, match="outside int8 range"):
            CTable(Row)


def test_nullable_policy_rejects_wrong_string_sentinel_type():
    @dataclass
    class Row:
        s: str = blosc2.field(blosc2.string(nullable=True))

    with blosc2.null_policy(blosc2.NullPolicy(column_null_values={"s": b"NA"})):
        with pytest.raises(TypeError, match="must be str"):
            CTable(Row)


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


def test_is_null_timestamp_sentinel():
    """Timestamp sentinels materialize as np.datetime64('NaT') (same bit
    pattern as INT64_MIN), so is_null() must special-case datetime64 arrays
    instead of comparing against the raw int sentinel."""

    @dataclass
    class TsRow:
        ts: int = blosc2.field(blosc2.timestamp(null_value=np.iinfo(np.int64).min))

    null_ts = np.iinfo(np.int64).min
    t = CTable(TsRow, new_data=[(1000,), (null_ts,), (2000,)])
    assert list(t["ts"].is_null()) == [False, True, False]
    assert t["ts"].null_count() == 1


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


def test_sum_where_pushdown_skips_int_null():
    t = CTable(IntRow, new_data=[(1, 10), (2, -1), (3, 20), (4, -1), (5, 30)])
    assert t["score"].sum(where=t.id < 5) == 30
    assert t[t.id < 5]["score"].sum() == 30


def test_sum_where_pushdown_skips_nan_null():
    t = CTable(FloatRow, new_data=[("a", 1.5), ("b", float("nan")), ("c", 2.5)])
    assert t["value"].sum(where=t.value < 2.0) == pytest.approx(1.5)
    assert t[t.value < 2.0]["value"].sum() == pytest.approx(1.5)


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


# ===========================================================================
# fillna / dropna
# ===========================================================================


def test_fillna_int_sentinel():
    t = CTable(IntRow, new_data=[(1, 10), (2, -1), (3, 20)])
    filled = t["score"].fillna(0)
    np.testing.assert_array_equal(filled, [10, 0, 20])


def test_fillna_nan_float():
    t = CTable(FloatRow, new_data=[("a", 1.0), ("b", float("nan")), ("c", 3.0)])
    filled = t["value"].fillna(-1.0)
    np.testing.assert_array_equal(filled, [1.0, -1.0, 3.0])


def test_fillna_timestamp_sentinel():
    @dataclass
    class TsRow:
        ts: int = blosc2.field(blosc2.timestamp(null_value=np.iinfo(np.int64).min))

    null_ts = np.iinfo(np.int64).min
    t = CTable(TsRow, new_data=[(1000,), (null_ts,), (2000,)])
    filled = t["ts"].fillna(np.datetime64(0, "us"))
    np.testing.assert_array_equal(filled, np.array([1000, 0, 2000], dtype="datetime64[us]"))


def test_fillna_dictionary_column():
    @dataclass
    class DictRow:
        vendor: str = blosc2.field(blosc2.dictionary())

    t = CTable(DictRow, new_data=[("Uber",), (None,), ("Lyft",)])
    assert t["vendor"].fillna("unknown") == ["Uber", "unknown", "Lyft"]


def test_fillna_varlen_string_column():
    @dataclass
    class VLRow:
        text: str = blosc2.field(blosc2.vlstring(nullable=True))

    t = CTable(VLRow, new_data=[("hello",), (None,), ("world",)])
    assert t["text"].fillna("N/A") == ["hello", "N/A", "world"]


def test_fillna_on_view_returns_view_rows_only():
    t = CTable(IntRow, new_data=[(1, 10), (2, -1), (3, 20), (4, -1)])
    view = t.where(t["id"] > 1)
    np.testing.assert_array_equal(view["score"].fillna(0), [0, 20, 0])
    np.testing.assert_array_equal(t["score"][:], [10, -1, 20, -1])  # base untouched


def test_fillna_on_non_nullable_column_is_identity():
    t = CTable(IntRow, new_data=[(1, 10), (2, 20)])
    np.testing.assert_array_equal(t["id"].fillna(0), [1, 2])


def test_dropna_default_subset():
    t = CTable(IntRow, new_data=[(1, 10), (2, -1), (3, 20), (4, -1)])
    result = t.dropna()
    assert result.nrows == 2
    np.testing.assert_array_equal(result["score"][:], [10, 20])
    assert result.base is t


def test_dropna_explicit_subset():
    @dataclass
    class TwoNullableRow:
        a: int = blosc2.field(blosc2.int64(null_value=-1))
        b: int = blosc2.field(blosc2.int64(null_value=-1))

    t = CTable(TwoNullableRow, new_data=[(1, -1), (-1, 2), (3, 4)])
    # only checking "a": row 1 (a=-1) is dropped, row 0 (b=-1) survives
    result = t.dropna(subset=["a"])
    np.testing.assert_array_equal(result["a"][:], [1, 3])
    np.testing.assert_array_equal(result["b"][:], [-1, 4])


def test_dropna_row_count_correct():
    t = CTable(IntRow, new_data=[(i, -1 if i % 3 == 0 else i) for i in range(10)])
    result = t.dropna(subset=["score"])
    assert result.nrows == sum(1 for i in range(10) if i % 3 != 0)


def test_dropna_of_a_filtered_view():
    t = CTable(IntRow, new_data=[(1, 10), (2, -1), (3, 20), (4, -1), (5, 30)])
    view = t.where(t["id"] > 1)  # rows with id 2,3,4,5 -> scores -1,20,-1,30
    result = view.dropna(subset=["score"])
    np.testing.assert_array_equal(result["score"][:], [20, 30])


if __name__ == "__main__":
    pytest.main(["-v", __file__])
