#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

"""Tests for the utf8 schema spec (variable-length strings as offsets + bytes)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

import blosc2
from blosc2 import CTable

STRING_DTYPE = np.dtypes.StringDType()


@dataclass
class Row:
    name: str = blosc2.field(blosc2.utf8())
    x: int = blosc2.field(blosc2.int64())


@dataclass
class NullableRow:
    name: str = blosc2.field(blosc2.utf8(nullable=True))
    x: int = blosc2.field(blosc2.int64())


# Mixed content: ASCII, non-ASCII (1..4-byte UTF-8), empty, 1-char, multi-KB.
SAMPLE = [
    "hello",
    "",
    "a",
    "café",
    "日本語のテキスト",
    "emoji 🎉🚀",
    "x" * 4096,
    "línea con acentos y çedilla",
]


def make_table(values=None, **kwargs):
    values = SAMPLE if values is None else values
    return CTable(Row, new_data={"name": list(values), "x": list(range(len(values)))}, **kwargs)


# ---------------------------------------------------------------------------
# Schema spec
# ---------------------------------------------------------------------------


def test_utf8_spec_defaults():
    spec = blosc2.utf8()
    assert spec.nullable is False
    assert spec.null_value is None
    assert spec.dtype is None
    assert spec.python_type is str


def test_utf8_spec_metadata_round_trip():
    from blosc2.schema_compiler import spec_from_metadata_dict

    spec = blosc2.utf8(nullable=True, null_value="<NA>")
    d = spec.to_metadata_dict()
    assert d["kind"] == "utf8"
    assert d["nullable"] is True
    assert d["null_value"] == "<NA>"

    restored = spec_from_metadata_dict(d)
    assert type(restored).__name__ == "Utf8Spec"
    assert restored.nullable is True
    assert restored.null_value == "<NA>"


def test_utf8_null_value_must_be_str():
    with pytest.raises(TypeError, match="null_value must be str"):
        blosc2.utf8(null_value=42)


def test_utf8_display_width():
    from blosc2.schema_compiler import compute_display_width

    assert compute_display_width(blosc2.utf8()) == 40


def test_utf8_not_inferred_from_plain_str_annotation():
    @dataclass
    class Plain:
        s: str
        x: int

    t = CTable(Plain, new_data=[("abc", 1)])
    cc = t.schema.columns_by_name["s"]
    assert type(cc.spec).__name__ == "string"  # fixed-width default is unchanged


# ---------------------------------------------------------------------------
# Utf8Array internal adapter
# ---------------------------------------------------------------------------


def test_utf8_array_basic_roundtrip():
    from blosc2.utf8_array import Utf8Array

    arr = Utf8Array(blosc2.utf8())
    arr.extend(SAMPLE)
    assert len(arr) == len(SAMPLE)
    assert list(arr[:]) == SAMPLE
    arr.flush()
    assert list(arr[:]) == SAMPLE
    assert arr[0] == "hello"
    assert arr[-1] == SAMPLE[-1]
    assert arr.dtype == STRING_DTYPE


def test_utf8_array_reads_across_pending_boundary():
    from blosc2.utf8_array import Utf8Array

    arr = Utf8Array(blosc2.utf8())
    arr.extend(SAMPLE[:4])
    arr.flush()
    arr.extend(SAMPLE[4:])  # stays pending
    assert list(arr[:]) == SAMPLE
    assert list(arr[2:6]) == SAMPLE[2:6]
    got = arr[np.array([7, 0, 5])]
    assert got.dtype == STRING_DTYPE
    assert list(got) == [SAMPLE[7], SAMPLE[0], SAMPLE[5]]
    mask = np.zeros(len(SAMPLE), dtype=np.bool_)
    mask[[1, 4]] = True
    assert list(arr[mask]) == [SAMPLE[1], SAMPLE[4]]


def test_utf8_array_setitem_shifts_offsets():
    from blosc2.utf8_array import Utf8Array

    arr = Utf8Array(blosc2.utf8())
    arr.extend(["aa", "bb", "cc"])
    arr.flush()
    arr[1] = "a longer replacement value"
    assert list(arr[:]) == ["aa", "a longer replacement value", "cc"]
    arr[1] = ""
    assert list(arr[:]) == ["aa", "", "cc"]


def test_utf8_array_rejects_non_str():
    from blosc2.utf8_array import Utf8Array

    arr = Utf8Array(blosc2.utf8())
    with pytest.raises(TypeError, match="Expected str"):
        arr.append(42)
    with pytest.raises(TypeError, match="not nullable"):
        arr.append(None)


# ---------------------------------------------------------------------------
# CTable integration: append / extend / read
# ---------------------------------------------------------------------------


def test_ctable_utf8_extend_and_read():
    t = make_table()
    assert t.nrows == len(SAMPLE)
    values = t["name"][:]
    assert isinstance(values, np.ndarray)
    assert values.dtype == STRING_DTYPE
    assert list(values) == SAMPLE
    assert t["name"][3] == "café"
    assert t["name"][-2] == "x" * 4096


def test_ctable_utf8_append_rows():
    t = CTable(Row)
    t.append(("first", 1))
    t.append({"name": "segundo", "x": 2})
    t.append(("日本", 3))
    assert list(t["name"][:]) == ["first", "segundo", "日本"]
    assert list(t["x"][:]) == [1, 2, 3]


def test_ctable_utf8_extend_numpy_fixed_width_input():
    t = CTable(Row)
    t.extend({"name": np.array(["uno", "dos", "tres"]), "x": np.arange(3)})
    assert list(t["name"][:]) == ["uno", "dos", "tres"]


def test_ctable_utf8_setitem():
    t = make_table()
    t["name"][0] = "replaced"
    assert t["name"][0] == "replaced"
    assert list(t["name"][1:4]) == SAMPLE[1:4]
    t["name"][2:4] = ["p", "q"]
    assert list(t["name"][:4]) == ["replaced", "", "p", "q"]


def test_ctable_utf8_iter_and_fancy_reads():
    t = make_table()
    assert list(t["name"]) == SAMPLE
    got = t["name"][[5, 1, 0]]
    assert list(got) == [SAMPLE[5], SAMPLE[1], SAMPLE[0]]
    mask = np.array([v.startswith("h") for v in SAMPLE])
    assert list(t["name"][mask]) == ["hello"]


def test_ctable_utf8_unique_and_value_counts():
    t = make_table(["b", "a", "b", "c", "a", "b"])
    assert list(t["name"].unique()) == ["a", "b", "c"]
    assert t["name"].value_counts() == {"b": 3, "a": 2, "c": 1}


def test_ctable_utf8_repr_and_str():
    t = make_table()
    text = str(t)
    assert "hello" in text
    assert "café" in text
    col_repr = repr(t["name"])
    assert "name" in col_repr
    info_text = repr(t.info)
    assert "utf8" in info_text


def test_ctable_utf8_delete_and_compact():
    t = make_table(["a", "bb", "ccc", "dddd", "eeeee"])
    t.delete([1, 3])
    assert t.nrows == 3
    assert list(t["name"][:]) == ["a", "ccc", "eeeee"]
    t.compact()
    assert list(t["name"][:]) == ["a", "ccc", "eeeee"]
    t.append(("tail", 99))
    assert list(t["name"][:]) == ["a", "ccc", "eeeee", "tail"]


def test_ctable_utf8_copy_and_take():
    t = make_table()
    c = t.copy()
    assert list(c["name"][:]) == SAMPLE
    sub = t.take([0, 3, 5])
    assert list(sub["name"][:]) == [SAMPLE[0], SAMPLE[3], SAMPLE[5]]


def test_ctable_utf8_view_reads():
    t = make_table(["a", "bb", "ccc", "dddd"])
    v = t.head(2)
    assert list(v["name"][:]) == ["a", "bb"]
    v2 = t[t.x > 1]
    assert list(v2["name"][:]) == ["ccc", "dddd"]


def test_ctable_utf8_add_and_drop_column():
    t = make_table(["a", "b"])
    t.add_column("note", blosc2.field(blosc2.utf8(), default="n/a"))
    assert list(t["note"][:]) == ["n/a", "n/a"]
    t.drop_column("note")
    assert "note" not in t.col_names


# ---------------------------------------------------------------------------
# Nulls (sentinel-based)
# ---------------------------------------------------------------------------


def test_ctable_utf8_nullable_sentinel_from_policy():
    t = CTable(NullableRow, new_data={"name": ["a", None, "c"], "x": [1, 2, 3]})
    nv = t["name"].null_value
    assert nv == "__BLOSC2_NULL__"
    assert list(t["name"].is_null()) == [False, True, False]
    assert t["name"].null_count() == 1
    # Reads surface the sentinel verbatim, like other sentinel-based columns.
    assert t["name"][1] == nv
    assert list(t["name"].fillna("<missing>")) == ["a", "<missing>", "c"]


def test_ctable_utf8_explicit_null_value():
    @dataclass
    class R:
        s: str = blosc2.field(blosc2.utf8(null_value="<NA>"))
        x: int = blosc2.field(blosc2.int64())

    t = CTable(R, new_data={"s": [None, "v"], "x": [0, 1]})
    assert t["s"].null_value == "<NA>"
    assert list(t["s"][:]) == ["<NA>", "v"]
    assert t["s"].null_count() == 1


def test_ctable_utf8_not_nullable_rejects_none():
    t = CTable(Row)
    with pytest.raises((TypeError, ValueError)):
        t.append((None, 1))


def test_ctable_utf8_dropna():
    t = CTable(NullableRow, new_data={"name": ["a", None, "c", None], "x": [1, 2, 3, 4]})
    kept = t.dropna(subset=["name"])
    assert list(kept["name"][:]) == ["a", "c"]


# ---------------------------------------------------------------------------
# Persistence round-trips
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ext", [".b2z", ".b2d"])
def test_ctable_utf8_persistence_roundtrip(tmp_path, ext):
    urlpath = str(tmp_path / f"utf8_table{ext}")
    t = make_table(urlpath=urlpath, mode="w")
    t.close()

    t2 = CTable.open(urlpath, mode="r")
    try:
        assert list(t2["name"][:]) == SAMPLE
        assert t2["name"][4] == SAMPLE[4]
        values = t2["name"][:]
        assert values.dtype == STRING_DTYPE
    finally:
        t2.close()


def test_ctable_utf8_persistence_append_reopen(tmp_path):
    urlpath = str(tmp_path / "utf8_append.b2z")
    t = make_table(["one", "two"], urlpath=urlpath, mode="w")
    t.close()

    t2 = CTable.open(urlpath, mode="a")
    try:
        t2.append(("three", 2))
        t2.extend({"name": ["four", ""], "x": [3, 4]})
    finally:
        t2.close()

    t3 = CTable.open(urlpath, mode="r")
    try:
        assert list(t3["name"][:]) == ["one", "two", "three", "four", ""]
    finally:
        t3.close()


def test_ctable_utf8_nullable_persists(tmp_path):
    urlpath = str(tmp_path / "utf8_null.b2z")
    t = CTable(NullableRow, new_data={"name": ["a", None], "x": [1, 2]}, urlpath=urlpath, mode="w")
    t.close()
    t2 = CTable.open(urlpath, mode="r")
    try:
        assert t2["name"].null_value == "__BLOSC2_NULL__"
        assert t2["name"].null_count() == 1
    finally:
        t2.close()


def test_ctable_utf8_load_into_memory(tmp_path):
    urlpath = str(tmp_path / "utf8_load.b2d")
    t = make_table(urlpath=urlpath, mode="w")
    t.close()
    t2 = CTable.load(urlpath)
    assert list(t2["name"][:]) == SAMPLE
    t2.append(("appended", 100))
    assert t2["name"][-1] == "appended"


def test_ctable_utf8_save_copy_to_disk(tmp_path):
    t = make_table()
    urlpath = str(tmp_path / "utf8_saved.b2z")
    t.save(urlpath)
    t2 = CTable.open(urlpath, mode="r")
    try:
        assert list(t2["name"][:]) == SAMPLE
    finally:
        t2.close()


def test_ctable_utf8_cframe_roundtrip():
    t = make_table()
    frame = t.to_cframe()
    t2 = blosc2.ctable_from_cframe(frame)
    assert list(t2["name"][:]) == SAMPLE


def test_ctable_utf8_rename_column_persistent(tmp_path):
    urlpath = str(tmp_path / "utf8_rename.b2d")
    t = make_table(["a", "b"], urlpath=urlpath, mode="w")
    t.rename_column("name", "title")
    assert list(t["title"][:]) == ["a", "b"]
    t.close()
    t2 = CTable.open(urlpath, mode="r")
    try:
        assert list(t2["title"][:]) == ["a", "b"]
    finally:
        t2.close()


# ---------------------------------------------------------------------------
# Unsupported operations fail clearly (lifted by later work)
# ---------------------------------------------------------------------------


def test_ctable_utf8_comparison_raises_clearly():
    t = make_table()
    with pytest.raises(NotImplementedError, match="utf8"):
        t.name == "hello"  # noqa: B015


def test_ctable_utf8_where_expression_raises_clearly():
    t = make_table()
    with pytest.raises(NotImplementedError, match="utf8"):
        t.where("name == 'hello'")


def test_ctable_utf8_groupby_raises_clearly():
    t = make_table(["a", "b", "a"])
    with pytest.raises(TypeError, match="variable-length"):
        t.group_by("name").sum("x")


def test_ctable_utf8_sort_raises_clearly():
    t = make_table()
    with pytest.raises(TypeError, match="utf8"):
        t.sort_by("name")


def test_ctable_utf8_arrow_export_large_string():
    pa = pytest.importorskip("pyarrow")
    t = make_table()
    at = t.to_arrow()
    assert at.schema.field("name").type == pa.large_string()
    assert at.column("name").to_pylist() == SAMPLE


# ---------------------------------------------------------------------------
# Arrow interop (P3.b)
# ---------------------------------------------------------------------------


def test_utf8_pa_table_roundtrip():
    pa = pytest.importorskip("pyarrow")
    t = make_table()
    at = pa.table(t)
    assert at.column("name").to_pylist() == SAMPLE


def test_utf8_pa_table_roundtrip_with_nulls():
    pa = pytest.importorskip("pyarrow")
    t = CTable(NullableRow, new_data={"name": ["a", None, "c"], "x": [1, 2, 3]})
    at = pa.table(t)
    assert at.schema.field("name").type == pa.large_string()
    assert at.column("name").to_pylist() == ["a", None, "c"]
    assert at.column("name").null_count == 1


def test_utf8_from_arrow_default_ingest():
    pa = pytest.importorskip("pyarrow")
    at = pa.table({"name": pa.array(SAMPLE, type=pa.string()), "x": pa.array(range(len(SAMPLE)))})
    t = CTable.from_arrow(at.schema, at.to_batches())
    assert t["name"].is_utf8
    assert list(t["name"][:]) == SAMPLE


def test_utf8_from_arrow_large_string_ingest():
    pa = pytest.importorskip("pyarrow")
    at = pa.table({"name": pa.array(SAMPLE, type=pa.large_string())})
    t = CTable.from_arrow(at.schema, at.to_batches())
    assert t["name"].is_utf8
    assert list(t["name"][:]) == SAMPLE


def test_utf8_from_arrow_nulls_use_sentinel():
    pa = pytest.importorskip("pyarrow")
    at = pa.table({"name": pa.array(["a", None, "c"], type=pa.string())})
    t = CTable.from_arrow(at.schema, at.to_batches())
    nv = t["name"].null_value
    assert nv is not None
    assert list(t["name"][:]) == ["a", nv, "c"]
    assert t["name"].null_count() == 1


def test_utf8_from_arrow_fixed_width_when_max_length_given():
    pa = pytest.importorskip("pyarrow")
    at = pa.table({"name": pa.array(["hi", "there"], type=pa.string())})
    t = CTable.from_arrow(at.schema, at.to_batches(), string_max_length=32)
    assert not t["name"].is_utf8
    assert t["name"].dtype.kind == "U"


def test_utf8_duckdb_query():
    duckdb = pytest.importorskip("duckdb")
    pytest.importorskip("pyarrow")
    t = make_table(["paris", "london", "paris", "tokyo"])
    arrow_tbl = t.to_arrow()
    result = duckdb.sql(
        "SELECT name, count(*) AS n FROM arrow_tbl WHERE name = 'paris' GROUP BY name"
    ).fetchall()
    assert result == [("paris", 2)]
