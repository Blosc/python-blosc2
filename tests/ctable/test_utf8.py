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

if not hasattr(np.dtypes, "StringDType"):
    pytest.skip("utf8 columns require NumPy >= 2.0 (StringDType)", allow_module_level=True)

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
    assert type(restored).__name__ == "UTF8Spec"
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
# UTF8Array internal adapter
# ---------------------------------------------------------------------------


def test_utf8_array_basic_roundtrip():
    from blosc2._utf8_array import UTF8Array

    arr = UTF8Array(blosc2.utf8())
    arr.extend(SAMPLE)
    assert len(arr) == len(SAMPLE)
    assert list(arr[:]) == SAMPLE
    arr.flush()
    assert list(arr[:]) == SAMPLE
    assert arr[0] == "hello"
    assert arr[-1] == SAMPLE[-1]
    assert arr.dtype == STRING_DTYPE


def test_utf8_array_reads_across_pending_boundary():
    from blosc2._utf8_array import UTF8Array

    arr = UTF8Array(blosc2.utf8())
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
    from blosc2._utf8_array import UTF8Array

    arr = UTF8Array(blosc2.utf8())
    arr.extend(["aa", "bb", "cc"])
    arr.flush()
    arr[1] = "a longer replacement value"
    assert list(arr[:]) == ["aa", "a longer replacement value", "cc"]
    arr[1] = ""
    assert list(arr[:]) == ["aa", "", "cc"]


def test_utf8_array_rejects_non_str():
    from blosc2._utf8_array import UTF8Array

    arr = UTF8Array(blosc2.utf8())
    with pytest.raises(TypeError, match="Expected str"):
        arr.append(42)
    with pytest.raises(TypeError, match="not nullable"):
        arr.append(None)


# ---------------------------------------------------------------------------
# Chunked bulk extend (write path)
# ---------------------------------------------------------------------------


def test_utf8_array_extend_empty_iterable_is_noop():
    from blosc2._utf8_array import UTF8Array

    arr = UTF8Array(blosc2.utf8())
    arr.extend([])
    assert len(arr) == 0
    arr.extend(iter([]))
    assert len(arr) == 0
    arr.flush()
    assert list(arr[:]) == []


def test_utf8_array_extend_many_rows_no_dropped_rows():
    """Regression for the flush-rebind pitfall: `flush()` rebinds
    `self._pending` to a fresh list rather than mutating it, so an
    `extend()` spanning several internal flushes must re-read
    `self._pending` after each one instead of caching a reference."""
    from blosc2._utf8_array import _FLUSH_ROWS, UTF8Array

    n = _FLUSH_ROWS * 3 + 7
    values = [f"row{i}" for i in range(n)]
    arr = UTF8Array(blosc2.utf8())
    arr.extend(values)
    assert len(arr) == n
    arr.flush()
    assert len(arr) == n
    assert list(arr[:]) == values


def test_utf8_array_extend_none_straddles_chunk():
    from blosc2._utf8_array import _FLUSH_ROWS, UTF8Array

    values = [f"v{i}" for i in range(_FLUSH_ROWS + 2)]
    values[_FLUSH_ROWS - 1] = None  # last row of first chunk
    values[_FLUSH_ROWS + 1] = None  # second row of second chunk
    arr = UTF8Array(blosc2.utf8(null_value="<NA>"))
    arr.extend(values)
    expected = [v if v is not None else "<NA>" for v in values]
    assert list(arr[:]) == expected


def test_utf8_array_extend_append_interleaved():
    from blosc2._utf8_array import UTF8Array

    arr = UTF8Array(blosc2.utf8())
    arr.append("first")
    arr.extend(["second", "third"])
    arr.append("fourth")
    arr.extend(["fifth"])
    assert list(arr[:]) == ["first", "second", "third", "fourth", "fifth"]


def test_utf8_array_extend_ascii_nul_byte_preserved():
    from blosc2._utf8_array import UTF8Array

    values = ["nul\x00in", "plain", "\x00leading", "trailing\x00"]
    assert all(v.isascii() for v in values)
    arr = UTF8Array(blosc2.utf8())
    arr.extend(values)
    arr.flush()
    assert list(arr[:]) == values


def test_utf8_array_extend_multi_mb_bounded():
    """~20 multi-MB ASCII strings: char-count flush bound is checked once
    per _FLUSH_ROWS-sized chunk (not per row), so this overshoots
    _FLUSH_CHARS by at most one chunk before flushing -- confirm read-back
    is still correct despite the coarser check."""
    from blosc2._utf8_array import UTF8Array

    values = [f"{i:06d}" + "x" * (2 * 1024 * 1024) for i in range(20)]
    arr = UTF8Array(blosc2.utf8())
    arr.extend(values)
    arr.flush()
    assert list(arr[:]) == values


# ---------------------------------------------------------------------------
# Bulk StringDType read: compiled kernel and its pure-Python fallback
# ---------------------------------------------------------------------------


@pytest.fixture(params=["kernel", "fallback"], ids=["kernel", "fallback"])
def force_kernel_mode(request, monkeypatch):
    """Exercise both the compiled bulk StringDType packer and its
    pure-Python per-row fallback, so the fallback stays covered even on a
    build where the compiled extension is available."""
    if request.param == "fallback":
        monkeypatch.setattr("blosc2._utf8_array._pack_utf8_kernel", lambda: None)
    return request.param


def test_pack_utf8_span_rejects_malformed_rel():
    """pack_utf8_span trusts its caller's rel/data invariants for speed, but
    still validates them cheaply up front so a malformed rel fails with a
    clear ValueError instead of driving the unchecked C loop out of bounds."""
    pytest.importorskip("blosc2.utf8_ext")
    from blosc2 import utf8_ext

    data = np.array([1, 2, 3], dtype=np.uint8)
    out = np.empty(2, dtype=STRING_DTYPE)

    with pytest.raises(ValueError, match="rel\\[0\\] must be 0"):
        utf8_ext.pack_utf8_span(np.array([1, 2, 3], dtype=np.int64), data, out)
    with pytest.raises(ValueError, match="non-decreasing"):
        utf8_ext.pack_utf8_span(np.array([0, 2, 1], dtype=np.int64), data, out)
    with pytest.raises(ValueError, match="must not exceed len\\(data\\)"):
        utf8_ext.pack_utf8_span(np.array([0, 2, 10], dtype=np.int64), data, out)

    # a well-formed rel still works after the added checks
    utf8_ext.pack_utf8_span(np.array([0, 1, 3], dtype=np.int64), data, out)
    assert list(out) == ["\x01", "\x02\x03"]


def test_utf8_array_bulk_read_kernel_and_fallback(force_kernel_mode):
    from blosc2._utf8_array import UTF8Array

    arr = UTF8Array(blosc2.utf8())
    arr.extend(SAMPLE)
    arr.flush()
    got = arr[:]
    assert got.dtype == STRING_DTYPE
    assert list(got) == SAMPLE


def test_utf8_array_bulk_read_matches_python(force_kernel_mode):
    """A wider mix of byte lengths and edge cases than SAMPLE: many distinct
    ASCII/multi-byte/empty/NUL-bearing values, read back in one bulk span."""
    from blosc2._utf8_array import UTF8Array

    rng = np.random.default_rng(5)
    pool = ["", "a", "café", "日本語", "x" * 5000, "nul\x00in", "nul\x00INSIDE", "emoji 🎉🚀"]
    values = [pool[i] for i in rng.integers(0, len(pool), 3000)]
    arr = UTF8Array(blosc2.utf8())
    arr.extend(values)
    arr.flush()
    assert list(arr[:]) == values


def test_ctable_utf8_extend_read_two_routes(force_kernel_mode):
    t = make_table()
    values = t["name"][:]
    assert values.dtype == STRING_DTYPE
    assert list(values) == SAMPLE


@pytest.mark.parametrize("ext", [".b2z", ".b2d"])
def test_ctable_utf8_persist_two_routes(tmp_path, ext, force_kernel_mode):
    urlpath = str(tmp_path / f"utf8_kernel_mode{ext}")
    t = make_table(urlpath=urlpath, mode="w")
    t.close()
    t2 = CTable.open(urlpath, mode="r")
    try:
        assert list(t2["name"][:]) == SAMPLE
    finally:
        t2.close()


# ---------------------------------------------------------------------------
# Bulk UTF-8 encode (write path): compiled kernel and its pure-Python fallback
# ---------------------------------------------------------------------------


@pytest.fixture(params=["kernel", "fallback"], ids=["kernel", "fallback"])
def force_write_kernel_mode(request, monkeypatch):
    """Exercise both the compiled bulk UTF-8 encoder and its pure-Python
    join+encode fallback, so the fallback stays covered even on a build
    where the compiled extension is available."""
    if request.param == "fallback":
        monkeypatch.setattr("blosc2._utf8_array._encode_utf8_kernel", lambda: None)
    return request.param


def test_utf8_array_extend_kernel_and_fallback(force_write_kernel_mode):
    from blosc2._utf8_array import UTF8Array

    arr = UTF8Array(blosc2.utf8())
    arr.extend(SAMPLE)
    arr.flush()
    assert list(arr[:]) == SAMPLE


def test_utf8_array_extend_matches_python_ground_truth(force_write_kernel_mode):
    """Same wider mix of byte lengths and edge cases as the read-side
    ground-truth test, exercised through the write path this time."""
    from blosc2._utf8_array import UTF8Array

    rng = np.random.default_rng(7)
    pool = ["", "a", "café", "日本語", "x" * 5000, "nul\x00in", "nul\x00INSIDE", "emoji 🎉🚀"]
    values = [pool[i] for i in rng.integers(0, len(pool), 3000)]
    arr = UTF8Array(blosc2.utf8())
    arr.extend(values)
    arr.flush()
    assert list(arr[:]) == values


def test_utf8_array_extend_nul_two_routes(force_write_kernel_mode):
    from blosc2._utf8_array import UTF8Array

    values = ["nul\x00in", "plain", "\x00leading", "trailing\x00"]
    arr = UTF8Array(blosc2.utf8())
    arr.extend(values)
    arr.flush()
    assert list(arr[:]) == values


def test_utf8_array_extend_mb_two_routes(force_write_kernel_mode):
    """A single multi-MB value alongside short ones -- sanity-checks the
    total-length/offset accumulation in the compiled kernel's two passes."""
    from blosc2._utf8_array import UTF8Array

    values = ["head", "x" * (8 * 1024 * 1024), "tail", "café" * 100_000]
    arr = UTF8Array(blosc2.utf8())
    arr.extend(values)
    arr.flush()
    assert list(arr[:]) == values


def test_ctable_utf8_extend_kernel_and_fallback(force_write_kernel_mode):
    t = make_table()
    assert list(t["name"][:]) == SAMPLE


def test_utf8_array_extend_surrogate_recovers(force_write_kernel_mode):
    """A lone surrogate is invalid UTF-8: flush() must raise
    UnicodeEncodeError, matching str.encode('utf-8')'s own behavior, and
    the array must remain usable afterwards -- a regression test for the
    compiled kernel's temp-buffer cleanup on the error path."""
    from blosc2._utf8_array import UTF8Array

    arr = UTF8Array(blosc2.utf8())
    arr.extend(["first"])
    arr.flush()
    arr.extend(["ok", "bad\udc80value"])
    with pytest.raises(UnicodeEncodeError):
        arr.flush()
    arr.extend(["second"])
    arr.flush()
    assert list(arr[:]) == ["first", "second"]


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


def test_ctable_utf8_add_column_values():
    t = make_table(["a", "b", "c"])
    t.add_column("note", blosc2.utf8(), values=["x", "yy", "zzz"])
    assert list(t["note"][:]) == ["x", "yy", "zzz"]


def test_ctable_utf8_add_col_values_from_expr():
    """The documented round trip: compute on <U, land the result back as utf8."""
    t = make_table(["a", "bb", "ccc"])
    arr = t["name"][:].astype("<U8")
    res = blosc2.lazyexpr("'x=' + a", {"a": arr}).compute()[:]
    t.add_column("out", blosc2.utf8(), values=res)
    assert list(t["out"][:]) == ["x=a", "x=bb", "x=ccc"]
    assert t["out"][:].dtype == STRING_DTYPE


def test_ctable_utf8_add_column_values_after_delete():
    t = make_table(["a", "b", "c", "d"])
    t.delete([0, 2])
    t.add_column("note", blosc2.utf8(), values=["P", "Q"])
    assert list(t["name"][:]) == ["b", "d"]
    assert list(t["note"][:]) == ["P", "Q"]


def test_ctable_utf8_add_column_values_persists(tmp_path):
    path = str(tmp_path / "utf8_values.b2d")
    t = make_table(["a", "bb"], urlpath=path, mode="w")
    t.add_column("note", blosc2.utf8(), values=["hello", "wörld"])
    t.close()
    t2 = CTable.open(path)
    assert list(t2["note"][:]) == ["hello", "wörld"]


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
# Comparisons and filtering
# ---------------------------------------------------------------------------


def test_ctable_utf8_eq_filters_rows():
    t = make_table(["paris", "london", "paris", "tokyo"])
    view = t[t.name == "paris"]
    assert list(view["name"][:]) == ["paris", "paris"]
    assert list(view["x"][:]) == [0, 2]


def test_ctable_utf8_ne_filters_rows():
    t = make_table(["paris", "london", "paris", "tokyo"])
    view = t[t.name != "paris"]
    assert list(view["name"][:]) == ["london", "tokyo"]


def test_ctable_utf8_ordering_comparisons():
    t = make_table(["paris", "london", "tokyo"])
    assert list(t[t.name < "paris"]["name"][:]) == ["london"]
    assert list(t[t.name <= "paris"]["name"][:]) == ["paris", "london"]
    assert list(t[t.name > "paris"]["name"][:]) == ["tokyo"]
    assert list(t[t.name >= "paris"]["name"][:]) == ["paris", "tokyo"]


def test_ctable_utf8_comparison_excludes_null_rows():
    """SQL WHERE semantics: a null value never satisfies any comparison."""
    t = CTable(NullableRow, new_data={"name": ["paris", None, "london"], "x": [1, 2, 3]})
    assert list(t[t.name == t["name"].null_value]["name"][:]) == []
    assert list(t[t.name != "paris"]["name"][:]) == ["london"]
    assert list(t[t.name < "z"]["name"][:]) == ["paris", "london"]


def test_ctable_utf8_column_vs_column_comparison():
    @dataclass
    class TwoCols:
        a: str = blosc2.field(blosc2.utf8(nullable=True))
        b: str = blosc2.field(blosc2.utf8(nullable=True))

    t = CTable(TwoCols, new_data={"a": ["x", "y", None, "z"], "b": ["x", "z", "q", None]})
    eq = t[t.a == t.b]
    assert list(eq["a"][:]) == ["x"]
    ne = t[t.a != t.b]
    # rows with a null on either side never satisfy != either (SQL semantics)
    assert list(ne["a"][:]) == ["y"]


def test_ctable_utf8_comparison_on_view():
    t = make_table(["paris", "london", "paris", "tokyo", "berlin"])
    head_view = t.head(3)
    filtered = head_view[head_view.name == "paris"]
    assert list(filtered["name"][:]) == ["paris", "paris"]


def test_ctable_utf8_cmp_non_string_raises():
    t = make_table()
    with pytest.raises(TypeError, match="utf8"):
        t.name == 42  # noqa: B015
    with pytest.raises(TypeError, match="utf8"):
        t.name < 3.14  # noqa: B015


def test_ctable_utf8_cmp_wrong_col_type_raises():
    @dataclass
    class Mixed:
        name: str = blosc2.field(blosc2.utf8())
        other: str = blosc2.field(blosc2.vlstring())

    t = CTable(Mixed, new_data={"name": ["a"], "other": ["b"]})
    with pytest.raises(TypeError, match="utf8"):
        t.name == t.other  # noqa: B015


def test_ctable_utf8_scalar_comparison_differential():
    """Every scalar comparison (byte-level, no decode) must match Python
    string semantics row-for-row, across ASCII, multi-byte, empty,
    NUL-bearing, and multi-KB values, plus a mix of null rows.

    Ground truth is computed with Python's own operators on the original
    values (not via np.unique/StringDType helpers, which have a known bug
    merging strings that differ only after an embedded NUL).
    """
    import operator

    pool = [
        "hello",
        "",
        "a",
        "café",
        "日本語のテキスト",
        "z",
        "é",
        "日",
        "Taxi",
        "Taxi Affiliation",
        "nul\x00in",
        "nul\x00INSIDE",
        "x" * 5000,
        "y" * 5000 + "!",
    ]
    n = 5000
    rng = np.random.default_rng(11)
    values = [pool[i] for i in rng.integers(0, len(pool), n)]
    null_positions = rng.choice(n, size=n // 20, replace=False)
    data = list(values)
    for i in null_positions:
        data[i] = None

    t = CTable(NullableRow, new_data={"name": data, "x": list(range(n))})
    nv = t["name"].null_value

    probes = {
        "present": "café",
        "absent": "not_in_pool_ZZZ",
        "prefix": "Taxi",
        "empty": "",
        "sentinel": nv,
    }
    ops = {
        "eq": (operator.eq, lambda c, p: c == p),
        "ne": (operator.ne, lambda c, p: c != p),
        "lt": (operator.lt, lambda c, p: c < p),
        "le": (operator.le, lambda c, p: c <= p),
        "gt": (operator.gt, lambda c, p: c > p),
        "ge": (operator.ge, lambda c, p: c >= p),
    }

    for probe_name, probe in probes.items():
        for op_name, (py_op, col_op) in ops.items():
            expected = [v for v in data if v is not None and py_op(v, probe)]
            got = list(t[col_op(t.name, probe)]["name"][:])
            assert got == expected, f"op={op_name} probe={probe_name!r} mismatch"


def test_ctable_utf8_ordering_prefix_edge_cases():
    """A probe that is a strict prefix of a value, and vice versa, at
    length-group boundaries."""
    t = make_table(["Taxi", "Taxi Affiliation", "Taxicab", "Tax"])
    assert list(t[t.name < "Taxi"]["name"][:]) == ["Tax"]
    assert list(t[t.name > "Taxi"]["name"][:]) == ["Taxi Affiliation", "Taxicab"]
    assert list(t[t.name == "Taxi"]["name"][:]) == ["Taxi"]
    assert list(t[t.name <= "Taxi"]["name"][:]) == ["Taxi", "Tax"]
    assert list(t[t.name >= "Taxi"]["name"][:]) == ["Taxi", "Taxi Affiliation", "Taxicab"]


def test_ctable_utf8_ordering_empty_string_probe():
    """Everything except "" is > the empty-string probe; "" is == it."""
    t = make_table(["", "a", "zzz"])
    assert list(t[t.name == ""]["name"][:]) == [""]
    assert list(t[t.name > ""]["name"][:]) == ["a", "zzz"]
    assert list(t[t.name < ""]["name"][:]) == []
    assert list(t[t.name >= ""]["name"][:]) == ["", "a", "zzz"]


def test_ctable_utf8_ordering_multibyte_bounds():
    """1-, 2-, and 3-byte UTF-8 encodings must byte-compare in code-point
    order (code points 0x7A < 0xE9 < 0x65E5)."""
    assert "z" < "é" < "日"
    t = make_table(["日", "z", "é"])
    s = t.sort_by("name")
    assert list(s["name"][:]) == ["z", "é", "日"]
    assert list(t[t.name < "é"]["name"][:]) == ["z"]
    # filtering preserves original row order (日, z, é), not sorted order
    assert list(t[t.name > "z"]["name"][:]) == ["日", "é"]


def test_ctable_utf8_ordering_nul_bearing_values():
    # Ground truth: "nul\x00INSIDE" < "nul\x00in" < ... at the byte position
    # right after the embedded NUL ('I' = 0x49 < 'i' = 0x69).
    assert sorted(["nul\x00in", "nul\x00INSIDE", "nul"]) == ["nul", "nul\x00INSIDE", "nul\x00in"]
    t = make_table(["nul\x00in", "nul\x00INSIDE", "nul"])  # original row order
    assert list(t[t.name == "nul\x00in"]["name"][:]) == ["nul\x00in"]
    assert list(t[t.name < "nul\x00in"]["name"][:]) == ["nul\x00INSIDE", "nul"]
    assert list(t[t.name > "nul"]["name"][:]) == ["nul\x00in", "nul\x00INSIDE"]


def test_ctable_utf8_ordering_probe_equals_sentinel():
    """All four ordering ops must exclude null rows even when the probe is
    the sentinel value itself (rows equal to the sentinel are the null
    rows)."""
    t = CTable(NullableRow, new_data={"name": ["alpha", None, "zeta"], "x": [1, 2, 3]})
    nv = t["name"].null_value
    for pred in (t.name < nv, t.name <= nv, t.name > nv, t.name >= nv):
        got = list(t[pred]["name"][:])
        assert nv not in got
        assert None not in got


def test_ctable_utf8_scalar_cmp_view_deletes():
    """The predicate mask is physical-length; it must stay correct through a
    view and after rows have been deleted (live-row mask intersection)."""
    t = make_table(["paris", "london", "paris", "tokyo", "berlin", "paris"])
    head_view = t.head(4)
    assert list(head_view[head_view.name == "paris"]["name"][:]) == ["paris", "paris"]
    assert list(head_view[head_view.name < "london"]["name"][:]) == []

    t.delete([0, 2])  # removes two of the three "paris" rows
    assert list(t["name"][:]) == ["london", "tokyo", "berlin", "paris"]
    assert list(t[t.name == "paris"]["name"][:]) == ["paris"]
    assert list(t[t.name != "paris"]["name"][:]) == ["london", "tokyo", "berlin"]


def test_ctable_utf8_startswith_endswith():
    t = make_table(["hello", "help", "world"])
    started = blosc2.startswith(t.name, "hel").compute()
    assert list(np.asarray(started)[:]) == [True, True, False]
    ended = blosc2.endswith(t.name, "lo").compute()
    assert list(np.asarray(ended)[:]) == [True, False, False]


# ---------------------------------------------------------------------------
# Groupby keys
# ---------------------------------------------------------------------------


def test_utf8_factorize_span_matches_np_unique():
    """The raw-bytes factorization keeps the np.unique contract: uniques
    sorted ascending, codes indexing them.  Ground truth is Python's set —
    numpy's np.unique on StringDType merges strings differing only after an
    embedded NUL (numpy bug), which the byte-exact factorization does not.
    """
    from blosc2._utf8_array import UTF8Array

    rng = np.random.default_rng(7)
    pool = ["", "a", "ab", "café", "日本語", "x" * 3000, "nul\x00in", "nul\x00IN", "Wien", "wien"]
    values = [pool[i] for i in rng.integers(0, len(pool), 5000)]
    arr = UTF8Array(blosc2.utf8())
    arr.extend(values)
    codes, uniques = arr.factorize_span(0, len(values))
    assert list(uniques) == sorted(set(values))
    assert all(uniques[c] == v for c, v in zip(codes, values, strict=True))


def test_utf8_factorizer_cross_span_codes_are_global():
    from blosc2._utf8_array import UTF8Array

    arr = UTF8Array(blosc2.utf8())
    arr.extend(["b", "a", "b", "c", "a", "d"])
    fact = arr.factorizer()
    c1 = fact.codes_for_span(0, 3)  # b, a, b
    c2 = fact.codes_for_span(3, 6)  # c, a, d
    uniques = fact.uniques()
    assert [uniques[c] for c in c1] == ["b", "a", "b"]
    assert [uniques[c] for c in c2] == ["c", "a", "d"]
    # "a" keeps the code it was assigned in the first span
    assert c1[1] == c2[1]


def test_ctable_utf8_groupby_lengths_non_ascii():
    rng = np.random.default_rng(3)
    pool = ["", "a", "bb", "café", "日本語のテキスト", "x" * 2000, "münchen"]
    names = [pool[i] for i in rng.integers(0, len(pool), 3000)]
    t = make_table(names)
    t.x[:] = np.ones(3000, dtype=np.int64)
    g = t.group_by("name").sum("x")
    got = dict(zip(g["name"][:].tolist(), g["x_sum"][:].tolist(), strict=True))
    exp: dict[str, int] = {}
    for v in names:
        exp[v] = exp.get(v, 0) + 1
    assert got == exp


def test_ctable_utf8_groupby_multi_key_negative_int():
    """Composite-int key packing must survive negative integer keys."""

    @dataclass
    class NegRow:
        ikey: int = blosc2.field(blosc2.int64())
        ukey: str = blosc2.field(blosc2.utf8())
        val: float = blosc2.field(blosc2.float64())

    ik = [-5, -5, 3, 3, -5, 3]
    uk = ["a", "b", "a", "b", "a", "a"]
    t = CTable(NegRow, new_data={"ikey": ik, "ukey": uk, "val": [1.0] * 6})
    g = t.group_by(["ikey", "ukey"]).sum("val")
    got = {
        (int(i), u): v
        for i, u, v in zip(g["ikey"][:], g["ukey"][:].tolist(), g["val_sum"][:].tolist(), strict=True)
    }
    assert got == {(-5, "a"): 2.0, (-5, "b"): 1.0, (3, "a"): 2.0, (3, "b"): 1.0}


def test_ctable_utf8_groupby_multi_key_float_fallback():
    """A float co-key forces the structured-dtype packing path with utf8 codes."""

    @dataclass
    class FloatRow:
        fkey: float = blosc2.field(blosc2.float64())
        ukey: str = blosc2.field(blosc2.utf8())
        val: float = blosc2.field(blosc2.float64())

    fk = [0.5, 0.5, 1.5, 1.5, 0.5]
    uk = ["a", "b", "a", "a", "a"]
    t = CTable(FloatRow, new_data={"fkey": fk, "ukey": uk, "val": [1.0] * 5})
    g = t.group_by(["fkey", "ukey"]).sum("val")
    got = {
        (f, u): v
        for f, u, v in zip(
            g["fkey"][:].tolist(), g["ukey"][:].tolist(), g["val_sum"][:].tolist(), strict=True
        )
    }
    assert got == {(0.5, "a"): 2.0, (0.5, "b"): 1.0, (1.5, "a"): 2.0}


def test_ctable_utf8_groupby_sum():
    t = make_table(["a", "b", "a", "b", "a"])
    t.x[:] = [1, 2, 3, 4, 5]
    g = t.group_by("name").sum("x")
    rows = dict(zip(g["name"][:].tolist(), g["x_sum"][:].tolist(), strict=False))
    assert rows == {"a": 9, "b": 6}


def test_ctable_utf8_groupby_size_and_dropna():
    t = CTable(NullableRow, new_data={"name": ["a", "b", "a", None, "b", "a"], "x": range(6)})
    g = t.group_by("name").size()  # dropna=True by default
    counts = dict(zip(g["name"][:].tolist(), g["size"][:].tolist(), strict=False))
    assert counts == {"a": 3, "b": 2}

    g_all = t.group_by("name", dropna=False).size()
    nv = t["name"].null_value
    counts_all = dict(zip(g_all["name"][:].tolist(), g_all["size"][:].tolist(), strict=False))
    assert counts_all == {"a": 3, "b": 2, nv: 1}


def test_ctable_utf8_groupby_sort():
    t = make_table(["gamma", "alpha", "beta", "alpha"])
    g = t.group_by("name", sort=True).size()
    assert list(g["name"][:]) == ["alpha", "beta", "gamma"]


def test_ctable_utf8_groupby_result_is_utf8_column():
    t = make_table(["a", "b", "a"])
    g = t.group_by("name").size()
    assert g["name"].is_utf8


def test_ctable_utf8_groupby_multi_key_with_int():
    @dataclass
    class MultiRow:
        cat: str = blosc2.field(blosc2.utf8())
        grp: int = blosc2.field(blosc2.int32())
        x: float = blosc2.field(blosc2.float64())

    t = CTable(
        MultiRow,
        new_data={
            "cat": ["a", "b", "a", "b", "a"],
            "grp": [1, 1, 1, 2, 2],
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
        },
    )
    g = t.group_by(["cat", "grp"]).sum("x")
    rows = sorted(zip(g["cat"][:].tolist(), g["grp"][:].tolist(), g["x_sum"][:].tolist(), strict=False))
    assert rows == [("a", 1, 4.0), ("a", 2, 5.0), ("b", 1, 2.0), ("b", 2, 4.0)]


def test_ctable_utf8_groupby_multi_chunk_merge():
    """A key set spanning many physical chunks exercises the merge path, not
    just a single-chunk factorization."""
    import random

    rng = random.Random(0)
    words = ["alpha", "beta", "gamma", "delta", "epsilon"]
    n = 200_000
    names = [rng.choice(words) for _ in range(n)]
    xs = [float(i % 7) for i in range(n)]
    t = CTable(Row, new_data={"name": names, "x": xs}, expected_size=n)
    g = t.group_by("name").sum("x")
    assert g.nrows == len(words)
    assert abs(sum(g["x_sum"][:].tolist()) - sum(xs)) < 1e-6


def test_ctable_utf8_groupby_still_rejects_vlstring():
    @dataclass
    class VlRow:
        name: str = blosc2.field(blosc2.vlstring())
        x: int = blosc2.field(blosc2.int64())

    t = CTable(VlRow, new_data={"name": ["a", "b"], "x": [1, 2]})
    with pytest.raises(TypeError, match="variable-length"):
        t.group_by("name").sum("x")


# ---------------------------------------------------------------------------
# Sort
# ---------------------------------------------------------------------------


def test_ctable_utf8_sort_ascending():
    t = make_table(["banana", "apple", "cherry"])
    s = t.sort_by("name")
    assert list(s["name"][:]) == ["apple", "banana", "cherry"]
    # row alignment: the "x" companion column follows its row, not its old position
    assert list(s["x"][:]) == [1, 0, 2]


def test_ctable_utf8_sort_descending():
    t = make_table(["banana", "apple", "cherry"])
    s = t.sort_by("name", ascending=False)
    assert list(s["name"][:]) == ["cherry", "banana", "apple"]


def test_ctable_utf8_sort_nulls_last_both_directions():
    t = CTable(NullableRow, new_data={"name": ["banana", None, "apple", "cherry"], "x": [1, 2, 3, 4]})
    nv = t["name"].null_value
    asc = t.sort_by("name")
    assert list(asc["name"][:]) == ["apple", "banana", "cherry", nv]
    desc = t.sort_by("name", ascending=False)
    assert list(desc["name"][:]) == ["cherry", "banana", "apple", nv]


def test_ctable_utf8_sort_view():
    t = make_table(["banana", "apple", "cherry"])
    view = t.sort_by("name", view=True)
    assert list(view["name"][:]) == ["apple", "banana", "cherry"]
    assert view.base is not None


def test_ctable_utf8_sort_inplace():
    t = make_table(["b", "a", "c"])
    result = t.sort_by("name", inplace=True)
    assert result is t
    assert list(t["name"][:]) == ["a", "b", "c"]


def test_ctable_utf8_sort_multi_key_bystander():
    """A non-key utf8 column in the same table must be reordered along with
    the sort, not just the sort key itself."""

    @dataclass
    class MultiRow:
        grp: int = blosc2.field(blosc2.int32())
        name: str = blosc2.field(blosc2.utf8())
        note: str = blosc2.field(blosc2.utf8())

    t = CTable(
        MultiRow,
        new_data={
            "grp": [1, 1, 2, 2],
            "name": ["b", "a", "d", "c"],
            "note": ["n-b", "n-a", "n-d", "n-c"],
        },
    )
    s = t.sort_by(["grp", "name"])
    rows = list(zip(s["grp"][:].tolist(), s["name"][:].tolist(), s["note"][:].tolist(), strict=True))
    assert rows == [(1, "a", "n-a"), (1, "b", "n-b"), (2, "c", "n-c"), (2, "d", "n-d")]


def test_ctable_utf8_sort_inplace_bystander_column():
    @dataclass
    class TwoCols:
        name: str = blosc2.field(blosc2.utf8())
        note: str = blosc2.field(blosc2.utf8())

    t = CTable(TwoCols, new_data={"name": ["b", "a", "c"], "note": ["n-b", "n-a", "n-c"]})
    t.sort_by("name", inplace=True)
    assert list(t["name"][:]) == ["a", "b", "c"]
    assert list(t["note"][:]) == ["n-a", "n-b", "n-c"]


@pytest.mark.parametrize("ext", [".b2z", ".b2d"])
def test_ctable_utf8_sort_inplace_persists(tmp_path, ext):
    """Regression: sort_by(inplace=True) on a file-backed table must write the
    sorted utf8 rows through to the store, keeping them aligned with the other
    (on-disk-sorted) columns after close/reopen."""
    urlpath = str(tmp_path / f"utf8_sort{ext}")
    t = make_table(["banana", "apple", "cherry"], urlpath=urlpath, mode="w")
    t.sort_by("name", inplace=True)
    assert list(t["name"][:]) == ["apple", "banana", "cherry"]
    t.close()

    t2 = CTable.open(urlpath, mode="r")
    try:
        assert list(t2["name"][:]) == ["apple", "banana", "cherry"]
        assert list(t2["x"][:]) == [1, 0, 2]  # row alignment survives the reopen
    finally:
        t2.close()


@pytest.mark.parametrize("ext", [".b2z", ".b2d"])
def test_ctable_utf8_compact_persists_after_reopen(tmp_path, ext):
    """Regression: compact() on a file-backed table must rewrite the utf8
    column in the store, not in a detached in-memory replacement."""
    urlpath = str(tmp_path / f"utf8_compact{ext}")
    t = make_table(["a", "bb", "ccc", "dddd"], urlpath=urlpath, mode="w")
    t.delete([1])
    t.compact()
    assert list(t["name"][:]) == ["a", "ccc", "dddd"]
    t.close()

    t2 = CTable.open(urlpath, mode="r")
    try:
        assert list(t2["name"][:]) == ["a", "ccc", "dddd"]
        assert list(t2["x"][:]) == [0, 2, 3]
    finally:
        t2.close()


def test_ctable_utf8_setitem_shifts_reopen(tmp_path):
    """__setitem__ on persisted rows shifts the byte blob in place; longer,
    shorter, equal-length, and empty replacements must all round-trip."""
    urlpath = str(tmp_path / "utf8_setitem.b2d")
    t = make_table(["aa", "bb", "cc", "dd"], urlpath=urlpath, mode="w")
    t["name"][1] = "a much longer replacement"  # grow
    t["name"][2] = "c"  # shrink
    t["name"][0] = "xx"  # equal length
    t["name"][3] = ""  # empty
    expected = ["xx", "a much longer replacement", "c", ""]
    assert list(t["name"][:]) == expected
    t.close()

    t2 = CTable.open(urlpath, mode="r")
    try:
        assert list(t2["name"][:]) == expected
    finally:
        t2.close()


def test_ctable_utf8_sort_non_ascii():
    t = make_table(["café", "日本語のテキスト", "banana"])
    s = t.sort_by("name")
    assert list(s["name"][:]) == sorted(["café", "日本語のテキスト", "banana"])


# ---------------------------------------------------------------------------
# Unsupported operations fail clearly (lifted by later work)
# ---------------------------------------------------------------------------


def test_ctable_utf8_create_index_builds_a_rank_index():
    """utf8 is indexed by the alphabetical rank of each row's value.

    Sorting by rank is sorting by decoded string, so an int32 rank column drives
    the existing numeric index machinery unchanged.
    """
    values = ["pear", "apple", "café", "banana", "apple"]
    t = make_table(values)
    index = t.create_index(col_name="name", kind="full")

    assert index.kind == "full"
    meta = t._get_index_catalog()["name"]["full"]["utf8_rank"]
    assert meta["vocab_len"] == len(set(values))
    assert meta["n_rows"] == len(values)

    # Ordering through the index must match a plain sort.
    assert list(t.sort_by("name", view=True)["name"][:]) == sorted(values)
    assert list(t.sorted_slice("name", slice(0, 2))["name"][:]) == sorted(values)[:2]


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


def test_utf8_arrow_export_from_view_and_after_delete():
    """Arrow export of non-dense tables (views, deleted rows) takes the
    materializing fallback path; values and nulls must match the fast path."""
    pa = pytest.importorskip("pyarrow")
    t = CTable(NullableRow, new_data={"name": ["a", None, "c", "d"], "x": [1, 2, 3, 4]})
    view = t[t.x > 1]
    at = pa.table(view)
    assert at.schema.field("name").type == pa.large_string()
    assert at.column("name").to_pylist() == [None, "c", "d"]

    t.delete([0])  # dense-table fast path no longer applies
    at2 = t.to_arrow()
    assert at2.column("name").to_pylist() == [None, "c", "d"]
    assert at2.column("name").null_count == 1


def test_utf8_arrow_export_pending_rows():
    """Rows still buffered in memory (not yet flushed) must export correctly."""
    pa = pytest.importorskip("pyarrow")
    t = make_table(["x", "y"])
    t.append(("pending", 99))
    at = pa.table(t)
    assert at.column("name").to_pylist() == ["x", "y", "pending"]


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


def test_utf8_from_arrow_fixed_width_max_len():
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


# ---------------------------------------------------------------------------
# utf8_array() constructor
# ---------------------------------------------------------------------------


def test_utf8_array_constructor():
    arr = blosc2.utf8_array(SAMPLE)
    assert isinstance(arr, blosc2.UTF8Array)
    assert len(arr) == len(SAMPLE)
    assert list(arr[:]) == SAMPLE


def test_utf8_array_constructor_with_spec_and_nulls():
    arr = blosc2.utf8_array(["a", None, "c"], blosc2.utf8(nullable=True, null_value="<NA>"))
    assert list(arr[:]) == ["a", "<NA>", "c"]


def test_utf8_array_ctor_rejects_none_if_not_null():
    with pytest.raises(TypeError, match="not nullable"):
        blosc2.utf8_array(["a", None])


def test_utf8_array_span_max_bytes_reads_only_offsets():
    arr = blosc2.utf8_array(["a", "café", "日本語"])  # 1, 5 and 9 UTF-8 bytes
    assert arr._span_max_bytes(0, 3) == 9
    assert arr._span_max_bytes(0, 2) == 5
    assert arr._span_max_bytes(0, 0) == 0
    # Pending (unflushed) rows are measured too.
    arr.append("x" * 20)
    assert arr._span_max_bytes(0, 4) == 20


# ---------------------------------------------------------------------------
# String expressions over utf8 columns (span-loop driver)
# ---------------------------------------------------------------------------


def test_ctable_utf8_where_expression_equality():
    t = make_table(["hello", "help", "world", "café"])
    assert list(t.where("name == 'hello'")["name"][:]) == ["hello"]
    assert list(t.where("name != 'hello'")["name"][:]) == ["help", "world", "café"]


def test_ctable_utf8_where_expr_vs_operator():
    t = make_table(["paris", "london", "tokyo", "paris"])
    for value in ("paris", "tokyo", "absent"):
        expr = list(t.where(f"name == '{value}'")["x"][:])
        operator = list(t[t.name == value]["x"][:])
        assert expr == operator, value


def test_ctable_utf8_where_expression_predicates():
    t = make_table(["hello", "help", "world"])
    assert list(t.where("startswith(name, 'hel')")["name"][:]) == ["hello", "help"]
    assert list(t.where("endswith(name, 'lo')")["name"][:]) == ["hello"]
    assert list(t.where("contains(name, 'l')")["name"][:]) == ["hello", "help", "world"]


def test_ctable_utf8_where_expr_mixes_numeric():
    t = make_table(["a", "b", "c", "d"])
    assert list(t.where("(name == 'b') | (x > 2)")["name"][:]) == ["b", "d"]
    assert list(t.where("(name != 'a') & (x < 2)")["name"][:]) == ["b"]


def test_ctable_utf8_where_expression_runs_on_miniexpr():
    """A silent NumPy fallback would produce the same values, so pin the engine.

    ``strict_miniexpr`` raises rather than falling back, which is the only
    assertion that distinguishes the two.
    """
    t = make_table(["hello", "help", "world"])
    got = t._utf8_span_eval("startswith(name, 'hel')", {}, ["name"], strict=True)
    assert list(got[:3]) == [True, True, False]


def test_ctable_utf8_where_expr_many_widths():
    # Exercises the power-of-two width bucketing: values straddle several
    # buckets and one of them is past the 255-byte typesize cap.
    values = ["a", "bb", "x" * 40, "y" * 300, "café", ""] * 30
    t = make_table(values)
    assert list(t.where("name == 'café'")["x"][:]) == [i for i, v in enumerate(values) if v == "café"]
    assert list(t.where("startswith(name, 'y')")["x"][:]) == [
        i for i, v in enumerate(values) if v.startswith("y")
    ]


def test_ctable_utf8_where_expr_splits_spans():
    # A single long row would size the whole span's <Un buffer; the driver must
    # split it rather than materialize rows x longest.
    values = ["hello", "help", "world"] * 10 + ["z" * 5000]
    t = make_table(values)
    t._UTF8_EXPR_BUDGET = 4096  # force the split path
    assert list(t.where("startswith(name, 'hel')")["x"][:]) == [
        i for i, v in enumerate(values) if v.startswith("hel")
    ]


def test_ctable_utf8_string_result_is_a_utf8_array():
    """utf8 is contagious: a string-returning expression stays variable-width.

    A ``<Un`` result would pad every row out to miniexpr's compile-time bound,
    which for ``lower()`` reserves a 2x case-expansion factor at 4 bytes per
    codepoint.  The driver returns a UTF8Array instead.
    """
    values = ["hello", "world", "café", ""]
    t = make_table(values)
    got = t._utf8_span_eval("'x=' + name", {}, ["name"], strict=True)
    assert isinstance(got, blosc2.UTF8Array)
    assert [str(v) for v in got[: len(values)]] == [f"x={v}" for v in values]
    # Physical length, like every other result from this driver.
    assert len(got) == len(t._valid_rows)
    assert all(str(v) == "" for v in got[len(values) :])


def test_ctable_utf8_bool_result_stays_a_numpy_array():
    t = make_table(["hello", "help", "world"])
    got = t._utf8_span_eval("startswith(name, 'hel')", {}, ["name"], strict=True)
    assert isinstance(got, np.ndarray)
    assert got.dtype == np.bool_


def test_ctable_utf8_result_across_spans():
    """The UTF8Array is extended span by span, so span order and row counts
    must line up exactly; a single-span run would not catch a drift."""
    values = [f"v{i}" for i in range(50)]
    t = make_table(values)
    t._UTF8_EXPR_SPAN = 7  # several spans, with a short final one
    got = t._utf8_span_eval("'x=' + name", {}, ["name"], strict=True)
    assert isinstance(got, blosc2.UTF8Array)
    assert [str(v) for v in got[: len(values)]] == [f"x={v}" for v in values]


def test_ctable_utf8_result_keeps_sentinel():
    t = CTable(NullableRow, new_data={"name": ["a", None, "c"], "x": [0, 1, 2]})
    got = t._utf8_span_eval("'p=' + name", {}, ["name"], strict=True)
    assert isinstance(got, blosc2.UTF8Array)
    assert got.spec.null_value == t["name"].null_value
    assert [str(v) for v in got[:3]] == ["p=a", t["name"].null_value, "p=c"]


def test_ctable_utf8_where_expr_multi_cols():
    @dataclass
    class TwoRow:
        first: str = blosc2.field(blosc2.utf8())
        second: str = blosc2.field(blosc2.utf8())

    t = CTable(
        TwoRow,
        new_data={"first": ["ab", "cd", "ef"], "second": ["ab", "xy", "ef"]},
    )
    assert list(t.where("first == second")["first"][:]) == ["ab", "ef"]


def test_ctable_utf8_where_expression_empty_table():
    t = make_table([])
    assert len(t.where("name == 'hello'")) == 0


def test_ctable_utf8_where_expr_view_delete():
    t = make_table(["paris", "london", "paris", "tokyo"])
    t.delete([0])
    assert list(t.where("name == 'paris'")["x"][:]) == [2]
    view = t[t.x > 1]
    assert list(view.where("name == 'paris'")["x"][:]) == [2]


# ---------------------------------------------------------------------------
# Null policy (3c): nulls are materialized to "" and re-masked afterwards
# ---------------------------------------------------------------------------


def _nullable_table(values):
    return CTable(
        NullableRow,
        new_data={"name": list(values), "x": list(range(len(values)))},
    )


def test_ctable_utf8_where_expr_nulls_no_match():
    t = _nullable_table(["hello", None, "help", None, "world"])
    assert list(t.where("name == 'hello'")["x"][:]) == [0]
    assert list(t.where("startswith(name, 'hel')")["x"][:]) == [0, 2]
    # Not even against the sentinel string itself: a null is not a value.
    assert list(t.where("name == '<NA>'")["x"][:]) == []


def test_ctable_utf8_where_expr_nulls_operator():
    t = _nullable_table(["hello", None, "help", None, "world"])
    for value in ("hello", "world", "<NA>"):
        assert list(t.where(f"name == '{value}'")["x"][:]) == list(t[t.name == value]["x"][:])
        assert list(t.where(f"name != '{value}'")["x"][:]) == list(t[t.name != value]["x"][:])


def test_ctable_utf8_where_expression_all_null_column():
    t = _nullable_table([None] * 5)
    assert list(t.where("name == 'hello'")["x"][:]) == []
    assert list(t.where("name != 'hello'")["x"][:]) == []


def test_ctable_utf8_where_expr_no_nulls_fast():
    # A nullable column with no actual nulls must not mask anything away.
    t = _nullable_table(["hello", "help", "world"])
    assert list(t.where("startswith(name, 'hel')")["x"][:]) == [0, 1]


def test_ctable_utf8_sum_where_expression():
    t = make_table(["hello", "help", "world"])
    assert t["x"].sum(where="startswith(name, 'hel')") == 1


# ---------------------------------------------------------------------------
# Scalar predicates take the raw-byte path instead of the span driver
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("expr", "predicate"),
    [
        ("name == 'help'", lambda v: v == "help"),
        ("name != 'help'", lambda v: v != "help"),
        ("name < 'm'", lambda v: v < "m"),
        ("name <= 'help'", lambda v: v <= "help"),
        ("name > 'm'", lambda v: v > "m"),
        ("name >= 'help'", lambda v: v >= "help"),
        ("'help' == name", lambda v: v == "help"),
        ("'help' != name", lambda v: v != "help"),
        ("'m' > name", lambda v: v < "m"),
        ("'m' <= name", lambda v: v >= "m"),
    ],
)
def test_ctable_utf8_scalar_predicates_match_python(expr, predicate):
    values = ["hello", "help", "world", "café", "日本語", "", "zz"]
    t = make_table(values)
    assert list(t.where(expr)["x"][:]) == [i for i, v in enumerate(values) if predicate(v)]


@pytest.mark.parametrize(
    ("expr", "rewritten_away"),
    [
        ("name == 'help'", True),
        ("'help' == name", True),
        ("name < 'm'", True),
        ("(name == 'help') | (x > 4)", True),
        ("(name == 'a') & (name != 'b')", True),
        # Not a scalar comparison: these still need the span driver.
        ("startswith(name, 'hel')", False),
        ("contains(name, 'l')", False),
        ("startswith(name, 'hel') | (name == 'zz')", False),
    ],
)
def test_ctable_utf8_preds_skip_span_driver(expr, rewritten_away):
    """The raw-byte scan is several times cheaper than decode -> <Un -> miniexpr.

    Correctness alone would not notice the difference, so assert on which route
    the expression takes: a utf8 name survives the rewrite only when something
    other than a scalar comparison still references it.
    """
    t = make_table(["hello", "help", "world", "zz", "a", "b"])
    operands = t._where_expression_operands(expr)
    _, _, remaining = t._rewrite_utf8_predicates(expr, operands, t._utf8_names_in(expr))
    assert (remaining == []) is rewritten_away


def test_ctable_utf8_rewritten_pred_vs_driver():
    """Both routes must agree, including on nulls and on the sentinel spelling."""
    values = ["hello", None, "help", None, "world"]
    t = _nullable_table(values)
    for expr in ("name == 'hello'", "name != 'hello'", "name < 'm'", "name == '<NA>'"):
        fast = list(t.where(expr)["x"][:])
        slow = list(np.flatnonzero(t._utf8_span_eval(expr, {}, ["name"])[: len(values)]))
        assert fast == slow, expr


def test_ctable_utf8_pred_literal_with_ops():
    # The literal is parsed with ast.literal_eval, so quoted operators and
    # spaces inside it must not be mistaken for expression syntax.
    values = ["a == b", "x > y", "plain"]
    t = make_table(values)
    assert list(t.where("name == 'a == b'")["x"][:]) == [0]
    assert list(t.where("name == 'x > y'")["x"][:]) == [1]


def test_ctable_utf8_pred_view_and_delete():
    t = make_table(["paris", "london", "paris", "tokyo"])
    t.delete([0])
    assert list(t.where("name == 'paris'")["x"][:]) == [2]
    view = t[t.x > 1]
    assert list(view.where("name == 'paris'")["x"][:]) == [2]


def test_ctable_utf8_two_preds_same_col():
    t = make_table(["a", "b", "c", "d"])
    assert list(t.where("(name > 'a') & (name < 'd')")["name"][:]) == ["b", "c"]


@pytest.mark.parametrize("op", ["==", "!=", "<", "<=", ">", ">="])
def test_utf8_array_comparisons_match_numpy(op):
    """Comparisons must be element-wise, not object identity.

    Without ``__eq__`` these fell through to identity, so ``arr == "hello"``
    was a plain ``False`` — silently wrong rather than an error.
    """
    import operator

    values = ["hello", "world", "héllo", "abc", "", "hello"]
    arr = blosc2.utf8_array(values)
    ref = np.array(values, dtype=arr.dtype)
    fn = getattr(operator, {"==": "eq", "!=": "ne", "<": "lt", "<=": "le", ">": "gt", ">=": "ge"}[op])

    for probe in ("hello", "héllo", "", "zzz"):
        got = fn(arr, probe)
        assert isinstance(got, np.ndarray)
        assert got.dtype == np.bool_
        np.testing.assert_array_equal(got, fn(ref, probe))


def test_utf8_array_comparison_against_array_likes():
    values = ["a", "bb", "ccc"]
    arr = blosc2.utf8_array(values)
    ref = np.array(values, dtype=arr.dtype)

    np.testing.assert_array_equal(arr == values, np.ones(3, dtype=bool))
    np.testing.assert_array_equal(arr == ref, np.ones(3, dtype=bool))
    np.testing.assert_array_equal(arr == blosc2.utf8_array(values), np.ones(3, dtype=bool))
    np.testing.assert_array_equal(arr != blosc2.utf8_array(["a", "x", "ccc"]), [False, True, False])


def test_utf8_array_comparison_edge_cases():
    # Unflushed pending rows take part in the comparison.
    arr = blosc2.utf8_array(["a"])
    arr.append("b")
    np.testing.assert_array_equal(arr == "b", [False, True])

    # Empty array yields an empty mask rather than raising.
    empty = blosc2.utf8_array([])
    assert (empty == "x").shape == (0,)

    # Defining __eq__ must not have made the container unhashable.
    assert isinstance(hash(arr), int)


def test_bare_utf8_expr_uses_span_driver():
    """A bare UTF8Array must not evaluate through the NumPy slices_eval path.

    That path returns correct-looking values while never reaching miniexpr,
    ignoring the span budget, and widening the result to a fixed ``<Un``.
    """
    values = ["hello", "world", "héllo"]
    arr = blosc2.utf8_array(values)

    result = blosc2.lazyexpr("'x=' + a", {"a": arr}).compute(strict_miniexpr=True)
    # Contagion: a string result over a utf8 operand stays variable-width.
    assert isinstance(result, blosc2.UTF8Array)
    assert list(result[:]) == ["x=" + v for v in values]

    # A boolean result is a plain NumPy array, as for a CTable column.
    mask = blosc2.lazyexpr("startswith(a, 'h')", {"a": arr}).compute(strict_miniexpr=True)
    assert isinstance(mask, np.ndarray)
    np.testing.assert_array_equal(mask, [True, False, True])


def test_bare_utf8_expr_mixed_operands():
    arr = blosc2.utf8_array(["hello", "world", "héllo"])
    other = blosc2.utf8_array(["A", "B", "C"])
    joined = blosc2.lazyexpr("a + b", {"a": arr, "b": other}).compute(strict_miniexpr=True)
    assert isinstance(joined, blosc2.UTF8Array)
    assert list(joined[:]) == ["helloA", "worldB", "hélloC"]

    nums = blosc2.asarray(np.array([1, 2, 3]))
    mixed = blosc2.lazyexpr("startswith(a, 'h') & (n > 1)", {"a": arr, "n": nums})
    np.testing.assert_array_equal(mixed.compute(strict_miniexpr=True), [False, False, True])


@pytest.mark.parametrize(("span_rows", "budget"), [(7, 64 << 20), (65536, 512)])
def test_bare_utf8_array_expression_splits_spans(span_rows, budget, monkeypatch):
    """Both the row-span and the byte-budget splits must hold over a bare array."""
    from blosc2 import _utf8_array

    values = [f"row-{i}" for i in range(50)]
    arr = blosc2.utf8_array(values)

    spans = []
    original = _utf8_array.utf8_spans
    monkeypatch.setattr(
        _utf8_array,
        "utf8_spans",
        lambda a, n, s, b: [spans.append(x) or x for x in original(a, n, s, b)],
    )
    monkeypatch.setattr(_utf8_array, "UTF8_EXPR_SPAN", span_rows)
    monkeypatch.setattr(_utf8_array, "UTF8_EXPR_BUDGET", budget)

    result = blosc2.lazyexpr("'x=' + a", {"a": arr}).compute(strict_miniexpr=True)
    assert len(spans) > 1, f"expected a split, got {spans}"
    assert list(result[:]) == ["x=" + v for v in values]


def test_bare_utf8_expr_rejects_unsupported():
    arr = blosc2.utf8_array(["a", "b"])
    lazy = blosc2.lazyexpr("upper(a)", {"a": arr})

    assert lazy.shape == (2,)
    assert len(lazy) == 2
    assert list(lazy[0:1]) == ["A"]

    with pytest.raises(NotImplementedError, match="whole-array only"):
        lazy.compute(item=slice(0, 1))
    with pytest.raises(NotImplementedError, match="not supported"):
        blosc2.lazyexpr("upper(a)", {"a": arr}, where=(arr, arr))


def test_ctable_utf8_index_reopen_nulls_last(tmp_path):
    """A persisted utf8 rank index must reopen and keep nulls at the end."""
    from dataclasses import make_dataclass

    path = str(tmp_path / "utf8_index.b2t")
    row_cls = make_dataclass("Row", [("name", str, blosc2.field(blosc2.utf8(nullable=True)))])
    values = ["pear", "apple", None, "banana"]
    t = blosc2.CTable(row_cls, urlpath=path, mode="w")
    t.extend({"name": values}, validate=False)
    t._flush_varlen_columns()
    t.create_index("name", kind="full")
    del t

    reopened = blosc2.open(path)
    assert reopened._get_index_catalog()["name"]["full"]["utf8_rank"]["null_rank"] == 3
    ordered = list(reopened.sort_by("name", view=True)["name"][:])
    assert ordered[:3] == ["apple", "banana", "pear"]
    assert ordered[3] == reopened["name"].null_value  # the null sentinel sorts last


def test_ctable_utf8_index_stale_on_change():
    """Appending a value ahead of existing ones invalidates every rank."""
    t = make_table(["pear", "banana"])
    t.create_index("name", kind="full")
    meta = t._get_index_catalog()["name"]["full"]["utf8_rank"]
    assert not t._utf8_rank_index_stale("name", meta)

    t.append({"name": "apple", "x": 2})
    t._flush_varlen_columns()
    assert t._utf8_rank_index_stale("name", meta)
    # The answer stays correct via the lexsort fallback.
    assert list(t.sort_by("name", view=True)["name"][:]) == ["apple", "banana", "pear"]


def test_utf8_rejects_a_lone_nul_null_value():
    """NumPy will not match a lone NUL against StringDType, so nulls would vanish."""
    import numpy as np

    probe = np.array(["\x00"], dtype=np.dtypes.StringDType())
    assert not (probe == "\x00")[0], "numpy started matching lone NUL; the guard can go"

    with pytest.raises(ValueError, match="NUL"):
        blosc2.utf8(null_value="\x00")
    # A NUL that is not the whole string is fine — numpy matches those.
    assert blosc2.utf8(null_value="\x00x").null_value == "\x00x"


@pytest.mark.parametrize("nullable", [False, True])
def test_ctable_utf8_index_answers_scalar_predicates(nullable, tmp_path):
    """With a rank index, a scalar comparison is a sidecar lookup, not a scan.

    The literal is located by one searchsorted over the stored vocabulary and
    the matching rows are a contiguous run of the sorted-positions sidecar.
    Results must be identical to the raw-byte scan, including that a null
    satisfies no comparison.
    """
    from dataclasses import make_dataclass

    import numpy as np

    values = ["pear", "apple", "café", "banana", "apple", "pear"]
    if nullable:
        values = [*values, None]
    spec = blosc2.utf8(nullable=True) if nullable else blosc2.utf8()
    row_cls = make_dataclass("Row", [("c", str, blosc2.field(spec))])

    masks = {}
    for tag in ("scan", "index"):
        t = blosc2.CTable(row_cls, urlpath=str(tmp_path / f"{tag}.b2t"), mode="w")
        t.extend({"c": values}, validate=False)
        t._flush_varlen_columns()
        if tag == "index":
            t.create_index("c", kind="full")
        col = t["c"]
        got = {}
        for name, op in (
            ("==", np.equal),
            ("!=", np.not_equal),
            ("<", np.less),
            ("<=", np.less_equal),
            (">", np.greater),
            (">=", np.greater_equal),
        ):
            for probe in ("apple", "pear", "zzz-absent", ""):
                got[(name, probe)] = col._utf8_scalar_mask(op, probe).copy()
        masks[tag] = got
        if tag == "index":
            # The fast path must really have been taken, not silently skipped.
            assert col._utf8_index_mask(np.equal, "apple") is not None
        del t

    for key, scanned in masks["scan"].items():
        np.testing.assert_array_equal(masks["index"][key], scanned, err_msg=f"{key}")


def test_ctable_utf8_index_pred_falls_back(tmp_path):
    """A stale rank index must not answer predicates from frozen ranks."""
    from dataclasses import make_dataclass

    import numpy as np

    row_cls = make_dataclass("Row", [("c", str, blosc2.field(blosc2.utf8()))])
    t = blosc2.CTable(row_cls, urlpath=str(tmp_path / "t.b2t"), mode="w")
    t.extend({"c": ["pear", "banana"]}, validate=False)
    t._flush_varlen_columns()
    t.create_index("c", kind="full")
    assert t["c"]._utf8_index_mask(np.equal, "pear") is not None

    t.append({"c": "apple"})  # a value ahead of the others invalidates every rank
    t._flush_varlen_columns()
    assert t["c"]._utf8_index_mask(np.equal, "pear") is None
    np.testing.assert_array_equal(t["c"]._utf8_scalar_mask(np.equal, "apple")[:3], [False, False, True])


# ---------------------------------------------------------------------------
# Fixed-width conversion pair: astype / from_utf8 / to_utf8
# ---------------------------------------------------------------------------


def test_utf8_astype_infers_exact_width():
    arr = blosc2.utf8_array(["hello", "café", "日本語", ""])
    out = arr.astype()
    assert out.dtype == np.dtype("<U5")
    assert list(out) == ["hello", "café", "日本語", ""]


def test_utf8_astype_unsized_u_also_infers():
    arr = blosc2.utf8_array(["hello", "café"])
    assert arr.astype("<U").dtype == np.dtype("<U5")
    assert arr.astype("U").dtype == np.dtype("<U5")


def test_utf8_astype_width_is_codepoints_not_bytes():
    """A byte-length bound would over-allocate 3-4x on non-ASCII text."""
    arr = blosc2.utf8_array(["日本語のテキスト", "a"])  # 8 chars, 24 bytes
    assert arr.astype().dtype == np.dtype("<U8")


def test_utf8_astype_width_truncates_like_np():
    arr = blosc2.utf8_array(["hello", "hi"])
    assert list(arr.astype("<U3")) == ["hel", "hi"]


def test_utf8_astype_rejects_non_string_dtype():
    arr = blosc2.utf8_array(["a"])
    with pytest.raises(ValueError, match="needs a U dtype"):
        arr.astype("int64")


def test_utf8_astype_empty_and_all_empty():
    assert blosc2.utf8_array([]).astype().shape == (0,)
    assert list(blosc2.utf8_array(["", ""]).astype()) == ["", ""]


@pytest.mark.parametrize("span_rows", [1, 3, 65536])
def test_utf8_astype_span_size_does_not_change_result(span_rows):
    values = ["", "a", "日本語", "x" * 40, "café", "🎉🚀"]
    arr = blosc2.utf8_array(values)
    out = arr.astype(span_rows=span_rows)
    assert out.dtype == np.dtype("<U40")
    assert list(out) == values


def test_utf8_astype_surfaces_null_sentinel():
    # A bare spec carries no sentinel until a table resolves its null policy.
    spec = blosc2.utf8(nullable=True, null_value="<NA>")
    arr = blosc2.utf8_array(["a", None], spec)
    assert list(arr.astype()) == ["a", "<NA>"]


def test_from_utf8_accepts_array_column_and_iterables():
    values = ["hello", "café"]
    arr = blosc2.utf8_array(values)
    t = make_table(values)

    for source in (arr, t["name"], np.array(values, dtype=STRING_DTYPE), values):
        out = blosc2.from_utf8(source)
        assert out.dtype == np.dtype("<U5"), source
        assert list(out) == values, source


def test_from_utf8_honours_explicit_dtype():
    arr = blosc2.utf8_array(["hello"])
    assert blosc2.from_utf8(arr, "<U2").dtype == np.dtype("<U2")


def test_to_utf8_from_fixed_width_and_lists():
    values = ["hello", "café", ""]
    for source in (np.array(values, dtype="<U5"), np.array(values, dtype=STRING_DTYPE), values):
        out = blosc2.to_utf8(source)
        assert isinstance(out, blosc2.UTF8Array)
        assert list(out[:]) == values, source


def test_to_utf8_honours_spec():
    spec = blosc2.utf8(nullable=True, null_value="<NA>")
    out = blosc2.to_utf8(["a", None], spec)
    assert out[1] == "<NA>"


def test_utf8_conversion_round_trip():
    values = ["", "a", "日本語のテキスト", "x" * 40, "🎉"]
    arr = blosc2.utf8_array(values)
    assert list(blosc2.to_utf8(blosc2.from_utf8(arr))[:]) == values


def test_utf8_conversion_round_trip_via_expr():
    """The documented compute rule, end to end."""
    t = make_table(["a", "bb", "ccc"])
    fixed = blosc2.from_utf8(t["name"])
    res = blosc2.lazyexpr("'x=' + a", {"a": fixed}).compute()[:]
    t.add_column("prefixed", blosc2.utf8(), values=blosc2.to_utf8(res))
    assert list(t["prefixed"][:]) == ["x=a", "x=bb", "x=ccc"]


# ---------------------------------------------------------------------------
# Column.assign on utf8
# ---------------------------------------------------------------------------


def test_ctable_utf8_column_assign():
    t = make_table(["a", "bb", "ccc"])
    t["name"].assign(["X", "YY", "ZZZ"])
    assert list(t["name"][:]) == ["X", "YY", "ZZZ"]


def test_ctable_utf8_col_assign_from_computed():
    t = make_table(["a", "bb"])
    fixed = blosc2.from_utf8(t["name"])
    res = blosc2.lazyexpr("a + '!'", {"a": fixed}).compute()[:]
    t["name"].assign(res)
    assert list(t["name"][:]) == ["a!", "bb!"]


def test_ctable_utf8_column_assign_skips_deleted_rows():
    t = make_table(["a", "b", "c", "d"])
    t.delete([0, 2])
    t["name"].assign(["P", "Q"])
    assert list(t["name"][:]) == ["P", "Q"]
    assert list(t["x"][:]) == [1, 3]


def test_ctable_utf8_column_assign_wrong_length_raises():
    t = make_table(["a", "bb"])
    with pytest.raises(ValueError, match="requires 2 values"):
        t["name"].assign(["only-one"])


def test_ctable_utf8_column_assign_persists(tmp_path):
    path = str(tmp_path / "utf8_assign.b2d")
    t = make_table(["a", "bb"], urlpath=path, mode="w")
    t["name"].assign(["hello", "wörld"])
    t.close()
    t2 = CTable.open(path)
    assert list(t2["name"][:]) == ["hello", "wörld"]


# ---------------------------------------------------------------------------
# Compute-side refusals: they must name the column and route to the conversion
# ---------------------------------------------------------------------------


@blosc2.dsl_kernel
def _shout(name):
    return name.upper()


def _assert_routes(message, source):
    """Every utf8 compute refusal must hand back a usable recipe."""
    assert "blosc2.from_utf8(" in message
    assert "blosc2.to_utf8(" in message
    assert source in message
    assert "Computing strings on a utf8 column" in message


def test_utf8_computed_col_names_workaround():
    t = make_table(["a", "bb"])
    with pytest.raises(NotImplementedError) as exc:
        t.add_computed_column("up", "upper(name)")
    _assert_routes(str(exc.value), "t['name']")
    assert "upper(name)" in str(exc.value)  # the user's own expression is echoed


def test_utf8_assign_expression_names_the_workaround():
    t = make_table(["a", "bb"])
    with pytest.raises(NotImplementedError) as exc:
        t.assign(up="upper(name)")
    _assert_routes(str(exc.value), "t['name']")


def test_utf8_generated_col_names_workaround():
    t = make_table(["a", "bb"])
    with pytest.raises(NotImplementedError) as exc:
        t.add_generated_column("g", values="upper(name)")
    _assert_routes(str(exc.value), "t['name']")


def test_utf8_kernel_refused_at_registration():
    """Regression: it used to register, then break every read *and* str(t)."""
    t = make_table(["a", "bb"])
    with pytest.raises(NotImplementedError) as exc:
        t.add_computed_column("up", _shout, inputs=["name"])
    _assert_routes(str(exc.value), "t['name']")
    # The table must be left untouched and usable.
    assert "up" not in t.col_names
    assert list(t["name"][:]) == ["a", "bb"]
    assert "name" in str(t)


def test_utf8_kernel_refused_whatever_returned():
    """It is the utf8 operand that cannot work, not the string output."""

    @blosc2.dsl_kernel
    def is_long(name):
        return name > "b"

    t = make_table(["a", "bb"])
    with pytest.raises(NotImplementedError, match="cannot be a UDF"):
        t.add_computed_column("flag", is_long, inputs=["name"])


def test_utf8_apply_names_the_column():
    t = make_table(["a", "bb"])
    with pytest.raises(NotImplementedError) as exc:
        t.apply(_shout, columns=["name"])
    _assert_routes(str(exc.value), "t['name']")


def test_utf8_lazyudf_over_a_column_names_the_column():
    t = make_table(["a", "bb"])
    with pytest.raises(NotImplementedError) as exc:
        blosc2.lazyudf(_shout, (t["name"],))
    _assert_routes(str(exc.value), "t['name']")


def test_utf8_lazyudf_over_a_bare_array_routes_too():
    arr = blosc2.utf8_array(["a", "bb"])
    with pytest.raises(NotImplementedError) as exc:
        blosc2.lazyudf(_shout, (arr,))
    msg = str(exc.value)
    assert "blosc2.from_utf8(arr)" in msg
    assert "blosc2.to_utf8(" in msg
    # No table to assign into, so the message must not suggest one.
    assert ".assign(" not in msg


def test_utf8_refusal_recipe_actually_works():
    """The recipe the error prints must run as printed."""
    t = make_table(["a", "bb"])
    fixed = blosc2.from_utf8(t["name"])
    res = blosc2.lazyexpr("upper(name)", {"name": fixed}).compute()[:]
    t.add_column("out", blosc2.utf8(), values=blosc2.to_utf8(res))
    assert list(t["out"][:]) == ["A", "BB"]

    res = blosc2.lazyudf(_shout, (fixed,)).compute()[:]
    assert list(blosc2.to_utf8(res)[:]) == ["A", "BB"]


def test_non_utf8_dsl_kernel_column_still_works():
    """The guard must not catch ordinary columns."""

    @blosc2.dsl_kernel
    def double(x):
        return x * 2

    t = make_table(["a", "bb"])
    t.add_computed_column("dbl", double, inputs=["x"])
    np.testing.assert_array_equal(t["dbl"][:], [0, 2])


# ---------------------------------------------------------------------------
# NumPy StringDType interop: dtype-based dispatch to UTF8Array
# ---------------------------------------------------------------------------


def test_utf8_array_satisfies_array_protocol():
    arr = blosc2.utf8_array(["a", "bb", "ccc"])
    assert isinstance(arr, blosc2.Array)
    assert arr.shape == (3,)
    assert arr.ndim == 1
    assert arr.size == 3
    assert arr.dtype == STRING_DTYPE


def test_utf8_array_np_asarray_keeps_string_dtype():
    """np.asarray() used to iterate the rows and infer a fixed-width <Un."""
    arr = blosc2.utf8_array(["x" * 3, "y" * 200])
    out = np.asarray(arr)
    assert out.dtype == STRING_DTYPE
    assert list(out) == ["x" * 3, "y" * 200]
    # ... and the same dtype arr[:] reports, which is the point.
    assert out.dtype == arr[:].dtype


def test_utf8_array_np_asarray_honours_dtype():
    arr = blosc2.utf8_array(["abc", "de"])
    assert list(np.asarray(arr, dtype="<U5")) == ["abc", "de"]
    assert np.asarray(arr, dtype="<U5").dtype == np.dtype("<U5")


def test_asarray_dispatches_string_dtype_to_utf8array():
    src = np.array(["a", "bb", "日本語"], dtype=STRING_DTYPE)
    out = blosc2.asarray(src)
    assert isinstance(out, blosc2.UTF8Array)
    assert list(out[:]) == ["a", "bb", "日本語"]


def test_asarray_dispatches_on_the_target_dtype():
    """A fixed-width source asked for StringDType becomes variable-length."""
    out = blosc2.asarray(np.array(["a", "bb"], dtype="<U2"), dtype=STRING_DTYPE)
    assert isinstance(out, blosc2.UTF8Array)
    assert list(out[:]) == ["a", "bb"]


def test_asarray_leaves_utf8_for_fixed_width():
    src = np.array(["a", "bb"], dtype=STRING_DTYPE)
    out = blosc2.asarray(src, dtype="<U8")
    assert isinstance(out, blosc2.NDArray)
    assert out.dtype == np.dtype("<U8")
    assert list(out[:]) == ["a", "bb"]

    out = blosc2.asarray(blosc2.utf8_array(["a", "bb"]), dtype="<U8")
    assert isinstance(out, blosc2.NDArray)
    assert list(out[:]) == ["a", "bb"]


def test_asarray_returns_a_utf8array_unchanged():
    arr = blosc2.utf8_array(["a", "bb"])
    assert blosc2.asarray(arr) is arr
    copied = blosc2.asarray(arr, copy=True)
    assert copied is not arr
    assert list(copied[:]) == ["a", "bb"]


def test_asarray_str_dtype_rejects_nd_kwargs():
    src2d = np.array([["a"], ["b"]], dtype=STRING_DTYPE)
    with pytest.raises(ValueError, match="1-D only"):
        blosc2.asarray(src2d)
    with pytest.raises(TypeError, match="utf8_array"):
        blosc2.asarray(np.array(["a"], dtype=STRING_DTYPE), urlpath="unused.b2nd")


def test_asarray_non_string_paths_are_unchanged():
    assert isinstance(blosc2.asarray(np.arange(3)), blosc2.NDArray)
    fixed = blosc2.asarray(np.array(["a", "bb"], dtype="<U2"))
    assert isinstance(fixed, blosc2.NDArray)
    assert fixed.dtype == np.dtype("<U2")
    # A plain list of str still infers a fixed width, as it always did.
    assert isinstance(blosc2.asarray(["a", "bb"]), blosc2.NDArray)


@pytest.mark.parametrize(
    ("call", "expected"),
    [
        (lambda d: blosc2.zeros(3, dtype=d), ["", "", ""]),
        (lambda d: blosc2.empty(3, dtype=d), ["", "", ""]),
        (lambda d: blosc2.ones(3, dtype=d), ["1", "1", "1"]),
        (lambda d: blosc2.full(3, "x", dtype=d), ["x", "x", "x"]),
    ],
)
def test_constructors_with_string_dtype_match_numpy(call, expected):
    out = call(STRING_DTYPE)
    assert isinstance(out, blosc2.UTF8Array)
    assert list(out[:]) == expected


def test_constructors_string_dtype_vs_numpy():
    d = STRING_DTYPE
    assert list(blosc2.zeros(3, dtype=d)[:]) == list(np.zeros(3, dtype=d))
    assert list(blosc2.ones(3, dtype=d)[:]) == list(np.ones(3, dtype=d))
    assert list(blosc2.full(3, "x", dtype=d)[:]) == list(np.full(3, "x", dtype=d))


def test_constructors_string_dtype_reject_nd():
    with pytest.raises(ValueError, match="1-D only"):
        blosc2.zeros((2, 3), dtype=STRING_DTYPE)
    with pytest.raises(TypeError, match="utf8_array"):
        blosc2.zeros(3, dtype=STRING_DTYPE, urlpath="unused.b2nd")


def test_constructors_string_dtype_do_not_materialize_a_fill_list():
    """The fill is one string repeated; building a list of it is pure overhead.

    A list would hold shape[0] pointers to the same object before the packer
    sees any of them, which is what makes zeros(10_000_000, StringDType())
    expensive for no reason.
    """
    import tracemalloc

    n = 1_000_000
    tracemalloc.start()
    try:
        arr = blosc2.zeros(n, dtype=STRING_DTYPE)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert len(arr) == n
    assert arr[0] == ""
    # A list of n pointers alone is 8n bytes (~7.6 MiB here) on top of the
    # array itself; the streamed build stays far below that.
    assert peak < 4 * 2**20, f"peak {peak / 2**20:.1f} MiB suggests a materialized fill list"


def test_utf8_dispatch_round_trips_conversion():
    out = blosc2.full(2, "hé", dtype=STRING_DTYPE)
    assert list(blosc2.to_utf8(blosc2.from_utf8(out))[:]) == ["hé", "hé"]


# ---------------------------------------------------------------------------
# Nested (dotted) utf8 leaves
# ---------------------------------------------------------------------------


def _nested_table(**kwargs):
    """A table whose utf8 column is addressed by a dotted path.

    Two leaves under overlapping prefixes, so the longest-name-first aliasing
    is exercised: rewriting "trip.who" first would corrupt "trip.begin.who".
    """
    names = ["alice", "bob", "carol", "dave"]
    t = CTable(
        Row,
        new_data={"name": names, "x": list(range(len(names)))},
        **kwargs,
    )
    t.rename_column("name", "trip.begin.who")
    t.add_column("trip.who", blosc2.utf8(), values=[n.upper() for n in names])
    return t


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        ('trip.begin.who == "bob"', [1]),
        ('"bob" == trip.begin.who', [1]),
        ('trip.begin.who != "bob"', [0, 2, 3]),
        ('trip.begin.who < "c"', [0, 1]),
        ('startswith(trip.begin.who, "c")', [2]),
        ('upper(trip.begin.who) == "DAVE"', [3]),
        ('startswith(trip.begin.who, "c") & (x > 1)', [2]),
        # Both leaves at once, and the shorter name is a prefix of the longer.
        ('(trip.begin.who == "bob") | (trip.who == "CAROL")', [1, 2]),
    ],
)
def test_ctable_utf8_nested_leaf_filters(expr, expected):
    # Dotted utf8 leaves are outside the operand namespace, so they reach the
    # utf8 driver still spelled with dots -- which no expression engine parses.
    t = _nested_table()
    assert list(t.where(expr)["x"][:]) == expected


def test_ctable_utf8_nested_leaf_matches_flat():
    """A dotted name must not change the answer the same data gives flat."""
    values = ["hello", "help", "world", "zz"]
    flat = make_table(values)
    nested = make_table(values)
    nested.rename_column("name", "trip.who")
    for flat_expr, nested_expr in (
        ("name == 'hello'", 'trip.who == "hello"'),
        ("startswith(name, 'hel')", 'startswith(trip.who, "hel")'),
        ("name < 'w'", 'trip.who < "w"'),
    ):
        assert list(flat.where(flat_expr)["x"][:]) == list(nested.where(nested_expr)["x"][:])


def test_ctable_utf8_nested_leaf_sum_persist(tmp_path):
    urlpath = str(tmp_path / "utf8_nested.b2z")
    t = _nested_table(urlpath=urlpath, mode="w")
    assert t["x"].sum(where='startswith(trip.begin.who, "c")') == 2
    t.close()

    reopened = CTable.open(urlpath, mode="r")
    try:
        assert list(reopened.where('trip.begin.who == "dave"')["x"][:]) == [3]
    finally:
        reopened.close()
