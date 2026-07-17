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
# Chunked bulk extend (write path)
# ---------------------------------------------------------------------------


def test_utf8_array_extend_empty_iterable_is_noop():
    from blosc2.utf8_array import Utf8Array

    arr = Utf8Array(blosc2.utf8())
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
    from blosc2.utf8_array import _FLUSH_ROWS, Utf8Array

    n = _FLUSH_ROWS * 3 + 7
    values = [f"row{i}" for i in range(n)]
    arr = Utf8Array(blosc2.utf8())
    arr.extend(values)
    assert len(arr) == n
    arr.flush()
    assert len(arr) == n
    assert list(arr[:]) == values


def test_utf8_array_extend_none_straddles_chunk_boundary():
    from blosc2.utf8_array import _FLUSH_ROWS, Utf8Array

    values = [f"v{i}" for i in range(_FLUSH_ROWS + 2)]
    values[_FLUSH_ROWS - 1] = None  # last row of first chunk
    values[_FLUSH_ROWS + 1] = None  # second row of second chunk
    arr = Utf8Array(blosc2.utf8(null_value="<NA>"))
    arr.extend(values)
    expected = [v if v is not None else "<NA>" for v in values]
    assert list(arr[:]) == expected


def test_utf8_array_extend_append_interleaved_before_flush():
    from blosc2.utf8_array import Utf8Array

    arr = Utf8Array(blosc2.utf8())
    arr.append("first")
    arr.extend(["second", "third"])
    arr.append("fourth")
    arr.extend(["fifth"])
    assert list(arr[:]) == ["first", "second", "third", "fourth", "fifth"]


def test_utf8_array_extend_ascii_nul_byte_preserved():
    from blosc2.utf8_array import Utf8Array

    values = ["nul\x00in", "plain", "\x00leading", "trailing\x00"]
    assert all(v.isascii() for v in values)
    arr = Utf8Array(blosc2.utf8())
    arr.extend(values)
    arr.flush()
    assert list(arr[:]) == values


def test_utf8_array_extend_multi_mb_strings_bounded_flush():
    """~20 multi-MB ASCII strings: char-count flush bound is checked once
    per _FLUSH_ROWS-sized chunk (not per row), so this overshoots
    _FLUSH_CHARS by at most one chunk before flushing -- confirm read-back
    is still correct despite the coarser check."""
    from blosc2.utf8_array import Utf8Array

    values = [f"{i:06d}" + "x" * (2 * 1024 * 1024) for i in range(20)]
    arr = Utf8Array(blosc2.utf8())
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
        monkeypatch.setattr("blosc2.utf8_array._pack_utf8_kernel", lambda: None)
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
    from blosc2.utf8_array import Utf8Array

    arr = Utf8Array(blosc2.utf8())
    arr.extend(SAMPLE)
    arr.flush()
    got = arr[:]
    assert got.dtype == STRING_DTYPE
    assert list(got) == SAMPLE


def test_utf8_array_bulk_read_matches_python_ground_truth(force_kernel_mode):
    """A wider mix of byte lengths and edge cases than SAMPLE: many distinct
    ASCII/multi-byte/empty/NUL-bearing values, read back in one bulk span."""
    from blosc2.utf8_array import Utf8Array

    rng = np.random.default_rng(5)
    pool = ["", "a", "café", "日本語", "x" * 5000, "nul\x00in", "nul\x00INSIDE", "emoji 🎉🚀"]
    values = [pool[i] for i in rng.integers(0, len(pool), 3000)]
    arr = Utf8Array(blosc2.utf8())
    arr.extend(values)
    arr.flush()
    assert list(arr[:]) == values


def test_ctable_utf8_extend_and_read_kernel_and_fallback(force_kernel_mode):
    t = make_table()
    values = t["name"][:]
    assert values.dtype == STRING_DTYPE
    assert list(values) == SAMPLE


@pytest.mark.parametrize("ext", [".b2z", ".b2d"])
def test_ctable_utf8_persistence_roundtrip_kernel_and_fallback(tmp_path, ext, force_kernel_mode):
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
        monkeypatch.setattr("blosc2.utf8_array._encode_utf8_kernel", lambda: None)
    return request.param


def test_utf8_array_extend_kernel_and_fallback(force_write_kernel_mode):
    from blosc2.utf8_array import Utf8Array

    arr = Utf8Array(blosc2.utf8())
    arr.extend(SAMPLE)
    arr.flush()
    assert list(arr[:]) == SAMPLE


def test_utf8_array_extend_matches_python_ground_truth(force_write_kernel_mode):
    """Same wider mix of byte lengths and edge cases as the read-side
    ground-truth test, exercised through the write path this time."""
    from blosc2.utf8_array import Utf8Array

    rng = np.random.default_rng(7)
    pool = ["", "a", "café", "日本語", "x" * 5000, "nul\x00in", "nul\x00INSIDE", "emoji 🎉🚀"]
    values = [pool[i] for i in rng.integers(0, len(pool), 3000)]
    arr = Utf8Array(blosc2.utf8())
    arr.extend(values)
    arr.flush()
    assert list(arr[:]) == values


def test_utf8_array_extend_ascii_nul_byte_kernel_and_fallback(force_write_kernel_mode):
    from blosc2.utf8_array import Utf8Array

    values = ["nul\x00in", "plain", "\x00leading", "trailing\x00"]
    arr = Utf8Array(blosc2.utf8())
    arr.extend(values)
    arr.flush()
    assert list(arr[:]) == values


def test_utf8_array_extend_multi_mb_string_kernel_and_fallback(force_write_kernel_mode):
    """A single multi-MB value alongside short ones -- sanity-checks the
    total-length/offset accumulation in the compiled kernel's two passes."""
    from blosc2.utf8_array import Utf8Array

    values = ["head", "x" * (8 * 1024 * 1024), "tail", "café" * 100_000]
    arr = Utf8Array(blosc2.utf8())
    arr.extend(values)
    arr.flush()
    assert list(arr[:]) == values


def test_ctable_utf8_extend_kernel_and_fallback(force_write_kernel_mode):
    t = make_table()
    assert list(t["name"][:]) == SAMPLE


def test_utf8_array_extend_lone_surrogate_raises_and_recovers(force_write_kernel_mode):
    """A lone surrogate is invalid UTF-8: flush() must raise
    UnicodeEncodeError, matching str.encode('utf-8')'s own behavior, and
    the array must remain usable afterwards -- a regression test for the
    compiled kernel's temp-buffer cleanup on the error path."""
    from blosc2.utf8_array import Utf8Array

    arr = Utf8Array(blosc2.utf8())
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


def test_ctable_utf8_comparison_with_non_string_scalar_raises():
    t = make_table()
    with pytest.raises(TypeError, match="utf8"):
        t.name == 42  # noqa: B015
    with pytest.raises(TypeError, match="utf8"):
        t.name < 3.14  # noqa: B015


def test_ctable_utf8_comparison_with_mismatched_column_type_raises():
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


def test_ctable_utf8_ordering_multibyte_byte_length_boundaries():
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


def test_ctable_utf8_scalar_comparison_view_and_deleted_rows():
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


def test_utf8_factorize_span_matches_np_unique_contract():
    """The raw-bytes factorization keeps the np.unique contract: uniques
    sorted ascending, codes indexing them.  Ground truth is Python's set —
    numpy's np.unique on StringDType merges strings differing only after an
    embedded NUL (numpy bug), which the byte-exact factorization does not.
    """
    from blosc2.utf8_array import Utf8Array

    rng = np.random.default_rng(7)
    pool = ["", "a", "ab", "café", "日本語", "x" * 3000, "nul\x00in", "nul\x00IN", "Wien", "wien"]
    values = [pool[i] for i in rng.integers(0, len(pool), 5000)]
    arr = Utf8Array(blosc2.utf8())
    arr.extend(values)
    codes, uniques = arr.factorize_span(0, len(values))
    assert list(uniques) == sorted(set(values))
    assert all(uniques[c] == v for c, v in zip(codes, values, strict=True))


def test_utf8_factorizer_cross_span_codes_are_global():
    from blosc2.utf8_array import Utf8Array

    arr = Utf8Array(blosc2.utf8())
    arr.extend(["b", "a", "b", "c", "a", "d"])
    fact = arr.factorizer()
    c1 = fact.codes_for_span(0, 3)  # b, a, b
    c2 = fact.codes_for_span(3, 6)  # c, a, d
    uniques = fact.uniques()
    assert [uniques[c] for c in c1] == ["b", "a", "b"]
    assert [uniques[c] for c in c2] == ["c", "a", "d"]
    # "a" keeps the code it was assigned in the first span
    assert c1[1] == c2[1]


def test_ctable_utf8_groupby_many_byte_lengths_and_non_ascii():
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


def test_ctable_utf8_sort_multi_key_with_bystander_utf8_column():
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
def test_ctable_utf8_sort_inplace_persists_after_reopen(tmp_path, ext):
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


def test_ctable_utf8_setitem_persisted_shifts_survive_reopen(tmp_path):
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


def test_ctable_utf8_where_expression_raises_clearly():
    t = make_table()
    with pytest.raises(NotImplementedError, match="utf8"):
        t.where("name == 'hello'")


def test_ctable_utf8_create_index_raises_clearly():
    t = make_table()
    with pytest.raises(NotImplementedError, match="utf8"):
        t.create_index(col_name="name")


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
