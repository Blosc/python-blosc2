#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

"""Tests for vlstring / vlbytes schema specs and CTable integration (Phases 2+3)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

import blosc2

# ---------------------------------------------------------------------------
# Schema spec round-trip tests
# ---------------------------------------------------------------------------


def test_vlstring_spec_defaults():
    spec = blosc2.vlstring()
    assert spec.nullable is False
    assert spec.serializer == "msgpack"
    assert spec.batch_rows == 2048
    assert spec.items_per_block is None
    assert spec.dtype is None
    assert spec.python_type is str


def test_vlstring_rejects_unsupported_serializer():
    with pytest.raises(ValueError, match="serializer='msgpack'"):
        blosc2.vlstring(serializer="arrow")


def test_vlbytes_spec_defaults():
    spec = blosc2.vlbytes()
    assert spec.nullable is False
    assert spec.serializer == "msgpack"
    assert spec.batch_rows == 2048
    assert spec.items_per_block is None
    assert spec.dtype is None
    assert spec.python_type is bytes


def test_vlstring_spec_metadata_round_trip():
    from blosc2.schema_compiler import spec_from_metadata_dict

    spec = blosc2.vlstring(nullable=True, batch_rows=512)
    d = spec.to_metadata_dict()
    assert d["kind"] == "vlstring"
    assert d["nullable"] is True
    assert d["batch_rows"] == 512
    assert "items_per_block" not in d  # None → omitted

    restored = spec_from_metadata_dict(d)
    assert type(restored).__name__ == "VLStringSpec"
    assert restored.nullable is True
    assert restored.batch_rows == 512


def test_vlbytes_spec_metadata_round_trip():
    from blosc2.schema_compiler import spec_from_metadata_dict

    spec = blosc2.vlbytes(nullable=False, items_per_block=64)
    d = spec.to_metadata_dict()
    assert d["kind"] == "vlbytes"
    assert d["nullable"] is False
    assert d["items_per_block"] == 64

    restored = spec_from_metadata_dict(d)
    assert type(restored).__name__ == "VLBytesSpec"
    assert restored.items_per_block == 64


def test_vlstring_display_width():
    from blosc2.schema_compiler import compute_display_width

    assert compute_display_width(blosc2.vlstring()) == 40
    assert compute_display_width(blosc2.vlbytes()) == 40
    assert compute_display_width(blosc2.list(blosc2.int64())) == 40


# ---------------------------------------------------------------------------
# _ScalarVarLenArray internal adapter tests
# ---------------------------------------------------------------------------


def _make_sva(spec):
    from blosc2.scalar_array import _ScalarVarLenArray

    return _ScalarVarLenArray(spec)


def test_scalar_varlen_array_str_basic():
    spec = blosc2.vlstring(batch_rows=4)
    sva = _make_sva(spec)

    sva.append("hello")
    sva.append("world")
    assert len(sva) == 2
    assert sva[0] == "hello"
    assert sva[1] == "world"
    assert sva[-1] == "world"


def test_scalar_varlen_array_bytes_basic():
    spec = blosc2.vlbytes(batch_rows=4)
    sva = _make_sva(spec)

    sva.append(b"foo")
    sva.append(bytearray(b"bar"))
    assert len(sva) == 2
    assert sva[0] == b"foo"
    assert sva[1] == b"bar"


def test_scalar_varlen_array_flush_and_persist():
    """Flushing moves items from pending to backend."""
    spec = blosc2.vlstring(batch_rows=4)
    sva = _make_sva(spec)

    sva.extend(["a", "bb", "ccc"])
    assert sva._persisted_row_count == 0
    assert len(sva._pending) == 3

    sva.flush()
    assert sva._persisted_row_count == 3
    assert len(sva._pending) == 0
    assert len(sva) == 3
    assert sva[0] == "a"
    assert sva[2] == "ccc"


def test_scalar_varlen_array_auto_flush_on_full_batch():
    """Auto-flush happens when pending reaches batch_rows."""
    spec = blosc2.vlstring(batch_rows=3)
    sva = _make_sva(spec)

    sva.extend(["x", "y", "z"])  # exactly batch_rows → auto-flush
    assert sva._persisted_row_count == 3
    assert len(sva._pending) == 0

    sva.append("w")
    assert len(sva) == 4
    assert sva[3] == "w"


def test_scalar_varlen_array_cross_batch_access():
    """Access items that span multiple persisted batches plus pending."""
    spec = blosc2.vlstring(batch_rows=3)
    sva = _make_sva(spec)

    values = [f"item{i}" for i in range(8)]
    sva.extend(values)  # 6 auto-flushed + 2 pending
    sva.flush()  # flush remaining 2

    for i, v in enumerate(values):
        assert sva[i] == v, f"mismatch at {i}"


def test_scalar_varlen_array_setitem_pending():
    spec = blosc2.vlstring(batch_rows=10)
    sva = _make_sva(spec)
    sva.extend(["a", "b", "c"])
    sva[1] = "B"
    assert sva[1] == "B"


def test_scalar_varlen_array_setitem_persisted():
    spec = blosc2.vlstring(batch_rows=3)
    sva = _make_sva(spec)
    sva.extend(["a", "b", "c"])  # auto-flushed
    assert sva._persisted_row_count == 3
    sva[1] = "B"
    assert sva[1] == "B"
    assert sva[0] == "a"
    assert sva[2] == "c"


def test_scalar_varlen_array_bulk_getitem():
    spec = blosc2.vlstring(batch_rows=3)
    sva = _make_sva(spec)
    values = ["a", "b", "c", "d", "e"]
    sva.extend(values)
    sva.flush()

    result = sva[[0, 2, 4]]
    assert result == ["a", "c", "e"]

    result = sva[1:4]
    assert result == ["b", "c", "d"]


def test_scalar_varlen_array_nullable():
    spec = blosc2.vlstring(nullable=True, batch_rows=4)
    sva = _make_sva(spec)
    sva.extend(["hello", None, "world", None])
    sva.flush()

    assert sva[0] == "hello"
    assert sva[1] is None
    assert sva[3] is None


def test_scalar_varlen_array_rejects_none_when_not_nullable():
    spec = blosc2.vlstring(nullable=False)
    sva = _make_sva(spec)
    with pytest.raises(TypeError, match="not nullable"):
        sva.append(None)


def test_scalar_varlen_array_rejects_wrong_type_str():
    spec = blosc2.vlstring()
    sva = _make_sva(spec)
    with pytest.raises(TypeError):
        sva.append(b"bytes instead of str")


def test_scalar_varlen_array_rejects_wrong_type_bytes():
    spec = blosc2.vlbytes()
    sva = _make_sva(spec)
    with pytest.raises(TypeError):
        sva.append("str instead of bytes")


def test_scalar_varlen_array_iter():
    spec = blosc2.vlstring(batch_rows=3)
    sva = _make_sva(spec)
    values = ["x", "y", "z", "w"]
    sva.extend(values)
    sva.flush()
    assert list(sva) == values


# ---------------------------------------------------------------------------
# CTable schema definition helpers
# ---------------------------------------------------------------------------


@dataclass
class VLRow:
    id: int = blosc2.field(blosc2.int64())
    text: str = blosc2.field(blosc2.vlstring())
    data: bytes = blosc2.field(blosc2.vlbytes())


@dataclass
class VLNullRow:
    id: int = blosc2.field(blosc2.int64())
    text: str = blosc2.field(blosc2.vlstring(nullable=True))


# ---------------------------------------------------------------------------
# CTable basic CRUD tests
# ---------------------------------------------------------------------------


ROWS = [
    (0, "hello world", b"bin0"),
    (1, "a" * 200, b"x" * 500),
    (2, "", b""),
    (3, "unicode: \u00e9\u00e0\u00fc", b"\x00\x01\x02"),
    (4, "short", b"also short"),
]


def test_ctable_vlstring_vl_bytes_append_getitem():
    ct = blosc2.CTable(VLRow, new_data=ROWS)
    assert len(ct) == 5

    for i, (exp_id, exp_text, exp_data) in enumerate(ROWS):
        assert int(ct.id[i]) == exp_id
        assert ct.text[i] == exp_text
        assert ct.data[i] == exp_data


def test_ctable_vlstring_column_getitem():
    ct = blosc2.CTable(VLRow, new_data=ROWS)
    col = ct.text

    assert col[0] == "hello world"
    assert col[-1] == "short"
    assert col[1:3] == ["a" * 200, ""]
    assert col[[0, 2, 4]] == ["hello world", "", "short"]


def test_ctable_vlbytes_column_getitem():
    ct = blosc2.CTable(VLRow, new_data=ROWS)
    col = ct.data
    assert col[0] == b"bin0"
    assert col[2] == b""


def test_ctable_vlstring_column_setitem():
    ct = blosc2.CTable(VLRow, new_data=ROWS)
    ct.text[0] = "changed"
    assert ct.text[0] == "changed"


def test_ctable_vlstring_column_iter():
    ct = blosc2.CTable(VLRow, new_data=ROWS)
    texts = list(ct.text)
    assert texts == [r[1] for r in ROWS]


def test_ctable_vlstring_column_is_not_list():
    ct = blosc2.CTable(VLRow, new_data=ROWS)
    assert not ct.text.is_list
    assert ct.text.is_varlen_scalar


def test_ctable_vlstring_column_null_count_non_nullable():
    ct = blosc2.CTable(VLRow, new_data=ROWS)
    # Non-nullable: no Nones → null_count = 0
    assert ct.text.null_count() == 0
    assert ct.text.null_value is None


def test_ctable_vlstring_nullable_null_count():
    data = [(0, "hello"), (1, None), (2, "world"), (3, None)]
    ct = blosc2.CTable(VLNullRow, new_data=data)
    assert ct.text.null_count() == 2
    assert list(ct.text.is_null()) == [False, True, False, True]
    assert list(ct.text.notnull()) == [True, False, True, False]


def test_ctable_vlstring_nullable_getitem():
    data = [(0, "hello"), (1, None), (2, "world")]
    ct = blosc2.CTable(VLNullRow, new_data=data)
    assert ct.text[0] == "hello"
    assert ct.text[1] is None
    assert ct.text[2] == "world"


def test_ctable_vlstring_iter_chunks_not_supported():
    ct = blosc2.CTable(VLRow, new_data=ROWS)
    with pytest.raises(TypeError, match="varlen scalar"):
        list(ct.text.iter_chunks())


def test_ctable_vlstring_dtype_is_none():
    ct = blosc2.CTable(VLRow, new_data=ROWS)
    assert ct.text.dtype is None


# ---------------------------------------------------------------------------
# CTable with deletions
# ---------------------------------------------------------------------------


def test_ctable_vlstring_with_deletions():
    ct = blosc2.CTable(VLRow, new_data=ROWS)
    ct.delete(1)  # delete "a" * 200
    assert len(ct) == 4
    texts = list(ct.text)
    assert "a" * 200 not in texts
    assert "hello world" in texts


def test_ctable_vlstring_compact():
    ct = blosc2.CTable(VLRow, new_data=ROWS)
    ct.delete([1, 3])
    ct.compact()
    assert len(ct) == 3
    texts = list(ct.text)
    assert texts == ["hello world", "", "short"]


# ---------------------------------------------------------------------------
# CTable.copy() with varlen-scalar (vlstring/vlbytes) columns
# ---------------------------------------------------------------------------


def test_ctable_vlstring_copy_dense_compact():
    ct = blosc2.CTable(VLRow, new_data=ROWS)
    copied = ct.copy(compact=True)
    assert len(copied) == len(ct)
    assert list(copied.text) == [r[1] for r in ROWS]
    assert list(copied.data) == [r[2] for r in ROWS]
    # Independent copy: mutating the source must not affect it.
    ct.text[0] = "mutated"
    assert copied.text[0] == ROWS[0][1]


def test_ctable_vlstring_copy_with_deletions_compact():
    ct = blosc2.CTable(VLRow, new_data=ROWS)
    ct.delete([1, 3])  # drop the non-dense live-position case
    copied = ct.copy(compact=True)
    expected = [r[1] for i, r in enumerate(ROWS) if i not in (1, 3)]
    assert len(copied) == len(expected)
    assert list(copied.text) == expected


def test_ctable_vlstring_copy_noncompact_preserves_tombstones():
    ct = blosc2.CTable(VLRow, new_data=ROWS)
    ct.delete([1, 3])
    copied = ct.copy(compact=False)
    n = len(copied._valid_rows)  # copy(compact=False) trims to the high watermark
    assert list(copied._valid_rows[:]) == list(ct._valid_rows[:n])
    # Index the raw backing array directly: the row-bound column accessor
    # (copied.text) is limited to live-row count, not physical capacity.
    raw_text = copied._cols["text"]
    live_texts = [raw_text[i] for i in range(n) if copied._valid_rows[i]]
    expected = [r[1] for i, r in enumerate(ROWS) if i not in (1, 3)]
    assert live_texts == expected


# ---------------------------------------------------------------------------
# CTable save / load / open (persistence)
# ---------------------------------------------------------------------------


def test_ctable_vlstring_save_load(tmp_path):
    urlpath = str(tmp_path / "vl_test.b2d")
    ct = blosc2.CTable(VLRow, new_data=ROWS, urlpath=urlpath, mode="w")
    ct.close()

    ct2 = blosc2.CTable.open(urlpath)
    assert len(ct2) == len(ROWS)
    for i, (_exp_id, exp_text, exp_data) in enumerate(ROWS):
        assert ct2.text[i] == exp_text
        assert ct2.data[i] == exp_data
    ct2.close()


def test_ctable_vlstring_backend_role_metadata(tmp_path):
    urlpath = str(tmp_path / "vl_meta.b2d")
    ct = blosc2.CTable(VLRow, new_data=ROWS[:2], urlpath=urlpath, mode="w")
    ct.close()

    backend = blosc2.open(str(tmp_path / "vl_meta.b2d" / "_cols" / "text.b2b"), mode="r")
    assert backend.schunk.meta["ctable_varlen_scalar"] == {
        "version": 1,
        "py_type": "str",
        "nullable": False,
        "batch_rows": 2048,
    }


def test_ctable_constructor_reopens_vlstring_persistent_table(tmp_path):
    urlpath = str(tmp_path / "vl_ctor_reopen.b2d")
    ct = blosc2.CTable(VLRow, new_data=ROWS[:2], urlpath=urlpath, mode="w")
    ct.close()

    reopened = blosc2.CTable(VLRow, urlpath=urlpath, mode="a")
    assert reopened.text[1] == ROWS[1][1]
    reopened.append((42, "ctor append", b"ctor bytes"))
    assert reopened.text[2] == "ctor append"
    reopened.close()


def test_ctable_vlstring_save_reload_b2z(tmp_path):
    urlpath = str(tmp_path / "vl_test.b2z")
    ct = blosc2.CTable(VLRow, new_data=ROWS, urlpath=urlpath, mode="w")
    ct.close()

    ct2 = blosc2.CTable.open(urlpath)
    assert len(ct2) == len(ROWS)
    for i, (_, exp_text, _) in enumerate(ROWS):
        assert ct2.text[i] == exp_text
    ct2.close()


def test_ctable_vlstring_load_into_memory(tmp_path):
    urlpath = str(tmp_path / "vl_load.b2d")
    ct = blosc2.CTable(VLRow, new_data=ROWS, urlpath=urlpath, mode="w")
    ct.close()

    ct_mem = blosc2.CTable.load(urlpath)
    assert len(ct_mem) == len(ROWS)
    assert list(ct_mem.text) == [r[1] for r in ROWS]

    # In-memory table is writable
    ct_mem.text[0] = "mutated"
    assert ct_mem.text[0] == "mutated"


def test_ctable_vlstring_save_from_memory(tmp_path):
    urlpath = str(tmp_path / "vl_save.b2d")
    ct = blosc2.CTable(VLRow, new_data=ROWS)  # in-memory
    ct.save(urlpath)

    ct2 = blosc2.CTable.open(urlpath)
    assert list(ct2.text) == [r[1] for r in ROWS]
    ct2.close()


# ---------------------------------------------------------------------------
# CTable append + extend after open (writable mode)
# ---------------------------------------------------------------------------


def test_ctable_vlstring_append_to_persistent(tmp_path):
    urlpath = str(tmp_path / "vl_append.b2d")
    ct = blosc2.CTable(VLRow, new_data=ROWS[:3], urlpath=urlpath, mode="w")
    ct.close()

    ct2 = blosc2.CTable.open(urlpath, mode="a")
    ct2.append((99, "appended", b"appended_bytes"))
    ct2.close()

    ct3 = blosc2.CTable.open(urlpath)
    assert len(ct3) == 4
    assert ct3.text[3] == "appended"
    ct3.close()


# ---------------------------------------------------------------------------
# Arrow export
# ---------------------------------------------------------------------------


def test_ctable_vlstring_arrow_export():
    pa = pytest.importorskip("pyarrow")
    ct = blosc2.CTable(VLRow, new_data=ROWS)
    table = ct.to_arrow()
    assert pa.types.is_string(table.schema.field("text").type)
    assert pa.types.is_large_binary(table.schema.field("data").type)

    texts = table.column("text").to_pylist()
    assert texts == [r[1] for r in ROWS]

    datas = table.column("data").to_pylist()
    assert datas == [r[2] for r in ROWS]


def test_ctable_vlstring_nullable_arrow_export():
    pa = pytest.importorskip("pyarrow")
    data = [(0, "hello"), (1, None), (2, "world")]
    ct = blosc2.CTable(VLNullRow, new_data=data)
    table = ct.to_arrow()
    texts = table.column("text").to_pylist()
    assert texts == ["hello", None, "world"]
    assert table.column("text").null_count == 1


# ---------------------------------------------------------------------------
# Sort / index guards
# ---------------------------------------------------------------------------


def test_ctable_vlstring_sort_raises():
    ct = blosc2.CTable(VLRow, new_data=ROWS)
    gen = ct.iter_sorted("text")
    with pytest.raises(TypeError, match="varlen scalar"):
        next(gen)


def test_ctable_vlstring_lazy_expression_raises():
    ct = blosc2.CTable(VLRow, new_data=ROWS)
    with pytest.raises(NotImplementedError, match="vlstring/vlbytes"):
        ct.where('text == "hello world"')
    with pytest.raises(NotImplementedError, match="vlstring/vlbytes"):
        _ = ct.text == "hello world"


def test_ctable_vlstring_build_index_raises(tmp_path):
    urlpath = str(tmp_path / "vl_idx.b2d")
    ct = blosc2.CTable(VLRow, new_data=ROWS, urlpath=urlpath, mode="w")
    with pytest.raises(NotImplementedError, match="vlstring"):
        ct.create_index("text")
    ct.close()


# ---------------------------------------------------------------------------
# add_column with vlstring
# ---------------------------------------------------------------------------


@dataclass
class SimpleRow:
    id: int = blosc2.field(blosc2.int64())


def test_ctable_add_vlstring_column():
    ct = blosc2.CTable(SimpleRow, new_data=[(i,) for i in range(5)])
    ct.add_column("label", blosc2.field(blosc2.vlstring(), default="unknown"))
    assert "label" in ct.col_names
    assert ct.label[0] == "unknown"
    assert ct.label[4] == "unknown"
    assert ct.label.is_varlen_scalar


# ---------------------------------------------------------------------------
# Schema serialization round-trip (schema_to_dict / schema_from_dict)
# ---------------------------------------------------------------------------


def test_ctable_schema_dict_round_trip():
    from blosc2.schema_compiler import schema_from_dict, schema_to_dict

    ct = blosc2.CTable(VLRow, new_data=ROWS)
    d = schema_to_dict(ct._schema)

    kinds = {c["name"]: c["kind"] for c in d["columns"]}
    assert kinds["text"] == "vlstring"
    assert kinds["data"] == "vlbytes"

    restored = schema_from_dict(d)
    assert {c.name: type(c.spec).__name__ for c in restored.columns} == {
        "id": "int64",
        "text": "VLStringSpec",
        "data": "VLBytesSpec",
    }


# ---------------------------------------------------------------------------
# Display / info sanity
# ---------------------------------------------------------------------------


def test_ctable_vlstring_str_display():
    ct = blosc2.CTable(VLRow, new_data=ROWS)
    previous = blosc2.get_printoptions()
    try:
        s = str(ct)
        assert "vlstring" not in s
        assert "vlbytes" not in s
        assert "hello world" in s

        blosc2.set_printoptions(fancy=True)
        s = str(ct)
        assert "vlstring" in s
        assert "vlbytes" in s
        assert "hello world" in s
    finally:
        blosc2.set_printoptions(
            display_index=previous["display_index"],
            display_rows=previous["display_rows"],
            display_precision=previous["display_precision"],
            fancy=previous["fancy"],
        )


def test_ctable_vlstring_repr():
    ct = blosc2.CTable(VLRow, new_data=ROWS)
    r = repr(ct)
    # repr is now the tabular view (same as str); a small table shows no footer.
    assert r == str(ct)
    assert "id" in r.splitlines()[0]  # column header present
