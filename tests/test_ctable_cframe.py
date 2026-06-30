"""Tests for CTable.to_cframe() / ctable_from_cframe()."""

from dataclasses import dataclass

import numpy as np

import blosc2
import blosc2.ctable as ct


def _check(label, got, want):
    assert got == want, f"{label}: {got!r} != {want!r}"


def test_scalar_string():
    @dataclass
    class R:
        x: int = blosc2.field(blosc2.int32())
        y: str = blosc2.field(blosc2.string(max_length=20))

    t = blosc2.CTable(R)
    for r in [(0, "n0"), (1, "n1"), (2, "n2")]:
        t.append(r)

    cf = t.to_cframe()
    rt = blosc2.ctable_from_cframe(cf)
    _check("nrows", len(rt), 3)
    _check("row0", tuple(rt[0]), (0, "n0"))
    _check("row2", tuple(rt[2]), (2, "n2"))
    _check("schema", rt.schema_dict(), t.schema_dict())

    # slice
    v = t.slice(1, 3)
    cfv = v.to_cframe()
    rtv = blosc2.ctable_from_cframe(cfv)
    _check("slice nrows", len(rtv), 2)
    _check("slice row0", tuple(rtv[0]), (1, "n1"))


def test_list_column():
    @dataclass
    class R:
        x: int = blosc2.field(blosc2.int32())
        tags: list[int] = blosc2.field(ct.ListSpec(ct.int32()))  # noqa: RUF009

    t = blosc2.CTable(R)
    for r in [(1, [1, 2, 3]), (2, [4]), (3, [])]:
        t.append(r)

    cf = t.to_cframe()
    rt = blosc2.ctable_from_cframe(cf)
    _check("nrows", len(rt), 3)
    _check("row0 tags", rt[0].tags, [1, 2, 3])
    _check("row2 tags", rt[2].tags, [])


def test_dictionary_column():
    @dataclass
    class R:
        x: int = blosc2.field(blosc2.int32())
        cat: str = blosc2.field(ct.DictionarySpec(value_type=blosc2.vlstring()))

    t = blosc2.CTable(R)
    for r in [(1, "a"), (2, "b"), (3, "a"), (4, "b")]:
        t.append(r)

    cf = t.to_cframe()
    rt = blosc2.ctable_from_cframe(cf)
    _check("nrows", len(rt), 4)
    _check("row0", rt[0].cat, "a")
    _check("row3", rt[3].cat, "b")


def test_vlstring():
    @dataclass
    class R:
        x: int = blosc2.field(blosc2.int32())
        label: str = blosc2.field(ct.VLStringSpec())

    t = blosc2.CTable(R)
    for r in [(1, "hi"), (2, "yo"), (3, "longer")]:
        t.append(r)

    cf = t.to_cframe()
    rt = blosc2.ctable_from_cframe(cf)
    _check("nrows", len(rt), 3)
    _check("vlstr row0", rt[0].label, "hi")
    _check("vlstr row2", rt[2].label, "longer")

    # slice must work (slice() materializes, bypassing the lazy-view copy() bug)
    v = t.slice(1, 3)
    cfv = v.to_cframe()
    rtv = blosc2.ctable_from_cframe(cfv)
    _check("vlstr slice nrows", len(rtv), 2)
    _check("vlstr slice row0", rtv[0].label, "yo")


def test_vlmeta():
    @dataclass
    class R:
        x: int = blosc2.field(blosc2.int32())

    t = blosc2.CTable(R)
    t.append((0,))
    t.vlmeta["author"] = "Alice"
    t.vlmeta["tags"] = [1, 2]

    cf = t.to_cframe()
    rt = blosc2.ctable_from_cframe(cf)
    _check("vlmeta author", rt.vlmeta["author"], "Alice")
    _check("vlmeta tags", rt.vlmeta["tags"], [1, 2])


def test_clean_failure():
    """ctable_from_cframe must raise cleanly on non-CTable cframes."""
    import pytest

    a = blosc2.asarray(np.arange(3))
    with pytest.raises(ValueError, match="Not an EmbedStore cframe"):
        blosc2.ctable_from_cframe(a.to_cframe())


def test_gc_independence():
    """Accessing a from_cframe() result must work after the cframe ref is dropped."""
    import gc

    @dataclass
    class R:
        x: int = blosc2.field(blosc2.int32())

    t = blosc2.CTable(R)
    t.append((42,))
    cf = t.to_cframe()
    rt = blosc2.ctable_from_cframe(cf)
    del cf
    gc.collect()
    _check("gc row0", rt[0].x, 42)


# ---------------------------------------------------------------------------
# Edge cases and larger coverage
# ---------------------------------------------------------------------------


def test_empty_table():
    """A table with no rows must serialize and deserialize correctly."""

    @dataclass
    class R:
        x: int = blosc2.field(blosc2.int32())

    t = blosc2.CTable(R)
    cf = t.to_cframe()
    rt = blosc2.ctable_from_cframe(cf)
    _check("nrows", len(rt), 0)
    _check("cols", rt.col_names, ["x"])


def test_single_row():
    @dataclass
    class R:
        x: int = blosc2.field(blosc2.int32())
        y: str = blosc2.field(blosc2.string(max_length=20))

    t = blosc2.CTable(R)
    t.append((99, "only"))
    cf = t.to_cframe()
    rt = blosc2.ctable_from_cframe(cf)
    _check("nrows", len(rt), 1)
    _check("row0", tuple(rt[0]), (99, "only"))


def test_large_table():
    """Ensure a table with many rows round-trips without bloating or losing data."""

    @dataclass
    class R:
        x: int = blosc2.field(blosc2.int32())

    t = blosc2.CTable(R)
    for i in range(1000):
        t.append((i,))
    cf = t.to_cframe()
    rt = blosc2.ctable_from_cframe(cf)
    _check("nrows", len(rt), 1000)
    # spot-check
    _check("row0", rt[0].x, 0)
    _check("row999", rt[999].x, 999)
    _check("col slice", rt["x"][:3].tolist(), [0, 1, 2])


def test_persistent_b2z_roundtrip():
    """A .b2z file opened from disk must serialize correctly via to_cframe."""
    import pathlib
    import tempfile

    @dataclass
    class R:
        x: int = blosc2.field(blosc2.int32())
        y: str = blosc2.field(blosc2.string(max_length=20))

    p = pathlib.Path(tempfile.mkdtemp()) / "t.b2z"
    t = blosc2.CTable(R, urlpath=str(p), mode="w", compact=True)
    for i in range(50):
        t.append((i, f"n{i}"))
    t.close()
    t = blosc2.open(p)
    cf = t.to_cframe()
    rt = blosc2.ctable_from_cframe(cf)
    _check("nrows", len(rt), 50)
    _check("row0", tuple(rt[0]), (0, "n0"))
    _check("row49", tuple(rt[49]), (49, "n49"))


def test_mixed_column_types():
    """A single table with scalar + list + dict columns must round-trip."""

    @dataclass
    class R:
        x: int = blosc2.field(blosc2.int32())
        tags: list[int] = blosc2.field(ct.ListSpec(ct.int32()))  # noqa: RUF009
        cat: str = blosc2.field(ct.DictionarySpec(value_type=blosc2.vlstring()))

    t = blosc2.CTable(R)
    for r in [(1, [10, 20], "alpha"), (2, [30], "beta")]:
        t.append(r)
    cf = t.to_cframe()
    rt = blosc2.ctable_from_cframe(cf)
    _check("nrows", len(rt), 2)
    _check("cols", rt.col_names, ["x", "tags", "cat"])
    _check("row0", (rt[0].x, rt[0].tags, rt[0].cat), (1, [10, 20], "alpha"))
    _check("row1", (rt[1].x, rt[1].tags, rt[1].cat), (2, [30], "beta"))


def test_where_view():
    """Lazy views (base is not None) must serialize via copy() for scalar columns."""

    @dataclass
    class R:
        x: int = blosc2.field(blosc2.int32())
        y: str = blosc2.field(blosc2.string(max_length=20))

    t = blosc2.CTable(R)
    for r in [(0, "n0"), (1, "n1"), (2, "n2"), (3, "n3")]:
        t.append(r)
    w = t.where(t["x"] > 1)
    _check("view nrows", len(w), 2)
    cf = w.to_cframe()
    rt = blosc2.ctable_from_cframe(cf)
    _check("view rt nrows", len(rt), 2)
    _check("view row0", tuple(rt[0]), (2, "n2"))
    _check("view row1", tuple(rt[1]), (3, "n3"))


def test_double_serialization():
    """A from_cframe result can be re-serialized (identity)."""

    @dataclass
    class R:
        x: int = blosc2.field(blosc2.int32())

    t = blosc2.CTable(R)
    for i in range(5):
        t.append((i,))
    cf1 = t.to_cframe()
    rt = blosc2.ctable_from_cframe(cf1)
    cf2 = rt.to_cframe()
    rt2 = blosc2.ctable_from_cframe(cf2)
    _check("double nrows", len(rt2), 5)
    _check("double row0", rt2[0].x, 0)


def test_schema_preservation():
    """The schema_dict must survive a round-trip unchanged."""

    @dataclass
    class R:
        a: int = blosc2.field(blosc2.int32())
        b: float = blosc2.field(blosc2.float64())
        c: str = blosc2.field(blosc2.string(max_length=10))

    t = blosc2.CTable(R)
    t.append((1, 2.5, "hi"))
    s_before = t.schema_dict()
    rt = blosc2.ctable_from_cframe(t.to_cframe())
    # only compare stable fields (not n_rows which changes)
    for col_b, col_a in zip(s_before["columns"], rt.schema_dict()["columns"], strict=False):
        _check(f"col {col_b['name']}", col_a, col_b)
