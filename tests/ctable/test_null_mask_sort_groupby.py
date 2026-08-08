#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""Sort, group_by and query paths over mask-storage columns (Phase 7).

Every path here had the same shape of bug, and it is worth stating once: a
mask column's null rows hold the column's *fill*, and a fill is an ordinary
value to anything that only looks at the values.  So

* ``sort_by`` sorted nulls by their fill — first for an ascending int column
  (``0``), first for a string one (``""``) — where the nulls-last contract and
  every sentinel column put them last;
* ``group_by`` merged the nulls into the genuine ``0`` / ``""`` group, and
  reduced over the fill instead of skipping it;
* ``where("a < 10")`` matched every null, because ``0 < 10``;
* a ``FULL`` index on a float column returned every null for ``f > 0.5``,
  because the NaN fill sorts into the ordered range the index hands back.

The tests are written as a differential oracle against sentinel storage: the
same logical data both ways, asserted to give the same answer.  That is the
strongest form available, because the sentinel path has been correct since
Phase 1 and is independently tested.  Where the two are *supposed* to differ,
the divergence is asserted directly instead.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from utf8_compat import needs_utf8, utf8_spec

import blosc2


def annotation_for(spec):
    if isinstance(spec, (blosc2.schema.NDArraySpec, blosc2.schema.timestamp)):
        return object
    return spec.python_type


def table(rows, capacity=64, urlpath=None, **cols):
    Row = dataclasses.make_dataclass(
        "SortRow", [(n, annotation_for(s), blosc2.field(s)) for n, s in cols.items()]
    )
    kwargs = {"urlpath": str(urlpath), "mode": "w"} if urlpath is not None else {}
    t = blosc2.CTable(Row, expected_size=max(capacity, len(rows)), **kwargs)
    if rows:
        t.extend(rows)
    return t


def one_col(values, spec, capacity=64, urlpath=None):
    return table([(v,) for v in values], capacity=capacity, urlpath=urlpath, a=spec)


#: The V1 kinds a sort key can be, each with a mask spec, a sentinel spec, and
#: the value the sentinel path has to write where the mask path writes ``None``.
KEY_KINDS = [
    ("int64", blosc2.int64, {}, -(2**62)),
    ("float64", blosc2.float64, {}, np.nan),
    ("string", blosc2.string, {"max_length": 4}, "\x7f\x7f"),
    pytest.param("utf8", blosc2.utf8, {}, "__BLOSC2_NULL__", marks=needs_utf8),
]


def pair(values, factory, kw, sentinel):
    """The same logical *values* as a mask table and a sentinel table."""
    mask = one_col(values, factory(null_storage="mask", **kw))
    raw = [sentinel if v is None else v for v in values]
    sent = one_col(raw, factory(nullable=True, null_value=sentinel, **kw))
    return mask, sent


def as_list(col):
    """A column's live values with nulls spelled ``None``."""
    null = col.is_null()
    return [None if null[i] else _scalar(col[i]) for i in range(len(col))]


def _scalar(value):
    return value.item() if hasattr(value, "item") else value


# ---------------------------------------------------------------------------
# sort_by: nulls last, in both directions, for every kind
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("label", "factory", "kw", "sentinel"), KEY_KINDS)
@pytest.mark.parametrize("ascending", [True, False])
def test_sort_by_puts_mask_nulls_last(label, factory, kw, sentinel, ascending):
    """The contract sentinel columns already keep: nulls last, either direction.

    Before this phase the null-indicator lexsort key was only built for a
    sentinel, so a mask column sorted its nulls by the fill — ``0`` and ``""``
    both sort *first* ascending, which silently reversed the contract.
    """
    values = ["e", None, "a", "z", None, "b"] if label in ("string", "utf8") else [5, None, 1, 9, None, 2]
    mask, sent = pair(values, factory, kw, sentinel)

    got = as_list(mask.sort_by("a", ascending=ascending)["a"])
    n_nulls = sum(v is None for v in values)
    assert got[-n_nulls:] == [None] * n_nulls
    assert got[:-n_nulls] == sorted((v for v in values if v is not None), reverse=not ascending)
    # And the same answer the sentinel path gives, null spelling aside.
    assert got == as_list(sent.sort_by("a", ascending=ascending)["a"])


def test_sort_by_null_free_mask_column_needs_no_indicator_key():
    """No sidecar, no nulls, no extra key — decision 9 all the way through."""
    t = one_col([3, 1, 2], blosc2.int64(null_storage="mask"))
    assert t._null_mask("a") is None
    assert as_list(t.sort_by("a")["a"]) == [1, 2, 3]


def test_sort_by_multi_key_orders_nulls_last_per_key():
    t = table(
        [(1, "b"), (1, None), (None, "a"), (1, "a"), (None, None)],
        a=blosc2.int64(null_storage="mask"),
        s=blosc2.string(max_length=2, null_storage="mask"),
    )
    st = t.sort_by(["a", "s"])
    assert as_list(st["a"]) == [1, 1, 1, None, None]
    assert as_list(st["s"]) == ["a", "b", None, "a", None]


def test_sort_by_inplace_keeps_values_with_their_nulls():
    t = one_col([5, None, 1], blosc2.int64(null_storage="mask"))
    t.sort_by("a", inplace=True)
    assert as_list(t["a"]) == [1, 5, None]


def test_sort_by_view_orders_nulls_last():
    t = one_col([5, None, 1], blosc2.int64(null_storage="mask"))
    assert as_list(t.sort_by("a", view=True)["a"]) == [1, 5, None]


@pytest.mark.parametrize(
    ("spec", "values", "expected"),
    [
        (blosc2.int8(null_storage="mask"), [3, None, -128, 7], [7, 3, -128, None]),
        (blosc2.uint8(null_storage="mask"), [3, None, 255, 0], [255, 3, 0, None]),
        (blosc2.bool(null_storage="mask"), [True, None, False], [True, False, None]),
        (blosc2.bytes(max_length=2, null_storage="mask"), [b"b", None, b""], [b"b", b"", None]),
    ],
)
def test_descending_sort_over_a_full_range_mask_column(spec, values, expected):
    """Two storage-independent sort bugs that only mask storage can reach.

    The descending value key is built by negating the values, and that broke on
    the two dtypes a nullable column could not previously *be*: a ``bool``
    column has no unary minus (a nullable bool used to be physically ``uint8``),
    and a narrow signed dtype wraps on its own minimum -- ``-(-128) == -128`` in
    int8 -- so that row sorted as if it were the largest.  A sentinel had to
    reserve int8's ``-128``, so no nullable int8 column could hold it.
    """
    assert as_list(one_col(values, spec).sort_by("a", ascending=False)["a"]) == expected


def test_descending_sort_of_a_plain_bool_column():
    """The same fix, stated for the case that was broken with no nulls at all."""
    t = one_col([True, False, True], blosc2.bool())
    assert as_list(t.sort_by("a", ascending=False)["a"]) == [True, True, False]


# ---------------------------------------------------------------------------
# The FULL-index sort path reads the sidecar, not the whole column
# ---------------------------------------------------------------------------


def indexed_pair(values, factory, kw, sentinel, tmp_path):
    mask = one_col(values, factory(null_storage="mask", **kw), urlpath=tmp_path / "m.b2t")
    raw = [sentinel if v is None else v for v in values]
    sent = one_col(raw, factory(nullable=True, null_value=sentinel, **kw), urlpath=tmp_path / "s.b2t")
    for t in (mask, sent):
        t.create_index("a", kind="full")
    return mask, sent


@pytest.mark.parametrize(("label", "factory", "kw", "sentinel"), KEY_KINDS)
@pytest.mark.parametrize("ascending", [True, False])
def test_full_index_sort_matches_the_lexsort(label, factory, kw, sentinel, ascending, tmp_path):
    """A FULL index sorts by stored value, so nulls have to be repartitioned.

    Which is not new — the sentinel path already did it — but a mask column had
    no branch, so its nulls came back wherever the fill sorted.
    """
    if label in ("string", "utf8"):
        pool = ["alfa", "beta", "", "zeta", "gam"]  # within string's max_length=4
    else:
        pool = [5, 1, 9, 2, 7]
    values = [None if i % 7 == 0 else pool[i % len(pool)] for i in range(200)]
    mask, sent = indexed_pair(values, factory, kw, sentinel, tmp_path)

    got = as_list(mask.sort_by("a", ascending=ascending)["a"])
    assert got == as_list(sent.sort_by("a", ascending=ascending)["a"])
    n_nulls = sum(v is None for v in values)
    assert got[-n_nulls:] == [None] * n_nulls


def whole_column_reads(t, name, monkeypatch, body):
    """Run *body*, returning which arrays *t* read whole (``arr[:]``)."""
    seen = []
    real_getitem = blosc2.NDArray.__getitem__
    values_arr = t._cols[name]
    sidecar = t._null_mask(name)

    def counting(self, key):
        if isinstance(key, slice) and key == slice(None):
            if self is values_arr:
                seen.append("values")
            elif sidecar is not None and self is sidecar:
                seen.append("sidecar")
        return real_getitem(self, key)

    monkeypatch.setattr(blosc2.NDArray, "__getitem__", counting)
    body()
    monkeypatch.undo()
    return seen


def test_full_index_sort_reads_the_sidecar_instead_of_the_values(tmp_path, monkeypatch):
    """The highest value-per-line change in the phase, measured as such.

    Locating the null rows in a FULL index's permutation means reading a whole
    column: the sentinel path reads the *values* to compare against the
    sentinel, 8 bytes a row for ``int64`` and 64 for a ``U16`` string.  A mask
    column reads its sidecar instead — one byte a row, and bool NDArrays
    compress to almost nothing.
    """
    values = [None if i % 5 == 0 else i for i in range(200)]
    mask = one_col(values, blosc2.int64(null_storage="mask"), urlpath=tmp_path / "m.b2t")
    sent = one_col(
        [-1 if v is None else v for v in values],
        blosc2.int64(nullable=True, null_value=-1),
        urlpath=tmp_path / "s.b2t",
    )
    for t in (mask, sent):
        t.create_index("a", kind="full")

    assert whole_column_reads(mask, "a", monkeypatch, lambda: mask.sort_by("a")) == ["sidecar"]
    assert whole_column_reads(sent, "a", monkeypatch, lambda: sent.sort_by("a")) == ["values"]


def test_sorted_slice_falls_back_when_the_column_has_nulls(tmp_path):
    """The window read cannot locate a mask column's null block.

    It finds the null block by bisecting the *sorted values* sidecar for the
    null's stored value; under mask storage that value is the fill, which
    genuine rows share, and the validity sidecar is indexed by physical
    position rather than sorted position so the window cannot consult it.  So
    it declines, and the full sorted view — which is mask-aware — answers.
    """
    values = [None if i % 9 == 0 else (i % 17) for i in range(300)]
    t = one_col(values, blosc2.int64(null_storage="mask"), urlpath=tmp_path / "m.b2t")
    t.create_index("a", kind="full")

    assert t._sorted_slice_positions("a", True, slice(0, 5)) is None
    for key in (slice(0, 5), slice(-5, None), slice(280, 300)):
        for ascending in (True, False):
            window = as_list(t.sorted_slice("a", key, ascending=ascending)["a"])
            full = as_list(t.sort_by("a", ascending=ascending, view=True)[key]["a"])
            assert window == full


def test_sorted_slice_keeps_its_window_read_when_there_are_no_nulls(tmp_path):
    """A mask column with no sidecar has no null block to locate."""
    t = one_col(list(range(300)), blosc2.int64(null_storage="mask"), urlpath=tmp_path / "m.b2t")
    t.create_index("a", kind="full")
    assert t._sorted_slice_positions("a", True, slice(0, 5)) is not None


# ---------------------------------------------------------------------------
# utf8 rank index: the fill must not factorize as an ordinary value
# ---------------------------------------------------------------------------


@needs_utf8
def test_utf8_rank_index_separates_nulls_from_genuine_empty_strings(tmp_path):
    """The landmine this phase was warned about, and it is a real one.

    A utf8 rank index factorizes the column; under mask storage the ``""`` fill
    is just another vocabulary entry, and it factorizes to **rank 0** — the
    smallest — so nulls both sorted first and answered ``a == ''`` alongside
    the rows that really are empty.
    """
    values = ["b", None, "", "a", None, ""]
    t = one_col(values, blosc2.utf8(null_storage="mask"), urlpath=tmp_path / "m.b2t")
    t.create_index("a", kind="full")

    assert len(t.where("a == ''")) == 2  # the two real empty strings, not the nulls
    assert as_list(t.where("a == ''")["a"]) == ["", ""]
    assert as_list(t.sort_by("a")["a"]) == ["", "", "a", "b", None, None]


@needs_utf8
def test_utf8_rank_arrays_stamps_nulls_with_the_null_rank():
    """Directly, since this is where the recoding happens."""
    from blosc2.ctable_indexing import _utf8_rank_arrays

    t = one_col(["b", None, "", "a"], blosc2.utf8(null_storage="mask"))
    col = t._cols["a"]
    valid = t._null_mask("a")
    ranks, meta, vocab = _utf8_rank_arrays(col, 4, None, valid=valid)
    assert meta["null_aware"] is True
    assert list(vocab) == ["", "a", "b"]
    assert int(ranks[1]) == meta["null_rank"] == 3
    assert int(ranks[2]) == 0  # the genuine "" keeps rank 0


@needs_utf8
def test_a_pre_mask_utf8_rank_index_is_treated_as_stale(tmp_path):
    """An index built before nulls got their own rank cannot be trusted.

    Neither of the O(1) staleness signals can catch it — the column's row count
    and byte size are exactly what they were — so the absence of ``null_aware``
    from the meta is what marks it, and only for a column that has a sidecar.
    """
    t = one_col(["b", None, ""], blosc2.utf8(null_storage="mask"), urlpath=tmp_path / "m.b2t")
    t.create_index("a", kind="full")
    meta = t._get_index_catalog()["a"]["full"]["utf8_rank"]
    assert t._utf8_rank_index_stale("a", meta) is False

    del meta["null_aware"]
    assert t._utf8_rank_index_stale("a", meta) is True


@needs_utf8
def test_a_sentinel_utf8_rank_index_stays_fresh_without_the_flag(tmp_path):
    """The staleness rule must not fire for sentinel columns.

    Their nulls have always carried ``null_rank``, so an index with no
    ``null_aware`` flag is merely an older one, not a wrong one — invalidating
    it would rebuild every stored utf8 index in the wild for nothing.
    """
    t = one_col(
        ["b", "__BLOSC2_NULL__", ""],
        blosc2.utf8(nullable=True),
        urlpath=tmp_path / "s.b2t",
    )
    t.create_index("a", kind="full")
    meta = dict(t._get_index_catalog()["a"]["full"]["utf8_rank"])
    meta.pop("null_aware", None)
    assert t._utf8_rank_index_stale("a", meta) is False


# ---------------------------------------------------------------------------
# where(): the string form, indexed and not
# ---------------------------------------------------------------------------

WHERE_CASES = [
    "a < 500",
    "a > 500",
    "a != 7",
    "a == 3",
    "~(a > 500)",
]


def numeric_pair(tmp_path, indexed):
    rng = np.random.default_rng(5)
    values = [None if i % 31 == 0 else int(rng.integers(0, 1000)) for i in range(2000)]
    mask = one_col(values, blosc2.int64(null_storage="mask"), urlpath=tmp_path / f"m{indexed}.b2t")
    sent = one_col(
        [-1 if v is None else v for v in values],
        blosc2.int64(nullable=True, null_value=-1),
        urlpath=tmp_path / f"s{indexed}.b2t",
    )
    if indexed:
        mask.create_index("a", kind="full")
        sent.create_index("a", kind="full")
    return mask, sent


@pytest.mark.parametrize("query", WHERE_CASES)
@pytest.mark.parametrize("indexed", [False, True])
def test_string_predicates_reject_mask_nulls(query, indexed, tmp_path):
    """``a < 10`` matched every null before this: the int fill is ``0``.

    The operator form was already correct (``_null_aware_compare`` collapses
    null to False at the leaf); it was only the string form that compared the
    stored fill, exactly as it used to compare the stored sentinel before
    Phase 1.
    """
    mask, sent = numeric_pair(tmp_path, indexed)
    got = mask.where(query)
    assert got["a"].null_count() == 0
    assert len(got) == len(sent.where(query))


@pytest.mark.parametrize("indexed", [False, True])
def test_or_over_a_mask_column_keeps_the_other_branch(indexed, tmp_path):
    """The per-leaf guard is what makes OR right; a global filter would not be.

    A row null in ``a`` but matching ``b`` must survive ``(a > x) | (b < y)``.
    """
    rng = np.random.default_rng(6)
    n = 2000
    a = [None if i % 31 == 0 else int(rng.integers(0, 1000)) for i in range(n)]
    b = [int(rng.integers(0, 1000)) for _ in range(n)]
    t = table(
        list(zip(a, b, strict=True)),
        capacity=n,
        urlpath=tmp_path / f"or{indexed}.b2t",
        a=blosc2.int64(null_storage="mask"),
        b=blosc2.int64(),
    )
    if indexed:
        t.create_index("a", kind="full")

    expected = sum(((av is not None and av > 800) or bv < 200) for av, bv in zip(a, b, strict=True))
    assert len(t.where("(a > 800) | (b < 200)")) == expected


def test_a_nan_fill_does_not_come_back_from_an_ordered_index(tmp_path):
    """The index failure this phase found, and the one masks make worse.

    An ordered index answers ``f > 0.5`` by taking a *range of the sorted
    column* — it never evaluates the predicate — and NaN sorts to the end of
    that range.  A mask float column's fill is NaN, so every null row was
    returned.  Making the expression null-aware cannot fix this; the
    positions have to be filtered, which is what a mask column now joins
    ``nullable_indexed`` to get.
    """
    rng = np.random.default_rng(9)
    values = [None if i % 23 == 0 else float(rng.random()) for i in range(2000)]
    t = one_col(values, blosc2.float64(null_storage="mask"), urlpath=tmp_path / "f.b2t")
    unindexed = len(t.where("a > 0.5"))
    t.create_index("a", kind="full")
    indexed = t.where("a > 0.5")
    assert indexed["a"].null_count() == 0
    assert len(indexed) == unindexed


def test_null_free_mask_column_stays_out_of_the_null_bookkeeping(tmp_path):
    """No sidecar means no guard operand and no post-filter to pay for."""
    t = one_col(list(range(2000)), blosc2.int64(null_storage="mask"), urlpath=tmp_path / "n.b2t")
    t.create_index("a", kind="full")
    rewritten, operands = t._rewrite_null_predicates("a < 500", {"a": t._cols["a"]})
    assert rewritten == "a < 500"
    assert list(operands) == ["a"]
    assert len(t.where("a < 500")) == 500


def test_a_guard_is_only_emitted_where_the_fill_could_match():
    """Keeping needless guards out is what keeps a query on its index.

    The fill takes the sentinel's place in the can-match test, and it is the
    right stand-in: it is what a null row actually holds.  ``0`` cannot satisfy
    ``a > 10``, so that leaf is left alone.
    """
    t = one_col([1, None, 20], blosc2.int64(null_storage="mask"))
    operands = {"a": t._cols["a"]}
    assert t._rewrite_null_predicates("a > 10", operands)[0] == "a > 10"
    guarded, extended = t._rewrite_null_predicates("a < 10", operands)
    assert guarded != "a < 10"
    assert any(name.startswith("__nv") for name in extended)


# ---------------------------------------------------------------------------
# utf8 comparisons and the span driver
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query",
    ["a < 'b'", "a > 'b'", "a != 'a'", "a == ''", "startswith(a, 'a')"],
)
@needs_utf8
def test_utf8_predicates_agree_between_storages(query):
    values = ["e", None, "a", "z", "", "b"]
    mask, sent = pair(values, blosc2.utf8, {}, "__BLOSC2_NULL__")
    assert as_list(mask.where(query)["a"]) == as_list(sent.where(query)["a"])


@needs_utf8
def test_utf8_span_driver_excludes_mask_nulls():
    """The span driver materializes nulls to ``""`` and re-applies nullity.

    It found them by comparing against the sentinel, which a mask column does
    not have — so a boolean result over the ``""`` fill came back True for
    nulls whenever the fill satisfied the expression.
    """
    t = one_col(["ax", None, "bx"], blosc2.utf8(null_storage="mask"))
    assert as_list(t.where("startswith(a, '')")["a"]) == ["ax", "bx"]


@needs_utf8
def test_utf8_column_vs_column_comparison_excludes_nulls():
    t = table(
        [("a", "a"), (None, "a"), ("b", None), ("c", "c")],
        a=blosc2.utf8(null_storage="mask"),
        b=blosc2.utf8(null_storage="mask"),
    )
    assert as_list(t.where(t["a"] == t["b"])["a"]) == ["a", "c"]


# ---------------------------------------------------------------------------
# group_by: as a key and as a value
# ---------------------------------------------------------------------------

AGGS = {
    "cnt": ("v", "count"),
    "sm": ("v", "sum"),
    "mn": ("v", "min"),
    "mx": ("v", "max"),
    "av": ("v", "mean"),
}


def grouped(t, keys, dropna):
    """``{key tuple -> {agg -> value}}``, nulls spelled ``None`` throughout."""
    g = t.group_by(keys, dropna=dropna, sort=True).agg(**AGGS)
    # Each column is read once, not once per row: as_list is O(rows), so
    # calling it inside the loop below made this O(rows^2 * columns) and cost
    # more than every other test in the file put together.
    cols = {name: as_list(g[name]) for name in (*keys, *AGGS)}
    out = {}
    for i in range(len(g)):
        key = tuple(cols[name][i] for name in keys)
        row = {}
        for name in AGGS:
            value = cols[name][i]
            if isinstance(value, float) and np.isnan(value):
                value = None  # a non-nullable float output spells "missing" NaN
            row[name] = value
        out[key] = row
    return out


def groupby_pair():
    rng = np.random.default_rng(11)
    n = 400
    k = [None if i % 13 == 0 else int(rng.integers(0, 4)) for i in range(n)]
    s = [None if i % 17 == 0 else ["aa", "bb", "cc"][int(rng.integers(0, 3))] for i in range(n)]
    v = [None if i % 7 == 0 else float(rng.integers(0, 100)) for i in range(n)]
    mask = table(
        list(zip(k, s, v, strict=True)),
        capacity=n,
        k=blosc2.int64(null_storage="mask"),
        s=blosc2.string(max_length=2, null_storage="mask"),
        v=blosc2.float64(null_storage="mask"),
    )
    sent = table(
        [
            (-1 if a is None else a, "ZZ" if b is None else b, np.nan if c is None else c)
            for a, b, c in zip(k, s, v, strict=True)
        ],
        capacity=n,
        k=blosc2.int64(nullable=True, null_value=-1),
        s=blosc2.string(max_length=2, nullable=True, null_value="ZZ"),
        v=blosc2.float64(nullable=True, null_value=np.nan),
    )
    return mask, sent


@pytest.mark.parametrize("keys", [["k"], ["s"], ["k", "s"]])
@pytest.mark.parametrize("dropna", [True, False])
def test_group_by_agrees_with_sentinel_storage(keys, dropna, tmp_path):
    """The differential oracle for group_by, keys and values together.

    A mask key column used to merge its nulls into the group of whatever its
    fill was — ``0`` for the int key, ``""`` for the string one — and a mask
    value column reduced over the fill, turning any group containing a null
    into ``NaN`` for ``sum``/``mean``.
    """
    mask, sent = groupby_pair()
    got = grouped(mask, keys, dropna)
    expected = {
        tuple(None if part in (-1, "ZZ") else part for part in key): row
        for key, row in grouped(sent, keys, dropna).items()
    }
    assert got == expected


def test_group_by_a_mask_key_keeps_nulls_out_of_the_fill_group():
    """Stated on its own, because it is the failure that is easiest to miss.

    ``0`` is a perfectly ordinary key, so a null row landing in its group
    inflates a real answer rather than producing an obviously wrong one.
    """
    t = table(
        [(0, 1.0), (None, 2.0), (0, 4.0)],
        k=blosc2.int64(null_storage="mask"),
        v=blosc2.float64(null_storage="mask"),
    )
    g = t.group_by(["k"], dropna=True).agg(total=("v", "sum"))
    assert as_list(g["k"]) == [0]
    assert as_list(g["total"]) == [5.0]


def test_group_by_a_mask_key_writes_its_null_group_as_a_null():
    """With ``dropna=False`` the null group's key is a *real* null.

    A sentinel column can only offer its sentinel here, which is why
    ``group_by(dropna=False)`` over one returns a group keyed ``-1``.  Mask
    storage can say what it means.
    """
    t = table(
        [(1, 1.0), (None, 2.0), (None, 4.0)],
        k=blosc2.int64(null_storage="mask"),
        v=blosc2.float64(null_storage="mask"),
    )
    g = t.group_by(["k"], dropna=False, sort=True).agg(total=("v", "sum"))
    assert as_list(g["k"]) == [None, 1]
    assert as_list(g["total"]) == [6.0, 1.0]


def test_group_by_a_mask_value_skips_nulls_not_fills():
    t = table(
        [("x", 3), ("x", None), ("y", None)],
        s=blosc2.string(max_length=1),
        v=blosc2.int64(null_storage="mask"),
    )
    g = t.group_by(["s"], sort=True).agg(n=("v", "count"), lo=("v", "min"))
    assert as_list(g["n"]) == [1, 0]
    # min over a group with no non-null value is a null, not the 0 fill.
    assert as_list(g["lo"]) == [3, None]


def test_group_by_a_mask_float_value_treats_nan_as_a_value():
    """Decision 6, in the one place it is observable in an aggregate.

    A sentinel float column's NaN *is* its null, so ``sum`` skips it.  A mask
    column's NaN is data, so it propagates — which is Arrow's answer and
    NumPy's.
    """
    t = table(
        [("x", 1.0), ("x", float("nan")), ("x", None)],
        s=blosc2.string(max_length=1),
        v=blosc2.float64(null_storage="mask"),
    )
    g = t.group_by(["s"]).agg(total=("v", "sum"), n=("v", "count"))
    assert np.isnan(g["total"][0])
    assert as_list(g["n"]) == [2]  # the NaN counts; the null does not


def test_group_by_a_mask_float_key_groups_nan_with_the_nulls():
    """Keys keep NaN-as-missing, so ``dropna`` stays predictable.

    This is the one place a float mask column does *not* follow decision 6, and
    deliberately: the rule is about values, and a key that sometimes forms its
    own NaN group and sometimes does not would make ``dropna`` unusable.  It is
    also what the sentinel path does.
    """
    t = table(
        [(1.0, 1), (float("nan"), 2), (None, 4)],
        k=blosc2.float64(null_storage="mask"),
        v=blosc2.int64(),
    )
    assert len(t.group_by(["k"], dropna=True).agg(total=("v", "sum"))) == 1
    g = t.group_by(["k"], dropna=False, sort=True).agg(total=("v", "sum"))
    assert as_list(g["total"]) == [6, 1]


@needs_utf8
def test_group_by_utf8_mask_key_separates_nulls_from_empty_strings():
    t = table(
        [("", 1), (None, 2), ("", 4)],
        s=blosc2.utf8(null_storage="mask"),
        v=blosc2.int64(),
    )
    g = t.group_by(["s"], dropna=False, sort=True).agg(total=("v", "sum"))
    assert as_list(g["s"]) == [None, ""]
    assert as_list(g["total"]) == [2, 5]


@pytest.mark.parametrize(
    ("spec", "values"),
    [
        (blosc2.int8(null_storage="mask"), [3, None, -128, 3]),
        (blosc2.uint8(null_storage="mask"), [255, None, 0, 255]),
        (blosc2.bool(null_storage="mask"), [True, None, False, True]),
        (blosc2.bytes(max_length=2, null_storage="mask"), [b"aa", None, b"", b"aa"]),
        pytest.param(utf8_spec(null_storage="mask"), ["aa", None, "", "aa"], marks=needs_utf8),
    ],
)
def test_group_by_a_mask_key_of_every_v1_kind(spec, values):
    """Each kind's fill is a value some real row could hold, so each needs the recode."""
    t = table([(v, i) for i, v in enumerate(values)], k=spec, v=blosc2.int64())
    distinct = {v for v in values if v is not None}
    dropped = t.group_by(["k"], dropna=True, sort=True).agg(total=("v", "sum"))
    assert len(dropped) == len(distinct)
    assert None not in as_list(dropped["k"])
    kept = t.group_by(["k"], dropna=False, sort=True).agg(total=("v", "sum"))
    assert as_list(kept["k"]).count(None) == 1
    assert sum(as_list(kept["total"])) == sum(range(len(values)))


def test_group_by_min_over_a_bool_value_column():
    """A storage-independent bug the mask path is what routed into.

    ``min``/``max`` over a group seed an accumulator with the dtype's opposite
    identity, and ``bool`` had none -- ``np.full(n, None, dtype=bool)`` is
    ``False``, so a min accumulator could never rise above it and every all-True
    group reduced to ``False``.  Reachable with a plain non-nullable bool column
    on any generic-path aggregation; a nullable one used to be ``uint8``, whose
    identities are fine, which is why mask storage surfaced it.
    """
    t = table(
        [("a", True), ("a", True), ("b", False), ("b", True)],
        k=blosc2.string(max_length=1),
        v=blosc2.bool(),
    )
    g = t.group_by(["k"], sort=True).agg(lo=("v", "min"), hi=("v", "max"))
    assert as_list(g["lo"]) == [True, False]
    assert as_list(g["hi"]) == [True, True]


def fast_path_taken(t, keys, **aggs):
    """Whether ``group_by(keys).agg(**aggs)`` is served by a fast path."""
    gb = t.group_by(keys)
    specs = gb._normalize_aggs((), aggs)
    return gb._mask_null_columns(specs), gb._try_fast_paths(specs, False) is not None


def test_null_free_mask_columns_keep_every_groupby_fast_path():
    """Decision 9 again: the deoptimization is scoped to columns with nulls."""
    t = table(
        [(1, 2), (1, 3), (2, 4)],
        k=blosc2.int64(null_storage="mask"),
        v=blosc2.int64(null_storage="mask"),
    )
    assert fast_path_taken(t, ["k"], total=("v", "sum")) == ([], True)


def test_a_mask_value_column_keeps_the_dense_single_key_path():
    """It is plain NumPy and already asks ``_null_mask``; it just needs the sidecar.

    Only the Cython kernels have to give up, because a kernel is handed a
    ``skip_nan`` flag rather than a validity array.  Keeping this path matters:
    deferring a mask value column all the way to the generic hash-and-merge path
    cost ~4.5x on a 2M-row sum.
    """
    t = table(
        [(1, 2), (1, 3), (2, 4)],
        k=blosc2.int64(),
        v=blosc2.int64(null_storage="mask"),
    )
    t["v"][0] = None
    assert fast_path_taken(t, ["k"], total=("v", "sum")) == (["v"], True)


def test_a_mask_key_column_leaves_the_fast_paths():
    """Its recoded chunk is a ``_CodedKeyChunk``, not an array of dense ints."""
    t = table(
        [(1, 2), (1, 3), (2, 4)],
        k=blosc2.int64(null_storage="mask"),
        v=blosc2.int64(),
    )
    t["k"][0] = None
    assert fast_path_taken(t, ["k"], total=("v", "sum")) == (["k"], False)
