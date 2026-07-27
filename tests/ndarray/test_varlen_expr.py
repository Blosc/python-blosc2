#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""Tests for blosc2.compute_varlen(): Arrow varlen results from expressions."""

import numpy as np
import pytest

import blosc2

VALUES = [
    "Flash Cab",
    "",
    "Sun Taxi",
    "café",
    "日本語のテキスト",
    "emoji 🎉🚀",
    "Yellow Cab",
    "x" * 200,
]


@pytest.fixture
def operands():
    co = np.array(VALUES, dtype="<U200")
    pt = np.array(["Cash", "Credit", "Prcard", "Cash"] * 2, dtype="<U11")
    return blosc2.asarray(co), blosc2.asarray(pt)


def test_varlen_matches_the_fixed_width_result(operands):
    a, b = operands
    expr = "co=" + a + "|pay=" + blosc2.lower(b)
    fixed = expr.compute(strict_miniexpr=True)
    got = blosc2.compute_varlen(expr)
    assert isinstance(got, blosc2.Utf8Array)
    assert [str(v) for v in got] == [str(v) for v in fixed[:]]


def test_varlen_is_tighter_than_the_width_bound(operands):
    """The point of the layout: rows cost their own length, not the bound."""
    a, b = operands
    expr = blosc2.lower(a)
    got = blosc2.compute_varlen(expr)
    # lower() on <U200 reserves a 2x case-expansion bound at 4 bytes/codepoint.
    assert expr.dtype.itemsize == 200 * 2 * 4
    assert int(got.offsets[-1]) < expr.dtype.itemsize * len(VALUES)


def test_varlen_over_a_dsl_kernel(operands):
    a, b = operands

    @blosc2.dsl_kernel
    def label(company, ptype):
        pay = ptype.lower()
        c = company.lower()
        if " cab" in c:
            return "cab|" + c.removesuffix(" cab") + "|" + pay
        return "other|" + c + "|" + pay

    udf = blosc2.lazyudf(label, (a, b))
    fixed = udf.compute()
    got = blosc2.compute_varlen(udf)
    assert isinstance(got, blosc2.Utf8Array)
    assert [str(v) for v in got] == [str(v) for v in fixed[:]]


@pytest.mark.parametrize("span", [1, 3, 8, 64])
def test_varlen_spans_do_not_shift_rows(operands, span):
    """Spans are appended in order; an off-by-one would misalign every later row."""
    a, b = operands
    expr = "x=" + a
    fixed = expr.compute(strict_miniexpr=True)
    got = blosc2.compute_varlen(expr, span=span)
    assert len(got) == len(VALUES)
    assert [str(v) for v in got] == [str(v) for v in fixed[:]]


def test_varlen_single_threaded_matches_pooled(operands):
    a, b = operands
    expr = "x=" + a
    one = blosc2.compute_varlen(expr, span=2, max_workers=1)
    many = blosc2.compute_varlen(expr, span=2, max_workers=4)
    assert [str(v) for v in one] == [str(v) for v in many]


def test_varlen_rejects_a_numeric_expression():
    x = blosc2.asarray(np.arange(10, dtype=np.float64))
    with pytest.raises(ValueError, match="string"):
        blosc2.compute_varlen(x + 1)


def test_varlen_rejects_a_non_lazy_argument(operands):
    a, _ = operands
    with pytest.raises(TypeError, match="LazyExpr"):
        blosc2.compute_varlen(a)


def test_varlen_empty_operands():
    a = blosc2.asarray(np.array([], dtype="<U8"))
    got = blosc2.compute_varlen("x=" + a)
    assert isinstance(got, blosc2.Utf8Array)
    assert len(got) == 0


def test_extend_encoded_round_trips():
    arr = blosc2.Utf8Array(blosc2.utf8())
    arr.extend(["first", "second"])
    arr.flush()
    payload = ["café", "", "日本語"]
    encoded = [v.encode("utf-8") for v in payload]
    offsets = np.concatenate([[0], np.cumsum([len(e) for e in encoded])]).astype(np.int64)
    data = np.frombuffer(b"".join(encoded), dtype=np.uint8)
    arr.extend_encoded(offsets, data)
    assert [str(v) for v in arr] == ["first", "second", *payload]


def test_extend_encoded_rejects_mismatched_lengths():
    arr = blosc2.Utf8Array(blosc2.utf8())
    with pytest.raises(ValueError, match="offsets"):
        arr.extend_encoded(np.array([0, 3], dtype=np.int64), np.zeros(5, dtype=np.uint8))
