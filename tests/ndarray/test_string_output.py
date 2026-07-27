#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""String-valued expression results (miniexpr string output).

The values are checked against NumPy, but the *width* matters just as much: the
output container is allocated before evaluation, so if it is sized from NumPy's
``result_type`` instead of miniexpr's inference the kernel silently truncates.
"""

import numpy as np
import pytest

import blosc2
from blosc2 import blosc2_ext

NAMES = [
    "Cozy Loft With City View",
    "Small Single Room",
    "Studio",
    "Double Room",
]


@pytest.fixture
def names():
    return np.array(NAMES * 64, dtype="<U24")


def test_concat_scalar_does_not_truncate():
    # An operand already full to its width plus a suffix: sizing the output from
    # numpy's view of the *operand* would drop the suffix entirely.
    full = np.array(["A" * 16] * 128, dtype="<U16")
    arr = blosc2.asarray(full)
    got = (arr + "XYZ").compute(strict_miniexpr=True)
    expected = full + "XYZ"
    assert got.dtype == expected.dtype
    assert list(got[:]) == list(expected)


def test_concat_two_arrays(names):
    other = np.array(["/x"] * names.size, dtype="<U4")
    a, b = blosc2.asarray(names), blosc2.asarray(other)
    got = (a + b).compute(strict_miniexpr=True)
    expected = names + other
    assert list(got[:]) == list(expected)
    # miniexpr's bound is conservative: at least wide enough, never narrower.
    assert got.dtype.itemsize >= expected.dtype.itemsize


@pytest.mark.parametrize("func", ["lower", "upper"])
def test_case_matches_numpy(names, func):
    arr = blosc2.asarray(names)
    got = getattr(blosc2, func)(arr).compute(strict_miniexpr=True)
    expected = getattr(np.strings, func)(names)
    assert list(got[:]) == list(expected)


def test_case_expansion_matches_numpy():
    # NumPy uses full case mapping; a 1:1 table would give "STRAßE" here.
    src = np.array(["straße", "ﬁx"] * 64, dtype="<U8")
    arr = blosc2.asarray(src)
    got = blosc2.upper(arr).compute(strict_miniexpr=True)
    expected = np.strings.upper(src)
    assert list(got[:]) == list(expected)


def test_nested_kernel_shape(names):
    arr = blosc2.asarray(names)
    got = ("room_type=" + blosc2.lower(arr)).compute(strict_miniexpr=True)
    expected = "room_type=" + np.strings.lower(names)
    assert list(got[:]) == list(expected)


def test_uses_miniexpr_not_the_numpy_fallback(names):
    # A silent fallback would still produce correct values, so assert the engine
    # by checking miniexpr can type the expression at all.
    out = blosc2_ext.me_output_dtype("o0 + 'x'", {"o0": names.dtype})
    assert out is not None
    assert out.kind == "U"
    # ...and that evaluation really goes through it.  Every other test here
    # passes strict_miniexpr=True for the same reason.
    arr = blosc2.asarray(names)
    assert list((arr + "x").compute(strict_miniexpr=True)[:]) == list(names + "x")


BLOG_PROPS = ["Entire home", "Private room", "Shared room", "Loft"]


def _blog_kernel_numpy(prop, name):
    """The pandas-3 blog kernel, run row by row in plain Python."""
    result = "property_type=" + prop
    desc = name.lower()
    if " with " not in desc:
        return result + ", room_type=" + desc.removesuffix(" room")
    before, after = desc.split(" with ", 1)
    r2 = result + ", room_type=" + before.removesuffix(" room")
    return r2 + ", amenity=" + after


def test_blog_kernel_as_dsl_kernel(names):
    """End-to-end acceptance: the motivating kernel, unmodified, over NDArrays.

    Exercises method syntax, tuple unpacking, an early return from an `if`, and
    branches whose string results have different widths.
    """

    @blosc2.dsl_kernel
    def format_room_info(property_type, name):
        result = "property_type=" + property_type
        desc = name.lower()
        if " with " not in desc:
            return result + ", room_type=" + desc.removesuffix(" room")
        before, after = desc.split(" with ", 1)
        r2 = result + ", room_type=" + before.removesuffix(" room")
        return r2 + ", amenity=" + after

    props = np.array((BLOG_PROPS * 64)[: names.size], dtype="<U16")
    lazy = blosc2.lazyudf(format_room_info, (blosc2.asarray(props), blosc2.asarray(names)))
    got = lazy.compute(strict_miniexpr=True)[:]
    expected = [_blog_kernel_numpy(p, n) for p, n in zip(props, names, strict=True)]
    assert list(got) == expected


def test_probe_reports_none_for_unsupported():
    # When miniexpr cannot type an expression the probe must say so rather than
    # guess, so the caller keeps its numpy path.
    assert blosc2_ext.me_output_dtype("nosuchfunc(o0)", {"o0": "<U8"}) is None
    assert blosc2_ext.me_output_dtype("o0 + o1", {"o0": "<U8", "o1": "f8"}) is None


def test_string_literal_with_equals_sign():
    # Regression: expressions containing a literal like 'property_type=' used to
    # be misrouted to miniexpr's DSL parser and fail to compile.
    src = np.array(["home", "loft"] * 64, dtype="<U8")
    arr = blosc2.asarray(src)
    got = ("property_type=" + arr).compute(strict_miniexpr=True)
    expected = "property_type=" + src
    assert list(got[:]) == list(expected)


# --- bytes ('S') columns: same kernels, 1-byte code units -------------------

BYTES_VALUES = [b"foo", b"Hello", b" pad  ", b"abcdefgh"]


@pytest.fixture
def raws():
    return np.array(BYTES_VALUES * 32, dtype="S8")


def test_bytes_concat_matches_numpy(raws):
    arr = blosc2.asarray(raws)
    got = (arr + b"-x").compute(strict_miniexpr=True)
    expected = raws + b"-x"
    assert got.dtype == expected.dtype
    assert list(got[:]) == list(expected)


@pytest.mark.parametrize("func", ["lower", "upper"])
def test_bytes_case_matches_numpy(raws, func):
    # NumPy's `S` case mapping is ASCII-only and 1:1, so unlike `U` the width
    # must not grow.
    arr = blosc2.asarray(raws)
    got = getattr(blosc2, func)(arr).compute(strict_miniexpr=True)
    expected = getattr(np.strings, func)(raws)
    assert got.dtype == expected.dtype
    assert list(got[:]) == list(expected)


def test_bytes_predicates_match_numpy(raws):
    arr = blosc2.asarray(raws)
    assert list((arr == b"foo").compute(strict_miniexpr=True)[:]) == list(raws == b"foo")
    got = blosc2.contains(arr, b"ell").compute(strict_miniexpr=True)
    assert list(got[:]) == list(np.strings.find(raws, b"ell") >= 0)


def test_bytes_dsl_kernel(raws):
    @blosc2.dsl_kernel
    def tag(x):
        if b"o" in x:
            return x
        return b"long-" + x

    arr = blosc2.asarray(raws)
    got = blosc2.lazyudf(tag, (arr,)).compute(strict_miniexpr=True)
    expected = [v if b"o" in v else b"long-" + v for v in raws]
    assert list(got[:]) == expected


def test_bytes_and_str_do_not_mix(raws):
    # NumPy raises on `S` + `U` too; miniexpr must not silently pick one.
    assert blosc2_ext.me_output_dtype("o0 + o1", {"o0": "S8", "o1": "<U8"}) is None


# ---------------------------------------------------------------------------
# Wide operands: itemsize above BLOSC_MAX_TYPESIZE (255 bytes)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("width", [63, 64, 128, 512])
def test_wide_string_operands_match_numpy(width):
    # c-blosc2 caps a typesize above 255 to 1 in the chunk header so its split
    # machinery keeps working, so the miniexpr prefilter must ask
    # blosc2_getitem_ctx() in *header* units.  Asking in element units read a
    # byte range instead: every block past the first was uninitialised memory
    # and predicates came back almost entirely False.  <U64 is 256 bytes, the
    # first width that trips it.
    values = np.array(["hello", "help", "world"] * 400, dtype=f"<U{width}")
    arr = blosc2.asarray(values)
    assert arr.dtype.itemsize == width * 4
    assert arr.blocks[0] < len(values), "need several blocks to catch the bug"

    got = (arr == "hello").compute(strict_miniexpr=True)
    assert list(got[:]) == list(values == "hello")

    upper = blosc2.upper(arr).compute(strict_miniexpr=True)
    assert list(upper[:]) == list(np.strings.upper(values))
