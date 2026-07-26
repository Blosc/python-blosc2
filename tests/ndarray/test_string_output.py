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
    got = (arr + "XYZ").compute()
    expected = full + "XYZ"
    assert got.dtype == expected.dtype
    assert list(got[:]) == list(expected)


def test_concat_two_arrays(names):
    other = np.array(["/x"] * names.size, dtype="<U4")
    a, b = blosc2.asarray(names), blosc2.asarray(other)
    got = (a + b).compute()
    expected = names + other
    assert list(got[:]) == list(expected)
    # miniexpr's bound is conservative: at least wide enough, never narrower.
    assert got.dtype.itemsize >= expected.dtype.itemsize


@pytest.mark.parametrize("func", ["lower", "upper"])
def test_case_matches_numpy(names, func):
    arr = blosc2.asarray(names)
    got = getattr(blosc2, func)(arr).compute()
    expected = getattr(np.strings, func)(names)
    assert list(got[:]) == list(expected)


def test_case_expansion_matches_numpy():
    # NumPy uses full case mapping; a 1:1 table would give "STRAßE" here.
    src = np.array(["straße", "ﬁx"] * 64, dtype="<U8")
    arr = blosc2.asarray(src)
    got = blosc2.upper(arr).compute()
    expected = np.strings.upper(src)
    assert list(got[:]) == list(expected)


def test_nested_kernel_shape(names):
    arr = blosc2.asarray(names)
    got = ("room_type=" + blosc2.lower(arr)).compute()
    expected = "room_type=" + np.strings.lower(names)
    assert list(got[:]) == list(expected)


def test_uses_miniexpr_not_the_numpy_fallback(names):
    # A silent fallback would still produce correct values, so assert the engine
    # by checking miniexpr can type the expression at all.
    out = blosc2_ext.me_output_dtype("o0 + 'x'", {"o0": names.dtype})
    assert out is not None
    assert out.kind == "U"


def test_probe_reports_none_for_unsupported():
    # Bytes are not wired up yet; the probe must say so rather than guess, so
    # the caller keeps its numpy path.
    assert blosc2_ext.me_output_dtype("o0 + o1", {"o0": "S8", "o1": "S8"}) is None
    assert blosc2_ext.me_output_dtype("nosuchfunc(o0)", {"o0": "<U8"}) is None


def test_string_literal_with_equals_sign():
    # Regression: expressions containing a literal like 'property_type=' used to
    # be misrouted to miniexpr's DSL parser and fail to compile.
    src = np.array(["home", "loft"] * 64, dtype="<U8")
    arr = blosc2.asarray(src)
    got = ("property_type=" + arr).compute()
    expected = "property_type=" + src
    assert list(got[:]) == list(expected)
