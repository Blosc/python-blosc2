#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

import numexpr as ne
import numpy as np
import pytest

import blosc2

NITEMS_SMALL = 1_000
NITEMS = 10_000


@pytest.fixture(
    params=[
        (np.float32, np.float64),
        pytest.param((np.float64, np.float64), marks=pytest.mark.heavy),
        (np.int32, np.float32),
        (np.int32, np.uint32),
        (np.int8, np.int16),
        # The next dtypes work, but running everything takes too much time
        pytest.param((np.int32, np.float64), marks=pytest.mark.heavy),
        pytest.param((np.int8, np.float64), marks=pytest.mark.heavy),
        pytest.param((np.uint8, np.uint16), marks=pytest.mark.heavy),
        pytest.param((np.uint8, np.uint32), marks=pytest.mark.heavy),
        pytest.param((np.uint8, np.float32), marks=pytest.mark.heavy),
        pytest.param((np.uint16, np.float64), marks=pytest.mark.heavy),
    ]
)
def dtype_fixture(request):
    return request.param


@pytest.fixture(params=[(NITEMS_SMALL,), (NITEMS,), (NITEMS // 100, 100)])
def shape_fixture(request):
    return request.param


# params: (same_chunks, same_blocks)
@pytest.fixture(
    params=[
        (True, True),
        (True, False),
        pytest.param((False, True), marks=pytest.mark.heavy),
        (False, False),
    ]
)
def chunks_blocks_fixture(request):
    return request.param


@pytest.fixture
def array_fixture(dtype_fixture, shape_fixture, chunks_blocks_fixture):
    nelems = np.prod(shape_fixture)
    dt1, dt2 = dtype_fixture
    na1_ = np.linspace(0, nelems, nelems, dtype=dt1).reshape(shape_fixture)
    na2_ = np.linspace(10, 10 + nelems, nelems, dtype=dt2).reshape(shape_fixture)
    na1 = np.empty(shape_fixture, dtype=[("a", dt1), ("b", dt2)])
    na1["a"] = na1_
    na1["b"] = na2_
    same_chunks_blocks = chunks_blocks_fixture[0] and chunks_blocks_fixture[1]
    same_chunks = chunks_blocks_fixture[0]
    same_blocks = chunks_blocks_fixture[1]
    if same_chunks_blocks:
        # For full generality, use partitions with padding
        chunks = chunks1 = [c // 11 for c in na1.shape]
        blocks = blocks1 = [c // 71 for c in na1.shape]
    elif same_chunks:
        chunks = [c // 11 for c in na1.shape]
        blocks = [c // 71 for c in na1.shape]
        chunks1 = [c // 11 for c in na1.shape]
        blocks1 = [c // 51 for c in na1.shape]
    elif same_blocks:
        chunks = [c // 11 for c in na1.shape]
        blocks = [c // 71 for c in na1.shape]
        chunks1 = [c // 23 for c in na1.shape]
        blocks1 = [c // 71 for c in na1.shape]
    else:
        # Different chunks and blocks
        chunks = [c // 17 for c in na1.shape]
        blocks = [c // 19 for c in na1.shape]
        chunks1 = [c // 23 for c in na1.shape]
        blocks1 = [c // 29 for c in na1.shape]
    a1 = blosc2.asarray(na1, chunks=chunks, blocks=blocks)
    fna1 = na1["a"]
    fna2 = na1["b"]
    fa1 = a1.fields["a"]
    fa2 = a1.fields["b"]
    na2 = np.copy(na1)
    a2 = blosc2.asarray(na2, chunks=chunks1, blocks=blocks1)
    fna3 = na2["a"]
    fna4 = na2["b"]
    fa3 = blosc2.NDField(a2, "a")
    fa4 = blosc2.NDField(a2, "b")
    return a1, a2, na1, na2, fa1, fa2, fa3, fa4, fna1, fna2, fna3, fna4


def test_simple_getitem(array_fixture):
    sa1, sa2, nsa1, nsa2, a1, a2, a3, a4, na1, na2, na3, na4 = array_fixture
    expr = a1 + a2 - a3 * a4
    nres = na1 + na2 - na3 * na4
    sl = slice(100)
    res = expr[sl]
    np.testing.assert_allclose(res, nres[sl], rtol=1e-6)


def test_simple_getitem_proxy(array_fixture):
    sa1, sa2, nsa1, nsa2, a1, a2, a3, a4, na1, na2, na3, na4 = array_fixture
    sa1 = blosc2.Proxy(sa1)
    a1 = sa1.fields["a"]
    a2 = sa1.fields["b"]
    expr = a1 + a2 - a3 * a4
    nres = na1 + na2 - na3 * na4
    sl = slice(100)
    res = expr[sl]
    np.testing.assert_allclose(res, nres[sl], rtol=1e-6)


# Add more test functions to test different aspects of the code
def test_simple_expression(array_fixture):
    sa1, sa2, nsa1, nsa2, a1, a2, a3, a4, na1, na2, na3, na4 = array_fixture
    expr = a1 + a2 - a3 * a4
    nres = na1 + na2 - na3 * na4
    res = expr.compute()
    np.testing.assert_allclose(res[:], nres, rtol=1e-6)


def test_simple_expression_proxy(array_fixture):
    sa1, sa2, nsa1, nsa2, a1, a2, a3, a4, na1, na2, na3, na4 = array_fixture
    sa1 = blosc2.Proxy(sa1)
    a1 = sa1.fields["a"]
    sa2 = blosc2.Proxy(sa2)
    a4 = sa2.fields["b"]
    expr = a1 + a2 - a3 * a4
    nres = na1 + na2 - na3 * na4
    res = expr.compute()
    np.testing.assert_allclose(res[:], nres, rtol=1e-6)


def test_iXXX(array_fixture):
    sa1, sa2, nsa1, nsa2, a1, a2, a3, a4, na1, na2, na3, na4 = array_fixture
    expr = a1**3 + a2**2 + a3**3 - a4 + 3
    expr += 5  # __iadd__
    expr -= 15  # __isub__
    expr *= 2  # __imul__
    expr /= 7  # __itruediv__
    expr **= 2.3  # __ipow__
    res = expr.compute()
    nres = ne.evaluate("(((((na1 ** 3 + na2 ** 2 + na3 ** 3 - na4 + 3) + 5) - 15) * 2) / 7) ** 2.3")
    # NumPy raises: RuntimeWarning: invalid value encountered in power
    # nres = (((((na1 ** 3 + na2 ** 2 + na3 ** 3 - na4 + 3) + 5) - 15) * 2) / 7) ** 2.3
    np.testing.assert_allclose(res[:], nres)


def test_complex_evaluate(array_fixture):
    sa1, sa2, nsa1, nsa2, a1, a2, a3, a4, na1, na2, na3, na4 = array_fixture
    expr = blosc2.tan(a1) * (blosc2.sin(a2) * blosc2.sin(a2) + blosc2.cos(a3)) + (blosc2.sqrt(a4) * 2)
    expr += 2
    nres = ne.evaluate("tan(na1) * (sin(na2) * sin(na2) + cos(na3)) + (sqrt(na4) * 2) + 2")
    # This slightly differs from numexpr, but it is correct (kind of)
    # nres = np.tan(na1) * (np.sin(na2) * np.sin(na2) + np.cos(na3)) + (np.sqrt(na4) * 2) + 2
    res = expr.compute()
    np.testing.assert_allclose(res[:], nres)


def test_complex_getitem_slice(array_fixture):
    sa1, sa2, nsa1, nsa2, a1, a2, a3, a4, na1, na2, na3, na4 = array_fixture
    expr = blosc2.tan(a1) * (blosc2.sin(a2) * blosc2.sin(a2) + blosc2.cos(a3)) + (blosc2.sqrt(a4) * 2)
    expr += 2
    nres = ne.evaluate("tan(na1) * (sin(na2) * sin(na2) + cos(na3)) + (sqrt(na4) * 2) + 2")
    sl = slice(100)
    res = expr[sl]
    np.testing.assert_allclose(res, nres[sl])


def test_reductions(array_fixture):
    sa1, sa2, nsa1, nsa2, a1, a2, a3, a4, na1, na2, na3, na4 = array_fixture
    expr = a1 + a2 - a3 * a4
    nres = ne.evaluate("na1 + na2 - na3 * na4")
    # Use relative tolerance for mean and std
    np.testing.assert_allclose(expr.sum()[()], nres.sum())
    np.testing.assert_allclose(expr.mean()[()], nres.mean(), rtol=1e-5)
    np.testing.assert_allclose(expr.min()[()], nres.min())
    np.testing.assert_allclose(expr.max()[()], nres.max())
    np.testing.assert_allclose(expr.std()[()], nres.std(), rtol=1e-3)


def test_mixed_operands(array_fixture):
    sa1, sa2, nsa1, nsa2, a1, a2, a3, a4, na1, na2, na3, na4 = array_fixture
    # All a1, a2, a3 and a4 are NDFields
    a3 = blosc2.asarray(na3)  # this is a NDArray now
    assert not isinstance(a3, blosc2.NDField)
    a4 = na4  # this is a NumPy array now
    assert not isinstance(a4, blosc2.NDField)
    expr = a1 + a2 - a3 * a4
    nres = na1 + na2 - na3 * na4
    res = expr.compute()
    np.testing.assert_allclose(res[:], nres, rtol=1e-6)


# Test expressions with where()
def test_where(array_fixture):
    sa1, sa2, nsa1, nsa2, a1, a2, a3, a4, na1, na2, na3, na4 = array_fixture
    expr = a1**2 + a2**2 > 2 * a1 * a2 + 1
    # Test with eval
    res = expr.where(0, 1).compute()
    nres = ne.evaluate("where(na1**2 + na2**2 > 2 * na1 * na2 + 1, 0, 1)")
    np.testing.assert_allclose(res[:], nres)
    # Test with getitem
    sl = slice(100)
    res = expr.where(0, 1)[sl]
    np.testing.assert_allclose(res, nres[sl])


# Test where with one parameter
def test_where_one_param(array_fixture):
    sa1, sa2, nsa1, nsa2, a1, a2, a3, a4, na1, na2, na3, na4 = array_fixture
    expr = a1**2 + a2**2 > 2 * a1 * a2 + 1
    # Test with eval
    res = expr.where(a1).compute()
    nres = na1[ne.evaluate("na1**2 + na2**2 > 2 * na1 * na2 + 1")]
    np.testing.assert_allclose(res[:], nres)
    # Test with getitem
    sl = slice(100)
    res = expr.where(a1)[sl]
    np.testing.assert_allclose(res, nres[sl])


# Test where indirectly via a condition in getitem in a NDArray
def test_where_getitem(array_fixture):
    sa1, sa2, nsa1, nsa2, a1, a2, a3, a4, na1, na2, na3, na4 = array_fixture

    # Test with complete slice
    res = sa1[a1**2 + a2**2 > 2 * a1 * a2 + 1].compute()
    nres = nsa1[ne.evaluate("na1**2 + na2**2 > 2 * na1 * na2 + 1")]
    np.testing.assert_allclose(res["a"][:], nres["a"])
    np.testing.assert_allclose(res["b"][:], nres["b"])
    # string version
    res = sa1["a**2 + b**2 > 2 * a * b + 1"].compute()
    np.testing.assert_allclose(res["a"][:], nres["a"])
    np.testing.assert_allclose(res["b"][:], nres["b"])

    # Test with partial slice
    sl = slice(100)
    res = sa1[a1**2 + a2**2 > 2 * a1 * a2 + 1][sl]
    np.testing.assert_allclose(res["a"], nres[sl]["a"])
    np.testing.assert_allclose(res["b"], nres[sl]["b"])
    # string version
    res = sa1["a**2 + b**2 > 2 * a * b + 1"][sl]
    np.testing.assert_allclose(res["a"], nres[sl]["a"])
    np.testing.assert_allclose(res["b"], nres[sl]["b"])


# Test where indirectly via a condition in getitem in a NDField
# Test boolean operators here too
@pytest.mark.parametrize("npflavor", [True, False])
@pytest.mark.parametrize("lazystr", [True, False])
def test_where_getitem_field(array_fixture, npflavor, lazystr):
    sa1, sa2, nsa1, nsa2, a1, a2, a3, a4, na1, na2, na3, na4 = array_fixture
    if a1.dtype == np.int8 or a2.dtype == np.int8:
        # Skip this test for short ints because of casting differences between NumPy and numexpr
        return
    if npflavor:
        a2 = na2
    # Let's put a *bitwise_or* at the front to test the ufunc mechanism of NumPy
    if lazystr:
        expr = blosc2.lazyexpr("(a2 < 0) | ~((a1**2 > a2**2) & ~(a1 * a2 > 1))")
    else:
        expr = (a2 < 0) | ~((a1**2 > a2**2) & ~(a1 * a2 > 1))
    assert expr.dtype == np.bool_
    # Compute and check
    res = a1[expr]
    nres = na1[ne.evaluate("(na2 < 0) | ~((na1**2 > na2**2) & ~(na1 * na2 > 1))")]
    np.testing.assert_allclose(res[:], nres)
    # Test with getitem
    sl = slice(100)
    ressl = res[sl]
    np.testing.assert_allclose(ressl, nres[sl])


# Test where combined with a reduction
def test_where_reduction1(array_fixture):
    sa1, sa2, nsa1, nsa2, a1, a2, a3, a4, na1, na2, na3, na4 = array_fixture
    expr = a1**2 + a2**2 > 2 * a1 * a2 + 1
    axis = None if sa1.ndim == 1 else 1
    res = expr.where(0, 1).sum(axis=axis)
    nres = ne.evaluate("where(na1**2 + na2**2 > 2 * na1 * na2 + 1, 0, 1)").sum(axis=axis)
    np.testing.assert_allclose(res, nres)


# Test *implicit* where (a query) combined with a reduction
def test_where_reduction2(array_fixture):
    sa1, sa2, nsa1, nsa2, a1, a2, a3, a4, na1, na2, na3, na4 = array_fixture
    axis = None if sa1.ndim == 1 else 1
    # We have to use the original names in fields here
    expr = sa1[f"(b * a.sum(axis={axis})) > 0"]
    res = expr[:]
    nres = nsa1[(na2 * na1.sum(axis=axis)) > 0]
    np.testing.assert_allclose(res["a"], nres["a"])


# More complex cases with where() calls combined with reductions,
# broadcasting, reusing the result in another expression and other
# funny stuff


# Two where() calls
def test_where_fusion1(array_fixture):
    sa1, sa2, nsa1, nsa2, a1, a2, a3, a4, na1, na2, na3, na4 = array_fixture
    expr = a1**2 + a2**2 > 2 * a1 * a2 + 1
    npexpr = ne.evaluate("na1**2 + na2**2 > 2 * na1 * na2 + 1")

    res = expr.where(0, 1) + expr.where(0, 1)
    nres = np.where(npexpr, 0, 1) + np.where(npexpr, 0, 1)
    np.testing.assert_allclose(res[:], nres)


# Two where() calls with a reduction (and using broadcasting)
def test_where_fusion2(array_fixture):
    sa1, sa2, nsa1, nsa2, a1, a2, a3, a4, na1, na2, na3, na4 = array_fixture
    expr = a1**2 + a2**2 > 2 * a1 * a2 + 1
    npexpr = ne.evaluate("na1**2 + na2**2 > 2 * na1 * na2 + 1")

    axis = None if sa1.ndim == 1 else 1
    res = expr.where(0.5, 0.2) + expr.where(0.3, 0.6).sum(axis=axis)
    nres = np.where(npexpr, 0.5, 0.2) + np.where(npexpr, 0.3, 0.6).sum(axis=axis)
    np.testing.assert_allclose(res[:], nres)


# Reuse the result in another expression
def test_where_fusion3(array_fixture):
    sa1, sa2, nsa1, nsa2, a1, a2, a3, a4, na1, na2, na3, na4 = array_fixture
    expr = a1**2 + a2**2 > 2 * a1 * a2 + 1
    npexpr = ne.evaluate("na1**2 + na2**2 > 2 * na1 * na2 + 1")

    res = expr.where(0, 1) + expr.where(0, 1)
    nres = np.where(npexpr, 0, 1) + np.where(npexpr, 0, 1)
    res = expr.where(0, 1) + res.sum()
    nres = np.where(npexpr, 0, 1) + nres.sum()
    np.testing.assert_allclose(res[:], nres)


# Reuse the result in another expression twice
def test_where_fusion4(array_fixture):
    sa1, sa2, nsa1, nsa2, a1, a2, a3, a4, na1, na2, na3, na4 = array_fixture
    expr = a1**2 + a2**2 > 2 * a1 * a2 + 1
    npexpr = ne.evaluate("na1**2 + na2**2 > 2 * na1 * na2 + 1")

    res = expr.where(0.1, 0.7) + expr.where(0.2, 5)
    nres = np.where(npexpr, 0.1, 0.7) + np.where(npexpr, 0.2, 5)
    res = 2 * res + 4 * res
    nres = 2 * nres + 4 * nres
    np.testing.assert_allclose(res[:], nres)


# Reuse the result in another expression twice II
def test_where_fusion5(array_fixture):
    sa1, sa2, nsa1, nsa2, a1, a2, a3, a4, na1, na2, na3, na4 = array_fixture
    expr = a1**2 + a2**2 > 2 * a1 * a2 + 1
    npexpr = ne.evaluate("na1**2 + na2**2 > 2 * na1 * na2 + 1")

    res = expr.where(-1, 7) + expr.where(2, 5)
    nres = np.where(npexpr, -1, 7) + np.where(npexpr, 2, 5)
    res = 2 * res + blosc2.sqrt(res)
    nres = 2 * nres + np.sqrt(nres)
    np.testing.assert_allclose(res[:], nres)


# Reuse the result in another expression twice III
def test_where_fusion6(array_fixture):
    sa1, sa2, nsa1, nsa2, a1, a2, a3, a4, na1, na2, na3, na4 = array_fixture
    expr = a1**2 + a2**2 > 2 * a1 * a2 + 1
    npexpr = ne.evaluate("na1**2 + na2**2 > 2 * na1 * na2 + 1")

    res = expr.where(-1, 1) + expr.where(2, 1)
    nres = np.where(npexpr, -1, 1) + np.where(npexpr, 2, 1)
    res = expr.where(6.1, 1) + res
    nres = np.where(npexpr, 6.1, 1) + nres
    np.testing.assert_allclose(res[:], nres)
