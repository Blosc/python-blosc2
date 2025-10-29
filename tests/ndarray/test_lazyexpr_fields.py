#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

import numpy as np
import pytest

import blosc2
from blosc2.lazyexpr import ne_evaluate

NITEMS_SMALL = 100
NITEMS = 1000


@pytest.fixture(
    params=[
        (np.float32, np.float64),
        pytest.param((np.float64, np.float64), marks=pytest.mark.heavy),
        (np.int32, np.float32),
        (np.int32, np.uint32),
        pytest.param(
            (np.int8, np.int16),
            marks=pytest.mark.skipif(
                np.__version__.startswith("1."), reason="NumPy < 2.0 has different casting rules"
            ),
        ),
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


@pytest.fixture(
    params=[(NITEMS_SMALL,), (NITEMS,), pytest.param((NITEMS // 10, 100), marks=pytest.mark.heavy)]
)
def shape_fixture(request):
    return request.param


# params: (same_chunks, same_blocks)
@pytest.fixture(
    params=[
        (True, True),
        (True, False),
        pytest.param((False, True), marks=pytest.mark.heavy),
        pytest.param((False, False), marks=pytest.mark.heavy),
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
    if not blosc2.IS_WASM:
        expr **= 2.3  # __ipow__
    res = expr.compute()
    if not blosc2.IS_WASM:
        nres = ne_evaluate("(((((na1 ** 3 + na2 ** 2 + na3 ** 3 - na4 + 3) + 5) - 15) * 2) / 7) ** 2.3")
    else:
        nres = ne_evaluate("(((((na1 ** 3 + na2 ** 2 + na3 ** 3 - na4 + 3) + 5) - 15) * 2) / 7)")
    # NumPy raises: RuntimeWarning: invalid value encountered in power
    # nres = (((((na1 ** 3 + na2 ** 2 + na3 ** 3 - na4 + 3) + 5) - 15) * 2) / 7) ** 2.3
    np.testing.assert_allclose(res[:], nres)


def test_complex_evaluate(array_fixture):
    sa1, sa2, nsa1, nsa2, a1, a2, a3, a4, na1, na2, na3, na4 = array_fixture
    expr = blosc2.tan(a1) * (blosc2.sin(a2) * blosc2.sin(a2) + blosc2.cos(a3)) + (blosc2.sqrt(a4) * 2)
    expr += 2
    nres = ne_evaluate("tan(na1) * (sin(na2) * sin(na2) + cos(na3)) + (sqrt(na4) * 2) + 2")
    # This slightly differs from numexpr, but it is correct (kind of)
    # nres = np.tan(na1) * (np.sin(na2) * np.sin(na2) + np.cos(na3)) + (np.sqrt(na4) * 2) + 2
    res = expr.compute()
    np.testing.assert_allclose(res[:], nres)


def test_complex_getitem_slice(array_fixture):
    sa1, sa2, nsa1, nsa2, a1, a2, a3, a4, na1, na2, na3, na4 = array_fixture
    expr = blosc2.tan(a1) * (blosc2.sin(a2) * blosc2.sin(a2) + blosc2.cos(a3)) + (blosc2.sqrt(a4) * 2)
    expr += 2
    nres = ne_evaluate("tan(na1) * (sin(na2) * sin(na2) + cos(na3)) + (sqrt(na4) * 2) + 2")
    sl = slice(100)
    res = expr[sl]
    np.testing.assert_allclose(res, nres[sl])


def test_reductions(array_fixture):
    sa1, sa2, nsa1, nsa2, a1, a2, a3, a4, na1, na2, na3, na4 = array_fixture
    expr = a1 + a2 - a3 * a4
    nres = ne_evaluate("na1 + na2 - na3 * na4")
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
    nres = ne_evaluate("where(na1**2 + na2**2 > 2 * na1 * na2 + 1, 0, 1)")
    np.testing.assert_allclose(res[:], nres)

    # Test with getitem
    sl = slice(100)
    res = expr.where(0, 1)[sl]
    np.testing.assert_allclose(res, nres[sl])

    # Test with string
    res = blosc2.evaluate("where(a1**2 + a2**2 > 2 * a1 * a2 + 1, a1 + 5, a2)")
    nres = ne_evaluate("where(na1**2 + na2**2 > 2 * na1 * na2 + 1, na1 + 5, na2)")
    np.testing.assert_allclose(res, nres)


# Test expressions with where() and string comps
def test_lazy_where(array_fixture):
    sa1, sa2, nsa1, nsa2, a1, a2, a3, a4, na1, na2, na3, na4 = array_fixture

    # Test 1: where
    # Test with string expression
    expr = blosc2.lazyexpr("where((a1 ** 2 + a2 ** 2) > (2 * a1 * a2 + 1), 0, a1)")
    # Test with eval
    res = expr.compute()
    nres = ne_evaluate("where(na1**2 + na2**2 > 2 * na1 * na2 + 1, 0, na1)")
    np.testing.assert_allclose(res[:], nres)
    # Test with getitem
    sl = slice(100)
    res = expr[sl]
    np.testing.assert_allclose(res, nres[sl])

    # Test 2: sum of wheres
    # Test with string expression
    expr = blosc2.lazyexpr("where(a1 < 0, 10, a1) + where(a2 < 0, 3, a2)")
    # Test with eval
    res = expr.compute()
    nres = ne_evaluate("where(na1 < 0, 10, na1) + where(na2 < 0, 3, na2)")
    np.testing.assert_allclose(res[:], nres)

    # Test 3: nested wheres
    # Test with string expression
    expr = blosc2.lazyexpr("where(where(a2 < 0, 3, a2) > 3, 10, a1)")
    # Test with eval
    res = expr.compute()
    nres = ne_evaluate("where(where(na2 < 0, 3, na2) > 3, 10, na1)")
    np.testing.assert_allclose(res[:], nres)

    # Test 4: multiplied wheres
    # Test with string expression
    expr = blosc2.lazyexpr("1 * where(a2 < 0, 3, a2)")
    # Test with eval
    res = expr.compute()
    nres = ne_evaluate("1 * where(na2 < 0, 3, na2)")
    np.testing.assert_allclose(res[:], nres)


# Test where with one parameter
def test_where_one_param(array_fixture):
    sa1, sa2, nsa1, nsa2, a1, a2, a3, a4, na1, na2, na3, na4 = array_fixture
    expr = a1**2 + a2**2 > 2 * a1 * a2 + 1
    # Test with eval
    res = expr.where(a1).compute()
    nres = na1[ne_evaluate("na1**2 + na2**2 > 2 * na1 * na2 + 1")]
    # On general chunked ndim arrays, we cannot guarantee the order of the results
    if not (len(a1.shape) == 1 or a1.chunks == a1.shape):
        res = np.sort(res)
        nres = np.sort(nres)
    np.testing.assert_allclose(res[:], nres)

    # Test with getitem
    sl = slice(100)
    res = expr.where(a1)[sl]
    nres = na1[sl][ne_evaluate("na1**2 + na2**2 > 2 * na1 * na2 + 1")[sl]]
    if len(a1.shape) == 1 or a1.chunks == a1.shape:
        # TODO: fix this, as it seems that is not working well for numexpr?
        if blosc2.IS_WASM:
            return
        np.testing.assert_allclose(res, nres)
    else:
        # In this case, we cannot compare results, only the length
        assert len(res) == len(nres)


# Test where indirectly via a condition in getitem in a NDArray
def test_where_getitem(array_fixture):
    sa1, sa2, nsa1, nsa2, a1, a2, a3, a4, na1, na2, na3, na4 = array_fixture

    # Test with complete slice
    res = sa1[a1**2 + a2**2 > 2 * a1 * a2 + 1].compute()
    nres = nsa1[ne_evaluate("na1**2 + na2**2 > 2 * na1 * na2 + 1")]
    resa = res["a"][:]
    resb = res["b"][:]
    nresa = nres["a"]
    nresb = nres["b"]
    # On general chunked ndim arrays, we cannot guarantee the order of the results
    if not (len(a1.shape) == 1 or a1.chunks == a1.shape):
        resa = np.sort(resa)
        resb = np.sort(resb)
        nresa = np.sort(nresa)
        nresb = np.sort(nresb)
    np.testing.assert_allclose(resa, nresa)
    np.testing.assert_allclose(resb, nresb)

    # string version
    res = sa1["a**2 + b**2 > 2 * a * b + 1"].compute()
    resa = res["a"][:]
    resb = res["b"][:]
    nresa = nres["a"]
    nresb = nres["b"]
    # On general chunked ndim arrays, we cannot guarantee the order of the results
    if not (len(a1.shape) == 1 or a1.chunks == a1.shape):
        resa = np.sort(resa)
        resb = np.sort(resb)
        nresa = np.sort(nresa)
        nresb = np.sort(nresb)
    np.testing.assert_allclose(resa, nresa)
    np.testing.assert_allclose(resb, nresb)

    # Test with partial slice
    sl = slice(100)
    res = sa1[a1**2 + a2**2 > 2 * a1 * a2 + 1][sl]
    nres = nsa1[sl][ne_evaluate("na1**2 + na2**2 > 2 * na1 * na2 + 1")[sl]]
    if len(a1.shape) == 1 or a1.chunks == a1.shape:
        # TODO: fix this, as it seems that is not working well for numexpr?
        if blosc2.IS_WASM:
            return
        np.testing.assert_allclose(res["a"], nres["a"])
        np.testing.assert_allclose(res["b"], nres["b"])
    else:
        # In this case, we cannot compare results, only the length
        assert len(res["a"]) == len(nres["a"])
        assert len(res["b"]) == len(nres["b"])
    # string version
    res = sa1["a**2 + b**2 > 2 * a * b + 1"][sl]
    if len(a1.shape) == 1 or a1.chunks == a1.shape:
        np.testing.assert_allclose(res["a"], nres["a"])
        np.testing.assert_allclose(res["b"], nres["b"])
    else:
        # We cannot compare the results here, other than the length
        assert len(res["a"]) == len(nres["a"])
        assert len(res["b"]) == len(nres["b"])


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
    res = a1[expr][:]
    nres = na1[ne_evaluate("(na2 < 0) | ~((na1**2 > na2**2) & ~(na1 * na2 > 1))")]
    # On general chunked ndim arrays, we cannot guarantee the order of the results
    if not (len(a1.shape) == 1 or a1.chunks == a1.shape):
        res = np.sort(res)
        nres = np.sort(nres)
    np.testing.assert_allclose(res, nres)
    # Test with getitem
    sl = slice(100)
    ressl = res[sl]
    if len(a1.shape) == 1 or a1.chunks == a1.shape:
        np.testing.assert_allclose(ressl, nres[sl])
    else:
        # In this case, we cannot compare results, only the length
        assert len(ressl) == len(nres[sl])


# Test where combined with a reduction
def test_where_reduction1(array_fixture):
    sa1, sa2, nsa1, nsa2, a1, a2, a3, a4, na1, na2, na3, na4 = array_fixture
    expr = a1**2 + a2**2 > 2 * a1 * a2 + 1
    axis = None if sa1.ndim == 1 else 1
    res = expr.where(0, 1).sum(axis=axis)
    nres = ne_evaluate("where(na1**2 + na2**2 > 2 * na1 * na2 + 1, 0, 1)").sum(axis=axis)
    np.testing.assert_allclose(res, nres)


# Test *implicit* where (a query) combined with a reduction
# TODO: fix this, as it seems that is not working well for numexpr?
@pytest.mark.skipif(blosc2.IS_WASM, reason="numexpr is not behaving as numpy(?")
def test_where_reduction2(array_fixture):
    sa1, sa2, nsa1, nsa2, a1, a2, a3, a4, na1, na2, na3, na4 = array_fixture
    # We have to use the original names in fields here
    expr = sa1["(b * a.sum()) > 0"]
    res = expr[:]
    nres = nsa1[(na2 * na1.sum()) > 0]
    # On general chunked ndim arrays, we cannot guarantee the order of the results
    if not (len(a1.shape) == 1 or a1.chunks == a1.shape):
        np.testing.assert_allclose(np.sort(res["a"]), np.sort(nres["a"]))
    else:
        np.testing.assert_allclose(res["a"], nres["a"])


# More complex cases with where() calls combined with reductions,
# broadcasting, reusing the result in another expression and other
# funny stuff


# Two where() calls
def test_where_fusion1(array_fixture):
    sa1, sa2, nsa1, nsa2, a1, a2, a3, a4, na1, na2, na3, na4 = array_fixture
    expr = a1**2 + a2**2 > 2 * a1 * a2 + 1
    npexpr = ne_evaluate("na1**2 + na2**2 > 2 * na1 * na2 + 1")

    res = expr.where(0, 1) + expr.where(0, 1)
    nres = np.where(npexpr, 0, 1) + np.where(npexpr, 0, 1)
    np.testing.assert_allclose(res[:], nres)


# Two where() calls with a reduction (and using broadcasting)
def test_where_fusion2(array_fixture):
    sa1, sa2, nsa1, nsa2, a1, a2, a3, a4, na1, na2, na3, na4 = array_fixture
    expr = a1**2 + a2**2 > 2 * a1 * a2 + 1
    npexpr = ne_evaluate("na1**2 + na2**2 > 2 * na1 * na2 + 1")

    res = expr.where(0.5, 0.2) + expr.where(0.3, 0.6).sum()
    nres = np.where(npexpr, 0.5, 0.2) + np.where(npexpr, 0.3, 0.6).sum()
    np.testing.assert_allclose(res[:], nres)


# Reuse the result in another expression
def test_where_fusion3(array_fixture):
    sa1, sa2, nsa1, nsa2, a1, a2, a3, a4, na1, na2, na3, na4 = array_fixture
    expr = a1**2 + a2**2 > 2 * a1 * a2 + 1
    npexpr = ne_evaluate("na1**2 + na2**2 > 2 * na1 * na2 + 1")

    res = expr.where(0, 1) + expr.where(0, 1)
    nres = np.where(npexpr, 0, 1) + np.where(npexpr, 0, 1)
    res = expr.where(0, 1) + res.sum()
    nres = np.where(npexpr, 0, 1) + nres.sum()
    np.testing.assert_allclose(res[:], nres)


# Reuse the result in another expression twice
def test_where_fusion4(array_fixture):
    sa1, sa2, nsa1, nsa2, a1, a2, a3, a4, na1, na2, na3, na4 = array_fixture
    expr = a1**2 + a2**2 > 2 * a1 * a2 + 1
    npexpr = ne_evaluate("na1**2 + na2**2 > 2 * na1 * na2 + 1")

    res = expr.where(0.1, 0.7) + expr.where(0.2, 5)
    nres = np.where(npexpr, 0.1, 0.7) + np.where(npexpr, 0.2, 5)
    res = 2 * res + 4 * res
    nres = 2 * nres + 4 * nres
    np.testing.assert_allclose(res[:], nres)


# Reuse the result in another expression twice II
def test_where_fusion5(array_fixture):
    sa1, sa2, nsa1, nsa2, a1, a2, a3, a4, na1, na2, na3, na4 = array_fixture
    expr = a1**2 + a2**2 > 2 * a1 * a2 + 1
    npexpr = ne_evaluate("na1**2 + na2**2 > 2 * na1 * na2 + 1")

    res = expr.where(-1, 7) + expr.where(2, 5)
    nres = np.where(npexpr, -1, 7) + np.where(npexpr, 2, 5)
    res = 2 * res + blosc2.sqrt(res)
    nres = 2 * nres + np.sqrt(nres)
    np.testing.assert_allclose(res[:], nres)


# Reuse the result in another expression twice III
# TODO: fix this, as it seems that is not working well for numexpr?
@pytest.mark.skipif(blosc2.IS_WASM, reason="numexpr is not behaving as numpy(?")
def test_where_fusion6(array_fixture):
    sa1, sa2, nsa1, nsa2, a1, a2, a3, a4, na1, na2, na3, na4 = array_fixture
    expr = a1**2 + a2**2 > 2 * a1 * a2 + 1
    npexpr = ne_evaluate("na1**2 + na2**2 > 2 * na1 * na2 + 1")

    res = expr.where(-1, 1) + expr.where(2, 1)
    nres = np.where(npexpr, -1, 1) + np.where(npexpr, 2, 1)
    res = expr.where(6.1, 1) + res
    nres = np.where(npexpr, 6.1, 1) + nres
    np.testing.assert_allclose(res[:], nres)


@pytest.mark.parametrize(
    ("shape", "chunks", "blocks", "field"),
    [
        ((5,), (2,), (1,), "a"),
        ((15,), (2,), (2,), "b"),
        ((100,), (44,), (33,), "b"),
    ],
)
@pytest.mark.parametrize("order", ["a", "b", None])
def test_indices(shape, chunks, blocks, field, order):
    na = np.arange(1, shape[0] + 1)
    nb = np.arange(2 * shape[0], shape[0], -1)
    nsa = np.empty(shape, dtype=[("a", np.int32), ("b", np.int32)])
    nsa["a"] = na
    nsa["b"] = nb
    sa = blosc2.asarray(nsa)

    # The expression
    res = sa[f"{field} > 2"].indices(order=order).compute()
    assert res.dtype == np.int64

    # Emulate that expression with NumPy
    if order:
        asort = nsa.argsort(order=order)
        nsa = nsa[asort]
        # nres = np.where(nsa[field] > 2)[0][asort]
    mask = nsa[field] > 2
    nres = np.where(mask)[0]
    if order:
        nres = asort[mask]

    # Check
    np.testing.assert_allclose(res[:], nres)


@pytest.mark.parametrize(
    ("shape", "chunks", "blocks", "order"),
    [
        ((5,), (2,), (1,), "a"),
        ((15,), (2,), (2,), "b"),
        ((100,), (44,), (33,), "b"),
        ((100,), (44,), (33,), None),
    ],
)
def test_sort(shape, chunks, blocks, order):
    na = np.arange(1, shape[0] + 1)
    nb = np.arange(2 * shape[0], shape[0], -1)
    nsa = np.empty(shape, dtype=[("a", np.int32), ("b", np.int32)])
    nsa["a"] = na
    nsa["b"] = nb
    sa = blosc2.asarray(nsa, chunks=chunks, blocks=blocks)

    # The expression
    res = sa["a > 2"].sort(order).compute()

    # Emulate that expression with NumPy
    nres = np.sort(nsa[na > 2], order=order)

    # Check
    np.testing.assert_allclose(res["a"][:], nres["a"])
    np.testing.assert_allclose(res["b"][:], nres["b"])


@pytest.mark.parametrize(
    ("shape", "chunks", "blocks", "order"),
    [
        ((5,), (2,), (1,), "a"),
        ((5,), (2,), (1,), "b"),
        ((10,), (4,), (3,), "b"),
        ((10,), (4,), (3,), None),
    ],
)
def test_sort_indices(shape, chunks, blocks, order):
    na = np.arange(1, shape[0] + 1)
    nb = np.arange(2 * shape[0], shape[0], -1)
    nsa = np.empty(shape, dtype=[("a", np.int32), ("b", np.int32)])
    nsa["a"] = na
    nsa["b"] = nb
    sa = blosc2.asarray(nsa, chunks=chunks, blocks=blocks)

    # The expression
    res = sa["a > 2"].indices(order).compute()

    # Emulate that expression with NumPy
    mask = nsa["a"] > 2
    if order:
        sorted_indices = np.argsort(nsa[order][mask])
    else:
        sorted_indices = np.argsort(nsa[mask])
    nres = np.where(mask)[0][sorted_indices]

    # Check
    np.testing.assert_allclose(res[:], nres)
    np.testing.assert_allclose(res[:], nres)


@pytest.mark.parametrize(
    ("shape", "chunks", "blocks"),
    [
        ((5,), (2,), (1,)),
        ((5,), (5,), (1,)),
        ((10,), (4,), (3,)),
    ],
)
def test_iter(shape, chunks, blocks):
    na = np.arange(int(np.prod(shape)), dtype=np.int32).reshape(shape)
    nb = np.arange(2 * int(np.prod(shape)), int(np.prod(shape)), -1, dtype=np.int32).reshape(shape)
    nsa = np.empty(shape, dtype=[("a", np.int32), ("b", np.int32)])
    nsa["a"] = na
    nsa["b"] = nb
    sa = blosc2.asarray(nsa, chunks=chunks, blocks=blocks)

    for _i, (a, b) in enumerate(zip(sa, nsa, strict=False)):
        np.testing.assert_equal(a, b)
        assert a.dtype == b.dtype
    assert _i == shape[0] - 1


@pytest.mark.parametrize("reduce_op", ["sum", "mean", "min", "max", "std", "var"])
def test_col_reduction(reduce_op):
    N = 1000
    rng = np.random.default_rng()
    it = ((-x + 1, x - 2, rng.normal()) for x in range(N))
    sa = blosc2.fromiter(it, dtype=[("A", "i4"), ("B", "f4"), ("C", "f8")], shape=(N,), chunks=(N // 2,))

    # The operations
    reduc = getattr(blosc2, reduce_op)
    C = sa.fields["C"]
    s = reduc(C[C > 0])
    s2 = reduc(C["C > 0"])  # string version

    # Check
    nreduc = getattr(np, reduce_op)
    nsa = sa[:]
    nC = nsa["C"]
    ns = nreduc(nC[nC > 0])
    np.testing.assert_allclose(s, ns)
    np.testing.assert_allclose(s2, ns)


def test_fields_indexing():
    N = 1000
    it = ((-x + 1, x - 2, 0.1 * x) for x in range(N))
    sa = blosc2.fromiter(
        it, dtype=[("A", "i4"), ("B", "f4"), ("C", "f8")], shape=(N,), urlpath="sa-1M.b2nd", mode="w"
    )
    expr = sa["(A < B)"]
    A = sa["A"][:]
    B = sa["B"][:]
    C = sa["C"][:]
    temp = sa[:]
    indices = A < B
    idx = np.argmax(indices)

    # Returns less than 10 elements in general
    sliced = expr.compute(slice(0, 10))
    gotitem = expr[:10]
    np.testing.assert_array_equal(sliced[:], gotitem)
    np.testing.assert_array_equal(gotitem, temp[:10][indices[:10]])
    # Actually this makes sense since one can understand this as a request to compute on a portion of operands.
    # If one desires a portion of the result, one should compute the whole expression and then slice it.
    # For a general slice it is quite difficult to simply stop when the desired slice has been obtained. Or
    # to try to optimise chunk computation order.

    # Get first true element
    sliced = expr.compute(idx)
    gotitem = expr[idx]
    np.testing.assert_array_equal(sliced[()], gotitem)
    np.testing.assert_array_equal(gotitem, temp[idx])

    # Should return void arrays here.
    sliced = expr.compute(0)  # typically gives array of zeros
    gotitem = expr[0]  # gives an error
    np.testing.assert_array_equal(sliced[()], gotitem)
    np.testing.assert_array_equal(gotitem, temp[0])

    # Remove file
    blosc2.remove_urlpath("sa-1M.b2nd")
