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
        (np.float64, np.float64),
        (np.int8, np.int16),
        (np.int8, np.float64),
    ]
)
def dtype_fixture(request):
    return request.param


@pytest.fixture(params=[(NITEMS_SMALL,), (NITEMS,), (NITEMS // 100, 100)])
def shape_fixture(request):
    return request.param


# params: (same_chunks, same_blocks)
@pytest.fixture(params=[(True, True), (True, False), (False, True), (False, False)])
def chunks_blocks_fixture(request):
    return request.param


@pytest.fixture
def array_fixture(dtype_fixture, shape_fixture, chunks_blocks_fixture):
    nelems = np.prod(shape_fixture)
    dt1, dt2 = dtype_fixture
    na1_ = np.linspace(0, 10, nelems, dtype=dt1).reshape(shape_fixture)
    na2_ = np.linspace(10, 20, nelems, dtype=dt2).reshape(shape_fixture)
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
    nres = ne.evaluate("na1 + na2 - na3 * na4")
    sl = slice(100)
    res = expr[sl]
    np.testing.assert_allclose(res, nres[sl])


# Add more test functions to test different aspects of the code
def test_simple_expression(array_fixture):
    sa1, sa2, nsa1, nsa2, a1, a2, a3, a4, na1, na2, na3, na4 = array_fixture
    expr = a1 + a2 - a3 * a4
    nres = ne.evaluate("na1 + na2 - na3 * na4")
    res = expr.eval()
    np.testing.assert_allclose(res[:], nres)


def test_iXXX(array_fixture):
    sa1, sa2, nsa1, nsa2, a1, a2, a3, a4, na1, na2, na3, na4 = array_fixture
    expr = a1**3 + a2**2 + a3**3 - a4 + 3
    expr += 5  # __iadd__
    expr -= 15  # __isub__
    expr *= 2  # __imul__
    expr /= 7  # __itruediv__
    expr **= 2.3  # __ipow__
    res = expr.eval()
    nres = ne.evaluate("(((((na1 ** 3 + na2 ** 2 + na3 ** 3 - na4 + 3) + 5) - 15) * 2) / 7) ** 2.3")
    np.testing.assert_allclose(res[:], nres)


def test_complex_evaluate(array_fixture):
    sa1, sa2, nsa1, nsa2, a1, a2, a3, a4, na1, na2, na3, na4 = array_fixture
    expr = blosc2.tan(a1) * (blosc2.sin(a2) * blosc2.sin(a2) + blosc2.cos(a3)) + (blosc2.sqrt(a4) * 2)
    expr += 2
    nres = ne.evaluate("tan(na1) * (sin(na2) * sin(na2) + cos(na3)) + (sqrt(na4) * 2) + 2")
    res = expr.eval()
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
    # Testing too much is a waste of time; just keep one reduction here
    # np.testing.assert_allclose(expr.sum()[()], nres.sum())
    np.testing.assert_allclose(expr.mean()[()], nres.mean())
    # np.testing.assert_allclose(expr.min()[()], nres.min())
    # np.testing.assert_allclose(expr.max()[()], nres.max())
    # np.testing.assert_allclose(expr.std()[()], nres.std())


def test_mixed_operands(array_fixture):
    sa1, sa2, nsa1, nsa2, a1, a2, a3, a4, na1, na2, na3, na4 = array_fixture
    # All a1, a2, a3 and a4 are NDFields
    a3 = blosc2.asarray(na3)  # this is a NDArray now
    assert not isinstance(a3, blosc2.NDField)
    a4 = na4  # this is a NumPy array now
    assert not isinstance(a4, blosc2.NDField)
    expr = a1 + a2 - a3 * a4
    nres = ne.evaluate("na1 + na2 - na3 * na4")
    res = expr.eval()
    np.testing.assert_allclose(res[:], nres)


# Test expressions with where()
def test_where(array_fixture):
    sa1, sa2, nsa1, nsa2, a1, a2, a3, a4, na1, na2, na3, na4 = array_fixture
    expr = a1**2 + a2**2 > 2 * a1 * a2 + 1
    # Test with eval
    res = expr.where(0, 1).eval()
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
    res = expr.where(a1).eval()
    nres = na1[na1**2 + na2**2 > 2 * na1 * na2 + 1]
    np.testing.assert_allclose(res[:], nres)
    # Test with getitem
    sl = slice(100)
    res = expr.where(a1)[sl]
    np.testing.assert_allclose(res, nres[sl])


# Test where indirectly via a condition in getitem in a NDArray
def test_where_getitem(array_fixture):
    sa1, sa2, nsa1, nsa2, a1, a2, a3, a4, na1, na2, na3, na4 = array_fixture

    # Test with eval
    res = sa1[a1**2 + a2**2 > 2 * a1 * a2 + 1].eval()
    nres = nsa1[na1**2 + na2**2 > 2 * na1 * na2 + 1]
    np.testing.assert_allclose(res["a"], nres["a"])
    np.testing.assert_allclose(res["b"], nres["b"])
    # string version
    res = sa1["a**2 + b**2 > 2 * a * b + 1"].eval()
    np.testing.assert_allclose(res["a"], nres["a"])
    np.testing.assert_allclose(res["b"], nres["b"])

    # Test with getitem
    sl = slice(100)
    res = sa1[a1**2 + a2**2 > 2 * a1 * a2 + 1][sl]
    np.testing.assert_allclose(res["a"], nres["a"][sl])
    np.testing.assert_allclose(res["b"], nres["b"][sl])
    # string version
    res = sa1["a**2 + b**2 > 2 * a * b + 1"][sl]
    np.testing.assert_allclose(res["a"], nres["a"][sl])
    np.testing.assert_allclose(res["b"], nres["b"][sl])


# Test where indirectly via a condition in getitem in a NDField
def test_where_getitem_field(array_fixture):
    sa1, sa2, nsa1, nsa2, a1, a2, a3, a4, na1, na2, na3, na4 = array_fixture
    # Test with eval
    res = a1[a1**2 + a2**2 > 2 * a1 * a2 + 1].eval()
    nres = na1[na1**2 + na2**2 > 2 * na1 * na2 + 1]
    np.testing.assert_allclose(res[:], nres)
    # Test with getitem
    sl = slice(100)
    res = a1[a1**2 + a2**2 > 2 * a1 * a2 + 1][sl]
    np.testing.assert_allclose(res, nres[sl])


# Test where combined with a reduction
def test_where_reduction(array_fixture):
    sa1, sa2, nsa1, nsa2, a1, a2, a3, a4, na1, na2, na3, na4 = array_fixture
    expr = a1**2 + a2**2 > 2 * a1 * a2 + 1
    axis = None if sa1.ndim == 1 else 1
    res = expr.where(0, 1).sum(axis=axis)
    nres = ne.evaluate("where(na1**2 + na2**2 > 2 * na1 * na2 + 1, 0, 1)").sum(axis=axis)
    np.testing.assert_allclose(res, nres)


# This is a more complex case with where() calls combined with reductions,
# broadcasting, reusing the result in another expression and other
# funny stuff that is not working yet.
def test_where_fusion(array_fixture):
    sa1, sa2, nsa1, nsa2, a1, a2, a3, a4, na1, na2, na3, na4 = array_fixture
    expr = a1**2 + a2**2 > 2 * a1 * a2 + 1
    npexpr = na1**2 + na2**2 > 2 * na1 * na2 + 1

    # Two where() calls
    res = expr.where(0, 1) + expr.where(0, 1)
    nres = np.where(npexpr, 0, 1) + np.where(npexpr, 0, 1)
    np.testing.assert_allclose(res[:], nres)

    # Two where() calls with a reduction (and using broadcasting)
    axis = None if sa1.ndim == 1 else 1
    res = expr.where(0, 1) + expr.where(0, 1).sum(axis=axis)
    nres = np.where(npexpr, 0, 1) + np.where(npexpr, 0, 1).sum(axis=axis)
    np.testing.assert_allclose(res[:], nres)

    # Reuse the result in another expression
    res = expr.where(0, 1) + res.sum()
    nres = np.where(npexpr, 0, 1) + nres.sum()
    np.testing.assert_allclose(res[:], nres)

    # Reuse the result in another expression twice
    res = 2 * res + 4 * res
    nres = 2 * nres + 4 * nres
    print(res[:], nres)
    np.testing.assert_allclose(res[:], nres)

    # TODO: this is not working yet
    # Reuse the result in another expression twice II
    # res = 2 * res + blosc2.sqrt(res)
    # nres = 2 * nres + nres.sqrt()
    # np.testing.assert_allclose(res[:], nres)

    # TODO: this is not working yet
    # Reuse the result in another expression twice III
    # res = expr.where(0, 1) + res
    # nres = np.where(npexpr, 0, 1) + nres
    # np.testing.assert_allclose(res[:], nres)
