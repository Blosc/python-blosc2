#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################
import pathlib

import numpy as np
import pytest

import blosc2
from blosc2.lazyexpr import ne_evaluate

pytestmark = pytest.mark.network

NITEMS_SMALL = 1_000
ROOT = "@public"
DIR = "expr/"


def get_arrays(shape, chunks_blocks):
    dtype = np.float64
    nelems = np.prod(shape)
    na1 = np.linspace(0, 10, nelems, dtype=dtype).reshape(shape)
    urlpath = f"ds-0-10-linspace-{dtype.__name__}-{chunks_blocks}-a1-{shape}d.b2nd"
    path = pathlib.Path(f"{ROOT}/{DIR + urlpath}").as_posix()
    a1 = blosc2.C2Array(path)
    urlpath = f"ds-0-10-linspace-{dtype.__name__}-{chunks_blocks}-a2-{shape}d.b2nd"
    path = pathlib.Path(f"{ROOT}/{DIR + urlpath}").as_posix()
    a2 = blosc2.C2Array(path)
    # Let other operands be local, on-disk NDArray copies
    urlpath = f"ds-0-10-linspace-{dtype.__name__}-{chunks_blocks}-a3-{shape}d.b2nd"
    a3 = blosc2.asarray(a2, urlpath=urlpath, mode="w")
    urlpath = f"ds-0-10-linspace-{dtype.__name__}-{chunks_blocks}-a4-{shape}d.b2nd"
    a4 = a3.copy(urlpath=urlpath, mode="w")
    assert isinstance(a1, blosc2.C2Array)
    assert isinstance(a2, blosc2.C2Array)
    assert isinstance(a3, blosc2.NDArray)
    assert isinstance(a4, blosc2.NDArray)
    return a1, a2, a3, a4, na1, np.copy(na1), np.copy(na1), np.copy(na1)


@pytest.mark.parametrize(
    "chunks_blocks",
    [
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    ],
)
def test_simple(chunks_blocks, cat2_context):
    shape = (60, 60)
    a1, a2, a3, a4, na1, na2, na3, na4 = get_arrays(shape, chunks_blocks)

    # Slice
    sl = slice(10)
    expr = a1 + a3
    nres = ne_evaluate("na1 + na3")
    res = expr.compute(item=sl)
    np.testing.assert_allclose(res[:], nres[sl])

    # All
    res = expr.compute()
    np.testing.assert_allclose(res[:], nres)


def test_simple_getitem(cat2_context):
    shape = (NITEMS_SMALL,)
    chunks_blocks = "default"
    a1, a2, a3, a4, na1, na2, na3, na4 = get_arrays(shape, chunks_blocks)
    expr = a1 + a2 - a3 * a4
    nres = ne_evaluate("na1 + na2 - na3 * na4")

    # slice
    sl = slice(10)
    res = expr[sl]
    np.testing.assert_allclose(res, nres[sl])
    # all
    res = expr[:]
    np.testing.assert_allclose(res, nres)


# Add more test functions to test different aspects of the code
@pytest.mark.parametrize(
    "chunks_blocks",
    [
        (True, False),
        (False, False),
    ],
)
def test_ixxx(chunks_blocks, cat2_context):
    shape = (60, 60)
    a1, a2, a3, a4, na1, na2, na3, na4 = get_arrays(shape, chunks_blocks)
    expr = a1**3 + a2**2 + a3**3 - a4 + 3
    expr += 5  # __iadd__
    expr /= 7  # __itruediv__
    expr **= 2.3  # __ipow__
    res = expr.compute()
    nres = ne_evaluate("(((na1 ** 3 + na2 ** 2 + na3 ** 3 - na4 + 3) + 5) / 7) ** 2.3")
    np.testing.assert_allclose(res[:], nres)


def test_complex(cat2_context):
    shape = (NITEMS_SMALL,)
    chunks_blocks = "default"
    a1, a2, a3, a4, na1, na2, na3, na4 = get_arrays(shape, chunks_blocks)
    expr = blosc2.tan(a1) * blosc2.sin(a2) + (blosc2.sqrt(a4) * 2)
    expr += 2
    nres = ne_evaluate("tan(na1) * sin(na2) + (sqrt(na4) * 2) + 2")
    # eval
    res = expr.compute()
    np.testing.assert_allclose(res[:], nres)
    # __getitem__
    res = expr[:]
    np.testing.assert_allclose(res, nres)
    # slice
    sl = slice(10)
    res = expr[sl]
    np.testing.assert_allclose(res, nres[sl])


# Test expr with remote & local operands
@pytest.mark.parametrize(
    "chunks_blocks",
    [
        pytest.param((True, True), marks=pytest.mark.heavy),
        pytest.param((True, False), marks=pytest.mark.heavy),
        pytest.param((False, True), marks=pytest.mark.heavy),
        (False, False),
    ],
)
def test_mix_operands(chunks_blocks, cat2_context):
    shape = (60, 60)
    a1, a2, a3, a4, na1, na2, na3, na4 = get_arrays(shape, chunks_blocks)
    b1 = blosc2.asarray(na1, chunks=a1.chunks, blocks=a1.blocks)
    b3 = blosc2.asarray(na3, chunks=a3.chunks, blocks=a3.blocks)

    expr = a1 + b1
    nres = ne_evaluate("na1 + na1")
    np.testing.assert_allclose(expr[:], nres)
    np.testing.assert_allclose(expr.compute()[:], nres)

    expr = a1 + b3
    nres = ne_evaluate("na1 + na3")
    np.testing.assert_allclose(expr[:], nres)
    np.testing.assert_allclose(expr.compute()[:], nres)

    expr = a1 + b1 + a2 + b3
    nres = ne_evaluate("na1 + na1 + na2 + na3")
    np.testing.assert_allclose(expr[:], nres)
    np.testing.assert_allclose(expr.compute()[:], nres)

    expr = a1 + a2 + b1 + b3
    nres = ne_evaluate("na1 + na2 + na1 + na3")
    np.testing.assert_allclose(expr[:], nres)
    np.testing.assert_allclose(expr.compute()[:], nres)

    # TODO: fix this
    # expr = a1 + na1 * b3
    # print(type(expr))
    # print("expression: ", expr.expression)
    # nres = ne_evaluate("na1 + na1 * na3")
    # np.testing.assert_allclose(expr[:], nres)
    # np.testing.assert_allclose(expr.compute()[:], nres)


# Tests related with save method
def test_save(cat2_context):
    shape = (60, 60)
    tol = 1e-17
    a1, a2, a3, a4, na1, na2, na3, na4 = get_arrays(shape, (False, True))

    expr = a1 * a2 + a3 - a4 * 3
    nres = ne_evaluate("na1 * na2 + na3 - na4 * 3")

    res = expr.compute()
    assert res.dtype == np.float64
    np.testing.assert_allclose(res[:], nres, rtol=tol, atol=tol)

    urlpath = "expr.b2nd"
    expr.save(urlpath=urlpath, mode="w")
    ops = [a1, a2, a3, a4]
    for op in ops:
        del op
    del expr
    expr = blosc2.open(urlpath)
    res = expr.compute()
    assert res.dtype == np.float64
    np.testing.assert_allclose(res[:], nres, rtol=tol, atol=tol)
    # Test getitem
    np.testing.assert_allclose(expr[:], nres, rtol=tol, atol=tol)

    blosc2.remove_urlpath(urlpath)


@pytest.fixture(
    params=[
        ((2, 5), (5,)),
        pytest.param(((2, 1), (5,)), marks=pytest.mark.heavy),
        pytest.param(((2, 5, 3), (5, 1)), marks=pytest.mark.heavy),
        ((2, 1, 3), (5, 3)),
        pytest.param(((2, 5, 3, 2), (5, 3, 1)), marks=pytest.mark.heavy),
        ((2, 5, 3, 2), (5, 1, 2)),
        pytest.param(((2, 5, 3, 2, 2), (5, 3, 2, 2)), marks=pytest.mark.heavy),
    ]
)
def broadcast_shape(request):
    return request.param


@pytest.fixture
def broadcast_fixture(broadcast_shape, cat2_context):
    shape1, shape2 = broadcast_shape
    dtype = np.float64
    na1 = np.linspace(0, 1, np.prod(shape1), dtype=dtype).reshape(shape1)
    na2 = np.linspace(1, 2, np.prod(shape2), dtype=dtype).reshape(shape2)
    urlpath = f"ds-0-1-linspace-{dtype.__name__}-b1-{shape1}d.b2nd"
    path = pathlib.Path(f"{ROOT}/{DIR + urlpath}").as_posix()
    b1 = blosc2.C2Array(path)
    urlpath = f"ds-1-2-linspace-{dtype.__name__}-b2-{shape2}d.b2nd"
    path = pathlib.Path(f"{ROOT}/{DIR + urlpath}").as_posix()
    b2 = blosc2.C2Array(path)

    return b1, b2, na1, na2


def test_broadcasting(broadcast_fixture):
    a1, a2, na1, na2 = broadcast_fixture
    expr1 = a1 + a2
    assert expr1.shape == np.broadcast_shapes(a1.shape, a2.shape)
    expr2 = a1 * a2 + 1
    assert expr2.shape == np.broadcast_shapes(a1.shape, a2.shape)
    expr = expr1 - expr2
    assert expr.shape == np.broadcast_shapes(a1.shape, a2.shape)
    nres = ne_evaluate("na1 + na2 - (na1 * na2 + 1)")
    res = expr.compute()
    np.testing.assert_allclose(res[:], nres)
    res = expr[:]
    np.testing.assert_allclose(res, nres)
