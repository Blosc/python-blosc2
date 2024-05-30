#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################
import pathlib

import httpx
import numexpr as ne
import numpy as np
import pytest

import blosc2

NITEMS_SMALL = 1_000
NITEMS = 50_000

SUB_URL = 'http://localhost:8002/'
ROOT = 'foo'
DIR = 'operands/'
# TODO: Uncomment this when all needed changes are merged and
#  the server runs with the latest version
# SUB_URL = 'https://demo.caterva2.net/'
# ROOT = 'b2tests'
# DIR = 'expr/'

# resp = httpx.post(f'{SUB_URL}auth/jwt/login',
#                   data=dict(username='user@example.com', password='foobar'))
# resp.raise_for_status()
# AUTH_COOKIE = '='.join(list(resp.cookies.items())[0])


@pytest.fixture(params=[
    np.float64
])
def dtype_fixture(request):
    return request.param


@pytest.fixture(params=[
    (NITEMS_SMALL,),
    (NITEMS,),
    (NITEMS // 100, 100)
])
def shape_fixture(request):
    return request.param


# params: (same_chunks, same_blocks)
@pytest.fixture(params=[(True, True),
                        (True, False),
                        (False, True),
                        (False, False)
                        ])
def chunks_blocks_fixture(request):
    return request.param


@pytest.fixture(params=[
    None,
    # AUTH_COOKIE,
])
def auth_cookie(request):
    return request.param


# @pytest.fixture
# def array_fixture(dtype_fixture, shape_fixture, chunks_blocks_fixture):
#     nelems = np.prod(shape_fixture)
#     na1 = np.linspace(0, 10, nelems, dtype=dtype_fixture).reshape(shape_fixture)
#
#     chunks = chunks1 = blocks = blocks1 = None  # silence linter
#     same_chunks_blocks = chunks_blocks_fixture[0] and chunks_blocks_fixture[1]
#     same_chunks = chunks_blocks_fixture[0]
#     same_blocks = chunks_blocks_fixture[1]
#     if same_chunks_blocks:
#         # For full generality, use partitions with padding
#         chunks = chunks1 = [c // 11 for c in na1.shape]
#         blocks = blocks1 = [c // 71 for c in na1.shape]
#     elif same_chunks:
#         chunks = [c // 11 for c in na1.shape]
#         blocks = [c // 71 for c in na1.shape]
#         chunks1 = [c // 11 for c in na1.shape]
#         blocks1 = [c // 51 for c in na1.shape]
#     elif same_blocks:
#         chunks = [c // 11 for c in na1.shape]
#         blocks = [c // 71 for c in na1.shape]
#         chunks1 = [c // 23 for c in na1.shape]
#         blocks1 = [c // 71 for c in na1.shape]
#     else:
#         # Different chunks and blocks
#         chunks = [c // 17 for c in na1.shape]
#         blocks = [c // 19 for c in na1.shape]
#         chunks1 = [c // 23 for c in na1.shape]
#         blocks1 = [c // 29 for c in na1.shape]
#
#     urlpath = f'ds-0-10-linspace-{dtype_fixture.__name__}-{chunks_blocks_fixture}-a1-{shape_fixture}d.b2nd'
#     a1 = blosc2.asarray(na1, chunks=chunks, blocks=blocks, urlpath=urlpath, mode="w")
#     na2 = np.copy(na1)
#     urlpath = f'ds-0-10-linspace-{dtype_fixture.__name__}-{chunks_blocks_fixture}-a2-{shape_fixture}d.b2nd'
#     a2 = blosc2.asarray(na2, chunks=chunks, blocks=blocks, urlpath=urlpath, mode="w")
#     na3 = np.copy(na1)
#     # Let other operands have chunks1 and blocks1
#     urlpath = f'ds-0-10-linspace-{dtype_fixture.__name__}-{chunks_blocks_fixture}-a3-{shape_fixture}d.b2nd'
#     a3 = blosc2.asarray(na3, chunks=chunks1, blocks=blocks1, urlpath=urlpath, mode="w")
#     na4 = np.copy(na1)
#     urlpath = f'ds-0-10-linspace-{dtype_fixture.__name__}-{chunks_blocks_fixture}-a4-{shape_fixture}d.b2nd'
#     a4 = blosc2.asarray(na4, chunks=chunks1, blocks=blocks1, urlpath=urlpath, mode="w")
#
#     na = np.linspace(-1, 1, nelems, dtype=dtype_fixture).reshape(shape_fixture)
#     urlpath = f'ds--1-1-linspace-{dtype_fixture.__name__}-a5-{shape_fixture}d.b2nd'
#     _ = blosc2.asarray(na, urlpath=urlpath, mode="w")
#
#     na = np.array([b"abc", b"def", b"aterr", b"oot", b"zu", b"ab c"])
#     urlpath = f'ds-str-a6.b2nd'
#     _ = blosc2.asarray(na, urlpath=urlpath, mode="w")
#     na = np.array([b"abc", b"ab c", b" abc", b" abc ", b"\tabc", b"c h"])
#     urlpath = f'ds-str-a7.b2nd'
#     _ = blosc2.asarray(na, urlpath=urlpath, mode="w")
#
#     return a1, a2, a3, a4, na1, np.copy(na1), np.copy(na1), np.copy(na1)


@pytest.fixture
def array_fixture(dtype_fixture, shape_fixture, chunks_blocks_fixture, auth_cookie):
    nelems = np.prod(shape_fixture)
    na1 = np.linspace(0, 10, nelems, dtype=dtype_fixture).reshape(shape_fixture)
    urlpath = f'ds-0-10-linspace-{dtype_fixture.__name__}-{chunks_blocks_fixture}-a1-{shape_fixture}d.b2nd'
    path = pathlib.Path(f'{ROOT}/{DIR + urlpath}')
    a1 = blosc2.C2Array(path, sub_url=SUB_URL, auth_cookie=auth_cookie)
    urlpath = f'ds-0-10-linspace-{dtype_fixture.__name__}-{chunks_blocks_fixture}-a2-{shape_fixture}d.b2nd'
    path = pathlib.Path(f'{ROOT}/{DIR + urlpath}')
    a2 = blosc2.C2Array(path, sub_url=SUB_URL, auth_cookie=auth_cookie)
    # Let other operands have chunks1 and blocks1
    urlpath = f'ds-0-10-linspace-{dtype_fixture.__name__}-{chunks_blocks_fixture}-a3-{shape_fixture}d.b2nd'
    path = pathlib.Path(f'{ROOT}/{DIR + urlpath}')
    a3 = blosc2.C2Array(path, sub_url=SUB_URL, auth_cookie=auth_cookie)
    urlpath = f'ds-0-10-linspace-{dtype_fixture.__name__}-{chunks_blocks_fixture}-a4-{shape_fixture}d.b2nd'
    path = pathlib.Path(f'{ROOT}/{DIR + urlpath}')
    a4 = blosc2.C2Array(path, sub_url=SUB_URL, auth_cookie=auth_cookie)
    assert isinstance(a1, blosc2.C2Array)
    assert isinstance(a2, blosc2.C2Array)
    assert isinstance(a3, blosc2.C2Array)
    assert isinstance(a4, blosc2.C2Array)
    return a1, a2, a3, a4, na1, np.copy(na1), np.copy(na1), np.copy(na1)


def test_simple(array_fixture):
    a1, a2, a3, a4, na1, na2, na3, na4 = array_fixture

    expr = a1 + a1
    nres = ne.evaluate("na1 + na1")
    np.testing.assert_allclose(expr[:], nres)
    np.testing.assert_allclose(expr.eval()[:], nres)

    expr = a1 + a3
    nres = ne.evaluate("na1 + na3")
    np.testing.assert_allclose(expr[:], nres)
    np.testing.assert_allclose(expr.eval()[:], nres)


def test_simple_getitem(array_fixture):
    a1, a2, a3, a4, na1, na2, na3, na4 = array_fixture
    expr = a1 + a2 - a3 * a4
    nres = ne.evaluate("na1 + na2 - na3 * na4")
    # slice
    sl = slice(100)
    res = expr[sl]
    np.testing.assert_allclose(res, nres[sl])
    # eval
    res = expr.eval()
    np.testing.assert_allclose(res[:], nres)


# Add more test functions to test different aspects of the code
def test_ixxx(array_fixture):
    a1, a2, a3, a4, na1, na2, na3, na4 = array_fixture
    expr = a1 ** 3 + a2 ** 2 + a3 ** 3 - a4 + 3
    expr += 5  # __iadd__
    expr -= 15  # __isub__
    expr *= 2  # __imul__
    expr /= 7  # __itruediv__
    expr **= 2.3  # __ipow__
    res = expr.eval()
    nres = ne.evaluate("(((((na1 ** 3 + na2 ** 2 + na3 ** 3 - na4 + 3) + 5) - 15) * 2) / 7) ** 2.3")
    np.testing.assert_allclose(res[:], nres)


def test_complex(array_fixture):
    a1, a2, a3, a4, na1, na2, na3, na4 = array_fixture
    expr = blosc2.tan(a1) * (blosc2.sin(a2) * blosc2.sin(a2) + blosc2.cos(a3)) + (blosc2.sqrt(a4) * 2)
    expr += 2
    nres = ne.evaluate("tan(na1) * (sin(na2) * sin(na2) + cos(na3)) + (sqrt(na4) * 2) + 2")
    # eval
    res = expr.eval()
    np.testing.assert_allclose(res[:], nres)
    # __getitem__
    res = expr[:]
    np.testing.assert_allclose(res, nres)
    # slice
    sl = slice(100)
    res = expr[sl]
    np.testing.assert_allclose(res, nres[sl])


def test_expression_with_constants(array_fixture):
    a1, a2, a3, a4, na1, na2, na3, na4 = array_fixture
    expr = a1 + 2 - a3 * 3.14
    nres = ne.evaluate("na1 + 2 - na3 * 3.14")
    np.testing.assert_allclose(expr[:], nres)

    # Test with operands with same chunks and blocks
    expr = a1 + 2 - a2 * 3.14
    nres = ne.evaluate("na1 + 2 - na2 * 3.14")
    np.testing.assert_allclose(expr[:], nres)


@pytest.mark.parametrize("compare_expressions", [True, False])
@pytest.mark.parametrize("comparison_operator", ["==", "!=", ">=", ">", "<=", "<"])
def test_comparison_operators(dtype_fixture, compare_expressions, comparison_operator, auth_cookie):
    shape_fixture = (NITEMS_SMALL,)
    nelems = np.prod(shape_fixture)
    na1 = np.linspace(0, 10, nelems, dtype=dtype_fixture).reshape(shape_fixture)
    na2 = np.copy(na1)  # noqa: F841
    urlpath = f'ds-0-10-linspace-{dtype_fixture.__name__}-(True, False)-a1-{shape_fixture}d.b2nd'
    path = pathlib.Path(f'{ROOT}/{DIR + urlpath}')
    a1 = blosc2.C2Array(path, sub_url=SUB_URL, auth_cookie=auth_cookie)
    urlpath = f'ds-0-10-linspace-{dtype_fixture.__name__}-(True, False)-a2-{shape_fixture}d.b2nd'
    path = pathlib.Path(f'{ROOT}/{DIR + urlpath}')
    a2 = blosc2.C2Array(path, sub_url=SUB_URL, auth_cookie=auth_cookie)
    # Construct the lazy expression
    if compare_expressions:
        expr = eval(f"a1 ** 2 {comparison_operator} (a1 + a2)", {"a1": a1, "a2": a2})
        expr_string = f"na1 ** 2 {comparison_operator} (na1 + na2)"
    else:
        expr = eval(f"a1 {comparison_operator} a2", {"a1": a1, "a2": a2})
        expr_string = f"na1 {comparison_operator} na2"
    res_lazyexpr = expr.eval()
    # Evaluate using NumExpr
    res_numexpr = ne.evaluate(expr_string)
    # Compare the results
    np.testing.assert_allclose(res_lazyexpr[:], res_numexpr)


@pytest.mark.parametrize(
    "function",
    [
        "sin",
        "cos",
        "tan",
        "sqrt",
        "sinh",
        "cosh",
        "tanh",
        "arcsin",
        "arccos",
        "arctan",
        "arcsinh",
        "arccosh",
        "arctanh",
        "exp",
        "expm1",
        "log",
        "log10",
        "log1p",
        "conj",
        "real",
        "imag",
    ],
)
def test_functions(function, dtype_fixture, shape_fixture, auth_cookie):
    nelems = np.prod(shape_fixture)
    na1 = np.linspace(0, 10, nelems, dtype=dtype_fixture).reshape(shape_fixture)
    urlpath = f'ds-0-10-linspace-{dtype_fixture.__name__}-{(True, False)}-a1-{shape_fixture}d.b2nd'
    path = pathlib.Path(f'{ROOT}/{DIR + urlpath}')
    a1 = blosc2.C2Array(path, sub_url=SUB_URL, auth_cookie=auth_cookie)
    # Construct the lazy expression based on the function name
    expr = blosc2.LazyExpr(new_op=(a1, function, None))
    res_lazyexpr = expr.eval()
    # Evaluate using NumExpr
    expr_string = f"{function}(na1)"
    res_numexpr = ne.evaluate(expr_string)
    # Compare the results
    np.testing.assert_allclose(res_lazyexpr[:], res_numexpr)


# TODO: support save


def test_abs(shape_fixture, dtype_fixture, auth_cookie):
    nelems = np.prod(shape_fixture)
    na = np.linspace(-1, 1, nelems, dtype=dtype_fixture).reshape(shape_fixture)
    urlpath = f'ds--1-1-linspace-{dtype_fixture.__name__}-a5-{shape_fixture}d.b2nd'
    path = pathlib.Path(f'{ROOT}/{DIR + urlpath}')
    a = blosc2.C2Array(path, sub_url=SUB_URL, auth_cookie=auth_cookie)
    expr = blosc2.LazyExpr(new_op=(a, "abs", None))
    res_lazyexpr = expr.eval()
    res_np = np.abs(na)
    np.testing.assert_allclose(res_lazyexpr[:], res_np)


@pytest.mark.parametrize("values", [("NDArray", "str"), ("NDArray", "NDArray"), ("str", "NDArray")])
def test_contains(values, auth_cookie):
    # Unpack the value fixture
    value1, value2 = values
    if value1 == "NDArray":
        urlpath = f'ds-str-a6.b2nd'
        path = pathlib.Path(f'{ROOT}/{DIR + urlpath}')
        a1_blosc = blosc2.C2Array(path, sub_url=SUB_URL, auth_cookie=auth_cookie)
        a1 = a1_blosc[:]
        if value2 == "str":  # ("NDArray", "str")
            value2 = b"test abc here"
            # Construct the lazy expression
            expr_lazy = blosc2.LazyExpr(new_op=(a1_blosc, "contains", value2))
            # Evaluate using NumExpr
            expr_numexpr = f"{'contains'}(a1, value2)"
            res_numexpr = ne.evaluate(expr_numexpr)
        else:  # ("NDArray", "NDArray")
            urlpath = f'ds-str-a7.b2nd'
            path = pathlib.Path(f'{ROOT}/{DIR + urlpath}')
            a2_blosc = blosc2.C2Array(path, sub_url=SUB_URL, auth_cookie=auth_cookie)
            a2 = a2_blosc[:]
            # Construct the lazy expression
            expr_lazy = blosc2.LazyExpr(new_op=(a1_blosc, "contains", a2_blosc))
            # Evaluate using NumExpr
            res_numexpr = ne.evaluate("contains(a2, a1)")
    else:  # ("str", "NDArray")
        value1 = b"abc"
        urlpath = f'ds-str-a6.b2nd'
        path = pathlib.Path(f'{ROOT}/{DIR + urlpath}')
        a2_blosc = blosc2.C2Array(path, sub_url=SUB_URL, auth_cookie=auth_cookie)
        a2 = a2_blosc[:]
        # Construct the lazy expression
        expr_lazy = blosc2.LazyExpr(new_op=(value1, "contains", a2_blosc))
        # Evaluate using NumExpr
        res_numexpr = ne.evaluate("contains(value1, a2)")
    res_lazyexpr = expr_lazy.eval()
    # Compare the results
    np.testing.assert_array_equal(res_lazyexpr[:], res_numexpr)


def test_negate(dtype_fixture, shape_fixture, auth_cookie):
    urlpath = f'ds--1-1-linspace-{dtype_fixture.__name__}-a5-{shape_fixture}d.b2nd'
    path = pathlib.Path(f'{ROOT}/{DIR + urlpath}')
    a = blosc2.C2Array(path, sub_url=SUB_URL, auth_cookie=auth_cookie)
    na = a[:]

    # Test with a single NDArray
    expr = -a
    res_lazyexpr = expr.eval()
    res_np = -na
    np.testing.assert_allclose(res_lazyexpr[:], res_np)

    # Test with a proper expression
    expr = -(a + 2)
    res_lazyexpr = expr.eval()
    res_np = -(na + 2)
    np.testing.assert_allclose(res_lazyexpr[:], res_np)


# Test expr with remote & local operands
def test_mix_operands(array_fixture):
    a1, a2, a3, a4, na1, na2, na3, na4 = array_fixture
    b1 = blosc2.asarray(na1, chunks=a1.chunks, blocks=a1.blocks)
    b3 = blosc2.asarray(na3, chunks=a3.chunks, blocks=a3.blocks)

    expr = a1 + b1
    nres = ne.evaluate("na1 + na1")
    np.testing.assert_allclose(expr[:], nres)
    np.testing.assert_allclose(expr.eval()[:], nres)

    expr = a1 + b3
    nres = ne.evaluate("na1 + na3")
    np.testing.assert_allclose(expr[:], nres)
    np.testing.assert_allclose(expr.eval()[:], nres)

    expr = a1 + b1 + a2 + b3
    nres = ne.evaluate("na1 + na1 + na2 + na3")
    np.testing.assert_allclose(expr[:], nres)
    np.testing.assert_allclose(expr.eval()[:], nres)

    expr = a1 + a2 + b1 + b3
    nres = ne.evaluate("na1 + na2 + na1 + na3")
    np.testing.assert_allclose(expr[:], nres)
    np.testing.assert_allclose(expr.eval()[:], nres)

    # TODO: fix this
    # expr = a1 + na1 * b3
    # print(type(expr))
    # print("expression: ", expr.expression)
    # nres = ne.evaluate("na1 + na1 * na3")
    # np.testing.assert_allclose(expr[:], nres)
    # np.testing.assert_allclose(expr.eval()[:], nres)


# Tests related with save method
def test_save(dtype_fixture, shape_fixture, auth_cookie):
    tol = 1e-17
    chunks_blocks_fixture = (False, False)
    nelems = np.prod(shape_fixture)
    na1 = np.linspace(0, 10, nelems, dtype=dtype_fixture).reshape(shape_fixture)
    urlpath = f'ds-0-10-linspace-{dtype_fixture.__name__}-{chunks_blocks_fixture}-a1-{shape_fixture}d.b2nd'
    path = pathlib.Path(f'{ROOT}/{DIR + urlpath}')
    a1 = blosc2.C2Array(path, sub_url=SUB_URL, auth_cookie=auth_cookie)
    urlpath = f'ds-0-10-linspace-{dtype_fixture.__name__}-{chunks_blocks_fixture}-a2-{shape_fixture}d.b2nd'
    path = pathlib.Path(f'{ROOT}/{DIR + urlpath}')
    a2 = blosc2.C2Array(path, sub_url=SUB_URL, auth_cookie=auth_cookie)
    # Let other operands have chunks1 and blocks1
    urlpath = f'ds-0-10-linspace-{dtype_fixture.__name__}-{chunks_blocks_fixture}-a3-{shape_fixture}d.b2nd'
    path = pathlib.Path(f'{ROOT}/{DIR + urlpath}')
    a3 = blosc2.C2Array(path, sub_url=SUB_URL, auth_cookie=auth_cookie)
    urlpath = f'ds-0-10-linspace-{dtype_fixture.__name__}-{chunks_blocks_fixture}-a4-{shape_fixture}d.b2nd'
    path = pathlib.Path(f'{ROOT}/{DIR + urlpath}')
    a4 = blosc2.C2Array(path, sub_url=SUB_URL, auth_cookie=auth_cookie)
    na2 = na1.copy()
    na3 = na1.copy()
    na4 = na1.copy()

    expr = a1 * a2 + a3 - a4 * 3
    nres = ne.evaluate('na1 * na2 + na3 - na4 * 3')

    res = expr.eval()
    assert res.dtype == np.float64
    np.testing.assert_allclose(res[:], nres, rtol=tol, atol=tol)

    urlpath = "expr.b2nd"
    expr.save(urlpath=urlpath, mode="w")
    ops = [a1, a2, a3, a4]
    for op in ops:
        del op
    del expr
    expr = blosc2.open(urlpath)
    res = expr.eval()
    assert res.dtype == np.float64
    np.testing.assert_allclose(res[:], nres, rtol=tol, atol=tol)
    # Test getitem
    np.testing.assert_allclose(expr[:], nres, rtol=tol, atol=tol)

    blosc2.remove_urlpath(urlpath)


@pytest.fixture(
    params=[
        ((2, 5), (5,)),
        ((2, 1), (5,)),
        ((2, 5, 3), (5, 3)),
        ((2, 5, 3), (5, 1)),
        ((2, 1, 3), (5, 3)),
        ((2, 5, 3, 2), (5, 3, 2)),
        ((2, 5, 3, 2), (5, 3, 1)),
        ((2, 5, 3, 2), (5, 1, 2)),
        ((2, 1, 3, 2), (5, 3, 2)),
        ((2, 1, 3, 2), (5, 1, 2)),
        ((2, 5, 3, 2, 2), (5, 3, 2, 2)),
        ((20, 20, 20), (20, 20)),
    ]
)
def broadcast_shape(request):
    return request.param


# Test broadcasting

# Generate datasets
# @pytest.fixture
# def broadcast_fixture(dtype_fixture, broadcast_shape):
#     shape1, shape2 = broadcast_shape
#     na1 = np.linspace(0, 1, np.prod(shape1), dtype=dtype_fixture).reshape(shape1)
#     na2 = np.linspace(1, 2, np.prod(shape2), dtype=dtype_fixture).reshape(shape2)
#     urlpath = f'ds-0-1-linspace-{dtype_fixture.__name__}-b1-{shape1}d.b2nd'
#     b1 = blosc2.asarray(na1, urlpath=urlpath, mode="w")
#     urlpath = f'ds-1-2-linspace-{dtype_fixture.__name__}-b2-{shape2}d.b2nd'
#     b2 = blosc2.asarray(na2, urlpath=urlpath, mode="w")
#
#     return b1, b2, na1, na2

@pytest.fixture
def broadcast_fixture(dtype_fixture, broadcast_shape, auth_cookie):
    shape1, shape2 = broadcast_shape
    na1 = np.linspace(0, 1, np.prod(shape1), dtype=dtype_fixture).reshape(shape1)
    na2 = np.linspace(1, 2, np.prod(shape2), dtype=dtype_fixture).reshape(shape2)
    urlpath = f'ds-0-1-linspace-{dtype_fixture.__name__}-b1-{shape1}d.b2nd'
    path = pathlib.Path(f'{ROOT}/{DIR + urlpath}')
    b1 = blosc2.C2Array(path, sub_url=SUB_URL, auth_cookie=auth_cookie)
    urlpath = f'ds-1-2-linspace-{dtype_fixture.__name__}-b2-{shape2}d.b2nd'
    path = pathlib.Path(f'{ROOT}/{DIR + urlpath}')
    b2 = blosc2.C2Array(path, sub_url=SUB_URL, auth_cookie=auth_cookie)

    return b1, b2, na1, na2


def test_broadcasting(broadcast_fixture):
    a1, a2, na1, na2 = broadcast_fixture
    expr1 = a1 + a2
    assert expr1.shape == a1.shape
    expr2 = a1 * a2 + 1
    assert expr2.shape == a1.shape
    expr = expr1 - expr2
    assert expr.shape == a1.shape
    nres = ne.evaluate("na1 + na2 - (na1 * na2 + 1)")
    res = expr.eval()
    np.testing.assert_allclose(res[:], nres)
    res = expr[:]
    np.testing.assert_allclose(res, nres)
