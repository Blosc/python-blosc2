#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################
import math
import pathlib

import numpy as np
import pytest
import torch

import blosc2
from blosc2.lazyexpr import ne_evaluate
from blosc2.ndarray import get_chunks_idx, npvecdot

NITEMS_SMALL = 100
NITEMS = 1000


@pytest.fixture(params=[np.float32, np.float64])
def dtype_fixture(request):
    return request.param


@pytest.fixture(params=[(NITEMS_SMALL,), (NITEMS,), (NITEMS // 10, 100)])
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
    na1 = np.linspace(0, 10, nelems, dtype=dtype_fixture).reshape(shape_fixture)
    chunks = chunks1 = blocks = blocks1 = None  # silence linter
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
    na2 = np.copy(na1)
    a2 = blosc2.asarray(na2, chunks=chunks, blocks=blocks)
    na3 = np.copy(na1)
    # Let other operands have chunks1 and blocks1
    a3 = blosc2.asarray(na3, chunks=chunks1, blocks=blocks1)
    na4 = np.copy(na1)
    a4 = blosc2.asarray(na4, chunks=chunks1, blocks=blocks1)
    return a1, a2, a3, a4, na1, na2, na3, na4


def test_simple_getitem(array_fixture):
    a1, a2, a3, a4, na1, na2, na3, na4 = array_fixture
    expr = a1 + a2 - a3 * a4
    nres = ne_evaluate("na1 + na2 - na3 * na4")
    sl = slice(100)
    res = expr[sl]
    np.testing.assert_allclose(res, nres[sl])


# Mix Proxy and NDArray operands
def test_proxy_simple_getitem(array_fixture):
    a1, a2, a3, a4, na1, na2, na3, na4 = array_fixture
    a1 = blosc2.Proxy(a1)
    a2 = blosc2.Proxy(a2)
    expr = a1 + a2 - a3 * a4
    nres = ne_evaluate("na1 + na2 - na3 * na4")
    sl = slice(100)
    res = expr[sl]
    np.testing.assert_allclose(res, nres[sl])


@pytest.mark.heavy
def test_mix_operands(array_fixture):
    a1, a2, a3, a4, na1, na2, na3, na4 = array_fixture
    expr = a1 + na2
    nres = ne_evaluate("na1 + na2")
    sl = slice(100)
    res = expr[sl]
    np.testing.assert_allclose(res, nres[sl])
    np.testing.assert_allclose(expr[:], nres)
    np.testing.assert_allclose(expr.compute()[:], nres)

    # TODO: fix this
    # expr = na2 + a1
    # nres = ne_evaluate("na2 + na1")
    # sl = slice(100)
    # res = expr[sl]
    # np.testing.assert_allclose(res, nres[sl])
    # np.testing.assert_allclose(expr[:], nres)
    # np.testing.assert_allclose(expr.compute()[:], nres)

    expr = a1 + na2 + a3
    nres = ne_evaluate("na1 + na2 + na3")
    res = expr[sl]
    np.testing.assert_allclose(res, nres[sl])
    np.testing.assert_allclose(expr[:], nres)
    np.testing.assert_allclose(expr.compute()[:], nres)

    expr = a1 * na2 + a3
    nres = ne_evaluate("na1 * na2 + na3")
    res = expr[sl]
    np.testing.assert_allclose(res, nres[sl])
    np.testing.assert_allclose(expr[:], nres)
    np.testing.assert_allclose(expr.compute()[:], nres)

    expr = a1 * na2 * a3
    nres = ne_evaluate("na1 * na2 * na3")
    res = expr[sl]
    np.testing.assert_allclose(res, nres[sl])
    np.testing.assert_allclose(expr[:], nres)
    np.testing.assert_allclose(expr.compute()[:], nres)

    expr = blosc2.LazyExpr(new_op=(na2, "*", a3))
    nres = ne_evaluate("na2 * na3")
    res = expr[sl]
    np.testing.assert_allclose(res, nres[sl])
    np.testing.assert_allclose(expr[:], nres)
    np.testing.assert_allclose(expr.compute()[:], nres)

    # TODO: support this case
    # expr = a1 + na2 * a3
    # print("--------------------------------------------------------")
    # print(type(expr))
    # print(expr.expression)
    # print(expr.operands)
    # print("--------------------------------------------------------")
    # nres = ne_evaluate("na1 + na2 * na3")
    # sl = slice(100)
    # res = expr[sl]
    # np.testing.assert_allclose(res, nres[sl])
    # np.testing.assert_allclose(expr[:], nres)
    # np.testing.assert_allclose(expr.compute()[:], nres)


# Add more test functions to test different aspects of the code
def test_simple_expression(array_fixture):
    a1, a2, a3, a4, na1, na2, na3, na4 = array_fixture
    expr = a1 + a2 - a3 * a4
    nres = ne_evaluate("na1 + na2 - na3 * na4")
    res = expr.compute(cparams=blosc2.CParams())
    np.testing.assert_allclose(res[:], nres)


# Mix Proxy and NDArray operands
def test_proxy_simple_expression(array_fixture):
    a1, a2, a3, a4, na1, na2, na3, na4 = array_fixture
    a1 = blosc2.Proxy(a1)
    a3 = blosc2.Proxy(a3)
    expr = a1 + a2 - a3 * a4
    nres = ne_evaluate("na1 + na2 - na3 * na4")
    res = expr.compute(storage=blosc2.Storage())
    np.testing.assert_allclose(res[:], nres)


def test_iXXX(array_fixture):
    a1, a2, a3, a4, na1, na2, na3, na4 = array_fixture
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
    np.testing.assert_allclose(res[:], nres)


def test_complex_evaluate(array_fixture):
    a1, a2, a3, a4, na1, na2, na3, na4 = array_fixture
    expr = blosc2.tan(a1) * (blosc2.sin(a2) * blosc2.sin(a2) + blosc2.cos(a3)) + (blosc2.sqrt(a4) * 2)
    expr += 2
    nres = ne_evaluate("tan(na1) * (sin(na2) * sin(na2) + cos(na3)) + (sqrt(na4) * 2) + 2")
    res = expr.compute()
    np.testing.assert_allclose(res[:], nres)


def test_complex_getitem(array_fixture):
    a1, a2, a3, a4, na1, na2, na3, na4 = array_fixture
    expr = blosc2.tan(a1) * (blosc2.sin(a2) * blosc2.sin(a2) + blosc2.cos(a3)) + (blosc2.sqrt(a4) * 2)
    expr += 2
    nres = ne_evaluate("tan(na1) * (sin(na2) * sin(na2) + cos(na3)) + (sqrt(na4) * 2) + 2")
    res = expr[:]
    np.testing.assert_allclose(res, nres)


def test_complex_getitem_slice(array_fixture):
    a1, a2, a3, a4, na1, na2, na3, na4 = array_fixture
    expr = blosc2.tan(a1) * (blosc2.sin(a2) * blosc2.sin(a2) + blosc2.cos(a3)) + (blosc2.sqrt(a4) * 2)
    expr += 2
    nres = ne_evaluate("tan(na1) * (sin(na2) * sin(na2) + cos(na3)) + (sqrt(na4) * 2) + 2")
    sl = slice(100)
    res = expr[sl]
    np.testing.assert_allclose(res, nres[sl])


def test_func_expression(array_fixture):
    a1, a2, a3, a4, na1, na2, na3, na4 = array_fixture
    expr = (a1 + a2) * a3 - a4
    expr = blosc2.sin(expr) + blosc2.cos(expr)
    nres = ne_evaluate("sin((na1 + na2) * na3 - na4) + cos((na1 + na2) * na3 - na4)")
    res = expr.compute(storage={})
    np.testing.assert_allclose(res[:], nres)


def test_expression_with_constants(array_fixture):
    a1, a2, a3, a4, na1, na2, na3, na4 = array_fixture
    # Test with operands with same chunks and blocks
    expr = a1 + 2 - a3 * 3.14
    nres = ne_evaluate("na1 + 2 - na3 * 3.14")
    np.testing.assert_allclose(expr[:], nres)


@pytest.mark.parametrize("compare_expressions", [True, False])
@pytest.mark.parametrize("comparison_operator", ["==", "!=", ">=", ">", "<=", "<"])
def test_comparison_operators(dtype_fixture, compare_expressions, comparison_operator):
    reshape = [30, 4]
    nelems = np.prod(reshape)
    cparams = {"clevel": 0, "codec": blosc2.Codec.LZ4}  # Compression parameters
    na1 = np.linspace(0, 10, nelems, dtype=dtype_fixture).reshape(reshape)
    na2 = np.copy(na1)
    a1 = blosc2.asarray(na1, cparams=cparams)
    a2 = blosc2.asarray(na1, cparams=cparams)
    # Construct the lazy expression
    if compare_expressions:
        expr = eval(f"a1 ** 2 {comparison_operator} (a1 + a2)", {"a1": a1, "a2": a2})
        expr_string = f"na1 ** 2 {comparison_operator} (na1 + na2)"
    else:
        expr = eval(f"a1 {comparison_operator} a2", {"a1": a1, "a2": a2})
        expr_string = f"na1 {comparison_operator} na2"
    res_lazyexpr = expr.compute(dparams={})
    # Evaluate using NumExpr
    res_numexpr = ne_evaluate(expr_string)
    # Compare the results
    np.testing.assert_allclose(res_lazyexpr[:], res_numexpr)


# Skip this test for blosc2.IS_WASM
@pytest.mark.skipif(blosc2.IS_WASM, reason="This test is not supported in WASM")
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
def test_functions(function, dtype_fixture, shape_fixture):
    nelems = np.prod(shape_fixture)
    cparams = {"clevel": 0, "codec": blosc2.Codec.LZ4}  # Compression parameters
    na1 = np.linspace(0, 10, nelems, dtype=dtype_fixture).reshape(shape_fixture)
    a1 = blosc2.asarray(na1, cparams=cparams)
    # Construct the lazy expression based on the function name
    expr = blosc2.LazyExpr(new_op=(a1, function, None))
    res_lazyexpr = expr.compute(cparams={})
    # Evaluate using NumExpr
    expr_string = f"{function}(na1)"
    res_numexpr = ne_evaluate(expr_string)
    # Compare the results
    np.testing.assert_allclose(res_lazyexpr[:], res_numexpr)
    np.testing.assert_allclose(expr.slice(slice(0, 10, 1)), res_numexpr[:10])  # slice test
    np.testing.assert_allclose(expr[:10], res_numexpr[:10])  # getitem test

    # For some reason real and imag are not supported by numpy's assert_allclose
    # (TypeError: bad operand type for abs(): 'LazyExpr' and segfaults are observed)
    if function in ("real", "imag"):
        return

    # Using numpy functions
    expr = eval(f"np.{function}(a1)", {"a1": a1, "np": np})
    # Compare the results
    np.testing.assert_allclose(expr[()], res_numexpr)

    # In combination with other operands
    na2 = np.linspace(0, 10, nelems, dtype=dtype_fixture).reshape(shape_fixture)
    a2 = blosc2.asarray(na2, cparams=cparams)
    # All the next work
    # expr = blosc2.lazyexpr(f"a1 + {function}(a2)", {"a1": a1, "a2": a2})
    # expr = eval(f"a1 + blosc2.{function}(a2)", {"a1": a1, "a2": a2, "blosc2": blosc2})
    expr = eval(f"a1 + np.{function}(a2)", {"a1": a1, "a2": a2, "np": np})
    res_lazyexpr = expr.compute(cparams={})
    # Evaluate using NumExpr
    expr_string = f"na1 + {function}(na2)"
    res_numexpr = ne_evaluate(expr_string)
    # Compare the results
    np.testing.assert_allclose(res_lazyexpr[:], res_numexpr)

    # Functions of the form np.function(a1 + a2)
    expr = eval(f"np.{function}(a1 + a2)", {"a1": a1, "a2": a2, "np": np})
    # Evaluate using NumExpr
    expr_string = f"{function}(na1 + na2)"
    res_numexpr = ne_evaluate(expr_string)
    # Compare the results
    np.testing.assert_allclose(expr[()], res_numexpr)


@pytest.mark.parametrize(
    "urlpath",
    ["arr.b2nd", None],
)
@pytest.mark.parametrize(
    "function",
    ["arctan2", "**"],
)
@pytest.mark.parametrize(
    ("value1", "value2"),
    [("NDArray", "scalar"), ("NDArray", "NDArray"), ("scalar", "NDArray"), ("scalar", "scalar")],
)
def test_arctan2_pow(urlpath, shape_fixture, dtype_fixture, function, value1, value2):
    nelems = np.prod(shape_fixture)
    if urlpath is None:
        urlpath1 = urlpath2 = urlpath_save = None
    else:
        urlpath1 = "a.b2nd"
        urlpath2 = "a2.b2nd"
        urlpath_save = "expr.b2nd"
    if value1 == "NDArray":  # ("NDArray", "scalar"), ("NDArray", "NDArray")
        na1 = np.linspace(0, 10, nelems, dtype=dtype_fixture).reshape(shape_fixture)
        a1 = blosc2.asarray(na1, urlpath=urlpath1, mode="w")
        if value2 == "NDArray":  # ("NDArray", "NDArray")
            na2 = np.linspace(0, 10, nelems, dtype=dtype_fixture).reshape(shape_fixture)
            a2 = blosc2.asarray(na1, urlpath=urlpath2, mode="w")
            # Construct the lazy expression based on the function name
            expr = blosc2.LazyExpr(new_op=(a1, function, a2))
            if urlpath is not None:
                expr.save(urlpath=urlpath_save)
                expr = blosc2.open(urlpath_save)
            res_lazyexpr = expr.compute()
            # Evaluate using NumExpr
            if function == "**":
                res_numexpr = ne_evaluate("na1**na2")
            else:
                expr_string = f"{function}(na1, na2)"
                res_numexpr = ne_evaluate(expr_string)
        else:  # ("NDArray", "scalar")
            value2 = 3
            # Construct the lazy expression based on the function name
            expr = blosc2.LazyExpr(new_op=(a1, function, value2))
            if urlpath is not None:
                expr.save(urlpath=urlpath_save)
                expr = blosc2.open(urlpath_save)
            res_lazyexpr = expr.compute()
            # Evaluate using NumExpr
            if function == "**":
                res_numexpr = ne_evaluate("na1**value2")
            else:
                expr_string = f"{function}(na1, value2)"
                res_numexpr = ne_evaluate(expr_string)
    elif value2 == "NDArray":  # ("scalar", "NDArray")
        value1 = 12
        na2 = np.linspace(0, 10, nelems, dtype=dtype_fixture).reshape(shape_fixture)
        a2 = blosc2.asarray(na2, urlpath=urlpath2, mode="w")
        # Construct the lazy expression based on the function name
        expr = blosc2.LazyExpr(new_op=(value1, function, a2))
        if urlpath is not None:
            expr.save(urlpath=urlpath_save)
            expr = blosc2.open(urlpath_save)
        res_lazyexpr = expr.compute()
        # Evaluate using NumExpr
        if function == "**":
            res_numexpr = ne_evaluate("value1**na2")
        else:
            expr_string = f"{function}(value1, na2)"
            res_numexpr = ne_evaluate(expr_string)
    else:  # ("scalar", "scalar")
        value1 = 12
        value2 = 3
        # Construct the lazy expression based on the function name
        expr = blosc2.LazyExpr(new_op=(value1, function, value2))
        res_lazyexpr = expr.compute()
        # Evaluate using NumExpr
        if function == "**":
            res_numexpr = ne_evaluate("value1**value2")
        else:
            expr_string = f"{function}(value1, value2)"
            res_numexpr = ne_evaluate(expr_string)
    # Compare the results
    tol = 1e-15 if dtype_fixture == "float64" else 1e-6
    np.testing.assert_allclose(res_lazyexpr[()], res_numexpr, atol=tol, rtol=tol)

    for path in [urlpath1, urlpath2, urlpath_save]:
        blosc2.remove_urlpath(path)


def test_abs(shape_fixture, dtype_fixture):
    nelems = np.prod(shape_fixture)
    na1 = np.linspace(-1, 1, nelems, dtype=dtype_fixture).reshape(shape_fixture)
    a1 = blosc2.asarray(na1)
    expr = blosc2.LazyExpr(new_op=(a1, "abs", None))
    res_lazyexpr = expr.compute(dparams={})
    res_np = np.abs(na1)
    np.testing.assert_allclose(res_lazyexpr[:], res_np)

    # Using np.abs
    expr = np.abs(a1)
    res_lazyexpr = expr.compute(dparams={})
    np.testing.assert_allclose(res_lazyexpr[:], res_np)


@pytest.mark.skipif(blosc2.IS_WASM, reason="This test is not supported in WASM")
@pytest.mark.parametrize("values", [("NDArray", "str"), ("NDArray", "NDArray"), ("str", "NDArray")])
def test_contains(values):
    # Unpack the value fixture
    value1, value2 = values
    if value1 == "NDArray":
        a1 = np.array([b"abc", b"def", b"aterr", b"oot", b"zu", b"ab c"])
        a1_blosc = blosc2.asarray(a1)
        if value2 == "str":  # ("NDArray", "str")
            value2 = b"test abc here"
            # Construct the lazy expression
            expr_lazy = blosc2.LazyExpr(new_op=(a1_blosc, "contains", value2))
            # Evaluate using NumExpr
            expr_numexpr = f"{'contains'}(a1, value2)"
            res_numexpr = ne_evaluate(expr_numexpr)
        else:  # ("NDArray", "NDArray")
            a2 = np.array([b"abc", b"ab c", b" abc", b" abc ", b"\tabc", b"c h"])
            a2_blosc = blosc2.asarray(a2)
            # Construct the lazy expression
            expr_lazy = blosc2.LazyExpr(new_op=(a1_blosc, "contains", a2_blosc))
            # Evaluate using NumExpr
            res_numexpr = ne_evaluate("contains(a2, a1)")
    else:  # ("str", "NDArray")
        value1 = b"abc"
        a2 = np.array([b"abc", b"def", b"aterr", b"oot", b"zu", b"ab c"])
        a2_blosc = blosc2.asarray(a2)
        # Construct the lazy expression
        expr_lazy = blosc2.LazyExpr(new_op=(value1, "contains", a2_blosc))
        # Evaluate using NumExpr
        res_numexpr = ne_evaluate("contains(value1, a2)")
    res_lazyexpr = expr_lazy.compute()
    # Compare the results
    np.testing.assert_array_equal(res_lazyexpr[:], res_numexpr)


def test_negate(dtype_fixture, shape_fixture):
    nelems = np.prod(shape_fixture)
    na1 = np.linspace(-1, 1, nelems, dtype=dtype_fixture).reshape(shape_fixture)
    a1 = blosc2.asarray(na1)

    # Test with a single NDArray
    expr = -a1
    res_lazyexpr = expr.compute()
    res_np = -na1
    np.testing.assert_allclose(res_lazyexpr[:], res_np)

    # Test with a proper expression
    expr = -(a1 + 2)
    res_lazyexpr = expr.compute()
    res_np = -(na1 + 2)
    np.testing.assert_allclose(res_lazyexpr[:], res_np)


@pytest.mark.skipif(blosc2.IS_WASM, reason="This test is not supported in WASM")
def test_params(array_fixture):
    a1, a2, a3, a4, na1, na2, na3, na4 = array_fixture
    expr = a1 + a2 - a3 * a4
    nres = ne_evaluate("na1 + na2 - na3 * na4")

    urlpath = "eval_expr.b2nd"
    blosc2.remove_urlpath(urlpath)
    cparams = blosc2.CParams(nthreads=2)
    dparams = {"nthreads": 4}
    chunks = tuple(i // 2 for i in nres.shape)
    blocks = tuple(i // 4 for i in nres.shape)
    res = expr.compute(urlpath=urlpath, cparams=cparams, dparams=dparams, chunks=chunks, blocks=blocks)
    np.testing.assert_allclose(res[:], nres)
    assert res.schunk.urlpath == urlpath
    assert res.schunk.cparams.nthreads == cparams.nthreads
    assert res.schunk.dparams.nthreads == dparams["nthreads"]
    assert res.chunks == chunks
    assert res.blocks == blocks

    blosc2.remove_urlpath(urlpath)


# Tests related with save method
def test_save():
    tol = 1e-17
    shape = (23, 23)
    nelems = np.prod(shape)
    na1 = np.linspace(0, 10, nelems, dtype=np.float32).reshape(shape)
    na2 = np.linspace(10, 20, nelems, dtype=np.float32).reshape(shape)
    na3 = np.linspace(0, 10, nelems).reshape(shape)
    na4 = np.linspace(0, 10, nelems).reshape(shape)
    a1 = blosc2.asarray(na1)
    a2 = blosc2.asarray(na2)
    a3 = blosc2.asarray(na3)
    a4 = blosc2.asarray(na4)
    ops = [a1, a2, a3, a4]
    op_urlpaths = ["a1.b2nd", "a2.b2nd", "a3.b2nd", "a4.b2nd"]
    for i, urlpath in enumerate(op_urlpaths):
        ops[i] = ops[i].copy(urlpath=urlpath, mode="w")

    # Construct the lazy expression with the on-disk operands
    da1, da2, da3, da4 = ops
    expr = da1 / da2 + da2 - da3 * da4
    nres = ne_evaluate("na1 / na2 + na2 - na3 * na4")
    urlpath_save = "expr.b2nd"
    expr.save(urlpath=urlpath_save)

    if not blosc2.IS_WASM:
        cparams = {"nthreads": 2}
        dparams = {"nthreads": 4}
    else:
        cparams = {}
        dparams = {}
    chunks = tuple(i // 2 for i in nres.shape)
    blocks = tuple(i // 4 for i in nres.shape)
    urlpath_eval = "eval_expr.b2nd"
    res = expr.compute(
        storage=blosc2.Storage(urlpath=urlpath_eval, mode="w"),
        chunks=chunks,
        blocks=blocks,
        cparams=cparams,
        dparams=dparams,
    )
    np.testing.assert_allclose(res[:], nres, rtol=tol, atol=tol)

    expr = blosc2.open(urlpath_save)
    # After opening, check that a lazy expression does have an array
    # and schunk attributes. This is to allow the .info() method to work.
    assert hasattr(expr, "array") is True
    assert hasattr(expr, "schunk") is True
    # Check the dtype (should be upcasted to float64)
    assert expr.array.dtype == np.float64
    res = expr.compute()
    assert res.dtype == np.float64
    np.testing.assert_allclose(res[:], nres, rtol=tol, atol=tol)
    # Test getitem
    np.testing.assert_allclose(expr[:], nres, rtol=tol, atol=tol)

    urlpath_save2 = "expr_str.b2nd"
    x = 3
    expr = "a1 / a2 + a2 - a3 * a4**x"
    var_dict = {"a1": ops[0], "a2": ops[1], "a3": ops[2], "a4": ops[3], "x": x}
    lazy_expr = eval(expr, var_dict)
    lazy_expr.save(urlpath=urlpath_save2)
    expr = blosc2.open(urlpath_save2)
    assert expr.array.dtype == np.float64
    res = expr.compute()
    nres = ne_evaluate("na1 / na2 + na2 - na3 * na4**3")
    np.testing.assert_allclose(res[:], nres, rtol=tol, atol=tol)
    # Test getitem
    np.testing.assert_allclose(expr[:], nres, rtol=tol, atol=tol)

    for urlpath in op_urlpaths + [urlpath_save, urlpath_eval, urlpath_save2]:
        blosc2.remove_urlpath(urlpath)


@pytest.mark.skipif(blosc2.IS_WASM, reason="This test is not supported in WASM")
def test_save_unsafe():
    na = np.arange(1000)
    nb = np.arange(1000)
    a = blosc2.asarray(na, urlpath="a.b2nd", mode="w")
    b = blosc2.asarray(nb, urlpath="b.b2nd", mode="w")
    disk_arrays = ["a.b2nd", "b.b2nd"]
    expr = a + b
    urlpath = "expr.b2nd"
    expr.save(urlpath=urlpath)
    disk_arrays.append(urlpath)

    expr = blosc2.open(urlpath)
    # Replace expression by a (potentially) unsafe expression
    expr.expression = "import os; os.system('touch /tmp/unsafe')"
    with pytest.raises(ValueError) as excinfo:
        expr.compute()
    assert expr.expression in str(excinfo.value)

    # Check that an invalid expression cannot be easily saved.
    # Although, as this can easily be worked around, the best protection is
    # during loading time (tested above).
    expr.expression_tosave = "import os; os.system('touch /tmp/unsafe')"
    with pytest.raises(ValueError) as excinfo:
        expr.save(urlpath=urlpath)
    assert expr.expression_tosave in str(excinfo.value)

    for urlpath in disk_arrays:
        blosc2.remove_urlpath(urlpath)


@pytest.mark.skipif(blosc2.IS_WASM, reason="This test is not supported in WASM")
@pytest.mark.parametrize(
    "function",
    [
        "sin",
        "sqrt",
        "cosh",
        "arctan",
        "arcsinh",
        "exp",
        "expm1",
        "log",
        "conj",
        "real",
        "imag",
    ],
)
def test_save_functions(function, dtype_fixture, shape_fixture):
    nelems = np.prod(shape_fixture)
    cparams = {"clevel": 0, "codec": blosc2.Codec.LZ4}  # Compression parameters
    na1 = np.linspace(0, 10, nelems, dtype=dtype_fixture).reshape(shape_fixture)
    urlpath_op = "a1.b2nd"
    a1 = blosc2.asarray(na1, cparams=cparams, urlpath=urlpath_op, mode="w")
    urlpath_save = "expr.b2nd"

    # Construct the lazy expression based on the function name
    expr = blosc2.LazyExpr(new_op=(a1, function, None))
    expr.save(urlpath=urlpath_save)
    del expr
    expr = blosc2.open(urlpath_save)
    res_lazyexpr = expr.compute()

    # Evaluate using NumExpr
    expr_string = f"{function}(na1)"
    res_numexpr = ne_evaluate(expr_string)
    # Compare the results
    np.testing.assert_allclose(res_lazyexpr[:], res_numexpr)

    expr_string = f"blosc2.{function}(a1)"
    expr = eval(expr_string, {"a1": a1, "blosc2": blosc2})
    expr.save(urlpath=urlpath_save)
    res_lazyexpr = expr.compute()
    np.testing.assert_allclose(res_lazyexpr[:], res_numexpr)

    expr = blosc2.open(urlpath_save)
    res_lazyexpr = expr.compute()
    np.testing.assert_allclose(res_lazyexpr[:], res_numexpr)

    for urlpath in [urlpath_op, urlpath_save]:
        blosc2.remove_urlpath(urlpath)


@pytest.mark.skipif(blosc2.IS_WASM, reason="This test is not supported in WASM")
@pytest.mark.parametrize("values", [("NDArray", "str"), ("NDArray", "NDArray"), ("str", "NDArray")])
def test_save_contains(values):
    # Unpack the value fixture
    value1, value2 = values
    urlpath = "a.b2nd"
    urlpath2 = "a2.b2nd"
    urlpath_save = "expr.b2nd"
    if value1 == "NDArray":
        a1 = np.array([b"abc(", b"def", b"aterr", b"oot", b"zu", b"ab c"])
        a1_blosc = blosc2.asarray(a1, urlpath=urlpath, mode="w")
        if value2 == "str":  # ("NDArray", "str")
            value2 = b"test abc( here"
            # Construct the lazy expression
            expr_lazy = blosc2.LazyExpr(new_op=(a1_blosc, "contains", value2))
            expr_lazy.save(urlpath=urlpath_save)
            expr_lazy = blosc2.open(urlpath_save)
            # Evaluate using NumExpr
            expr_numexpr = f"{'contains'}(a1, value2)"
            res_numexpr = ne_evaluate(expr_numexpr)
        else:  # ("NDArray", "NDArray")
            a2 = np.array([b"abc(", b"ab c", b" abc", b" abc ", b"\tabc", b"c h"])
            a2_blosc = blosc2.asarray(a2, urlpath=urlpath2, mode="w")
            # Construct the lazy expression
            expr_lazy = blosc2.LazyExpr(new_op=(a1_blosc, "contains", a2_blosc))
            expr_lazy.save(urlpath=urlpath_save)
            expr_lazy = blosc2.open(urlpath_save)
            # Evaluate using NumExpr
            res_numexpr = ne_evaluate("contains(a2, a1)")
    else:  # ("str", "NDArray")
        value1 = b"abc"
        a2 = np.array([b"abc(", b"def", b"aterr", b"oot", b"zu", b"ab c"])
        a2_blosc = blosc2.asarray(a2, urlpath=urlpath2, mode="w")
        # Construct the lazy expression
        expr_lazy = blosc2.LazyExpr(new_op=(value1, "contains", a2_blosc))
        expr_lazy.save(urlpath=urlpath_save)
        expr_lazy = blosc2.open(urlpath_save)
        # Evaluate using NumExpr
        res_numexpr = ne_evaluate("contains(value1, a2)")
    res_lazyexpr = expr_lazy.compute()
    # Compare the results
    np.testing.assert_array_equal(res_lazyexpr[:], res_numexpr)

    for path in [urlpath, urlpath2, urlpath_save]:
        blosc2.remove_urlpath(path)


@pytest.mark.skipif(blosc2.IS_WASM, reason="This test is not supported in WASM")
def test_save_many_functions(dtype_fixture, shape_fixture):
    rtol = 1e-6 if dtype_fixture == np.float32 else 1e-15
    atol = 1e-6 if dtype_fixture == np.float32 else 1e-15
    nelems = np.prod(shape_fixture)
    cparams = {"clevel": 0, "codec": blosc2.Codec.LZ4}  # Compression parameters
    na1 = np.linspace(0, 10, nelems, dtype=dtype_fixture).reshape(shape_fixture)
    na2 = np.linspace(0, 10, nelems, dtype=dtype_fixture).reshape(shape_fixture)
    urlpath_op = "a1.b2nd"
    urlpath_op2 = "a1.b2nd"
    a1 = blosc2.asarray(na1, cparams=cparams, urlpath=urlpath_op, mode="w")
    a2 = blosc2.asarray(na2, cparams=cparams, urlpath=urlpath_op2, mode="w")

    # Evaluate using NumExpr
    expr_string = "sin(x)**3 + cos(y)**2 + cos(x) * arcsin(y) + arcsinh(x) + sinh(x)"
    res_numexpr = ne_evaluate(expr_string, {"x": na1, "y": na2})

    urlpath_save = "expr.b2nd"
    expr = blosc2.lazyexpr(expr_string, {"x": a1, "y": a2})
    expr.save(urlpath=urlpath_save)
    res_lazyexpr = expr.compute()
    np.testing.assert_allclose(res_lazyexpr[:], res_numexpr, rtol=rtol, atol=atol)

    expr = blosc2.open(urlpath_save)
    res_lazyexpr = expr.compute()
    np.testing.assert_allclose(res_lazyexpr[:], res_numexpr, rtol=rtol, atol=atol)

    for urlpath in [urlpath_op, urlpath_op2, urlpath_save]:
        blosc2.remove_urlpath(urlpath)


@pytest.mark.skipif(blosc2.IS_WASM, reason="This test is not supported in WASM")
@pytest.mark.parametrize(
    "constructor", ["arange", "linspace", "fromiter", "reshape", "zeros", "ones", "full"]
)
@pytest.mark.parametrize("shape", [(10,), (10, 10), (10, 10, 10)])
@pytest.mark.parametrize("dtype", ["int32", "float64", "i2"])
@pytest.mark.parametrize("disk", [True, False])
def test_save_constructor(disk, shape, dtype, constructor):  # noqa: C901
    lshape = math.prod(shape)
    urlpath = "a.b2nd" if disk else None
    b2func = getattr(blosc2, constructor)
    a, expr = None, None
    if constructor in ("zeros", "ones"):
        a = b2func(shape, dtype=dtype, urlpath=urlpath, mode="w")
        expr = f"a + {constructor}({shape}, dtype={dtype}) + 1"
    elif constructor == "full":
        a = b2func(shape, 10, dtype=dtype, urlpath=urlpath, mode="w")
        expr = f"a + {constructor}(10, {shape}, dtype={dtype}) + 1"
    elif constructor == "fromiter":
        a = b2func(range(lshape), dtype=dtype, shape=shape, urlpath=urlpath, mode="w")
        expr = f"a + {constructor}(range({lshape}), dtype={dtype}, shape={shape}) + 1"
    elif constructor == "reshape":
        # Let's put a nested arange array here
        a = blosc2.arange(lshape, dtype=dtype, shape=shape, urlpath=urlpath, mode="w")
        b = f"arange({lshape}, dtype={dtype})"
        # Both expressions below are equivalent, but use the method variant for testing purposes
        # expr = f"a + {constructor}({b}, shape={shape}) + 1"
        expr = f"a + {b}.reshape({shape}) + 1"
        # The one below is also supported, but should be rarely used
        # expr = f"a + {b}.reshape(shape={shape}) + 1"
    elif constructor == "linspace":
        a = b2func(0, 10, lshape, dtype=dtype, shape=shape, urlpath=urlpath, mode="w")
        expr = f"a + {constructor}(0, 10, {lshape}, dtype={dtype}, shape={shape}) + 1"
    elif constructor == "arange":
        a = b2func(lshape, dtype=dtype, shape=shape, urlpath=urlpath, mode="w")
        expr = f"a + {constructor}({lshape}, dtype={dtype}, shape={shape}) + 1"
    if disk:
        a = blosc2.open(urlpath)
    npfunc = getattr(np, constructor)
    if constructor == "linspace":
        na = npfunc(0, 10, lshape, dtype=dtype).reshape(shape)
    elif constructor == "fromiter":
        na = np.fromiter(range(lshape), dtype=dtype, count=lshape).reshape(shape)
    elif constructor == "reshape":
        na = np.arange(lshape, dtype=dtype).reshape(shape)
    elif constructor == "full":
        na = npfunc(shape, 10, dtype=dtype)
    else:
        na = npfunc(lshape, dtype=dtype).reshape(shape)

    # An expression involving the constructor
    lexpr = blosc2.lazyexpr(expr)
    assert lexpr.shape == a.shape
    if disk:
        lexpr.save("out.b2nd")
        lexpr = blosc2.open("out.b2nd")
    res = lexpr.compute()
    nres = na + na + 1
    assert np.allclose(res[()], nres)

    if disk:
        blosc2.remove_urlpath("a.b2nd")
        blosc2.remove_urlpath("out.b2nd")


@pytest.mark.parametrize("shape", [(10,), (10, 10), (10, 10, 10)])
@pytest.mark.parametrize("disk", [True, False])
def test_save_2_constructors(shape, disk):
    lshape = math.prod(shape)
    urlpath_a = "a.b2nd" if disk else None
    urlpath_b = "b.b2nd" if disk else None
    a = blosc2.arange(lshape, shape=shape, urlpath=urlpath_a, mode="w")
    b = blosc2.ones(shape, urlpath=urlpath_b, mode="w")
    expr = f"arange({lshape}, shape={shape}) + a + ones({shape}) + b + 1"
    lexpr = blosc2.lazyexpr(expr)
    if disk:
        lexpr.save("out.b2nd")
        lexpr = blosc2.open("out.b2nd")
    res = lexpr.compute()
    na = np.arange(lshape).reshape(shape)
    nb = np.ones(shape)
    nres = na + a[:] + nb + b[:] + 1
    assert np.allclose(res[()], nres)
    if disk:
        blosc2.remove_urlpath(urlpath_a)
        blosc2.remove_urlpath(urlpath_b)
        blosc2.remove_urlpath("out.b2nd")


@pytest.mark.parametrize("shape", [(10,), (10, 10), (10, 10, 10)])
@pytest.mark.parametrize("disk", [True, False])
def test_save_constructor_reshape(shape, disk):
    lshape = math.prod(shape)
    urlpath_a = "a.b2nd" if disk else None
    urlpath_b = "b.b2nd" if disk else None
    a = blosc2.arange(lshape, shape=shape, urlpath=urlpath_a, mode="w")
    b = blosc2.ones(shape, urlpath=urlpath_b, mode="w")
    # All the next work
    # expr = f"arange({lshape}).reshape({shape}) + a + ones({shape}) + b + 1"
    # expr = f"arange({lshape}).reshape(shape={shape}) + a + ones({shape}) + b + 1"
    expr = f"arange({lshape}).reshape(shape  = {shape}) + a + ones({shape}) + b + 1"
    lexpr = blosc2.lazyexpr(expr)
    if disk:
        lexpr.save("out.b2nd")
        lexpr = blosc2.open("out.b2nd")
    res = lexpr.compute()
    na = np.arange(lshape).reshape(shape)
    nb = np.ones(shape)
    nres = na + a[:] + nb + b[:] + 1
    assert np.allclose(res[()], nres)
    if disk:
        blosc2.remove_urlpath(urlpath_a)
        blosc2.remove_urlpath(urlpath_b)
        blosc2.remove_urlpath("out.b2nd")


@pytest.mark.parametrize("shape", [(10,), (10, 10), (10, 10, 10)])
@pytest.mark.parametrize("disk", [True, False])
def test_save_2equal_constructors(shape, disk):
    lshape = math.prod(shape)
    urlpath_a = "a.b2nd" if disk else None
    urlpath_b = "b.b2nd" if disk else None
    a = blosc2.ones(shape, dtype=np.int8, urlpath=urlpath_a, mode="w")
    b = blosc2.ones(shape, urlpath=urlpath_b, mode="w")
    expr = f"ones({shape}, dtype=int8) + a + ones({shape}) + b + 1"
    lexpr = blosc2.lazyexpr(expr)
    if disk:
        lexpr.save("out.b2nd")
        lexpr = blosc2.open("out.b2nd")
    res = lexpr.compute()
    na = np.ones(shape, dtype=np.int8)
    nb = np.ones(shape)
    nres = na + a[:] + nb + b[:] + 1
    assert np.allclose(res[()], nres)
    assert res.dtype == nres.dtype
    if disk:
        blosc2.remove_urlpath(urlpath_a)
        blosc2.remove_urlpath(urlpath_b)
        blosc2.remove_urlpath("out.b2nd")


@pytest.fixture(
    params=[
        ((10, 1), (10,)),
        ((2, 5), (5,)),
        ((2, 1), (5,)),
        ((2, 5, 3), (5, 3)),
        ((2, 5, 3), (5, 1)),
        ((2, 1, 3), (5, 3)),
        ((2, 5, 3, 2), (5, 3, 2)),
        ((2, 5, 3, 2), (5, 3, 1)),
        pytest.param(((2, 5, 3, 2), (5, 1, 2)), marks=pytest.mark.heavy),
        ((2, 1, 3, 2), (5, 3, 2)),
        pytest.param(((2, 1, 3, 2), (5, 1, 2)), marks=pytest.mark.heavy),
        pytest.param(((2, 5, 3, 2, 2), (5, 3, 2, 2)), marks=pytest.mark.heavy),
        pytest.param(((100, 100, 100), (100, 100)), marks=pytest.mark.heavy),
        ((1_000, 1), (1_000,)),
    ]
)
def broadcast_shape(request):
    return request.param


# Test broadcasting
@pytest.fixture
def broadcast_fixture(dtype_fixture, broadcast_shape):
    shape1, shape2 = broadcast_shape
    na1 = np.linspace(0, 1, np.prod(shape1), dtype=dtype_fixture).reshape(shape1)
    na2 = np.linspace(1, 2, np.prod(shape2), dtype=dtype_fixture).reshape(shape2)
    a1 = blosc2.asarray(na1)
    a2 = blosc2.asarray(na2)
    return a1, a2, na1, na2


def test_broadcasting(broadcast_fixture):
    a1, a2, na1, na2 = broadcast_fixture
    expr1 = a1 + a2
    assert expr1.shape == np.broadcast_shapes(a1.shape, a2.shape)
    expr2 = a1 * a2 + 1
    assert expr2.shape == np.broadcast_shapes(a1.shape, a2.shape)
    expr = expr1 - expr2
    assert expr.shape == np.broadcast_shapes(expr1.shape, expr2.shape)
    nres = ne_evaluate("na1 + na2 - (na1 * na2 + 1)")
    res = expr.compute()
    np.testing.assert_allclose(res[:], nres)
    res = expr[:]
    np.testing.assert_allclose(res, nres)


def test_broadcasting_str(broadcast_fixture):
    a1, a2, na1, na2 = broadcast_fixture
    expr1 = blosc2.lazyexpr("a1 + a2")
    assert expr1.shape == np.broadcast_shapes(a1.shape, a2.shape)
    expr2 = blosc2.lazyexpr("a1 * a2 + 1")
    assert expr2.shape == np.broadcast_shapes(a1.shape, a2.shape)
    expr = blosc2.lazyexpr("expr1 - expr2")
    assert expr.shape == np.broadcast_shapes(expr1.shape, expr2.shape)
    nres = ne_evaluate("na1 + na2 - (na1 * na2 + 1)")
    assert expr.shape == nres.shape
    res = expr.compute()
    np.testing.assert_allclose(res[:], nres)
    res = expr[:]
    np.testing.assert_allclose(res, nres)


@pytest.mark.parametrize(
    "operand_mix",
    [
        ("NDArray", "numpy"),
        ("NDArray", "NDArray"),
        ("numpy", "NDArray"),
        ("numpy", "numpy"),
    ],
)
@pytest.mark.parametrize("operand_guess", [True, False])
def test_lazyexpr(array_fixture, operand_mix, operand_guess):
    a1, a2, a3, a4, na1, na2, na3, na4 = array_fixture
    if operand_mix[0] == "NDArray" and operand_mix[1] == "NDArray":
        operands = {"a1": a1, "a2": a2, "a3": a3, "a4": a4}
    elif operand_mix[0] == "NDArray" and operand_mix[1] == "numpy":
        operands = {"a1": a1, "a2": na2, "a3": a3, "a4": na4}
    elif operand_mix[0] == "numpy" and operand_mix[1] == "NDArray":
        operands = {"a1": na1, "a2": a2, "a3": na3, "a4": a4}
    else:
        operands = {"a1": na1, "a2": na2, "a3": na3, "a4": na4}

    # Check eval()
    if operand_guess:
        expr = blosc2.lazyexpr("a1 + a2 - a3 * a4")
    else:
        expr = blosc2.lazyexpr("a1 + a2 - a3 * a4", operands=operands)
    nres = ne_evaluate("na1 + na2 - na3 * na4")
    assert expr.shape == nres.shape
    res = expr.compute()
    np.testing.assert_allclose(res[:], nres)
    # With selections
    res = expr.compute(item=0)
    np.testing.assert_allclose(res[()], nres[0])
    res = expr.compute(item=slice(10))
    np.testing.assert_allclose(res[()], nres[:10])
    res = expr.compute(item=slice(0, 10, 2))
    np.testing.assert_allclose(res[()], nres[0:10:2])

    # Check getitem
    res = expr[:]
    np.testing.assert_allclose(res, nres)
    # With selections
    res = expr[0]
    np.testing.assert_allclose(res, nres[0])
    res = expr[0:10]
    np.testing.assert_allclose(res, nres[0:10])
    res = expr[0:10:2]
    np.testing.assert_allclose(res, nres[0:10:2])


@pytest.mark.parametrize(
    "operand_mix",
    [
        ("NDArray", "numpy"),
        ("NDArray", "NDArray"),
        ("numpy", "NDArray"),
        ("numpy", "numpy"),
    ],
)
@pytest.mark.parametrize(
    "out_param",
    ["NDArray", "numpy"],
)
def test_lazyexpr_out(array_fixture, out_param, operand_mix):
    a1, a2, a3, a4, na1, na2, na3, na4 = array_fixture
    if operand_mix[0] == "NDArray" and operand_mix[1] == "NDArray":
        operands = {"a1": a1, "a2": a2}
    elif operand_mix[0] == "NDArray" and operand_mix[1] == "numpy":
        operands = {"a1": a1, "a2": na2}
    elif operand_mix[0] == "numpy" and operand_mix[1] == "NDArray":
        operands = {"a1": na1, "a2": a2}
    else:
        operands = {"a1": na1, "a2": na2}
    if out_param == "NDArray":
        out = a3
    else:
        out = na3
    expr = blosc2.lazyexpr("a1 + a2", operands=operands, out=out)
    res = expr.compute()  # res should be equal to out
    assert res is out
    nres = ne_evaluate("na1 + na2", out=na4)
    assert nres is na4
    if out_param == "NDArray":
        np.testing.assert_allclose(res[:], nres)
    else:
        np.testing.assert_allclose(na3, na4)

    # Use an existing LazyExpr as expression
    expr = blosc2.lazyexpr("a1 - a2", operands=operands)
    operands = {"a1": a1, "a2": a2}
    expr2 = blosc2.lazyexpr(expr, operands=operands, out=out)
    assert expr2.compute() is out
    nres = ne_evaluate("na1 - na2")
    np.testing.assert_allclose(out[:], nres)


# Test compute with an item parameter
def test_eval_item(array_fixture):
    a1, a2, a3, a4, na1, na2, na3, na4 = array_fixture
    expr = blosc2.lazyexpr("a1 + a2 - a3 * a4", operands={"a1": a1, "a2": a2, "a3": a3, "a4": a4})
    nres = ne_evaluate("na1 + na2 - na3 * na4")
    res = expr.compute(item=0)
    np.testing.assert_allclose(res[()], nres[0])
    res = expr.compute(item=slice(10))
    np.testing.assert_allclose(res[()], nres[:10])
    res = expr.compute(item=slice(0, 10, 2))
    np.testing.assert_allclose(res[()], nres[0:10:2])


# Test getitem with an item parameter
def test_eval_getitem(array_fixture):
    a1, a2, a3, a4, na1, na2, na3, na4 = array_fixture
    expr = blosc2.lazyexpr("a1 + a2 - a3 * a4", operands={"a1": a1, "a2": a2, "a3": a3, "a4": a4})
    nres = ne_evaluate("na1 + na2 - na3 * na4")
    np.testing.assert_allclose(expr[0], nres[0])
    np.testing.assert_allclose(expr[:10], nres[:10])
    np.testing.assert_allclose(expr[0:10:2], nres[0:10:2])


def test_eval_getitem2():
    # Small test for non-isomorphic shape
    shape = (2, 10, 5)
    test_arr = blosc2.linspace(0, 10, np.prod(shape), shape=shape, chunks=(1, 5, 1))
    expr = test_arr * 30
    nres = test_arr[:] * 30
    np.testing.assert_allclose(expr[0], nres[0])
    np.testing.assert_allclose(expr[1:, :7], nres[1:, :7])
    np.testing.assert_allclose(expr[0:10:2], nres[0:10:2])
    # Now relies on inefficient blosc2.ndarray.slice for non-unit steps but only per chunk (not for whole result)
    np.testing.assert_allclose(expr.slice((slice(None, None, None), slice(0, 10, 2)))[:], nres[:, 0:10:2])

    # Small test for broadcasting
    expr = test_arr + test_arr.slice(1)
    nres = test_arr[:] + test_arr[1]
    np.testing.assert_allclose(expr[0], nres[0])
    np.testing.assert_allclose(expr[1:, :7], nres[1:, :7])
    np.testing.assert_allclose(expr[:, 0:10:2], nres[:, 0:10:2])
    # Now relies on inefficient blosc2.ndarray.slice for non-unit steps but only per chunk (not for whole result)
    np.testing.assert_allclose(expr.slice((slice(None, None, None), slice(0, 10, 2)))[:], nres[:, 0:10:2])


# Test lazyexpr's slice method
def test_eval_slice(array_fixture):
    a1, a2, a3, a4, na1, na2, na3, na4 = array_fixture
    expr = blosc2.lazyexpr("a1 + a2 - (a3 * a4)", operands={"a1": a1, "a2": a2, "a3": a3, "a4": a4})
    nres = ne_evaluate("na1 + na2 - (na3 * na4)")
    res = expr.slice(slice(0, 8, 2))
    assert isinstance(res, blosc2.ndarray.NDArray)
    np.testing.assert_allclose(res[:], nres[:8:2])
    res = expr[:8:2]
    assert isinstance(res, np.ndarray)
    np.testing.assert_allclose(res, nres[:8:2])

    # string lazy expressions automatically use .slice internally
    expr1 = blosc2.lazyexpr("a1 * a2", operands={"a1": a1, "a2": a2})
    expr2 = blosc2.lazyexpr("expr1[:2] + a3[:2]")
    nres = ne_evaluate("(na1 * na2) + na3")[:2]
    assert isinstance(expr2, blosc2.LazyExpr)
    res = expr2.compute()
    assert isinstance(res, blosc2.ndarray.NDArray)
    np.testing.assert_allclose(res[()], nres)


def test_rebasing(array_fixture):
    a1, a2, a3, a4, na1, na2, na3, na4 = array_fixture
    expr = blosc2.lazyexpr("a1 + a2 - (a3 * a4)", operands={"a1": a1, "a2": a2, "a3": a3, "a4": a4})
    assert expr.expression == "(o0 + o1 - o2 * o3)"

    expr = blosc2.lazyexpr("a1")
    assert expr.expression == "(o0)"

    expr = blosc2.lazyexpr("a1[:10]")
    assert expr.expression == "(o0.slice((slice(None, 10, None),)))"


# Test get_chunk method
@pytest.mark.heavy
def test_get_chunk(array_fixture):
    a1, a2, a3, a4, na1, na2, na3, na4 = array_fixture
    expr = blosc2.lazyexpr(
        "a1 + a2 - a3 * a4",
        operands={"a1": a1, "a2": a2, "a3": a3, "a4": a4},
    )
    nres = ne_evaluate("na1 + na2 - na3 * na4")
    chunksize = np.prod(expr.chunks) * expr.dtype.itemsize
    blocksize = np.prod(expr.blocks) * expr.dtype.itemsize
    _, nchunks = get_chunks_idx(expr.shape, expr.chunks)
    out = blosc2.empty(expr.shape, dtype=expr.dtype, chunks=expr.chunks, blocks=expr.blocks)
    for nchunk in range(nchunks):
        chunk = expr.get_chunk(nchunk)
        out.schunk.update_chunk(nchunk, chunk)
        chunksize_ = int.from_bytes(chunk[4:8], byteorder="little")
        blocksize_ = int.from_bytes(chunk[8:12], byteorder="little")
        # Sometimes the actual chunksize is smaller than the expected chunks due to padding
        assert chunksize <= chunksize_
        assert blocksize == blocksize_
    np.testing.assert_allclose(out[:], nres)


@pytest.mark.skipif(blosc2.IS_WASM, reason="This test is not supported in WASM")
@pytest.mark.parametrize(
    ("chunks", "blocks"),
    [
        ((10, 100), (6, 100)),  # behaved
        ((15, 100), (5, 100)),  # not behaved
        ((15, 15), (5, 5)),  # not behaved
        ((10, 10), (5, 5)),  # not behaved
    ],
)
@pytest.mark.parametrize(
    "disk",
    [True, False],
)
@pytest.mark.parametrize("fill_value", [0, 1, np.nan])
def test_fill_disk_operands(chunks, blocks, disk, fill_value):
    N = 100

    apath = bpath = cpath = None
    if disk:
        apath = "a.b2nd"
        bpath = "b.b2nd"
        cpath = "c.b2nd"
    if fill_value != 0:
        a = blosc2.full((N, N), fill_value, urlpath=apath, mode="w", chunks=chunks, blocks=blocks)
        b = blosc2.full((N, N), fill_value, urlpath=bpath, mode="w", chunks=chunks, blocks=blocks)
        c = blosc2.full((N, N), fill_value, urlpath=cpath, mode="w", chunks=chunks, blocks=blocks)
    else:
        a = blosc2.zeros((N, N), urlpath=apath, mode="w", chunks=chunks, blocks=blocks)
        b = blosc2.zeros((N, N), urlpath=bpath, mode="w", chunks=chunks, blocks=blocks)
        c = blosc2.zeros((N, N), urlpath=cpath, mode="w", chunks=chunks, blocks=blocks)
    if disk:
        a = blosc2.open("a.b2nd")
        b = blosc2.open("b.b2nd")
        c = blosc2.open("c.b2nd")

    expr = ((a**3 + blosc2.sin(c * 2)) < b) & ~(c > 0)

    out = expr.compute()
    assert out.shape == (N, N)
    assert out.dtype == np.bool_
    assert out.schunk.urlpath is None
    np.testing.assert_allclose(out[:], ((a[:] ** 3 + np.sin(c[:] * 2)) < b[:]) & (c[:] > 0))

    if disk:
        blosc2.remove_urlpath("a.b2nd")
        blosc2.remove_urlpath("b.b2nd")
        blosc2.remove_urlpath("c.b2nd")


@pytest.mark.parametrize(
    ("expression", "expected_operands"),
    [
        ("a + b * sin(c) + max(e, axis=1, keepdims=True)", ["a", "b", "c", "e"]),
        ("x + y + z", ["x", "y", "z"]),
        ("sum(sin(a) + b)", ["a", "b"]),
        ("sum(sin(a + c)**2 + cos(b + c)**2 + b) + 1", ["a", "b", "c"]),
        ("func1(a, b) + method1(x)", ["a", "b", "x"]),
        ("u + v * cos(w) + sqrt(x)", ["u", "v", "w", "x"]),
        ("data.mean(axis=0) + sum(data, axis=1)", ["data"]),
        ("a + b + custom_func1(c, d)", ["a", "b", "c", "d"]),
        ("k + l.method1(m, n=3) + max(o, p=q)", ["k", "l", "m", "o", "q"]),
        ("func_with_no_args() + method_with_no_args().attribute", []),
        ("a*b + c/d - e**f + g%h", ["a", "b", "c", "d", "e", "f", "g", "h"]),
        ("single_operand", ["single_operand"]),
        ("func1(arg1, kwarg1=True) + var.method2(arg2, kwarg2=False)", ["arg1", "arg2", "var"]),
    ],
)
def test_get_expr_operands(expression, expected_operands):
    assert blosc2.get_expr_operands(expression) == set(expected_operands)


@pytest.mark.skipif(np.__version__.startswith("1."), reason="NumPy < 2.0 has different casting rules")
@pytest.mark.parametrize(
    "scalar",
    [
        "np.int8(0)",
        "np.uint8(0)",
        "np.int16(0)",
        "np.uint16(0)",
        "np.int32(0)",
        "np.uint32(0)",
        "np.int64(0)",
        "np.float32(0)",
        "np.float64(0)",
        "np.complex64(0)",
        "np.complex128(0)",
    ],
)
@pytest.mark.parametrize(
    ("dtype1", "dtype2"),
    [
        (np.int8, np.int8),
        (np.int8, np.int16),
        (np.int8, np.int32),
        (np.int8, np.int64),
        (np.int8, np.float32),
        (np.int8, np.float64),
        (np.uint16, np.uint16),
        (np.uint16, np.uint32),
        # (np.uint16, np.uint64), # numexpr does not support uint64
        (np.uint16, np.float32),
        # (np.uint16, np.float64),
        # (np.int32, np.int32),
        (np.int32, np.int64),
        (np.float32, np.float32),
        (np.float32, np.float64),
        (np.complex64, np.complex64),
        (np.complex64, np.complex128),
    ],
)
def test_dtype_infer(dtype1, dtype2, scalar):
    shape = (5, 10)
    na = np.linspace(0, 1, np.prod(shape), dtype=dtype1).reshape(shape)
    nb = np.linspace(1, 2, np.prod(shape), dtype=dtype2).reshape(shape)
    a = blosc2.asarray(na)
    b = blosc2.asarray(nb)

    # Using compute()
    expr = blosc2.lazyexpr(f"a + b * {scalar}", operands={"a": a, "b": b})
    nres = na + nb * eval(scalar)
    res = expr.compute()
    np.testing.assert_allclose(res[()], nres)
    assert res.dtype == nres.dtype

    # Using __getitem__
    res = expr[()]
    np.testing.assert_allclose(res, nres)
    assert res.dtype == nres.dtype

    # Check dtype not changed by expression creation (bug fix)
    assert a.dtype == dtype1
    assert b.dtype == dtype2


@pytest.mark.parametrize(
    "cfunc", ["np.int8", "np.int16", "np.int32", "np.int64", "np.float32", "np.float64"]
)
def test_dtype_infer_scalars(cfunc):
    castfunc = eval(cfunc)
    o1 = blosc2.arange(10, dtype=castfunc(1))
    la1 = o1 + castfunc(1)
    res = la1[()]
    n1 = np.arange(10, dtype=castfunc)
    nres = n1 + castfunc(1)
    assert res.dtype == nres.dtype
    np.testing.assert_equal(res, nres)

    expr = f"(o1 + {cfunc}(1))"
    print(expr)
    la2 = blosc2.lazyexpr(expr)
    res = la2[()]
    assert res.dtype == nres.dtype
    np.testing.assert_equal(res, nres)


def test_indices():
    shape = (20,)
    na = np.arange(shape[0])
    a = blosc2.asarray(na)
    expr = a > 1
    # TODO: Implement the indices method for LazyExpr more generally
    with pytest.raises(NotImplementedError):
        expr.indices().compute()


def test_sort():
    shape = (20,)
    na = np.arange(shape[0])
    a = blosc2.asarray(na)
    expr = a > 1
    # TODO: Implement the sort method for LazyExpr more generally
    with pytest.raises(NotImplementedError):
        expr.sort().compute()


def test_listargs():
    # lazyexpr tries to convert [] to slice, but could
    # have problems for arguments which are lists
    shape = (20,)
    na = np.arange(shape[0])
    a = blosc2.asarray(na)
    b = blosc2.asarray(na)
    expr = blosc2.lazyexpr("stack([a, b])")
    np.testing.assert_array_equal(expr[:], np.stack([a[:], b[:]]))


def test_str_constructors():
    shape = (1000, 1)
    chunks = (100, 1)
    a = blosc2.lazyexpr(f"linspace(0, 100, {np.prod(shape)}, shape={shape}, chunks={chunks})")
    assert a.chunks == chunks
    b = blosc2.lazyexpr("a.T")  # this fails unless chunkshape is assigned to a on creation

    b = blosc2.ones((1000, 10))
    a = blosc2.lazyexpr(f"b + linspace(0, 100, {np.prod(shape)}, shape={shape}, chunks={chunks})")
    assert a.shape == np.broadcast_shapes(shape, b.shape)

    # failed before dtype handling improved
    x = blosc2.lazyexpr("linspace(-1, 1, 10, shape=(1, 10))")
    lexpr = blosc2.sin(blosc2.sqrt(x**2))


@pytest.mark.parametrize(
    "obj",
    [
        blosc2.arange(10),
        blosc2.ones(10),
        blosc2.zeros(10),
        blosc2.arange(10) + blosc2.ones(10),
        blosc2.arange(10) + np.ones(10),
        "arange(10)",
        "arange(10) + arange(10)",
        "arange(10) + linspace(0, 1, 10)",
        "arange(10, shape=(10,))",
        "arr",
        "arange(10) + arr",
    ],
)
@pytest.mark.parametrize("getitem", [True, False])
@pytest.mark.parametrize("item", [(), slice(10), slice(0, 10, 2)])
def test_only_ndarrays_or_constructors(obj, getitem, item):
    arr = blosc2.arange(10)  # is a test case
    larr = blosc2.lazyexpr(obj)
    if not isinstance(obj, str):
        assert larr.shape == obj.shape
        assert larr.dtype == obj.dtype
    if getitem:
        b = larr[item]
        assert isinstance(b, np.ndarray)
    else:
        b = larr.compute(item)
        assert isinstance(b, blosc2.NDArray)
    if item == ():
        assert b.shape == larr.shape
    assert b.dtype == larr.dtype
    if not isinstance(obj, str):
        assert np.allclose(b[:], obj[item])


@pytest.mark.parametrize("func", ["cumsum", "cumulative_sum", "cumprod"])
def test_numpy_funcs(array_fixture, func):
    a1, a2, a3, a4, na1, na2, na3, na4 = array_fixture
    try:
        npfunc = getattr(np, func)
        d_blosc2 = npfunc(((a1**3 + blosc2.sin(na2 * 2)) < a3) & (na2 > 0), axis=0)
        d_numpy = npfunc(((na1**3 + np.sin(na2 * 2)) < na3) & (na2 > 0), axis=0)
        np.testing.assert_equal(d_blosc2, d_numpy)
    except AttributeError:
        pytest.skip("NumPy version has no cumulative_sum function.")


# Test the LazyExpr when some operands are missing (e.g. removed file)
def test_missing_operator():
    a = blosc2.arange(10, urlpath="a.b2nd", mode="w")
    b = blosc2.arange(10, urlpath="b.b2nd", mode="w")
    expr = blosc2.lazyexpr("a + b")
    expr.save("expr.b2nd", mode="w")
    # Remove the file for operand b
    blosc2.remove_urlpath("b.b2nd")
    # Re-open the lazy expression
    with pytest.raises(blosc2.exceptions.MissingOperands) as excinfo:
        blosc2.open("expr.b2nd")

    # Check that some operand is missing
    assert "a" not in excinfo.value.missing_ops
    assert excinfo.value.missing_ops["b"] == pathlib.Path("b.b2nd")
    assert excinfo.value.expr == "a + b"

    # Clean up
    blosc2.remove_urlpath("a.b2nd")
    blosc2.remove_urlpath("expr.b2nd")


# Test the chaining of multiple lazy expressions
def test_chain_expressions():
    N = 1_000
    dtype = "float64"
    a = blosc2.linspace(0, 1, N * N, dtype=dtype, shape=(N, N))
    b = blosc2.linspace(1, 2, N * N, dtype=dtype, shape=(N, N))
    c = blosc2.linspace(0, 1, N, dtype=dtype, shape=(N,))

    le1 = a**3 + blosc2.sin(a**2)
    le2 = le1 < c
    le3 = le2 & (b < 0)
    le1_ = blosc2.lazyexpr("a ** 3 + sin(a ** 2)", {"a": a})
    le2_ = blosc2.lazyexpr("(le1 < c)", {"le1": le1_, "c": c})
    le3_ = blosc2.lazyexpr("(le2 & (b < 0))", {"le2": le2_, "b": b})
    assert (le3_[:] == le3[:]).all()

    le1 = a**3 + blosc2.sin(a**2)
    le2 = le1 < c
    le3 = b < 0
    le4 = le2 & le3
    le1_ = blosc2.lazyexpr("a ** 3 + sin(a ** 2)", {"a": a})
    le2_ = blosc2.lazyexpr("(le1 < c)", {"le1": le1_, "c": c})
    le3_ = blosc2.lazyexpr("(b < 0)", {"b": b})
    le4_ = blosc2.lazyexpr("(le2 & le3)", {"le2": le2_, "le3": le3_})
    assert (le4_[:] == le4[:]).all()

    expr1 = blosc2.lazyexpr("arange(N) + b")
    expr2 = blosc2.lazyexpr("a * b + 1")
    expr = blosc2.lazyexpr("expr1 - expr2")
    expr_final = blosc2.lazyexpr("expr * expr")
    nres = (expr * expr)[:]
    res = expr_final.compute()
    np.testing.assert_allclose(res[:], nres)

    # Test that update_expr does not alter expr1
    expr1 = "a + b"
    expr2 = "sin(a) + tan(c)"
    lexpr1 = blosc2.lazyexpr(expr1)
    lexpr2 = blosc2.lazyexpr(expr2)
    lexpr3 = lexpr1 + lexpr2
    assert lexpr1.expression == lexpr1.expression
    assert lexpr1.operands == lexpr1.operands
    assert lexpr2.expression == lexpr2.expression
    assert lexpr2.operands == lexpr2.operands
    lexpr1 += lexpr2
    assert lexpr1.expression == lexpr3.expression
    assert lexpr1.operands == lexpr3.operands

    # chain constructors
    expr1 = "linspace(0, 10, 100)"
    lexpr1 = blosc2.lazyexpr(expr1)
    lexpr1 *= 2
    assert lexpr1.expression == "((linspace(0, 10, 100)) * 2)"
    assert lexpr1.shape == (100,)


# Test the chaining of multiple persistent lazy expressions
def test_chain_persistentexpressions():
    N = 1_000
    dtype = "float64"
    a = blosc2.linspace(0, 1, N * N, dtype=dtype, shape=(N, N), urlpath="a.b2nd", mode="w")
    b = blosc2.linspace(1, 2, N * N, dtype=dtype, shape=(N, N), urlpath="b.b2nd", mode="w")
    c = blosc2.linspace(0, 1, N, dtype=dtype, shape=(N,), urlpath="c.b2nd", mode="w")

    le1 = a**3 + blosc2.sin(a**2)
    le2 = le1 < c
    le3 = le2 & (b < 0)
    le4 = le2 & le3

    le1_ = blosc2.lazyexpr("a ** 3 + sin(a ** 2)", {"a": a})
    le1_.save("expr1.b2nd", mode="w")
    myle1 = blosc2.open("expr1.b2nd")

    le2_ = blosc2.lazyexpr("(le1 < c)", {"le1": myle1, "c": c})
    le2_.save("expr2.b2nd", mode="w")
    myle2 = blosc2.open("expr2.b2nd")

    le3_ = blosc2.lazyexpr("(b < 0)", {"b": b})
    le3_.save("expr3.b2nd", mode="w")
    myle3 = blosc2.open("expr3.b2nd")

    le4_ = blosc2.lazyexpr("(le2 & le3)", {"le2": myle2, "le3": myle3})
    le4_.save("expr4.b2nd", mode="w")
    myle4 = blosc2.open("expr4.b2nd")
    assert (myle4[:] == le4[:]).all()

    # Remove files
    for f in ["expr1.b2nd", "expr2.b2nd", "expr3.b2nd", "expr4.b2nd", "a.b2nd", "b.b2nd", "c.b2nd"]:
        blosc2.remove_urlpath(f)


@pytest.mark.parametrize(
    "values",
    [
        (np.ones(10, dtype=np.uint16), 2),
        (np.ones(10, dtype=np.uint16), np.uint32(2)),
        (2, np.ones(10, dtype=np.uint16)),
        (np.uint32(2), np.ones(10, dtype=np.uint16)),
        (np.ones(10, dtype=np.uint16), 2.0),
        (np.ones(10, dtype=np.float32), 2.0),
        (np.ones(10, dtype=np.float32), 2.0j),
    ],
)
def test_scalar_dtypes(values):
    value1, value2 = values
    dtype1 = (value1 + value2).dtype
    avalue1 = blosc2.asarray(value1) if not np.isscalar(value1) else value1
    avalue2 = blosc2.asarray(value2) if not np.isscalar(value2) else value2
    dtype2 = (avalue1 * avalue2).dtype
    assert dtype1 == dtype2, f"Expected {dtype1} but got {dtype2}"

    # test scalars
    value = value1 if np.isscalar(value1) else value2
    assert blosc2.sin(value)[()] == np.sin(value)
    assert (value + blosc2.sin(value))[()] == value + np.sin(value)


def test_to_cframe():
    N = 1_000
    dtype = "float64"
    a = blosc2.linspace(0, 1, N * N, dtype=dtype, shape=(N, N))
    expr = a**3 + blosc2.sin(a**2)
    cframe = expr.to_cframe()
    assert len(cframe) > 0
    arr = blosc2.ndarray_from_cframe(cframe)
    assert arr.shape == expr.shape
    assert arr.dtype == expr.dtype
    assert np.allclose(arr[:], expr[:])


# Test for the bug where multiplying two complex lazy expressions would fail with:
# ValueError: invalid literal for int() with base 10: '0,'
def test_complex_lazy_expression_multiplication():
    # Create test data similar to the animated plot scenario
    width, height = 64, 64
    x = np.linspace(-4 * np.pi, 4 * np.pi, width)
    y = np.linspace(-4 * np.pi, 4 * np.pi, height)
    X, Y = np.meshgrid(x, y)

    # Convert to blosc2 arrays
    X_b2 = blosc2.asarray(X)
    Y_b2 = blosc2.asarray(Y)

    # Create the complex expressions that were causing the bug
    time_factor = 0.5

    # First complex expression: R * 4 - time_factor * 2
    R = np.sqrt(X_b2**2 + Y_b2**2)
    expr1 = R * 4 - time_factor * 2

    # Second complex expression: theta * 6
    theta = np.arctan2(Y_b2, X_b2)
    expr2 = theta * 6

    # Apply functions to create more complex expressions
    sin_expr = np.sin(expr1)
    cos_expr = np.cos(expr2)

    # This multiplication was failing before the fix
    result_expr = sin_expr * cos_expr

    # Evaluate the expression - this should not raise an error
    result = result_expr.compute()

    # Verify the result matches numpy computation using the same approach
    # Use the blosc2 arrays converted to numpy to ensure consistency
    R_np = np.sqrt(X_b2[:] ** 2 + Y_b2[:] ** 2)
    theta_np = np.arctan2(Y_b2[:], X_b2[:])
    expected = np.sin(R_np * 4 - time_factor * 2) * np.cos(theta_np * 6)

    np.testing.assert_allclose(result, expected, rtol=1e-14, atol=1e-14)

    # Also test getitem access
    np.testing.assert_allclose(result_expr[:], expected, rtol=1e-14, atol=1e-14)


# Test checking that objects following the blosc2.Array protocol can be operated with
def test_minimal_protocol():
    class NewObj:
        def __init__(self, a):
            self.a = a

        @property
        def shape(self):
            return self.a.shape

        @property
        def dtype(self):
            return self.a.dtype

        def __getitem__(self, key):
            return self.a[key]

        def __len__(self):
            return len(self.a)

    a = np.arange(100, dtype=np.int64).reshape(10, 10)
    b = NewObj(a)
    c = blosc2.asarray(a)
    lb = blosc2.lazyexpr("b + c + 1")

    np.testing.assert_array_equal(lb[:], a + a + 1)


def test_not_numexpr():
    shape = (20, 20)
    a = blosc2.linspace(0, 20, num=np.prod(shape), shape=shape)
    b = blosc2.ones((20, 1))
    d_blosc2 = blosc2.evaluate("logaddexp(a, b) + a")
    npa = a[()]
    npb = b[()]
    np.testing.assert_array_almost_equal(d_blosc2, np.logaddexp(npa, npb) + npa)
    # TODO: Implement __add__ etc. for LazyUDF so this line works
    # d_blosc2 = blosc2.evaluate(f"logaddexp(a, b) + clip(a, 6, 12)")
    arr = blosc2.lazyexpr("matmul(a, b)")
    assert isinstance(arr, blosc2.LazyExpr)
    np.testing.assert_array_almost_equal(arr[()], np.matmul(npa, npb))


def test_lazylinalg():
    """
    Test the shape parser for linear algebra funcs
    """
    # --- define base shapes ---
    shapes = {
        "A": (3, 4),
        "B": (4, 5),
        "C": (2, 3, 4),
        "D": (1, 5, 1),
        "x": (10,),
        "y": (10,),
    }
    s = shapes["x"]
    x = blosc2.linspace(0, np.prod(s), shape=s)
    s = shapes["y"]
    y = blosc2.linspace(0, np.prod(s), shape=s)
    s = shapes["A"]
    A = blosc2.linspace(0, np.prod(s), shape=s)
    s = shapes["B"]
    B = blosc2.linspace(0, np.prod(s), shape=s)
    s = shapes["C"]
    C = blosc2.linspace(0, np.prod(s), shape=s)
    s = shapes["D"]
    D = blosc2.linspace(0, np.prod(s), shape=s)

    npx = x[()]
    npy = y[()]
    npA = A[()]
    npB = B[()]
    npC = C[()]
    npD = D[()]

    # --- concat ---
    out = blosc2.lazyexpr("concat((x, y), axis=0)")
    npres = np.concatenate((npx, npy), axis=0)
    assert out.shape == npres.shape
    np.testing.assert_array_almost_equal(out[()], npres)

    # --- diagonal ---
    out = blosc2.lazyexpr("diagonal(A)")
    npres = np.diagonal(npA)
    assert out.shape == npres.shape
    np.testing.assert_array_almost_equal(out[()], npres)

    # --- expand_dims ---
    out = blosc2.lazyexpr("expand_dims(x, axis=0)")
    npres = np.expand_dims(npx, axis=0)
    assert out.shape == npres.shape
    np.testing.assert_array_almost_equal(out[()], npres)

    # --- matmul ---
    out = blosc2.lazyexpr("matmul(A, B)")
    npres = np.matmul(npA, npB)
    assert out.shape == npres.shape
    np.testing.assert_array_almost_equal(out[()], npres)

    # --- matrix_transpose ---
    out = blosc2.lazyexpr("matrix_transpose(A)")
    npres = np.matrix_transpose(npA) if np.__version__.startswith("2.") else npA.T
    assert out.shape == npres.shape
    np.testing.assert_array_almost_equal(out[()], npres)
    out = blosc2.lazyexpr("C.mT")
    npres = C.mT
    assert out.shape == npres.shape
    np.testing.assert_array_almost_equal(out[()], npres)
    out = blosc2.lazyexpr("A.T")
    npres = npA.T
    assert out.shape == npres.shape
    np.testing.assert_array_almost_equal(out[()], npres)

    # --- outer ---
    out = blosc2.lazyexpr("outer(x, y)")
    npres = np.outer(npx, npy)
    assert out.shape == npres.shape
    np.testing.assert_array_almost_equal(out[()], npres)

    # --- permute_dims ---
    out = blosc2.lazyexpr("permute_dims(C, axes=(2,0,1))")
    npres = np.transpose(npC, axes=(2, 0, 1))
    assert out.shape == npres.shape
    np.testing.assert_array_almost_equal(out[()], npres)

    # --- squeeze ---
    out = blosc2.lazyexpr("squeeze(D, axis=-1)")
    npres = np.squeeze(npD, -1)
    assert out.shape == npres.shape
    np.testing.assert_array_almost_equal(out[()], npres)
    out = blosc2.lazyexpr("D.squeeze(axis=-1)")
    npres = np.squeeze(npD, -1)
    assert out.shape == npres.shape
    np.testing.assert_array_almost_equal(out[()], npres)

    # --- stack ---
    out = blosc2.lazyexpr("stack((x, y), axis=0)")
    npres = np.stack((npx, npy), axis=0)
    assert out.shape == npres.shape
    np.testing.assert_array_almost_equal(out[()], npres)
    # --- stack ---
    # repeat with list arg instead of tuple
    out = blosc2.lazyexpr("stack([x, y], axis=0)")
    npres = np.stack((npx, npy), axis=0)
    assert out.shape == npres.shape
    np.testing.assert_array_almost_equal(out[()], npres)

    # --- tensordot ---
    out = blosc2.lazyexpr("tensordot(A, B, axes=1)")  # test with int axes
    npres = np.tensordot(npA, npB, axes=1)
    assert out.shape == npres.shape
    np.testing.assert_array_almost_equal(out[()], npres)
    out = blosc2.lazyexpr("tensordot(A, B, axes=((1,) , (0,)))")  # test with tuple axes
    npres = np.tensordot(npA, npB, axes=((1,), (0,)))
    assert out.shape == npres.shape
    np.testing.assert_array_almost_equal(out[()], npres)

    # --- vecdot ---
    out = blosc2.lazyexpr("vecdot(x, y)")
    npres = npvecdot(npx, npy)
    assert out.shape == npres.shape
    np.testing.assert_array_almost_equal(out[()], npres)

    # --- batched matmul ---
    shapes = {
        "A": (1, 3, 4),
        "B": (3, 4, 5),
    }
    s = shapes["A"]
    A = blosc2.linspace(0, np.prod(s), shape=s)
    npA = A[()]  # actual numpy array
    s = shapes["B"]
    B = blosc2.linspace(0, np.prod(s), shape=s)
    npB = B[()]  # actual numpy array

    out = blosc2.lazyexpr("matmul(A, B)")
    npres = np.matmul(npA, npB)
    assert out.shape == npres.shape
    np.testing.assert_array_almost_equal(out[()], npres)


# Test for issue #503 (LazyArray.compute() should honor out param)
def test_lazyexpr_compute_out():
    # check reductions
    a = blosc2.ones(10)
    out = blosc2.zeros(1)
    lexpr = blosc2.lazyexpr("sum(a)")
    assert lexpr.compute(out=out) is out
    assert out[0] == 10
    assert lexpr.compute() is not out

    # check normal expression
    a = blosc2.ones(10)
    out = blosc2.zeros(10)
    lexpr = blosc2.lazyexpr("sin(a)")
    assert lexpr.compute(out=out) is out
    assert out[0] == np.sin(1)
    assert lexpr.compute() is not out


def test_lazyexpr_2args():
    a = blosc2.ones(10)
    lexpr = blosc2.lazyexpr("sin(a)")
    newexpr = blosc2.hypot(lexpr, 3)
    assert newexpr.expression == "hypot((sin(o0)), 3)"
    assert newexpr.operands["o0"] is a


@pytest.mark.parametrize(
    "xp",
    [torch, np],
)
@pytest.mark.parametrize(
    "dtype",
    ["bool", "int32", "int64", "float32", "float64", "complex128"],
)
def test_simpleproxy(xp, dtype):
    try:
        dtype_ = getattr(xp, dtype) if hasattr(xp, dtype) else np.dtype(dtype)
    except FutureWarning:
        dtype_ = np.dtype(dtype)
    if dtype == "bool":
        blosc_matrix = blosc2.asarray([True, False, False], dtype=np.dtype(dtype), chunks=(2,))
        foreign_matrix = xp.zeros((3,), dtype=dtype_)
        # Create a lazy expression object
        lexpr = blosc2.lazyexpr(
            "(b & a) | (~b)", operands={"a": blosc_matrix, "b": foreign_matrix}
        )  # this does not
        # Compare with numpy computation result
        npb = np.asarray(foreign_matrix)
        npa = blosc_matrix[()]
        res = (npb & npa) | np.logical_not(npb)
    else:
        N = 5
        shape_a = (N, N, N)
        blosc_matrix = blosc2.full(shape=shape_a, fill_value=3, dtype=np.dtype(dtype), chunks=(N // 2,) * 3)
        foreign_matrix = xp.ones(shape_a, dtype=dtype_)
        if dtype == "complex128":
            foreign_matrix += 0.5j
            blosc_matrix = blosc2.full(
                shape=shape_a, fill_value=3 + 2j, dtype=np.dtype(dtype), chunks=(N // 3,) * 3
            )

        # Create a lazy expression object
        lexpr = blosc2.lazyexpr(
            "b + sin(a) + sum(b) - tensordot(a, b, axes=1)",
            operands={"a": blosc_matrix, "b": foreign_matrix},
        )  # this does not
        # Compare with numpy computation result
        npb = np.asarray(foreign_matrix)
        npa = blosc_matrix[()]
        res = npb + np.sin(npa) + np.sum(npb) - np.tensordot(npa, npb, axes=1)

    # Test object metadata and result
    assert isinstance(lexpr, blosc2.LazyExpr)
    assert lexpr.dtype == res.dtype
    assert lexpr.shape == res.shape
    np.testing.assert_array_equal(lexpr[()], res)
