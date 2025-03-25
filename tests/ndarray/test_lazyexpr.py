#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################
import math

import numpy as np
import pytest

import blosc2
from blosc2.lazyexpr import ne_evaluate
from blosc2.ndarray import get_chunks_idx

NITEMS_SMALL = 1_000
NITEMS = 10_000


@pytest.fixture(params=[np.float32, np.float64])
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
    [
        ("NDArray", "scalar"),
        ("NDArray", "NDArray"),
        ("scalar", "NDArray"),
        # ("scalar", "scalar") # Not supported by LazyExpr
    ],
)
def test_arctan2_pow(urlpath, shape_fixture, dtype_fixture, function, value1, value2):  # noqa: C901
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
    else:  # ("scalar", "NDArray")
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
    # Compare the results
    tol = 1e-15 if dtype_fixture == "float64" else 1e-6
    np.testing.assert_allclose(res_lazyexpr[:], res_numexpr, atol=tol, rtol=tol)

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


# Test eval with an item parameter
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

    expr = ((a**3 + blosc2.sin(c * 2)) < b) & (c > 0)

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
@pytest.mark.parametrize("item", [None, slice(10), slice(0, 10, 2)])
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
    if item is None:
        assert b.shape == larr.shape
    assert b.dtype == larr.dtype
    if not isinstance(obj, str):
        assert np.allclose(b[:], obj[item])


@pytest.mark.parametrize("func", ["cumsum", "cumulative_sum", "cumprod"])
def test_numpy_funcs(array_fixture, func):
    a1, a2, a3, a4, na1, na2, na3, na4 = array_fixture
    npfunc = getattr(np, func)
    d_blosc2 = npfunc(((a1**3 + blosc2.sin(na2 * 2)) < a3) & (na2 > 0), axis=0)
    npfunc = getattr(np, func)
    d_numpy = npfunc(((na1**3 + np.sin(na2 * 2)) < na3) & (na2 > 0), axis=0)
    np.testing.assert_equal(d_blosc2, d_numpy)
