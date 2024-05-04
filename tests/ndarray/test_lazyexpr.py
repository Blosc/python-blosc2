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
NITEMS = 50_000


@pytest.fixture(params=[np.float32, np.float64])
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
    nres = ne.evaluate("na1 + na2 - na3 * na4")
    sl = slice(100)
    res = expr[sl]
    np.testing.assert_allclose(res, nres[sl])


# Add more test functions to test different aspects of the code
def test_simple_expression(array_fixture):
    a1, a2, a3, a4, na1, na2, na3, na4 = array_fixture
    expr = a1 + a2 - a3 * a4
    nres = ne.evaluate("na1 + na2 - na3 * na4")
    res = expr.eval()
    np.testing.assert_allclose(res[:], nres)


def test_iXXX(array_fixture):
    a1, a2, a3, a4, na1, na2, na3, na4 = array_fixture
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
    a1, a2, a3, a4, na1, na2, na3, na4 = array_fixture
    expr = blosc2.tan(a1) * (blosc2.sin(a2) * blosc2.sin(a2) + blosc2.cos(a3)) + (blosc2.sqrt(a4) * 2)
    expr += 2
    nres = ne.evaluate("tan(na1) * (sin(na2) * sin(na2) + cos(na3)) + (sqrt(na4) * 2) + 2")
    res = expr.eval()
    np.testing.assert_allclose(res[:], nres)


def test_complex_getitem(array_fixture):
    a1, a2, a3, a4, na1, na2, na3, na4 = array_fixture
    expr = blosc2.tan(a1) * (blosc2.sin(a2) * blosc2.sin(a2) + blosc2.cos(a3)) + (blosc2.sqrt(a4) * 2)
    expr += 2
    nres = ne.evaluate("tan(na1) * (sin(na2) * sin(na2) + cos(na3)) + (sqrt(na4) * 2) + 2")
    res = expr[:]
    np.testing.assert_allclose(res, nres)


def test_complex_getitem_slice(array_fixture):
    a1, a2, a3, a4, na1, na2, na3, na4 = array_fixture
    expr = blosc2.tan(a1) * (blosc2.sin(a2) * blosc2.sin(a2) + blosc2.cos(a3)) + (blosc2.sqrt(a4) * 2)
    expr += 2
    nres = ne.evaluate("tan(na1) * (sin(na2) * sin(na2) + cos(na3)) + (sqrt(na4) * 2) + 2")
    sl = slice(100)
    res = expr[sl]
    np.testing.assert_allclose(res, nres[sl])


def test_expression_with_constants(array_fixture):
    a1, a2, a3, a4, na1, na2, na3, na4 = array_fixture
    # Test with operands with same chunks and blocks
    expr = a1 + 2 - a3 * 3.14
    nres = ne.evaluate("na1 + 2 - na3 * 3.14")
    np.testing.assert_allclose(expr[:], nres)


@pytest.mark.parametrize("compare_expressions", [True, False])
@pytest.mark.parametrize("comparison_operator", ["==", "!=", ">=", ">", "<=", "<"])
def test_comparison_operators(dtype_fixture, compare_expressions, comparison_operator):
    reshape = [30, 4]
    nelems = np.prod(reshape)
    cparams = {"clevel": 0, "codec": blosc2.Codec.LZ4}  # Compression parameters
    na1 = np.linspace(0, 10, nelems, dtype=dtype_fixture).reshape(reshape)
    na2 = np.copy(na1)  # noqa: F841
    a1 = blosc2.asarray(na1, cparams=cparams)
    a2 = blosc2.asarray(na1, cparams=cparams)
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
def test_functions(function, dtype_fixture, shape_fixture):
    nelems = np.prod(shape_fixture)
    cparams = {"clevel": 0, "codec": blosc2.Codec.LZ4}  # Compression parameters
    na1 = np.linspace(0, 10, nelems, dtype=dtype_fixture).reshape(shape_fixture)
    a1 = blosc2.asarray(na1, cparams=cparams)
    # Construct the lazy expression based on the function name
    expr = blosc2.LazyExpr(new_op=(a1, function, None))
    res_lazyexpr = expr.eval()
    # Evaluate using NumExpr
    expr_string = f"{function}(na1)"
    res_numexpr = ne.evaluate(expr_string)
    # Compare the results
    np.testing.assert_allclose(res_lazyexpr[:], res_numexpr)


@pytest.mark.parametrize(
    "function",
    ["arctan2", "**"],
)
@pytest.mark.parametrize(
    "value1, value2",
    [
        ("NDArray", "scalar"),
        ("NDArray", "NDArray"),
        ("scalar", "NDArray"),
        # ("scalar", "scalar") # Not supported by LazyExpr
    ],
)
def test_arctan2_pow(shape_fixture, dtype_fixture, function, value1, value2):
    nelems = np.prod(shape_fixture)
    if value1 == "NDArray":  # ("NDArray", "scalar"), ("NDArray", "NDArray")
        na1 = np.linspace(0, 10, nelems, dtype=dtype_fixture).reshape(shape_fixture)
        a1 = blosc2.asarray(na1)
        if value2 == "NDArray":  # ("NDArray", "NDArray")
            na2 = np.linspace(0, 10, nelems, dtype=dtype_fixture).reshape(shape_fixture)
            a2 = blosc2.asarray(na1)
            # Construct the lazy expression based on the function name
            expr = blosc2.LazyExpr(new_op=(a1, function, a2))
            res_lazyexpr = expr.eval()
            # Evaluate using NumExpr
            if function == "**":
                res_numexpr = ne.evaluate("na1**na2")
            else:
                expr_string = f"{function}(na1, na2)"
                res_numexpr = ne.evaluate(expr_string)
        else:  # ("NDArray", "scalar")
            value2 = 3
            # Construct the lazy expression based on the function name
            expr = blosc2.LazyExpr(new_op=(a1, function, value2))
            res_lazyexpr = expr.eval()
            # Evaluate using NumExpr
            if function == "**":
                res_numexpr = ne.evaluate("na1**value2")
            else:
                expr_string = f"{function}(na1, value2)"
                res_numexpr = ne.evaluate(expr_string)
    else:  # ("scalar", "NDArray")
        value1 = 12
        na2 = np.linspace(0, 10, nelems, dtype=dtype_fixture).reshape(shape_fixture)
        a2 = blosc2.asarray(na2)
        # Construct the lazy expression based on the function name
        expr = blosc2.LazyExpr(new_op=(value1, function, a2))
        res_lazyexpr = expr.eval()
        # Evaluate using NumExpr
        if function == "**":
            res_numexpr = ne.evaluate("value1**na2")
        else:
            expr_string = f"{function}(value1, na2)"
            res_numexpr = ne.evaluate(expr_string)
    # Compare the results
    tolerancy = 1e-15 if dtype_fixture == "float64" else 1e-6
    np.testing.assert_allclose(res_lazyexpr[:], res_numexpr, atol=tolerancy, rtol=tolerancy)


def test_abs(shape_fixture, dtype_fixture):
    nelems = np.prod(shape_fixture)
    na1 = np.linspace(-1, 1, nelems, dtype=dtype_fixture).reshape(shape_fixture)
    a1 = blosc2.asarray(na1)
    expr = blosc2.LazyExpr(new_op=(a1, "abs", None))
    res_lazyexpr = expr.eval()
    res_np = np.abs(na1)
    np.testing.assert_allclose(res_lazyexpr[:], res_np)


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
            res_numexpr = ne.evaluate(expr_numexpr)
        else:  # ("NDArray", "NDArray")
            a2 = np.array([b"abc", b"ab c", b" abc", b" abc ", b"\tabc", b"c h"])
            a2_blosc = blosc2.asarray(a2)
            # Construct the lazy expression
            expr_lazy = blosc2.LazyExpr(new_op=(a1_blosc, "contains", a2_blosc))
            # Evaluate using NumExpr
            res_numexpr = ne.evaluate("contains(a2, a1)")
    else:  # ("str", "NDArray")
        value1 = b"abc"
        a2 = np.array([b"abc", b"def", b"aterr", b"oot", b"zu", b"ab c"])
        a2_blosc = blosc2.asarray(a2)
        # Construct the lazy expression
        expr_lazy = blosc2.LazyExpr(new_op=(value1, "contains", a2_blosc))
        # Evaluate using NumExpr
        res_numexpr = ne.evaluate("contains(value1, a2)")
    res_lazyexpr = expr_lazy.eval()
    # Compare the results
    np.testing.assert_array_equal(res_lazyexpr[:], res_numexpr)
