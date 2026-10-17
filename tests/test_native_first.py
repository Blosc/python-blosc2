"""Automatic native selection is conservative and never replays execution."""

import numpy as np
import pytest

import blosc2
from blosc2.native_graph import prepare_native_first
from conftest import orthogonal_cases


@pytest.fixture
def native_first_runtime(native_graph_runtime):
    expr = blosc2.lazyexpr("x + 1", {"x": np.arange(4.0)})
    if prepare_native_first(expr, ()) is None:
        pytest.skip("Automatic native execution requires an available graph JIT")


@pytest.mark.parametrize("dtype", ["int8", "uint64", "float32", "float64", "bool"])
@pytest.mark.parametrize("safe", [False, True])
def test_native_first_arithmetic(dtype, safe, native_first_runtime, monkeypatch):
    x = np.arange(6).reshape(2, 3).astype(dtype)
    y = np.ones((1, 3), dtype=dtype)
    expression = "x == y" if dtype == "bool" else "x + (y - 1)"
    expr = blosc2.lazyexpr(expression, {"x": x, "y": y}, evaluation="safe" if safe else "full")

    def forbidden(*args, **kwargs):
        pytest.fail("Eligible expression used legacy execution")

    monkeypatch.setattr(blosc2.LazyExpr, "_compute_expr", forbidden)
    expected = x == y if dtype == "bool" else x + (y - 1)
    np.testing.assert_array_equal(expr[:], expected)
    assert expr._native_execution_report["backend"] == "portable-jit"
    np.testing.assert_array_equal(expr.compute()[:], expected)
    x[:] = 0
    expected = x == y if dtype == "bool" else x + (y - 1)
    np.testing.assert_array_equal(expr[:], expected)


@pytest.mark.parametrize("expression", ["x[1:3]", "sum(sum(x))", "x and y", "unknown(x)"])
def test_native_first_excluded_syntax(expression):
    from blosc2.native_graph import _native_first_syntax

    assert not _native_first_syntax(expression)


@pytest.mark.parametrize("kind", ["compressed", "float16", "complex", "empty", "non_native"])
def test_native_first_excluded_inputs(kind):
    x = np.arange(4.0)
    if kind == "compressed":
        x = blosc2.asarray(x)
    elif kind == "float16":
        x = x.astype("float16")
    elif kind == "complex":
        x = x.astype("complex128")
    elif kind == "empty":
        x = x[:0]
    else:
        x = x.astype(x.dtype.newbyteorder("S"))
    expr = blosc2.lazyexpr("x + 1", {"x": x})
    assert prepare_native_first(expr, ()) is None


@pytest.mark.parametrize("item", [slice(0, 0), slice(1, 3), 0, [0, 2], (Ellipsis,)])
def test_native_first_partial_selection_stays_legacy(item):
    expr = blosc2.lazyexpr("x + 1", {"x": np.arange(4.0)})
    assert prepare_native_first(expr, item) is None


@pytest.mark.parametrize("options", [{"out": None}, {"_reduce_args": {"op_str": "sum"}}, {"chunks": (2,)}])
def test_native_first_excluded_options(options):
    expr = blosc2.lazyexpr("x + 1", {"x": np.arange(4.0)})
    assert prepare_native_first(expr, (), options=options) is None


def test_native_first_explicit_opt_out(native_graph_runtime, monkeypatch):
    expr = blosc2.lazyexpr("x + 1", {"x": np.arange(4.0)})
    import blosc2.native_graph as native

    def forbidden(*args, **kwargs):
        pytest.fail("Explicit opt-out attempted native preparation")

    monkeypatch.setattr(native, "prepare_native_first", forbidden)
    np.testing.assert_array_equal(expr.compute(_require_native=False)[:], np.arange(4.0) + 1)


def test_native_first_unavailable_runtime_falls_back(monkeypatch):
    import blosc2.native_graph as native

    def unavailable(*args, **kwargs):
        raise NotImplementedError("Old dependency")

    monkeypatch.setattr(native, "lower_native_graph", unavailable)
    expr = blosc2.lazyexpr("x + 1", {"x": np.arange(4.0)})
    np.testing.assert_array_equal(expr[:], np.arange(4.0) + 1)
    with pytest.raises(NotImplementedError, match="Old dependency"):
        expr.compute(_require_native=True)


def test_native_first_execution_failure_is_not_replayed(native_first_runtime, monkeypatch):
    import blosc2.native_graph as native

    def failure(*args, **kwargs):
        raise FloatingPointError("Invocation failed")

    def forbidden(*args, **kwargs):
        pytest.fail("Native execution failure was replayed")

    expr = blosc2.lazyexpr("x + 1", {"x": np.arange(4.0)})
    monkeypatch.setattr(native, "compute_native_graph", failure)
    monkeypatch.setattr(blosc2.LazyExpr, "_compute_expr", forbidden)
    with pytest.raises(FloatingPointError, match="Invocation failed"):
        expr[:]


@pytest.mark.parametrize("status", ["invalid_signature", "unsupported"])
def test_native_first_unsupported_preparation_falls_back(status, monkeypatch):
    import blosc2.native_graph as native

    def unsupported(*args, **kwargs):
        raise blosc2.PortableArtifactError("Not supported", status=status)

    monkeypatch.setattr(native, "lower_native_graph", unsupported)
    expr = blosc2.lazyexpr("x + 1", {"x": np.arange(4.0)})
    np.testing.assert_array_equal(expr[:], np.arange(4.0) + 1)


def test_native_first_unexpected_preparation_error_propagates(monkeypatch):
    import blosc2.native_graph as native

    def failure(*args, **kwargs):
        raise blosc2.PortableArtifactError("Out of memory", status="out_of_memory")

    monkeypatch.setattr(native, "lower_native_graph", failure)
    expr = blosc2.lazyexpr("x + 1", {"x": np.arange(4.0)})
    with pytest.raises(blosc2.PortableArtifactError, match="Out of memory"):
        expr.compute()


def test_native_first_preserves_backend_policy(monkeypatch):
    expr = blosc2.lazyexpr("x + 1", {"x": np.arange(4.0)})
    assert prepare_native_first(expr, (), jit_backend="tcc") is None
    assert prepare_native_first(expr, (), fp_accuracy=blosc2.FPAccuracy.HIGH) is None
    called = []
    original = blosc2.LazyExpr._compute_expr

    def legacy(self, item, kwargs):
        called.append(kwargs["jit_backend"])
        # The assertion is about dispatch, independent of compiler availability.
        kwargs.pop("jit_backend")
        kwargs.pop("jit", None)
        return original(self, item, kwargs)

    monkeypatch.setattr(blosc2.LazyExpr, "_compute_expr", legacy)
    monkeypatch.setenv("BLOSC_ME_JIT", "tcc")
    np.testing.assert_array_equal(expr[:], np.arange(4.0) + 1)
    assert called == ["tcc"]


def test_native_first_empty_slice_and_excess_indices():
    expr = blosc2.lazyexpr("x + 1", {"x": np.arange(4.0).reshape(1, 4)})
    assert expr[:0].shape == (0, 4)
    with pytest.raises(IndexError):
        expr[:, :, :]


def test_native_first_report_cleared_on_opt_out(native_first_runtime):
    expr = blosc2.lazyexpr("x + 1", {"x": np.arange(4.0)})
    expr.compute()
    assert hasattr(expr, "_native_execution_report")
    expr.compute(_require_native=False)
    assert not hasattr(expr, "_native_execution_report")


@pytest.mark.parametrize(
    ("dtype", "expression"),
    orthogonal_cases(
        ["int16", "int32", "int64", "uint8", "uint16", "uint32"], ["x + 1", "-x", "x * 2", "x < 3", "x & 3"]
    ),
)
def test_native_first_numeric_coverage(dtype, expression, native_first_runtime):
    x = np.arange(6).astype(dtype)
    expr = blosc2.lazyexpr(expression, {"x": x})
    expected = eval(expression, {"x": x})  # fixed test expressions only
    np.testing.assert_array_equal(expr[:], expected)
    assert expr._native_execution_report["backend"] == "portable-jit"


@pytest.mark.parametrize("layout", ["strided", "fortran", "scalar"])
def test_native_first_layout(layout, native_first_runtime):
    x = np.arange(12.0).reshape(3, 4)
    if layout == "strided":
        x = x[::-1, ::2]
    elif layout == "fortran":
        x = np.asfortranarray(x)
    else:
        x = np.array(3.0)
    expr = blosc2.lazyexpr("x * 2 + 1", {"x": x})
    # A zero-dimensional expression has no axis to slice.
    result = expr.compute()
    np.testing.assert_array_equal(result[()], x * 2 + 1)
    assert hasattr(expr, "_native_execution_report")


def test_native_first_jit_request(native_first_runtime):
    x = np.arange(8.0)
    expr = blosc2.lazyexpr("x + 1", {"x": x})
    np.testing.assert_array_equal(expr.compute(jit=True)[:], x + 1)
    assert expr._native_execution_report["backend"] == "portable-jit"


def test_native_first_unavailable_jit_uses_interpreter(native_graph_runtime, monkeypatch):
    import blosc2.native_graph as native

    original = native.lower_native_graph

    def interpreted(expr, **kwargs):
        return original(expr, jit=False)

    monkeypatch.setattr(native, "lower_native_graph", interpreted)
    expr = blosc2.lazyexpr("x + 1", {"x": np.arange(4.0)})
    np.testing.assert_array_equal(expr[:], np.arange(4.0) + 1)
    assert expr._native_execution_report["backend"] == "portable-interpreter"


def test_native_first_jit_false_uses_interpreter(native_graph_runtime, monkeypatch):
    import blosc2.native_graph as native

    original = native.lower_native_graph
    calls = []

    def interpreted(*args, **kwargs):
        calls.append(kwargs.get("jit"))
        return original(*args, **kwargs)

    monkeypatch.setattr(native, "lower_native_graph", interpreted)
    expr = blosc2.lazyexpr("x + 1", {"x": np.arange(4.0)})
    np.testing.assert_array_equal(expr.compute(jit=False)[:], np.arange(4.0) + 1)
    assert calls
    assert not any(calls)
    assert expr._native_execution_report["backend"] == "portable-interpreter"


@pytest.mark.parametrize("compressed", [False, True])
@pytest.mark.parametrize("scoped", [False, True])
@pytest.mark.parametrize(
    "policy",
    [{"jit": False}, {"jit_backend": "tcc"}, {"fp_accuracy": blosc2.FPAccuracy.HIGH}],
)
def test_native_first_inherited_policy_retains_legacy(monkeypatch, compressed, scoped, policy):
    import blosc2.native_graph as native

    monkeypatch.delenv("BLOSC_ME_JIT", raising=False)
    values = np.arange(32.0)
    operand = blosc2.asarray(values) if compressed else values
    expr = blosc2.lazyexpr("x + 1", {"x": operand})

    original = native.lower_native_graph
    calls = []

    def record(*args, **kwargs):
        calls.append(kwargs.get("jit"))
        return original(*args, **kwargs)

    monkeypatch.setattr(native, "lower_native_graph", record)
    previous = blosc2.get_jit_options()
    try:
        if scoped:
            with blosc2.jit_options(**policy):
                result = expr.compute()
        else:
            blosc2.set_jit_options(**policy)
            result = expr.compute()
        np.testing.assert_array_equal(result[:], values + 1)
        if policy == {"jit": False} and hasattr(expr, "_native_execution_report"):
            assert expr._native_execution_report["backend"] == "portable-interpreter"
        elif policy != {"jit": False}:
            assert not hasattr(expr, "_native_execution_report")
    finally:
        blosc2.set_jit_options(**previous)


@pytest.mark.parametrize(
    ("op", "axis", "keepdims"),
    orthogonal_cases(["sum", "prod", "min", "max", "any", "all"], [None, 0, -1, (), (0, 1)], [False, True]),
)
def test_native_first_reductions(op, axis, keepdims, native_first_runtime, monkeypatch):
    x = np.arange(1, 7.0).reshape(2, 3)
    expr = blosc2.lazyexpr("x + 1", {"x": x})

    def forbidden(*args, **kwargs):
        pytest.fail("Eligible reduction used legacy execution")

    monkeypatch.setattr(blosc2.LazyExpr, "_compute_expr", forbidden)
    result = getattr(expr, op)(axis=axis, keepdims=keepdims)
    expected = getattr(np, op)(x + 1, axis=axis, keepdims=keepdims)
    np.testing.assert_allclose(result, expected)
    assert result.shape == expected.shape
    assert result.dtype == expected.dtype
    assert expr._native_execution_report["jit_stages"] == 1


@pytest.mark.parametrize("expression", ["sum(x + 1)", "(x + 1).sum(axis=-1, keepdims=True)"])
def test_native_first_root_reductions(expression, native_first_runtime, monkeypatch):
    x = np.arange(6.0).reshape(2, 3)
    expr = blosc2.lazyexpr(expression, {"x": x}, evaluation="safe")
    expected = expr.compute(_require_native=True)

    def forbidden(*args, **kwargs):
        pytest.fail("Root reduction used legacy execution")

    monkeypatch.setattr(blosc2.LazyExpr, "_compute_expr", forbidden)
    np.testing.assert_array_equal(expr.compute(), expected)
    assert expr._native_execution_report["jit_stages"] == 1


@pytest.mark.parametrize("dtype", ["float32", "float64", "int32", "uint32"])
def test_native_first_reduction_dtype_initial(dtype, native_first_runtime):
    x = np.arange(6).astype(dtype)
    expr = blosc2.lazyexpr("x + x", {"x": x})
    result = expr.sum(dtype="float64", initial=3)
    np.testing.assert_array_equal(result, np.sum(x + x, dtype="float64", initial=3))
    assert result.dtype == np.dtype("float64")
    assert hasattr(expr, "_native_execution_report")


@pytest.mark.parametrize("keepdims", [False, True])
def test_native_first_parallel_sum_and_opt_out(keepdims, native_first_runtime):
    previous = blosc2.set_nthreads(2)
    try:
        x = np.linspace(0, 1, 4 * 128 * 1024).reshape(4, -1)
        expr = blosc2.lazyexpr("x + 1", {"x": x})
        result = expr.sum(keepdims=keepdims)
        np.testing.assert_allclose(result, np.sum(x + 1, keepdims=keepdims), rtol=1e-12)
        assert expr._native_execution_report["route"] == "parallel-resident-reduction"
        assert expr._native_execution_report["threads"] == 2
        serial = expr.sum(keepdims=keepdims, _require_native=True, jit=True)
        np.testing.assert_allclose(result, serial, rtol=1e-12)
        assert expr._native_execution_report.get("route") != "parallel-resident-reduction"
        legacy = expr.sum(keepdims=keepdims, _require_native=False)
        np.testing.assert_allclose(result, legacy, rtol=1e-12)
        assert not hasattr(expr, "_native_execution_report")
    finally:
        blosc2.set_nthreads(previous)


@pytest.mark.parametrize("case", ["broadcast", "strided", "initial", "axis", "single_thread"])
def test_native_first_sum_keeps_serial_when_not_parallel_eligible(case, native_first_runtime):
    previous = blosc2.set_nthreads(1 if case == "single_thread" else 2)
    try:
        x = np.ones((4, 128 * 1024))
        if case == "strided":
            x = x[:, ::-1]
        y = np.ones((1, x.shape[1])) if case == "broadcast" else np.ones_like(x)
        expr = blosc2.lazyexpr("x + y", {"x": x, "y": y})
        options = {"initial": 3} if case == "initial" else {"axis": 0} if case == "axis" else {}
        result = expr.sum(**options)
        np.testing.assert_array_equal(result, np.sum(x + y, **options))
        assert expr._native_execution_report.get("route") != "parallel-resident-reduction"
    finally:
        blosc2.set_nthreads(previous)


def test_native_first_parallel_sum_failure_is_not_replayed(native_first_runtime, monkeypatch):
    import blosc2.resident_execution as resident

    class FailedPool:
        def submit(self, *args, **kwargs):
            from concurrent.futures import Future

            future = Future()
            future.set_exception(FloatingPointError("Reduction worker failed"))
            return future

    def forbidden(*args, **kwargs):
        pytest.fail("Reduction failure was replayed")

    previous = blosc2.set_nthreads(2)
    try:
        monkeypatch.setattr(resident, "resident_pool", lambda: FailedPool())
        monkeypatch.setattr(blosc2.LazyExpr, "_compute_expr", forbidden)
        expr = blosc2.lazyexpr("x + 1", {"x": np.ones(4 * 128 * 1024)})
        with pytest.raises(FloatingPointError, match="Reduction worker failed"):
            expr.sum()
    finally:
        blosc2.set_nthreads(previous)


def test_native_first_parallel_sum_combination_status(native_first_runtime):
    previous = blosc2.set_nthreads(2)
    try:
        x = np.zeros(4 * 128 * 1024)
        x[0] = x[x.size // 2] = np.finfo("float64").max * 0.75
        expr = blosc2.lazyexpr("x + 0", {"x": x})
        assert np.isposinf(expr.sum())
        assert expr._native_execution_report["fp_flags"] & 4
        assert expr._native_execution_report["route"] == "parallel-resident-reduction"
    finally:
        blosc2.set_nthreads(previous)


def test_native_first_small_compressed_reduction_stays_legacy(native_first_runtime):
    x = blosc2.asarray(np.arange(100.0))
    expr = blosc2.lazyexpr("x + 1", {"x": x})
    assert expr.sum() == 5050
    assert not hasattr(expr, "_native_execution_report")
