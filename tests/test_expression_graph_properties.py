"""Deterministic differential and lifetime probes for the experimental boundary."""

import gc
import importlib
import weakref

import numpy as np
import pytest

import blosc2


def test_value_dependent_numpy_promotion_defers_scalar_metadata(monkeypatch):
    module = importlib.import_module("blosc2.expression_graph")
    monkeypatch.setattr(module, "_NUMPY_VALUE_PROMOTION", True)
    x = np.arange(4, dtype="uint8")
    graph = module.parse_expression("x + scalar")
    assert graph.infer_dtype({"x": x, "scalar": 256}) is None
    assert graph.infer_dtype({"x": x, "scalar": np.array(256)}) is None
    assert graph.infer_dtype({"x": x, "scalar": x}) == np.dtype("uint8")


@pytest.mark.parametrize("dtype", ["int32", "float32", "float64"])
@pytest.mark.parametrize(
    "text",
    [
        "(x + y).sum(axis=0, keepdims=True)",
        "(x / y).mean(axis=1)",
        "sum(x + y, axis=1)",
    ],
)
def test_numpy_intermediate_reduction_metadata_without_execution(monkeypatch, dtype, text):
    module = importlib.import_module("blosc2.lazyexpr")
    operands = {"x": np.arange(1, 33, dtype=dtype).reshape(4, 8), "y": np.full((4, 8), 2, dtype=dtype)}
    trusted = blosc2.lazyexpr(text, operands, evaluation="full")
    expected = trusted[:]
    with monkeypatch.context() as patch:
        patch.setattr(module, "_numpy_eval_expr", lambda *a, **k: pytest.fail("Dummy numerical inference"))
        safe = blosc2.lazyexpr(text, operands, evaluation="safe")
        assert safe.shape == trusted.shape
        assert safe.dtype == trusted.dtype
    np.testing.assert_allclose(safe[:], expected, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("cast", [False, True])
def test_numpy_intermediate_extended_contracts_without_dummies(monkeypatch, cast):
    module = importlib.import_module("blosc2.lazyexpr")
    x = np.arange(1, 33, dtype="float32").reshape(4, 8)
    y = np.full_like(x, 2)
    # The trusted validator/inference does not support these forms. Compare
    # the explicit NumPy contract rather than calling that a parity success.
    text = "(x + y).astype('float64').sum(axis=0)" if cast else "np.std(x + y, axis=0, ddof=1)"
    expected = (x + y).astype("float64").sum(axis=0) if cast else np.std(x + y, axis=0, ddof=1)
    with monkeypatch.context() as patch:
        patch.setattr(module, "_numpy_eval_expr", lambda *a, **k: pytest.fail("Dummy numerical inference"))
        safe = blosc2.lazyexpr(text, {"x": x, "y": y}, evaluation="safe")
        assert safe.shape == expected.shape
        assert safe.dtype == expected.dtype
    np.testing.assert_allclose(safe[:], expected, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("fails", [False, True])
def test_graph_evaluation_releases_operands_without_cyclic_gc(fails):
    from blosc2.expression_graph import parse_expression

    graph = parse_expression("reshape(x, (3,))" if fails else "sqrt(x) + 1")
    enabled = gc.isenabled()
    gc.disable()
    try:
        source = np.arange(4, dtype="float64")
        reference = weakref.ref(source)
        if fails:
            with pytest.raises(ValueError):
                graph.evaluate({"x": source})
        else:
            result = graph.evaluate({"x": source})
            np.testing.assert_allclose(result, np.sqrt(source) + 1)
        del source
        assert reference() is None
    finally:
        if enabled:
            gc.enable()


@pytest.mark.parametrize(
    ("left_backend", "right_backend"),
    [("numpy", "numpy"), ("numpy", "blosc2"), ("blosc2", "numpy"), ("blosc2", "blosc2")],
)
@pytest.mark.parametrize("dtype", ["int32", "float32", "float64"])
@pytest.mark.parametrize(
    "text",
    [
        "(x + y).sum(axis=0)",
        "(x / y).mean(axis=1)",
        "sum(sqrt(x) + y, axis=0)",
        "sqrt(x + y) - mean(y)",
        "(x + y).std(axis=1)",
        "(x + y).var(axis=1)",
    ],
)
def test_mixed_backend_intermediates_match_trusted_contract(left_backend, right_backend, dtype, text):
    data = np.arange(1, 33, dtype=dtype).reshape(4, 8)
    other = np.full((4, 8), 2, dtype=dtype)
    operands = {
        "x": data if left_backend == "numpy" else blosc2.asarray(data),
        "y": other if right_backend == "numpy" else blosc2.asarray(other),
    }
    trusted = blosc2.lazyexpr(text, operands, evaluation="full")
    safe = blosc2.lazyexpr(text, operands, evaluation="safe")
    assert safe.shape == trusted.shape
    assert safe.dtype == trusted.dtype
    np.testing.assert_allclose(safe[:], trusted[:], rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("backend", ["numpy", "blosc2"])
@pytest.mark.parametrize("dtype", ["int8", "uint8", "int32", "float32", "float64"])
@pytest.mark.parametrize("literal", ["1", "0.5", "128", "256"])
def test_weak_scalar_value_and_error_contract_matches_trusted(backend, dtype, literal):
    data = np.arange(1, 5, dtype=dtype)
    operands = {"x": data if backend == "numpy" else blosc2.asarray(data)}

    def evaluate(mode):
        try:
            expr = blosc2.lazyexpr(f"x + {literal}", operands, evaluation=mode)
            return expr.dtype, expr[:], None
        except OverflowError as error:
            return None, None, type(error)

    trusted_dtype, trusted_values, trusted_error = evaluate("full")
    safe_dtype, safe_values, safe_error = evaluate("safe")
    assert safe_error == trusted_error
    if trusted_error is None:
        assert safe_dtype == trusted_dtype
        np.testing.assert_array_equal(safe_values, trusted_values)


def numerical_case(seed, x, y):
    rng = np.random.default_rng(seed)
    text, expected = "x", x
    for _ in range(3):
        operation = int(rng.integers(4))
        if operation == 0:
            text, expected = f"sqrt(({text}) * ({text}) + 1)", np.sqrt(expected * expected + 1)
        elif operation == 1:
            text, expected = f"sin({text})", np.sin(expected)
        elif operation == 2:
            text, expected = f"(({text}) + y)", expected + y
        else:
            text, expected = f"(({text}) * 0.5)", expected * 0.5
    reduction = seed % 4
    if reduction == 0:
        return f"sum({text}, axis=0, keepdims=True)", expected.sum(axis=0, keepdims=True)
    if reduction == 1:
        return f"mean({text}, axis=1)", expected.mean(axis=1)
    if reduction == 2:
        return f"(({text}) - mean(x))", expected - x.mean()
    return text, expected


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("route", ["disk", "frame", "structured"])
def test_seeded_persisted_graph_lifetime_without_python_text_bridge(
    tmp_path, monkeypatch, seed, dtype, route
):
    graph_module = importlib.import_module("blosc2.expression_graph")
    x = np.linspace(1, 3, 32, dtype=dtype).reshape(4, 8)
    y = np.linspace(0.25, 0.5, 8, dtype=dtype)
    text, expected = numerical_case(seed, x, y)
    source = blosc2.asarray(x, urlpath=tmp_path / "x.b2nd", mode="w")
    other = blosc2.asarray(y, urlpath=tmp_path / "y.b2nd", mode="w")
    # NumExpr/Blosc2 accumulation need not share NumPy's intermediate dtype.
    # Capture the existing trusted contract before disabling the text bridge.
    trusted = blosc2.lazyexpr(text, {"x": source, "y": other}, evaluation="full")
    expected_dtype = trusted.dtype
    monkeypatch.setattr(
        graph_module, "eval", lambda *a, **k: pytest.fail("Reached Python text bridge"), raising=False
    )
    expr = blosc2.lazyexpr(text, {"x": source, "y": other}, evaluation="safe")
    if route == "disk":
        expr.save(tmp_path / "expr.b2nd")
        opened = blosc2.open(tmp_path / "expr.b2nd")
    elif route == "frame":
        opened = blosc2.from_cframe(expr.to_cframe())
    else:
        objects = importlib.import_module("blosc2.b2objects")
        opened = objects.decode_b2object_payload(objects.encode_b2object_payload(expr))
    with blosc2.expression_evaluation("full"):
        assert opened.shape == expected.shape
        assert opened.dtype == expected_dtype
        np.testing.assert_allclose(opened[:], expected, rtol=2e-6, atol=2e-6)
        np.testing.assert_allclose(opened[:1], expected[:1], rtol=2e-6, atol=2e-6)
        combined = np.square(opened + 1)
        np.testing.assert_allclose(combined[:], np.square(expected + 1), rtol=4e-6, atol=4e-6)
        source[:] = x + 0.25
        _, changed = numerical_case(seed, x + 0.25, y)
        np.testing.assert_allclose(opened[:], changed, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("text", ["sqrt(x) + 1", "sum(sqrt(x))", "x - mean(x)"])
def test_partial_result_does_not_coerce_whole_source_to_numpy(monkeypatch, text):
    graph_module = importlib.import_module("blosc2.expression_graph")
    monkeypatch.setattr(
        graph_module, "eval", lambda *a, **k: pytest.fail("Python text bridge"), raising=False
    )
    data = np.linspace(1, 4, 8192)
    source = blosc2.asarray(data, chunks=(256,), blocks=(64,))
    original_array = getattr(blosc2.NDArray, "__array__", None)
    original_getitem = blosc2.NDArray.__getitem__
    reads = []

    def array(self, *args, **kwargs):
        if self is source:
            pytest.fail("Whole-source NumPy coercion")
        if original_array is not None:
            return original_array(self, *args, **kwargs)
        return np.asarray(original_getitem(self, ()), *args, **kwargs)

    def getitem(self, item):
        result = original_getitem(self, item)
        if self is source and type(result) is np.ndarray:
            reads.append(result.size)
            assert result.size <= source.chunks[0]
        return result

    monkeypatch.setattr(blosc2.NDArray, "__array__", array, raising=False)
    monkeypatch.setattr(blosc2.NDArray, "__getitem__", getitem)
    expr = blosc2.lazyexpr(text, {"x": source}, evaluation="safe")
    if text == "sum(sqrt(x))":
        np.testing.assert_allclose(expr[()], np.sqrt(data).sum())
    else:
        expected = np.sqrt(data) + 1 if text == "sqrt(x) + 1" else data - data.mean()
        np.testing.assert_allclose(expr[17:41], expected[17:41])
    # This probes Python-visible reads, not all native allocations or schedules.
    assert not reads or max(reads) <= 256


@pytest.mark.parametrize("seed", range(24))
def test_bounded_forbidden_syntax_variations_reject_before_resolution(monkeypatch, seed):
    objects = importlib.import_module("blosc2.b2objects")
    monkeypatch.setattr(
        objects, "decode_operand_mapping", lambda *a, **k: pytest.fail("Resolved forbidden graph")
    )
    forbidden = [
        f"x.__getattribute__('shape{seed}')",
        f"[x for x in range({seed + 1})]",
        f"(lambda x: x + {seed})(x)",
        f"unknown_{seed}(x)",
        f"sum(x, **options_{seed})",
        f"np.lib.function_{seed}(x)",
    ]
    with pytest.raises((blosc2.UnsafeDeserializationError, ValueError)):
        objects.decode_structured_lazyexpr({"expression": forbidden[seed % len(forbidden)], "operands": {}})
