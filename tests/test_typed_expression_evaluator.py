"""The rollout path may use typed native execution or NumPy, never uniform typing."""

import importlib

import numpy as np
import pytest

import blosc2


@pytest.fixture(autouse=True)
def typed_route(monkeypatch):
    monkeypatch.setenv("BLOSC2_EXPRESSION_EVALUATOR", "typed")

    def forbidden(*args, **kwargs):
        pytest.fail("Ordinary typed expressions reached the uniform-dtype evaluator")

    monkeypatch.setattr(blosc2.NDArray, "_set_pref_expr", forbidden)


@pytest.mark.parametrize("force_wasm", [False, True])
@pytest.mark.parametrize("case", ["division", "hypot", "nextafter", "float_condition", "int_condition"])
def test_typed_map_and_reduction_semantics(monkeypatch, force_wasm, case):
    # This exercises WASM's dispatch restrictions on the host, not the actual
    # wasm32 ABI. The same test also runs in the real Pyodide test job.
    if force_wasm:
        monkeypatch.setattr(blosc2, "IS_WASM", True)
    na = np.arange(1, 25, dtype=np.int64).reshape(4, 6)
    if case == "float_condition":
        na = np.full((4, 6), 0.1, dtype=np.float32)
    elif case == "int_condition":
        na = 2**53 + na
    a = blosc2.asarray(na, chunks=(2, 6), blocks=(1, 3))
    b = blosc2.asarray(np.full_like(na, 2), chunks=(2, 6), blocks=(1, 3))
    if case == "division":
        expr, expected = (a < 5) / b, (na < 5) / np.full_like(na, 2)
    elif case == "hypot":
        expr, expected = blosc2.hypot(a + 1, b), np.hypot(na + 1, np.full_like(na, 2))
    elif case == "nextafter":
        expr, expected = blosc2.nextafter(a + 1, b), np.nextafter(na + 1, np.full_like(na, 2))
    elif case == "float_condition":
        expr, expected = blosc2.where(a > 0.1, 1, 0), np.where(na > 0.1, 1, 0)
    else:
        expr, expected = a == 2**53 + 1, na == 2**53 + 1
    assert expr.dtype == expected.dtype
    for result in (expr[:], expr.compute()[:], expr[:2]):
        np.testing.assert_array_equal(result, expected[:2] if result.shape == (2, 6) else expected)
    np.testing.assert_allclose(expr.sum(), expected.sum())
    np.testing.assert_allclose(expr.sum(axis=0), expected.sum(axis=0))


@pytest.mark.parametrize("operation", ["compare", "add"])
def test_typed_strong_and_weak_scalar(operation):
    na = np.full(24, 0.1, dtype=np.float32)
    a = blosc2.asarray(na, chunks=(12,), blocks=(4,))
    for scalar in (0.1, np.float64(0.1), np.array(0.1, dtype=np.float64)):
        if operation == "compare":
            expr, expected = (a + 0) > scalar, (na + 0) > scalar
        else:
            expr, expected = (a + 0) + scalar, (na + 0) + scalar
        assert expr.dtype == expected.dtype
        np.testing.assert_array_equal(expr[:], expected)
        np.testing.assert_array_equal(expr.compute()[:], expected)
        np.testing.assert_array_equal(expr.sum(), expected.sum())


def test_typed_large_unsigned_branch():
    na = np.arange(24, dtype=np.uint64)
    a = blosc2.asarray(na, chunks=(12,), blocks=(4,))
    big = np.uint64(2**63 + 1)
    expr = blosc2.where(a > 5, big, a)
    expected = np.where(na > 5, big, na)
    np.testing.assert_array_equal(expr[:], expected)
    np.testing.assert_array_equal(expr.compute()[:], expected)


def test_typed_weak_scalar_overflow_preserves_error_contract():
    if np.__version__.startswith("1."):
        pytest.skip("NumPy 1 has value-based scalar promotion")
    a = blosc2.asarray(np.arange(4, dtype=np.uint8))
    with pytest.raises(OverflowError):
        (a + 256)[:]


@pytest.mark.parametrize("layout", ["C", "F", "strided"])
def test_typed_output_contract(layout):
    na = np.arange(99, dtype=np.float32).reshape(9, 11)
    a = blosc2.asarray(na, chunks=(4, 6), blocks=(2, 3))
    storage = np.full((18, 22), -99, dtype=np.float32)
    out = storage[::2, ::2] if layout == "strided" else np.empty_like(na, order=layout)
    result = (a + 1).compute(out=out)
    assert result is out
    np.testing.assert_array_equal(out, na + 1)
    if layout == "strided":
        np.testing.assert_array_equal(storage[1::2], -99)
        np.testing.assert_array_equal(storage[::2, 1::2], -99)


@pytest.mark.parametrize(
    ("shape", "chunks", "blocks", "direct"),
    [
        ((7, 8, 10), (3, 8, 10), (1, 4, 10), True),
        ((4, 12, 30), (1, 4, 30), (1, 2, 30), True),
        ((4, 10, 30), (1, 4, 30), (1, 2, 30), False),
        ((12, 10, 9), (4, 5, 9), (2, 5, 3), False),
    ],
)
def test_typed_flat_chunk_prefilter(monkeypatch, shape, chunks, blocks, direct):
    if (
        blosc2.IS_WASM
        or not getattr(blosc2.blosc2_ext, "menudet_prefilter_flat_chunks_available", lambda: False)()
    ):
        pytest.skip("Rebuilt native flat-chunk prefilter bindings are required")
    values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    a = blosc2.asarray(values, chunks=chunks, blocks=blocks)
    b = blosc2.asarray(values + 1, chunks=chunks, blocks=blocks)
    inputs = {"a": a, "b": b}
    assert blosc2.blosc2_ext.menudet_prefilter_inputs_eligible(inputs, output_dtype=np.float32) is direct
    original = blosc2.NDArray._execute_menudet_prefilter
    calls = []

    def record(self, *args, **kwargs):
        calls.append(True)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(blosc2.NDArray, "_execute_menudet_prefilter", record)
    expr = a + b * 2
    expected = values + (values + 1) * 2
    np.testing.assert_array_equal(expr[:], expected)
    np.testing.assert_array_equal(expr.compute()[:], expected)
    assert bool(calls) is direct


def test_typed_unsupported_signature_uses_numpy(monkeypatch):
    native = importlib.import_module("blosc2.native_graph")
    values = np.arange(24, dtype=np.float64)
    a = blosc2.asarray(values, chunks=(12,), blocks=(4,))
    expr = a + 1

    def unsupported(*args, **kwargs):
        raise blosc2.MenudetError("Not supported", status="unsupported")

    monkeypatch.setattr(native, "lower_native_graph", unsupported)
    np.testing.assert_array_equal(expr.compute(_require_native=False)[:], values + 1)


@pytest.mark.skipif(blosc2.IS_WASM, reason="Typed compression prefilters are not admitted on WASM yet")
@pytest.mark.skipif(np.__version__.startswith("1."), reason="NumPy 1 uses the typed NumPy fallback")
def test_typed_prefilter_failure_is_not_replayed(monkeypatch, native_graph_runtime):
    values = np.arange(24, dtype=np.float64)
    a = blosc2.asarray(values, chunks=(12,), blocks=(4,))
    expr = a + 1
    output = np.full_like(values, -99)

    def failure(self, *args, **kwargs):
        output[0] = 123
        raise RuntimeError("Typed prefilter execution failed")

    def no_replay(*args, **kwargs):
        pytest.fail("Native execution was replayed through a chunk evaluator")

    monkeypatch.setattr(blosc2.NDArray, "_execute_menudet_prefilter", failure)
    monkeypatch.setattr(importlib.import_module("blosc2.lazyexpr"), "evaluate_chunk", no_replay)
    with pytest.raises(RuntimeError, match="Typed prefilter execution failed"):
        expr.compute(out=output, _require_native=False)
    assert output[0] == 123
    np.testing.assert_array_equal(output[1:], -99)


def test_invalid_evaluator_selector(monkeypatch):
    monkeypatch.setenv("BLOSC2_EXPRESSION_EVALUATOR", "typo")
    with pytest.raises(ValueError, match="BLOSC2_EXPRESSION_EVALUATOR"):
        blosc2.asarray(np.arange(4.0)) + 1


@pytest.mark.parametrize(
    ("dtype", "op", "explicit_dtype"),
    [
        (np.float32, "sum", None),
        (np.float32, "sum", np.float64),
        (np.float64, "prod", None),
        (np.int8, "sum", None),
        (np.int8, "sum", np.int8),
        (np.int64, "sum", None),
        (np.uint64, "sum", None),
        (np.float32, "min", None),
        (np.int16, "max", None),
        (np.int32, "any", None),
        (np.int32, "all", None),
        (np.float32, "cmp_any", None),
        (np.float32, "cmp_all", None),
        (np.float32, "mean", None),
        (np.float32, "cmp_sum", None),
    ],
)
def test_typed_scalar_reduction_prefilter(monkeypatch, dtype, op, explicit_dtype):
    if (
        blosc2.IS_WASM
        or np.__version__.startswith("1.")
        or not getattr(blosc2.blosc2_ext, "menudet_prefilter_reductions_available", lambda: False)()
    ):
        pytest.skip("Rebuilt native prefilter reduction bindings are required")
    shape, chunks, blocks = (5, 6, 10), (2, 6, 10), (1, 3, 10)
    rng = np.random.default_rng(7)
    if np.issubdtype(dtype, np.integer):
        na = rng.integers(1, 5, size=shape, dtype=dtype)
        nb = rng.integers(1, 3, size=shape, dtype=dtype)
    else:
        na = rng.random(shape, dtype=np.float64).astype(dtype) + 0.5
        nb = rng.random(shape, dtype=np.float64).astype(dtype) + 0.5
    a = blosc2.asarray(na, chunks=chunks, blocks=blocks)
    b = blosc2.asarray(nb, chunks=chunks, blocks=blocks)
    original = blosc2.NDArray._execute_menudet_prefilter
    reports = []

    def record(self, *args, **kwargs):
        result, report = original(self, *args, **kwargs)
        reports.append(report)
        return result, report

    monkeypatch.setattr(blosc2.NDArray, "_execute_menudet_prefilter", record)
    if op.startswith("cmp_"):
        expr, nexpr = a > b, na > nb
        method, kwargs = op.removeprefix("cmp_"), {}
    else:
        expr, nexpr = a + b, na + nb
        method = op
        kwargs = {} if explicit_dtype is None else {"dtype": explicit_dtype}
    got = getattr(expr, method)(**kwargs)
    expected = getattr(nexpr, method)(**kwargs)
    assert np.dtype(type(got)) == np.dtype(type(expected))
    if np.issubdtype(expected.dtype, np.floating):
        np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)
    else:
        np.testing.assert_array_equal(got, expected)
    assert len(reports) == 1
    assert reports[0]["output_destination"] == "reduction"
    assert reports[0]["evaluated_blocks"] == 10


@pytest.mark.parametrize(
    ("op", "values"),
    [
        ("sum", np.array([1, -1], dtype=np.int64)),
        ("sum", np.zeros(8, dtype=np.int64)),
        ("prod", np.full(64, 2, dtype=np.uint64)),
        ("prod", np.array([1, 0, -1], dtype=np.int64)),
        ("sum", np.array([0.0, np.nan], dtype=np.float64)),
        ("prod", np.array([1.0, -1.0, np.nan], dtype=np.float64)),
    ],
)
def test_typed_boolean_accumulator_uses_logical_reduction(monkeypatch, op, values):
    calls = []
    original = blosc2.NDArray._execute_menudet_prefilter

    def record(self, *args, **kwargs):
        calls.append(kwargs.get("reduction_op"))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(blosc2.NDArray, "_execute_menudet_prefilter", record)
    a = blosc2.asarray(values, chunks=(8,), blocks=(2,))
    result = getattr(a, op)(dtype=np.bool_)
    assert isinstance(result, np.bool_)
    assert result == getattr(values, op)(dtype=np.bool_)
    if (
        not blosc2.IS_WASM
        and not np.__version__.startswith("1.")
        and getattr(blosc2.blosc2_ext, "menudet_prefilter_reductions_available", lambda: False)()
    ):
        assert calls == [5 if op == "sum" else 6]  # B2_REDUCE_ANY / B2_REDUCE_ALL


@pytest.mark.parametrize("op", ["sum", "prod"])
def test_typed_empty_boolean_accumulator_identity(op):
    values = np.array([], dtype=np.int64)
    result = getattr(blosc2.asarray(values), op)(dtype=np.bool_)
    assert isinstance(result, np.bool_)
    assert result == getattr(values, op)(dtype=np.bool_)


@pytest.mark.parametrize("prune_chunks", [False, True])
def test_typed_candidate_block_prefilter(monkeypatch, prune_chunks):
    if (
        blosc2.IS_WASM
        or np.__version__.startswith("1.")
        or not getattr(blosc2.blosc2_ext, "menudet_prefilter_reductions_available", lambda: False)()
    ):
        pytest.skip("Rebuilt native prefilter reduction bindings are required")
    values = np.arange(30, dtype=np.float64)
    a = blosc2.asarray(values, chunks=(12,), blocks=(4,))
    # Three chunks of three blocks (nine physical blocks, eight active in 30 items).
    bitmap = np.array([0, 1, 0, 0, 0, 0, 1, 0, 0], dtype=np.uint8)
    original = blosc2.NDArray._execute_menudet_prefilter
    reports = []

    def record(self, *args, **kwargs):
        result, report = original(self, *args, **kwargs)
        reports.append(report)
        return result, report

    monkeypatch.setattr(blosc2.NDArray, "_execute_menudet_prefilter", record)
    expr = a > -1
    result = expr.compute(_candidate_blocks=bitmap, _prune_chunks=prune_chunks)
    assert len(reports) == 1
    assert reports[0]["evaluated_blocks"] == 2
    expected = np.zeros(30, dtype=np.bool_)
    expected[4:8] = True
    expected[24:28] = True
    if prune_chunks:
        assert reports[0]["input_chunk_fetches"] == 2
        np.testing.assert_array_equal(
            np.frombuffer(result.schunk.decompress_chunk(0), dtype=np.bool_), expected[:12]
        )
        np.testing.assert_array_equal(
            np.frombuffer(result.schunk.decompress_chunk(2), dtype=np.bool_)[:6], expected[24:30]
        )
    else:
        np.testing.assert_array_equal(result[:], expected)
