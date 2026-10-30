"""Scalar and elementwise correctness of the engine comparison benchmark."""

import os
import random
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest

import blosc2


@pytest.mark.parametrize("reduc", [None, "sum"])
def test_cold_selected_engine_results(engine_benchmark, reduc):
    inputs, values = engine_benchmark["build_operands"](257, "numpy", "int64")
    expected = values.sum() if reduc else values
    for name in engine_benchmark["ENGINES"]:
        with engine_benchmark["cold_preparation"]():
            result = engine_benchmark["make_engines"](inputs, "numpy", reduc, selected=name)[name]()
        if isinstance(result, blosc2.NDArray):
            result = result[:]
        np.testing.assert_array_equal(result, expected)


def test_cold_samples_rebuild_after_qualification(engine_benchmark):
    created = []
    previous = os.environ.get("ME_DSL_JIT_CACHE_DIR")

    def factory(name):
        assert name == "probe"
        created.append(os.environ["ME_DSL_JIT_CACHE_DIR"])
        assert os.environ["ME_DSL_JIT_POS_CACHE"] == "0"
        return lambda: np.arange(3)

    times = engine_benchmark["benchmark"](
        {"probe": lambda: np.arange(3)}, np.arange(3), 3, 0, random.Random(42), cold_factory=factory
    )
    assert times["probe"] > 0
    assert len(set(created)) == 3
    assert os.environ.get("ME_DSL_JIT_CACHE_DIR") == previous


@pytest.fixture(scope="module")
def engine_benchmark():
    pytest.importorskip("numexpr")
    return runpy.run_path(str(Path(__file__).resolve().parents[1] / "bench" / "menudet-engines.py"))


@pytest.mark.parametrize("dtype", ["float32", "float64", "complex64", "complex128"])
@pytest.mark.parametrize("epsilons", [4, 16])
def test_array_mixed_tolerance(engine_benchmark, dtype, epsilons):
    expected = np.array([0.0, 1.0, -1.0], dtype=dtype)
    result = expected.copy()
    result.real[:] += epsilons * np.finfo(expected.real.dtype).eps
    failures = {}
    times = engine_benchmark["benchmark"](
        {"probe": lambda: result}, expected, 1, 0, random.Random(42), failures=failures
    )
    assert ("probe" in times) == (epsilons <= 4)
    assert bool(failures) == (epsilons > 4)
    if failures:
        assert "max absolute error=" in failures["probe"]
        assert "index=(0,): actual=" in failures["probe"]
        assert "expected=" in failures["probe"]


def test_array_mixed_tolerance_diagnostics(engine_benchmark):
    expected = np.zeros((2, 65536))
    result = expected.copy()
    result.flat[[0, 1, 2, 65536]] = [1, 2, 3, 4]
    with pytest.raises(AssertionError) as error:
        engine_benchmark["assert_float_arrays_close"](result, expected)
    message = str(error.value)
    assert "4 mismatches; max absolute error=4;" in message
    assert message.count("index=") == 3
    assert "index=(0, 2):" in message


def test_array_mixed_tolerance_nonfinite(engine_benchmark):
    check = engine_benchmark["assert_float_arrays_close"]
    expected = np.array([np.nan, np.inf, -np.inf])
    check(expected.copy(), expected)
    with pytest.raises(AssertionError, match="max absolute error=inf"):
        check(np.array([0.0, np.inf, -np.inf]), expected)


@pytest.mark.parametrize("dtype", ["int64", "bool"])
def test_array_exact_tolerance(engine_benchmark, dtype):
    expected = np.array([0, 1], dtype=dtype)
    result = np.array([1, 1], dtype=dtype)
    failures = {}
    times = engine_benchmark["benchmark"](
        {"probe": lambda: result}, expected, 1, 0, random.Random(42), failures=failures
    )
    assert not times
    assert failures == {"probe": "different numerical results"}


@pytest.mark.parametrize(
    ("options", "expected"),
    [
        ([], [100_000, 1_000_000, 8_000_000, 100_000_000]),
        (["--maxitems", "1000000"], [100_000, 1_000_000]),
        (["--maxitems", "1m"], [100_000, 1_000_000]),
        (["--maxsize", "8M"], [100_000, 1_000_000, 8_000_000]),
        (["--items", "3", "1", "2", "--maxitems", "2"], [1, 2]),
        (["--maxitems", "1"], []),
    ],
)
def test_maxitems_cli(engine_benchmark, monkeypatch, capsys, options, expected):
    main = engine_benchmark["main"]
    namespace = main.__globals__
    sizes = []

    def build_operands(size, *args, **kwargs):
        sizes.append(size)
        return {}, np.zeros(1)

    monkeypatch.setattr(sys, "argv", ["menudet-engines.py", "--operands", "numpy", *options])
    monkeypatch.setitem(namespace, "build_operands", build_operands)
    monkeypatch.setitem(namespace, "make_engines", lambda *args, **kwargs: {})
    monkeypatch.setitem(namespace, "benchmark", lambda *args, **kwargs: {})
    monkeypatch.setattr(blosc2, "set_nthreads", lambda threads: None)
    monkeypatch.setattr(namespace["ne"], "set_num_threads", lambda threads: None)
    main()
    assert sizes == expected
    if not expected:
        assert "nothing to benchmark" in capsys.readouterr().out


@pytest.mark.parametrize("value", ["0", "-1", "invalid", "8MB", "1.5M"])
def test_maxitems_invalid(engine_benchmark, monkeypatch, value):
    monkeypatch.setattr(sys, "argv", ["menudet-engines.py", "--maxitems", value])
    with pytest.raises(SystemExit) as error:
        engine_benchmark["main"]()
    assert error.value.code == 2


@pytest.mark.parametrize(
    ("value", "expected"), [("123", 123), ("100K", 100_000), ("8m", 8_000_000), ("1G", 1_000_000_000)]
)
def test_parse_item_count(engine_benchmark, value, expected):
    assert engine_benchmark["parse_item_count"](value) == expected


@pytest.mark.parametrize("reduc", [None, "sum", "mean", "min", "max", "any", "all"])
@pytest.mark.parametrize(("size", "kind"), [(1, "numpy"), (257, "blosc2")])
def test_engine_results(engine_benchmark, size, kind, reduc):
    inputs, values = engine_benchmark["build_operands"](size, "numpy")
    expected = getattr(np, reduc)(values) if reduc else values
    if kind == "blosc2":
        # Multiple chunks plus a partial tail for the nontrivial size.
        inputs = {name: blosc2.asarray(value, chunks=(128,), blocks=(32,)) for name, value in inputs.items()}
    engines = engine_benchmark["make_engines"](inputs, kind, reduc)
    assert tuple(engines) == engine_benchmark["ENGINES"]
    for name, operation in engines.items():
        result = operation()
        if reduc:
            assert np.ndim(result) == 0, name
            np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12, err_msg=name)
        else:
            if name == "LazyExpr" or (kind == "blosc2" and name == "Menudet"):
                assert isinstance(result, blosc2.NDArray), name
                assert result.cparams.clevel == 5, name
                if kind == "blosc2":
                    assert result.chunks == inputs["x"].chunks, name
                    assert result.blocks == inputs["x"].blocks, name
                result = result[:]
            else:
                assert isinstance(result, np.ndarray), name
            np.testing.assert_array_equal(result, expected, err_msg=name)


@pytest.mark.parametrize("kind", ["numpy", "blosc2"])
def test_evaluate_column_calls_public_api_every_time(engine_benchmark, monkeypatch, kind):
    inputs, expected = engine_benchmark["build_operands"](257, kind)
    calls = []

    def evaluate(expression, *, local_dict):
        calls.append((expression, local_dict))
        return expected.copy()

    monkeypatch.setattr(blosc2, "evaluate", evaluate)
    engines = engine_benchmark["make_engines"](inputs, kind)
    first = engines["evaluate"]()
    second = engines["evaluate"]()
    assert first is not second
    assert calls == [(engine_benchmark["EXPRESSION"], inputs)] * 2


def test_allocation_control_outputs(engine_benchmark):
    inputs, expected = engine_benchmark["build_operands"](257, "numpy", "int64")
    engines = engine_benchmark["make_allocation_engines"](inputs)
    reused = engines["numexpr reused"]()
    np.testing.assert_array_equal(reused, expected)
    assert engines["evaluate reused"]() is reused
    for name in ("numexpr fresh", "evaluate fresh"):
        first = engines[name]()
        second = engines[name]()
        assert first is not second
        np.testing.assert_array_equal(first, expected)
    np.testing.assert_array_equal(reused, expected)


def test_allocation_control_cli(engine_benchmark, monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "menudet-engines.py",
            "--items",
            "17",
            "--samples",
            "1",
            "--warmups",
            "1",
            "--operands",
            "numpy",
            "--allocation-control",
        ],
    )
    engine_benchmark["main"]()
    text = capsys.readouterr().out
    assert "main table above remains fresh-output" in text
    assert "numexpr reused" in text
    assert "evaluate reused" in text


@pytest.mark.parametrize("backend", [None, "tcc", "cc"])
def test_jit_backend_selection(engine_benchmark, monkeypatch, capsys, backend):
    main = engine_benchmark["main"]
    namespace = main.__globals__
    monkeypatch.setenv("ME_DSL_JIT_COMPILER", "tcc")
    monkeypatch.setenv("BLOSC_ME_JIT", "0")
    argv = ["menudet-engines.py", "--items", "1", "--samples", "1", "--warmups", "1"]
    argv += ["--dtype", "uint64", "--codec", "LZ4", "--clevel", "3"]
    if backend:
        argv += ["--jit", backend]
    monkeypatch.setattr(sys, "argv", argv)
    observed = []

    def make_engines(inputs, kind, reduc, cparams=None, expression=None):
        assert inputs["x"].dtype == np.dtype("uint64")
        assert cparams.codec == blosc2.Codec.LZ4
        assert cparams.clevel == 3
        observed.append((kind, os.environ["ME_DSL_JIT_COMPILER"]))
        return {name: lambda: np.zeros(1) for name in namespace["ENGINES"]}

    monkeypatch.setitem(namespace, "make_engines", make_engines)
    monkeypatch.setattr(blosc2, "set_nthreads", lambda threads: None)
    monkeypatch.setattr(namespace["ne"], "set_num_threads", lambda threads: None)
    main()
    expected = backend or "tcc"
    assert observed == [("numpy", expected), ("blosc2", expected)]
    assert f"Miniexpr JIT backend request: {expected}" in capsys.readouterr().out


@pytest.mark.parametrize("dtype", ["int16", "uint64", "float32"])
@pytest.mark.parametrize("kind", ["numpy", "blosc2"])
def test_arange_operands_and_compression(engine_benchmark, dtype, kind):
    cparams = blosc2.CParams(codec=blosc2.Codec.LZ4, clevel=3)
    # Cross a narrow-integer wrap boundary without relying on Python scalar casts.
    size = 40003 if dtype == "int16" else 257
    inputs, expected = engine_benchmark["build_operands"](size, kind, dtype, cparams)
    x, y = np.arange(size, dtype=dtype), np.arange(1, size + 1, dtype=dtype)
    for name, values in (("x", x), ("y", y)):
        operand = inputs[name]
        assert operand.dtype == np.dtype(dtype)
        if kind == "blosc2":
            assert operand.cparams.codec == blosc2.Codec.LZ4
            assert operand.cparams.clevel == 3
            operand = operand[:]
        np.testing.assert_array_equal(operand, values)
    np.testing.assert_array_equal(expected, x + (y - 1))
    for name in ("LazyExpr", "Menudet"):
        if kind == "numpy" and name == "Menudet":
            continue
        result = engine_benchmark["make_engines"](inputs, kind, cparams=cparams)[name]()
        assert result.cparams.codec == blosc2.Codec.LZ4
        assert result.cparams.clevel == 3
        np.testing.assert_array_equal(result[:], expected)


def test_benchmark_reports_nonparity(engine_benchmark):
    import random

    engines = {
        "NumPy": lambda: np.array([1], dtype="int16"),
        "numexpr": lambda: np.array([2], dtype="int32"),
    }
    failures = {}
    times = engine_benchmark["benchmark"](
        engines, np.array([1], dtype="int16"), 1, 0, random.Random(42), failures=failures
    )
    assert set(times) == {"NumPy"}
    assert failures == {"numexpr": "different numerical results"}


@pytest.mark.parametrize("expression", ["z * 2", "x * y + z", "sqrt(y + z)"])
@pytest.mark.parametrize("kind", ["numpy", "blosc2"])
def test_custom_expression(engine_benchmark, expression, kind):
    inputs, expected = engine_benchmark["build_operands"](257, kind, expression=expression)
    assert tuple(inputs) == engine_benchmark["operand_names"](expression)
    for reduc in (None, "sum"):
        engines = engine_benchmark["make_engines"](inputs, kind, reduc, expression=expression)
        reference = expected.sum() if reduc else expected
        for name, operation in engines.items():
            result = operation()
            if isinstance(result, blosc2.NDArray):
                result = result[:]
            np.testing.assert_allclose(result, reference, rtol=1e-12, atol=1e-12, err_msg=name)


def test_custom_expression_allocation_and_cold(engine_benchmark):
    expression = "y > z"
    inputs, expected = engine_benchmark["build_operands"](17, "numpy", expression=expression)
    for operation in engine_benchmark["make_allocation_engines"](inputs, expression=expression).values():
        np.testing.assert_array_equal(operation(), expected)
    with engine_benchmark["cold_preparation"]():
        operation = engine_benchmark["make_engines"](
            inputs, "numpy", selected="Menudet", expression=expression
        )["Menudet"]
        np.testing.assert_array_equal(operation(), expected)


def test_custom_expression_cli(engine_benchmark, monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        ["menudet-engines.py", "--expr", "z * 2", "--items", "17", "--samples", "1", "--warmups", "1"],
    )
    engine_benchmark["main"]()
    text = capsys.readouterr().out
    assert "Expression: z * 2" in text
    assert "N/A" not in text


@pytest.mark.parametrize("expression", ["x +", "x + w", "42"])
def test_invalid_expression_cli(engine_benchmark, monkeypatch, expression):
    monkeypatch.setattr(sys, "argv", ["menudet-engines.py", "--expr", expression])
    with pytest.raises(SystemExit, match="2"):
        engine_benchmark["main"]()
