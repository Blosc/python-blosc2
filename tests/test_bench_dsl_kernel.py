"""Small API/binding checks for the public Menudet benchmark (no timing thresholds)."""

import os
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import blosc2


@pytest.fixture(scope="module")
def benchmark():
    return runpy.run_path(str(Path(__file__).resolve().parents[1] / "bench/ndarray/dsl-kernel-bench.py"))


def test_benchmark_kernels_are_public_lazy_float32(benchmark):
    # Non-divisible multidimensional partitions exercise binding/materialization,
    # not an exhaustive native numerical/backend matrix.
    x = np.linspace(0, 1, 63, dtype=np.float32).reshape(7, 9)
    y = x[::-1].copy()
    a = blosc2.asarray(x, chunks=(4, 6), blocks=(2, 3))
    b = blosc2.asarray(y, chunks=a.chunks, blocks=a.blocks)
    with blosc2.jit_options(jit=False):
        for name in ("loop1", "loop2", "loop4", "loop4_heavy", "nested2"):
            author = benchmark[f"kernel_{name}"]
            assert isinstance(author, blosc2.MenudetFunction)
            lazy = author(a, b)
            assert isinstance(lazy, blosc2.LazyArray)
            assert lazy.dtype == x.dtype
            acc = np.zeros_like(x)
            indices = (
                ((i, j) for i in range(2) for j in range(2))
                if name == "nested2"
                else ((i, 0) for i in range(int(name[4])))
            )
            for i, j in indices:
                step = i + j
                tmp = (
                    np.where(x < y, y + step, x - step)
                    if step % 2 == 0
                    else np.where(x > y, x + step, y - step)
                )
                acc += tmp * (step + 1)
                if name == "loop4_heavy":
                    acc += tmp * tmp * 0.05
            result = lazy.compute(chunks=a.chunks, blocks=a.blocks)
            assert result.shape == x.shape
            assert result.dtype == x.dtype
            np.testing.assert_allclose(result[:], acc, rtol=1e-5, atol=2e-6)


@pytest.mark.parametrize("case", ["loop2", "transcend4"])
@pytest.mark.parametrize("warmups", [0, 1])
def test_benchmark_qualifies_all_routes(benchmark, case, warmups):
    a = blosc2.asarray(np.linspace(0, 1, 63, dtype=np.float32).reshape(7, 9), chunks=(4, 6), blocks=(2, 3))
    inputs = {"x": a}
    expr = benchmark["expr_transcendentals"]()
    if case == "loop2":
        inputs["y"] = a
        expr = benchmark["expr_for_steps"](2)
    with blosc2.jit_options(jit=False):
        row = benchmark["bench_case"](
            case, benchmark[f"kernel_{case}"], expr, inputs, samples=1, warmups=warmups
        )
    assert row["has_jit"] is False
    assert set(row["preparation_s"]) == {"lazyexpr", "menudet"}
    assert set(row["execution_reports"]) == {"lazyexpr", "menudet"}
    assert set(row["execution_modes"]) == {"lazyexpr", "menudet"}
    assert row["execution_modes"]["menudet"] == "interp"
    # This tiny input stays below the compressed native-prefilter minimum, so a
    # non-native LazyExpr route is legitimate here.
    assert row["execution_modes"]["lazyexpr"] in {"interp", "other"}
    assert "baseline" not in row
    assert all(len(samples) == 1 for samples in row["samples_s"].values())
    assert row["mode"] == ("preparation-cold" if warmups == 0 else "warm")
    assert "miniexpr_enabled" not in benchmark


def test_benchmark_speedup_and_headers(benchmark):
    row = {
        "case": "example",
        "lazyexpr": 0.006,
        "menudet": 0.002,
        "lazyexpr_gbps": 1.0,
        "menudet_gbps": 3.0,
        "execution_modes": {"lazyexpr": "interp", "menudet": "jit"},
    }
    headers, fmt, _ = benchmark["table_formatter"]()
    values = benchmark["format_row"](row)
    assert headers == [
        "Case",
        "LazyExpr ms",
        "LazyExpr GB/s",
        "LazyExpr mode",
        "Menudet ms",
        "Menudet GB/s",
        "Menudet mode",
        "Menudet speedup",
    ]
    assert values == ["example", "6.00", "1.00", "interp", "2.00", "3.00", "jit", "3.00x"]
    assert "3.00x" in fmt.format(*values)
    # Rows from callers that do not report a mode still render.
    assert benchmark["format_row"]({**row, "execution_modes": {}}) == [
        "example",
        "6.00",
        "1.00",
        "?",
        "2.00",
        "3.00",
        "?",
        "3.00x",
    ]


@pytest.mark.parametrize(
    ("report", "has_jit", "expected"),
    [
        ({"backend": "menudet-jit"}, None, "jit"),
        ({"backend": "menudet-interpreter"}, None, "interp"),
        ({"backend": "menudet-jit"}, False, "jit"),
        ({}, True, "jit"),
        ({}, False, "interp"),
        (None, None, "other"),
        ({"backend": "graph-bounded-readers"}, None, "graph-bounded-readers"),
        ({"backend": "menudet-interpreter"}, True, "interp"),
    ],
)
def test_execution_mode_reports_actual_engine(benchmark, report, has_jit, expected):
    assert benchmark["execution_mode"](report, has_jit) == expected


@pytest.mark.skipif(blosc2.IS_WASM, reason="Native compressed prefilter needs host threads")
def test_benchmark_records_engine_for_native_routes(benchmark):
    # Above the compressed-map minimum so both routes actually use native code.
    values = np.linspace(0, 1, 2_000_000, dtype=np.float32)
    a = blosc2.asarray(values, chunks=(250_000,), blocks=(25_000,))
    with blosc2.jit_options(jit=None, jit_backend=None):
        row = benchmark["bench_case"](
            "loop1",
            benchmark["kernel_loop1"],
            benchmark["expr_for_steps"](1),
            {"x": a, "y": a},
            samples=1,
            warmups=1,
        )
    lazy = row["execution_modes"]["lazyexpr"]
    assert lazy == benchmark["execution_mode"](row["execution_reports"]["lazyexpr"], None)
    assert lazy in {"jit", "interp"}
    assert row["execution_modes"]["menudet"] == ("jit" if row["has_jit"] else "interp")


def test_warm_samples_use_median_and_untimed_warmups(benchmark, monkeypatch):
    calls = []
    clock = iter([0, 0.003, 1, 1.001, 2, 2.002])
    monkeypatch.setattr(benchmark["time"], "perf_counter", lambda: next(clock))
    elapsed, _, timings = benchmark["time_it"](lambda: calls.append(1), samples=3, warmups=2)
    assert len(calls) == 5
    assert len(timings) == 3
    assert elapsed == pytest.approx(0.002)


def test_cold_samples_isolate_caches_and_include_factory(benchmark, monkeypatch):
    from blosc2.native_graph import configured_map_kernel

    author = SimpleNamespace(_kernels={"stale": object()})
    directories = []
    monkeypatch.setenv("ME_DSL_JIT_CACHE_DIR", "original-cache")
    monkeypatch.setenv("ME_DSL_JIT_POS_CACHE", "1")

    def forbidden():
        pytest.fail("Cold timing reused a prepared engine")

    def factory():
        assert not author._kernels
        assert configured_map_kernel.cache_info().currsize == 0
        directory = os.environ["ME_DSL_JIT_CACHE_DIR"]
        assert Path(directory).is_dir()
        assert blosc2.jit_config.execution_options()["cache_dir"] == directory
        assert os.environ["ME_DSL_JIT_POS_CACHE"] == "0"
        directories.append(directory)
        author._kernels["stale"] = object()
        return np.arange(3)

    elapsed, result, timings = benchmark["time_it"](
        forbidden, samples=3, warmups=0, cold_factory=factory, author=author
    )
    assert elapsed > 0
    assert len(timings) == 3
    assert len(set(directories)) == 3
    assert all(not Path(directory).exists() for directory in directories)
    assert os.environ["ME_DSL_JIT_CACHE_DIR"] == "original-cache"
    assert os.environ["ME_DSL_JIT_POS_CACHE"] == "1"
    np.testing.assert_array_equal(result, np.arange(3))


def test_cold_preparation_restores_policy_on_failure(benchmark, monkeypatch):
    monkeypatch.delenv("ME_DSL_JIT_CACHE_DIR", raising=False)
    monkeypatch.delenv("ME_DSL_JIT_POS_CACHE", raising=False)
    with blosc2.jit_options(cache_dir="outer-cache"):
        previous = blosc2.jit_config.execution_options()
        with (
            pytest.raises(RuntimeError, match="compile failed"),
            benchmark["cold_preparation"](SimpleNamespace(_kernels={})),
        ):
            raise RuntimeError("compile failed")
        assert blosc2.jit_config.execution_options() == previous
    assert "ME_DSL_JIT_CACHE_DIR" not in os.environ
    assert "ME_DSL_JIT_POS_CACHE" not in os.environ


@pytest.mark.parametrize(
    ("options", "warmups", "backend", "jit"),
    [
        ([], 3, None, False),
        (["--jit", "cc", "--warmups", "0"], 0, "cc", True),
        (["--jit", "tcc"], 3, "tcc", True),
        (["--no-jit"], 3, None, False),
    ],
)
def test_benchmark_cli_modes(benchmark, monkeypatch, tmp_path, capsys, options, warmups, backend, jit):
    import json

    output = tmp_path / "results.json"
    calls = []

    def bench_case(name, kernel, expr, inputs, *, samples, warmups):
        calls.append((samples, warmups, blosc2.jit_config.execution_options()))
        return {"case": name, "lazyexpr": 0.006, "menudet": 0.002, "lazyexpr_gbps": 1.0, "menudet_gbps": 3.0}

    main = benchmark["main"]
    monkeypatch.setitem(main.__globals__, "bench_case", bench_case)
    monkeypatch.setattr(blosc2, "set_nthreads", lambda threads: None)
    monkeypatch.setattr(
        sys,
        "argv",
        ["dsl-kernel-bench.py", "--size", "3", "--samples", "2", "--output", str(output), *options],
    )
    with blosc2.jit_options(jit=False, jit_backend=None):
        main()
    assert len(calls) == 10
    assert all(sample_count == 2 and warmup_count == warmups for sample_count, warmup_count, _ in calls)
    assert all(policy["jit"] is jit and policy["jit_backend"] == backend for _, _, policy in calls)
    metadata = json.loads(output.read_text())["metadata"]
    assert metadata["samples"] == 2
    assert metadata["warmups"] == warmups
    assert metadata["statistic"] == "median"
    assert metadata["mode"] == ("preparation-cold" if warmups == 0 else "warm")
    assert "compiler_requested" not in metadata
    text = capsys.readouterr().out
    assert ("Preparation-cold medians" if warmups == 0 else "Warm medians") in text


@pytest.mark.parametrize(
    ("options", "backend"),
    [
        ([], None),
        (["--math-backend", "system"], "system"),
        (["--math-backend", "accelerate"], "accelerate"),
    ],
)
def test_benchmark_cli_math_backend(benchmark, monkeypatch, tmp_path, capsys, options, backend):
    import json

    output = tmp_path / "results.json"
    seen = []

    def bench_case(name, kernel, expr, inputs, *, samples, warmups):
        seen.append(blosc2.jit_config.execution_options()["math_backend"])
        return {
            "case": name,
            "lazyexpr": 0.006,
            "menudet": 0.002,
            "lazyexpr_gbps": 1.0,
            "menudet_gbps": 3.0,
        }

    main = benchmark["main"]
    monkeypatch.setitem(main.__globals__, "bench_case", bench_case)
    # Availability depends on the build; the preflight itself is covered separately.
    monkeypatch.setitem(main.__globals__, "require_math_backend", lambda parser, value: None)
    monkeypatch.setattr(blosc2, "set_nthreads", lambda threads: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dsl-kernel-bench.py",
            "--size",
            "3",
            "--samples",
            "1",
            "--warmups",
            "1",
            "--output",
            str(output),
            *options,
        ],
    )
    main()
    assert set(seen) == {backend}
    metadata = json.loads(output.read_text())["metadata"]
    assert metadata["math_backend_requested"] == backend
    assert f"Miniexpr math backend request: {backend or 'runtime default'}" in capsys.readouterr().out


def test_math_backend_preflight_reports_native_error(benchmark, monkeypatch, capsys):
    class Unavailable:
        def specialize(self, values):
            raise blosc2.MenudetError(
                "requested math backend 'sleef' is unavailable on this build", status="invalid_source"
            )

    main = benchmark["main"]
    monkeypatch.setitem(main.__globals__, "kernel_math_probe", Unavailable())
    monkeypatch.setattr(blosc2, "set_nthreads", lambda threads: None)
    monkeypatch.setattr(sys, "argv", ["dsl-kernel-bench.py", "--size", "3", "--math-backend", "sleef"])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    captured = capsys.readouterr()
    assert "requested math backend 'sleef' is unavailable" in captured.err
    # The preflight runs before any table is printed.
    assert "LazyExpr vs Menudet" not in captured.out


def test_math_backend_preflight_inherits_policy(benchmark):
    import argparse

    benchmark["require_math_backend"](argparse.ArgumentParser(), None)


def test_math_probe_does_not_share_measured_signatures(benchmark):
    # float64 keeps the probe out of every measured float32 kernel cache.
    assert benchmark["kernel_math_probe"]._signature.parameters["x"].annotation is np.float64


@pytest.mark.parametrize(
    "options",
    [
        ["--warmups", "-1"],
        ["--samples", "0"],
        ["--jit", "cc", "--no-jit"],
        ["--compiler", "cc"],
        ["--repeat", "1"],
        ["--math-backend", "mkl"],
    ],
)
def test_benchmark_cli_rejects_invalid_options(benchmark, monkeypatch, options):
    monkeypatch.setattr(sys, "argv", ["dsl-kernel-bench.py", *options])
    with pytest.raises(SystemExit) as error:
        benchmark["main"]()
    assert error.value.code == 2
