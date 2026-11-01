"""Small API/binding checks for the public Menudet benchmark (no timing thresholds)."""

import runpy
from pathlib import Path

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
def test_benchmark_qualifies_all_routes(benchmark, case):
    a = blosc2.asarray(np.linspace(0, 1, 63, dtype=np.float32).reshape(7, 9), chunks=(4, 6), blocks=(2, 3))
    inputs = {"x": a}
    expr = benchmark["expr_transcendentals"]()
    if case == "loop2":
        inputs["y"] = a
        expr = benchmark["expr_for_steps"](2)
    old = benchmark["lazyexpr_mod"].try_miniexpr
    with blosc2.jit_options(jit=False):
        row = benchmark["bench_case"](case, benchmark[f"kernel_{case}"], expr, inputs, repeat=1)
    assert row["has_jit"] is False
    assert set(row["preparation_s"]) == {"baseline", "lazyexpr", "menudet"}
    assert all(len(samples) == 1 for samples in row["samples_s"].values())
    assert benchmark["lazyexpr_mod"].try_miniexpr is old
