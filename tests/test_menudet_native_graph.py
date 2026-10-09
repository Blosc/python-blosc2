"""Native-required graph execution, plan reuse and NumExpr-unavailable deployment."""

import os
import subprocess
import sys

import numpy as np
import pytest

import blosc2
from blosc2.expression_graph import ExpressionGraph
from blosc2.native_graph import compile_plan


@pytest.fixture(autouse=True)
def array_runtime():
    from blosc2.dsl_kernel import DSLKernel

    try:
        source = DSLKernel.from_source("def k(x):\n    return x\n")
        plan = blosc2.PortableKernel.from_json(source.export({"x": "float64"}, "float64", version="1.1"))
        plan.evaluate_array({"x": np.ones(1)})
    except blosc2.PortableArtifactError as error:
        if os.environ.get("MENUDET_REQUIRE_ARRAY_RUNTIME"):
            pytest.fail(str(error))
        pytest.skip("Installed dependency does not implement native logical arrays")


@pytest.mark.parametrize("dtype", ["int8", "int64", "uint64", "float32", "float64"])
def test_required_graph_no_fallback_and_plan_reuse(monkeypatch, dtype):
    x = np.arange(60).reshape(10, 6).astype(dtype)
    y = np.ones((1, 6), dtype=dtype)
    with blosc2.expression_evaluation("safe"):
        expr = blosc2.lazyexpr("x * 2 + y", {"x": x, "y": y})

    def forbidden(*args, **kwargs):
        pytest.fail("Python/NumExpr numerical fallback occurred")

    import importlib

    module = importlib.import_module("blosc2.lazyexpr")
    monkeypatch.setattr(module, "ne_evaluate", forbidden)
    monkeypatch.setattr(ExpressionGraph, "evaluate", forbidden)
    result = expr.compute(_require_native=True)
    np.testing.assert_array_equal(result[:], x * 2 + y)
    previous = compile_plan.cache_info().hits
    x[:] = 3
    np.testing.assert_array_equal(expr.compute(_require_native=True)[:], x * 2 + y)
    assert compile_plan.cache_info().hits > previous
    assert expr._native_execution_report["backend"] == "portable-interpreter"


def test_partial_reads_and_reductions():
    x = np.arange(60.0).reshape(10, 6)
    y = np.ones((1, 6))
    with blosc2.expression_evaluation("safe"):
        expr = blosc2.lazyexpr("x + y", {"x": x, "y": y})
    np.testing.assert_array_equal(
        expr.compute((slice(2, 8, 2), slice(None, None, -1)), _require_native=True)[:], (x + y)[2:8:2, ::-1]
    )
    for op in ("sum", "prod", "min", "max", "any", "all"):
        np.testing.assert_array_equal(
            getattr(expr, op)(axis=0, _require_native=True), getattr(np, op)(x + y, axis=0)
        )


def test_root_reduction_and_export(tmp_path):
    x = blosc2.asarray(
        np.arange(24.0).reshape(4, 6), urlpath=tmp_path / "input.b2nd", chunks=(2, 6), blocks=(1, 3)
    )
    with blosc2.expression_evaluation("safe"):
        expr = blosc2.lazyexpr("sum(x * 2, axis=-1, keepdims=True)", {"x": x})
        elementwise = blosc2.lazyexpr("x * 2", {"x": x})
    np.testing.assert_array_equal(
        expr.compute(_require_native=True), np.sum(x[:] * 2, axis=-1, keepdims=True)
    )
    with pytest.raises(ValueError, match="Root"):
        expr.native_kernel()
    plan = elementwise.native_kernel()
    inputs = {name: value for name, value in elementwise.operands.items() if name in plan.input_dtypes}
    portable = plan.lazy(inputs, partitions=(2, 3))
    portable.save(tmp_path / "plan.b2nd")
    loaded = blosc2.open(tmp_path / "plan.b2nd")
    np.testing.assert_array_equal(loaded[:], x[:] * 2)


def test_unsupported_rejects_before_destination_write(tmp_path):
    with blosc2.expression_evaluation("safe"):
        expr = blosc2.lazyexpr("cumsum(x)", {"x": np.arange(8.0)})
    target = tmp_path / "output.b2nd"
    with pytest.raises((ValueError, blosc2.PortableArtifactError)):
        expr.compute(_require_native=True, urlpath=target)
    assert not target.exists()


def test_numexpr_unavailable_subprocess(tmp_path):
    script = r"""
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname == 'numexpr' or fullname.startswith('numexpr.'):
            raise ModuleNotFoundError('NumExpr deliberately unavailable', name='numexpr')
sys.meta_path.insert(0, Block())
import blosc2, numpy as np
assert blosc2.numexpr is None
with blosc2.expression_evaluation('safe'):
    x = blosc2.asarray(np.arange(24.).reshape(4,6))
    e = blosc2.lazyexpr('sqrt(x) + 2', {'x':x})
    result = e.compute(_require_native=True)
    np.testing.assert_allclose(result[:], np.sqrt(x[:]) + 2)
    np.testing.assert_allclose(e.sum(axis=0, _require_native=True), np.sum(np.sqrt(x[:]) + 2,axis=0))
    plan = e.native_kernel()
    imported = blosc2.PortableKernel.from_json(plan.to_json())
    bindings = {name:value[()] for name,value in e.operands.items() if name in imported.input_dtypes}
    np.testing.assert_allclose(imported.evaluate_array(bindings), np.sqrt(x[:]) + 2)
    from pathlib import Path
    root = Path(sys.argv[1])
    stored = blosc2.asarray(x[:], urlpath=root/'input.b2nd')
    e = blosc2.lazyexpr('sqrt(x) + 2', {'x':stored})
    plan = e.native_kernel()
    bindings = {name:value for name,value in e.operands.items() if name in plan.input_dtypes}
    plan.lazy(bindings, partitions=(2,3)).save(root/'portable.b2nd')
    restored = blosc2.open(root/'portable.b2nd', deserialize='safe')
    np.testing.assert_allclose(restored[:], np.sqrt(stored[:])+2)
assert not any(n == 'numexpr' or n.startswith('numexpr.') for n in sys.modules)
"""
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
