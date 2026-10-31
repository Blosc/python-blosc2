"""NumExpr installation must not affect runtime import or expression dispatch."""

import ast
import importlib.metadata
import json
import runpy
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

import blosc2


def test_package_has_no_numexpr_import_or_requirement():
    package = Path(blosc2.__file__).parent
    imports = []
    for source in package.glob("*.py"):
        tree = ast.parse(source.read_text(), filename=str(source))
        for node in ast.walk(tree):
            names = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else []
            )
            if any(name.split(".")[0] == "numexpr" for name in names):
                imports.append(f"{source.name}:{node.lineno}")
    assert not imports
    requirements = importlib.metadata.requires("blosc2") or []
    assert not any(item.lower().startswith("numexpr") for item in requirements)


@pytest.mark.skipif(blosc2.IS_WASM, reason="Subprocess isolation is unavailable on WASM")
@pytest.mark.parametrize("installed", [False, True])
@pytest.mark.parametrize("split_package", [False, True])
def test_runtime_never_imports_numexpr(installed, split_package):
    source = textwrap.dedent("""
        import sys, json, importlib.abc, importlib.util, types, os
        sys.path[:] = json.loads(sys.argv[1])
        if sys.argv[2] == "installed":
            try:
                import numexpr
            except ImportError:
                class ForbiddenNumExpr(types.ModuleType):
                    def __getattr__(self, name):
                        if name.startswith("__"):
                            raise AttributeError(name)
                        raise AssertionError("Unexpected NumExpr access: " + name)
                sys.modules["numexpr"] = ForbiddenNumExpr("numexpr")
            else:
                def forbidden(*args, **kwargs):
                    raise AssertionError("Unexpected NumExpr backend/thread call")
                for name in ("evaluate", "re_evaluate", "validate", "NumExpr", "set_num_threads"):
                    setattr(numexpr, name, forbidden)
        os.environ["NUMEXPR_MAX_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"
        class BlockNumExpr(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "numexpr" or fullname.startswith("numexpr."):
                    raise AssertionError("Unexpected NumExpr runtime import")
        sys.meta_path.insert(0, BlockNumExpr())
        import numpy as np
        package = json.loads(sys.argv[3])
        spec = importlib.util.spec_from_file_location(
            "blosc2", package["origin"], submodule_search_locations=package["paths"])
        blosc2 = importlib.util.module_from_spec(spec)
        sys.modules["blosc2"] = blosc2
        spec.loader.exec_module(blosc2)
        assert blosc2.numexpr is None
        x = np.arange(24.).reshape(4,6)
        for order in ("C", "F"):
            result=blosc2.evaluate("x * 2 + 1", local_dict={"x":x}, order=order)
            np.testing.assert_array_equal(result,x*2+1)
        stored=blosc2.asarray(x)
        np.testing.assert_array_equal(blosc2.evaluate("x + 1",local_dict={"x":stored}),x+1)
        out=np.empty(x.shape,dtype="complex128")
        assert blosc2.evaluate("complex(x,x)",local_dict={"x":x},out=out) is out
        np.testing.assert_array_equal(out.real,x)
        np.testing.assert_array_equal(out.imag,x)
        mixed=blosc2.evaluate("a + b",local_dict={"a":stored,"b":x})
        np.testing.assert_array_equal(mixed,2*x)
        words=np.array(["red", "blue", "red"])
        np.testing.assert_array_equal(
            blosc2.evaluate("contains(words, 'red')",local_dict={"words":words}),
            [True,False,True])
        dates=np.array(["2026-01-01", "NaT"],dtype="datetime64[D]")
        np.testing.assert_array_equal(
            blosc2.evaluate("dates == dates",local_dict={"dates":dates}),[True,False])
        with blosc2.jit_options(jit=False):
            from blosc2.graph_plan import NativeGraph
            graph=NativeGraph.from_expression("x * 2 + 1",{"x":"float64"},jit=False)
            result,report=graph.evaluate({"x":x},out=np.empty_like(x),return_report=True)
            np.testing.assert_array_equal(result,x*2+1)
            assert not graph.has_jit
            assert report["interpreter_stages"] > 0 and report["jit_stages"] == 0
        evaluator=blosc2.prepare_evaluate("x+1")
        np.testing.assert_array_equal(evaluator({"x":x}),x+1)
        blosc2.evaluate("sum(x)",local_dict={"x":x})
        assert blosc2.re_evaluate({"x":x}) == x.sum()
        assert blosc2.numexpr is None
        if sys.argv[2] != "installed":
            assert "numexpr" not in sys.modules
    """)
    package = {"origin": blosc2.__file__, "paths": list(blosc2.__path__)}
    if split_package:
        root = Path(__file__).resolve().parents[1] / "src/blosc2"
        package = {"origin": str(root / "__init__.py"), "paths": [str(root), *blosc2.__path__]}
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            source,
            json.dumps(sys.path),
            "installed" if installed else "absent",
            json.dumps(package),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_interpreter_bounded_stored_output(native_graph_runtime):
    from blosc2.native_graph import _compute_native_stored_output

    values = np.arange(17 * 19, dtype="float64").reshape(17, 19)
    inputs = {
        "x": blosc2.asarray(values, chunks=(7, 8), blocks=(3, 4)),
        "y": np.arange(19, dtype="float64")[None, :],
    }
    expression = blosc2.lazyexpr("x + y", inputs)
    out = blosc2.empty(values.shape, dtype="float64", chunks=(9, 10), blocks=(3, 5))
    result = _compute_native_stored_output(expression, out, {"out": out}, jit=False)
    assert result is out
    np.testing.assert_array_equal(out[:], values + inputs["y"])
    report = expression._native_execution_report
    assert report["backend"] == "menudet-interpreter"
    assert report["route"] == "bounded-stored-map"
    assert report["maximum_input_items"] <= 262144


@pytest.mark.parametrize("legacy", [False, True])
def test_chunk_evaluator_preserves_caller_scope(legacy):
    from blosc2.lazyexpr import evaluate_chunk, ne_evaluate

    x = np.arange(7, dtype="float64")
    evaluate = ne_evaluate if legacy else evaluate_chunk
    np.testing.assert_array_equal(evaluate("x + 1"), x + 1)


@pytest.mark.parametrize("owner", ["blosc2.audit_probe", "external_reference"])
def test_absence_qualification_detects_swallowed_runtime_imports(monkeypatch, owner):
    script = Path(__file__).resolve().parents[1] / "scripts/qualify_without_numexpr.py"
    blocker = runpy.run_path(str(script))["BlockNumExpr"]()
    monkeypatch.delitem(sys.modules, "numexpr", raising=False)
    monkeypatch.setattr(sys, "meta_path", [blocker, *sys.meta_path])
    namespace = {"__name__": owner}
    exec("def request():\n    import numexpr\n", namespace)
    with pytest.raises(ModuleNotFoundError, match="excluded"):
        namespace["request"]()
    assert blocker.runtime_imports == ([owner] if owner.startswith("blosc2.") else [])


def test_external_backend_benchmark_without_numexpr(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    blocker = runpy.run_path(str(root / "scripts/qualify_without_numexpr.py"))["BlockNumExpr"]()
    monkeypatch.delitem(sys.modules, "numexpr", raising=False)
    monkeypatch.setattr(sys, "meta_path", [blocker, *sys.meta_path])
    worker = runpy.run_path(str(root / "tools/menudet_backend_bench.py"))["worker"]
    result = worker("numpy", "float64", 16, 1)
    assert result["warm_median_ns"] > 0
    assert result["numexpr_threads"] is None
    assert worker("numexpr", "float64", 16, 1)["status"] == "unavailable"
    for tool, args in (
        ("tools/menudet_jit_bench.py", ("numexpr", "affine", "float64", 16, "flat")),
        ("tools/menudet_graph_bench.py", (16, "float64", "numexpr", "C", False)),
    ):
        unavailable = runpy.run_path(str(root / tool))["worker"](*args)
        assert unavailable["status"] == "unavailable"
    assert not blocker.runtime_imports


def test_graph_benchmark_without_resource(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    worker = runpy.run_path(str(root / "tools/menudet_graph_bench.py"))["worker"]
    monkeypatch.setitem(worker.__globals__, "resource", None)
    result = worker(16, "float64", "numpy", "C", False)
    assert result["warm_median_ms"] > 0
    assert result["peak_rss_bytes"] is None
