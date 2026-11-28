"""M8 cutover verification: default NativeGraph dispatch, fallback, and pilot retirement."""

import json

import numpy as np
import pytest

import blosc2
from blosc2.graph_plan import NativeGraph


def _exec_available():
    return bool(getattr(blosc2.blosc2_ext, "_b2nd_exec_available", lambda: False)())


def _adapter_available():
    return bool(getattr(blosc2.blosc2_ext, "menudet_legacy_adapter_available", lambda: False)())


def test_m8_math_environment_precedence(monkeypatch):
    if not _exec_available():
        pytest.skip("Requires C-Blosc2 typed executor")
    from blosc2.typed_graph import typed_graph_policy

    monkeypatch.setenv("ME_SIMD_MATH_BACKEND", "system")
    monkeypatch.setenv("ME_DSL_JIT_COMPILER", "cc")
    with blosc2.compute_options(math_backend="sleef", jit_backend="tcc"):
        policy, _, compiler, math = typed_graph_policy(False, False)
        assert (compiler, math) == (2, 2)
        assert policy["math_backend"] == "system"
        plan = NativeGraph.from_expression("sin(x)", {"x": "float64"}, jit=False)
    assert plan.info()["math_backend"] == "system"
    imported = NativeGraph.from_json(plan.to_json(), jit=False)
    assert imported.info()["math_backend"] == "system"
    np.testing.assert_allclose(plan.evaluate({"x": np.arange(5.0)}), np.sin(np.arange(5.0)))

    # These are environment values, not accepted compute_options values.
    for backend, code in (("auto", 1), ("scalar", 5)):
        monkeypatch.setenv("ME_SIMD_MATH_BACKEND", backend)
        with blosc2.compute_options(fp_accuracy=blosc2.FPAccuracy.MEDIUM):
            assert typed_graph_policy(False, False)[3] == code
        with pytest.raises(ValueError, match="math_backend"):
            with blosc2.compute_options(math_backend=backend):
                pass


def test_m8_native_auto_policy(monkeypatch):
    if not _exec_available():
        pytest.skip("Requires C-Blosc2 typed executor")
    from blosc2.typed_graph import typed_graph_policy

    for name in ("BLOSC_ME_JIT", "ME_DSL_JIT", "ME_DSL_JIT_COMPILER"):
        monkeypatch.delenv(name, raising=False)
    with blosc2.compute_options(jit=None, math_backend="system"):
        assert typed_graph_policy(None, False)[1] == 3
        assert typed_graph_policy(False, False)[1] == 0
        assert typed_graph_policy(True, False)[1] == 1
        assert typed_graph_policy(None, True)[1] == 2
        plan = NativeGraph.from_expression("x + 1", {"x": "float64"})
        assert not plan.info()["has_jit"]  # Native expression-only AUTO heuristic.


def test_m8_direct_handle_auto_jit(monkeypatch):
    if not _exec_available() or _adapter_available():
        pytest.skip("Requires artifact-off typed handle delegation")
    from blosc2.blosc2_ext import NativeGraphHandle

    monkeypatch.delenv("BLOSC_ME_JIT", raising=False)
    monkeypatch.setenv("ME_DSL_JIT", "1")
    with blosc2.compute_options(jit=None, jit_backend="cc", math_backend="system"):
        graph = NativeGraph.from_expression("x + 1", {"x": "float64"}, jit=False).to_json().encode()
        handles = [
            NativeGraphHandle(graph, 3),
            NativeGraphHandle.from_expression(b"x + 1", {"x": "float64"}, 3),
        ]
        for handle in handles:
            assert handle.info()["has_jit"]
            schedule = handle.specialize({"x": (np.dtype("float64"), (5,))})
            values, report = schedule.execute({"x": np.arange(5.0)})
            np.testing.assert_array_equal(values, np.arange(5.0) + 1)
            assert report["jit_calls"] > 0
        with pytest.raises(ValueError, match="JIT mode"):
            NativeGraphHandle(graph, 4)
        with pytest.raises(ValueError, match="JIT mode"):
            NativeGraphHandle.from_expression(b"x", {"x": "float64"}, -1)


def test_m8_default_executor_selects_c_blosc2(monkeypatch):
    if not _exec_available():
        pytest.skip("Requires C-Blosc2 typed executor")

    class _ForbidAdapter:
        def __new__(cls, *args, **kwargs):
            raise AssertionError("adapter forbidden for admitted M8 path")

        @classmethod
        def from_expression(cls, *args, **kwargs):
            raise AssertionError("adapter forbidden for admitted M8 path")

    monkeypatch.setattr(blosc2.graph_plan, "NativeGraphHandle", _ForbidAdapter)

    x = np.arange(1, 13, dtype=np.float64).reshape(3, 4)
    y = np.arange(4, dtype=np.float64)

    expr_plan = NativeGraph.from_expression("x * 2.5 + y", {"x": "float64", "y": "float64"})
    assert expr_plan.info()["executor"] == "c-blosc2"
    np.testing.assert_allclose(expr_plan.evaluate({"x": x, "y": y}), x * 2.5 + y)
    assert json.loads(expr_plan.to_json())["format"] == "menudet-graph-1"
    assert json.loads(expr_plan.map_json())["schema_version"] == "0.1"

    roundtrip = NativeGraph.from_json(expr_plan.to_json())
    assert roundtrip.info()["executor"] == "c-blosc2"
    np.testing.assert_allclose(roundtrip.evaluate({"x": x, "y": y}), x * 2.5 + y)

    staged_plan = NativeGraph(NativeGraph.from_expression("x - sum(x, axis=0)", {"x": "float64"}).to_json())
    assert staged_plan.info()["executor"] == "c-blosc2"
    f_out = np.empty(x.shape, dtype=np.float64, order="F")
    result, report = staged_plan.evaluate({"x": x}, out=f_out, return_report=True)
    assert result is f_out
    np.testing.assert_allclose(result, x - np.sum(x, axis=0))
    assert report["stages"] == 2

    red_plan = NativeGraph.from_expression("sum(x, dtype=int8)", {"x": "int64"})
    assert red_plan.info()["executor"] == "c-blosc2"
    arr_i64 = np.array([120, 120, 120], dtype=np.int64)
    red_out = red_plan.evaluate({"x": arr_i64})
    assert red_out.dtype == np.dtype("int8")
    assert red_out == np.sum(arr_i64, dtype="int8")

    # Compressed source execution and lazyexpr compute without touching adapter.
    bx = blosc2.asarray(np.arange(1, 101, dtype=np.float64), chunks=(32,), blocks=(16,))
    stream_plan = NativeGraph.from_expression("sum(x * 2 + 1)", {"x": "float64"})
    assert stream_plan.info()["executor"] == "c-blosc2"
    np.testing.assert_allclose(
        stream_plan.evaluate_sources({"x": bx}, tile_items=17),
        np.sum(bx[:] * 2 + 1),
    )

    lazy_res = blosc2.evaluate(
        "x**3 + 1", local_dict={"x": np.arange(8, dtype=np.int64)}, optimization="aggressive"
    )
    np.testing.assert_array_equal(lazy_res, np.arange(8, dtype=np.int64) ** 3 + 1)


def test_m8_fallback_and_explicit_executor_modes():
    if _adapter_available():
        x = np.arange(8, dtype=np.float64)
        explicit_adapter = NativeGraph.from_expression("x + 1", {"x": "float64"}, _executor="adapter")
        assert explicit_adapter.info()["executor"] == "adapter"
        np.testing.assert_array_equal(explicit_adapter.evaluate({"x": x}), x + 1)

    if not _exec_available():
        if _adapter_available():
            plan = NativeGraph.from_expression("x + 1", {"x": "float64"})
            assert plan.info()["executor"] == "adapter"
            with pytest.raises(NotImplementedError):
                NativeGraph.from_expression("x + 1", {"x": "float64"}, _executor="c-blosc2")
        return

    # Accuracy preferences now remain on the typed route, including artifact-off.
    with blosc2.compute_options(fp_accuracy=blosc2.FPAccuracy.HIGH):
        auto_plan = NativeGraph.from_expression("sin(x)", {"x": "float64"})
        assert auto_plan.info()["executor"] == "c-blosc2"
        assert (
            NativeGraph.from_expression("sin(x)", {"x": "float64"}, _executor="c-blosc2").info()["executor"]
            == "c-blosc2"
        )

    with pytest.raises(ValueError, match="Unknown graph executor"):
        NativeGraph.from_expression("x + 1", {"x": "float64"}, _executor="invalid")


@pytest.mark.parametrize("fp", ["strict", "contract", "fast", "relaxed"])
@pytest.mark.parametrize("accuracy", [blosc2.FPAccuracy.HIGH, blosc2.FPAccuracy.MEDIUM])
def test_m8_fp_policy_remains_typed(monkeypatch, fp, accuracy):
    if not _exec_available():
        pytest.skip("Requires C-Blosc2 typed executor")
    monkeypatch.setenv("ME_DSL_FP_MODE", fp)
    x = np.linspace(0.1, 1, 24).reshape(6, 4)
    with blosc2.compute_options(fp_accuracy=accuracy, jit_backend="cc"):
        plan = NativeGraph.from_expression("sin(x) + cos(x)", {"x": "float64"}, jit=False)
        assert plan.info()["executor"] == "c-blosc2"
        assert plan.info()["fp_mode"] == ("fast" if fp == "relaxed" else fp)
        if plan.info()["math_backend"] == "sleef":
            assert plan.info()["sleef_accuracy"] == ("u10" if accuracy == blosc2.FPAccuracy.HIGH else "u35")
        expected = np.sin(x) + np.cos(x)
        for operand in (x, blosc2.asarray(x, chunks=(3, 4), blocks=(1, 4))):
            np.testing.assert_allclose(plan.evaluate_sources({"x": operand}), expected, rtol=1e-6)
        imported = NativeGraph.from_json(plan.to_json(), jit=False)
        assert imported.info()["fp_mode"] == plan.info()["fp_mode"]
        np.testing.assert_allclose(imported.evaluate({"x": x}), expected, rtol=1e-6)
        staged = NativeGraph.from_expression("x - sum(sin(x), axis=0)", {"x": "float64"}, jit=False)
        assert staged.info()["executor"] == "c-blosc2"
        np.testing.assert_allclose(staged.evaluate({"x": x}), x - np.sin(x).sum(axis=0), rtol=1e-6)
        exported = blosc2.MenudetKernel.from_json(plan.map_json(), jit=False)
        assert exported._handle.info()["fp_mode"] == plan.info()["fp_mode"]
        np.testing.assert_allclose(exported.evaluate({"x": x}), expected, rtol=1e-6)


def test_m8_accuracy_preference_fallback_and_cache(monkeypatch):
    if not _exec_available():
        pytest.skip("Requires C-Blosc2 typed executor")
    import blosc2.typed_graph as typed

    monkeypatch.setattr(typed, "_typed_sleef_available", lambda: False)
    with blosc2.compute_options(fp_accuracy=blosc2.FPAccuracy.HIGH):
        high = NativeGraph.from_expression("sin(x)", {"x": "float64"}, jit=False)
    assert high.info()["executor"] == "c-blosc2"
    assert high.info()["math_backend"] in {"auto", "system"}
    monkeypatch.setenv("ME_DSL_FP_MODE", "contract")
    with blosc2.compute_options(jit_backend="cc"):
        contract = NativeGraph.from_expression("sin(x)", {"x": "float64"}, jit=False)
    assert contract.info()["fp_mode"] == "contract"
    assert high.info()["fp_mode"] == "strict"
    assert high._handle._plan is not contract._handle._plan


@pytest.mark.parametrize("fp", ["contract", "fast"])
def test_m8_fp_policy_required_system_jit(monkeypatch, fp):
    if not _exec_available():
        pytest.skip("Requires C-Blosc2 typed executor")
    monkeypatch.setenv("ME_DSL_FP_MODE", fp)
    with blosc2.compute_options(jit_backend="cc", math_backend="system", fp_accuracy=blosc2.FPAccuracy.HIGH):
        plan = NativeGraph.from_expression("sqrt(x) + 1", {"x": "float64"}, require_jit=True)
    assert plan.info()["executor"] == "c-blosc2"
    assert plan.info()["fp_mode"] == fp
    assert plan.info()["math_backend"] == "system"  # Explicit backend beats HIGH preference.
    assert plan.info()["sleef_accuracy"] == ""
    x = np.arange(1.0, 10.0)
    values, report = plan.evaluate({"x": x}, return_report=True)
    np.testing.assert_allclose(values, np.sqrt(x) + 1)
    assert report["jit_calls"] > 0


@pytest.mark.parametrize("accuracy", [blosc2.FPAccuracy.HIGH, blosc2.FPAccuracy.MEDIUM])
def test_m8_explicit_sleef_accuracy(monkeypatch, accuracy):
    if not _exec_available():
        pytest.skip("Requires C-Blosc2 typed executor")
    import blosc2.typed_graph as typed

    if not typed._typed_sleef_available():
        pytest.skip("Explicit SLEEF requires a SLEEF-enabled core")
    monkeypatch.delenv("ME_DSL_FP_MODE", raising=False)
    with blosc2.compute_options(math_backend="sleef", fp_accuracy=accuracy):
        plan = NativeGraph.from_expression("sin(x)", {"x": "float64"}, jit=False)
    assert plan.info()["math_backend"] == "sleef"
    assert plan.info()["sleef_accuracy"] == ("u10" if accuracy == blosc2.FPAccuracy.HIGH else "u35")
    x = np.linspace(0.1, 1.0, 17)
    np.testing.assert_allclose(plan.evaluate({"x": x}), np.sin(x), rtol=1e-6)


def test_m8_non_strict_tcc_fails_without_adapter_replay(monkeypatch):
    if not _exec_available():
        pytest.skip("Requires C-Blosc2 typed executor")

    class ForbidAdapter:
        @classmethod
        def from_expression(cls, *args, **kwargs):
            raise AssertionError("Invalid compiler/FP policy must not replay the adapter")

    monkeypatch.setattr(blosc2.graph_plan, "NativeGraphHandle", ForbidAdapter)
    monkeypatch.setenv("ME_DSL_FP_MODE", "fast")
    with blosc2.compute_options(jit_backend="tcc", math_backend="system"):
        with pytest.raises(blosc2.MenudetError, match="supports strict floating point only"):
            NativeGraph.from_expression("sin(x)", {"x": "float64"}, jit=False)


def test_m8_auto_preserves_admission_error_without_adapter(monkeypatch):
    if not _exec_available():
        pytest.skip("Requires C-Blosc2 typed executor")
    from blosc2.typed_graph import TypedGraphHandle

    graph = NativeGraph.from_expression("x + 1", {"x": "float64"}, jit=False).to_json()
    failure = NotImplementedError("typed admission rejected")

    def reject(*args, **kwargs):
        raise failure

    def forbid(*args, **kwargs):
        raise AssertionError("missing adapter must not be entered")

    monkeypatch.setattr(blosc2.graph_plan, "_adapter_available", lambda: False)
    monkeypatch.setattr(blosc2.graph_plan, "NativeGraphHandle", forbid)
    monkeypatch.setattr(TypedGraphHandle, "from_json", reject)
    monkeypatch.setattr(TypedGraphHandle, "__init__", reject)
    for prepare in (
        lambda: NativeGraph.from_json(graph, jit=False),
        lambda: NativeGraph.from_expression("x + 1", {"x": "float64"}, jit=False),
    ):
        with pytest.raises(NotImplementedError) as exc:
            prepare()
        assert exc.value is failure


def test_m8_core_array_accumulator_and_mask():
    if not _exec_available():
        pytest.skip("Requires C-Blosc2 typed executor")

    @blosc2.menudet(jit=False)
    def root(x):
        return np.sqrt(x)

    x = np.array([[-1, 4, 9], [-1, 16, 25]], dtype=np.float64)
    mask = np.array([[False, True, True]])
    kernel = root.specialize(x)
    for tile in (1, 2, 16):
        values, report = kernel.evaluate_array(
            {"x": x},
            reduction="sum",
            axis=1,
            dtype="float32",
            initial=1,
            where=mask,
            tile_items=tile,
            return_report=True,
        )
        assert values.dtype == np.dtype("float32")
        np.testing.assert_array_equal(values, np.array([6, 10], dtype="float32"))
        assert report["fp_flags"] == 0
    values = kernel.evaluate_array(
        {"x": x[:, 1:]},
        reduction="sum",
        dtype="float32",
        tile_items=2,
    )
    assert values.dtype == np.dtype("float32")
    assert values == 14


def test_m8_retires_single_case_pilots():
    for attr in (
        "_ResidentMinI64Plan",
        "_ResidentMapMinI64Plan",
        "_b2nd_kernel_pilot_available",
        "_b2nd_min_i64_available",
        "_b2nd_map_min_i64_available",
    ):
        assert not hasattr(blosc2.blosc2_ext, attr)
    assert not hasattr(blosc2.NDArray, "_evaluate_b2nd_kernel_pilot")

    if not _exec_available():
        pytest.skip("Requires opt-in C-Blosc2 typed executor")

    data = np.arange(1024, dtype=np.int64)[::-1].copy()
    assert blosc2.lazyexpr("min(x)", {"x": data}).compute() == 0

    bdata = blosc2.asarray(np.arange(1024, dtype=np.float64), chunks=(256,), blocks=(128,))

    @blosc2.menudet(cardinality="block_scalar", partitions=(256,))
    def my_sum(x):
        return block_sum(x)  # noqa: F821

    lazy = my_sum(bdata)
    np.testing.assert_allclose(
        lazy[:], [np.sum(np.arange(i, i + 256, dtype=np.float64)) for i in range(0, 1024, 256)]
    )


def test_m8_core_menudet_v1_parity_and_execution():
    if not _exec_available():
        pytest.skip("Requires opt-in C-Blosc2 core executor")

    from blosc2 import blosc2_ext

    # 1. Both from_recipe and from_json prepare core me_kernel handles.
    @blosc2.menudet(jit=False)
    def scaled_shift(x):
        return x * np.float64(1.5) + 2

    src = blosc2.asarray(np.arange(16, dtype=np.float64), chunks=(8,), blocks=(4,))
    recipe_kernel = scaled_shift.specialize(src)
    assert recipe_kernel._handle._has_core_kernel() is True

    manifest_json = recipe_kernel.to_json()
    json_kernel = blosc2.MenudetKernel.from_json(manifest_json, jit=False)
    assert json_kernel._handle._has_core_kernel() is True
    json_info = json_kernel._handle.info()
    recipe_info = recipe_kernel._handle.info()
    for key in ("source", "entry_point", "inputs", "output_dtype", "jit", "execution_kind", "fp_mode"):
        assert json_info[key] == recipe_info[key]
    assert json_info["math_backend"] in {"auto", "system"}
    assert recipe_info["math_backend"] in {"auto", "system"}
    json_desc = json_kernel._handle.descriptor_info()
    recipe_desc = recipe_kernel._handle.descriptor_info()
    for key in (
        "schema_version",
        "inputs",
        "output_dtype",
        "inferred_dtype",
        "cardinality",
        "ndim",
        "native_partitionable",
    ):
        assert json_desc[key] == recipe_desc[key]

    arr_x = np.arange(8, dtype=np.float64)
    np.testing.assert_array_equal(json_kernel.evaluate({"x": arr_x}), arr_x * 1.5 + 2.0)
    np.testing.assert_array_equal(json_kernel.evaluate_block({"x": arr_x}), arr_x * 1.5 + 2.0)
    np.testing.assert_array_equal(
        recipe_kernel.lazy({"x": src})[:], np.arange(16, dtype=np.float64) * 1.5 + 2.0
    )
    np.testing.assert_array_equal(
        json_kernel.lazy({"x": src})[:], np.arange(16, dtype=np.float64) * 1.5 + 2.0
    )

    # 2. Inferred output & ND context v1 manifest lowering.
    nd_manifest = json.dumps(
        {
            "schema_version": "0.1",
            "language": {"name": "menudet", "version": "0.1"},
            "requires": ["numeric", "nd-context"],
            "source": "def coord_scale(x):\n    return x + _i0 * _n1 + _ndim\n",
            "entry_point": "coord_scale",
            "inputs": [{"name": "x", "dtype": "float64"}],
            "constants": [],
            "output": {"dtype": "auto", "contract": "elementwise"},
            "semantics": {"numeric": "numpy-2.5", "casting": "unsafe"},
            "context": {"ndim": 2},
        }
    ).encode("utf-8")
    nd_handle = blosc2_ext.MenudetArtifactHandle(nd_manifest, 0, True)
    assert nd_handle._has_core_kernel() is True
    nd_info = nd_handle.descriptor_info()
    assert nd_info["ndim"] == 2
    assert nd_info["inferred_dtype"] == np.dtype("float64")
    block_out = nd_handle.evaluate_block(
        {"x": np.zeros((2, 3), dtype=np.float64)},
        (2, 3),
        logical_shape=(4, 5),
        block_origin=(1, 2),
    )
    expected_nd = np.array(
        [[1 * 5 + 2, 1 * 5 + 2, 1 * 5 + 2], [2 * 5 + 2, 2 * 5 + 2, 2 * 5 + 2]], dtype=np.float64
    )
    np.testing.assert_array_equal(block_out, expected_nd)

    # 3. Line diagnostic adjustment ignores injected semantics header pragmas.
    bad_manifest = {
        "schema_version": "0.1",
        "language": {"name": "menudet", "version": "0.1"},
        "requires": ["numeric", "flexible-math"],
        "source": "def bad(x):\n    return (\n",
        "entry_point": "bad",
        "inputs": [{"name": "x", "dtype": "float64"}],
        "constants": [],
        "output": {"dtype": "float64", "contract": "elementwise"},
        "semantics": {"numeric": "numpy-2.5", "casting": "unsafe", "fp": "contract", "math": "system"},
        "context": {"ndim": 0},
    }
    with pytest.raises(blosc2.MenudetError) as err_info:
        blosc2.MenudetKernel.from_json(json.dumps(bad_manifest).encode("utf-8"), jit=False)
    assert err_info.value.status == "invalid_source"
    assert err_info.value.line <= 2

    # 4. Resident prefilter works through core handles when eligible.
    res_np = np.empty(16, dtype=np.float64)
    recipe_kernel._handle.evaluate_prefilter({"x": src}, numpy_out=res_np)
    np.testing.assert_array_equal(res_np, np.arange(16, dtype=np.float64) * 1.5 + 2.0)
