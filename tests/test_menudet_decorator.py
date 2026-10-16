from __future__ import annotations

import numpy as np
import pytest

import blosc2


@blosc2.menudet
def inferred(x, y):
    return x + (y - 1)


@blosc2.menudet
def annotated(x: np.float64, y: np.float64) -> np.float64:
    return x + (y - 1)


@pytest.mark.parametrize("dtype", ["int8", "int32", "uint64", "float32", "float64"])
@pytest.mark.parametrize("compressed", [False, True])
def test_inferred_arrays(dtype, compressed, native_graph_runtime):
    x = np.arange(12).reshape(3, 4).astype(dtype)
    y = np.ones((1, 4), dtype=dtype)
    values = {"x": blosc2.asarray(x) if compressed else x, "y": y}
    lazy = inferred(**values)
    assert lazy.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(lazy[:], x + (y - 1))
    np.testing.assert_array_equal(lazy.compute()[:], x + (y - 1))


def test_annotations_and_no_cast(native_graph_runtime):
    x, y = np.arange(4.0), np.ones(4)
    np.testing.assert_array_equal(annotated(x, y)[:], x + y - 1)
    with pytest.raises(TypeError, match="expected dtype float64"):
        annotated(x.astype("float32"), y)


def test_specialization_cache_and_mutation(native_graph_runtime):
    x, y = np.arange(4.0), np.ones(4)
    kernel = inferred.specialize(x, y)
    assert kernel is inferred.specialize(y=y, x=x.reshape(2, 2))
    assert kernel is not inferred.specialize(x.astype("float32"), y.astype("float32"))
    lazy = inferred(x, y)
    x[:] = 10
    np.testing.assert_array_equal(lazy[:], np.full(4, 10.0))
    assert isinstance(kernel, blosc2.PortableKernel)


def test_interpretation_and_python_never_called(native_graph_runtime, monkeypatch):
    @blosc2.menudet(jit=False)
    def k(x: np.float64) -> np.float64:
        return x * 2

    def forbidden(*args, **kwargs):
        pytest.fail("Original Python function executed")

    monkeypatch.setattr(blosc2.DSLKernel, "__call__", forbidden)
    monkeypatch.setattr(k._author, "func", forbidden)
    x = np.arange(4.0)
    np.testing.assert_array_equal(k(x)[:], x * 2)
    assert not k.specialize(x).has_jit


def test_default_jit(native_graph_runtime):
    @blosc2.menudet
    def k(x):
        return x + 1

    assert k._jit is True
    x = np.arange(4.0)
    np.testing.assert_array_equal(k(x)[:], x + 1)


def test_libm_kernel_roundtrip(native_graph_runtime):
    # Numerical/operator matrices live in miniexpr. Check decorator binding,
    # lazy reads and native artifact reconstruction with the resolved callbacks.
    @blosc2.menudet
    def trig(x):
        return np.sin(x) + np.cos(x) + 1

    x = np.linspace(-np.pi, np.pi, 257)
    expected = np.sin(x) + np.cos(x) + 1
    np.testing.assert_allclose(trig(x)[:], expected, rtol=1e-14, atol=1e-14)
    kernel = blosc2.PortableKernel.from_json(trig.specialize(x).to_json())
    np.testing.assert_allclose(kernel.evaluate({"x": x}), expected, rtol=1e-14, atol=1e-14)


def test_binding_defaults_and_scalars(native_graph_runtime):
    @blosc2.menudet(jit=False)
    def k(x, y: np.float64 = 2.0):
        return x + y

    x = np.arange(4.0)
    np.testing.assert_array_equal(k(x)[:], x + 2)
    np.testing.assert_array_equal(k(y=3.0, x=x)[:], x + 3)
    with pytest.raises(TypeError, match="missing"):
        k()
    with pytest.raises(TypeError, match="unannotated inputs"):
        inferred(1, 2)


def test_capture_snapshot(native_graph_runtime):
    scale = 2

    @blosc2.menudet(jit=False)
    def k(x):
        return x * scale

    scale = 7
    for dtype in ("float32", "float64"):
        x = np.arange(4).astype(dtype)
        np.testing.assert_array_equal(k(x)[:], x * 2)


def test_inferred_statement_return(native_graph_runtime):
    @blosc2.menudet(jit=False)
    def k(x):
        y = x * 2
        return y + 1

    x = np.arange(4, dtype="int32")
    assert k(x).dtype == np.dtype("int32")
    np.testing.assert_array_equal(k(x)[:], x * 2 + 1)


def test_block_reduction_contract(native_graph_runtime):
    @blosc2.menudet(jit=False, cardinality="block_scalar", partitions=(2,))
    def k(x: np.int32) -> np.int64:
        return block_sum(x)  # noqa: F821 -- native Menudet builtin

    np.testing.assert_array_equal(k(np.arange(4, dtype="int32"))[:], [1, 5])


def test_required_jit_unavailable(native_graph_runtime, monkeypatch):
    @blosc2.menudet(require_jit=True)
    def k(x: np.float64) -> np.float64:
        return x + 1

    monkeypatch.setattr(blosc2.PortableKernel, "has_jit", property(lambda self: False))
    with pytest.raises(RuntimeError, match="Required Menudet JIT"):
        k(np.arange(4.0))


def test_invalid_decorator_options():
    with pytest.raises(ValueError, match="conflicts"):
        blosc2.menudet(jit=False, require_jit=True)(inferred.__wrapped__)
    with pytest.raises(TypeError, match="must be bool"):
        blosc2.menudet(jit="tcc")(inferred.__wrapped__)
    with pytest.raises(TypeError, match="ordinary Python function"):
        blosc2.menudet(np.add)


def test_unsupported_annotations():
    with pytest.raises(TypeError, match="dtype annotation"):

        @blosc2.menudet
        def bad(x: list) -> np.float64:
            return x


def test_native_execution_error_not_replayed(native_graph_runtime, monkeypatch):
    lazy = inferred(np.arange(4.0), np.ones(4))

    def failed(*args, **kwargs):
        raise blosc2.PortableArtifactError("Invocation failed", status="evaluation_error")

    monkeypatch.setattr(blosc2.PortableKernel, "evaluate_block", failed)
    with pytest.raises(blosc2.PortableArtifactError, match="Invocation failed"):
        lazy[:]


def test_unavailable_return_inference(native_graph_runtime, monkeypatch):
    @blosc2.menudet(jit=False)
    def k(x):
        return x + 1

    monkeypatch.setattr(blosc2.PortableKernel, "inferred_dtype", property(lambda self: None))
    with pytest.raises(TypeError, match="return dtype annotation"):
        k(np.arange(4.0))


def test_full_read_fast_path_preserves_block_reduction(native_graph_runtime, monkeypatch):
    @blosc2.menudet(jit=False)
    def k(x: np.int32) -> np.int64:
        return x - block_sum(x)  # noqa: F821 -- native Menudet builtin

    x = np.arange(4, dtype="int32")
    lazy = k(x)

    def forbidden(*args, **kwargs):
        pytest.fail("Full single-group read built a selection-index array")

    monkeypatch.setattr(np, "arange", forbidden)
    np.testing.assert_array_equal(lazy[:], [-6, -5, -4, -3])


def test_full_read_fast_path_preserves_coordinates(native_graph_runtime):
    @blosc2.menudet(jit=False, ndim=1)
    def k(x: np.float64) -> np.float64:
        return x + _i0 + _n0  # noqa: F821 -- native coordinate context

    np.testing.assert_array_equal(k(np.ones(4))[:], [5, 6, 7, 8])


def test_decorated_artifact_is_standalone(native_graph_runtime):
    x, y = np.arange(4.0), np.ones(4)
    artifact = annotated.specialize(x, y).to_json()
    kernel = blosc2.PortableKernel.from_json(artifact, jit=False)
    np.testing.assert_array_equal(kernel.evaluate_array({"x": x, "y": y}), x + (y - 1))


def test_none_is_not_a_dtype_annotation():
    with pytest.raises(TypeError, match="dtype annotation: None"):

        @blosc2.menudet
        def bad(x) -> None:
            return x
