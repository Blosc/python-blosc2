"""Bindings and dispatch contracts for source-authored Menudet execution."""

import importlib
import sys

import numpy as np
import pytest

import blosc2

pytestmark = pytest.mark.skipif(
    not blosc2.blosc2_ext.portable_descriptor_available(),
    reason="Portable runtime required",
)


@pytest.fixture
def forbid_legacy(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Supported DSL execution entered the legacy evaluator")

    monkeypatch.setattr(blosc2.NDArray, "_set_pref_expr", forbidden)
    monkeypatch.setattr(importlib.import_module("blosc2.lazyexpr"), "chunked_eval", forbidden)


def test_dsl_udf_compute_slice_and_out(forbid_legacy, tmp_path):
    source = blosc2.DSLKernel.from_source("def k(x, bias):\n    return x + bias\n")
    values = np.arange(48, dtype="float64").reshape(6, 8)
    x = blosc2.asarray(values, chunks=(6, 8), blocks=(2, 4))
    lazy = blosc2.lazyudf(source, (x, 3.0), dtype="float64", jit=False)
    np.testing.assert_array_equal(lazy[1:5:2, 2:7], (values + 3)[1:5:2, 2:7])
    output = np.empty_like(values)
    assert lazy.compute(out=output) is output
    np.testing.assert_array_equal(output, values + 3)
    result = lazy.compute(urlpath=tmp_path / "output.b2nd", chunks=(3, 4), blocks=(1, 2))
    assert result.chunks == (3, 4)
    np.testing.assert_array_equal(result[:], values + 3)
    assert lazy._native_execution_report["route"] == "menudet-lazyudf"


def test_dsl_input_free_and_jit_wrapper(forbid_legacy):
    kernel = blosc2.DSLKernel.from_source("def k(start):\n    return start + _flat_idx\n")
    lazy = blosc2.lazyudf(kernel, (7,), dtype="int64", shape=(5, 7), jit=False)
    np.testing.assert_array_equal(lazy[:], 7 + np.arange(35).reshape(5, 7))

    @blosc2.jit(jit=False)
    def bounded(x, limit=3):
        if x > limit:
            return limit
        return x

    np.testing.assert_array_equal(bounded(np.arange(10.0)), np.minimum(np.arange(10.0), 3))


def test_small_constructors_use_menudet_without_jit(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Supported small ramps must not create a legacy LazyUDF")

    monkeypatch.setattr(blosc2, "lazyudf", forbidden)
    original = blosc2.PortableKernel.from_json
    observed = []

    def load(artifact, *, jit=None):
        observed.append(jit)
        return original(artifact, jit=jit)

    monkeypatch.setattr(blosc2.PortableKernel, "from_json", load)
    np.testing.assert_array_equal(blosc2.arange(31)[:], np.arange(31))
    assert observed[-1] is False
    np.testing.assert_allclose(blosc2.linspace(0, 1, 31)[:], np.linspace(0, 1, 31))
    assert observed[-1] is False
    blosc2.arange(31, jit=True)
    assert observed[-1] is (not blosc2.IS_WASM)


@pytest.mark.parametrize("scalar_return", [False, True])
def test_padded_groups_and_scalar_broadcast(forbid_legacy, scalar_return):
    values = np.arange(11 * 13, dtype="float64").reshape(11, 13)
    x = blosc2.asarray(values, chunks=(6, 8), blocks=(4, 3))
    body = "block_sum(x)" if scalar_return else "x + block_sum(x) + _i0"
    kernel = blosc2.DSLKernel.from_source(f"def k(x):\n    return {body}\n")
    lazy = blosc2.lazyudf(kernel, (x,), dtype="float64", jit=False, in_place=True)
    expected = np.empty_like(values)
    if scalar_return:
        expected = np.array(
            [
                [values[r : min(r + 4, 11), c : min(c + 3, 13)].sum() for c in (0, 3, 6, 9, 12)]
                for r in (0, 4, 8)
            ]
        )
        np.testing.assert_array_equal(lazy[:], expected)
        return
    for row in range(0, 11, 6):
        for col in range(0, 13, 8):
            for r in range(row, min(row + 6, 11), 4):
                for c in range(col, min(col + 8, 13), 3):
                    item = (slice(r, min(r + 4, row + 6, 11)), slice(c, min(c + 3, col + 8, 13)))
                    block = values[item]
                    expected[item] = block + block.sum() + np.arange(r, item[0].stop)[:, None]
    np.testing.assert_array_equal(lazy[:], expected)
    np.testing.assert_array_equal(lazy[1:10:2, 2:12:3], expected[1:10:2, 2:12:3])
    np.testing.assert_array_equal(lazy.compute(chunks=(5, 5), blocks=(2, 2))[:], expected)


def test_portable_loader_options(forbid_legacy, tmp_path, capfd):
    if blosc2.IS_WASM or sys.platform == "win32":
        pytest.skip("No host compiler backend on WASM/Windows")
    if not blosc2.blosc2_ext.portable_artifact_options_available():
        pytest.skip("Requires the native artifact options API")
    kernel = blosc2.DSLKernel.from_source("def k(x):\n    return x + 1\n")
    lazy = blosc2.lazyudf(kernel, (np.arange(31.0),), dtype="float64")
    with blosc2.jit_options(jit_backend="cc", compiler="/no/compiler", cache_dir=tmp_path, trace=True):
        np.testing.assert_array_equal(lazy[:], np.arange(31.0) + 1)
    assert "compiler unavailable" in capfd.readouterr().err
    assert lazy._native_execution_report["route"] == "menudet-lazyudf"
    assert not lazy._native_execution_report["jit"]
    assert not blosc2.get_jit_options()["trace"]


@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
def test_complex_kernels_use_menudet(forbid_legacy, dtype):
    z = np.array([1 + 2j, 3 - 4j, -2 + 0.5j, 0 + 0j], dtype=dtype)
    kernel = blosc2.DSLKernel.from_source(
        "def k(z):\n"
        "    v = z\n"
        "    for i in range(2):\n"
        "        v = v * z\n"
        "    if imag(v) < 0:\n"
        "        return v + 1\n"
        "    return v\n"
    )
    lazy = blosc2.lazyudf(kernel, (z,), dtype=dtype, jit=True)
    expected = z * z * z
    expected = np.where(expected.imag < 0, expected + 1, expected)
    np.testing.assert_allclose(lazy[:], expected, rtol=1e-6)
    assert lazy._native_execution_report["route"] == "menudet-lazyudf"
    # Complex is interpreter-only; requesting JIT must not compile partial code.
    assert lazy._native_execution_report["jit"] is False
    result = lazy.compute(chunks=(2,), blocks=(1,))
    np.testing.assert_allclose(result[:], expected, rtol=1e-6)


def test_complex_real_results_constants_and_reductions(forbid_legacy):
    z = np.array([1 + 2j, 3 - 4j, -2 + 0.5j, 0 + 0j], dtype=np.complex128)
    real = blosc2.lazyudf(
        blosc2.DSLKernel.from_source("def k(z):\n    return abs(z) + real(z) - imag(z)\n"),
        (z,),
        dtype=np.float64,
        jit=False,
    )
    np.testing.assert_allclose(real[:], np.abs(z) + z.real - z.imag)
    shifted = blosc2.lazyudf(
        blosc2.DSLKernel.from_source("def k(z, c):\n    return z + c\n"),
        (z, 2 + 3j),
        dtype=z.dtype,
        jit=False,
    )
    np.testing.assert_allclose(shifted[:], z + (2 + 3j))
    summed = blosc2.lazyudf(
        blosc2.DSLKernel.from_source("def k(z):\n    return block_sum(z)\n"),
        (blosc2.asarray(z, chunks=(4,), blocks=(4,)),),
        dtype=z.dtype,
        jit=False,
    )
    np.testing.assert_allclose(summed[()], [z.sum()])


@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
@pytest.mark.parametrize("compressed", [False, True])
@pytest.mark.parametrize("operation", ["real", "conj"])
def test_complex_callback_identity(forbid_legacy, dtype, compressed, operation):
    # Optimized WASM links must keep real and conj distinct despite their
    # identical real-valued callbacks in the classic builtin registry.
    z = np.array([1 + 2j, 3 - 4j, -2 + 0.5j, 0j], dtype=dtype)
    values = blosc2.asarray(z, chunks=(2,), blocks=(1,)) if compressed else z
    output_dtype = z.real.dtype if operation == "real" else z.dtype
    source = blosc2.DSLKernel.from_source(f"def k(z):\n    return {operation}(z)\n")
    lazy = blosc2.lazyudf(source, (values,), dtype=output_dtype, jit=False)
    expected = z.real if operation == "real" else z.conj()
    np.testing.assert_array_equal(lazy[:], expected)
    np.testing.assert_array_equal(lazy[1::2], expected[1::2])
    np.testing.assert_array_equal(lazy.compute(chunks=(2,), blocks=(1,))[:], expected)
    assert lazy._native_execution_report["route"] == "menudet-lazyudf"


def test_complex_comparisons_and_where(forbid_legacy):
    z = np.array([1 + 2j, 3 - 4j, -2 + 0.5j, 0 + 0j])
    comparisons = blosc2.lazyudf(
        blosc2.DSLKernel.from_source("def k(z):\n    return z < z\n"),
        (z,),
        dtype=bool,
        jit=False,
    )
    np.testing.assert_array_equal(comparisons[:], z < z)
    selected = blosc2.lazyudf(
        blosc2.DSLKernel.from_source("def k(z):\n    return where(z, 1.0, 2.0)\n"),
        (z,),
        dtype=np.float64,
        jit=False,
    )
    np.testing.assert_array_equal(selected[:], np.where(z, 1.0, 2.0))


@pytest.mark.parametrize("unsupported", ["z // 2", "ceil(z)", "float(z)"])
def test_complex_unsupported_operations_raise(forbid_legacy, unsupported):
    kernel = blosc2.DSLKernel.from_source(f"def k(z):\n    return {unsupported}\n")
    lazy = blosc2.lazyudf(kernel, (np.array([1 + 2j]),), dtype=np.complex128, jit=False)
    with pytest.raises(RuntimeError, match="Menudet compilation failed"):
        lazy[:]
