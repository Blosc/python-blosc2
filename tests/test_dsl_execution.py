"""Bindings and dispatch contracts for source-authored Menudet execution."""

import importlib

import numpy as np
import pytest

import blosc2

pytestmark = pytest.mark.skipif(
    blosc2.IS_WASM or not blosc2.blosc2_ext.portable_descriptor_available(),
    reason="Native portable runtime required",
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
    assert observed[-1] is True
