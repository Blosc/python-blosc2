"""Small workload/binding regression; native operator matrices live in miniexpr."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

import blosc2

pytestmark = pytest.mark.skipif(blosc2.IS_WASM, reason="Host JIT workload is qualified outside WASM")


def test_tip11_hash_domain_partitions_and_signed_output():
    path = Path(__file__).resolve().parents[1] / "bench/optim_tips/tip_11_dsl_random.py"
    spec = importlib.util.spec_from_file_location("tip11", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        native = module.random_int32.specialize()
    except RuntimeError as error:
        if "Required Menudet JIT is unavailable" in str(error):
            pytest.skip(str(error))
        raise
    assert native.has_jit
    lazy = native.lazy({}, shape=(10003,), partitions=(257,))
    expected = module.reference(np.arange(10003))
    assert lazy.dtype == np.dtype("int32")
    np.testing.assert_array_equal(lazy[:], expected)
    np.testing.assert_array_equal(lazy[251:523], expected[251:523])
    np.testing.assert_array_equal(
        lazy.compute(chunks=(1024,), blocks=(128,), cparams={"clevel": 0})[:], expected
    )
    np.testing.assert_array_equal(
        lazy.compute(chunks=(1028,), blocks=(257,), cparams={"clevel": 0})[:], expected
    )
    # A small high-origin block checks the full uint32 domain without allocating it.
    for origin in (2**31 - 3, 2**32 - 7):
        actual = native.evaluate_block({}, block_shape=(7,), logical_shape=(2**32,), block_origin=(origin,))
        np.testing.assert_array_equal(actual, module.reference(np.arange(origin, origin + 7)))
    interpreter = blosc2.MenudetKernel.from_json(native.to_json(), jit=False)
    np.testing.assert_array_equal(interpreter.lazy({}, shape=(10003,), partitions=(257,))[:], expected)
