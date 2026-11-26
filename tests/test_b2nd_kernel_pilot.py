import numpy as np
import pytest

import blosc2
from blosc2 import blosc2_ext

SOURCE = "def f(x):\n    return x * 2.0 + 1.0\n"
AVAILABLE = blosc2_ext._b2nd_kernel_pilot_available()


@blosc2.menudet(jit=False)
def f(x):
    return x * 2.0 + 1.0


@pytest.mark.skipif(not AVAILABLE, reason="requires the C-Blosc2 scheduler pilot build")
@pytest.mark.parametrize(("size", "chunk", "block"), [(12, 6, 3), (7, 5, 2), (1, 1, 1)])
def test_b2nd_kernel_pilot_matches_adapter(size, chunk, block, tmp_path):
    values = np.arange(size, dtype=np.float64) - 2
    source = blosc2.asarray(values, chunks=(chunk,), blocks=(block,), urlpath=tmp_path / "input.b2nd")
    budget = 2 * block * values.itemsize
    actual = source._evaluate_b2nd_kernel_pilot(SOURCE, max_scratch=budget)
    np.testing.assert_array_equal(actual, f(source)[:], strict=True)
    np.testing.assert_array_equal(actual, values * 2.0 + 1.0, strict=True)
    with pytest.raises(ValueError, match="scratch budget"):
        source._evaluate_b2nd_kernel_pilot(SOURCE, max_scratch=budget - 1)


@pytest.mark.skipif(not AVAILABLE, reason="requires the C-Blosc2 scheduler pilot build")
def test_b2nd_kernel_pilot_rejects_unsupported_contracts():
    source = blosc2.asarray(np.arange(4, dtype=np.float64))
    with pytest.raises(ValueError):
        source._evaluate_b2nd_kernel_pilot("def f(x):\n    return x > 0.0\n")
    with pytest.raises(ValueError, match="native float64 vector"):
        blosc2.asarray(np.arange(4, dtype=np.float32))._evaluate_b2nd_kernel_pilot(SOURCE)


@pytest.mark.skipif(AVAILABLE, reason="requires a standard build")
def test_b2nd_kernel_pilot_is_disabled_by_default():
    with pytest.raises(NotImplementedError, match="B2_EXPERIMENTAL_SHARED_MINIEXPR_CORE"):
        blosc2.asarray(np.arange(4, dtype=np.float64))._evaluate_b2nd_kernel_pilot(SOURCE)
