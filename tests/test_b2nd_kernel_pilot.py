import numpy as np
import pytest

import blosc2
from blosc2 import blosc2_ext

AVAILABLE = blosc2_ext._b2nd_kernel_pilot_available()
SCALE = np.float64(2)


@blosc2.menudet(jit=False)
def f(x):
    return x * SCALE + 1.0


@blosc2.menudet(jit=False)
def positive(x):
    return x > 0.0


@blosc2.menudet(jit=False, cardinality="block_scalar")
def partition_sum(x):
    return block_sum(x)  # noqa: F821


@pytest.mark.skipif(not AVAILABLE, reason="requires the C-Blosc2 scheduler pilot build")
@pytest.mark.parametrize(("size", "chunk", "block"), [(12, 6, 3), (7, 5, 2), (1, 1, 1)])
def test_b2nd_kernel_pilot_matches_adapter(size, chunk, block, tmp_path):
    values = np.arange(size, dtype=np.float64) - 2
    source = blosc2.asarray(values, chunks=(chunk,), blocks=(block,), urlpath=tmp_path / "input.b2nd")
    kernel = f.specialize(source)
    budget = 2 * block * values.itemsize
    actual = source._evaluate_b2nd_kernel_pilot(kernel, max_scratch=budget)
    np.testing.assert_array_equal(actual, kernel.lazy({"x": source})[:], strict=True)
    np.testing.assert_array_equal(actual, values * 2.0 + 1.0, strict=True)
    np.testing.assert_array_equal(
        source._evaluate_b2nd_kernel_pilot(kernel, max_scratch=64, partition=4), actual, strict=True
    )
    with pytest.raises(ValueError, match="scratch budget"):
        source._evaluate_b2nd_kernel_pilot(kernel, max_scratch=budget - 1)


@pytest.mark.skipif(not AVAILABLE, reason="requires the C-Blosc2 scheduler pilot build")
def test_b2nd_kernel_pilot_rejects_unsupported_contracts():
    source = blosc2.asarray(np.arange(4, dtype=np.float64))
    with pytest.raises(ValueError, match="float64 input and output"):
        source._evaluate_b2nd_kernel_pilot(positive.specialize(source))
    with pytest.raises(TypeError, match="prepared MenudetKernel"):
        source._evaluate_b2nd_kernel_pilot("def f(x):\n    return x + 1.0\n")
    kernel = f.specialize(source)
    with pytest.raises(ValueError, match="native float64 vector"):
        blosc2.asarray(np.arange(4, dtype=np.float32))._evaluate_b2nd_kernel_pilot(kernel)
    with pytest.raises(ValueError, match="explicit logical partition"):
        source._evaluate_b2nd_kernel_pilot(partition_sum.specialize(source))
    with pytest.raises(ValueError, match="positive int32"):
        source._evaluate_b2nd_kernel_pilot(kernel, partition=0)
    with pytest.raises(ValueError, match="directly prepared Menudet kernel"):
        source._evaluate_b2nd_kernel_pilot(blosc2.MenudetKernel.from_json(kernel.to_json()))


@pytest.mark.skipif(not AVAILABLE, reason="requires the C-Blosc2 scheduler pilot build")
@pytest.mark.parametrize(("chunk", "block"), [(5, 2), (4, 4)])
def test_b2nd_kernel_pilot_preserves_persisted_partitions(chunk, block, tmp_path):
    values = np.array([1e16, 1, -1e16, 3, 4, 5, 6], dtype=np.float64)
    source = blosc2.asarray(values, chunks=(chunk,), blocks=(block,), urlpath=tmp_path / "input.b2nd")
    operation = partition_sum.specialize(source).lazy({"x": source}, partitions=(3,))
    path = tmp_path / "reduction.b2nd"
    operation.save(path)
    restored = blosc2.open(path, mode="r")
    assert restored.partitions == (3,)
    stored = restored.inputs["x"]
    expected = restored[:]
    budget = (restored.partitions[0] + 1) * values.itemsize
    actual = stored._evaluate_b2nd_kernel_pilot(
        restored.kernel, max_scratch=budget, partition=restored.partitions[0]
    )
    np.testing.assert_array_equal(actual, expected, strict=True)
    np.testing.assert_array_equal(actual, np.array([0.0, 12.0, 6.0]), strict=True)
    with pytest.raises(ValueError, match="scratch budget"):
        stored._evaluate_b2nd_kernel_pilot(restored.kernel, max_scratch=budget - 1, partition=3)


@pytest.mark.skipif(not AVAILABLE, reason="requires the C-Blosc2 scheduler pilot build")
def test_b2nd_kernel_pilot_empty_logical_domain():
    source = blosc2.asarray(np.empty(0, dtype=np.float64))
    actual = source._evaluate_b2nd_kernel_pilot(
        partition_sum.specialize(source), max_scratch=0, partition=3
    )
    np.testing.assert_array_equal(actual, np.empty(0, dtype=np.float64), strict=True)


@pytest.mark.skipif(AVAILABLE, reason="requires a standard build")
def test_b2nd_kernel_pilot_is_disabled_by_default():
    with pytest.raises(NotImplementedError, match="B2_EXPERIMENTAL_SHARED_MINIEXPR_CORE"):
        blosc2.asarray(np.arange(4, dtype=np.float64))._evaluate_b2nd_kernel_pilot(None)
