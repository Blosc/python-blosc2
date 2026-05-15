import numpy as np
import pytest

import blosc2


def test_group_reduce_size_and_sum_integer_keys():
    keys = np.array([2, 1, 2, 1, 2], dtype=np.int16)
    values = np.array([10, 1, 30, 3, 50], dtype=np.int32)

    groups, sizes = blosc2.group_reduce(keys, op="size", sort=True)
    groups2, sums = blosc2.group_reduce(keys, values, op="sum", sort=True)

    assert groups.dtype == keys.dtype
    np.testing.assert_array_equal(groups, np.array([1, 2], dtype=np.int16))
    np.testing.assert_array_equal(sizes, np.array([2, 3]))
    np.testing.assert_array_equal(groups2, np.array([1, 2], dtype=np.int16))
    np.testing.assert_array_equal(sums, np.array([4, 90]))


def test_group_reduce_integer_keys_float_aggs_with_nan_values():
    keys = np.array([0, 1, 0, 1, 2], dtype=np.uint16)
    values = np.array([1.0, np.nan, 3.0, np.nan, 10.0])

    groups, counts = blosc2.group_reduce(keys, values, op="count", sort=True)
    _, means = blosc2.group_reduce(keys, values, op="mean", sort=True)
    _, mins = blosc2.group_reduce(keys, values, op="min", sort=True)
    _, maxs = blosc2.group_reduce(keys, values, op="max", sort=True)

    np.testing.assert_array_equal(groups, np.array([0, 1, 2], dtype=np.uint16))
    np.testing.assert_array_equal(counts, np.array([2, 0, 1]))
    assert means[0] == 2.0
    assert np.isnan(means[1])
    assert means[2] == 10.0
    assert mins[0] == 1.0
    assert np.isnan(mins[1])
    assert mins[2] == 10.0
    assert maxs[0] == 3.0
    assert np.isnan(maxs[1])
    assert maxs[2] == 10.0


def test_group_reduce_arbitrary_float_keys_and_nan_key_group():
    keys = np.array([0.5, np.nan, 0.5, -0.0, 0.0, np.nan])
    values = np.array([1.0, 2.0, 3.0, 10.0, 20.0, 5.0])

    groups, sums = blosc2.group_reduce(keys, values, op="sum", sort=True, dropna=False)

    assert groups[0] == 0.0
    assert sums[0] == 30.0
    assert groups[1] == 0.5
    assert sums[1] == 4.0
    assert np.isnan(groups[2])
    assert sums[2] == 7.0


def test_group_reduce_dropna_default_skips_nan_keys():
    keys = np.array([1.0, np.nan, 1.0])
    values = np.array([2.0, 10.0, 3.0])

    groups, sums = blosc2.group_reduce(keys, values, op="sum", sort=True)

    np.testing.assert_array_equal(groups, np.array([1.0]))
    np.testing.assert_array_equal(sums, np.array([5.0]))


def test_group_reduce_rejects_bad_inputs():
    with pytest.raises(ValueError):
        blosc2.group_reduce(np.ones((2, 2)), op="size")
    with pytest.raises(ValueError):
        blosc2.group_reduce(np.arange(3), op="sum")
    with pytest.raises(ValueError):
        blosc2.group_reduce(np.arange(3), np.arange(2), op="sum")
    with pytest.raises(ValueError):
        blosc2.group_reduce(np.arange(3), op="bad")
