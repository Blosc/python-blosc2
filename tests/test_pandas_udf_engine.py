#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

import numpy as np
import pytest

import blosc2


class TestPandasUDF:
    def test_map(self):
        def add_one(x):
            return x + 1

        data = np.array([1, 2])

        result = blosc2.jit.__pandas_udf__.map(
            data,
            add_one,
            args=(),
            kwargs={},
            decorator=blosc2.jit,
            skip_na=False,
        )
        assert np.array_equal(result, np.array([2, 3]))

    def test_map_skip_na_not_supported(self):
        def add_one(x):
            return x + 1

        data = np.array([1, 2])

        with pytest.raises(NotImplementedError):
            blosc2.jit.__pandas_udf__.map(
                data,
                add_one,
                args=(),
                kwargs={},
                decorator=blosc2.jit,
                skip_na=True,
            )

    def test_apply_1d(self):
        def add_one(x):
            return x + 1

        data = np.array([1, 2])

        result = blosc2.jit.__pandas_udf__.apply(
            data,
            add_one,
            args=(),
            kwargs={},
            decorator=blosc2.jit,
            axis=0,
        )
        assert result.shape == (2,)
        assert result[0] == 2
        assert result[1] == 3

    def test_apply_1d_with_args(self):
        def add_numbers(x, num1, num2):
            return x + num1 + num2

        data = np.array([1, 2])

        result = blosc2.jit.__pandas_udf__.apply(
            data,
            add_numbers,
            args=(10,),
            kwargs={"num2": 100},
            decorator=blosc2.jit,
            axis=0,
        )
        assert result.shape == (2,)
        assert result[0] == 111
        assert result[1] == 112

    def test_apply_2d(self):
        def add_one(x):
            assert x.shape == (2, 3)
            return x + 1

        data = np.array([[1, 2, 3], [4, 5, 6]])

        result = blosc2.jit.__pandas_udf__.apply(
            data,
            add_one,
            args=(),
            kwargs={},
            decorator=blosc2.jit,
            axis=None,
        )
        expected = np.array([[2, 3, 4], [5, 6, 7]])
        assert np.array_equal(result, expected)

    def test_apply_2d_by_column(self):
        def add_one(x):
            assert x.shape == (2,)
            return x + 1

        data = np.array([[1, 2, 3], [4, 5, 6]])

        result = blosc2.jit.__pandas_udf__.apply(
            data,
            add_one,
            args=(),
            kwargs={},
            decorator=blosc2.jit,
            axis=0,
        )
        expected = np.array([[2, 3, 4], [5, 6, 7]])
        assert np.array_equal(result, expected)

    def test_apply_2d_by_row(self):
        def add_one(x):
            assert x.shape == (3,)
            return x + 1

        data = np.array([[1, 2, 3], [4, 5, 6]])

        result = blosc2.jit.__pandas_udf__.apply(
            data,
            add_one,
            args=(),
            kwargs={},
            decorator=blosc2.jit,
            axis=1,
        )
        expected = np.array([[2, 3, 4], [5, 6, 7]])
        assert np.array_equal(result, expected)


try:
    import pandas as pd

    _pandas_too_old = pd.__version__ < "3"
except ImportError:
    pd = None
    _pandas_too_old = False


@pytest.mark.skipif(pd is None, reason="pandas not installed")
@pytest.mark.skipif(_pandas_too_old, reason="engine= integration targets pandas 3.x")
class TestPandasEngineEndToEnd:
    """Exercises engine=blosc2.jit through real pandas, not the adapter directly."""

    def test_apply_axis0_matches_default_engine(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        expected = df.apply(lambda x: x + 1)
        result = df.apply(lambda x: x + 1, engine=blosc2.jit)
        pd.testing.assert_frame_equal(result, expected)

    def test_apply_axis1_matches_default_engine(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        expected = df.apply(lambda x: x + 1, axis=1)
        result = df.apply(lambda x: x + 1, engine=blosc2.jit, axis=1)
        pd.testing.assert_frame_equal(result, expected)

    def test_apply_args_and_kwargs_forwarded(self):
        def add_numbers(x, num1, num2=0):
            return x + num1 + num2

        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        expected = df.apply(add_numbers, args=(10,), num2=100)
        result = df.apply(add_numbers, engine=blosc2.jit, args=(10,), num2=100)
        pd.testing.assert_frame_equal(result, expected)

    def test_series_map_matches_default_engine(self):
        s = pd.Series([1.0, 2.0, 3.0])
        expected = s.map(lambda x: x + 1)
        result = s.map(lambda x: x + 1, engine=blosc2.jit)
        pd.testing.assert_series_equal(result, expected)

    def test_apply_object_dtype_raises_clear_error(self):
        df = pd.DataFrame({"a": ["x", "y"]})
        with pytest.raises(ValueError, match="numeric dtype"):
            df.apply(lambda x: x + 1, engine=blosc2.jit)
