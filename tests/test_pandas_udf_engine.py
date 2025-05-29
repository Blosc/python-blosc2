#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

import numpy as np
import pytest

import blosc2


class TestPandasUDF:
    def test_map(self):
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
                skip_na=False,
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
