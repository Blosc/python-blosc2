#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

import numpy as np

import blosc2


class TestPandasUDF:
    def test_map_1d(self):
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
        assert result.shape == (2,)
        assert result[0] == 2
        assert result[1] == 3

    def test_map_1d_with_args(self):
        def add_numbers(x, num1, num2):
            return x + num1 + num2

        data = np.array([1, 2])

        result = blosc2.jit.__pandas_udf__.map(
            data,
            add_numbers,
            args=(10,),
            kwargs={"num2": 100},
            decorator=blosc2.jit,
            skip_na=False,
        )
        assert result.shape == (2,)
        assert result[0] == 111
        assert result[1] == 112

    def test_map_2d(self):
        def add_one(x):
            return x + 1

        data = np.array([[1, 2], [3, 4]])

        result = blosc2.jit.__pandas_udf__.map(
            data,
            add_one,
            args=(),
            kwargs={},
            decorator=blosc2.jit,
            skip_na=False,
        )
        assert result.shape == (2, 2)
        assert result[0, 0] == 2
        assert result[0, 1] == 3
        assert result[1, 0] == 4
        assert result[1, 1] == 5

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
            return x + 1

        data = np.array([[1, 2], [3, 4]])

        result = blosc2.jit.__pandas_udf__.apply(
            data,
            add_one,
            args=(),
            kwargs={},
            decorator=blosc2.jit,
            axis=0,
        )
        assert result.shape == (2, 2)
        assert result[0, 0] == 2
        assert result[0, 1] == 3
        assert result[1, 0] == 4
        assert result[1, 1] == 5
