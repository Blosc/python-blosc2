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

    _pandas_too_old = int(pd.__version__.split(".")[0]) < 3
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

    def test_apply_axis1_row_subscript_idiom_matches_default_engine(self):
        def add_people(row):
            return row["max_people"] + row["max_children"]

        df = pd.DataFrame({"max_people": [4, 2, 8], "max_children": [1, 0, 3]})
        expected = df.apply(add_people, axis=1)
        result = df.apply(add_people, engine=blosc2.jit, axis=1)
        pd.testing.assert_series_equal(result, expected)

    def test_apply_axis1_row_subscript_args_kwargs_forwarded(self):
        def combine(row, num1, num2=0):
            return row["a"] + row["b"] + num1 + num2

        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        expected = df.apply(combine, axis=1, args=(10,), num2=100)
        result = df.apply(combine, engine=blosc2.jit, axis=1, args=(10,), num2=100)
        pd.testing.assert_series_equal(result, expected)

    def test_apply_axis1_row_subscript_preserves_column_dtype(self):
        # a mixed-dtype frame would be upcast by DataFrame.values; the row
        # proxy must extract columns from the original frame instead.
        def add(row):
            return row["i"] + row["f"]

        df = pd.DataFrame({"i": np.array([1, 2, 3], dtype=np.int64), "f": [0.5, 0.5, 0.5]})
        result = df.apply(add, engine=blosc2.jit, axis=1)
        np.testing.assert_allclose(result.to_numpy(), [1.5, 2.5, 3.5])

    def test_apply_axis1_row_subscript_with_loop_raises_clear_error(self):
        def kepler_row(row):
            m, ecc = row["m"], row["ecc"]
            e = m + ecc * np.sin(m)
            for _ in range(50):
                diff = (e - ecc * np.sin(e) - m) / (1.0 - ecc * np.cos(e))
                e = e - diff
            return e

        df = pd.DataFrame({"m": [0.1, 0.5], "ecc": [0.1, 0.2]})
        with pytest.raises(TypeError, match="for/while loop"):
            df.apply(kepler_row, engine=blosc2.jit, axis=1)

    def test_apply_axis1_row_subscript_duplicate_column_raises(self):
        def add(row):
            return row["a"] + 1

        df = pd.DataFrame(np.ones((2, 2)), columns=["a", "a"])
        with pytest.raises(KeyError, match="duplicated"):
            df.apply(add, engine=blosc2.jit, axis=1)

    def test_apply_axis1_row_subscript_attribute_access_raises(self):
        def bad(row):
            return row["a"] + row.b

        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        with pytest.raises(AttributeError, match="row\\['b'\\]"):
            df.apply(bad, engine=blosc2.jit, axis=1)

    def test_apply_axis1_row_subscript_non_numeric_column_raises(self):
        # Whole-frame numeric-dtype validation (`_ensure_numpy_data`) already
        # gates this ahead of row-proxy dispatch; `_PandasRowProxy` carries
        # its own per-column check too, for callers that construct it
        # directly.
        def bad(row):
            return row["a"] + len(row["b"])

        df = pd.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"]})
        with pytest.raises(ValueError, match="numeric dtype"):
            df.apply(bad, engine=blosc2.jit, axis=1)

    def test_apply_axis1_positional_idiom_still_uses_per_row_loop(self):
        # No `row["..."]` subscript: falls back to the historical per-row
        # loop, unaffected by the row-proxy dispatch added for the subscript
        # idiom above.
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        expected = df.apply(lambda row: row * 2, axis=1)
        result = df.apply(lambda row: row * 2, engine=blosc2.jit, axis=1)
        pd.testing.assert_frame_equal(result, expected)

    def test_apply_already_jitted_function_is_not_decorated_twice(self):
        # Decorating and passing engine= both request the same thing. Applying
        # the decorator a second time used to wrap the array in a SimpleProxy
        # before the inner DSL kernel saw it, which then failed asking for
        # `shape=`. Traced functions tolerated it, so only branches broke.
        def branch(col):
            if col >= 0:
                out = col + 1.0
            else:
                out = col - 1.0
            return out

        df = pd.DataFrame({"a": [-2.0, 1.0, 3.0], "b": [4.0, -5.0, 6.0]})
        expected = df.apply(lambda col: np.where(col >= 0, col + 1.0, col - 1.0))

        for func in (branch, blosc2.jit(branch)):
            result = df.apply(func, engine=blosc2.jit)
            pd.testing.assert_frame_equal(result, expected)

    def test_map_already_jitted_function_is_not_decorated_twice(self):
        def branch(col):
            if col >= 0:
                out = col * 2.0
            else:
                out = col
            return out

        s = pd.Series([-2.0, 1.0, 3.0])
        expected = np.where(s.to_numpy() >= 0, s.to_numpy() * 2.0, s.to_numpy())

        for func in (branch, blosc2.jit(branch)):
            result = s.map(func, engine=blosc2.jit)
            np.testing.assert_allclose(np.asarray(result), expected)

    def test_apply_engine_accepts_configured_jit(self):
        # pandas gates on hasattr(engine, "__pandas_udf__") and then uses the
        # engine object as the decorator, so a configured blosc2.jit(...) call
        # is a valid engine -- the only way to reach strict= through apply().
        from blosc2.dsl_kernel import DSLSyntaxError

        def branch(col):
            if col >= 0:
                out = col + 1.0
            else:
                out = col - 1.0
            return out

        def not_dsl(col):
            return np.where(col >= 0, col.mean() + 1.0, col - 1.0)

        df = pd.DataFrame({"a": [-2.0, 1.0, 3.0], "b": [4.0, -5.0, 6.0]})
        expected = df.apply(lambda col: np.where(col >= 0, col + 1.0, col - 1.0))

        result = df.apply(branch, engine=blosc2.jit(strict=True))
        pd.testing.assert_frame_equal(result, expected)

        # strict=True refuses to silently fall back to tracing
        with pytest.raises(DSLSyntaxError):
            df.apply(not_dsl, engine=blosc2.jit(strict=True))

        # strict=False forces the tracing route instead
        df.apply(not_dsl, engine=blosc2.jit(strict=False))

    def test_columns_by_keyword_unpacking(self):
        # doc/guides/pandas_engine.md's row-wise pattern: a DataFrame is a
        # mapping of column name to Series, so `kernel(**df)` passes each
        # column as a keyword argument. Both jit routes must accept that.
        @blosc2.jit
        def traced(a, b):
            return np.sqrt(a * a + b * b)

        @blosc2.jit
        def dsl(a, b):
            if a > b:
                out = a - b
            else:
                out = b - a
            return out

        df = pd.DataFrame({"a": [-2.0, 1.0, 3.0], "b": [4.0, -5.0, 6.0]})

        np.testing.assert_allclose(np.asarray(traced(**df)), np.hypot(df["a"], df["b"]))
        np.testing.assert_allclose(np.asarray(dsl(**df)), np.abs(df["a"] - df["b"]))
