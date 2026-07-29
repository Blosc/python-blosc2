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

    def test_axis1_subscript_matches_default_engine(self):
        def add_people(row):
            return row["max_people"] + row["max_children"]

        df = pd.DataFrame({"max_people": [4, 2, 8], "max_children": [1, 0, 3]})
        expected = df.apply(add_people, axis=1)
        result = df.apply(add_people, engine=blosc2.jit, axis=1)
        pd.testing.assert_series_equal(result, expected)

    def test_axis1_subscript_args_kwargs_forwarded(self):
        def combine(row, num1, num2=0):
            return row["a"] + row["b"] + num1 + num2

        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        expected = df.apply(combine, axis=1, args=(10,), num2=100)
        result = df.apply(combine, engine=blosc2.jit, axis=1, args=(10,), num2=100)
        pd.testing.assert_series_equal(result, expected)

    def test_axis1_subscript_keeps_column_dtype(self):
        # a mixed-dtype frame would be upcast by DataFrame.values; the row
        # proxy must extract columns from the original frame instead.
        def add(row):
            return row["i"] + row["f"]

        df = pd.DataFrame({"i": np.array([1, 2, 3], dtype=np.int64), "f": [0.5, 0.5, 0.5]})
        result = df.apply(add, engine=blosc2.jit, axis=1)
        np.testing.assert_allclose(result.to_numpy(), [1.5, 2.5, 3.5])

    def test_axis1_subscript_with_loop_raises(self):
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

    def test_axis1_subscript_duplicate_col_raises(self):
        def add(row):
            return row["a"] + 1

        df = pd.DataFrame(np.ones((2, 2)), columns=["a", "a"])
        with pytest.raises(KeyError, match="duplicated"):
            df.apply(add, engine=blosc2.jit, axis=1)

    def test_axis1_subscript_attr_access_raises(self):
        def bad(row):
            return row["a"] + row.b

        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        with pytest.raises(AttributeError, match="row\\['b'\\]"):
            df.apply(bad, engine=blosc2.jit, axis=1)

    def test_axis1_subscript_unvectorizable_raises(self):
        # String columns are supported now, so the per-column check in
        # `_PandasRowProxy` is what still rejects a dtype the engine cannot
        # vectorize at all.
        def bad(row):
            return row["a"] + row["b"]

        df = pd.DataFrame({"a": [1.0, 2.0], "b": pd.to_datetime(["2020-01-01", "2020-01-02"])})
        with pytest.raises(ValueError, match="cannot vectorize"):
            df.apply(bad, engine=blosc2.jit, axis=1)

    def test_axis1_positional_uses_per_row_loop(self):
        # No `row["..."]` subscript: falls back to the historical per-row
        # loop, unaffected by the row-proxy dispatch added for the subscript
        # idiom above.
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        expected = df.apply(lambda row: row * 2, axis=1)
        result = df.apply(lambda row: row * 2, engine=blosc2.jit, axis=1)
        pd.testing.assert_frame_equal(result, expected)

    def test_apply_jitted_func_not_decorated_twice(self):
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

    def test_map_jitted_func_not_decorated_twice(self):
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
        # column as a keyword argument. Both jit routes must accept that, and
        # bind by name: the columns below are in neither the parameter order
        # nor alphabetical order, and the operations are asymmetric, so a
        # positional binding would give a different (wrong) answer.
        @blosc2.jit
        def traced(a, b):
            return b - a * 2.0

        @blosc2.jit
        def dsl(a, b):
            if a > b:
                out = a - b
            else:
                out = b - a * 2.0
            return out

        df = pd.DataFrame({"b": [4.0, -5.0, 6.0], "a": [-2.0, 1.0, 3.0]})
        assert list(df.columns) == ["b", "a"]

        np.testing.assert_allclose(np.asarray(traced(**df)), np.asarray(traced(df["a"], df["b"])))
        np.testing.assert_allclose(np.asarray(traced(**df)), df["b"] - df["a"] * 2.0)
        np.testing.assert_allclose(np.asarray(dsl(**df)), np.asarray(dsl(df["a"], df["b"])))

    def test_wide_frame_kwargs_error_names_the_fix(self):
        # Extra columns are rejected, not dropped: a keyword that goes nowhere
        # would otherwise fail silently. The message must name the subsetting fix.
        @blosc2.jit
        def traced(a, b):
            return a + b

        @blosc2.jit
        def dsl(a, b):
            if a > b:
                out = a - b
            else:
                out = b - a
            return out

        df = pd.DataFrame({"a": [1.0, 2.0], "b": [4.0, 5.0], "note": [7.0, 8.0]})

        # (`func.__name__` is the jit wrapper's; the message uses the kernel's)
        for name, func in (("traced", traced), ("dsl", dsl)):
            with pytest.raises(TypeError) as excinfo:
                func(**df)
            message = str(excinfo.value)
            assert name in message
            assert "'note'" in message
            assert "**df[['a', 'b']]" in message

        # A missing operand is a different mistake and keeps its own message
        with pytest.raises(TypeError, match="missing a required argument"):
            dsl(a=df["a"])


@pytest.mark.skipif(pd is None, reason="pandas not installed")
@pytest.mark.skipif(_pandas_too_old, reason="engine= integration targets pandas 3.x")
class TestRowKernelsWithControlFlow:
    """`row["colname"]` combined with an `if`.

    Neither dispatch route could run these before: tracing evaluates the `if`
    over a whole column ("truth value ... is ambiguous") and the DSL parser
    rejected the subscript.  They are now rewritten into named parameters.
    """

    def test_numeric_row_kernel_with_branch(self):
        def pick(row):
            if row["a"] > 2:
                return row["a"] + row["b"]
            return row["a"] - row["b"]

        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [10.0, 20.0, 30.0, 40.0]})
        expected = df.apply(pick, axis=1)
        result = df.apply(pick, axis=1, engine=blosc2.jit)
        pd.testing.assert_series_equal(result, expected)

    def test_blog_kernel_matches_default_engine(self):
        """End-to-end acceptance: the pandas-3 blog kernel, run unmodified."""

        def format_room_info(row):
            result = "property_type=" + row["property_type"]
            desc = row["name"].lower()
            if " with " not in desc:
                return result + ", room_type=" + desc.removesuffix(" room")
            before, after = desc.split(" with ", 1)
            r2 = result + ", room_type=" + before.removesuffix(" room")
            return r2 + ", amenity=" + after

        df = pd.DataFrame(
            {
                "property_type": ["Entire home", "Private room", "Shared room", "Loft"] * 8,
                "name": [
                    "Cozy Loft With City View",
                    "Small Single Room",
                    "Studio with balcony",
                    "Double Room",
                ]
                * 8,
            }
        )
        expected = df.apply(format_room_info, axis=1)
        result = df.apply(format_room_info, axis=1, engine=blosc2.jit)
        pd.testing.assert_series_equal(result, expected)

    def test_column_named_like_a_dsl_function(self):
        """A column name that shadows a DSL builtin must not change meaning.

        The rewrite turns row["sqrt"] into a parameter literally called
        `sqrt`, which then coexists with a real sqrt() call in the same
        expression; operands and calls are distinguished by position, so both
        resolve correctly.  Index symbols (`_i0`) are checked for the same
        reason.
        """

        def collide(row):
            return row["sqrt"] + np.sqrt(row["b"])

        def index_symbol(row):
            return row["_i0"] + row["b"]

        df = pd.DataFrame({"sqrt": [1.0, 2.0], "b": [4.0, 9.0], "_i0": [5.0, 6.0]})
        for fn in (collide, index_symbol):
            pd.testing.assert_series_equal(df.apply(fn, axis=1, engine=blosc2.jit), df.apply(fn, axis=1))

    def test_non_identifier_column_label(self):
        def tag(row):
            if row["room type"] == "loft":
                return "L"
            return "-"

        df = pd.DataFrame({"room type": ["loft", "studio", "loft"]})
        expected = df.apply(tag, axis=1)
        result = df.apply(tag, axis=1, engine=blosc2.jit)
        pd.testing.assert_series_equal(result, expected)

    def test_null_string_column_is_rejected(self):
        # pandas raises on a row kernel over a null too; substituting "" would
        # invent a value it never produces.
        def concat(row):
            return "p=" + row["x"]

        df = pd.DataFrame({"x": ["a", None, "c"]})
        with pytest.raises(TypeError):
            df.apply(concat, axis=1)
        with pytest.raises(ValueError, match="contains nulls"):
            df.apply(concat, axis=1, engine=blosc2.jit)

    def test_positional_row_access_still_rejected(self):
        # Only `row["literal"]` is rewritten; anything else must keep failing
        # loudly rather than silently taking a different route.
        def positional(row):
            if row[0] > 1:
                return row[0]
            return row[1]

        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        with pytest.raises((TypeError, ValueError, RuntimeError)):
            df.apply(positional, axis=1, engine=blosc2.jit)
