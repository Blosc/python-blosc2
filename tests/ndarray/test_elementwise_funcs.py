import sys
import warnings

import numpy as np
import pytest
import torch

import blosc2

warnings.simplefilter("always")

# Functions to test (add more as needed)
UNARY_FUNC_PAIRS = []
BINARY_FUNC_PAIRS = []
UNSUPPORTED_UFUNCS = []

for name, obj in vars(np).items():
    if isinstance(obj, np.ufunc):
        if hasattr(blosc2, name):
            blosc_func = getattr(blosc2, name)
            if obj.nin == 1:
                UNARY_FUNC_PAIRS.append((obj, blosc_func))
            elif obj.nin == 2:
                BINARY_FUNC_PAIRS.append((obj, blosc_func))
        else:
            UNSUPPORTED_UFUNCS.append(obj)

# If you want to see which ones are enabled and which not, uncomment following
# print("Unary functions supported:", [f[0].__name__ for f in UNARY_FUNC_PAIRS])
# print("Binary functions supported:", [f[0].__name__ for f in BINARY_FUNC_PAIRS])
# print("NumPy ufuncs not in Blosc2:", [f.__name__ for f in UNSUPPORTED_UFUNCS]) <- all not in array-api
UNARY_FUNC_PAIRS.append((np.round, blosc2.round))
UNARY_FUNC_PAIRS.append((np.count_nonzero, blosc2.count_nonzero))

DTYPES = [blosc2.bool_, blosc2.int32, blosc2.int64, blosc2.float32, blosc2.float64, blosc2.complex128]
STR_DTYPES = ["bool", "int32", "int64", "float32", "float64", "complex128"]
SHAPES_CHUNKS = [((10,), (3,)), ((20, 20), (4, 7))]
SHAPES_CHUNKS_HEAVY = [((10, 13, 13), (3, 5, 2))]


def _test_unary_func_impl(np_func, blosc_func, dtype, shape, chunkshape):
    """Helper function containing the actual test logic for unary functions."""
    if np_func.__name__ in ("arccos", "arcsin", "arctanh"):
        a_blosc = blosc2.linspace(
            0.01, stop=0.99, num=np.prod(shape), chunks=chunkshape, shape=shape, dtype=dtype
        )
        if not blosc2.isdtype(dtype, "integral"):
            a_blosc[tuple(i // 2 for i in shape)] = blosc2.nan
        if dtype == blosc2.complex128:
            a_blosc = (a_blosc * (1 + 1j)).compute()
            a_blosc[tuple(i // 2 for i in shape)] = blosc2.nan + blosc2.nan * 1j
        if dtype == blosc2.bool_ and np_func.__name__ == "arctanh":
            a_blosc = blosc2.zeros(chunks=chunkshape, shape=shape, dtype=dtype)
    else:
        a_blosc = blosc2.linspace(
            1, stop=np.prod(shape), num=np.prod(shape), chunks=chunkshape, shape=shape, dtype=dtype
        )
        if not blosc2.isdtype(dtype, "integral"):
            a_blosc[tuple(i // 2 for i in shape)] = blosc2.nan
        if dtype == blosc2.complex128:
            a_blosc = (
                a_blosc
                + blosc2.linspace(
                    1j,
                    stop=np.prod(shape) * 1j,
                    num=np.prod(shape),
                    chunks=chunkshape,
                    shape=shape,
                    dtype=dtype,
                )
            ).compute()
            a_blosc[tuple(i // 2 for i in shape)] = blosc2.nan + blosc2.nan * 1j

    arr = a_blosc[()]
    success = False
    try:
        expected = np_func(arr) if np_func.__name__ != "reciprocal" else 1.0 / arr
        success = True
    except TypeError:
        assert True
    except RuntimeWarning as e:
        assert True
    if success:
        try:
            result = blosc_func(a_blosc)[...]
            np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)
        except TypeError as e:
            # some functions don't support certain dtypes and that's fine
            assert True
        except ValueError as e:
            if np_func.__name__ == "logical_not" and dtype in (
                blosc2.float32,
                blosc2.float64,
                blosc2.complex128,
            ):
                assert True
            else:
                raise e


def _test_binary_func_proxy(np_func, blosc_func, dtype, shape, chunkshape, xp):  # noqa: C901
    dtype_ = getattr(xp, dtype) if hasattr(xp, dtype) else np.dtype(dtype)
    dtype = np.dtype(dtype)
    not_blosc1 = xp.ones(shape, dtype=dtype_)
    if np_func.__name__ in ("right_shift", "left_shift"):
        a_blosc2 = blosc2.asarray(2, copy=True)
    else:
        a_blosc2 = blosc2.linspace(
            start=np.prod(shape) * 2,
            stop=np.prod(shape),
            num=np.prod(shape),
            chunks=chunkshape,
            shape=shape,
            dtype=dtype,
        )
        if not blosc2.isdtype(dtype, "integral"):
            a_blosc2[tuple(i // 2 for i in shape)] = blosc2.nan
        if dtype == blosc2.complex128:
            a_blosc2 = (
                a_blosc2
                + blosc2.linspace(
                    1j,
                    stop=np.prod(shape) * 1j,
                    num=np.prod(shape),
                    chunks=chunkshape,
                    shape=shape,
                    dtype=dtype,
                )
            ).compute()
            a_blosc2[tuple(i // 2 for i in shape)] = blosc2.nan + blosc2.nan * 1j
    arr1 = np.asarray(not_blosc1)
    arr2 = a_blosc2[()]
    success = False
    try:
        expected = np_func(arr1, arr2)
        success = True
    except TypeError:
        assert True
    except RuntimeWarning as e:
        assert True
    if success:
        try:
            result = blosc_func(not_blosc1, a_blosc2)[()]
            np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)
        except TypeError as e:
            # some functions don't support certain dtypes and that's fine
            assert True
        except ValueError as e:  # shouldn't be allowed for non-booleans
            if np_func.__name__ in ("logical_and", "logical_or", "logical_xor"):
                assert True
            if (
                np_func.__name__ in ("less", "less_equal", "greater", "greater_equal", "minimum", "maximum")
                and dtype == blosc2.complex128
            ):  # not supported for complex dtypes
                assert True
            else:
                raise e
        except NotImplementedError as e:
            if np_func.__name__ in ("left_shift", "right_shift", "floor_divide", "power", "remainder"):
                assert True
            else:
                raise e
        except AssertionError as e:
            if np_func.__name__ == "power" and blosc2.isdtype(
                dtype, "integral"
            ):  # overflow causes disagreement, no problem
                assert True
            elif np_func.__name__ in ("maximum", "minimum") and blosc2.isdtype(dtype, "real floating"):
                warnings.showwarning(
                    "minimum and maximum for numexpr do not match NaN behaviour for numpy",
                    UserWarning,
                    __file__,
                    0,
                    file=sys.stderr,
                )
                pytest.skip("minimum and maximum for numexpr do not match NaN behaviour for numpy")
            else:
                raise e


def _test_unary_func_proxy(np_func, blosc_func, dtype, shape, xp):
    dtype_ = getattr(xp, dtype) if hasattr(xp, dtype) else np.dtype(dtype)
    dtype = np.dtype(dtype)
    a_blosc = xp.ones(shape, dtype=dtype_)
    if not blosc2.isdtype(dtype, "integral"):
        a_blosc[tuple(i // 2 for i in shape)] = xp.nan
    if dtype == blosc2.complex128:
        a_blosc[tuple(i // 4 for i in shape)] = 1 + 1j
        a_blosc[tuple(i // 2 for i in shape)] = xp.nan + xp.nan * 1j
    if dtype == blosc2.bool_ and np_func.__name__ == "arctanh":
        a_blosc = xp.zeros(shape, dtype=dtype_)

    arr = np.asarray(a_blosc)
    success = False
    try:
        expected = np_func(arr) if np_func.__name__ != "reciprocal" else 1.0 / arr
        success = True
    except TypeError:
        assert True
    except RuntimeWarning as e:
        assert True
    if success:
        try:
            result = blosc_func(a_blosc)[...]
            np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)
        except TypeError as e:
            # some functions don't support certain dtypes and that's fine
            assert True
        except ValueError as e:
            if np_func.__name__ == "logical_not" and dtype in (
                blosc2.float32,
                blosc2.float64,
                blosc2.complex128,
            ):
                assert True
            else:
                raise e


def _test_binary_func_impl(np_func, blosc_func, dtype, shape, chunkshape):  # noqa: C901
    """Helper function containing the actual test logic for binary functions."""
    a_blosc1 = blosc2.linspace(
        1, stop=np.prod(shape), num=np.prod(shape), chunks=chunkshape, shape=shape, dtype=dtype
    )
    if np_func.__name__ in ("right_shift", "left_shift"):
        a_blosc2 = blosc2.asarray(2, copy=True)
    else:
        a_blosc2 = blosc2.linspace(
            start=np.prod(shape) * 2,
            stop=np.prod(shape),
            num=np.prod(shape),
            chunks=chunkshape,
            shape=shape,
            dtype=dtype,
        )
    if not blosc2.isdtype(dtype, "integral"):
        a_blosc1[tuple(i // 2 for i in shape)] = blosc2.nan
    if dtype == blosc2.complex128:
        a_blosc1 = (
            a_blosc1
            + blosc2.linspace(
                1j, stop=np.prod(shape) * 1j, num=np.prod(shape), chunks=chunkshape, shape=shape, dtype=dtype
            )
        ).compute()
        a_blosc1[tuple(i // 2 for i in shape)] = blosc2.nan + blosc2.nan * 1j
    arr1 = a_blosc1[()]
    arr2 = a_blosc2[()]
    success = False
    try:
        expected = np_func(arr1, arr2)
        success = True
    except TypeError:
        assert True
    except RuntimeWarning as e:
        assert True
    if success:
        try:
            result = blosc_func(a_blosc1, a_blosc2)[...]
            np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)
        except TypeError as e:
            # some functions don't support certain dtypes and that's fine
            assert True
        except ValueError as e:  # shouldn't be allowed for non-booleans
            if np_func.__name__ in ("logical_and", "logical_or", "logical_xor"):
                assert True
            if (
                np_func.__name__ in ("less", "less_equal", "greater", "greater_equal", "minimum", "maximum")
                and dtype == blosc2.complex128
            ):  # not supported for complex dtypes
                assert True
            else:
                raise e
        except NotImplementedError as e:
            if np_func.__name__ in ("left_shift", "right_shift", "floor_divide", "power", "remainder"):
                assert True
            else:
                raise e
        except AssertionError as e:
            if np_func.__name__ == "power" and blosc2.isdtype(
                dtype, "integral"
            ):  # overflow causes disagreement, no problem
                assert True
            elif np_func.__name__ in ("maximum", "minimum") and blosc2.isdtype(dtype, "real floating"):
                warnings.showwarning(
                    "minimum and maximum for numexpr do not match NaN behaviour for numpy",
                    UserWarning,
                    __file__,
                    0,
                    file=sys.stderr,
                )
                pytest.skip("minimum and maximum for numexpr do not match NaN behaviour for numpy")
            else:
                raise e


@pytest.mark.parametrize(("np_func", "blosc_func"), UNARY_FUNC_PAIRS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(("shape", "chunkshape"), SHAPES_CHUNKS)
def test_unary_funcs(np_func, blosc_func, dtype, shape, chunkshape):
    _test_unary_func_impl(np_func, blosc_func, dtype, shape, chunkshape)


@pytest.mark.parametrize(("np_func", "blosc_func"), UNARY_FUNC_PAIRS)
@pytest.mark.parametrize("dtype", STR_DTYPES)
@pytest.mark.parametrize("shape", [(10,), (20, 20)])
@pytest.mark.parametrize("xp", [torch])
def test_unfuncs_proxy(np_func, blosc_func, dtype, shape, xp):
    _test_unary_func_proxy(np_func, blosc_func, dtype, shape, xp)


@pytest.mark.heavy
@pytest.mark.parametrize(("np_func", "blosc_func"), UNARY_FUNC_PAIRS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(("shape", "chunkshape"), SHAPES_CHUNKS_HEAVY)
def test_unary_funcs_heavy(np_func, blosc_func, dtype, shape, chunkshape):
    _test_unary_func_impl(np_func, blosc_func, dtype, shape, chunkshape)


@pytest.mark.parametrize(("np_func", "blosc_func"), BINARY_FUNC_PAIRS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(("shape", "chunkshape"), SHAPES_CHUNKS)
def test_binary_funcs(np_func, blosc_func, dtype, shape, chunkshape):
    _test_binary_func_impl(np_func, blosc_func, dtype, shape, chunkshape)


@pytest.mark.parametrize(("np_func", "blosc_func"), BINARY_FUNC_PAIRS)
@pytest.mark.parametrize("dtype", STR_DTYPES)
@pytest.mark.parametrize(("shape", "chunkshape"), SHAPES_CHUNKS)
@pytest.mark.parametrize("xp", [torch])
def test_binfuncs_proxy(np_func, blosc_func, dtype, shape, chunkshape, xp):
    _test_binary_func_proxy(np_func, blosc_func, dtype, shape, chunkshape, xp)


@pytest.mark.heavy
@pytest.mark.parametrize(("np_func", "blosc_func"), BINARY_FUNC_PAIRS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(("shape", "chunkshape"), SHAPES_CHUNKS_HEAVY)
def test_binary_funcs_heavy(np_func, blosc_func, dtype, shape, chunkshape):
    _test_binary_func_impl(np_func, blosc_func, dtype, shape, chunkshape)
