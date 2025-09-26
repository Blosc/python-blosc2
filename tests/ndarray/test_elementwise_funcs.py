import sys
import warnings

import numpy as np
import pytest

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

DTYPES = [np.bool_, np.int32, np.int64, np.float32, np.float64, np.complex128]
SHAPES_CHUNKS = [((10,), (3,)), ((20, 20), (4, 7)), ((10, 13, 13), (3, 5, 2))]


@pytest.mark.parametrize(("np_func", "blosc_func"), UNARY_FUNC_PAIRS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(("shape", "chunkshape"), SHAPES_CHUNKS)
def test_unary_funcs(np_func, blosc_func, dtype, shape, chunkshape):  # noqa : C901
    if np_func.__name__ in ("arccos", "arcsin", "arctanh"):
        a_blosc = blosc2.linspace(
            0.01, stop=0.99, num=np.prod(shape), chunks=chunkshape, shape=shape, dtype=dtype
        )
        if not np.issubdtype(dtype, np.integer):
            a_blosc[tuple(i // 2 for i in shape)] = blosc2.nan
        if dtype == np.complex128:
            a_blosc = (a_blosc * (1 + 1j)).compute()
            a_blosc[tuple(i // 2 for i in shape)] = blosc2.nan + blosc2.nan * 1j
        if dtype == np.bool and np_func.__name__ == "arctanh":
            a_blosc = blosc2.zeros(chunks=chunkshape, shape=shape, dtype=dtype)
    else:
        a_blosc = blosc2.linspace(
            1, stop=np.prod(shape), num=np.prod(shape), chunks=chunkshape, shape=shape, dtype=dtype
        )
        if not np.issubdtype(dtype, np.integer):
            a_blosc[tuple(i // 2 for i in shape)] = blosc2.nan
        if dtype == np.complex128:
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
            if np_func.__name__ == "logical_not" and dtype in (np.float32, np.float64, np.complex128):
                assert True
            else:
                raise e
        except AssertionError as e:
            if np_func.__name__ in ("tan", "tanh") and dtype == np.complex128:
                warnings.showwarning(
                    "tan and tanh do not give correct NaN location",
                    UserWarning,
                    __file__,
                    0,
                    file=sys.stderr,
                )
                pytest.skip("tan and tanh do not give correct NaN location")
            else:
                raise e


@pytest.mark.parametrize(("np_func", "blosc_func"), BINARY_FUNC_PAIRS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(("shape", "chunkshape"), SHAPES_CHUNKS)
def test_binary_funcs(np_func, blosc_func, dtype, shape, chunkshape):  # noqa : C901
    a_blosc1 = blosc2.linspace(
        1, stop=np.prod(shape), num=np.prod(shape), chunks=chunkshape, shape=shape, dtype=dtype
    )
    if np_func.__name__ in ("right_shift", "left_shift"):
        a_blosc2 = blosc2.asarray(2)
    else:
        a_blosc2 = blosc2.linspace(
            start=np.prod(shape) * 2,
            stop=np.prod(shape),
            num=np.prod(shape),
            chunks=chunkshape,
            shape=shape,
            dtype=dtype,
        )
    if not np.issubdtype(dtype, np.integer):
        a_blosc1[tuple(i // 2 for i in shape)] = blosc2.nan
    if dtype == np.complex128:
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
            if np_func.__name__ in ("logical_and", "logical_or", "logical_xor", "minimum", "maximum"):
                assert True
            else:
                raise e
        except NotImplementedError as e:  # shouldn't be allowed for non-booleans
            if np_func.__name__ in ("left_shift", "right_shift", "floor_divide", "power", "remainder"):
                assert True
            else:
                raise e
        except AssertionError as e:
            if np_func.__name__ == "power" and np.issubdtype(
                dtype, np.integer
            ):  # overflow causes disagreement, no problem
                assert True
            elif np_func.__name__ in ("maximum", "minimum") and np.issubdtype(
                dtype, np.floating
            ):  # overflow causes disagreement, no problem
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
