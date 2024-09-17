#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

from __future__ import annotations

import builtins
import math
from collections import namedtuple
from typing import TYPE_CHECKING, Iterator, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Sequence

import ndindex
import numpy as np

import blosc2
from blosc2 import SpecialValue, blosc2_ext, compute_chunks_blocks
from blosc2.info import InfoReporter
from blosc2.schunk import SChunk

def is_documented_by(original):
    def wrapper(target):
        target.__doc__ = original.__doc__
        return target

    return wrapper

def make_key_hashable(key):
    if isinstance(key, slice):
        return (key.start, key.stop, key.step)
    elif isinstance(key, tuple | list):
        return tuple(make_key_hashable(k) for k in key)
    elif isinstance(key, np.ndarray):
        return tuple(key.tolist())
    else:
        return key


def process_key(key, shape):
    key = ndindex.ndindex(key).expand(shape).raw
    mask = tuple(isinstance(k, int) for k in key)
    key = tuple(k if isinstance(k, slice) else slice(k, k + 1, None) for k in key)
    return key, mask


def get_ndarray_start_stop(ndim, key, shape):
    start = [s.start if s.start is not None else 0 for s in key]
    stop = [s.stop if s.stop is not None else sh for s, sh in zip(key, shape, strict=False)]
    # Check that start and stop values do not exceed the shape
    for i in range(ndim):
        if start[i] < 0:
            start[i] = shape[i] + start[i]
        if start[i] > shape[i]:
            start[i] = shape[i]
        if stop[i] < 0:
            stop[i] = shape[i] + stop[i]
        if stop[i] > shape[i]:
            stop[i] = shape[i]
    step = tuple(s.step if s.step is not None else 1 for s in key)
    return start, stop, step


def are_partitions_aligned(shape, chunks, blocks):
    """
    Check whether the partitions defined by chunks and blocks are aligned with shape.

    This checks that shape is aligned with the chunks and the chunks are aligned
    with the blocks.

    Returns
    -------
    bool
        True if the partitions are aligned, False otherwise.
    """
    # Check alignment
    alignment_shape_chunks = builtins.all(s % c == 0 for s, c in zip(shape, chunks, strict=True))
    if not alignment_shape_chunks:
        return False
    return builtins.all(c % b == 0 for c, b in zip(chunks, blocks, strict=True))


def are_partitions_behaved(shape, chunks, blocks):
    """
    Check whether the partitions defined by chunks and blocks are well-behaved with shape.

    This checks that partitions are C-contiguous with respect the outer container.

    Returns
    -------
    bool
        True if the partitions are well-behaved, False otherwise.
    """

    # Check C-contiguity among partitions
    def check_contiguity(shape, part):
        ndims = len(shape)
        inner_dim = ndims - 1
        for i, size, unit in zip(reversed(range(ndims)), reversed(shape), reversed(part), strict=True):
            if size > unit:
                if i < inner_dim:
                    if size % unit != 0:
                        return False
                else:
                    if size != unit:
                        return False
                inner_dim = i
        return True

    # Check C-contiguity for blocks inside chunks
    if not check_contiguity(chunks, blocks):
        return False

    # Check C-contiguity for chunks inside shape
    return check_contiguity(shape, chunks)


def get_chunks_idx(shape, chunks):
    chunks_idx = tuple(math.ceil(s / c) for s, c in zip(shape, chunks, strict=True))
    nchunks = math.prod(chunks_idx)
    return chunks_idx, nchunks


def _check_allowed_dtypes(
    value: bool | int | float | str | NDArray | NDField | blosc2.C2Array | blosc2.Proxy,
):
    if not (
        isinstance(
            value,
            blosc2.LazyExpr
            | NDArray
            | NDField
            | blosc2.C2Array
            | blosc2.Proxy
            | blosc2.ProxyNDField
            | np.ndarray,
        )
        or np.isscalar(value)
    ):
        raise RuntimeError(
            "Expected LazyExpr, NDArray, NDField, C2Array, np.ndarray or scalar instances"
            f" and you provided a '{type(value)}' instance"
        )


def sum(ndarr: NDArray | NDField | blosc2.C2Array | blosc2.LazyExpr, axis: int | tuple[int] = None,
        dtype: np.dtype = None, keepdims: bool = False, **kwargs: dict
        ) -> np.ndarray | NDArray | int | float | complex | bool:
    """
    Return the sum of array elements over a given axis.

    Parameters
    ----------
    ndarr: :ref:`NDArray` or :ref:`NDField` or :ref:`C2Array` or :ref:`LazyExpr`
        The input array or expression.
    axis: int or tuple of ints, optional
        Axis or axes along which a sum is performed. The default, axis=None,
        will sum all the elements of the input array. If axis is negative
        it counts from the last to the first axis.
    dtype: np.dtype, optional
        The type of the returned array and of the accumulator in which the
        elements are summed. The dtype of :paramref:`ndarr` is used by default unless it has
        an integer dtype of less precision than the default platform integer.
    keepdims: bool, optional
        If this is set to True, the axes which are reduced are left in the result
        as dimensions with size one. With this option, the result will broadcast
        correctly against the input array.
    kwargs: dict, optional
        Keyword arguments that are supported by the :func:`empty` constructor.

    Returns
    -------
    sum_along_axis: np.ndarray or :ref:`NDArray` or scalar
        The sum of the elements along the axis.

    References
    ----------
    `np.sum <https://numpy.org/doc/stable/reference/generated/numpy.sum.html>`_

    Examples
    --------
    >>> import numpy as np
    >>> import blosc2
    >>> # Example array
    >>> array = np.array([[1, 2, 3], [4, 5, 6]])
    >>> nd_array = blosc2.asarray(array)
    >>> # Sum all elements in the array (axis=None)
    >>> total_sum = blosc2.sum(nd_array)
    >>> print("Sum of all elements:", total_sum)
    21
    >>> # Sum along axis 0 (columns)
    >>> sum_axis_0 = blosc2.sum(nd_array, axis=0)
    >>> print("Sum along axis 0 (columns):", sum_axis_0)
    Sum along axis 0 (columns): [5 7 9]
    """
    return ndarr.sum(axis=axis, dtype=dtype, keepdims=keepdims, **kwargs)


def mean(ndarr: NDArray | NDField | blosc2.C2Array | blosc2.LazyExpr, axis: int | tuple[int] = None,
         dtype: np.dtype = None, keepdims: bool = False, **kwargs: dict
         ) -> np.ndarray | NDArray | int | float | complex | bool:
    """
    Return the arithmetic mean along the specified axis.

    Parameters
    ----------
    ndarr: :ref:`NDArray` or :ref:`NDField` or :ref:`C2Array` or :ref:`LazyExpr`
        The input array or expression.
    axis: int or tuple of ints, optional
        Axis or axes along which the means are computed. The default is to compute
        the mean of the flattened array.
    dtype: np.dtype, optional
        Type to use in computing the mean. For integer inputs, the default is
        float32; for floating point inputs, it is the same as the input dtype.
    keepdims: bool, optional
        If this is set to True, the axes which are reduced are left in the result
        as dimensions with size one. With this option, the result will broadcast
        correctly against the input array.
    kwargs: dict, optional
        Keyword arguments that are supported by the :func:`empty` constructor.

    Returns
    -------
    mean_along_axis: np.ndarray or :ref:`NDArray` or scalar
        The mean of the elements along the axis.

    References
    ----------
    `np.mean <https://numpy.org/doc/stable/reference/generated/numpy.mean.html>`_

    Examples
    --------
    >>> import numpy as np
    >>> import blosc2
    >>> # Example array
    >>> array = np.array([[1, 2, 3], [4, 5, 6]]
    >>> nd_array = blosc2.asarray(array)
    >>> # Compute the mean of all elements in the array (axis=None)
    >>> overall_mean = blosc2.mean(nd_array)
    >>> print("Mean of all elements:", overall_mean)
    Mean of all elements: 3.5
    """
    return ndarr.mean(axis=axis, dtype=dtype, keepdims=keepdims, **kwargs)


def std(ndarr: NDArray | NDField | blosc2.C2Array | blosc2.LazyExpr, axis: int | tuple[int] = None,
        dtype: np.dtype = None, ddof: int = 0, keepdims: bool = False, **kwargs: dict
        ) -> np.ndarray | NDArray | int | float | bool:
    """
    Return the standard deviation along the specified axis.

    Parameters
    ----------
    ndarr: :ref:`NDArray` or :ref:`NDField` or :ref:`C2Array` or :ref:`LazyExpr`
        The input array or expression.
    axis: int or tuple of ints, optional
        Axis or axes along which the standard deviation is computed. The default is
        to compute the standard deviation of the flattened array.
    dtype: np.dtype, optional
        Type to use in computing the standard deviation. For integer inputs, the
        default is float32; for floating point inputs, it is the same as the input dtype.
    ddof: int, optional
        Means Delta Degrees of Freedom. The divisor used in calculations is N - ddof,
        where N represents the number of elements. By default, ddof is zero.
    keepdims: bool, optional
        If this is set to True, the axes which are reduced are left in the result as
        dimensions with size one. With this option, the result will broadcast correctly
        against the input array.
    kwargs: dict, optional
        Keyword arguments that are supported by the :func:`empty` constructor.

    Returns
    -------
    std_along_axis: np.ndarray or :ref:`NDArray` or scalar
        The standard deviation of the elements along the axis.

    References
    ----------
    `np.std <https://numpy.org/doc/stable/reference/generated/numpy.std.html>`_

    Examples
    --------
    >>> import numpy as np
    >>> import blosc2
    >>> # Create an instance of NDArray with some data
    >>> array = np.array([[1, 2, 3], [4, 5, 6]])
    >>> nd_array = blosc2.asarray(array)
    >>> # Compute the standard deviation of the entire array
    >>> std_all = blosc2.std(nd_array)
    >>> print("Standard deviation of the entire array:", std_all)
    Standard deviation of the entire array: 1.707825127659933
    >>> # Compute the standard deviation along axis 0 (columns)
    >>> std_axis0 = blosc2.std(nd_array, axis=0)
    >>> print("Standard deviation along axis 0:", std_axis0)
    Standard deviation along axis 0: [1.5 1.5 1.5]
    """
    return ndarr.std(axis=axis, dtype=dtype, ddof=ddof, keepdims=keepdims, **kwargs)


def var(ndarr: NDArray | NDField | blosc2.C2Array | blosc2.LazyExpr, axis: int | tuple[int] = None,
        dtype: np.dtype = None, ddof: int = 0, keepdims: bool = False, **kwargs: dict
        ) -> np.ndarray | NDArray | int | float | bool:
    """
    Return the variance along the specified axis.

    Parameters
    ----------
    ndarr: :ref:`NDArray` or :ref:`NDField` or :ref:`C2Array` or :ref:`LazyExpr`
        The input array or expression.
    axis: int or tuple of ints, optional
        Axis or axes along which the variance is computed. The default is to compute
        the variance of the flattened array.
    dtype: np.dtype, optional
        Type to use in computing the variance. For integer inputs, the default is
        float32; for floating point inputs, it is the same as the input dtype.
    ddof: int, optional
        Means Delta Degrees of Freedom. The divisor used in calculations is N - ddof,
        where N represents the number of elements. By default, ddof is zero.
    keepdims: bool, optional
        If this is set to True, the axes which are reduced are left in the result as
        dimensions with size one. With this option, the result will broadcast correctly
        against the input array.
    kwargs: dict, optional
        Keyword arguments that are supported by the :func:`empty` constructor.

    Returns
    -------
    var_along_axis: np.ndarray or :ref:`NDArray` or scalar
        The variance of the elements along the axis.

    References
    ----------
    `np.var <https://numpy.org/doc/stable/reference/generated/numpy.var.html>`_


    Examples
    --------
    >>> import numpy as np
    >>> import blosc2
    >>> # Create an instance of NDArray with some data
    >>> array = np.array([[1, 2, 3], [4, 5, 6]])
    >>> nd_array = blosc2.asarray(array)
    >>> # Compute the variance of the entire array
    >>> var_all = blosc2.var(nd_array)
    >>> print("Variance of the entire array:", var_all)
    Variance of the entire array: 2.9166666666666665
    >>> # Compute the variance along axis 0 (columns)
    >>> var_axis0 = blosc2.var(nd_array, axis=0)
    >>> print("Variance along axis 0:", var_axis0)
    Variance along axis 0: [2.25 2.25 2.25]
    """
    return ndarr.var(axis=axis, dtype=dtype, ddof=ddof, keepdims=keepdims, **kwargs)


def prod(ndarr: NDArray | NDField | blosc2.C2Array | blosc2.LazyExpr, axis: int | tuple[int] = None,
        dtype: np.dtype = None, keepdims: bool = False, **kwargs: dict
         ) -> np.ndarray | NDArray | int | float | complex | bool:
    """
    Return the product of array elements over a given axis.

    Parameters
    ----------
    ndarr: :ref:`NDArray` or :ref:`NDField` or :ref:`C2Array` or :ref:`LazyExpr`
        The input array or expression.
    axis: int or tuple of ints, optional
        Axis or axes along which a product is performed. The default, axis=None,
        will multiply all the elements of the input array. If axis is negative
        it counts from the last to the first axis.
    dtype: np.dtype, optional
        The type of the returned array and of the accumulator in which the
        elements are multiplied. The dtype of :paramref:`ndarr` is used by default unless
        :paramref:`ndarr` has an integer dtype of less precision than the
        default platform integer.
    keepdims: bool, optional
        If this is set to True, the axes which are reduced are left in the result
        as dimensions with size one. With this option, the result will broadcast
        correctly against the input array.
    kwargs: dict, optional
        Keyword arguments that are supported by the :func:`empty` constructor.

    Returns
    -------
    product_along_axis: np.ndarray or :ref:`NDArray` or scalar
        The product of the elements along the axis.

    References
    ----------
    `np.prod <https://numpy.org/doc/stable/reference/generated/numpy.prod.html>`_

    Examples
    --------
    >>> import numpy as np
    >>> import blosc2
    >>> # Create an instance of NDArray with some data
    >>> array = np.array([[11, 22, 33], [4, 15, 36]])
    >>> nd_array = blosc2.asarray(array)
    >>> # Compute the product of all elements in the array
    >>> prod_all = blosc2.prod(nd_array)
    >>> print("Product of all elements in the array:", prod_all)
    Product of all elements in the array: 17249760
    >>> # Compute the product along axis 1 (rows)
    >>> prod_axis1 = blosc2.prod(nd_array, axis=1)
    >>> print("Product along axis 1:", prod_axis1)
    Product along axis 1: [7986 2160]
    """
    return ndarr.prod(axis=axis, dtype=dtype, keepdims=keepdims, **kwargs)


def min(ndarr: NDArray | NDField | blosc2.C2Array | blosc2.LazyExpr, axis: int | tuple[int] = None,
        keepdims: bool = False, **kwargs: dict
        ) -> np.ndarray | NDArray | int | float | complex | bool:
    """
    Return the minimum along a given axis.

    Parameters
    ----------
    ndarr: :ref:`NDArray` or :ref:`NDField` or :ref:`C2Array` or :ref:`LazyExpr`
        The input array or expression.
    axis: int or tuple of ints, optional
        Axis or axes along which to operate. By default, flattened input is used.
    keepdims: bool, optional
        If this is set to True, the axes which are reduced are left in the result as
        dimensions with size one. With this option, the result will broadcast correctly
        against the input array.
    kwargs: dict, optional
        Keyword arguments that are supported by the :func:`empty` constructor.

    Returns
    -------
    min_along_axis: np.ndarray or :ref:`NDArray` or scalar
        The minimum of the elements along the axis.

    References
    ----------
    `np.min <https://numpy.org/doc/stable/reference/generated/numpy.min.html>`_

    Examples
    --------
    >>> import numpy as np
    >>> import blosc2
    >>> array = np.array([1, 3, 7, 8, 9, 31])
    >>> nd_array = blosc2.asarray(array)
    >>> min_all = blosc2.min(nd_array)
    >>> print("Minimum of all elements in the array:", min_all)
    Minimum of all elements in the array: 1
    >>> # Compute the minimum along axis 0 with keepdims=True
    >>> min_keepdims = blosc2.min(nd_array, axis=0, keepdims=True)
    >>> print("Minimum along axis 0 with keepdims=True:", min_keepdims)
    Minimum along axis 0 with keepdims=True:  [1]
    """
    return ndarr.min(axis=axis, keepdims=keepdims, **kwargs)


def max(ndarr: NDArray | NDField | blosc2.C2Array | blosc2.LazyExpr, axis: int | tuple[int] = None,
        keepdims: bool = False, **kwargs: dict
        ) -> np.ndarray | NDArray | int | float | complex | bool:
    """
    Return the maximum along a given axis.

    Parameters
    ----------
    ndarr: :ref:`NDArray` or :ref:`NDField` or :ref:`C2Array` or :ref:`LazyExpr`
        The input array or expression.
    axis: int or tuple of ints, optional
        Axis or axes along which to operate. By default, flattened input is used.
    keepdims: bool, optional
        If this is set to True, the axes which are reduced are left in the result as
        dimensions with size one. With this option, the result will broadcast correctly
        against the input array.
    kwargs: dict, optional
        Keyword arguments that are supported by the :func:`empty` constructor.

    Returns
    -------
    max_along_axis: np.ndarray or :ref:`NDArray` or scalar
        The maximum of the elements along the axis.

    References
    ----------
    `np.max <https://numpy.org/doc/stable/reference/generated/numpy.max.html>`_

    Examples
    --------
    >>> import blosc2
    >>> import numpy as np
    >>> data = np.array([[11, 2, 36, 24, 5, 69], [73, 81, 49, 6, 73, 0]])
    >>> ndarray = blosc2.asarray(data)
    >>> print("NDArray data:", ndarray[:])
    NDArray data:  [[11  2 36 24  5 69]
                    [73 81 49  6 73  0]]
    >>> # Compute the maximum along axis 0 and 1
    >>> max_along_axis_0 = blosc2.max(ndarray, axis=0)
    >>> print("Maximum along axis 0:", max_along_axis_0)
    Maximum along axis 0: [73 81 49 24 73 69]
    >>> max_along_axis_1 = blosc2.max(ndarray, axis=1)
    >>> print("Maximum along axis 1:", max_along_axis_1)
    Maximum along axis 1: [69 81]
    >>> max_flattened = blosc2.max(ndarray)
    >>> print("Maximum of the flattened array:", max_flattened)
    Maximum of the flattened array: 81
    """
    return ndarr.max(axis=axis, keepdims=keepdims, **kwargs)


def any(ndarr: NDArray | NDField | blosc2.C2Array | blosc2.LazyExpr, axis: int | tuple[int] = None,
        keepdims: bool = False, **kwargs: dict
        ) -> np.ndarray | NDArray | bool:
    """
    Test whether any array element along a given axis evaluates to True.

    Parameters
    ----------
    ndarr: :ref:`NDArray` or :ref:`NDField` or :ref:`C2Array` or :ref:`LazyExpr`
        The input array or expression.
    axis: int or tuple of ints, optional
        Axis or axes along which to operate. By default, flattened input is used.
    keepdims: bool, optional
        If this is set to True, the axes which are reduced are left in the result as
        dimensions with size one. With this option, the result will broadcast correctly
        against the input array.
    kwargs: dict, optional
        Keyword arguments that are supported by the :func:`empty` constructor.

    Returns
    -------
    any_along_axis: np.ndarray or :ref:`NDArray` or scalar
        The result of the evaluation along the axis.

    References
    ----------
    `np.any <https://numpy.org/doc/stable/reference/generated/numpy.any.html>`_

    Examples
    --------
    >>> import blosc2
    >>> import numpy as np
    >>> data = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
    >>> # Convert the NumPy array to a Blosc2 NDArray
    >>> ndarray = blosc2.asarray(data)
    >>> print("NDArray data:", ndarray[:])
    NDArray data: [[1 0 0]
                    [0 1 0]
                    [0 0 0]]
    >>> any_along_axis_0 = blosc2.any(ndarray, axis=0)
    >>> print("Any along axis 0:", any_along_axis_0)
    Any along axis 0: [True True False]
    >>> any_flattened = blosc2.any(ndarray)
    >>> print("Any in the flattened array:", any_flattened)
    Any in the flattened array: True
    """
    return ndarr.any(axis=axis, keepdims=keepdims, **kwargs)


def all(ndarr: NDArray | NDField | blosc2.C2Array | blosc2.LazyExpr, axis: int | tuple[int] = None,
        keepdims: bool = False, **kwargs: dict
        ) -> np.ndarray | NDArray | bool:
    """
    Test whether all array elements along a given axis evaluate to True.

    Parameters
    ----------
    ndarr: :ref:`NDArray` or :ref:`NDField` or :ref:`C2Array` or :ref:`LazyExpr`
        The input array or expression.
    axis: int or tuple of ints, optional
        Axis or axes along which to operate. By default, flattened input is used.
    keepdims: bool, optional
        If this is set to True, the axes which are reduced are left in the result as
        dimensions with size one. With this option, the result will broadcast correctly
        against the input array.
    kwargs: dict, optional
        Keyword arguments that are supported by the :func:`empty` constructor.

    Returns
    -------
    all_along_axis: np.ndarray or :ref:`NDArray` or scalar
        The result of the evaluation along the axis.

    References
    ----------
    `np.all <https://numpy.org/doc/stable/reference/generated/numpy.all.html>`_

    Examples
    --------
    >>> import numpy as np
    >>> import blosc2
    >>> data = np.array([True, True, False, True, True, True])
    >>> ndarray = blosc2.asarray(data)
    >>> # Test if all elements are True along the default axis (flattened array)
    >>> result_flat = blosc2.all(ndarray)
    >>> print("All elements are True (flattened):", result_flat)
    All elements are True (flattened): False
    """
    return ndarr.all(axis=axis, keepdims=keepdims, **kwargs)


class Operand:
    """Base class for all operands in expressions."""

    def __neg__(self) -> blosc2.LazyExpr:
        return blosc2.LazyExpr(new_op=(0, "-", self))

    def __and__(self, value: int | float | NDArray | NDField | blosc2.C2Array, /) -> blosc2.LazyExpr:
        _check_allowed_dtypes(value)
        return blosc2.LazyExpr(new_op=(self, "&", value))

    def __add__(self, value: int | float | NDArray | NDField | blosc2.C2Array, /) -> blosc2.LazyExpr:
        _check_allowed_dtypes(value)
        return blosc2.LazyExpr(new_op=(self, "+", value))

    def __iadd__(self, value: int | float | NDArray | NDField | blosc2.C2Array, /) -> blosc2.LazyExpr:
        _check_allowed_dtypes(value)
        return blosc2.LazyExpr(new_op=(self, "+", value))

    def __radd__(self, value: int | float | NDArray | NDField | blosc2.C2Array, /) -> blosc2.LazyExpr:
        _check_allowed_dtypes(value)
        return blosc2.LazyExpr(new_op=(value, "+", self))

    def __sub__(self, value: int | float | NDArray | NDField | blosc2.C2Array, /) -> blosc2.LazyExpr:
        _check_allowed_dtypes(value)
        return blosc2.LazyExpr(new_op=(self, "-", value))

    def __isub__(self, value: int | float | NDArray | NDField | blosc2.C2Array, /) -> blosc2.LazyExpr:
        _check_allowed_dtypes(value)
        return blosc2.LazyExpr(new_op=(self, "-", value))

    def __rsub__(self, value: int | float | NDArray | NDField | blosc2.C2Array, /) -> blosc2.LazyExpr:
        _check_allowed_dtypes(value)
        return blosc2.LazyExpr(new_op=(value, "-", self))

    def __mul__(self, value: int | float | NDArray | NDField | blosc2.C2Array, /) -> blosc2.LazyExpr:
        _check_allowed_dtypes(value)
        return blosc2.LazyExpr(new_op=(self, "*", value))

    def __imul__(self, value: int | float | NDArray | NDField | blosc2.C2Array, /) -> blosc2.LazyExpr:
        _check_allowed_dtypes(value)
        return blosc2.LazyExpr(new_op=(self, "*", value))

    def __rmul__(self, value: int | float | NDArray | NDField | blosc2.C2Array, /) -> blosc2.LazyExpr:
        _check_allowed_dtypes(value)
        return blosc2.LazyExpr(new_op=(value, "*", self))

    def __truediv__(self, value: int | float | NDArray | NDField | blosc2.C2Array, /) -> blosc2.LazyExpr:
        _check_allowed_dtypes(value)
        return blosc2.LazyExpr(new_op=(self, "/", value))

    def __itruediv__(self, value: int | float | NDArray | NDField | blosc2.C2Array, /) -> blosc2.LazyExpr:
        _check_allowed_dtypes(value)
        return blosc2.LazyExpr(new_op=(self, "/", value))

    def __rtruediv__(self, value: int | float | NDArray | NDField | blosc2.C2Array, /) -> blosc2.LazyExpr:
        _check_allowed_dtypes(value)
        return blosc2.LazyExpr(new_op=(value, "/", self))

    def __lt__(self, value: int | float | NDArray | NDField | blosc2.C2Array, /) -> blosc2.LazyExpr:
        _check_allowed_dtypes(value)
        return blosc2.LazyExpr(new_op=(self, "<", value))

    def __le__(self, value: int | float | NDArray | NDField | blosc2.C2Array, /) -> blosc2.LazyExpr:
        _check_allowed_dtypes(value)
        return blosc2.LazyExpr(new_op=(self, "<=", value))

    def __gt__(self, value: int | float | NDArray | NDField | blosc2.C2Array, /) -> blosc2.LazyExpr:
        _check_allowed_dtypes(value)
        return blosc2.LazyExpr(new_op=(self, ">", value))

    def __ge__(self, value: int | float | NDArray | NDField | blosc2.C2Array, /) -> blosc2.LazyExpr:
        _check_allowed_dtypes(value)
        return blosc2.LazyExpr(new_op=(self, ">=", value))

    def __eq__(self, value: int | float | NDArray | NDField | blosc2.C2Array, /):
        _check_allowed_dtypes(value)
        if blosc2._disable_overloaded_equal:
            return self is value
        return blosc2.LazyExpr(new_op=(self, "==", value))

    def __ne__(self, value: int | float | NDArray | NDField | blosc2.C2Array, /) -> blosc2.LazyExpr:
        _check_allowed_dtypes(value)
        return blosc2.LazyExpr(new_op=(self, "!=", value))

    def __pow__(self, value: int | float | NDArray | NDField | blosc2.C2Array, /) -> blosc2.LazyExpr:
        _check_allowed_dtypes(value)
        return blosc2.LazyExpr(new_op=(self, "**", value))

    def __ipow__(self, value: int | float | NDArray | NDField | blosc2.C2Array, /) -> blosc2.LazyExpr:
        _check_allowed_dtypes(value)
        return blosc2.LazyExpr(new_op=(self, "**", value))

    def __rpow__(self, value: int | float | NDArray | NDField | blosc2.C2Array, /) -> blosc2.LazyExpr:
        _check_allowed_dtypes(value)
        return blosc2.LazyExpr(new_op=(value, "**", self))

    @is_documented_by(sum)
    def sum(self, axis=None, dtype=None, keepdims=False, **kwargs):
        expr = blosc2.LazyExpr(new_op=(self, None, None))
        return expr.sum(axis=axis, dtype=dtype, keepdims=keepdims, **kwargs)

    @is_documented_by(mean)
    def mean(self, axis=None, dtype=None, keepdims=False, **kwargs):
        expr = blosc2.LazyExpr(new_op=(self, None, None))
        return expr.mean(axis=axis, dtype=dtype, keepdims=keepdims, **kwargs)

    @is_documented_by(std)
    def std(self, axis=None, dtype=None, ddof=0, keepdims=False, **kwargs):
        expr = blosc2.LazyExpr(new_op=(self, None, None))
        return expr.std(axis=axis, dtype=dtype, ddof=ddof, keepdims=keepdims, **kwargs)

    @is_documented_by(var)
    def var(self, axis=None, dtype=None, ddof=0, keepdims=False, **kwargs):
        expr = blosc2.LazyExpr(new_op=(self, None, None))
        return expr.var(axis=axis, dtype=dtype, ddof=ddof, keepdims=keepdims, **kwargs)

    @is_documented_by(prod)
    def prod(self, axis=None, dtype=None, keepdims=False, **kwargs):
        expr = blosc2.LazyExpr(new_op=(self, None, None))
        return expr.prod(axis=axis, dtype=dtype, keepdims=keepdims, **kwargs)

    @is_documented_by(min)
    def min(self, axis=None, keepdims=False, **kwargs):
        expr = blosc2.LazyExpr(new_op=(self, None, None))
        return expr.min(axis=axis, keepdims=keepdims, **kwargs)

    @is_documented_by(max)
    def max(self, axis=None, keepdims=False, **kwargs):
        expr = blosc2.LazyExpr(new_op=(self, None, None))
        return expr.max(axis=axis, keepdims=keepdims, **kwargs)

    @is_documented_by(any)
    def any(self, axis=None, keepdims=False, **kwargs):
        expr = blosc2.LazyExpr(new_op=(self, None, None))
        return expr.any(axis=axis, keepdims=keepdims, **kwargs)

    @is_documented_by(all)
    def all(self, axis=None, keepdims=False, **kwargs):
        expr = blosc2.LazyExpr(new_op=(self, None, None))
        return expr.all(axis=axis, keepdims=keepdims, **kwargs)


class NDArray(blosc2_ext.NDArray, Operand):
    def __init__(self, **kwargs):
        self._schunk = SChunk(_schunk=kwargs["_schunk"], _is_view=True)  # SChunk Python instance
        self._keep_last_read = False
        # Where to store the last read data
        self._last_read = {}
        super().__init__(kwargs["_array"])
        # Accessor to fields
        self._fields = {}
        if self.dtype.fields:
            for field in self.dtype.fields:
                self._fields[field] = NDField(self, field)

    @property
    def fields(self) -> dict:
        """
        Dictionary with the fields of the structured array.

        Returns
        -------
        fields: dict
            A dictionary with the fields of the structured array.

        See Also
        --------
        :ref:`NDField`

        Examples
        --------
        >>> import blosc2
        >>> import numpy as np
        >>> shape = (10,)
        >>> dtype = np.dtype([('a', np.int32), ('b', np.float64)])
        >>> # Create a structured array
        >>> sa = blosc2.zeros(shape, dtype=dtype)
        >>> # Check that fields are equal
        >>> assert sa.fields['a'] == sa.fields['b']
        """
        return self._fields

    @property
    def keep_last_read(self) -> bool:
        """Indicates whether the last read data should be kept in memory."""
        return self._keep_last_read

    @keep_last_read.setter
    def keep_last_read(self, value: bool) -> None:
        """Set whether the last read data should be kept in memory.

        This always clear the last read data (if any).
        """
        if not isinstance(value, bool):
            raise TypeError("keep_last_read should be a boolean")
        # Reset last read data
        self._last_read.clear()
        self._keep_last_read = value

    @property
    def info(self) -> InfoReporter:
        """
        Print information about this array.

        Examples
        --------
        >>> import numpy as np
        >>> import blosc2
        >>> my_array = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        >>> array = blosc2.asarray(my_array)
        >>> print(array.info)
        type    : NDArray
        shape   : (10,)
        chunks  : (10,)
        blocks  : (10,)
        dtype   : int64
        cratio  : 0.73
        cparams : {'blocksize': 80,
        'clevel': 1,
        'codec': <Codec.ZSTD: 5>,
        'codec_meta': 0,
        'filters': [<Filter.NOFILTER: 0>,
                <Filter.NOFILTER: 0>,
                <Filter.NOFILTER: 0>,
                <Filter.NOFILTER: 0>,
                <Filter.NOFILTER: 0>,
                <Filter.SHUFFLE: 1>],
        'filters_meta': [0, 0, 0, 0, 0, 0],
        'nthreads': 4,
        'splitmode': <SplitMode.ALWAYS_SPLIT: 1>,
        'typesize': 8,
        'use_dict': 0}
        dparams : {'nthreads': 4}
        """
        return InfoReporter(self)

    @property
    def info_items(self) -> list:
        items = []
        items += [("type", f"{self.__class__.__name__}")]
        items += [("shape", self.shape)]
        items += [("chunks", self.chunks)]
        items += [("blocks", self.blocks)]
        items += [("dtype", self.dtype)]
        items += [("cratio", f"{self.schunk.cratio:.2f}")]
        items += [("cparams", self.schunk.cparams)]
        items += [("dparams", self.schunk.dparams)]
        return items

    @property
    def schunk(self) -> blosc2.SChunk:
        """
        The :ref:`SChunk <SChunk>` reference of the :ref:`NDArray`.
        All the attributes from the :ref:`SChunk <SChunk>` can be accessed through
        this instance as `self.schunk`.

        See Also
        --------
        :ref:`SChunk Attributes <SChunkAttributes>`
        """
        return self._schunk

    @property
    def shape(self) -> tuple[int]:
        """The data shape of this container.

        In case it is multiple in each dimension of :attr:`chunks`,
        it will be the same as :attr:`ext_shape`.

        See Also
        --------
        :attr:`ext_shape`
        """
        return super().shape

    @property
    def ext_shape(self) -> tuple[int]:
        """The padded data shape.

        The padded data is filled with zeros to make the real data fit into blocks and chunks, but it
        will never be retrieved as actual data (so the user can ignore this).
        In case :attr:`shape` is multiple in each dimension of :attr:`chunks` it will be the same
        as :attr:`shape`.

        See Also
        --------
        :attr:`shape`
        :attr:`chunks`
        """
        return super().ext_shape

    @property
    def chunks(self) -> tuple[int]:
        """The data chunk shape of this container.

        In case it is multiple in each dimension of :attr:`blocks`,
        it will be the same as :attr:`ext_chunks`.

        See Also
        --------
        :attr:`ext_chunks`
        """
        return super().chunks

    @property
    def ext_chunks(self) -> tuple[int]:
        """The padded chunk shape which defines the chunksize in the associated schunk.

        This will be the chunk shape used to store each chunk, filling the extra positions
        with zeros (padding). In case :attr:`chunks` is multiple in
        each dimension of :attr:`blocks` it will be the same as :attr:`chunks`.

        See Also
        --------
        :attr:`chunks`
        """
        return super().ext_chunks

    @property
    def blocks(self) -> tuple[int]:
        """The block shape of this container."""
        return super().blocks

    @property
    def ndim(self) -> int:
        """The number of dimensions of this container."""
        return super().ndim

    @property
    def size(self) -> int:
        """The size (in bytes) for this container."""
        return super().size

    @property
    def chunksize(self) -> int:
        """The data chunk size (in bytes) for this container.

        This will not be the same as
        :attr:`SChunk.chunksize <blosc2.schunk.SChunk.chunksize>`
        in case :attr:`chunks` is not multiple in
        each dimension of :attr:`blocks` (or equivalently, in case :attr:`chunks` is
        not the same as :attr:`ext_chunks`.

        See Also
        --------
        :attr:`chunks`
        :attr:`ext_chunks`
        """
        return super().chunksize

    @property
    def dtype(self) -> np.dtype:
        """
        Data-type of the array’s elements.
        """
        return super().dtype

    @property
    def blocksize(self) -> int:
        """The block size (in bytes) for this container.

        This is a shortcut to
        :attr:`SChunk.blocksize <blosc2.schunk.SChunk.blocksize>` and can be accessed
        through the :attr:`schunk` attribute as well.

        See Also
        --------
        :attr:`schunk`

        Examples
        --------
        >>> import blosc2
        >>> import numpy as np
        >>> array = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        >>> ndarray = blosc2.asarray(array)
        >>> print("Block size:", ndarray.blocksize)
        Block size: 80
        """
        return self._schunk.blocksize

    def __getitem__(self, key: int | slice | Sequence[slice] | blosc2.LazyExpr | str) -> np.ndarray | blosc2.LazyExpr:
        """Get a (multidimensional) slice as specified in key.

        Parameters
        ----------
        key: int, slice, sequence of slices, LazyExpr or str
            The slice(s) to be retrieved. Note that step parameter is not honored yet
            in slices. If a LazyExpr is provided, the expression is supposed to be of boolean
            type and the result will be the values of this array where the expression is True.
            If the key is a string, it will be converted to a LazyExpr, and will search for the
            operands in the fields of this structured array.

        Returns
        -------
        out: np.ndarray | blosc2.LazyExpr
            An array (or LazyExpr) with the requested data.

        Examples
        --------
        >>> import blosc2
        >>> shape = [25, 10]
        >>> # Create an array
        >>> a = blosc2.full(shape, 3.3333)
        >>> # Get slice as a NumPy array
        >>> a[:5, :5]
        array([[3.3333, 3.3333, 3.3333, 3.3333, 3.3333],
               [3.3333, 3.3333, 3.3333, 3.3333, 3.3333],
               [3.3333, 3.3333, 3.3333, 3.3333, 3.3333],
               [3.3333, 3.3333, 3.3333, 3.3333, 3.3333],
               [3.3333, 3.3333, 3.3333, 3.3333, 3.3333]])
        """
        # First try some fast paths for common cases
        if isinstance(key, np.integer):
            # Massage the key to a tuple and go the fast path
            key_ = (slice(key, key + 1), *(slice(None),) * (self.ndim - 1))
            start, stop, step = get_ndarray_start_stop(self.ndim, key_, self.shape)
            shape = tuple(sp - st for st, sp in zip(start, stop, strict=True))
        elif isinstance(key, tuple) and (
            builtins.sum(isinstance(k, builtins.slice) for k in key) == self.ndim
        ):
            # This can be processed in a fast way already
            start, stop, step = get_ndarray_start_stop(self.ndim, key, self.shape)
            shape = tuple(sp - st for st, sp in zip(start, stop, strict=True))
        else:
            # The more general case (this is quite slow)
            # If the key is a LazyExpr, decorate with ``where`` and return it
            if isinstance(key, blosc2.LazyExpr):
                return key.where(self)
            if isinstance(key, str):
                if self.dtype.fields is None:
                    raise ValueError("The array is not structured (its dtype does not have fields)")
                expr = blosc2.LazyExpr._new_expr(key, self.fields)
                return expr.where(self)
            key_, mask = process_key(key, self.shape)
            start, stop, step = get_ndarray_start_stop(self.ndim, key_, self.shape)
            shape = np.array([sp - st for st, sp in zip(start, stop, strict=True)])
            shape = tuple(shape[[not m for m in mask]])

        # Create the array to store the result
        arr = np.empty(shape, dtype=self.dtype)
        nparr = super().get_slice_numpy(arr, (start, stop))
        if step != (1,) * self.ndim:
            if len(step) == 1:
                return nparr[:: step[0]]
            slice_ = tuple(slice(None, None, st) for st in step)
            return nparr[slice_]

        if self._keep_last_read:
            self._last_read.clear()
            inmutable_key = make_key_hashable(key)
            self._last_read[inmutable_key] = nparr

        return nparr

    def __setitem__(self, key: int | slice | Sequence[slice], value: object):
        """Set a slice.

        Parameters
        ----------
        key: int, slice or sequence of slices
            The index for the slices to be updated. Note that step parameter
            is not honored yet.
        value: Py_Object Supporting the Buffer Protocol
            An object supporting the
            `Buffer Protocol <https://docs.python.org/3/c-api/buffer.html>`_
            used to overwrite the slice.

        Examples
        --------
        >>> import blosc2
        >>> # Create an array
        >>> a = blosc2.full([8, 8], 3.3333)
        >>> # Set a slice to 0
        >>> a[:5, :5] = 0
        >>> a[:]
        array([[0.    , 0.    , 0.    , 0.    , 0.    , 3.3333, 3.3333, 3.3333],
               [0.    , 0.    , 0.    , 0.    , 0.    , 3.3333, 3.3333, 3.3333],
               [0.    , 0.    , 0.    , 0.    , 0.    , 3.3333, 3.3333, 3.3333],
               [0.    , 0.    , 0.    , 0.    , 0.    , 3.3333, 3.3333, 3.3333],
               [0.    , 0.    , 0.    , 0.    , 0.    , 3.3333, 3.3333, 3.3333],
               [3.3333, 3.3333, 3.3333, 3.3333, 3.3333, 3.3333, 3.3333, 3.3333],
               [3.3333, 3.3333, 3.3333, 3.3333, 3.3333, 3.3333, 3.3333, 3.3333],
               [3.3333, 3.3333, 3.3333, 3.3333, 3.3333, 3.3333, 3.3333, 3.3333]])
        """
        blosc2_ext.check_access_mode(self.schunk.urlpath, self.schunk.mode)
        key, _ = process_key(key, self.shape)
        start, stop, step = get_ndarray_start_stop(self.ndim, key, self.shape)
        if step != (1,) * self.ndim:
            raise ValueError("Step parameter is not supported yet")
        key = (start, stop)

        shape = [sp - st for sp, st in zip(stop, start, strict=False)]
        if isinstance(value, int | float | bool):
            value = np.full(shape, value, dtype=self.dtype)
        elif isinstance(value, np.ndarray):
            if value.dtype != self.dtype:
                raise ValueError("The dtype of the value should be the same as the array")
            if value.shape == ():
                value = np.full(shape, value, dtype=self.dtype)
        elif isinstance(value, NDArray):
            value = value[...]

        return super().set_slice(key, value)

    def get_chunk(self, nchunk: int) -> bytes:
        """Shortcut to :meth:`SChunk.get_chunk <blosc2.schunk.SChunk.get_chunk>`. This can be accessed
        through the :attr:`schunk` attribute as well.

        Parameters
        ----------
        nchunk: int
            The index of the chunk to retrieve.

        Returns
        -------
        chunk: bytes
            The chunk data at the specified index.

        See Also
        --------
        :attr:`schunk`
            The attribute that provides access to the underlying `SChunk` object.

        Examples
        --------
        >>> import blosc2
        >>> import numpy as np
        >>> # Create an SChunk with some data
        >>> array = np.arange(10)
        >>> ndarray = blosc2.asarray(array)
        >>> chunk = ndarray.get_chunk(0)
        >>> # Decompress the chunk to convert it into a numpy array
        >>> decompressed_chunk = blosc2.decompress(chunk)
        >>> np_array_chunk = np.frombuffer(decompressed_chunk, dtype=np.int64)
        >>> # Verify the content of the chunk
        >>> if isinstance(np_array_chunk, np.ndarray):
        >>>         print(np_array_chunk)
        >>>         print(np_array_chunk.shape) # Assuming chunk is a list or numpy array
        [ 0  1  2  3  4  5  6  7  8  9]
        (10,)
        """
        return self.schunk.get_chunk(nchunk)

    def iterchunks_info(self) -> Iterator[
                NamedTuple("info",
                           nchunk = int,
                           coords = tuple,
                           cratio = float,
                           special = blosc2.SpecialValue,
                           repeated_value = bytes | None,
                           lazychunk = bytes
                           )
            ]:
        """
        Iterate over :paramref:`self` chunks, providing info on index and special values.

        Yields
        ------
        info: namedtuple
            A namedtuple with the following fields:

                nchunk: int
                    The index of the chunk.
                coords: tuple
                    The coordinates of the chunk, in chunk units.
                cratio: float
                    The compression ratio of the chunk.
                special: :class:`SpecialValue`
                    The special value enum of the chunk; if 0, the chunk is not special.
                repeated_value: :attr:`self.dtype` or None
                    The repeated value for the chunk; if not SpecialValue.VALUE, it is None.
                lazychunk: bytes
                    A buffer with the complete lazy chunk.

        Examples
        --------
        >>> import blosc2
        >>> a = blosc2.full(shape=(1000, ) * 3, fill_value=9, chunks=(500, ) * 3, dtype="f4")
        >>> for info in a.iterchunks_info():
        ...     print(info.coords)
        (0, 0, 0)
        (0, 0, 1)
        (0, 1, 0)
        (0, 1, 1)
        (1, 0, 0)
        (1, 0, 1)
        (1, 1, 0)
        (1, 1, 1)
        """
        ChunkInfoNDArray = namedtuple(
            "ChunkInfoNDArray", ["nchunk", "coords", "cratio", "special", "repeated_value", "lazychunk"]
        )
        chunks_idx = np.array(self.ext_shape) // np.array(self.chunks)
        for cinfo in self.schunk.iterchunks_info():
            nchunk, cratio, special, repeated_value, lazychunk = cinfo
            coords = tuple(np.unravel_index(cinfo.nchunk, chunks_idx))
            if cinfo.special == SpecialValue.VALUE:
                repeated_value = np.frombuffer(cinfo.repeated_value, dtype=self.dtype)[0]
            yield ChunkInfoNDArray(nchunk, coords, cratio, special, repeated_value, lazychunk)


    def tobytes(self) -> bytes:
        """Returns a buffer with the data contents.

        Returns
        -------
        out: bytes
            The buffer containing the data of the whole array.

        Examples
        --------
        >>> import blosc2
        >>> import numpy as np
        >>> dtype = np.dtype("i4")
        >>> shape = [23, 11]
        >>> a = np.arange(0, int(np.prod(shape)), dtype=dtype).reshape(shape)
        >>> # Create an array
        >>> b = blosc2.asarray(a)
        >>> b.tobytes() == bytes(a[...])
        True
        """
        return super().tobytes()

    def to_cframe(self) -> bytes:
        """Get a bytes object containing the serialized :ref:`NDArray` instance.

        Returns
        -------
        out: bytes
            The buffer containing the serialized :ref:`NDArray` instance.

        See Also
        --------
        :func:`~blosc2.ndarray_from_cframe`
            This function can be used to reconstruct a NDArray from the serialized bytes.

        Examples
        --------
        >>> import blosc2
        >>> a = blosc2.full(shape=(1000, 1000), fill_value=9, dtype='i4')
        >>> # Get the bytes object containing the serialized instance
        >>> cframe_bytes = a.to_cframe()
        >>> blosc_array = blosc2.ndarray_from_cframe(cframe_bytes)
        >>> print("Shape of the NDArray:", blosc_array.shape)
        >>> print("Data type of the NDArray:", blosc_array.dtype)
        Shape of the NDArray: (1000, 1000)
        Data type of the NDArray: int32
        """
        return super().to_cframe()

    def copy(self, dtype: np.dtype = None, **kwargs: dict) -> NDArray:
        """Create a copy of an array with same parameters.

        Parameters
        ----------
        dtype: np.dtype
            The new array dtype. Default `self.dtype`.

        Other Parameters
        ----------------
        kwargs: dict, optional
            Keyword arguments that are supported by the :func:`empty` constructor.
            If some are not specified, the default will be the ones from the original
            array (except for the urlpath).

        Returns
        -------
        out: :ref:`NDArray`
            A :ref:`NDArray` with a copy of the data.

        See Also
        --------
        :func:`copy`

        Examples
        --------
        >>> import blosc2
        >>> import numpy as np
        >>> shape = (10, 10)
        >>> blocks = (10, 10)
        >>> dtype = np.bool_
        >>> # Create a NDArray with default chunks
        >>> a = blosc2.zeros(shape, blocks=blocks, dtype=dtype)
        >>> # Get a copy with default chunks and blocks
        >>> b = a.copy(chunks=None, blocks=None)
        >>> np.array_equal(b[...], a[...])
        True
        """
        if dtype is None:
            dtype = self.dtype
        kwargs["cparams"] = kwargs.get("cparams", self.schunk.cparams).copy()
        kwargs["dparams"] = kwargs.get("dparams", self.schunk.dparams).copy()
        if "meta" not in kwargs:
            # Copy metalayers as well
            meta_dict = {meta: self.schunk.meta[meta] for meta in self.schunk.meta}
            kwargs["meta"] = meta_dict
        _check_ndarray_kwargs(**kwargs)

        return super().copy(dtype, **kwargs)

    def resize(self, newshape: tuple | list) -> None:
        """Change the shape of the array by growing or shrinking one or more dimensions.

        Parameters
        ----------
        newshape : tuple or list
            The new shape of the array. It should have the same dimensions
            as :paramref:`self`.

        Returns
        -------
        out: None

        Notes
        -----
        The array values corresponding to the added positions are not initialized.
        Thus, the user is in charge of initializing them.

        Examples
        --------
        >>> import blosc2
        >>> import numpy as np
        >>> dtype = np.dtype(np.float32)
        >>> shape = [23, 11]
        >>> a = np.linspace(1, 3, num=int(np.prod(shape))).reshape(shape)
        >>> # Create an array
        >>> b = blosc2.asarray(a)
        >>> newshape = [50, 10]
        >>> # Extend first dimension, shrink second dimension
        >>> b.resize(newshape)
        >>> b.shape
        (50, 10)
        """
        blosc2_ext.check_access_mode(self.schunk.urlpath, self.schunk.mode)
        super().resize(newshape)

    def slice(self, key: int | slice | Sequence[slice], **kwargs: dict) -> NDArray:
        """Get a (multidimensional) slice as a new :ref:`NDArray`.

        Parameters
        ----------
        key: int, slice or sequence of slices
            The index for the slices to be updated. Note that the step parameter is
            not honored yet in slices.

        Other Parameters
        ----------------
        kwargs: dict, optional
            Keyword arguments that are supported by the :func:`empty` constructor.

        Returns
        -------
        out: :ref:`NDArray`
            An array with the requested data. The dtype will be the same as `self`.

        Examples
        --------
        >>> import blosc2
        >>> import numpy as np
        >>> shape = [23, 11]
        >>> a = np.arange(np.prod(shape)).reshape(shape)
        >>> # Create an array
        >>> b = blosc2.asarray(a)
        >>> slices = (slice(3, 7), slice(1, 11))
        >>> # Get a slice as a new NDArray
        >>> c = b.slice(slices)
        >>> print(c.shape)
        (4, 10)
        >>> print(type(c))
        <class 'blosc2.ndarray.NDArray'>
        """
        _check_ndarray_kwargs(**kwargs)
        key, mask = process_key(key, self.shape)
        start, stop, step = get_ndarray_start_stop(self.ndim, key, self.shape)
        key = (start, stop)
        ndslice = super().get_slice(key, mask, **kwargs)

        # This is memory intensive, but we have not a better way to do it yet
        # TODO: perhaps add a step param in the get_slice method in the future?
        if step != (1,) * self.ndim:
            nparr = ndslice[...]
            if len(step) == 1:
                nparr = nparr[:: step[0]]
            else:
                slice_ = tuple(slice(None, None, st) for st in step)
                nparr = nparr[slice_]
            return asarray(nparr, **kwargs)

        return ndslice

    def squeeze(self) -> None:
        """Remove the 1's in array's shape.

        Returns
        -------
        out: None

        Examples
        --------
        >>> import blosc2
        >>> shape = [1, 23, 1, 11, 1]
        >>> # Create an array
        >>> a = blosc2.full(shape, 2**30)
        >>> a.shape
        (1, 23, 1, 11, 1)
        >>> # Squeeze the array
        >>> a.squeeze()
        >>> a.shape
        (23, 11)
        """
        super().squeeze()


def sin(ndarr: NDArray | NDField | blosc2.C2Array | blosc2.LazyExpr, /) -> blosc2.LazyExpr:
    """
    Trigonometric sine, element-wise.

    Parameters
    ----------
    ndarr: :ref:`NDArray` or :ref:`NDField` or :ref:`C2Array` or :ref:`LazyExpr`
        Angle, in radians.

    Returns
    -------
    out: :ref:`LazyExpr`
        A lazy expression that can be evaluated.

    References
    ----------
    `np.sin <https://numpy.org/doc/stable/reference/generated/numpy.sin.html>`_

    Examples
    --------
    >>> import numpy as np
    >>> import blosc2
    >>> angles = np.array([0, np.pi/6, np.pi/4, np.pi/2, np.pi])
    >>> nd_array = blosc2.asarray(angles)
    >>> result_ = blosc2.sin(nd_array)
    >>> result = result_[:]
    >>> print("Angles in radians:", angles)
    Angles in radians: [0.         0.52359878 0.78539816 1.57079633 3.14159265]
    >>> print("Sine of the angles:", result)
    Sine of the angles: [0.00000000e+00 5.00000000e-01 7.07106781e-01 1.00000000e+00
    1.22464680e-16]
    """
    return blosc2.LazyExpr(new_op=(ndarr, "sin", None))


def cos(ndarr: NDArray | NDField | blosc2.C2Array | blosc2.LazyExpr, /) -> blosc2.LazyExpr:
    """
    Trigonometric cosine, element-wise.

    Parameters
    ----------
    ndarr: :ref:`NDArray` or :ref:`NDField` or :ref:`C2Array` or :ref:`LazyExpr`
        Angle, in radians.

    Returns
    -------
    out: :ref:`LazyExpr`
        A lazy expression that can be evaluated.

    References
    ----------
    `np.cos <https://numpy.org/doc/stable/reference/generated/numpy.cos.html>`_

    Examples
    --------
    >>> import numpy as np
    >>> import blosc2
    >>> angles = np.array([0, np.pi/6, np.pi/4, np.pi/2, np.pi])
    >>> nd_array = blosc2.asarray(angles)
    >>> result_ = blosc2.cos(nd_array)
    >>> result = result_[:]
    >>> print("Angles in radians:", angles)
    Angles in radians: [0.         0.52359878 0.78539816 1.57079633 3.14159265]
    >>> print("Cosine of the angles:", result)
    Cosine of the angles: [ 1.00000000e+00  8.66025404e-01  7.07106781e-01  6.12323400e-17
    -1.00000000e+00]
    """
    return blosc2.LazyExpr(new_op=(ndarr, "cos", None))


def tan(ndarr: NDArray | NDField | blosc2.C2Array | blosc2.LazyExpr, /) -> blosc2.LazyExpr:
    """
    Trigonometric tangent, element-wise.

    Parameters
    ----------
    ndarr: :ref:`NDArray` or :ref:`NDField` or :ref:`C2Array` or :ref:`LazyExpr`
        Angle, in radians.

    Returns
    -------
    out: :ref:`LazyExpr`
        A lazy expression that can be evaluated.

    References
    ----------
    `np.tan <https://numpy.org/doc/stable/reference/generated/numpy.tan.html>`_

    Examples
    --------
    >>> import numpy as np
    >>> import blosc2
    >>> angles = np.array([0, np.pi/6, np.pi/4, np.pi/2, np.pi])
    >>> nd_array = blosc2.asarray(angles)
    >>> result_ = blosc2.tan(nd_array)
    >>> result = result_[:]
    >>> print("Angles in radians:", angles)
    Angles in radians: [0.         0.52359878 0.78539816 1.57079633 3.14159265]
    >>> print("Tangent of the angles:", result)
    Tangent of the angles: [ 0.00000000e+00  5.77350269e-01  1.00000000e+00  1.63312394e+16
    -1.22464680e-16]
    """
    return blosc2.LazyExpr(new_op=(ndarr, "tan", None))


def sqrt(ndarr: NDArray | NDField | blosc2.C2Array | blosc2.LazyExpr, /) -> blosc2.LazyExpr:
    """
    Return the non-negative square-root of an array, element-wise.

    Parameters
    ----------
    ndarr: :ref:`NDArray` or :ref:`NDField` or :ref:`C2Array` or :ref:`LazyExpr`
        The input array.

    Returns
    -------
    out: :ref:`LazyExpr`
        A lazy expression that can be evaluated.

    References
    ----------
    `np.sqrt <https://numpy.org/doc/stable/reference/generated/numpy.sqrt.html>`_

    Examples
    --------
    >>> import numpy as np
    >>> import blosc2
    >>> data = np.array([0, np.pi/6, np.pi/4, np.pi/2, np.pi])
    >>> nd_array = blosc2.asarray(data)
    >>> result_ = blosc2.sqrt(nd_array)
    >>> result = result_[:]
    >>> print("Original numbers:", data)
    Original numbers: [ 0  1  4  9 16 25]
    >>> print("Square roots:", result)
    Square roots: [0. 1. 2. 3. 4. 5.]
    """
    return blosc2.LazyExpr(new_op=(ndarr, "sqrt", None))


def sinh(ndarr: NDArray | NDField | blosc2.C2Array | blosc2.LazyExpr, /) -> blosc2.LazyExpr:
    """
    Hyperbolic sine, element-wise.

    Parameters
    ----------
    ndarr: :ref:`NDArray` or :ref:`NDField` or :ref:`C2Array` or :ref:`LazyExpr`
        The input array.

    Returns
    -------
    out: :ref:`LazyExpr`
        A lazy expression that can be evaluated.

    References
    ----------
    `np.sinh <https://numpy.org/doc/stable/reference/generated/numpy.sinh.html>`_

    Examples
    --------
    >>> import numpy as np
    >>> import blosc2
    >>> numbers = np.array([-2, -1, 0, 1, 2])
    >>> ndarray = blosc2.asarray(numbers)
    >>> result_lazy = blosc2.sinh(ndarray)
    >>> result = result_lazy[:]
    >>> print("Original numbers:", numbers)
    Original numbers: [-2 -1  0  1  2]
    >>> print("Hyperbolic sine:", result)
    Hyperbolic sine: [-3.62686041 -1.17520119  0.          1.17520119  3.62686041]
    """
    return blosc2.LazyExpr(new_op=(ndarr, "sinh", None))


def cosh(ndarr: NDArray | NDField | blosc2.C2Array | blosc2.LazyExpr, /) -> blosc2.LazyExpr:
    """
    Hyperbolic cosine, element-wise.

    Parameters
    ----------
    ndarr: :ref:`NDArray` or :ref:`NDField` or :ref:`C2Array` or :ref:`LazyExpr`
        The input array.

    Returns
    -------
    out: :ref:`LazyExpr`
        A lazy expression that can be evaluated.

    References
    ----------
    `np.cosh <https://numpy.org/doc/stable/reference/generated/numpy.cosh.html>`_

    Examples
    --------
    >>> import numpy as np
    >>> import blosc2
    >>> numbers = np.array([-2, -1, 0, 1, 2])
    >>> ndarray = blosc2.asarray(numbers)
    >>> result_lazy = blosc2.cosh(ndarray)
    >>> result = result_lazy[:]
    >>> print("Original numbers:", numbers)
    Original numbers: [-2 -1  0  1  2]
    >>> print("Hyperbolic cosine:", result)
    Hyperbolic cosine: [3.76219569 1.54308063 1.         1.54308063 3.76219569]
    """
    return blosc2.LazyExpr(new_op=(ndarr, "cosh", None))


def tanh(ndarr: NDArray | NDField | blosc2.C2Array | blosc2.LazyExpr, /) -> blosc2.LazyExpr:
    """
    Hyperbolic tangent, element-wise.

    Parameters
    ----------
    ndarr: :ref:`NDArray` or :ref:`NDField` or :ref:`C2Array` or :ref:`LazyExpr`
        The input array.

    Returns
    -------
    out: :ref:`LazyExpr`
        A lazy expression that can be evaluated.

    References
    ----------
    `np.tanh <https://numpy.org/doc/stable/reference/generated/numpy.tanh.html>`_

    Examples
    --------
    >>> import numpy as np
    >>> import blosc2
    >>> numbers = np.array([-2, -1, 0, 1, 2])
    >>> ndarray = blosc2.asarray(numbers)
    >>> result_lazy = blosc2.tanh(ndarray)
    >>> result = result_lazy[:]
    >>> print("Original numbers:", numbers)
    Original numbers: [-2 -1  0  1  2]
    >>> print("Hyperbolic tangent:", result)
    Hyperbolic tangent: [-0.96402758 -0.76159416  0.          0.76159416  0.96402758]
    """
    return blosc2.LazyExpr(new_op=(ndarr, "tanh", None))


def arcsin(ndarr: NDArray | NDField | blosc2.C2Array | blosc2.LazyExpr, /) -> blosc2.LazyExpr:
    """
    Inverse sine, element-wise.

    Parameters
    ----------
    ndarr: :ref:`NDArray` or :ref:`NDField` or :ref:`C2Array` or :ref:`LazyExpr`
        The input array.

    Returns
    -------
    out: :ref:`LazyExpr`
        A lazy expression that can be evaluated.

    References
    ----------
    `np.arcsin <https://numpy.org/doc/stable/reference/generated/numpy.arcsin.html>`_

    Examples
    --------
    >>> import numpy as np
    >>> import blosc2
    >>> numbers = np.array([-1, -0.5, 0, 0.5, 1])
    >>> ndarray = blosc2.asarray(numbers)
    >>> result_lazy = blosc2.arcsin(ndarray)
    >>> result = result_lazy[:]
    >>> print("Original numbers:", numbers)
    Original numbers: [-1.  -0.5  0.   0.5  1. ]
    >>> print("Arcsin:", result)
    Arcsin: [-1.57079633 -0.52359878  0.          0.52359878  1.57079633]
    """
    return blosc2.LazyExpr(new_op=(ndarr, "arcsin", None))


def arccos(ndarr: NDArray | NDField | blosc2.C2Array | blosc2.LazyExpr, /) -> blosc2.LazyExpr:
    """
    Trigonometric inverse cosine, element-wise.

    Parameters
    ----------
    ndarr: :ref:`NDArray` or :ref:`NDField` or :ref:`C2Array` or :ref:`LazyExpr`
        The input array.

    Returns
    -------
    out: :ref:`LazyExpr`
        A lazy expression that can be evaluated.

    References
    ----------
    `np.arccos <https://numpy.org/doc/stable/reference/generated/numpy.arccos.html>`_

    Examples
    --------
    >>> import numpy as np
    >>> import blosc2
    >>> numbers = np.array([-1, -0.5, 0, 0.5, 1])
    >>> ndarray = blosc2.asarray(numbers)
    >>> result_lazy = blosc2.arccos(ndarray)
    >>> result = result_lazy[:]
    >>> print("Original numbers:", numbers)
    Original numbers: [-1.  -0.5  0.   0.5  1. ]
    >>> print("Arccos:", result)
    Arccos: [3.14159265 2.0943951  1.57079633 1.04719755 0.        ]
    """
    return blosc2.LazyExpr(new_op=(ndarr, "arccos", None))


def arctan(ndarr: NDArray | NDField | blosc2.C2Array | blosc2.LazyExpr, /) -> blosc2.LazyExpr:
    """
    Trigonometric inverse tangent, element-wise.

    Parameters
    ----------
    ndarr: :ref:`NDArray` or :ref:`NDField` or :ref:`C2Array` or :ref:`LazyExpr`
        The input array.

    Returns
    -------
    out: :ref:`LazyExpr`
        A lazy expression that can be evaluated.

    References
    ----------
    `np.arctan <https://numpy.org/doc/stable/reference/generated/numpy.arctan.html>`_

    Examples
    --------
    >>> import numpy as np
    >>> import blosc2
    >>> numbers = np.array([-1, -0.5, 0, 0.5, 1])
    >>> ndarray = blosc2.asarray(numbers)
    >>> result_lazy = blosc2.arctan(ndarray)
    >>> result = result_lazy[:]
    >>> print("Original numbers:", numbers)
    Original numbers: [-1.  -0.5  0.   0.5  1. ]
    >>> print("Arctan:", result)
    Arctan: [-0.78539816 -0.46364761  0.          0.46364761  0.78539816]
    """
    return blosc2.LazyExpr(new_op=(ndarr, "arctan", None))


def arctan2(ndarr1: NDArray | NDField | blosc2.C2Array,
            ndarr2: NDArray | NDField | blosc2.C2Array, /) -> blosc2.LazyExpr:
    """
    Element-wise arc tangent of ``ndarr1 / ndarr2`` choosing the quadrant correctly.

    Parameters
    ----------
    ndarr1: :ref:`NDArray` or :ref:`NDField` or :ref:`C2Array`
        The input array.
    ndarr2: :ref:`NDArray` or :ref:`NDField` or :ref:`C2Array`
        The input array.

    Returns
    -------
    out: :ref:`LazyExpr`
        A lazy expression that can be evaluated.

    References
    ----------
    `np.arctan2 <https://numpy.org/doc/stable/reference/generated/numpy.arctan2.html>`_

    Examples
    --------
    >>> import numpy as np
    >>> import blosc2
    >>> y = np.array([0, 1, 0, -1, 1])
    >>> x = np.array([1, 1, -1, -1, 0])
    >>> ndarray_y = blosc2.asarray(y)
    >>> ndarray_x = blosc2.asarray(x)
    >>> result_lazy = blosc2.arctan2(ndarray_y, ndarray_x)
    >>> result = result_lazy[:]
    >>> print("y:", y)
    y: [ 0  1  0 -1  1]
    >>> print("x:", x)
    x: [ 1  1 -1 -1  0]
    >>> print("Arctan2(y, x):", result)
    Arctan2(y, x): [ 0.          0.78539816  3.14159265 -2.35619449  1.57079633]
    """
    return blosc2.LazyExpr(new_op=(ndarr1, "arctan2", ndarr2))


def arcsinh(ndarr: NDArray | NDField | blosc2.C2Array | blosc2.LazyExpr, /) -> blosc2.LazyExpr:
    """
    Inverse hyperbolic sine, element-wise.

    Parameters
    ----------
    ndarr: :ref:`NDArray` or :ref:`NDField` or :ref:`C2Array` or :ref:`LazyExpr`
        The input array.

    Returns
    -------
    out: :ref:`LazyExpr`
        A lazy expression that can be evaluated.

    References
    ----------
    `np.arcsinh <https://numpy.org/doc/stable/reference/generated/numpy.arcsinh.html>`_

    Examples
    --------
    >>> import numpy as np
    >>> import blosc2
    >>> values = np.array([-2, -1, 0, 1, 2])
    >>> ndarray = blosc2.asarray(values)
    >>> result_lazy = blosc2.arcsinh(ndarray)
    >>> result = result_lazy[:]
    >>> print("Original values:", values)
    Original values: [-2 -1  0  1  2]
    >>> print("Arcsinh:", result)
    Arcsinh: [-1.44363548 -0.88137359  0.          0.88137359  1.44363548]
    """
    return blosc2.LazyExpr(new_op=(ndarr, "arcsinh", None))


def arccosh(ndarr: NDArray | NDField | blosc2.C2Array | blosc2.LazyExpr, /) -> blosc2.LazyExpr:
    """
    Inverse hyperbolic cosine, element-wise.

    Parameters
    ----------
    ndarr: :ref:`NDArray` or :ref:`NDField` or :ref:`C2Array` or :ref:`LazyExpr`
        The input array.

    Returns
    -------
    out: :ref:`LazyExpr`
        A lazy expression that can be evaluated.

    References
    ----------
    `np.arccosh <https://numpy.org/doc/stable/reference/generated/numpy.arccosh.html>`_

    Examples
    --------
    >>> import numpy as np
    >>> import blosc2
    >>> values = np.array([1, 2, 3, 4, 5])
    >>> ndarray = blosc2.asarray(values)
    >>> result_lazy = blosc2.arccosh(ndarray)
    >>> result = result_lazy[:]
    >>> print("Original values:", values)
    Original values: [1 2 3 4 5]
    >>> print("Arccosh:", result)
    Arccosh: [0.         1.3169579  1.76274717 2.06343707 2.29243167]
    """
    return blosc2.LazyExpr(new_op=(ndarr, "arccosh", None))


def arctanh(ndarr: NDArray | NDField | blosc2.C2Array | blosc2.LazyExpr, /) -> blosc2.LazyExpr:
    """
    Inverse hyperbolic tangent, element-wise.

    Parameters
    ----------
    ndarr: :ref:`NDArray` or :ref:`NDField` or :ref:`C2Array` or :ref:`LazyExpr`
        The input array.

    Returns
    -------
    out: :ref:`LazyExpr`
        A lazy expression that can be evaluated.

    References
    ----------
    `np.arctanh <https://numpy.org/doc/stable/reference/generated/numpy.arctanh.html>`_

    Examples
    --------
    >>> import numpy as np
    >>> import blosc2
    >>> values = np.array([-0.9, -0.5, 0, 0.5, 0.9])
    >>> ndarray = blosc2.asarray(values)
    >>> result_lazy = blosc2.arctanh(ndarray)
    >>> result = result_lazy[:]
    >>> print("Original values:", values)
    Original values: [-0.9 -0.5  0.   0.5  0.9]
    >>> print("Arctanh:", result)
    Arctanh: [-1.47221949 -0.54930614  0.          0.54930614  1.47221949]
    """
    return blosc2.LazyExpr(new_op=(ndarr, "arctanh", None))


def exp(ndarr: NDArray | NDField | blosc2.C2Array | blosc2.LazyExpr, /) -> blosc2.LazyExpr:
    """
    Calculate the exponential of all elements in the input array.

    Parameters
    ----------
    ndarr: :ref:`NDArray` or :ref:`NDField` or :ref:`C2Array` or :ref:`LazyExpr`
        The input array.

    Returns
    -------
    out: :ref:`LazyExpr`
        A lazy expression that can be evaluated.

    References
    ----------
    `np.exp <https://numpy.org/doc/stable/reference/generated/numpy.exp.html>`_

    Examples
    --------
    >>> import numpy as np
    >>> import blosc2
    >>> values = np.array([0, 1, 2, 3, 4])
    >>> ndarray = blosc2.asarray(values)
    >>> result_lazy = blosc2.exp(ndarray)
    >>> result = result_lazy[:]
    >>> print("Original values:", values)
    Original values: [0 1 2 3 4]
    >>> print("Exponential:", result)
    Exponential: [ 1.          2.71828183  7.3890561  20.08553692 54.59815003]
    """
    return blosc2.LazyExpr(new_op=(ndarr, "exp", None))


def expm1(ndarr: NDArray | NDField | blosc2.C2Array | blosc2.LazyExpr, /) -> blosc2.LazyExpr:
    """
    Calculate ``exp(ndarr) - 1`` for all elements in the array.

    Parameters
    ----------
    ndarr: :ref:`NDArray` or :ref:`NDField` or :ref:`C2Array` or :ref:`LazyExpr`
        The input array.

    Returns
    -------
    out: :ref:`LazyExpr`
        A lazy expression that can be evaluated.

    References
    ----------
    `np.expm1 <https://numpy.org/doc/stable/reference/generated/numpy.expm1.html>`_

    Examples
    --------
    >>> import numpy as np
    >>> import blosc2
    >>> values = np.array([-1, -0.5, 0, 0.5, 1])
    >>> ndarray = blosc2.asarray(values)
    >>> result_lazy = blosc2.expm1(ndarray)
    >>> result = result_lazy[:]
    >>> print("Original values:", values)
    Original values: [-1.  -0.5  0.   0.5  1. ]
    >>> print("Expm1:", result)
    Expm1: [-0.63212056 -0.39346934  0.          0.64872127  1.71828183]
    """
    return blosc2.LazyExpr(new_op=(ndarr, "expm1", None))


def log(ndarr: NDArray | NDField | blosc2.C2Array | blosc2.LazyExpr, /) -> blosc2.LazyExpr:
    """
    Natural logarithm, element-wise.

    Parameters
    ----------
    ndarr: :ref:`NDArray` or :ref:`NDField` or :ref:`C2Array` or :ref:`LazyExpr`
        The input array.

    Returns
    -------
    out: :ref:`LazyExpr`
        A lazy expression that can be evaluated.

    References
    ----------
    `np.log <https://numpy.org/doc/stable/reference/generated/numpy.log.html>`_

    Examples
    --------
    >>> import numpy as np
    >>> import blosc2
    >>> values = np.array([1, 2, 3, 4, 5])
    >>> ndarray = blosc2.asarray(values)
    >>> result_lazy = blosc2.log(ndarray)
    >>> result = result_lazy[:]
    >>> print("Original values:", values)
    Original values: [1 2 3 4 5]
    >>> print("Logarithm (base e):", result)
    Logarithm (base e): [0.         0.69314718 1.09861229 1.38629436 1.60943791]
    """
    return blosc2.LazyExpr(new_op=(ndarr, "log", None))


def log10(ndarr: NDArray | NDField | blosc2.C2Array | blosc2.LazyExpr, /) -> blosc2.LazyExpr:
    """
    Return the base 10 logarithm of the input array, element-wise.

    Parameters
    ----------
    ndarr: :ref:`NDArray` or :ref:`NDField` or :ref:`C2Array` or :ref:`LazyExpr`
        The input array.

    Returns
    -------
    out: :ref:`LazyExpr`
        A lazy expression that can be evaluated.

    References
    ----------
    `np.log10 <https://numpy.org/doc/stable/reference/generated/numpy.log10.html>`_

    Examples
    --------
    >>> import numpy as np
    >>> import blosc2
    >>> values = np.array([1, 10, 100, 1000, 10000])
    >>> ndarray = blosc2.asarray(values)
    >>> result_lazy = blosc2.log10(ndarray)
    >>> result = result_lazy[:]
    >>> print("Original values:", values)
    Original values: [    1    10   100  1000 10000]
    >>> print("Logarithm (base 10):", result)
    Logarithm (base 10): [0. 1. 2. 3. 4.]
    """
    return blosc2.LazyExpr(new_op=(ndarr, "log10", None))


def log1p(ndarr: NDArray | NDField | blosc2.C2Array | blosc2.LazyExpr, /) -> blosc2.LazyExpr:
    """
    Return the natural logarithm of one plus the input array, element-wise.

    Parameters
    ----------
    ndarr: :ref:`NDArray` or :ref:`NDField` or :ref:`C2Array` or :ref:`LazyExpr`
        The input array.

    Returns
    -------
    out: :ref:`LazyExpr`
        A lazy expression that can be evaluated.

    References
    ----------
    `np.log1p <https://numpy.org/doc/stable/reference/generated/numpy.log1p.html>`_

    Examples
    --------
    >>> import numpy as np
    >>> import blosc2
    >>> values = np.array([-0.9, -0.5, 0, 0.5, 0.9])
    >>> ndarray = blosc2.asarray(values)
    >>> result_lazy = blosc2.log1p(ndarray)
    >>> result = result_lazy[:]
    >>> print("Original values:", values)
    Original values: [-0.9 -0.5  0.   0.5  0.9]
    >>> print("Log1p (log(1 + x)):", result)
    Log1p (log(1 + x)): [-2.30258509 -0.69314718  0.          0.40546511  0.64185389]
    """
    return blosc2.LazyExpr(new_op=(ndarr, "log1p", None))


def conj(ndarr: NDArray | NDField | blosc2.C2Array | blosc2.LazyExpr, /) -> blosc2.LazyExpr:
    """
    Return the complex conjugate, element-wise.

    Parameters
    ----------
    ndarr: :ref:`NDArray` or :ref:`NDField` or :ref:`C2Array` or :ref:`LazyExpr`
        The input array.

    Returns
    -------
    out: :ref:`LazyExpr`
        A lazy expression that can be evaluated.

    References
    ----------
    `np.conj <https://numpy.org/doc/stable/reference/generated/numpy.conj.html>`_

    Examples
    --------
    >>> import numpy as np
    >>> import blosc2
    >>> values = np.array([1+2j, 3-4j, -5+6j, 7-8j])
    >>> ndarray = blosc2.asarray(values)
    >>> result_ = blosc2.conj(ndarray)
    >>> result = result_[:]
    >>> print("Original values:", values)
    Original values: [ 1.+2.j  3.-4.j -5.+6.j  7.-8.j]
    >>> print("Complex conjugates:", result)
    Complex conjugates: [ 1.-2.j  3.+4.j -5.-6.j  7.+8.j]
    """
    return blosc2.LazyExpr(new_op=(ndarr, "conj", None))


def real(ndarr: NDArray | NDField | blosc2.C2Array | blosc2.LazyExpr, /) -> blosc2.LazyExpr:
    """
    Return the real part of the complex array, element-wise.

    Parameters
    ----------
    ndarr: :ref:`NDArray` or :ref:`NDField` or :ref:`C2Array` or :ref:`LazyExpr`
        The input array.

    Returns
    -------
    out: :ref:`LazyExpr`
        A lazy expression that can be evaluated.

    References
    ----------
    `np.real <https://numpy.org/doc/stable/reference/generated/numpy.real.html>`_

    Examples
    --------
    >>> import numpy as np
    >>> import blosc2
    >>> complex_values = np.array([1+2j, 3-4j, -5+6j, 7-8j])
    >>> ndarray = blosc2.asarray(complex_values)
    >>> result_ = blosc2.real(ndarray)
    >>> result = result_[:]
    >>> print("Original complex values:", complex_values)
    Original values: [ 1.+2.j  3.-4.j -5.+6.j  7.-8.j]
    >>> print("Real parts:", result)
    Real parts: [ 1.  3. -5.  7.]
    """
    return blosc2.LazyExpr(new_op=(ndarr, "real", None))


def imag(ndarr: NDArray | NDField | blosc2.C2Array | blosc2.LazyExpr, /) -> blosc2.LazyExpr:
    """
    Return the imaginary part of the complex array, element-wise.

    Parameters
    ----------
    ndarr: :ref:`NDArray` or :ref:`NDField` or :ref:`C2Array` or :ref:`LazyExpr`
        The input array.

    Returns
    -------
    out: :ref:`LazyExpr`
        A lazy expression that can be evaluated.

    References
    ----------
    `np.imag <https://numpy.org/doc/stable/reference/generated/numpy.imag.html>`_

    Examples
    --------
    >>> import numpy as np
    >>> import blosc2
    >>> complex_values = np.array([2+3j, -1+4j, 0-2j, 5+6j])
    >>> ndarray = blosc2.asarray(complex_values)
    >>> result_ = blosc2.imag(ndarray)
    >>> result = result_[:]
    >>> print("Original complex values:", complex_values)
    Original complex values: [ 2.+3.j -1.+4.j  0.-2.j  5.+6.j]
    >>> print("Imaginary parts:", result)
    Imaginary parts: [ 3.  4. -2.  6.]
    """
    return blosc2.LazyExpr(new_op=(ndarr, "imag", None))


def contains(
    ndarr: NDArray | NDField | blosc2.C2Array, value: str | bytes | NDArray | NDField | blosc2.C2Array, /
) -> blosc2.LazyExpr:
    """
    Check if the array contains a string value.

    Parameters
    ----------
    ndarr: :ref:`NDArray` or :ref:`NDField` or :ref:`C2Array`
        The input array.
    value: str or bytes or :ref:`NDArray` or :ref:`NDField` or :ref:`C2Array`
        The value to be checked.

    Returns
    -------
    out: :ref:`LazyExpr`
        A lazy expression that can be evaluated.

    Examples
    --------
    >>> import numpy as np
    >>> import blosc2
    >>> values = np.array([b"apple", b"xxbananaxxx", b"cherry", b"date"])
    >>> text_values = blosc2.asarray(values)
    >>> value_to_check = b"banana"
    >>> expr = blosc2.contains(text_values, value_to_check)
    >>> result = expr.eval()
    >>> print("Contains 'banana':", result[:])
    Contains 'banana': [False  True False False]
    """
    if not isinstance(value, str | bytes | NDArray):
        raise TypeError("value should be a string, bytes or a NDArray!")
    return blosc2.LazyExpr(new_op=(ndarr, "contains", value))


def abs(ndarr: NDArray | NDField | blosc2.C2Array | blosc2.LazyExpr, /) -> blosc2.LazyExpr:
    """
    Calculate the absolute value element-wise.

    Parameters
    ----------
    ndarr: :ref:`NDArray` or :ref:`NDField` or :ref:`C2Array` or :ref:`LazyExpr`
        The input array.

    Returns
    -------
    out: :ref:`LazyExpr`
        A lazy expression that can be evaluated.

    References
    ----------
    `np.abs <https://numpy.org/doc/stable/reference/generated/numpy.absolute.html#numpy.absolute>`_

    Examples
    --------
    >>> import numpy as np
    >>> import blosc2
    >>> values = np.array([-5, -3, 0, 2, 4])
    >>> ndarray = blosc2.asarray(values)
    >>> result_ = blosc2.abs(ndarray)
    >>> result = result_[:]
    >>> print("Original values:", values)
    Original values: [-5 -3  0  2  4]
    >>> print("Absolute values:", result)
    Absolute values: [5. 3. 0. 2. 4.]
    """
    return blosc2.LazyExpr(new_op=(ndarr, "abs", None))


def where(
    condition: blosc2.LazyExpr,
    x: NDArray | NDField | np.ndarray | int | float | complex | bool | str | bytes | None = None,
    y: NDArray | NDField | np.ndarray | int | float | complex | bool | str | bytes | None = None,
) -> blosc2.LazyExpr:
    """
    Return elements chosen from `x` or `y` depending on `condition`.

    Parameters
    ----------
    condition: :ref:`LazyExpr`
        Where True, yield `x`, otherwise yield `y`.
    x: :ref:`NDArray` or :ref:`NDField` or np.ndarray or scalar or bytes
        Values from which to choose when `condition` is True.
    y: :ref:`NDArray` or :ref:`NDField` or np.ndarray or scalar or bytes
        Values from which to choose when `condition` is False.

    References
    ----------
    `np.where <https://numpy.org/doc/stable/reference/generated/numpy.where.html>`_
    """
    return condition.where(x, y)


def lazywhere(value1=None, value2=None):
    """Decorator to apply a where condition to a LazyExpr."""

    def inner_decorator(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs).where(value1, value2)

        return wrapper

    return inner_decorator


def _check_shape(shape):
    if isinstance(shape, int | np.integer):
        shape = (shape,)
    elif not isinstance(shape, tuple | list):
        raise TypeError("shape should be a tuple or a list!")
    return shape


def empty(shape: int | tuple | list, dtype: np.dtype = np.uint8, **kwargs: dict) -> NDArray:
    """Create an empty array.

    Parameters
    ----------
    shape: int, tuple or list
        The shape for the final array.
    dtype: np.dtype
        The ndarray dtype in NumPy format. Default is `np.uint8`.
        This will override the `typesize`
        in the cparams in case they are passed.

    Other Parameters
    ----------------
    kwargs: dict, optional
        Keyword arguments supported:
            chunks: tuple or list
                The chunk shape. If None (default), Blosc2 will compute
                an efficient chunk shape.
            blocks: tuple or list
                The block shape. If None (default), Blosc2 will compute
                an efficient block shape. This will override the `blocksize`
                in the cparams in case they are passed.

        The other keyword arguments supported are the same as for the
        :obj:`SChunk.__init__ <blosc2.schunk.SChunk.__init__>` constructor.

    Returns
    -------
    out: :ref:`NDArray`
        A :ref:`NDArray` is returned.

    Examples
    --------
    >>> import blosc2
    >>> import numpy as np
    >>> shape = [20, 20]
    >>> dtype = np.int32
    >>> # Create empty array with default chunks and blocks
    >>> array = blosc2.empty(shape, dtype=dtype)
    >>> array.shape
    (20, 20)
    >>> array.dtype
    dtype('int32')
    """
    shape = _check_shape(shape)
    _check_ndarray_kwargs(**kwargs)
    chunks = kwargs.pop("chunks", None)
    blocks = kwargs.pop("blocks", None)
    chunks, blocks = compute_chunks_blocks(shape, chunks, blocks, dtype, **kwargs)
    return blosc2_ext.empty(shape, chunks, blocks, dtype, **kwargs)


def uninit(shape: int | tuple | list, dtype: np.dtype = np.uint8, **kwargs: dict) -> NDArray:
    """Create an array with uninitialized values.

    The parameters and keyword arguments are the same as for the
    :func:`empty` constructor.

    Returns
    -------
    out: :ref:`NDArray`
        A :ref:`NDArray` is returned.

    Examples
    --------
    >>> import blosc2
    >>> shape = [8, 8]
    >>> chunks = [6, 5]
    >>> # Create uninitialized array
    >>> array = blosc2.uninit(shape, dtype='f8', chunks=chunks)
    >>> array.shape
    (8, 8)
    >>> array.chunks
    (6, 5)
    >>> array.dtype
    dtype('float64')
    """
    shape = _check_shape(shape)
    _check_ndarray_kwargs(**kwargs)
    chunks = kwargs.pop("chunks", None)
    blocks = kwargs.pop("blocks", None)
    chunks, blocks = compute_chunks_blocks(shape, chunks, blocks, dtype, **kwargs)
    return blosc2_ext.uninit(shape, chunks, blocks, dtype, **kwargs)


def nans(shape: int | tuple | list, dtype: np.dtype = np.float64, **kwargs: dict) -> NDArray:
    """Create an array with NaNs values.

    The parameters and keyword arguments are the same as for the
    :func:`empty` constructor.

    Returns
    -------
    out: :ref:`NDArray <NDArray>`
        A :ref:`NDArray <NDArray>` is returned.

    Examples
    --------
    >>> import blosc2
    >>> shape = [8, 8]
    >>> chunks = [6, 5]
    >>> # Create an array of NaNs
    >>> array = blosc2.nans(shape, dtype='f8', chunks=chunks)
    >>> array.shape
    (8, 8)
    >>> array.chunks
    (6, 5)
    >>> array.dtype
    dtype('float64')
    """
    shape = _check_shape(shape)
    _check_ndarray_kwargs(**kwargs)
    chunks = kwargs.pop("chunks", None)
    blocks = kwargs.pop("blocks", None)
    chunks, blocks = compute_chunks_blocks(shape, chunks, blocks, dtype, **kwargs)
    return blosc2_ext.nans(shape, chunks, blocks, dtype, **kwargs)


def zeros(shape: int | tuple | list, dtype: np.dtype = np.uint8, **kwargs: dict) -> NDArray:
    """Create an array, with zero being used as the default value
    for uninitialized portions of the array.

    The parameters and keyword arguments are the same as for the
    :func:`empty` constructor.

    Returns
    -------
    out: :ref:`NDArray`
        A :ref:`NDArray` is returned.

    Examples
    --------
    >>> import blosc2
    >>> import numpy as np
    >>> shape = [8, 8]
    >>> chunks = [6, 5]
    >>> blocks = [5, 5]
    >>> dtype = np.float64
    >>> # Create zeros array
    >>> array = blosc2.zeros(shape, dtype=dtype, chunks=chunks, blocks=blocks)
    >>> array.shape
    (8, 8)
    >>> array.chunks
    (6, 5)
    >>> array.blocks
    (5, 5)
    >>> array.dtype
    dtype('float64')
    """
    shape = _check_shape(shape)
    _check_ndarray_kwargs(**kwargs)
    chunks = kwargs.pop("chunks", None)
    blocks = kwargs.pop("blocks", None)
    chunks, blocks = compute_chunks_blocks(shape, chunks, blocks, dtype, **kwargs)
    return blosc2_ext.zeros(shape, chunks, blocks, dtype, **kwargs)


def full(shape: int | tuple | list, fill_value: bytes | int | float | bool, dtype: np.dtype = None,
         **kwargs: dict) -> NDArray:
    """Create an array, with :paramref:`fill_value` being used as the default value
    for uninitialized portions of the array.

    Parameters
    ----------
    shape: int, tuple or list
        The shape for the final array.
    fill_value: bytes, int, float or bool
        Default value to use for uninitialized portions of the array.
        Its size will override the `typesize`
        in the cparams in case they are passed.
    dtype: np.dtype
        The ndarray dtype in NumPy format. By default, this will
        be taken from the :paramref:`fill_value`.
        This will override the `typesize`
        in the cparams in case they are passed.

    Other Parameters
    ----------------
    kwargs: dict, optional
        Keyword arguments that are supported by the :func:`empty` constructor.

    Returns
    -------
    out: :ref:`NDArray`
        A :ref:`NDArray` is returned.

    Examples
    --------
    >>> import blosc2
    >>> import numpy as np
    >>> shape = [25, 10]
    >>> # Create array filled with True
    >>> array = blosc2.full(shape, True)
    >>> array.shape
    (25, 10)
    >>> array.dtype
    dtype('bool')
    """
    if isinstance(fill_value, bytes):
        dtype = np.dtype(f"S{len(fill_value)}")
    if dtype is None:
        dtype = np.dtype(type(fill_value))
    shape = _check_shape(shape)
    _check_ndarray_kwargs(**kwargs)
    chunks = kwargs.pop("chunks", None)
    blocks = kwargs.pop("blocks", None)
    chunks, blocks = compute_chunks_blocks(shape, chunks, blocks, dtype, **kwargs)
    return blosc2_ext.full(shape, chunks, blocks, fill_value, dtype, **kwargs)


def frombuffer(
    buffer: bytes, shape: int | tuple | list, dtype: np.dtype = np.uint8, **kwargs: dict | list
) -> NDArray:
    """Create an array out of a buffer.

    Parameters
    ----------
    buffer: bytes
        The buffer of the data to populate the container.
    shape: int, tuple or list
        The shape for the final container.
    dtype: np.dtype
        The ndarray dtype in NumPy format. Default is `np.uint8`.
        This will override the `typesize`
        in the cparams in case they are passed.

    Other Parameters
    ----------------
    kwargs: dict, optional
        Keyword arguments that are supported by the :func:`empty` constructor.

    Returns
    -------
    out: :ref:`NDArray`
        A :ref:`NDArray` is returned.

    Examples
    --------
    >>> import blosc2
    >>> import numpy as np
    >>> shape = [25, 10]
    >>> chunks = (49, 49)
    >>> dtype = np.dtype("|S8")
    >>> typesize = dtype.itemsize
    >>> # Create a buffer
    >>> buffer = bytes(np.random.normal(0, 1, np.prod(shape)) * typesize)
    >>> # Create a NDArray from a buffer with default blocks
    >>> a = blosc2.frombuffer(buffer, shape, chunks=chunks, dtype=dtype)
    """
    shape = _check_shape(shape)
    _check_ndarray_kwargs(**kwargs)
    chunks = kwargs.pop("chunks", None)
    blocks = kwargs.pop("blocks", None)
    chunks, blocks = compute_chunks_blocks(shape, chunks, blocks, dtype, **kwargs)
    return blosc2_ext.from_buffer(buffer, shape, chunks, blocks, dtype, **kwargs)


def copy(array: NDArray, dtype: np.dtype = None, **kwargs: dict) -> NDArray:
    """
    This is equivalent to :meth:`NDArray.copy`

    Examples
    --------
    >>> import numpy as np
    >>> import blosc2
    >>> # Create an instance of MyNDArray with some data
    >>> original_array = np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]])
    >>> # Create a copy of the array without changing dtype
    >>> copied_array = blosc2.copy(original_array)
    >>> print("Copied array (default dtype):")
    >>> print(copied_array)
    Copied array (default dtype):
    [[1.1 2.2 3.3]
    [4.4 5.5 6.6]]
    """
    return array.copy(dtype, **kwargs)


def asarray(array: np.ndarray | blosc2.C2Array, **kwargs: dict | list) -> NDArray:
    """Convert the `array` to an `NDArray`.

    Parameters
    ----------
    array: array_like
        An array supporting numpy array interface.

    Other Parameters
    ----------------
    kwargs: dict, optional
        Keyword arguments that are supported by the :func:`empty` constructor.

    Returns
    -------
    out: :ref:`NDArray`
        An new NDArray made of :paramref:`array`.

    Notes
    -----
    This will create the NDArray chunk-by-chunk directly from the input array,
    without the need to create a contiguous NumPy array internally.  This can
    be used for ingesting e.g. disk or network based arrays very effectively
    and without consuming lots of memory.

    Examples
    --------
    >>> import blosc2
    >>> import numpy as np
    >>> # Create some data
    >>> shape = [25, 10]
    >>> a = np.arange(0, np.prod(shape), dtype=np.int64).reshape(shape)
    >>> # Create a NDArray from a NumPy array
    >>> nda = blosc2.asarray(a)
    """
    _check_ndarray_kwargs(**kwargs)
    chunks = kwargs.pop("chunks", None)
    blocks = kwargs.pop("blocks", None)
    # Use the chunks and blocks from the array if they are not passed
    if chunks is None and hasattr(array, "chunks"):
        chunks = array.chunks
    if blocks is None and hasattr(array, "blocks"):
        blocks = array.blocks
    chunks, blocks = compute_chunks_blocks(array.shape, chunks, blocks, array.dtype, **kwargs)

    # Fast path for small arrays. This is not too expensive in terms of memory consumption.
    shape = array.shape
    small_size = 2**24  # 16 MB
    array_nbytes = np.prod(shape) * array.dtype.itemsize
    if array_nbytes < small_size:
        if not isinstance(array, np.ndarray):
            if hasattr(array, "chunks"):
                # A getitem operation should be enough to get a numpy array
                array = array[:]
        else:
            if not array.flags.contiguous:
                array = np.ascontiguousarray(array)
        return blosc2_ext.asarray(array, chunks, blocks, **kwargs)

    # Create the empty array
    ndarr = empty(shape, array.dtype, chunks=chunks, blocks=blocks, **kwargs)
    behaved = are_partitions_behaved(shape, chunks, blocks)

    # Get the coordinates of the chunks
    chunks_idx, nchunks = get_chunks_idx(shape, chunks)

    # Iterate over the chunks and update the empty array
    for nchunk in range(nchunks):
        # Compute current slice coordinates
        coords = tuple(np.unravel_index(nchunk, chunks_idx))
        slice_ = tuple(
            slice(c * s, builtins.min((c + 1) * s, shape[i]))
            for i, (c, s) in enumerate(zip(coords, chunks, strict=True))
        )
        # Ensure the array slice is contiguous
        array_slice = np.ascontiguousarray(array[slice_])
        if behaved:
            # The whole chunk is to be updated, so this fastpath is safe
            ndarr.schunk.update_data(nchunk, array_slice, copy=False)
        else:
            ndarr[slice_] = array_slice

    return ndarr


def _check_ndarray_kwargs(**kwargs):
    supported_keys = [
        "chunks",
        "blocks",
        "cparams",
        "dparams",
        "meta",
        "urlpath",
        "contiguous",
        "mode",
        "mmap_mode",
        "initial_mapping_size",
    ]
    for key in kwargs:
        if key not in supported_keys:
            raise KeyError(
                f"Only {supported_keys} are supported as keyword arguments, and you passed '{key}'"
            )
    if "cparams" in kwargs and "chunks" in kwargs["cparams"]:
        raise ValueError("You cannot pass chunks in cparams, use `chunks` argument instead")
    if "cparams" in kwargs and "blocks" in kwargs["cparams"]:
        raise ValueError("You cannot pass chunks in cparams, use `blocks` argument instead")


def get_slice_nchunks(schunk: blosc2.SChunk,
                      key: tuple[(int, int)] | int | slice | Sequence[slice]
                      ) -> np.ndarray:
    """
    Get the unidimensional chunk indexes needed to get a
    slice of a :ref:`SChunk <SChunk>` or a :ref:`NDArray`.

    Parameters
    ----------
    schunk: :ref:`SChunk <SChunk>` or :ref:`NDArray`
        The super-chunk or ndarray container.
    key: tuple(int, int), int, slice or sequence of slices
        If it is a super-chunk, a tuple with the start and stop of the slice, an integer,
        or a single slice.
        If it is a ndarray, sequences of slices (one per dim) are accepted too.

    Returns
    -------
    out: np.ndarray
        An array with the unidimensional chunk indexes.
    """
    if isinstance(schunk, NDArray):
        array = schunk
        key, _ = process_key(key, array.shape)
        start, stop, step = get_ndarray_start_stop(array.ndim, key, array.shape)
        if step != (1,) * array.ndim:
            raise IndexError("Step parameter is not supported yet")
        key = (start, stop)
        return blosc2_ext.array_get_slice_nchunks(array, key)
    else:
        if isinstance(key, int):
            key = (key, key + 1)
        elif isinstance(key, slice):
            if key.step not in (1, None):
                raise IndexError("Only step=1 is supported")
            key = (key.start, key.stop)
        return blosc2_ext.schunk_get_slice_nchunks(schunk, key)


# Class for dealing with fields in an NDArray
# This will allow to access fields by name in the dtype of the NDArray
class NDField(Operand):
    def __init__(self, ndarr: NDArray, field: str):
        """
        Create a new NDField.

        Parameters
        ----------
        ndarr: :ref:`NDArray`
            The NDArray to which assign the field.
        field: str
            The field's name.

        Returns
        -------
        out: :ref:`NDField`
            The corresponding :ref:`NDField`.
        """
        if not isinstance(ndarr, NDArray):
            raise TypeError("ndarr should be a NDArray!")
        if not isinstance(field, str):
            raise TypeError("field should be a string!")
        if ndarr.dtype.fields is None:
            raise TypeError("NDArray does not have a structured dtype!")
        if field not in ndarr.dtype.fields:
            raise TypeError(f"Field {field} not found in the dtype of the NDArray")
        # Store immutable properties
        self.ndarr = ndarr
        self.chunks = ndarr.chunks
        self.blocks = ndarr.blocks
        self.field = field
        self.dtype = ndarr.dtype.fields[field][0]
        self.offset = ndarr.dtype.fields[field][1]

    def __repr__(self):
        """
        Get a string as a representation.

        Returns
        -------
        out: str
        """
        return f"NDField({self.ndarr}, {self.field})"

    @property
    def shape(self) -> tuple[int]:
        """The shape of the associated :ref:`NDArray`."""
        return self.ndarr.shape

    @property
    def schunk(self) -> blosc2.SChunk:
        """The associated :ref:`SChunk <SChunk>`."""
        return self.ndarr.schunk

    def __getitem__(self, key: int | slice | Sequence[slice]) -> np.ndarray:
        """
        Get a slice of :paramref:`self`.

        Parameters
        ----------
        key: int or slice or Sequence[slice]
            The slice to be retrieved.

        Returns
        -------
        out: NumPy.ndarray
            A NumPy array with the data slice.

        """
        # If key is a LazyExpr, decorate it with ``where`` and return it
        if isinstance(key, blosc2.LazyExpr):
            return key.where(self)

        if isinstance(key, str):
            raise TypeError("This array is a NDField; use a structured NDArray for bool expressions")

        # Check if the key is in the last read cache
        inmutable_key = make_key_hashable(key)
        if inmutable_key in self.ndarr._last_read:
            return self.ndarr._last_read[inmutable_key][self.field]

        # Do the actual read in the parent NDArray
        nparr = self.ndarr[key]
        # And return the field
        return nparr[self.field]
