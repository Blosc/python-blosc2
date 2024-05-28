#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

from __future__ import annotations

from collections import namedtuple
from collections.abc import Sequence

import ndindex
import numpy as np

import blosc2
from blosc2 import SpecialValue, blosc2_ext, compute_chunks_blocks

from .info import InfoReporter
from .schunk import SChunk


def make_key_hashable(key):
    if isinstance(key, slice):
        return (key.start, key.stop, key.step)
    elif isinstance(key, tuple):
        return tuple(make_key_hashable(k) for k in key)
    elif isinstance(key, list):
        return tuple(make_key_hashable(k) for k in key)
    elif isinstance(key, np.ndarray):
        return tuple(key.tolist())
    else:
        return key


def process_key(key, shape):
    key = ndindex.ndindex(key).expand(shape).raw
    mask = tuple(True if isinstance(k, int) else False for k in key)
    key = tuple(k if isinstance(k, slice) else slice(k, k + 1, None) for k in key)
    return key, mask


def get_ndarray_start_stop(ndim, key, shape):
    start = tuple(s.start if s.start is not None else 0 for s in key)
    stop = tuple(s.stop if s.stop is not None else sh for s, sh in zip(key, shape, strict=False))
    step = tuple(s.step if s.step is not None else 1 for s in key)
    return start, stop, step


def _check_allowed_dtypes(value: bool | int | float | NDArray | NDField, dtype_category: str, op: str):
    if not (isinstance(value, blosc2.LazyExpr | NDArray | NDField | np.ndarray) or np.isscalar(value)):
        raise RuntimeError(
            "Expected LazyExpr, NDArray, NDField, np.ndarray or scalar instances"
            f" and you provided a '{type(value)}' instance"
        )


class Operand:
    """Base class for all operands in expressions."""

    def __neg__(self):
        return blosc2.LazyExpr(new_op=(0, "-", self))

    def __and__(self, value: int | float | NDArray | NDField, /):
        _check_allowed_dtypes(value, "numeric", "__and__")
        return blosc2.LazyExpr(new_op=(self, "&", value))

    def __add__(self, value: int | float | NDArray | NDField, /):
        _check_allowed_dtypes(value, "numeric", "__add__")
        return blosc2.LazyExpr(new_op=(self, "+", value))

    def __iadd__(self, value: int | float | NDArray | NDField, /):
        _check_allowed_dtypes(value, "numeric", "__iadd__")
        return blosc2.LazyExpr(new_op=(self, "+", value))

    def __radd__(self, value: int | float | NDArray | NDField, /):
        _check_allowed_dtypes(value, "numeric", "__radd__")
        return blosc2.LazyExpr(new_op=(value, "+", self))

    def __sub__(self, value: int | float | NDArray | NDField, /):
        _check_allowed_dtypes(value, "numeric", "__sub__")
        return blosc2.LazyExpr(new_op=(self, "-", value))

    def __isub__(self, value: int | float | NDArray | NDField, /):
        _check_allowed_dtypes(value, "numeric", "__isub__")
        return blosc2.LazyExpr(new_op=(self, "-", value))

    def __rsub__(self, value: int | float | NDArray | NDField, /):
        _check_allowed_dtypes(value, "numeric", "__rsub__")
        return blosc2.LazyExpr(new_op=(value, "-", self))

    def __array_namespace__(self, *, api_version: str | None = None):
        if api_version is not None and not api_version.startswith("2021."):
            raise ValueError(f"Unrecognized array API version: {api_version!r}")
        return blosc2

    def __mul__(self, value: int | float | NDArray | NDField, /):
        _check_allowed_dtypes(value, "numeric", "__mul__")
        return blosc2.LazyExpr(new_op=(self, "*", value))

    def __imul__(self, value: int | float | NDArray | NDField, /):
        _check_allowed_dtypes(value, "numeric", "__imul__")
        return blosc2.LazyExpr(new_op=(self, "*", value))

    def __rmul__(self, value: int | float | NDArray | NDField, /):
        _check_allowed_dtypes(value, "numeric", "__rmul__")
        return blosc2.LazyExpr(new_op=(value, "*", self))

    def __truediv__(self, value: int | float | NDArray | NDField, /):
        _check_allowed_dtypes(value, "numeric", "__truediv__")
        return blosc2.LazyExpr(new_op=(self, "/", value))

    def __itruediv__(self, value: int | float | NDArray | NDField, /):
        _check_allowed_dtypes(value, "numeric", "__itruediv__")
        return blosc2.LazyExpr(new_op=(self, "/", value))

    def __rtruediv__(self, value: int | float | NDArray | NDField, /):
        _check_allowed_dtypes(value, "numeric", "__rtruediv__")
        return blosc2.LazyExpr(new_op=(value, "/", self))

    def __lt__(self, value: int | float | NDArray | NDField, /):
        _check_allowed_dtypes(value, "numeric", "__lt__")
        return blosc2.LazyExpr(new_op=(self, "<", value))

    def __le__(self, value: int | float | NDArray | NDField, /):
        _check_allowed_dtypes(value, "numeric", "__le__")
        return blosc2.LazyExpr(new_op=(self, "<=", value))

    def __gt__(self, value: int | float | NDArray | NDField, /):
        _check_allowed_dtypes(value, "numeric", "__gt__")
        return blosc2.LazyExpr(new_op=(self, ">", value))

    def __ge__(self, value: int | float | NDArray | NDField, /):
        _check_allowed_dtypes(value, "numeric", "__ge__")
        return blosc2.LazyExpr(new_op=(self, ">=", value))

    def __eq__(self, value: int | float | NDArray | NDField, /):
        _check_allowed_dtypes(value, "all", "__eq__")
        if blosc2._disable_overloaded_equal:
            return self is value
        return blosc2.LazyExpr(new_op=(self, "==", value))

    def __ne__(self, value: int | float | NDArray | NDField, /):
        _check_allowed_dtypes(value, "all", "__ne__")
        return blosc2.LazyExpr(new_op=(self, "!=", value))

    def __pow__(self, value: int | float | NDArray | NDField, /):
        _check_allowed_dtypes(value, "numeric", "__pow__")
        return blosc2.LazyExpr(new_op=(self, "**", value))

    def __ipow__(self, value: int | float | NDArray | NDField, /):
        _check_allowed_dtypes(value, "numeric", "__ipow__")
        return blosc2.LazyExpr(new_op=(self, "**", value))

    def __rpow__(self, value: int | float | NDArray | NDField, /):
        _check_allowed_dtypes(value, "numeric", "__rpow__")
        return blosc2.LazyExpr(new_op=(value, "**", self))

    def sum(self, axis=None, dtype=None, out=None, keepdims=False, **kwargs):
        """
        Returns the sum of array elements over a given axis.

        Parameters
        ----------
        axis: int or tuple of ints, optional
            Axis or axes along which a sum is performed. The default, axis=None,
            will sum all the elements of the input array. If axis is negative
            it counts from the last to the first axis.
        dtype: np.dtype, optional
            The type of the returned array and of the accumulator in which the
            elements are summed. The dtype of a is used by default unless a has
            an integer dtype of less precision than the default platform integer.
        keepdims: bool, optional
            If this is set to True, the axes which are reduced are left in the result
            as dimensions with size one. With this option, the result will broadcast
            correctly against the input array.
        kwargs: dict, optional
            Keyword arguments that are supported by the :func:`empty` constructor.

        Returns
        -------
        sum_along_axis: :ref:`NDArray` or :ref:`NDField`
            A NDArray with the sum of the elements along the axis.

        References
        ----------
        `np.sum <https://numpy.org/doc/stable/reference/generated/numpy.sum.html>`_
        """
        expr = blosc2.LazyExpr(new_op=(self, None, None))
        return expr.sum(axis=axis, dtype=dtype, keepdims=keepdims, **kwargs)

    def mean(self, axis=None, dtype=None, keepdims=False, **kwargs):
        """
        Returns the arithmetic mean along the specified axis.

        Parameters
        ----------
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
        mean_along_axis: :ref:`NDArray` or :ref:`NDField`
            A NDArray with the mean of the elements along the axis.

        References
        ----------
        `np.mean <https://numpy.org/doc/stable/reference/generated/numpy.mean.html>`_
        """
        expr = blosc2.LazyExpr(new_op=(self, None, None))
        return expr.mean(axis=axis, dtype=dtype, keepdims=keepdims, **kwargs)

    def std(self, axis=None, dtype=None, ddof=0, keepdims=False, **kwargs):
        """
        Returns the standard deviation along the specified axis.

        Parameters
        ----------
        axis: int or tuple of ints, optional
            Axis or axes along which the standard deviation is computed. The default is
            to compute the standard deviation of the flattened array.
        dtype: np.dtype, optional
            Type to use in computing the standard deviation. For integer inputs, the
            default is float32; for floating point inputs, it is the same as the input dtype.
        ddof: int, optional
            Means Delta Degrees of Freedom. The divisor used in calculations is N - ddof,
            where N represents the number of elements. By default ddof is zero.
        keepdims: bool, optional
            If this is set to True, the axes which are reduced are left in the result as
            dimensions with size one. With this option, the result will broadcast correctly
            against the input array.
        kwargs: dict, optional
            Keyword arguments that are supported by the :func:`empty` constructor.

        Returns
        -------
        std_along_axis: :ref:`NDArray` or :ref:`NDField`
            A NDArray with the standard deviation of the elements along the axis.

        References
        ----------
        `np.std <https://numpy.org/doc/stable/reference/generated/numpy.std.html>`_
        """
        expr = blosc2.LazyExpr(new_op=(self, None, None))
        return expr.std(axis=axis, dtype=dtype, ddof=ddof, keepdims=keepdims, **kwargs)

    def var(self, axis=None, dtype=None, ddof=0, keepdims=False, **kwargs):
        """
        Returns the variance along the specified axis.

        Parameters
        ----------
        axis: int or tuple of ints, optional
            Axis or axes along which the variance is computed. The default is to compute
            the variance of the flattened array.
        dtype: np.dtype, optional
            Type to use in computing the variance. For integer inputs, the default is
            float32; for floating point inputs, it is the same as the input dtype.
        ddof: int, optional
            Means Delta Degrees of Freedom. The divisor used in calculations is N - ddof,
            where N represents the number of elements. By default ddof is zero.
        keepdims: bool, optional
            If this is set to True, the axes which are reduced are left in the result as
            dimensions with size one. With this option, the result will broadcast correctly
            against the input array.
        kwargs: dict, optional
            Keyword arguments that are supported by the :func:`empty` constructor.

        Returns
        -------
        var_along_axis: :ref:`NDArray` or :ref:`NDField`
            A NDArray with the variance of the elements along the axis.

        References
        ----------
        `np.var <https://numpy.org/doc/stable/reference/generated/numpy.var.html>`_
        """
        expr = blosc2.LazyExpr(new_op=(self, None, None))
        return expr.var(axis=axis, dtype=dtype, ddof=ddof, keepdims=keepdims, **kwargs)

    def prod(self, axis=None, dtype=None, out=None, keepdims=False, **kwargs):
        """
        Returns the product of array elements over a given axis.

        Parameters
        ----------
        axis: int or tuple of ints, optional
            Axis or axes along which a product is performed. The default, axis=None,
            will multiply all the elements of the input array. If axis is negative
            it counts from the last to the first axis.
        dtype: np.dtype, optional
            The type of the returned array and of the accumulator in which the
            elements are multiplied. The dtype of a is used by default unless a has
            an integer dtype of less precision than the default platform integer.
        keepdims: bool, optional
            If this is set to True, the axes which are reduced are left in the result
            as dimensions with size one. With this option, the result will broadcast
            correctly against the input array.
        kwargs: dict, optional
            Keyword arguments that are supported by the :func:`empty` constructor.

        Returns
        -------
        product_along_axis: :ref:`NDArray` or :ref:`NDField`
            A NDArray with the product of the elements along the axis.

        References
        ----------
        `np.prod <https://numpy.org/doc/stable/reference/generated/numpy.prod.html>`_
        """
        expr = blosc2.LazyExpr(new_op=(self, None, None))
        return expr.prod(axis=axis, dtype=dtype, keepdims=keepdims, **kwargs)

    def min(self, axis=None, keepdims=False, **kwargs):
        """
        Returns the minimum along a given axis.

        Parameters
        ----------
        axis: int or tuple of ints, optional
            Axis or axes along which to operate. By default, flattened input is used.
        keepdims: bool, optional
            If this is set to True, the axes which are reduced are left in the result
            as dimensions with size one. With this option, the result will broadcast
            correctly against the input array.

        Returns
        -------
        min_along_axis: :ref:`NDArray` or :ref:`NDField`
            A NDArray with the minimum of the elements along the axis.

        References
        ----------
        `np.min <https://numpy.org/doc/stable/reference/generated/numpy.min.html>`_
        """
        expr = blosc2.LazyExpr(new_op=(self, None, None))
        return expr.min(axis=axis, keepdims=keepdims, **kwargs)

    def max(self, axis=None, keepdims=False, **kwargs):
        """
        Returns the maximum along a given axis.

        Parameters
        ----------
        axis: int or tuple of ints, optional
            Axis or axes along which to operate. By default, flattened input is used.
        keepdims: bool, optional
            If this is set to True, the axes which are reduced are left in the result
            as dimensions with size one. With this option, the result will broadcast
            correctly against the input array.

        Returns
        -------
        max_along_axis: :ref:`NDArray` or :ref:`NDField`
            A NDArray with the maximum of the elements along the axis.

        References
        ----------
        `np.max <https://numpy.org/doc/stable/reference/generated/numpy.max.html>`_
        """
        expr = blosc2.LazyExpr(new_op=(self, None, None))
        return expr.max(axis=axis, keepdims=keepdims, **kwargs)

    def any(self, axis=None, keepdims=False, **kwargs):
        """
        Test whether any array element along a given axis evaluates to True.

        Parameters
        ----------
        axis: int or tuple of ints, optional
            Axis or axes along which to operate. By default, flattened input is used.
        keepdims: bool, optional
            If this is set to True, the axes which are reduced are left in the result as
            dimensions with size one. With this option, the result will broadcast correctly
            against the input array.

        Returns
        -------
        any_along_axis: :ref:`NDArray` or :ref:`NDField`
            A NDArray with the result of the evaluation along the axis.

        References
        ----------
        `np.any <https://numpy.org/doc/stable/reference/generated/numpy.any.html>`_
        """
        expr = blosc2.LazyExpr(new_op=(self, None, None))
        return expr.any(axis=axis, keepdims=keepdims, **kwargs)

    def all(self, axis=None, keepdims=False, **kwargs):
        """
        Test whether all array elements along a given axis evaluate to True.

        Parameters
        ----------
        axis: int or tuple of ints, optional
            Axis or axes along which to operate. By default, flattened input is used.
        keepdims: bool, optional
            If this is set to True, the axes which are reduced are left in the result as
            dimensions with size one. With this option, the result will broadcast correctly
            against the input array.

        Returns
        -------
        all_along_axis: :ref:`NDArray` or :ref:`NDField`
            A NDArray with the result of the evaluation along the axis.

        References
        ----------
        `np.all <https://numpy.org/doc/stable/reference/generated/numpy.all.html>`_
        """
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
    def fields(self):
        """
        Dictionary with the fields of the structured array.

        Returns
        -------
        fields: dict
            A dictionary with the fields of the structured array.

        Examples
        --------
        >>> import blosc2
        >>> shape = (10,)
        >>> dtype = np.dtype([('a', np.int32), ('b', np.float64)])
        >>> # Create a structured array
        >>> sa = blosc2.zeros(shape, dtype=dtype)
        >>> # Evaluate if the fields are equal
        >>> np.all((sa.fields['a'] == sa.fields['b'])[:])
        True

        """
        return self._fields

    @property
    def keep_last_read(self):
        return self._keep_last_read

    @keep_last_read.setter
    def keep_last_read(self, value: bool):
        if not isinstance(value, bool):
            raise ValueError("keep_last_read should be a boolean")
        # Reset last read data
        self._last_read.clear()
        self._keep_last_read = value

    @property
    def info(self):
        """
        Print information about this array.
        """
        return InfoReporter(self)

    @property
    def info_items(self):
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
    def schunk(self):
        """
        The :ref:`SChunk <SChunk>` reference of the :ref:`NDArray <NDArray>`.
        All the attributes from the :ref:`SChunk <SChunk>` can be accessed through
        this instance as `self.schunk`.

        See Also
        --------
        :ref:`SChunk Attributes <SChunkAttributes>`
        """
        return self._schunk

    @property
    def blocksize(self):
        """The block size (in bytes) for this container.

        This is a shortcut to
        :attr:`SChunk.blocksize <blosc2.schunk.SChunk.blocksize>` and can be accessed
        through the :attr:`schunk` attribute as well.

        See Also
        --------
        :attr:`schunk`
        """
        return self._schunk.blocksize

    def __getitem__(self, key: int | slice | Sequence[slice]) -> np.ndarray:
        """Get a (multidimensional) slice as specified in key.

        Parameters
        ----------
        key: int, slice or sequence of slices
            The slice(s) to be retrieved. Note that step parameter is not honored yet
            in slices.

        Returns
        -------
        out: np.ndarray
            An array with the requested data.

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
        inmutable_key = None
        if self._keep_last_read:
            inmutable_key = make_key_hashable(key)
        key, mask = process_key(key, self.shape)
        start, stop, step = get_ndarray_start_stop(self.ndim, key, self.shape)
        key = (start, stop)
        shape = np.array([sp - st for st, sp in zip(start, stop, strict=False)])
        shape = tuple(shape[[not m for m in mask]])
        # arr = np.zeros(shape, dtype=self.dtype)
        # Benchmarking shows that empty is faster than zeros
        # (besides we don't need to fill padding with zeros)
        arr = np.empty(shape, dtype=self.dtype)
        nparr = super().get_slice_numpy(arr, key)
        if step != (1,) * self.ndim:
            if len(step) == 1:
                return nparr[:: step[0]]
            slice_ = tuple(slice(None, None, st) for st in step)
            return nparr[slice_]

        if self._keep_last_read:
            self._last_read.clear()
            self._last_read[inmutable_key] = nparr
        return nparr

    def __setitem__(self, key, value):
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

        if isinstance(value, int | float | bool):
            shape = [sp - st for sp, st in zip(stop, start, strict=False)]
            value = np.full(shape, value, dtype=self.dtype)
        elif isinstance(value, NDArray):
            value = value[...]
        elif isinstance(value, np.ndarray):
            if value.dtype != self.dtype:
                raise ValueError("The dtype of the value should be the same as the array")

        return super().set_slice(key, value)

    def iterchunks_info(self):
        """
        Iterate over :paramref:`self` chunks, providing info on index and special values.

        Yields
        ------
        info: namedtuple
            A namedtuple with the following fields:
            nchunk: the index of the chunk (int).
            coords: the coordinates of the chunk, in chunk units (tuple).
            cratio: the compression ratio of the chunk (float).
            special: the special value enum of the chunk; if 0, the chunk is not special (SpecialValue).
            repeated_value: the repeated value for the chunk; if not SpecialValue.VALUE, it is None.
            lazychunk: a buffer with the complete lazy chunk (bytes).
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

    def tobytes(self):
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

    def to_cframe(self):
        """Get a bytes object containing the serialized :ref:`NDArray <NDArray>` instance.

        Returns
        -------
        out: bytes
            The buffer containing the serialized :ref:`NDArray <NDArray>` instance.

        See Also
        --------
        :func:`~blosc2.ndarray_from_cframe`

        """
        return super().to_cframe()

    def copy(self, dtype=None, **kwargs):
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
        out: :ref:`NDArray <NDArray>`
            A :ref:`NDArray <NDArray>` with a copy of the data.

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
            meta_dict = {}
            for meta in self.schunk.meta.keys():
                meta_dict[meta] = self.schunk.meta[meta]
            kwargs["meta"] = meta_dict
        _check_ndarray_kwargs(**kwargs)

        return super().copy(dtype, **kwargs)

    def resize(self, newshape):
        """Change the shape of the array by growing or shrinking one or more dimensions.

        Parameters
        ----------
        newshape : tuple or list
            The new shape of the array. It should have the same dimensions
            as :paramref:`self`.

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
        >>> _ = b.resize(newshape)
        >>> b.shape
        (50, 10)
        """
        blosc2_ext.check_access_mode(self.schunk.urlpath, self.schunk.mode)
        return super().resize(newshape)

    def slice(self, key, **kwargs):
        """Get a (multidimensional) slice as a new :ref:`NDArray <NDArray>`.

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
        out: :ref:`NDArray <NDArray>`
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
        >>> c.shape
        (4, 10)
        """
        _check_ndarray_kwargs(**kwargs)
        key, mask = process_key(key, self.shape)
        start, stop, step = get_ndarray_start_stop(self.ndim, key, self.shape)
        if step != (1,) * self.ndim:
            raise ValueError("Step parameter is not supported yet")
        key = (start, stop)
        return super().get_slice(key, mask, **kwargs)

    def squeeze(self):
        """Remove the 1's in array's shape.

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


def sum(ndarr: NDArray | NDField, axis=None, dtype=None, keepdims=False, **kwargs):
    """
    Returns the sum of array elements over a given axis.

    Parameters
    ----------
    ndarr: :ref:`NDArray` or :ref:`NDField` | :ref:`LazyExpr`
        The input array or expression.
    axis: int or tuple of ints, optional
        Axis or axes along which a sum is performed. The default, axis=None,
        will sum all the elements of the input array. If axis is negative
        it counts from the last to the first axis.
    dtype: np.dtype, optional
        The type of the returned array and of the accumulator in which the
        elements are summed. The dtype of a is used by default unless a has
        an integer dtype of less precision than the default platform integer.
    keepdims: bool, optional
        If this is set to True, the axes which are reduced are left in the result
        as dimensions with size one. With this option, the result will broadcast
        correctly against the input array.
    kwargs: dict, optional
        Keyword arguments that are supported by the :func:`empty` constructor.

    Returns
    -------
    sum_along_axis: :ref:`NDArray` or :ref:`NDField`
            A NDArray with the sum of the elements along the axis.

    References
    ----------
    `np.sum <https://numpy.org/doc/stable/reference/generated/numpy.sum.html>`_
    """
    return ndarr.sum(axis=axis, dtype=dtype, keepdims=keepdims, **kwargs)


def mean(ndarr: NDArray | NDField, axis=None, dtype=None, keepdims=False, **kwargs):
    """
    Returns the arithmetic mean along the specified axis.

    Parameters
    ----------
    ndarr: :ref:`NDArray` or :ref:`NDField` | :ref:`LazyExpr`
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
    mean_along_axis: :ref:`NDArray` or :ref:`NDField`
        A NDArray with the mean of the elements along the axis.

    References
    ----------
    `np.mean <https://numpy.org/doc/stable/reference/generated/numpy.mean.html>`_
    """
    return ndarr.mean(axis=axis, dtype=dtype, keepdims=keepdims, **kwargs)


def std(ndarr: NDArray | NDField, axis=None, dtype=None, ddof=0, keepdims=False, **kwargs):
    """
    Returns the standard deviation along the specified axis.

    Parameters
    ----------
    ndarr: :ref:`NDArray` or :ref:`NDField` | :ref:`LazyExpr`
        The input array or expression.
    axis: int or tuple of ints, optional
        Axis or axes along which the standard deviation is computed. The default is
        to compute the standard deviation of the flattened array.
    dtype: np.dtype, optional
        Type to use in computing the standard deviation. For integer inputs, the
        default is float32; for floating point inputs, it is the same as the input dtype.
    ddof: int, optional
        Means Delta Degrees of Freedom. The divisor used in calculations is N - ddof,
        where N represents the number of elements. By default ddof is zero.
    keepdims: bool, optional
        If this is set to True, the axes which are reduced are left in the result as
        dimensions with size one. With this option, the result will broadcast correctly
        against the input array.
    kwargs: dict, optional
        Keyword arguments that are supported by the :func:`empty` constructor.

    Returns
    -------
    std_along_axis: :ref:`NDArray` or :ref:`NDField`
            A NDArray with the standard deviation of the elements along the axis.

    References
    ----------
    `np.std <https://numpy.org/doc/stable/reference/generated/numpy.std.html>`_
    """
    return ndarr.std(axis=axis, dtype=dtype, ddof=ddof, keepdims=keepdims, **kwargs)


def var(ndarr: NDArray | NDField, axis=None, dtype=None, ddof=0, keepdims=False, **kwargs):
    """
    Returns the variance along the specified axis.

    Parameters
    ----------
    ndarr: :ref:`NDArray` or :ref:`NDField` | :ref:`LazyExpr`
        The input array or expression.
    axis: int or tuple of ints, optional
        Axis or axes along which the variance is computed. The default is to compute
        the variance of the flattened array.
    dtype: np.dtype, optional
        Type to use in computing the variance. For integer inputs, the default is
        float32; for floating point inputs, it is the same as the input dtype.
    ddof: int, optional
        Means Delta Degrees of Freedom. The divisor used in calculations is N - ddof,
        where N represents the number of elements. By default ddof is zero.
    keepdims: bool, optional
        If this is set to True, the axes which are reduced are left in the result as
        dimensions with size one. With this option, the result will broadcast correctly
        against the input array.
    kwargs: dict, optional
        Keyword arguments that are supported by the :func:`empty` constructor.

    Returns
    -------
    var_along_axis: :ref:`NDArray` or :ref:`NDField`
            A NDArray with the variance of the elements along the axis.

    References
    ----------
    `np.var <https://numpy.org/doc/stable/reference/generated/numpy.var.html>`_
    """
    return ndarr.var(axis=axis, dtype=dtype, ddof=ddof, keepdims=keepdims, **kwargs)


def prod(ndarr: NDArray | NDField, axis=None, dtype=None, keepdims=False, **kwargs):
    """
    Returns the product of array elements over a given axis.

    Parameters
    ----------
    ndarr: :ref:`NDArray` or :ref:`NDField` | :ref:`LazyExpr`
        The input array or expression.
    axis: int or tuple of ints, optional
        Axis or axes along which a product is performed. The default, axis=None,
        will multiply all the elements of the input array. If axis is negative
        it counts from the last to the first axis.
    dtype: np.dtype, optional
        The type of the returned array and of the accumulator in which the
        elements are multiplied. The dtype of a is used by default unless a has
        an integer dtype of less precision than the default platform integer.
    keepdims: bool, optional
        If this is set to True, the axes which are reduced are left in the result
        as dimensions with size one. With this option, the result will broadcast
        correctly against the input array.
    kwargs: dict, optional
        Keyword arguments that are supported by the :func:`empty` constructor.

    Returns
    -------
    product_along_axis: :ref:`NDArray` or :ref:`NDField`
        A NDArray with the product of the elements along the axis.

    References
    ----------
    `np.prod <https://numpy.org/doc/stable/reference/generated/numpy.prod.html>`_
    """
    return ndarr.prod(axis=axis, dtype=dtype, keepdims=keepdims, **kwargs)


def min(ndarr: NDArray | NDField, axis=None, keepdims=False, **kwargs):
    """
    Returns the minimum along a given axis.

    Parameters
    ----------
    ndarr: :ref:`NDArray` or :ref:`NDField` | :ref:`LazyExpr`
        The input array or expression.
    axis: int or tuple of ints, optional
        Axis or axes along which to operate. By default, flattened input is used.
    keepdims: bool, optional
        If this is set to True, the axes which are reduced are left in the result as
        dimensions with size one. With this option, the result will broadcast correctly
        against the input array.

    Returns
    -------
    min_along_axis: :ref:`NDArray` or :ref:`NDField`
        A NDArray with the minimum of the elements along the axis.

    References
    ----------
    `np.min <https://numpy.org/doc/stable/reference/generated/numpy.min.html>`_
    """
    return ndarr.min(axis=axis, keepdims=keepdims, **kwargs)


def max(ndarr: NDArray | NDField, axis=None, keepdims=False, **kwargs):
    """
    Returns the maximum along a given axis.

    Parameters
    ----------
    ndarr: :ref:`NDArray` or :ref:`NDField` | :ref:`LazyExpr`
        The input array or expression.
    axis: int or tuple of ints, optional
        Axis or axes along which to operate. By default, flattened input is used.
    keepdims: bool, optional
        If this is set to True, the axes which are reduced are left in the result as
        dimensions with size one. With this option, the result will broadcast correctly
        against the input array.

    Returns
    -------
    max_along_axis: :ref:`NDArray` or :ref:`NDField`
        A NDArray with the maximum of the elements along the axis.

    References
    ----------
    `np.max <https://numpy.org/doc/stable/reference/generated/numpy.max.html>`_
    """
    return ndarr.max(axis=axis, keepdims=keepdims, **kwargs)


def any(ndarr: NDArray | NDField, axis=None, keepdims=False, **kwargs):
    """
    Test whether any array element along a given axis evaluates to True.

    Parameters
    ----------
    ndarr: :ref:`NDArray` or :ref:`NDField` | :ref:`LazyExpr`
        The input array or expression.
    axis: int or tuple of ints, optional
        Axis or axes along which to operate. By default, flattened input is used.
    keepdims: bool, optional
        If this is set to True, the axes which are reduced are left in the result as
        dimensions with size one. With this option, the result will broadcast correctly
        against the input array.

    Returns
    -------
    any_along_axis: :ref:`NDArray` or :ref:`NDField`
        A NDArray with the result of the evaluation along the axis.

    References
    ----------
    `np.any <https://numpy.org/doc/stable/reference/generated/numpy.any.html>`_
    """
    return ndarr.any(axis=axis, keepdims=keepdims, **kwargs)


def all(ndarr: NDArray | NDField, axis=None, keepdims=False, **kwargs):
    """
    Test whether all array elements along a given axis evaluate to True.

    Parameters
    ----------
    ndarr: :ref:`NDArray` or :ref:`NDField` | :ref:`LazyExpr`
        The input array or expression.
    axis: int or tuple of ints, optional
        Axis or axes along which to operate. By default, flattened input is used.
    keepdims: bool, optional
        If this is set to True, the axes which are reduced are left in the result as
        dimensions with size one. With this option, the result will broadcast correctly
        against the input array.

    Returns
    -------
    all_along_axis: :ref:`NDArray` or :ref:`NDField`
        A NDArray with the result of the evaluation along the axis.

    References
    ----------
    `np.all <https://numpy.org/doc/stable/reference/generated/numpy.all.html>`_
    """
    return ndarr.all(axis=axis, keepdims=keepdims, **kwargs)


def sin(ndarr: NDArray | NDField, /):
    """
    Trigonometric sine, element-wise.

    Parameters
    ----------
    ndarr: :ref:`NDArray` or :ref:`NDField`
        Angle, in radians.

    Returns
    -------
    out: :ref:`LazyExpr`
        A lazy expression that can be evaluated.

    References
    ----------
    `np.sin <https://numpy.org/doc/stable/reference/generated/numpy.sin.html>`_
    """
    return blosc2.LazyExpr(new_op=(ndarr, "sin", None))


def cos(ndarr: NDArray | NDField, /):
    """
    Trigonometric cosine, element-wise.

    Parameters
    ----------
    ndarr: :ref:`NDArray` or :ref:`NDField`
        Angle, in radians.

    Returns
    -------
    out: :ref:`LazyExpr`
        A lazy expression that can be evaluated.

    References
    ----------
    `np.cos <https://numpy.org/doc/stable/reference/generated/numpy.cos.html>`_
    """
    return blosc2.LazyExpr(new_op=(ndarr, "cos", None))


def tan(ndarr: NDArray | NDField, /):
    """
    Trigonometric tangent, element-wise.

    Parameters
    ----------
    ndarr: :ref:`NDArray` or :ref:`NDField`
            Angle, in radians.

    Returns
    -------
    out: :ref:`LazyExpr`
        A lazy expression that can be evaluated.

    References
    ----------
    `np.tan <https://numpy.org/doc/stable/reference/generated/numpy.tan.html>`_
    """
    return blosc2.LazyExpr(new_op=(ndarr, "tan", None))


def sqrt(ndarr: NDArray | NDField, /):
    """
    Return the non-negative square-root of an array, element-wise.

    Parameters
    ----------
    ndarr: :ref:`NDArray` or :ref:`NDField`
            The input array.

    Returns
    -------
    out: :ref:`LazyExpr`
        A lazy expression that can be evaluated.

    References
    ----------
    `np.sqrt <https://numpy.org/doc/stable/reference/generated/numpy.sqrt.html>`_
    """
    return blosc2.LazyExpr(new_op=(ndarr, "sqrt", None))


def sinh(ndarr: NDArray | NDField, /):
    """
    Hyperbolic sine, element-wise.

    Parameters
    ----------
    ndarr: :ref:`NDArray` or :ref:`NDField`
            The input array.

    Returns
    -------
    out: :ref:`LazyExpr`
        A lazy expression that can be evaluated.

    References
    ----------
    `np.sinh <https://numpy.org/doc/stable/reference/generated/numpy.sinh.html>`_
    """
    return blosc2.LazyExpr(new_op=(ndarr, "sinh", None))


def cosh(ndarr: NDArray | NDField, /):
    """
    Hyperbolic cosine, element-wise.

    Parameters
    ----------
    ndarr: :ref:`NDArray` or :ref:`NDField`
            The input array.

    Returns
    -------
    out: :ref:`LazyExpr`
        A lazy expression that can be evaluated.

    References
    ----------
    `np.cosh <https://numpy.org/doc/stable/reference/generated/numpy.cosh.html>`_
    """
    return blosc2.LazyExpr(new_op=(ndarr, "cosh", None))


def tanh(ndarr: NDArray | NDField, /):
    """
    Hyperbolic tangent, element-wise.

    Parameters
    ----------
    ndarr: :ref:`NDArray` or :ref:`NDField`
            The input array.

    Returns
    -------
    out: :ref:`LazyExpr`
        A lazy expression that can be evaluated.

    References
    ----------
    `np.tanh <https://numpy.org/doc/stable/reference/generated/numpy.tanh.html>`_
    """
    return blosc2.LazyExpr(new_op=(ndarr, "tanh", None))


def arcsin(ndarr: NDArray | NDField, /):
    """
    Inverse sine, element-wise.

    Parameters
    ----------
    ndarr: :ref:`NDArray` or :ref:`NDField`
            The input array.

    Returns
    -------
    out: :ref:`LazyExpr`
        A lazy expression that can be evaluated.

    References
    ----------
    `np.arcsin <https://numpy.org/doc/stable/reference/generated/numpy.arcsin.html>`_
    """
    return blosc2.LazyExpr(new_op=(ndarr, "arcsin", None))


def arccos(ndarr: NDArray | NDField, /):
    """
    Trigonometric inverse cosine, element-wise.

    Parameters
    ----------
    ndarr: :ref:`NDArray` or :ref:`NDField`
            The input array.

    Returns
    -------
    out: :ref:`LazyExpr`
        A lazy expression that can be evaluated.

    References
    ----------
    `np.arccos <https://numpy.org/doc/stable/reference/generated/numpy.arccos.html>`_
    """
    return blosc2.LazyExpr(new_op=(ndarr, "arccos", None))


def arctan(ndarr: NDArray | NDField, /):
    """
    Trigonometric inverse tangent, element-wise.

    Parameters
    ----------
    ndarr: :ref:`NDArray` or :ref:`NDField`
            The input array.

    Returns
    -------
    out: :ref:`LazyExpr`
        A lazy expression that can be evaluated.

    References
    ----------
    `np.arctan <https://numpy.org/doc/stable/reference/generated/numpy.arctan.html>`_
    """
    return blosc2.LazyExpr(new_op=(ndarr, "arctan", None))


def arctan2(ndarr1: NDArray | NDField, ndarr2: NDArray | NDField, /):
    """
    Element-wise arc tangent of ``ndarr1 / ndarr2`` choosing the quadrant correctly.

    Parameters
    ----------
    ndarr1: :ref:`NDArray` or :ref:`NDField`
            The input array.
    ndarr2: :ref:`NDArray` or :ref:`NDField`
            The input array.

    Returns
    -------
    out: :ref:`LazyExpr`
        A lazy expression that can be evaluated.

    References
    ----------
    `np.arctan2 <https://numpy.org/doc/stable/reference/generated/numpy.arctan2.html>`_
    """
    return blosc2.LazyExpr(new_op=(ndarr1, "arctan2", ndarr2))


def arcsinh(ndarr: NDArray | NDField, /):
    """
    Inverse hyperbolic sine, element-wise.

    Parameters
    ----------
    ndarr: :ref:`NDArray` or :ref:`NDField`
            The input array.

    Returns
    -------
    out: :ref:`LazyExpr`
        A lazy expression that can be evaluated.

    References
    ----------
    `np.arcsinh <https://numpy.org/doc/stable/reference/generated/numpy.arcsinh.html>`_
    """
    return blosc2.LazyExpr(new_op=(ndarr, "arcsinh", None))


def arccosh(ndarr: NDArray | NDField, /):
    """
    Inverse hyperbolic cosine, element-wise.

    Parameters
    ----------
    ndarr: :ref:`NDArray` or :ref:`NDField`
        The input array.

    Returns
    -------
    out: :ref:`LazyExpr`
        A lazy expression that can be evaluated.

    References
    ----------
    `np.arccosh <https://numpy.org/doc/stable/reference/generated/numpy.arccosh.html>`_
    """
    return blosc2.LazyExpr(new_op=(ndarr, "arccosh", None))


def arctanh(ndarr: NDArray | NDField, /):
    """
    Inverse hyperbolic tangent, element-wise.

    Parameters
    ----------
    ndarr: :ref:`NDArray` or :ref:`NDField`
        The input array.

    Returns
    -------
    out: :ref:`LazyExpr`
        A lazy expression that can be evaluated.

    References
    ----------
    `np.arctanh <https://numpy.org/doc/stable/reference/generated/numpy.arctanh.html>`_
    """
    return blosc2.LazyExpr(new_op=(ndarr, "arctanh", None))


def exp(ndarr: NDArray | NDField, /):
    """
    Calculate the exponential of all elements in the input array.

    Parameters
    ----------
    ndarr: :ref:`NDArray` or :ref:`NDField`
            The input array.

    Returns
    -------
    out: :ref:`LazyExpr`
        A lazy expression that can be evaluated.

    References
    ----------
    `np.exp <https://numpy.org/doc/stable/reference/generated/numpy.exp.html>`_
    """
    return blosc2.LazyExpr(new_op=(ndarr, "exp", None))


def expm1(ndarr: NDArray | NDField, /):
    """
    Calculate ``exp(ndarr) - 1`` for all elements in the array.

    Parameters
    ----------
    ndarr: :ref:`NDArray` or :ref:`NDField`
            The input array.

    Returns
    -------
    out: :ref:`LazyExpr`
        A lazy expression that can be evaluated.

    References
    ----------
    `np.expm1 <https://numpy.org/doc/stable/reference/generated/numpy.expm1.html>`_
    """
    return blosc2.LazyExpr(new_op=(ndarr, "expm1", None))


def log(ndarr: NDArray | NDField, /):
    """
    Natural logarithm, element-wise.

    Parameters
    ----------
    ndarr: :ref:`NDArray` or :ref:`NDField`
        The input array.

    Returns
    -------
    out: :ref:`LazyExpr`
        A lazy expression that can be evaluated.

    References
    ----------
    `np.log <https://numpy.org/doc/stable/reference/generated/numpy.log.html>`_
    """
    return blosc2.LazyExpr(new_op=(ndarr, "log", None))


def log10(ndarr: NDArray | NDField, /):
    """
    Return the base 10 logarithm of the input array, element-wise.

    Parameters
    ----------
    ndarr: :ref:`NDArray` or :ref:`NDField`
        The input array.

    Returns
    -------
    out: :ref:`LazyExpr`
        A lazy expression that can be evaluated.

    References
    ----------
    `np.log10 <https://numpy.org/doc/stable/reference/generated/numpy.log10.html>`_
    """
    return blosc2.LazyExpr(new_op=(ndarr, "log10", None))


def log1p(ndarr: NDArray | NDField, /):
    """
    Return the natural logarithm of one plus the input array, element-wise.

    Parameters
    ----------
    ndarr: :ref:`NDArray` or :ref:`NDField`
        The input array.

    Returns
    -------
    out: :ref:`LazyExpr`
        A lazy expression that can be evaluated.

    References
    ----------
    `np.log1p <https://numpy.org/doc/stable/reference/generated/numpy.log1p.html>`_
    """
    return blosc2.LazyExpr(new_op=(ndarr, "log1p", None))


def conj(ndarr: NDArray | NDField, /):
    """
    Return the complex conjugate, element-wise.

    Parameters
    ----------
    ndarr: :ref:`NDArray` or :ref:`NDField`
        The input array.

    Returns
    -------
    out: :ref:`LazyExpr`
        A lazy expression that can be evaluated.

    References
    ----------
    `np.conj <https://numpy.org/doc/stable/reference/generated/numpy.conj.html>`_
    """
    return blosc2.LazyExpr(new_op=(ndarr, "conj", None))


def real(ndarr: NDArray | NDField, /):
    """
    Return the real part of the complex array, element-wise.

    Parameters
    ----------
    ndarr: :ref:`NDArray` or :ref:`NDField`
            The input array.

    Returns
    -------
    out: :ref:`LazyExpr`
            A lazy expression that can be evaluated.

    References
    ----------
    `np.real <https://numpy.org/doc/stable/reference/generated/numpy.real.html>`_
    """
    return blosc2.LazyExpr(new_op=(ndarr, "real", None))


def imag(ndarr: NDArray | NDField, /):
    """
    Return the imaginary part of the complex array, element-wise.

    Parameters
    ----------
    ndarr: :ref:`NDArray` or :ref:`NDField`
        The input array.

    Returns
    -------
    out: :ref:`LazyExpr`
        A lazy expression that can be evaluated.

    References
    ----------
    `np.imag <https://numpy.org/doc/stable/reference/generated/numpy.imag.html>`_
    """
    return blosc2.LazyExpr(new_op=(ndarr, "imag", None))


def contains(ndarr: NDArray | NDField, value: str | bytes | NDArray, /):
    """
    Check if the array contains a string value.

    Parameters
    ----------
    ndarr: :ref:`NDArray` or :ref:`NDField`
        The input array.
    value: str or :ref:`NDArray` or :ref:`NDField`
        The value to be checked.

    Returns
    -------
    out: :ref:`LazyExpr`
        A lazy expression that can be evaluated.
    """
    if not isinstance(value, str | bytes | NDArray):
        raise ValueError("value should be a string, bytes or a NDArray!")
    return blosc2.LazyExpr(new_op=(ndarr, "contains", value))


def abs(ndarr: NDArray | NDField, /):
    """
    Calculate the absolute value element-wise.

    Parameters
    ----------
    ndarr: :ref:`NDArray` or :ref:`NDField`
        The input array.

    Returns
    -------
    out: :ref:`LazyExpr`
        A lazy expression that can be evaluated.

    References
    ----------
    `np.abs <https://numpy.org/doc/stable/reference/generated/numpy.abs.html>`_
    """
    return blosc2.LazyExpr(new_op=(ndarr, "abs", None))


def where(
    condition: blosc2.LazyExpr,
    x: NDArray | NDField | np.ndarray | int | float | complex | bool | str | bytes | None = None,
    y: NDArray | NDField | np.ndarray | int | float | complex | bool | str | bytes | None = None,
):
    """
    Return elements chosen from `x` or `y` depending on `condition`.

    Parameters
    ----------
    condition: :ref:`LazyExpr`
        Where True, yield `x`, otherwise yield `y`.
    x: :ref:`NDArray` or :ref:`NDField` or np.ndarray or scalar
        Values from which to choose when `condition` is True.
    y: :ref:`NDArray` or :ref:`NDField` or np.ndarray or scalar
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
    if isinstance(shape, int):
        shape = (shape,)
    elif not isinstance(shape, tuple | list):
        raise ValueError("shape should be a tuple or a list!")
    return shape


def empty(shape, dtype=np.uint8, **kwargs):
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
    out: :ref:`NDArray <NDArray>`
        A :ref:`NDArray <NDArray>` is returned.

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
    arr = blosc2_ext.empty(shape, chunks, blocks, dtype, **kwargs)
    return arr


def uninit(shape, dtype=np.uint8, **kwargs):
    """Create an array with uninitialized values.

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
    arr = blosc2_ext.uninit(shape, chunks, blocks, dtype, **kwargs)
    return arr


def zeros(shape, dtype=np.uint8, **kwargs):
    """Create an array, with zero being used as the default value
    for uninitialized portions of the array.

    The parameters and keyword arguments are the same as for the
    :func:`empty` constructor.

    Returns
    -------
    out: :ref:`NDArray <NDArray>`
        A :ref:`NDArray <NDArray>` is returned.

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
    arr = blosc2_ext.zeros(shape, chunks, blocks, dtype, **kwargs)
    return arr


def full(shape, fill_value, dtype=None, **kwargs):
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
    out: :ref:`NDArray <NDArray>`
        A :ref:`NDArray <NDArray>` is returned.

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
    arr = blosc2_ext.full(shape, chunks, blocks, fill_value, dtype, **kwargs)
    return arr


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
    out: :ref:`NDArray <NDArray>`
        A :ref:`NDArray <NDArray>` is returned.

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
    arr = blosc2_ext.from_buffer(buffer, shape, chunks, blocks, dtype, **kwargs)
    return arr


def copy(array, dtype=None, **kwargs):
    """
    This is equivalent to :meth:`NDArray.copy`
    """
    arr = array.copy(dtype, **kwargs)
    return arr


def asarray(array: np.ndarray, **kwargs: dict | list) -> NDArray:
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
    out: :ref:`NDArray <NDArray>`
        An new NDArray made of :paramref:`array`.

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
    chunks, blocks = compute_chunks_blocks(array.shape, chunks, blocks, array.dtype, **kwargs)
    if not isinstance(array, np.ndarray):
        array = np.array(array)
    if not array.flags.c_contiguous:
        # A contiguous array is needed
        array = np.ascontiguousarray(array)

    return blosc2_ext.asarray(array, chunks, blocks, **kwargs)


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


def get_slice_nchunks(schunk, key):
    """
    Get the unidimensional chunk indexes needed to get a
    slice of a :ref:`SChunk <SChunk>` or a :ref:`NDArray <NDArray>`.

    Parameters
    ----------
    schunk: :ref:`SChunk <SChunk>` or :ref:`NDArray <NDArray>`
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
    def __init__(self, ndarr: NDArray | NDField, field: str):
        if not isinstance(ndarr, NDArray):
            raise ValueError("ndarr should be a NDArray!")
        if not isinstance(field, str):
            raise ValueError("field should be a string!")
        if ndarr.dtype.fields is None:
            raise ValueError("NDArray does not have a structured dtype!")
        if field not in ndarr.dtype.fields:
            raise ValueError(f"Field {field} not found in the dtype of the NDArray")
        # Store immutable properties
        self.ndarr = ndarr
        self.chunks = ndarr.chunks
        self.blocks = ndarr.blocks
        self.field = field
        self.dtype = ndarr.dtype.fields[field][0]
        self.offset = ndarr.dtype.fields[field][1]
        # Activate the last read cache in parent NDArray
        self.ndarr.keep_last_read = True

    def __repr__(self):
        return f"NDField({self.ndarr}, {self.field})"

    @property
    def shape(self):
        return self.ndarr.shape

    @property
    def ext_shape(self):
        return self.ndarr.ext_shape

    @property
    def schunk(self):
        return self.ndarr.schunk

    def __getitem__(self, key):
        # Check if the key is in the last read cache
        inmutable_key = make_key_hashable(key)
        if inmutable_key in self.ndarr._last_read:
            return self.ndarr._last_read[inmutable_key][self.field]
        npbuf = self.ndarr[key]
        # Get the field from the buffer
        return npbuf[self.field]
