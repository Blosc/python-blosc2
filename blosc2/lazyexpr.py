#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################
import copy
import math
import pathlib
from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import Enum
from pathlib import Path

import ndindex
import numexpr as ne
import numpy as np

import blosc2
from blosc2.info import InfoReporter


class ReduceOp(Enum):
    """
    Available reduce operations.
    """

    SUM = np.add
    PROD = np.multiply
    MEAN = np.mean
    # Computing a median from partial results is not straightforward because the median
    # is a positional statistic, which means it depends on the relative ordering of all
    # the data points. Unlike statistics such as the sum or mean, you can't compute a median
    # from partial results without knowing the entire dataset, and this is way too expensive
    # for arrays that cannot typically fit in-memory (e.g. disk-based NDArray).
    # MEDIAN = np.median
    MAX = np.maximum
    MIN = np.minimum
    ANY = np.any
    ALL = np.all


class LazyArrayEnum(Enum):
    """
    Available LazyArrays.
    """

    Expr = 0
    UDF = 1


class LazyArray(ABC):
    @abstractmethod
    def eval(self, item, **kwargs):
        """
        Get a :ref:`NDArray <NDArray>` containing the evaluation of the :ref:`LazyUDF <LazyUDF>`
        or :ref:`LazyExpr <LazyExpr>`.

        Parameters
        ----------
        item: slice, list of slices, optional
            If not None, only the chunks that intersect with the slices
            in items will be evaluated.

        kwargs: dict, optional
            Keyword arguments that are supported by the :func:`empty` constructor.
            These arguments will be set in the resulting :ref:`NDArray <NDArray>`.

        Returns
        -------
        out: :ref:`NDArray <NDArray>`
            A :ref:`NDArray <NDArray>` containing the result of evaluating the
            :ref:`LazyUDF <LazyUDF>` or :ref:`LazyExpr <LazyExpr>`.

        Notes
        -----
        * If self is a LazyArray from an udf, the kwargs used to store the resulting
          array will be the ones passed to the constructor in :func:`lazyudf` (except the
          `urlpath`) updated with the kwargs passed when calling this method.
        """
        pass

    @abstractmethod
    def __getitem__(self, item):
        """
        Get the result of evaluating a slice.

        Parameters
        ----------
        item: int, slice or sequence of slices
            The slice(s) to be retrieved. Note that step parameter is not honored yet.

        Returns
        -------
        out: np.ndarray
            An array with the data containing the slice evaluated.
        """
        pass

    @abstractmethod
    def save(self, **kwargs):
        """
        Save the :ref:`LazyArray` on disk.

        Parameters
        ----------
        kwargs: dict, optional
            Keyword arguments that are supported by the :func:`empty` constructor.
            The `urlpath` must always be provided.

        Returns
        -------
        out: None

        Notes
        -----
        * All the operands of the LazyArray must be Python scalars or on-disk stored :ref:`NDArray <NDArray>`.
        * This is only supported for :ref:`LazyExpr <LazyExpr>`.
        """
        pass

    @property
    @abstractmethod
    def dtype(self):
        """
        Get the data type of the :ref:`LazyArray`.

        Returns
        -------
        out: np.dtype
            The data type of the :ref:`LazyArray`.
        """
        pass

    @property
    @abstractmethod
    def shape(self):
        """
        Get the shape of the :ref:`LazyArray`.

        Returns
        -------
        out: tuple
                The shape of the :ref:`LazyArray`.
        """
        pass

    @property
    @abstractmethod
    def info(self):
        """
        Get information about the :ref:`LazyArray`.

        Returns
        -------
        out: InfoReporter
            A printable class with information about the :ref:`LazyArray`.
        """
        pass


def convert_inputs(inputs):
    inputs_ = []
    for obj in inputs:
        if not isinstance(obj, np.ndarray | blosc2.NDArray | blosc2.C2Array) and not np.isscalar(obj):
            try:
                obj = np.asarray(obj)
            except:
                print(
                    "Inputs not being np.ndarray, NDArray or Python scalar objects"
                    " should be convertible to np.ndarray."
                )
                raise
        inputs_.append(obj)
    return inputs_


def check_broadcast_compatible(arrays):
    shapes = [arr.shape for arr in arrays]
    max_len = max(map(len, shapes))
    # Pad shorter shapes with 1s
    shapes_ = [(1,) * (max_len - len(shape)) + shape for shape in shapes]
    # Reverse the shapes to compare from last dimension
    shapes_ = [shape[::-1] for shape in shapes_]
    # Check
    for dims in zip(*shapes_, strict=True):
        max_dim = max(dims)
        if not all((dim == max_dim) or (dim == 1) for dim in dims):
            _shapes = " ".join(str(shape) for shape in shapes)
            raise ValueError(f"operands could not be broadcast together with shapes {_shapes}")


def compute_broadcast_shape(arrays):
    """
    Returns the shape of the outcome of an operation with the input arrays.
    """
    # When dealing with UDFs, one can arrive params that are not arrays
    shapes = [np.array(arr.shape) for arr in arrays if hasattr(arr, "shape")]
    max_len = max(map(len, shapes))

    # Pad shorter shapes with 1s
    shapes = np.array(
        [np.concatenate([np.ones(max_len - len(shape), dtype=int), shape]) for shape in shapes], dtype=int
    )

    # Compare dimensions from last dimension, take maximum size
    result_shape = np.max(shapes, axis=0)

    return tuple(result_shape)


def check_smaller_shape(value, shape, slice_shape):
    """Check whether the shape of the value is smaller than the shape of the array.

    This follows the NumPy broadcasting rules.
    """
    is_smaller_shape = any(
        s > (1 if i >= len(value.shape) else value.shape[i]) for i, s in enumerate(slice_shape)
    )
    if len(value.shape) < len(shape) or is_smaller_shape:
        return True
    else:
        return False


def _compute_smaller_slice(larger_shape, smaller_shape, larger_slice):
    """
    Returns the slice of the smaller array that corresponds to the slice of the larger array.
    """
    smaller_slice = []
    diff_dims = len(larger_shape) - len(smaller_shape)

    for i in range(len(larger_shape)):
        if i < diff_dims:
            # For leading dimensions of the larger array that the smaller array doesn't have,
            # we don't add anything to the smaller slice
            pass
        else:
            # For dimensions that both arrays have, the slice for the smaller array should be
            # the same as the larger array unless the smaller array's size along that dimension
            # is 1, in which case we use None to indicate the full slice
            if smaller_shape[i - diff_dims] != 1:
                smaller_slice.append(larger_slice[i])
            else:
                smaller_slice.append(slice(None))

    return tuple(smaller_slice)


# A more compact version of the function above, albeit less readable
def compute_smaller_slice(larger_shape, smaller_shape, larger_slice):
    diff_dims = len(larger_shape) - len(smaller_shape)
    return tuple(
        larger_slice[i] if smaller_shape[i - diff_dims] != 1 else slice(None)
        for i in range(diff_dims, len(larger_shape))
    )


def validate_inputs(inputs: dict, getitem=False, out=None) -> tuple:
    """Validate the inputs for the expression."""
    if len(inputs) == 0:
        raise ValueError(
            "You need to pass at least one array.  Use blosc2.empty() if values are not really needed."
        )

    inputs = list(input for input in inputs.values() if hasattr(input, "shape"))

    # All array inputs should have a compatible shape
    if len(inputs) > 1:
        check_broadcast_compatible(inputs)

    ref = inputs[0]
    if not all(np.array_equal(ref.shape, input.shape) for input in inputs):
        # If inputs have different shapes, we cannot take the fast path
        return ref.shape, ref.dtype, False

    # More checks specific of NDArray inputs
    NDinputs = list(input for input in inputs if hasattr(input, "chunks"))
    if len(NDinputs) == 0:
        # All inputs are NumPy arrays, so we cannot take the fast path
        dtype = inputs[0].dtype if out is None else out.dtype
        return inputs[0].shape, dtype, False

    # Check if we can take the fast path
    # For this we need that the chunks and blocks for all inputs (and a possible output)
    # are the same
    equal_chunks, equal_blocks = True, True
    first_input = NDinputs[0]
    # Check the out NDArray (if present) first
    if isinstance(out, blosc2.NDArray):
        if first_input.shape != out.shape:
            raise ValueError("Output shape does not match the first input shape")
        if first_input.blocks != out.blocks:
            equal_blocks = False
        if first_input.chunks != out.chunks:
            equal_chunks = False
    # Then, the rest of the operands
    for input_ in NDinputs:
        if first_input.chunks != input_.chunks:
            equal_chunks = False
        if first_input.blocks != input_.blocks:
            equal_blocks = False
        if input_.blocks[1:] != input_.chunks[1:]:
            # For some reason, the trailing dimensions not being the same is not supported in fast path
            equal_blocks = False
    fast_path = equal_chunks and equal_blocks

    dtype = first_input.dtype if out is None else out.dtype
    return first_input.shape, dtype, fast_path


def is_full_slice(item):
    """Check whether the slice represented by item is a full slice."""
    if item is None:
        # This is the case when the user does not pass any slice in eval() method
        return True
    if isinstance(item, tuple):
        return all((isinstance(i, slice) and i == slice(None, None, None)) or i == Ellipsis for i in item)
    elif isinstance(item, int | bool):
        return False
    else:
        return item == slice(None, None, None) or item == Ellipsis


def do_slices_intersect(slice1, slice2):
    """
    Check whether two slices intersect.

    Parameters
    ----------
    slice1: list of slices
        The first slice
    slice2: list of slices
        The second slice

    Returns
    -------
    bool
        Whether the slices intersect
    """

    # Pad the shorter slice list with full slices (:)
    while len(slice1) < len(slice2):
        slice1.append(slice(None))
    while len(slice2) < len(slice1):
        slice2.append(slice(None))

    # Check each dimension for intersection
    for s1, s2 in zip(slice1, slice2, strict=True):
        if s1 is Ellipsis or s2 is Ellipsis:
            return True
        if s1.start is not None and s2.stop is not None and s1.start >= s2.stop:
            return False
        if s1.stop is not None and s2.start is not None and s1.stop <= s2.start:
            return False

    return True


def fill_chunk_operands(operands, shape, slice_, chunks_, full_chunk, nchunk, chunk_operands):
    """Get the chunk operands for the expression evaluation.

    This function offers a fast path for full chunks and a slow path for the rest.
    """
    for key, value in operands.items():
        if np.isscalar(value):
            chunk_operands[key] = value
            continue
        if value.shape == ():
            chunk_operands[key] = value[()]
            continue

        if isinstance(value, np.ndarray | blosc2.C2Array):
            chunk_operands[key] = value[slice_]
            continue

        # TODO: broadcast is not in the fast path yet, so no need to check for it
        # slice_shape = tuple(s.stop - s.start for s in slice_)
        # if check_smaller_shape(value, shape, slice_shape):
        #     # We need to fetch the part of the value that broadcasts with the operand
        #     smaller_slice = compute_smaller_slice(shape, value.shape, slice_)
        #     chunk_operands[key] = value[smaller_slice]
        #     continue

        if not full_chunk or not isinstance(value, blosc2.NDArray):
            # The chunk is not a full one, or has padding, or is not a blosc2.NDArray,
            # so we need to go the slow path
            chunk_operands[key] = value[slice_]
            continue

        # Fast path for full chunks
        if key in chunk_operands:
            # We already have a buffer for this operand
            value.schunk.decompress_chunk(nchunk, dst=chunk_operands[key])
            continue

        # We don't have a buffer for this operand yet
        # Decompress the whole chunk and store it
        buff = value.schunk.decompress_chunk(nchunk)
        bsize = value.dtype.itemsize * math.prod(chunks_)
        chunk_operands[key] = np.frombuffer(buff[:bsize], dtype=value.dtype).reshape(chunks_)

    return None


def get_chunks_idx(shape, chunks):
    chunks_idx = tuple(math.ceil(s / c) for s, c in zip(shape, chunks, strict=True))
    nchunks = math.prod(chunks_idx)
    return chunks_idx, nchunks


def fast_eval(
    expression: str | Callable, operands: dict, getitem: bool, **kwargs
) -> blosc2.NDArray | np.ndarray:
    """Evaluate the expression in chunks of operands using a fast path.

    Parameters
    ----------
    expression: str or callable
        The expression or udf to evaluate.
    operands: dict
        A dictionary with the operands.
    getitem: bool, optional
        Whether the expression is being evaluated for a getitem operation or eval().
    kwargs: dict, optional
        Keyword arguments that are supported by the :func:`empty` constructor.

    Returns
    -------
    :ref:`NDArray` or np.ndarray
        The output array.
    """
    out = kwargs.pop("_output", None)
    where: dict | None = kwargs.pop("_where_args", None)
    if isinstance(out, blosc2.NDArray):
        # If 'out' has been passed, and is a NDArray, use it as the base array
        basearr = out
    else:
        # Otherwise, find the operand with the 'chunks' attribute and the longest shape
        operands_with_chunks = [o for o in operands.values() if hasattr(o, "chunks")]
        basearr = max(operands_with_chunks, key=lambda x: len(x.shape))

    # Get the shape of the base array
    shape = basearr.shape
    chunks = basearr.chunks
    has_padding = basearr.ext_shape != shape

    chunk_operands = {}
    chunks_idx, nchunks = get_chunks_idx(shape, chunks)

    # Iterate over the chunks and evaluate the expression
    for nchunk in range(nchunks):
        coords = tuple(np.unravel_index(nchunk, chunks_idx))
        slice_ = tuple(
            slice(c * s, min((c + 1) * s, shape[i]))
            for i, (c, s) in enumerate(zip(coords, chunks, strict=True))
        )
        offset = tuple(s.start for s in slice_)  # offset for the udf
        chunks_ = tuple(s.stop - s.start for s in slice_)

        full_chunk = (chunks_ == chunks) and not has_padding
        fill_chunk_operands(operands, shape, slice_, chunks_, full_chunk, nchunk, chunk_operands)

        if isinstance(out, np.ndarray) and not where:
            # Fast path: put the result straight in the output array (avoiding a memory copy)
            if callable(expression):
                expression(tuple(chunk_operands.values()), out[slice_], offset=offset)
            else:
                ne.evaluate(expression, chunk_operands, out=out[slice_])
            continue
        if callable(expression):
            result = np.empty(chunks_, dtype=out.dtype)
            expression(tuple(chunk_operands.values()), result, offset=offset)
        else:
            if where is None:
                result = ne.evaluate(expression, chunk_operands)
            else:
                # Apply the where condition (in result)
                if len(where) == 2:
                    new_expr = f"where({expression}, _where_x, _where_y)"
                    result = ne.evaluate(new_expr, chunk_operands)
                else:
                    # We do not support one or zero operands in the fast path yet
                    raise ValueError("The where condition must be a tuple with one or two elements")

            if out is None:
                # We can enter here when using any of the eval() or __getitem__() methods
                if getitem:
                    out = np.empty(shape, dtype=result.dtype)
                else:
                    out = blosc2.empty(
                        shape, chunks=chunks, blocks=basearr.blocks, dtype=result.dtype, **kwargs
                    )
        # Store the result in the output array
        if getitem:
            out[slice_] = result
        else:
            if has_padding:
                out[slice_] = result
            else:
                out.schunk.update_data(nchunk, result, copy=False)

    return out


def slices_eval(
    expression: str | Callable, operands: dict, getitem: bool, _slice=None, **kwargs
) -> blosc2.NDArray | np.ndarray:
    """Evaluate the expression in chunks of operands.

    This can be used when the operands in the expression have different chunk shapes.
    Also, it can be used when only a slice of the output array is needed.

    This is also flexible enough to be used when the operands have different shapes.

    Parameters
    ----------
    expression: str or callable
        The expression or udf to evaluate.
    operands: dict
        A dictionary with the operands.
    getitem: bool, optional
        Whether the expression is being evaluated for a getitem operation.
    _slice: slice, list of slices, optional
        If not None, only the chunks that intersect with this slice
        will be evaluated.
    kwargs: dict, optional
        Keyword arguments that are supported by the :func:`empty` constructor.

    Returns
    -------
    :ref:`NDArray` or np.ndarray
        The output array.
    """
    out = kwargs.pop("_output", None)
    chunks = kwargs.get("chunks", None)
    where: dict | None = kwargs.pop("_where_args", None)
    # Compute the shape and chunks of the output array, including broadcasting
    shape = compute_broadcast_shape(operands.values())

    # Any operand with chunks attrs will be used to get the chunks
    operands_ = [o for o in operands.values() if hasattr(o, "chunks")]
    if out is None or len(operands_) == 0:
        # operand will be a 'fake' NDArray just to get the necessary chunking information
        operand = blosc2.empty(shape, chunks=chunks)
    else:
        # Typically, we enter here when using UDFs, and out is a NumPy array.
        # Use operands to get the shape and chunks
        operand = operands_[0]
    chunks = operand.chunks

    chunks_idx, nchunks = get_chunks_idx(shape, chunks)
    del operand

    # Iterate over the operands and get the chunks
    lenout = 0
    for nchunk in range(nchunks):
        coords = tuple(np.unravel_index(nchunk, chunks_idx))
        chunk_operands = {}
        # Calculate the shape of the (chunk) slice_ (specially at the end of the array)
        slice_ = tuple(
            slice(c * s, min((c + 1) * s, shape[i]))
            for i, (c, s) in enumerate(zip(coords, chunks, strict=True))
        )
        offset = tuple(s.start for s in slice_)  # offset for the udf
        # Check whether current slice_ intersects with _slice
        if _slice is not None and _slice != ():
            # Ensure that _slice is of type slice
            key = ndindex.ndindex(_slice).expand(shape).raw
            _slice = tuple(k if isinstance(k, slice) else slice(k, k + 1, None) for k in key)
            intersects = do_slices_intersect(_slice, slice_)
            if not intersects:
                continue
        slice_shape = tuple(s.stop - s.start for s in slice_)
        # Get the slice of each operand
        for key, value in operands.items():
            if np.isscalar(value):
                chunk_operands[key] = value
                continue
            if value.shape == ():
                chunk_operands[key] = value[()]
                continue
            if check_smaller_shape(value, shape, slice_shape):
                # We need to fetch the part of the value that broadcasts with the operand
                smaller_slice = compute_smaller_slice(shape, value.shape, slice_)
                chunk_operands[key] = value[smaller_slice]
                continue
            chunk_operands[key] = value[slice_]

        # Evaluate the expression using chunks of operands

        if callable(expression):
            if getitem:
                # Call the udf directly and use out as the output array
                expression(tuple(chunk_operands.values()), out[slice_], offset=offset)
            else:
                result = np.empty(slice_shape, dtype=out.dtype)
                expression(tuple(chunk_operands.values()), result, offset=offset)
                out[slice_] = result
            continue

        if where is None:
            result = ne.evaluate(expression, chunk_operands)
        else:
            # Apply the where condition (in result)
            if len(where) == 2:
                # x = chunk_operands["_where_x"]
                # y = chunk_operands["_where_y"]
                # result = np.where(result, x, y)
                # numexpr is a bit faster than np.where, and we can fuse operations in this case
                new_expr = f"where({expression}, _where_x, _where_y)"
                result = ne.evaluate(new_expr, chunk_operands)
            elif len(where) == 1:
                result = ne.evaluate(expression, chunk_operands)
                x = chunk_operands["_where_x"]
                result = x[result]
            else:
                # result = np.asarray(result).nonzero()
                raise ValueError("The where condition must be a tuple with one or two elements")

        if out is None:
            shape_ = shape
            if where is not None and len(where) < 2:
                # The result is a linear array
                shape_ = math.prod(shape)
            if getitem:
                out = np.empty(shape_, dtype=result.dtype)
            else:
                if "chunks" not in kwargs and (where is None or len(where) == 2):
                    # Let's use the same chunks as the first operand (it could have been automatic too)
                    out = blosc2.empty(shape_, chunks=chunks, dtype=result.dtype, **kwargs)
                elif "chunks" in kwargs and (where is not None and len(where) < 2 and len(shape_) > 1):
                    # Remove the chunks argument if the where condition is not a tuple with two elements
                    kwargs.pop("chunks")
                    out = blosc2.empty(shape_, dtype=result.dtype, **kwargs)
                else:
                    out = blosc2.empty(shape_, dtype=result.dtype, **kwargs)

        if where is None:
            out[slice_] = result
        else:
            if len(where) == 2:
                out[slice_] = result
            elif len(where) == 1:
                lenres = len(result)
                out[lenout : lenout + lenres] = result
                lenout += lenres
            else:
                raise ValueError("The where condition must be a tuple with one or two elements")

    if where is not None and len(where) < 2:
        out = out[:lenout]
    return out


def reduce_slices(
    expression: str | Callable, operands: dict, reduce_args, _slice=None, **kwargs
) -> blosc2.NDArray | np.ndarray:
    """Evaluate the expression in chunks of operands.

    This can be used when the operands in the expression have different chunk shapes.
    Also, it can be used when only a slice of the output array is needed.

    Parameters
    ----------
    expression: str or callable
        The expression or udf to evaluate.
    operands: dict
        A dictionary with the operands.
    reduce_args: dict
        A dictionary with some of the arguments to be passed to np.reduce.
    _slice: slice, list of slices, optional
        If not None, only the chunks that intersect with this slice
        will be evaluated.
    kwargs: dict, optional
        Keyword arguments that are supported by the :func:`empty` constructor.

    Returns
    -------
    :ref:`NDArray` or np.ndarray
        The output array.
    """
    out = kwargs.pop("_output", None)
    where: dict | None = kwargs.pop("_where_args", None)
    reduce_op = reduce_args.pop("op")
    axis = reduce_args["axis"]
    keepdims = reduce_args["keepdims"]

    # Compute the shape and chunks of the output array, including broadcasting
    shape = compute_broadcast_shape(operands.values())

    if axis is None:
        axis = tuple(range(len(shape)))
    elif not isinstance(axis, tuple):
        axis = (axis,)
    if keepdims:
        reduced_shape = tuple(1 if i in axis else s for i, s in enumerate(shape))
    else:
        reduced_shape = tuple(s for i, s in enumerate(shape) if i not in axis)

    # Choose the array with the largest shape as the reference for chunks
    operand = max((o for o in operands.values() if hasattr(o, "chunks")), key=lambda x: len(x.shape))
    chunks = operand.chunks

    # Iterate over the operands and get the chunks
    chunk_operands = {}
    chunks_idx, nchunks = get_chunks_idx(shape, chunks)

    # Iterate over the operands and get the chunks
    for nchunk in range(nchunks):
        coords = tuple(np.unravel_index(nchunk, chunks_idx))
        # Calculate the shape of the (chunk) slice_ (specially at the end of the array)
        slice_ = tuple(
            slice(c * s, min((c + 1) * s, shape[i]))
            for i, (c, s) in enumerate(zip(coords, chunks, strict=True))
        )
        if keepdims:
            reduced_slice = tuple(slice(None) if i in axis else sl for i, sl in enumerate(slice_))
        else:
            reduced_slice = tuple(sl for i, sl in enumerate(slice_) if i not in axis)
        offset = tuple(s.start for s in slice_)  # offset for the udf
        # Check whether current slice_ intersects with _slice
        if _slice is not None:
            intersects = do_slices_intersect(_slice, slice_)
            if not intersects:
                continue
        slice_shape = tuple(s.stop - s.start for s in slice_)
        # reduced_slice_shape = tuple(s.stop - s.start for s in reduced_slice)
        if len(slice_) == 1:
            slice_ = slice_[0]
        if len(reduced_slice) == 1:
            reduced_slice = reduced_slice[0]
        # Get the slice of each operand
        for key, value in operands.items():
            if np.isscalar(value):
                chunk_operands[key] = value
                continue
            if value.shape == ():
                chunk_operands[key] = value[()]
                continue
            if check_smaller_shape(value, shape, slice_shape):
                # We need to fetch the part of the value that broadcasts with the operand
                smaller_slice = compute_smaller_slice(operand.shape, value.shape, slice_)
                chunk_operands[key] = value[smaller_slice]
                continue
            chunk_operands[key] = value[slice_]

        # Evaluate and reduce the expression using chunks of operands

        if callable(expression):
            # TODO: Implement the reductions for UDFs (and test them)
            result = np.empty(slice_shape, dtype=out.dtype)
            expression(tuple(chunk_operands.values()), result, offset=offset)
            # Reduce the result
            result = reduce_op.value.reduce(result, **reduce_args)
            # Update the output array with the result
            out[reduced_slice] = reduce_op.value(out[reduced_slice], result)
            continue

        if where is None:
            result = ne.evaluate(expression, chunk_operands)
        else:
            # Apply the where condition (in result)
            if len(where) == 2:
                # x = chunk_operands["_where_x"]
                # y = chunk_operands["_where_y"]
                # result = np.where(result, x, y)
                # numexpr is a bit faster than np.where, and we can fuse operations in this case
                new_expr = f"where({expression}, _where_x, _where_y)"
                result = ne.evaluate(new_expr, chunk_operands)
            else:
                raise ValueError(
                    "A where condition with less than 2 params is not supported"
                    " yet in combination with reductions"
                )

        # Reduce the result
        if reduce_op == ReduceOp.ANY:
            result = np.any(result, **reduce_args)
        elif reduce_op == ReduceOp.ALL:
            result = np.all(result, **reduce_args)
        else:
            result = reduce_op.value.reduce(result, **reduce_args)
        dtype = reduce_args["dtype"] if reduce_op in (ReduceOp.SUM, ReduceOp.PROD) else None
        if dtype is None:
            dtype = result.dtype

        if out is None:
            # out will be a proper numpy.ndarray
            if reduce_op == ReduceOp.SUM:
                out = np.zeros(reduced_shape, dtype=dtype)
            elif reduce_op == ReduceOp.PROD:
                out = np.ones(reduced_shape, dtype=dtype)
            elif reduce_op == ReduceOp.MIN:
                if np.issubdtype(dtype, np.integer):
                    out = np.iinfo(dtype).max * np.ones(reduced_shape, dtype=dtype)
                else:
                    out = np.inf * np.ones(reduced_shape, dtype=dtype)
            elif reduce_op == ReduceOp.MAX:
                if np.issubdtype(dtype, np.integer):
                    out = np.iinfo(dtype).min * np.ones(reduced_shape, dtype=dtype)
                else:
                    out = -np.inf * np.ones(reduced_shape, dtype=dtype)
            elif reduce_op == ReduceOp.ANY:
                # out = blosc2.zeros(reduced_shape, dtype=np.bool_, **kwargs)
                out = np.zeros(reduced_shape, dtype=np.bool_)
            elif reduce_op == ReduceOp.ALL:
                # out = blosc2.full(reduced_shape, True, dtype=np.bool_, **kwargs)
                out = np.ones(reduced_shape, dtype=np.bool_)

        # Update the output array with the result
        if reduce_op == ReduceOp.ANY:
            out[reduced_slice] += result
        elif reduce_op == ReduceOp.ALL:
            out[reduced_slice] *= result
        else:
            if reduced_slice == ():
                out = reduce_op.value(out, result)
            else:
                out[reduced_slice] = reduce_op.value(out[reduced_slice], result)

    return out


def chunked_eval(expression: str | Callable, operands: dict, item=None, **kwargs):
    getitem = kwargs.pop("_getitem", False)
    out = kwargs.get("_output", None)
    where: dict | None = kwargs.get("_where_args", None)
    if where:
        # Make the where arguments part of the operands
        operands = {**operands, **where}
    shape, dtype_, fast_path = validate_inputs(operands, getitem, out)

    reduce_args = kwargs.pop("_reduce_args", {})
    if reduce_args:
        # Eval and reduce the expression in a single step
        return reduce_slices(expression, operands, reduce_args=reduce_args, _slice=item, **kwargs)

    if not is_full_slice(item) or (where is not None and len(where) < 2):
        # The fast path is not possible when using partial slices or where returning
        # a variable number of elements
        return slices_eval(expression, operands, getitem=getitem, _slice=item, **kwargs)

    if fast_path:
        if getitem:
            # When using getitem, taking the fast path is always possible
            return fast_eval(expression, operands, getitem=True, **kwargs)
        elif (kwargs.get("chunks", None) is None and kwargs.get("blocks", None) is None) and (
            out is None or isinstance(out, blosc2.NDArray)
        ):
            # If not, the conditions to use the fast path are a bit more restrictive
            # e.g. the user cannot specify chunks or blocks, or an output that is not a blosc2.NDArray
            return fast_eval(expression, operands, getitem=False, **kwargs)

    return slices_eval(expression, operands, getitem=getitem, **kwargs)


def fuse_operands(operands1, operands2):
    new_operands = {}
    dup_operands = {}
    new_pos = len(operands1)
    for k2, v2 in operands2.items():
        try:
            k1 = list(operands1.keys())[list(operands1.values()).index(v2)]
            # The operand is duplicated; keep track of it
            dup_operands[k2] = k1
        except ValueError:
            # The value is not among operands1, so rebase it
            new_op = f"o{new_pos}"
            new_pos += 1
            new_operands[new_op] = operands2[k2]
    return new_operands, dup_operands


def fuse_expressions(expr, new_base, dup_op):
    new_expr = ""
    skip_to_char = 0
    old_base = 0
    prev_pos = {}
    for i in range(len(expr)):
        if i < skip_to_char:
            continue
        if expr[i] == "o":
            if i > 0 and (expr[i - 1] != " " and expr[i - 1] != "("):
                # Not a variable
                new_expr += expr[i]
                continue
            # This is a variable.  Find the end of it.
            j = i + 1
            for k in range(len(expr[j:])):
                if expr[j + k] in " )[":
                    j = k
                    break
            if expr[i + j] == ")":
                j -= 1
            old_pos = int(expr[i + 1 : i + j + 1])
            old_op = f"o{old_pos}"
            if old_op not in dup_op:
                if old_pos in prev_pos:
                    # Keep track of duplicated old positions inside expr
                    new_pos = prev_pos[old_pos]
                else:
                    new_pos = old_base + new_base
                    old_base += 1
                new_expr += f"o{new_pos}"
                prev_pos[old_pos] = new_pos
            else:
                new_expr += dup_op[old_op]
            skip_to_char = i + j + 1
        else:
            new_expr += expr[i]
    return new_expr


functions = [
    "sin",
    "cos",
    "tan",
    "sqrt",
    "sinh",
    "cosh",
    "tanh",
    "arcsin",
    "arccos",
    "arctan",
    "arctan2",
    "arcsinh",
    "arccosh",
    "arctanh",
    "exp",
    "expm1",
    "log",
    "log10",
    "log1p",
    "conj",
    "real",
    "imag",
    "contains",
    "abs",
]


class LazyExpr(LazyArray):
    """Class for hosting lazy expressions.

    This is not meant to be called directly from user space.

    Once the lazy expression is created, it can be evaluated via :func:`LazyExpr.eval`.
    """

    def __init__(self, new_op):
        if new_op is None:
            self.expression = ""
            self.operands = {}
            return
        value1, op, value2 = new_op
        if value2 is None:
            if isinstance(value1, LazyExpr):
                self.expression = f"{op}({self.expression})"
            else:
                self.operands = {"o0": value1}
                self.expression = "o0" if op is None else f"{op}(o0)"
            return
        elif op in ("arctan2", "contains", "pow"):
            if np.isscalar(value1) and np.isscalar(value2):
                self.expression = f"{op}(o0, o1)"
            elif np.isscalar(value2):
                self.operands = {"o0": value1}
                self.expression = f"{op}(o0, {value2})"
            elif np.isscalar(value1):
                self.operands = {"o0": value2}
                self.expression = f"{op}({value1} , o0)"
            else:
                self.operands = {"o0": value1, "o1": value2}
                self.expression = f"{op}(o0, o1)"
            return

        if np.isscalar(value1) and np.isscalar(value2):
            self.expression = f"({value1} {op} {value2})"
        elif np.isscalar(value2):
            self.operands = {"o0": value1}
            self.expression = f"(o0 {op} {value2})"
        elif hasattr(value2, "shape") and value2.shape == ():
            self.operands = {"o0": value1}
            self.expression = f"(o0 {op} {value2[()]})"
        elif np.isscalar(value1):
            self.operands = {"o0": value2}
            self.expression = f"({value1} {op} o0)"
        elif hasattr(value1, "shape") and value1.shape == ():
            self.operands = {"o0": value2}
            self.expression = f"({value1[()]} {op} o0)"
        else:
            if value1 is value2:
                self.operands = {"o0": value1}
                self.expression = f"(o0 {op} o0)"
            elif isinstance(value1, LazyExpr) or isinstance(value2, LazyExpr):
                if isinstance(value1, LazyExpr):
                    self.expression = value1.expression
                    self.operands = {"o0": value2}
                else:
                    self.expression = value2.expression
                    self.operands = {"o0": value1}
                newexpr = self.update_expr(new_op)
                self.expression = newexpr.expression
                self.operands = newexpr.operands
            else:
                # This is the very first time that a LazyExpr is formed from two operands
                # that are not LazyExpr themselves
                self.operands = {"o0": value1, "o1": value2}
                self.expression = f"(o0 {op} o1)"

    def update_expr(self, new_op):
        # We use a lot of the original NDArray.__eq__ as 'is', so deactivate the overloaded one
        blosc2._disable_overloaded_equal = True
        # One of the two operands are LazyExpr instances
        value1, op, value2 = new_op
        # The new expression and operands
        expression = None
        new_operands = {}
        # where() handling requires evaluating the expression prior to merge.
        # This is different from reductions, where the expression is evaluated
        # and returned an NumPy array (for usability convenience).
        # We do things like this to enable the fusion of operations like
        # `a.where(0, 1).sum()`.
        # Another possibility would have been to always evaluate where() and produce
        # an NDArray, but that would have been less efficient for the case above.
        if hasattr(value1, "_where_args"):
            value1 = value1.eval()
        if hasattr(value2, "_where_args"):
            value2 = value2.eval()
        if not isinstance(value1, LazyExpr) and not isinstance(value2, LazyExpr):
            # We converted some of the operands to NDArray (where() handling above)
            new_operands = {"o0": value1, "o1": value2}
            expression = f"(o0 {op} o1)"
        elif isinstance(value1, LazyExpr) and isinstance(value2, LazyExpr):
            # Expression fusion
            # Fuse operands in expressions and detect duplicates
            new_operands, dup_op = fuse_operands(value1.operands, value2.operands)
            # Take expression 2 and rebase the operands while removing duplicates
            new_expr = fuse_expressions(value2.expression, len(value1.operands), dup_op)
            expression = f"({self.expression} {op} {new_expr})"
        elif isinstance(value1, LazyExpr):
            if op == "not":
                expression = f"({op}{self.expression})"
            elif np.isscalar(value2):
                expression = f"({self.expression} {op} {value2})"
            elif hasattr(value2, "shape") and value2.shape == ():
                expression = f"({self.expression} {op} {value2[()]})"
            else:
                try:
                    op_name = list(value1.operands.keys())[list(value1.operands.values()).index(value2)]
                except ValueError:
                    op_name = f"o{len(self.operands)}"
                    new_operands = {op_name: value2}
                expression = f"({self.expression} {op} {op_name})"
        else:
            if np.isscalar(value1):
                expression = f"({value1} {op} {self.expression})"
            elif hasattr(value1, "shape") and value1.shape == ():
                expression = f"({value1[()]} {op} {self.expression})"
            else:
                try:
                    op_name = list(value2.operands.keys())[list(value2.operands.values()).index(value1)]
                except ValueError:
                    op_name = f"o{len(self.operands)}"
                    new_operands = {op_name: value1}
                if op == "[]":  # syntactic sugar for slicing
                    expression = f"({op_name}[{self.expression}])"
                else:
                    expression = f"({op_name} {op} {self.expression})"
        blosc2._disable_overloaded_equal = False
        # Return a new expression
        operands = self.operands | new_operands
        return self._new_expr(expression, operands, out=None, where=None)

    @property
    def dtype(self):
        # Updating the expression can change the dtype
        # Infer the dtype by evaluating the scalar version of the expression
        scalar_inputs = {}
        for key, value in self.operands.items():
            single_item = (0,) * len(value.shape)
            scalar_inputs[key] = value[single_item]
        # Evaluate the expression with scalar inputs (it is cheap)
        return ne.evaluate(self.expression, scalar_inputs).dtype

    @property
    def shape(self):
        if hasattr(self, "_shape"):
            # Contrarily to dtype, shape cannot change after creation of the expression
            return self._shape
        shape, dtype_, fast_path = validate_inputs(self.operands)
        self._shape = shape
        return shape

    def __neg__(self):
        return self.update_expr(new_op=(0, "-", self))

    def __add__(self, value):
        return self.update_expr(new_op=(self, "+", value))

    def __iadd__(self, other):
        return self.update_expr(new_op=(self, "+", other))

    def __radd__(self, value):
        return self.update_expr(new_op=(value, "+", self))

    def __sub__(self, value):
        return self.update_expr(new_op=(self, "-", value))

    def __isub__(self, value):
        return self.update_expr(new_op=(self, "-", value))

    def __rsub__(self, value):
        return self.update_expr(new_op=(value, "-", self))

    def __mul__(self, value):
        return self.update_expr(new_op=(self, "*", value))

    def __imul__(self, value):
        return self.update_expr(new_op=(self, "*", value))

    def __rmul__(self, value):
        return self.update_expr(new_op=(value, "*", self))

    def __truediv__(self, value):
        return self.update_expr(new_op=(self, "/", value))

    def __itruediv__(self, value):
        return self.update_expr(new_op=(self, "/", value))

    def __rtruediv__(self, value):
        return self.update_expr(new_op=(value, "/", self))

    def __and__(self, value):
        return self.update_expr(new_op=(self, "and", value))

    def __rand__(self, value):
        return self.update_expr(new_op=(value, "and", self))

    def __or__(self, value):
        return self.update_expr(new_op=(self, "or", value))

    def __ror__(self, value):
        return self.update_expr(new_op=(value, "or", self))

    def __invert__(self):
        return self.update_expr(new_op=(self, "not", None))

    def __pow__(self, value):
        return self.update_expr(new_op=(self, "**", value))

    def __rpow__(self, value):
        return self.update_expr(new_op=(value, "**", self))

    def __ipow__(self, value):
        return self.update_expr(new_op=(self, "**", value))

    def __lt__(self, value):
        return self.update_expr(new_op=(self, "<", value))

    def __le__(self, value):
        return self.update_expr(new_op=(self, "<=", value))

    def __eq__(self, value):
        return self.update_expr(new_op=(self, "==", value))

    def __ne__(self, value):
        return self.update_expr(new_op=(self, "!=", value))

    def __gt__(self, value):
        return self.update_expr(new_op=(self, ">", value))

    def __ge__(self, value):
        return self.update_expr(new_op=(self, ">=", value))

    def where(self, value1=None, value2=None):
        if self.dtype != np.bool_:
            raise ValueError("where() can only be used with boolean expressions")
        # This just acts as a 'decorator' for the existing expression
        if value1 is not None and value2 is not None:
            args = dict(_where_x=value1, _where_y=value2)
        elif value1 is not None:
            args = dict(_where_x=value1)
        elif value2 is not None:
            raise ValueError("where() requires value1 when using value2")
        else:
            args = {}
        self._where_args = args
        return self

    def sum(self, axis=None, dtype=None, keepdims=False):
        # Always evaluate the expression prior the reduction
        reduce_args = {
            "op": ReduceOp.SUM,
            "axis": axis,
            "dtype": dtype,
            "keepdims": keepdims,
        }
        return self.eval(_reduce_args=reduce_args)

    def mean(self, axis=None, dtype=None, keepdims=False):
        # Always evaluate the expression prior the reduction
        total_sum = self.sum(axis=axis, dtype=dtype, keepdims=keepdims)
        if np.isscalar(axis):
            axis = (axis,)
        num_elements = np.prod(self.shape) if axis is None else np.prod([self.shape[i] for i in axis])
        mean_expr = total_sum / num_elements
        return mean_expr

    def std(self, axis=None, dtype=None, keepdims=False, ddof=0):
        # Always evaluate the expression prior the reduction
        mean_value = self.mean(axis=axis, dtype=dtype, keepdims=True)
        std_expr = (self - mean_value) ** 2
        std_expr = std_expr.mean(axis=axis, dtype=dtype, keepdims=keepdims)
        if ddof != 0:
            if axis is None:
                num_elements = np.prod(self.shape)
            else:
                num_elements = np.prod([self.shape[i] for i in axis])
            std_expr = np.sqrt(std_expr * num_elements / (num_elements - ddof))
        else:
            std_expr = np.sqrt(std_expr)
        return std_expr

    def var(self, axis=None, dtype=None, keepdims=False, ddof=0):
        # Always evaluate the expression prior the reduction
        mean_value = self.mean(axis=axis, dtype=dtype, keepdims=True)
        var_expr = (self - mean_value) ** 2
        if ddof != 0:
            var_expr = var_expr.mean(axis=axis, dtype=dtype, keepdims=keepdims)
            if axis is None:
                num_elements = np.prod(self.shape)
            else:
                num_elements = np.prod([self.shape[i] for i in axis])
            var_values = var_expr * num_elements / (num_elements - ddof)
        else:
            var_values = var_expr.mean(axis=axis, dtype=dtype, keepdims=keepdims)
        return var_values

    def prod(self, axis=None, dtype=None, keepdims=False):
        # Always evaluate the expression prior the reduction
        reduce_args = {
            "op": ReduceOp.PROD,
            "axis": axis,
            "dtype": dtype,
            "keepdims": keepdims,
        }
        return self.eval(_reduce_args=reduce_args)

    def min(self, axis=None, keepdims=False):
        # Always evaluate the expression prior the reduction
        reduce_args = {
            "op": ReduceOp.MIN,
            "axis": axis,
            "keepdims": keepdims,
        }
        return self.eval(_reduce_args=reduce_args)

    def max(self, axis=None, keepdims=False):
        # Always evaluate the expression prior the reduction
        reduce_args = {
            "op": ReduceOp.MAX,
            "axis": axis,
            "keepdims": keepdims,
        }
        return self.eval(_reduce_args=reduce_args)

    def any(self, axis=None, keepdims=False):
        # Always evaluate the expression prior the reduction
        reduce_args = {
            "op": ReduceOp.ANY,
            "axis": axis,
            "keepdims": keepdims,
        }
        return self.eval(_reduce_args=reduce_args)

    def all(self, axis=None, keepdims=False):
        # Always evaluate the expression prior the reduction
        reduce_args = {
            "op": ReduceOp.ALL,
            "axis": axis,
            "keepdims": keepdims,
        }
        return self.eval(_reduce_args=reduce_args)

    def eval(self, item=None, **kwargs) -> blosc2.NDArray:
        if hasattr(self, "_output"):
            kwargs["_output"] = self._output
        if hasattr(self, "_where_args"):
            kwargs["_where_args"] = self._where_args
        return chunked_eval(self.expression, self.operands, item, **kwargs)

    def __getitem__(self, item):
        kwargs = {"_getitem": True}
        if hasattr(self, "_output"):
            kwargs["_output"] = self._output
        if hasattr(self, "_where_args"):
            kwargs["_where_args"] = self._where_args
        array = chunked_eval(self.expression, self.operands, item, **kwargs)
        full_slice = is_full_slice(item)
        # When a partial slice is requested, only the data delimited by item is filled
        return array[item] if not full_slice else array

    def __str__(self):
        expression = f"{self.expression}"
        return expression

    @property
    def info(self):
        return InfoReporter(self)

    @property
    def info_items(self):
        items = []
        items += [("type", f"{self.__class__.__name__}")]
        items += [("expression", self.expression)]
        opsinfo = {
            key: str(value) if value.schunk.urlpath is None else value.schunk.urlpath
            for key, value in self.operands.items()
        }
        items += [("operands", opsinfo)]
        items += [("shape", self.shape)]
        items += [("dtype", self.dtype)]
        return items

    def save(self, **kwargs):
        if kwargs.get("urlpath", None) is None:
            raise ValueError("To save a LazyArray you must provide an urlpath")

        meta = kwargs.get("meta", {})
        meta["LazyArray"] = LazyArrayEnum.Expr.value
        kwargs["meta"] = meta
        kwargs["mode"] = "w"  # always overwrite the file in urlpath

        # Create an empty array; useful for providing the shape and dtype of the outcome
        array = blosc2.empty(shape=self.shape, dtype=self.dtype, **kwargs)

        # Save the expression and operands in the metadata
        operands = {}
        for key, value in self.operands.items():
            if isinstance(value, blosc2.C2Array):
                operands[key] = {
                    "path": str(value.path),
                    "sub_url": value.sub_url,
                    "auth_cookie": value.auth_cookie,
                }
                continue
            if not isinstance(value, blosc2.NDArray):
                raise ValueError(
                    "To save a LazyArray, all operands must be blosc2.NDArray or blosc2.C2Array objects"
                )
            if value.schunk.urlpath is None:
                raise ValueError("To save a LazyArray, all operands must be stored on disk/network")
            operands[key] = value.schunk.urlpath
        # Check that the expression is valid
        ne.validate(self.expression, locals=operands)
        array.schunk.vlmeta["_LazyArray"] = {
            "expression": self.expression,
            "UDF": None,
            "operands": operands,
        }
        return

    @classmethod
    def _new_expr(cls, expression, operands, out=None, where=None):
        # Create a new LazyExpr object
        new_expr = cls(None)
        ne.validate(expression, locals=operands)
        new_expr.expression = expression
        new_expr.operands = operands
        if out is not None:
            new_expr._output = out
        if where is not None:
            new_expr._where_args = where
        return new_expr


class LazyUDF(LazyArray):
    def __init__(self, func, inputs, dtype, chunked_eval=True, **kwargs):
        # After this, all the inputs should be np.ndarray or NDArray objects
        self.inputs = convert_inputs(inputs)
        self.chunked_eval = chunked_eval
        # Get res shape
        self._shape = compute_broadcast_shape(self.inputs)
        if self._shape is None:
            raise NotImplementedError("If all operands are scalars, use python, numpy or numexpr")

        self.kwargs = kwargs
        self._dtype = dtype
        self.func = func

        # Prepare internal array for __getitem__
        # Deep copy the kwargs to avoid modifying them
        kwargs_getitem = copy.deepcopy(self.kwargs)
        # Cannot use multithreading when applying a postfilter, dparams['nthreads'] ignored
        dparams = kwargs_getitem.get("dparams", {})
        if isinstance(dparams, dict):
            dparams["nthreads"] = 1
        else:
            raise ValueError("dparams should be a dictionary")
        kwargs_getitem["dparams"] = dparams

        self.res_getitem = blosc2.empty(self._shape, self._dtype, **kwargs_getitem)
        # Register a postfilter for getitem
        self.res_getitem._set_postf_udf(self.func, id(self.inputs))

        self.inputs_dict = {f"o{i}": obj for i, obj in enumerate(self.inputs)}

    @property
    def dtype(self):
        return self._dtype

    @property
    def shape(self):
        return self._shape

    @property
    def info(self):
        return InfoReporter(self)

    @property
    def info_items(self):
        items = []
        items += [("type", f"{self.__class__.__name__}")]
        inputs = {}
        for key, value in self.inputs_dict.items():
            if isinstance(value, np.ndarray | blosc2.NDArray | blosc2.C2Array):
                inputs[key] = f"<{value.__class__.__name__}> {value.shape} {value.dtype}"
            else:
                inputs[key] = str(value)
        items += [("inputs", inputs)]
        items += [("shape", self.shape)]
        items += [("dtype", self.dtype)]
        return items

    def eval(self, item=None, **kwargs):
        # Get kwargs
        if kwargs is None:
            kwargs = {}
        # Do copy to avoid modifying the original parameters
        aux_kwargs = copy.deepcopy(self.kwargs)
        # Update is not recursive
        cparams = aux_kwargs.get("cparams", {})
        cparams.update(kwargs.get("cparams", {}))
        aux_kwargs["cparams"] = cparams
        dparams = aux_kwargs.get("dparams", {})
        dparams.update(kwargs.get("dparams", {}))
        aux_kwargs["dparams"] = dparams
        _ = kwargs.pop("cparams", None)
        _ = kwargs.pop("dparams", None)
        urlpath = kwargs.get("urlpath", None)
        if urlpath is not None and urlpath == aux_kwargs.get("urlpath", None):
            raise ValueError("Cannot use the same urlpath for LazyArray and eval NDArray")
        _ = aux_kwargs.pop("urlpath", None)
        aux_kwargs.update(kwargs)

        if item is None:
            if self.chunked_eval:
                res_eval = blosc2.empty(self.shape, self.dtype, **aux_kwargs)
                chunked_eval(self.func, self.inputs_dict, None, _getitem=False, _output=res_eval)
                return res_eval

            # Cannot use multithreading when applying a prefilter, save nthreads to set them
            # after the evaluation
            cparams = aux_kwargs.get("cparams", {})
            if isinstance(cparams, dict):
                self._cnthreads = cparams.get("nthreads", blosc2.cparams_dflts["nthreads"])
                cparams["nthreads"] = 1
            else:
                raise ValueError("cparams should be a dictionary")
            aux_kwargs["cparams"] = cparams

            res_eval = blosc2.empty(self.shape, self.dtype, **aux_kwargs)
            # Register a prefilter for eval
            res_eval._set_pref_udf(self.func, id(self.inputs))

            aux = np.empty(res_eval.shape, res_eval.dtype)
            res_eval[...] = aux
            res_eval.schunk.remove_prefilter(self.func.__name__)
            res_eval.schunk.cparams["nthreads"] = self._cnthreads

            return res_eval
        else:
            # Get only a slice
            np_array = self.__getitem__(item)
            return blosc2.asarray(np_array, **aux_kwargs)

    def __getitem__(self, item):
        if self.chunked_eval:
            output = np.empty(self.shape, self.dtype)
            chunked_eval(self.func, self.inputs_dict, item, _getitem=True, _output=output)
            return output[item]
        return self.res_getitem[item]

    def save(self, **kwargs):
        raise NotImplementedError("For safety reasons, this is not implemented for UDFs")


def _open_lazyarray(array):
    value = array.schunk.meta["LazyArray"]
    if value == LazyArrayEnum.UDF.value:
        raise NotImplementedError("For safety reasons, persistent UDFs are not supported")

    # LazyExpr
    lazyarray = array.schunk.vlmeta["_LazyArray"]
    operands = lazyarray["operands"]
    parent_path = Path(array.schunk.urlpath).parent
    operands_dict = {}
    for key, value in operands.items():
        if isinstance(value, str):
            value = parent_path / value
            op = blosc2.open(value)
            operands_dict[key] = op
        elif isinstance(value, dict):
            # C2Array
            operands_dict[key] = blosc2.C2Array(
                pathlib.Path(value["path"]), sub_url=value["sub_url"], auth_cookie=value["auth_cookie"]
            )
        else:
            raise ValueError("Error when retrieving the operands")

    expr = lazyarray["expression"]
    globals = {}
    for func in functions:
        if func in expr:
            globals[func] = getattr(blosc2, func)

    # Validate the expression (prevent security issues)
    ne.validate(expr, globals, operands_dict)
    # Create the expression as such
    expr = eval(expr, globals, operands_dict)
    # Make the array info available for the user (only available when opened from disk)
    expr.array = array
    return expr


def lazyudf(func, inputs, dtype, chunked_eval=True, **kwargs):
    """
    Get a LazyUDF from a python user-defined function.

    Parameters
    ----------
    func: Python function
        User defined function to apply to each block. This function will
        always receive the same parameters: `inputs_tuple`, `output` and `offset`.
        The first one will contain the corresponding slice for the block of each
        input in :paramref:`inputs`. The second, the buffer to be filled as a multidimensional
        numpy.ndarray. And the third one, the multidimensional offset corresponding
        to the start of the block that it is being computed.
    inputs: tuple or list
        The sequence of inputs. The supported inputs are NumPy.ndarray,
        Python scalars, and :ref:`NDArray <NDArray>`.
    dtype: np.dtype
        The resulting ndarray dtype in NumPy format.
    chunked_eval: bool, optional
        Whether to evaluate the expression in chunks or not (blocks).
    kwargs: dict, optional
        Keyword arguments that are supported by the :func:`empty` constructor.
        These arguments will be used by the :meth:`LazyArray.__getitem__` and
        :meth:`LazyArray.eval` methods. The
        last one will ignore the `urlpath` parameter passed in this function.

    Returns
    -------
    out: :ref:`LazyUDF <LazyUDF>`
        A :ref:`LazyUDF <LazyUDF>` is returned.

    """
    return LazyUDF(func, inputs, dtype, chunked_eval, **kwargs)


def lazyexpr(expression, operands=None, out=None, where=None):
    """
    Get a LazyExpr from an expression.

    Parameters
    ----------
    expression: str or bytes or LazyExpr
        The expression to evaluate. This can be any valid expression that can be
        ingested by numexpr. If a LazyExpr is passed, the expression will be
        updated with the new operands.
    operands: dict
        The dictionary with operands. Supported values are NumPy.ndarray,
        Python scalars, and :ref:`NDArray <NDArray>` instances.
    out: NDArray or np.ndarray, optional
        The output array where the result will be stored. If not provided,
        a new array will be created.
    where: tuple, list, optional
        A sequence with the where arguments. This is useful when the expression
        contains a where clause. The where arguments should be provided as a sequence.

    Returns
    -------
    out: :ref:`LazyExpr <LazyExpr>`
        A :ref:`LazyExpr <LazyExpr>` is returned.

    """
    if isinstance(expression, LazyExpr):
        if operands is not None:
            expression.operands.update(operands)
        if out is not None:
            expression._output = out
        if where is not None:
            where_args = dict(_where_x=where[0], _where_y=where[1])
            expression._where_args = where_args
        return expression
    if operands is None:
        raise ValueError("`operands` must be provided for a string expression")
    return LazyExpr._new_expr(expression, operands, out=out, where=where)


if __name__ == "__main__":
    from time import time

    # Create initial containers
    na1 = np.linspace(0, 10, 10_000_000, dtype=np.float64)
    a1 = blosc2.asarray(na1)
    na2 = np.copy(na1)
    a2 = blosc2.asarray(na2)
    na3 = np.copy(na1)
    a3 = blosc2.asarray(na3)
    na4 = np.copy(na1)
    a4 = blosc2.asarray(na4)
    # Interesting slice
    # sl = None
    sl = slice(0, 10_000)
    # Create a simple lazy expression
    expr = a1 + a2
    print(expr)
    t0 = time()
    nres = na1 + na2
    print(f"Elapsed time (numpy, [:]): {time() - t0:.3f} s")
    t0 = time()
    nres = ne.evaluate("na1 + na2")
    print(f"Elapsed time (numexpr, [:]): {time() - t0:.3f} s")
    nres = nres[sl] if sl is not None else nres
    t0 = time()
    res = expr.eval(item=sl)
    print(f"Elapsed time (evaluate): {time() - t0:.3f} s")
    res = res[sl] if sl is not None else res[:]
    t0 = time()
    res2 = expr[sl]
    print(f"Elapsed time (getitem): {time() - t0:.3f} s")
    np.testing.assert_allclose(res, nres)
    np.testing.assert_allclose(res2, nres)

    # Complex lazy expression
    expr = blosc2.tan(a1) * (blosc2.sin(a2) * blosc2.sin(a2) + blosc2.cos(a3)) + (blosc2.sqrt(a4) * 2)
    # expr = blosc2.sin(a1) + 2 * a1 + 1
    expr += 2
    print(expr)
    t0 = time()
    nres = np.tan(na1) * (np.sin(na2) * np.sin(na2) + np.cos(na3)) + (np.sqrt(na4) * 2) + 2
    # nres = np.sin(na1[:]) + 2 * na1[:] + 1 + 2
    print(f"Elapsed time (numpy, [:]): {time() - t0:.3f} s")
    t0 = time()
    nres = ne.evaluate("tan(na1) * (sin(na2) * sin(na2) + cos(na3)) + (sqrt(na4) * 2) + 2")
    print(f"Elapsed time (numexpr, [:]): {time() - t0:.3f} s")
    nres = nres[sl] if sl is not None else nres
    t0 = time()
    res = expr.eval(sl)
    print(f"Elapsed time (evaluate): {time() - t0:.3f} s")
    res = res[sl] if sl is not None else res[:]
    t0 = time()
    res2 = expr[sl]
    print(f"Elapsed time (getitem): {time() - t0:.3f} s")
    np.testing.assert_allclose(res, nres)
    np.testing.assert_allclose(res2, nres)
    print("Everything is working fine")
