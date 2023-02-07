import ndindex

import numpy as np
from blosc2 import blosc2_ext

from .SChunk import SChunk


def process_key(key, shape):
    key = ndindex.ndindex(key).expand(shape).raw
    key = tuple(k if isinstance(k, slice) else slice(k, k+1, None) for k in key)
    return key


def prod(list):
    prod = 1
    for li in list:
        prod *= li
    return prod


def get_ndarray_start_stop(ndim, key, shape):
    start = tuple(s.start if s.start is not None else 0 for s in key)
    stop = tuple(s.stop if s.stop is not None else sh for s, sh in zip(key, shape))

    size = prod([stop[i] - start[i] for i in range(ndim)])

    return start, stop, size


class NDArray(blosc2_ext.NDArray):
    def __init__(self, **kwargs):
        self.schunk = SChunk(_schunk=kwargs["_schunk"], _is_view=True)  # SChunk Python instance
        super(NDArray, self).__init__(kwargs["_array"])

    def __getitem__(self, key):
        """ Get a (multidimensional) slice as specified in key.

        Parameters
        ----------
        key: int, slice or sequence of slices
            The index for the slices to be updated. Note that step parameter is not honored yet
            in slices.

        Returns
        -------
        out: NDArray
            An array, stored in a non-compressed buffer, with the requested data.
        """
        key = process_key(key, self.shape)
        start, stop, _ = get_ndarray_start_stop(self.ndim, key, self.shape)
        key = (start, stop)
        shape = [sp - st for st, sp in zip(start, stop)]
        arr = np.zeros(shape, dtype=f"S{self.schunk.typesize}")

        return super(NDArray, self).get_slice_numpy(arr, key)


def empty(shape, chunks, blocks, typesize, **kwargs):
    """Create an empty array.

    Parameters
    ----------
    shape: tuple or list
        The shape for the final array.
    chunks: tuple or list
        The chunk shape.
    blocks: tuple or list
        The block shape. This will override the `blocksize`
        in the cparams in case they are passed.
    typesize: int
        The size, in bytes, of each element. This will override the `typesize`
        in the cparams in case they are passed.

    Other Parameters
    ----------------
    kwargs: dict, optional
        Keyword arguments supported:

        The keyword arguments supported are the same than for the :py_meth:`SChunk`.

    Returns
    -------
    out: NDArray
        A `NDArray` is returned.
    """
    arr = blosc2_ext.empty(shape, chunks, blocks, typesize, **kwargs)
    return arr


def zeros(shape, chunks, blocks, typesize, **kwargs):
    """Create an array, with zero being used as the default value
    for uninitialized portions of the array.

    Parameters
    ----------
    The parameters are the same than for the :py:meth:`caterva.empty` constructor.

    Returns
    -------
    out: NDArray
        A `NDArray` is returned.
    """
    arr = blosc2_ext.zeros(shape, chunks, blocks, typesize, **kwargs)
    return arr


def full(shape, chunks, blocks, fill_value, **kwargs):
    """Create an array, with @p fill_value being used as the default value
    for uninitialized portions of the array.

    Parameters
    ----------
    shape: tuple or list
        The shape for the final array.
    chunks: tuple or list
        The chunk shape.
    blocks: tuple or list
        The block shape. This will override the `blocksize`
        in the cparams in case they are passed.
    fill_value: bytes
        Default value to use for uninitialized portions of the array.
        Its size will override the `typesize`
        in the cparams in case they are passed.

    Other Parameters
    ----------------
    kwargs: dict, optional
        Keyword arguments that are supported by the :py:meth:`caterva.empty` constructor.

    Returns
    -------
    out: NDArray
        A `NDArray` is returned.
    """
    arr = blosc2_ext.full(shape, chunks, blocks, fill_value, **kwargs)
    return arr
