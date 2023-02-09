import ndindex

import numpy as np
from blosc2 import blosc2_ext

from .info import InfoReporter
from .SChunk import SChunk


def process_key(key, shape):
    key = ndindex.ndindex(key).expand(shape).raw
    mask = tuple(True if isinstance(k, int) else False for k in key)
    key = tuple(k if isinstance(k, slice) else slice(k, k+1, None) for k in key)
    return key, mask


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

    @classmethod
    def cast(cls, cont):
        cont.__class__ = cls
        assert isinstance(cont, NDArray)
        return cont

    @property
    def info(self):
        """
        Print information about this array.
        """
        return InfoReporter(self)

    @property
    def info_items(self):
        items = []
        items += [("Type", f"{self.__class__.__name__}")]
        items += [("Typesize", self.schunk.typesize)]
        items += [("Shape", self.shape)]
        items += [("Chunks", self.chunks)]
        items += [("Blocks", self.blocks)]
        items += [("Comp. codec", self.schunk.cparams["codec"].name)]
        items += [("Comp. level", self.schunk.cparams["clevel"])]
        filters = [f.name for f in self.schunk.cparams["filters"] if f.name != "NOFILTER"]
        items += [("Comp. filters", f"[{', '.join(map(str, filters))}]")]
        items += [("Comp. ratio", f"{self.schunk.cratio:.2f}")]
        return items

    def __getitem__(self, key):
        """ Get a (multidimensional) slice as specified in key.

        Parameters
        ----------
        key: int, slice or sequence of slices
            The index for the slices to be updated. Note that step parameter is not honored yet
            in slices.

        Returns
        -------
        out: :ref:`NDArray <NDArray>`
            An array, stored in a non-compressed buffer, with the requested data.
        """
        key, _ = process_key(key, self.shape)
        start, stop, _ = get_ndarray_start_stop(self.ndim, key, self.shape)
        key = (start, stop)
        shape = [sp - st for st, sp in zip(start, stop)]
        arr = np.zeros(shape, dtype=f"S{self.schunk.typesize}")

        return super(NDArray, self).get_slice_numpy(arr, key)

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

        """
        key, _ = process_key(key, self.shape)
        start, stop, _ = get_ndarray_start_stop(self.ndim, key, self.shape)
        key = (start, stop)

        return super(NDArray, self).set_slice(key, value)

    def to_buffer(self):
        """Returns a buffer with the data contents.

        Returns
        -------
        bytes
            The buffer containing the data of the whole array.
        """
        return super(NDArray, self).to_buffer()

    def copy(self, **kwargs):
        """Create a copy of an array.

        Parameters
        ----------
        array: :ref:`NDArray <NDArray>`
            The array to be copied.

        Other Parameters
        ----------------
        kwargs: dict, optional
            Keyword arguments that are supported by the :func:`empty` constructor.

        Returns
        -------
        out: :ref:`NDArray <NDArray>`
            A :ref:`NDArray <NDArray>` with a copy of the data.
        """
        return super(NDArray, self).copy(**kwargs)

    def resize(self, newshape):
        """Change the shape of the array by growing one or more dimensions.

        Parameters
        ----------
        newshape : tuple or list
            The new shape of the array. It should have the same dimensions
            as :paramref:`self`.

        Notes
        -----
        The array values corresponding to the added positions are not initialized.
        Thus, the user is in charge of initializing them.
        """
        return super(NDArray, self).resize(newshape)

    def slice(self, key, **kwargs):
        """ Get a (multidimensional) slice as specified in key. Generalizes :meth:`__getitem__`.

        Parameters
        ----------
        key: int, slice or sequence of slices
            The index for the slices to be updated. Note that step parameter is not honored yet in
            slices.

        Other Parameters
        ----------------
        kwargs: dict, optional
            Keyword arguments that are supported by the :func:`empty` constructor.

        Returns
        -------
        out: :ref:`NDArray <NDArray>`
            An array with the requested data.
        """
        key, mask = process_key(key, self.shape)
        start, stop, _ = get_ndarray_start_stop(self.ndim, key, self.shape)
        key = (start, stop)
        return super(NDArray, self).get_slice(self, key, mask, **kwargs)

    def squeeze(self):
        """Remove the 1's in array's shape."""
        super(NDArray, self).squeeze()


def empty(shape, chunks, blocks, typesize=1, **kwargs):
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
    typesize: int, optional
        The size, in bytes, of each element. Default is 1.
        This will override the `typesize`
        in the cparams in case they are passed.

    Other Parameters
    ----------------
    kwargs: dict, optional
        Keyword arguments supported:

        The keyword arguments supported are the same than for the
        :obj:`SChunk.__init__ <blosc2.SChunk.SChunk.__init__>`.

    Returns
    -------
    out: :ref:`NDArray <NDArray>`
        A :ref:`NDArray <NDArray>` is returned.
    """
    arr = blosc2_ext.empty(shape, chunks, blocks, typesize, **kwargs)
    return arr


def zeros(shape, chunks, blocks, typesize=1, **kwargs):
    """Create an array, with zero being used as the default value
    for uninitialized portions of the array.

    The parameters and keyword arguments are the same than for the
    :func:`empty` constructor.

    Returns
    -------
    out: :ref:`NDArray <NDArray>`
        A :ref:`NDArray <NDArray>` is returned.
    """
    arr = blosc2_ext.zeros(shape, chunks, blocks, typesize, **kwargs)
    return arr


def full(shape, chunks, blocks, fill_value, **kwargs):
    """Create an array, with :paramref:`fill_value` being used as the default value
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
        Keyword arguments that are supported by the :func:`empty` constructor.

    Returns
    -------
    out: :ref:`NDArray <NDArray>`
        A :ref:`NDArray <NDArray>` is returned.
    """
    arr = blosc2_ext.full(shape, chunks, blocks, fill_value, **kwargs)
    return arr


def from_buffer(buffer, shape, chunks, blocks, typesize=1, **kwargs):
    """Create an array out of a buffer.

    Parameters
    ----------
    buffer: bytes
        The buffer of the data to populate the container.
    shape: tuple or list
        The shape for the final container.
    chunks: tuple or list
        The chunk shape.
    blocks: tuple or list
        The block shape. This will override the `blocksize`
        in the cparams in case they are passed.
    typesize: int, optional
        The size, in bytes, of each element. Default is 1.
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
    """
    arr = blosc2_ext.from_buffer(buffer, shape, chunks, blocks, typesize, **kwargs)
    return arr


def copy(array, **kwargs):
    """Create a copy of an array.

    Parameters
    ----------
    array: :ref:`NDArray <NDArray>`
        The array to be copied.

    Other Parameters
    ----------------
    kwargs: dict, optional
        Keyword arguments that are supported by the :func:`empty` constructor.

    Returns
    -------
    out: :ref:`NDArray <NDArray>`
        A :ref:`NDArray <NDArray>` with a copy of the data.
    """
    arr = array.copy(**kwargs)
    return arr


def asarray(array, chunks, blocks, **kwargs):
    """Convert the input to an array.

    Parameters
    ----------
    array: array_like
        An array supporting the python buffer protocol and the numpy array interface.
    chunks: tuple or list
        The chunk shape.
    blocks: tuple or list
        The block shape. This will override the `blocksize`
        in the cparams in case they are passed.

    Other Parameters
    ----------------
    kwargs: dict, optional
        Keyword arguments that are supported by the :func:`empty` constructor.

    Returns
    -------
    out: :ref:`NDArray <NDArray>`
        An array interpretation of :paramref:`array`.
    """
    return blosc2_ext.asarray(array, chunks, blocks, **kwargs)
