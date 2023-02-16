import ndindex

import numpy as np
from blosc2 import blosc2_ext, compute_chunks_blocks

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
        self._schunk = SChunk(_schunk=kwargs["_schunk"], _is_view=True)  # SChunk Python instance
        super(NDArray, self).__init__(kwargs["_array"])

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
        All the attributes from the :ref:`SChunk <SChunk>` can be accessed through this instance
        as `self.schunk`.

        See Also
        --------
        :ref:`SChunk Attributes <SChunkAttributes>`
        """
        return self._schunk

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

        Examples
        --------
        >>> import blosc2
        >>> import numpy as np
        >>> shape = [25, 10]
        >>> # Create an array
        >>> a = blosc2.full(shape, 3.3333)
        >>> b = np.full(shape, 3.3333)
        >>> # Get slice as a NumPy array
        >>> c = a[...]
        >>> np.testing.assert_allclose(c, b)
        """
        key, _ = process_key(key, self.shape)
        start, stop, _ = get_ndarray_start_stop(self.ndim, key, self.shape)
        key = (start, stop)
        shape = [sp - st for st, sp in zip(start, stop)]
        arr = np.zeros(shape, dtype=self.dtype)

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

        Examples
        --------
        >>> import blosc2
        >>> import numpy as np
        >>> shape = [25, 10]
        >>> # Create an array
        >>> a = blosc2.full(shape, 3.3333)
        >>> # Set a slice to 0
        >>> a[:5, :5] = 0
        >>> b = np.zeros([5, 5])
        >>> assert np.array_equal(a[:5, :5], b)
        """
        key, _ = process_key(key, self.shape)
        start, stop, _ = get_ndarray_start_stop(self.ndim, key, self.shape)
        key = (start, stop)

        if isinstance(value, (int, float, bool)):
            shape = [sp - st for sp, st in zip(stop, start)]
            value = np.full(shape, value, dtype=self.dtype)
        elif isinstance(value, NDArray):
            value = value[...]

        return super(NDArray, self).set_slice(key, value)

    def to_buffer(self):
        """Returns a buffer with the data contents.

        Returns
        -------
        bytes
            The buffer containing the data of the whole array.

        Examples
        --------
        >>> import blosc2
        >>> import numpy as np
        >>> dtype = np.int32
        >>> shape = [23, 11]
        >>> a = np.arange(0, int(np.prod(shape)), dtype=dtype).reshape(shape)
        >>> # Create an array
        >>> b = blosc2.asarray(a, dtype=dtype)
        >>> assert b.to_buffer() == bytes(a[...])
        """
        return super(NDArray, self).to_buffer()

    def copy(self, dtype=None, **kwargs):
        """Create a copy of an array with same parameters.

        Parameters
        ----------
        dtype: NumPy.dtype
            The new array dtype. Default `self.dtype`.

        Other Parameters
        ----------------
        kwargs: dict, optional
            Keyword arguments that are supported by the :func:`empty` constructor. If some
            are not specified, the default will be the ones from the original array (except for the urlpath).

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
        >>> # Create a NDArray  with default chunks
        >>> a = blosc2.zeros(shape, blocks=blocks, dtype=dtype)
        >>> # Get a copy with default chunks and blocks
        >>> b = a.copy(chunks=None, blocks=None)
        >>> assert np.array_equal(b[...], a[...])
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

        return super(NDArray, self).copy(dtype, **kwargs)

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

        Examples
        --------
        >>> import blosc2
        >>> import numpy as np
        >>> dtype = np.float32
        >>> shape = [23, 11]
        >>> a = np.linspace(1, 3, num=int(np.prod(shape))).reshape(shape)
        >>> # Create an array
        >>> b = blosc2.asarray(a, dtype=dtype)
        >>> newshape = [50, 10]
        >>> # Extend first dimension, shrink second dimension
        >>> _ = b.resize(newshape)
        >>> assert b.shape == tuple(newshape)
        """
        return super(NDArray, self).resize(newshape)

    def slice(self, key, **kwargs):
        """ Get a (multidimensional) slice as specified in key as a new :ref:`NDArray <NDArray>`.
        The dtype used will be the same as `self`.

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
        _check_ndarray_kwargs(**kwargs)
        key, mask = process_key(key, self.shape)
        start, stop, _ = get_ndarray_start_stop(self.ndim, key, self.shape)
        key = (start, stop)
        return super(NDArray, self).get_slice(key, mask, **kwargs)

    def squeeze(self):
        """Remove the 1's in array's shape."""
        super(NDArray, self).squeeze()


def _check_shape(shape):
    if type(shape) is int:
        shape = (shape,)
    if type(shape) not in (tuple, list):
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
        :obj:`SChunk.__init__ <blosc2.SChunk.SChunk.__init__>`.

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
    >>> assert array.shape == tuple(shape)
    >>> assert array.dtype == dtype
    """
    shape = _check_shape(shape)
    _check_ndarray_kwargs(**kwargs)
    chunks = kwargs.pop("chunks", None)
    blocks = kwargs.pop("blocks", None)
    chunks, blocks = compute_chunks_blocks(shape, chunks, blocks, dtype, **kwargs)
    arr = blosc2_ext.empty(shape, chunks, blocks, dtype, **kwargs)
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
    >>> shape = [25, 10]
    >>> chunks = [10, 10]
    >>> blocks = [5, 5]
    >>> dtype = np.float64
    >>> # Create zeros array
    >>> array = blosc2.zeros(shape, dtype=dtype, chunks=chunks, blocks=blocks)
    >>> assert array.shape == tuple(shape)
    >>> assert array.chunks == tuple(chunks)
    >>> assert array.blocks == tuple(blocks)
    >>> assert array.dtype == dtype
    >>> # Get array data as a NumPy array ?? posar-ho en la gertitem???
    >>> nparray = array[...]
    >>> assert nparray.shape == array.shape
    >>> assert nparray.dtype == array.dtype
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
         The ndarray dtype in NumPy format. By default this will
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
    >>> assert array.shape == tuple(shape)
    >>> assert array.dtype == np.bool_
    >>> # Get array data as a NumPy array
    >>> nparray = array[...]
    >>> assert nparray.shape == array.shape
    >>> assert nparray.dtype == array.dtype
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


def from_buffer(buffer, shape, dtype=np.dtype("|S1"), **kwargs):
    """Create an array out of a buffer.

    Parameters
    ----------
    buffer: bytes
        The buffer of the data to populate the container.
    shape: int, tuple or list
        The shape for the final container.
    dtype: np.dtype
        The ndarray dtype in NumPy format. Default is `|S1`.
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
    >>> a = blosc2.from_buffer(buffer, shape, chunks=chunks, dtype=dtype)
    >>> # Convert the array to a buffer
    >>> buffer2 = a.to_buffer()
    >>> assert buffer == buffer2
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


def asarray(array, dtype=np.uint8, **kwargs):
    """Convert the input to an array.

    Parameters
    ----------
    array: array_like
        An array supporting the python buffer protocol and the numpy array interface.
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
        An array interpretation of :paramref:`array`.

    Examples
    --------
    >>> import blosc2
    >>> import numpy as np
    >>> shape = [25, 10]
    >>> dtype = np.int64
    >>> # Create some data
    >>> nparray = np.arange(0, np.prod(shape), dtype=dtype)
    >>> # Create a NDArray from a NumPy array
    >>> a = blosc2.asarray(nparray, dtype)
    >>> # Convert the array to a buffer
    >>> buffer2 = a.to_buffer()
    >>> assert nparray.tobytes() == buffer2
    """
    _check_ndarray_kwargs(**kwargs)
    chunks = kwargs.pop("chunks", None)
    blocks = kwargs.pop("blocks", None)
    chunks, blocks = compute_chunks_blocks(array.shape, chunks, blocks, dtype, **kwargs)
    return blosc2_ext.asarray(array, chunks, blocks, dtype, **kwargs)


def _check_ndarray_kwargs(**kwargs):
    supported_keys = ["chunks", "blocks", "cparams", "dparams", "meta", "urlpath", "contiguous"]
    for key in kwargs.keys():
        if key not in supported_keys:
            raise KeyError(f"Only {str(supported_keys)} are supported as keyword arguments")
