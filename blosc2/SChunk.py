########################################################################
#
#       Author:  The Blosc development team - blosc@blosc.org
#
########################################################################

from collections.abc import MutableMapping

from msgpack import packb, unpackb

from blosc2 import blosc2_ext

# See https://github.com/dask/distributed/issues/3716#issuecomment-632913789
def encode_tuple(obj):
    if isinstance(obj, tuple):
        obj = ["__tuple__", *obj]
    return obj

def decode_tuple(obj):
    if obj[0] == "__tuple__":
        obj = tuple(obj[1:])
    return obj


class vlmeta(MutableMapping, blosc2_ext.vlmeta):
    def __init__(self, schunk, urlpath, mode):
        self.urlpath = urlpath
        self.mode = mode
        super(vlmeta, self).__init__(schunk)

    def __setitem__(self, name, content):
        blosc2_ext._check_access_mode(self.urlpath, self.mode)
        cparams = {"typesize": 1}
        content = packb(content, default=encode_tuple, strict_types=True, use_bin_type=True)
        super(vlmeta, self).set_vlmeta(name, content, **cparams)

    def __getitem__(self, name):
        return unpackb(super(vlmeta, self).get_vlmeta(name), list_hook=decode_tuple)

    def __delitem__(self, name):
        blosc2_ext._check_access_mode(self.urlpath, self.mode)
        super(vlmeta, self).del_vlmeta(name)

    def __len__(self):
        return super(vlmeta, self).nvlmetalayers()

    def __iter__(self):
        keys = super(vlmeta, self).get_names()
        for name in keys:
            yield name

    def getall(self):
        """
        Return all the variable length metalayers as a dictionary

        """
        return super(vlmeta, self).to_dict()


class SChunk(blosc2_ext.SChunk):
    def __init__(self, chunksize=2 ** 24, data=None, **kwargs):
        """Create a new super-chunk.

        Parameters
        ----------
        chunksize: int
            The size, in bytes, of the chunks from the super-chunk. If not provided,
            it is set to 16 MB.

        data: bytes-like object, optional
            The data to be split into different chunks of size :paramref:`chunksize`.
            If None, the Schunk instance will be empty initially.

        Other parameters
        ----------------
        kwargs: dict, optional
            Keyword arguments supported:

                contiguous: bool
                    If the chunks are stored contiguously or not.
                urlpath: String
                    If the storage is persistent, the name of the file (when `contiguous = True`) or
                    the directory (if `contiguous = False`).
                    If the storage is in-memory, then this field is `None`.
                mode: str, optional
                    Persistence mode: ‘r’ means read only (must exist);
                    ‘a’ means read/write (create if it doesn’t exist);
                    ‘w’ means create (overwrite if it exists).
                cparams: dict
                    A dictionary with the compression parameters, which are the same that can be
                    used in the :func:`~blosc2.compress2` function.
                dparams: dict
                    A dictionary with the decompression parameters, which are the same that can be
                    used in the :func:`~blosc2.decompress2` function.

        Examples
        --------
        >>> import blosc2
        >>> storage = {"contiguous": True, "cparams": {}, "dparams": {}}
        >>> schunk = blosc2.SChunk(**storage)
        """
        if kwargs is not None:
            # This a private param to get an SChunk from a blosc2_schunk*
            sc = kwargs.pop("schunk", None)
            self.urlpath = kwargs.get("urlpath", None)
        else:
            self.urlpath = None
            sc = None
        super(SChunk, self).__init__(schunk=sc, chunksize=chunksize, data=data, **kwargs)
        self.vlmeta = vlmeta(super(SChunk, self).c_schunk, self.urlpath, self.mode)

    def append_data(self, data):
        """Append a data buffer to the SChunk.

        The data buffer must be of size `chunksize` specified in
        :func:`SChunk.__init__ <blosc2.SChunk.SChunk.__init__>`.

        Parameters
        ----------
        data: bytes-like object
            The data to be compressed and added as a chunk.

        Returns
        -------
        out: int
            The number of chunks in the SChunk.

        Raises
        ------
        RunTimeError
            If :paramref:`data` could not be appended.

        Examples
        --------
        >>> import blosc2
        >>> import numpy
        >>> schunk = blosc2.SChunk(chunksize=200*1000*4)
        >>> data = numpy.arange(200 * 1000, dtype='int32')
        >>> schunk.append_data(data)
        1
        """
        blosc2_ext._check_access_mode(self.urlpath, self.mode)
        return super(SChunk, self).append_data(data)

    def decompress_chunk(self, nchunk, dst=None):
        """Decompress the chunk given by its index :paramref:`nchunk`.

        Parameters
        ----------
        nchunk: int
            The index of the chunk that will be decompressed.
        dst: NumPy object or bytearray
            The destination NumPy object or bytearray to fill, the length
            of which must be greater than 0. The user must make sure
            that it has enough capacity for hosting the decompressed
            chunk. Default is None, meaning that a new bytes object
            is created, filled and returned.

        Returns
        -------
        out: str/bytes
            The decompressed chunk in form of a Python str / bytes object if
            :paramref:`dst` is `None`. Otherwise, it will return `None` because the result
            will already be in :paramref:`dst`.

        Raises
        ------
        RunTimeError
            If some problem was detected.

        Examples
        --------
        >>> import blosc2
        >>> cparams = {'typesize': 1}
        >>> storage = {'cparams': cparams}
        >>> schunk = blosc2.SChunk(chunksize=11, **storage)
        >>> buffer = b"wermqeoir23"
        >>> schunk.append_data(buffer)
        1
        >>> schunk.decompress_chunk(0)
        b'wermqeoir23'
        >>> bytes_obj = bytearray(len(buffer))
        >>> schunk.decompress_chunk(0, dst=bytes_obj)
        >>> bytes_obj == buffer
        True
        """
        return super(SChunk, self).decompress_chunk(nchunk, dst)

    def get_chunk(self, nchunk):
        """Return the compressed chunk that is in the SChunk.

        Parameters
        ----------
        nchunk: int
            The chunk index that identifies the chunk that will be returned.

        Returns
        -------
        out: bytes object
            The compressed chunk.

        Raises
        ------
        RunTimeError
            If some problem is detected.
        """
        return super(SChunk, self).get_chunk(nchunk)

    def delete_chunk(self, nchunk):
        """Delete the specified chunk from the SChunk.

        Parameters
        ----------
        nchunk: int
            The index of the chunk that will be removed.

        Returns
        -------
        out: int
            The number of chunks in the SChunk.

        Raises
        ------
        RunTimeError
            If some problem was detected.
        """
        blosc2_ext._check_access_mode(self.urlpath, self.mode)
        return super(SChunk, self).delete_chunk(nchunk)

    def insert_chunk(self, nchunk, chunk):
        """Insert an already compressed chunk in the SChunk.

        Parameters
        ----------
        nchunk: int
            The position in which the chunk will be inserted.
        chunk: bytes object
            The compressed chunk.

        Returns
        -------
        out: int
            The number of chunks in the SChunk.

        Raises
        ------
        RunTimeError
            If some problem was detected.
        """
        blosc2_ext._check_access_mode(self.urlpath, self.mode)
        return super(SChunk, self).insert_chunk(nchunk, chunk)

    def insert_data(self, nchunk, data, copy):
        """Insert the data in the specified position in the SChunk.

        Parameters
        ----------
        nchunk: int
            The position in which the chunk will be inserted.
        data: bytes object
            The data that will be compressed and inserted as a chunk.
        copy: bool
            Whether to internally do a copy of the chunk to insert it or not.

        Returns
        -------
        out: int
            The number of chunks in the SChunk.

        Raises
        ------
        RunTimeError
            If some problem was detected.
        """
        blosc2_ext._check_access_mode(self.urlpath, self.mode)
        return super(SChunk, self).insert_data(nchunk, data, copy)

    def update_chunk(self, nchunk, chunk):
        """Update an existing chunk in the SChunk.

        Parameters
        ----------
        nchunk: int
            The position identifying the chunk that will be updated.
        chunk: bytes object
            The new compressed chunk that will replace the content of the old one.

        Returns
        -------
        out: int
            The number of chunks in the SChunk.

        Raises
        ------
        RunTimeError
            If some problem was detected.
        """
        blosc2_ext._check_access_mode(self.urlpath, self.mode)
        return super(SChunk, self).update_chunk(nchunk, chunk)

    def update_data(self, nchunk, data, copy):
        """Update the chunk in the :paramref:`nchunk`-th position with the given data.

        Parameters
        ----------
        nchunk: int
            The position identifying the chunk that will be updated.
        data: bytes object
            The data that will be compressed and will replace the content of the old one.
        copy: bool
            Whether to internally do a copy of the chunk to update it or not.

        Returns
        -------
        out: int
            The number of chunks in the SChunk.

        Raises
        ------
        RunTimeError
            If some problem was detected.
        """
        blosc2_ext._check_access_mode(self.urlpath, self.mode)
        return super(SChunk, self).update_data(nchunk, data, copy)

    def get_slice(self, start=0, stop=None, out=None):
        """Get a slice from :paramref:`start` to :paramref:`stop`.

        Parameters
        ----------
        start: int
            The index where the slice will begin. Default is 0.
        stop: int
            The index where the slice will end (without including it).
            Default is until the SChunk ends.
        out: bytes-like object or bytearray
            The destination object (supporting the
            `Buffer Protocol <https://docs.python.org/3/c-api/buffer.html>`_) to fill.
            The user must make sure
            that it has enough capacity for hosting the decompressed
            data. Default is None, meaning that a new bytes object
            is created, filled and returned.

        Returns
        -------
        out: str/bytes or None
            The decompressed slice in form of a Python str / bytes object if
            :paramref:`out` is `None`. Otherwise, it will return `None` as the result
            will already be in :paramref:`out`.

        Raises
        ------
        ValueError
            If the size to get is negative.
            If there is not enough space in :paramref:`out`.
            If :paramref:`start` is greater or equal than the SChunk nitems
        RunTimeError
            If some problem was detected.

        See Also
        --------
        :func:`__getitem__`

        """
        return super(SChunk, self).get_slice(start, stop, out)

    def __getitem__(self, item):
        """ Get a slice from the SChunk.

        Parameters
        ----------
        item: int or slice
            The index for the slice. Note that the step parameter is not honored.

        Returns
        -------
        out: str/bytes
            The decompressed slice in form of a Python str / bytes object.

        Raises
        ------
        ValueError
            If the size to get is negative.
            If :paramref:`item`.start is greater or equal than the SChunk nitems
        RunTimeError
            If some problem was detected.
        IndexError
            If `step` is not 1.

        See Also
        --------
        :func:`get_slice`

        """
        if item.step is not None and item.step != 1:
            raise IndexError("`step` must be 1")
        return self.get_slice(item.start, item.stop)

    def __setitem__(self, key, value):
        """Set slice to :paramref:`value`.

        Parameters
        ----------
        key: int or slice
            The index of the slice to update. Note that step parameter is not honored.
        value: bytes-like object
            An object supporting the
            `Buffer Protocol <https://docs.python.org/3/c-api/buffer.html>`_ used to overwrite the slice.

        Returns
        -------
        out: None

        Raises
        ------
        ValueError
            If you cannot modify :paramref:`self`.
            If the size to get is negative.
            If there is not enough space in :paramref:`value` to update the slice.
            If :paramref:`start` is greater than the SChunk nitems
        RunTimeError
            If some problem was detected.
        IndexError
            If `step` is not 1.

        Notes
        -----
        This method can also be used to append new data if :paramref:`key`.stop
        is greater than the SChunk nitems.

        """
        if key.step is not None and key.step != 1:
            raise IndexError("`step` must be 1")
        blosc2_ext._check_access_mode(self.urlpath, self.mode)
        return super(SChunk, self).set_slice(start=key.start, stop=key.stop, value=value)

    def to_cframe(self):
        """ Get a bytes object containing the serialized :class:`~blosc2.SChunk` instance.

        Returns
        -------
        out: bytes
            The buffer containing the serialized :class:`~blosc2.SChunk` instance.

        See Also
        --------
        :func:`~blosc2.schunk_from_cframe`

        """
        return super(SChunk, self).to_cframe()

    def __dealloc__(self):
        super(SChunk, self).__dealloc__()


def open(urlpath, mode="a", **kwargs):
    """Open an already persistently stored :class:`~blosc2.SChunk`.

    Parameters
    ----------
    urlpath: str
        The path where the :class:`~blosc2.SChunk` is stored.
    mode: str, optional
        The open mode.

    Other parameters
    ----------------
    kwargs: dict, optional
            Keyword arguments supported:
                cparams: dict
                    A dictionary with the compression parameters, which are the same that can be
                    used in the :func:`~blosc2.compress2` function. Typesize and blocksize cannot
                    be changed.
                dparams: dict
                    A dictionary with the decompression parameters, which are the same that can be
                    used in the :func:`~blosc2.decompress2` function.

    Examples
    --------
    >>> import blosc2
    >>> import numpy
    >>> storage = {"contiguous": True, "urlpath": "b2frame", "cparams": {}, "dparams": {}}
    >>> nelem = 20 * 1000
    >>> nchunks = 5
    >>> chunksize = nelem * 4 // nchunks
    >>> data = numpy.arange(nelem, dtype="int32")
    >>> # Create SChunk and append data
    >>> schunk = blosc2.SChunk(chunksize=chunksize, data=data.tobytes(), mode="w", **storage)
    >>> # Open SChunk
    >>> sc_open = blosc2.open(urlpath=storage["urlpath"])
    >>> for i in range(nchunks):
    ...     dest = numpy.empty(nelem // nchunks, dtype=data.dtype)
    ...     schunk.decompress_chunk(i, dest)
    ...     dest1 = numpy.empty(nelem // nchunks, dtype=data.dtype)
    ...     sc_open.decompress_chunk(i, dest1)
    ...     numpy.array_equal(dest, dest1)
    True
    True
    True
    True
    True
    """
    return blosc2_ext.schunk_open(urlpath, mode, **kwargs)
