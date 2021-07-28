import blosc2
from blosc2 import blosc2_ext


class vlmeta(blosc2_ext.vlmeta):
    def __init__(self, schunk):
        self.vlmeta = {}
        super(vlmeta, self).__init__(schunk)

    def __setitem__(self, name, content):
        self.vlmeta[name] = content
        cparams = {"typesize": 1}
        super(vlmeta, self).set_vlmeta(name, content, **cparams)

    def __getitem__(self, name):
        return self.vlmeta[name]

    def __delitem__(self, name):
        raise NotImplementedError
        #del self.vlmeta[name]

    def __len__(self):
        return len(self.vlmeta)

    def __contains__(self, name):
        return name in self.vlmeta

    def getall(self):
        return self.vlmeta.copy()


class SChunk(blosc2_ext.SChunk):
    def __init__(self, chunksize=8 * 10 ** 6, data=None, **kwargs):
        """Create a new super-chunk.

        If `data` is diferent than `None`, the `data` is split into
        chunks of size `chunksize` and these chunks are appended into the created SChunk.

        Parameters
        ----------
        chunksize: int
            The size, in bytes, of the chunks from the super-chunk. If the chunksize is not provided
            it is set to 8MB.

        data: bytes-like object, optional
            The data to be splitted into different chunks of size `chunksize`.

        Other parameters
        ----------------
        kwargs: dict, optional
            Keyword arguments supported:

                contiguous: bool
                    If the chunks are stored contiguously or not.
                urlpath: String
                    If the storage is persistent the name of the file (when `contiguous = True`) or
                    the directory (if `contiguous = False`).
                    If the storage is in-memory, then this field is `None`.
                cparams: dict
                    A dictionary with the compression parameters, which are the same that can be
                    used in the :func:`~blosc2.compress2` function.
                dparams: dict
                    A dictionary with the decompression parameters, which are the same that can be
                    used in the :func:`~blosc2.decompress2` function.

        Examples
        --------
        >>> storage = {"contiguous": True, "cparams": {}, "dparams": {}}
        >>> schunk = blosc2.SChunk(**storage)
        """
        super(SChunk, self).__init__(chunksize, data, **kwargs)
        self.vlmeta = vlmeta(super(SChunk, self).c_schunk)

    def append_data(self, data):
        """Append a data data to the SChunk.

        Tha data buffer must be of size `chunksize` specified in
        :func:`~blosc2.SChunk.__init__` .

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
            If the data could not be appended.

        Examples
        --------
        >>> import numpy
        >>> schunk = blosc2.SChunk(chunksize=200*1000*4)
        >>> data =  numpy.arange(200 * 1000, dtype='int32')
        >>> schunk.append_data(data)
        1
        """
        return super(SChunk, self).append_data(data)

    def decompress_chunk(self, nchunk, dst=None):
        """Decompress the chunk given by its index `nchunk`.

        Parameters
        ----------
        nchunk: int
            The index of the chunk that will be decompressed.
        dst: NumPy object or bytearray
            The destination NumPy object or bytearray to fill wich
            length must be greater than 0. The user must make sure
            that it has enough capacity for hosting the decompressed
            chunk. Default is None, meaning that a new bytes object
            is created, filled and returned.

        Returns
        -------
        out: str/bytes
            The decompressed chunk in form of a Python str / bytes object if
            `dst` is `None`. Otherwise, it will return `None` because the result
            will already be in `dst`.

        Raises
        ------
        RunTimeError
            If some problem was detected.

        Examples
        --------
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
        return super(SChunk, self).update_chunk(nchunk, chunk)

    def update_data(self, nchunk, data, copy):
        """Update the chunk in the `nchunk`-th position with the given data.

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
        return super(SChunk, self).update_data(nchunk, data, copy)

    def __dealloc__(self):
        super(SChunk, self).__dealloc__()
