from blosc2 import blosc2_ext

class SChunk(blosc2_ext.SChunk):
    def __init__(self, **kwargs):
        """Create a new SuperChunk

        Other parameters
        ----------------
        kwargs: dict
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
        """
        super(SChunk, self).__init__(**kwargs)

    def append_buffer(self, data):
        """Append a data buffer to the SChunk.

        Parameters
        ----------
        data: The buffer of data to compress and add as a chunk.

        Returns
        -------
        out: int
            The number of chunks in the SChunk.

        Raises
        ------
        RunTimeError
            If the data could not be appended.
        """
        return super(SChunk, self).append_buffer(data)

    def decompress_chunk(self, nchunk, dst=None):
        """Decompress the chunk given by the its index `nchunk`.

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
            The chunk index that will be removed.

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

    def insert_buffer(self, nchunk, buffer, copy):
        """Insert the buffer in the specified position in the SChunk.

        Parameters
        ----------
        nchunk: int
            The position in which the chunk will be inserted.
        buffer: bytes object
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
        return super(SChunk, self).insert_buffer(nchunk, buffer, copy)

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

    def update_buffer(self, nchunk, buffer, copy):
        """Update the chunk in the `nchunk`-th position with the given buffer.

        Parameters
        ----------
        nchunk: int
            The position identifying the chunk that will be updated.
        buffer: bytes object
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
        return super(SChunk, self).update_buffer(nchunk, buffer, copy)

    def __dealloc__(self):
        super(SChunk, self).__dealloc__()


