#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

import os
import pathlib
from collections import namedtuple
from collections.abc import Mapping, MutableMapping

import numpy as np
from msgpack import packb, unpackb

import blosc2
from blosc2 import SpecialValue, blosc2_ext
from blosc2.helpers import _inherit_doc_parameter


class vlmeta(MutableMapping, blosc2_ext.vlmeta):
    def __init__(self, schunk, urlpath, mode, mmap_mode, initial_mapping_size):
        self.urlpath = urlpath
        self.mode = mode
        self.mmap_mode = mmap_mode
        self.initial_mapping_size = initial_mapping_size
        super().__init__(schunk)

    def __setitem__(self, name, content):
        blosc2_ext.check_access_mode(self.urlpath, self.mode)
        cparams = {"typesize": 1}
        content = packb(
            content,
            default=blosc2_ext.encode_tuple,
            strict_types=True,
            use_bin_type=True,
        )
        super().set_vlmeta(name, content, **cparams)

    def __getitem__(self, name):
        return unpackb(super().get_vlmeta(name), list_hook=blosc2_ext.decode_tuple)

    def __delitem__(self, name):
        blosc2_ext.check_access_mode(self.urlpath, self.mode)
        super().del_vlmeta(name)

    def __len__(self):
        return super().nvlmetalayers()

    def __iter__(self):
        keys = super().get_names()
        yield from keys

    def getall(self):
        """
        Return all the variable length metalayers as a dictionary

        """
        return super().to_dict()


class Meta(Mapping):
    """
    Class providing access to user meta on a :ref:`SChunk`.
    It will be available via the `.meta` property of a :ref:`SChunk`.
    """

    def get(self, key, default=None):
        """Return the value for `key` if `key` is in the dictionary, else `default`.
        If `default` is not given, it defaults to ``None``."""
        return self.get(key, default)

    def __del__(self):
        pass

    def __init__(self, schunk):
        self.schunk = schunk

    def __contains__(self, key):
        """Check if the `key` metalayer exists or not."""
        return blosc2_ext.meta__contains__(self.schunk, key)

    def __delitem__(self, key):
        return None

    def __setitem__(self, key, value):
        """Update the `key` metalayer with `value`.

        Parameters
        ----------
        key: str
            The name of the metalayer to update.
        value: bytes
            The buffer containing the new content for the metalayer.

            ..warning: Note that the *length* of the metalayer cannot not change,
            else an exception will be raised.
        """
        value = packb(value, default=blosc2_ext.encode_tuple, strict_types=True, use_bin_type=True)
        return blosc2_ext.meta__setitem__(self.schunk, key, value)

    def __getitem__(self, item):
        """Return the `item` metalayer.

        Parameters
        ----------
        item: str
            The name of the metalayer to return.

        Returns
        -------
        bytes
            The buffer containing the metalayer info.
        """
        if self.__contains__(item):
            return unpackb(
                blosc2_ext.meta__getitem__(self.schunk, item),
                list_hook=blosc2_ext.decode_tuple,
            )
        else:
            raise KeyError(f"{item} not found")

    def keys(self):
        """Return the metalayers keys."""
        return blosc2_ext.meta_keys(self.schunk)

    def values(self):
        raise NotImplementedError("Values can not be accessed")

    def items(self):
        raise NotImplementedError("Items can not be accessed")

    def __iter__(self):
        """Iter over the keys of the metalayers."""
        return iter(self.keys())

    def __len__(self):
        """Return the number of metalayers."""
        return blosc2_ext.meta__len__(self.schunk)


class SChunk(blosc2_ext.SChunk):
    def __init__(self, chunksize=None, data=None, **kwargs):
        """Create a new super-chunk, or open an existing one.

        Parameters
        ----------
        chunksize: int, optional
            The size, in bytes, of the chunks from the super-chunk. If not provided,
            it is set automatically to a reasonable value.

        data: bytes-like object, optional
            The data to be split into different chunks of size :paramref:`chunksize`.
            If None, the Schunk instance will be empty initially.

        Other parameters
        ----------------
        kwargs: dict, optional
            Keyword arguments supported:

                contiguous: bool, optional
                    If the chunks are stored contiguously or not.
                    Default is True when :paramref:`urlpath` is not None;
                    False otherwise.
                urlpath: str | pathlib.Path, optional
                    If the storage is persistent, the name of the file (when
                    `contiguous = True`) or the directory (if `contiguous = False`).
                    If the storage is in-memory, then this field is `None`.
                mode: str, optional
                    Persistence mode: ‘r’ means read only (must exist);
                    ‘a’ means read/write (create if it doesn’t exist);
                    ‘w’ means create (overwrite if it exists).
                mmap_mode: str, optional
                    If set, the file will be memory-mapped instead of using the default
                    I/O functions and the `mode` argument will be ignored. The memory-mapping
                    modes are similar as used by the
                    `numpy.memmap <https://numpy.org/doc/stable/reference/generated/numpy.memmap.html>`_
                    function, but it is possible to extend the file:

                    .. list-table::
                        :widths: 10 90
                        :header-rows: 1

                        * - mode
                          - description
                        * - 'r'
                          - Open an existing file for reading only.
                        * - 'r+'
                          - Open an existing file for reading and writing. Use this mode if you want
                            to append data to an existing schunk file.
                        * - 'w+'
                          - Create or overwrite an existing file for reading and writing. Use this
                            mode if you want to create a new schunk.
                        * - 'c'
                          - Open an existing file in copy-on-write mode: all changes affect the data
                            in memory but changes are not saved to disk. The file on disk is
                            read-only. On Windows, the size of the mapping cannot change.

                    Only contiguous storage can be memory-mapped. Hence, `urlpath` must point to a
                    file (and not a directory).

                    .. note::
                        Memory-mapped files are opened once and the file contents remain in (virtual)
                        memory for the lifetime of the schunk. Using memory-mapped I/O can be faster
                        than using the default I/O functions depending on the use case. Whereas
                        reading performance is generally better, writing performance may also be
                        slower in some cases on certain systems. In any case, memory-mapped files
                        can be especially beneficial when operating with network file systems
                        (like NFS).

                        This is currently a beta feature (especially write operations) and we
                        recommend trying it out and reporting any issues you may encounter.

                initial_mapping_size: int, optional
                    The initial size of the mapping for the memory-mapped file when writes are
                    allowed (r+ w+, or c mode). Once a file is memory-mapped and extended beyond the
                    initial mapping size, the file must be remapped which may be expensive. This
                    parameter allows to decouple the mapping size from the actual file size to early
                    reserve memory for future writes and avoid remappings. The memory is only
                    reserved virtually and does not occupy physical memory unless actual writes
                    happen. Since the virtual address space is large enough, it is ok to be generous
                    with this parameter (with special consideration on Windows, see note below).
                    For best performance, set this to the maximum expected size of the compressed
                    data (see example in :obj:`SChunk.__init__ <blosc2.schunk.SChunk.__init__>`).
                    The size is in bytes.

                    Default: 1 GiB.

                    .. note::
                        On Windows, the size of the mapping is directly coupled to the file size.
                        When the schunk gets destroyed, the file size will be truncated to the
                        actual size of the schunk.

                cparams: dict
                    A dictionary with the compression parameters, which are the same
                    as those can be used in the :func:`~blosc2.compress2` function.
                dparams: dict
                    A dictionary with the decompression parameters, which are the same
                    as those that can be used in the :func:`~blosc2.decompress2`
                    function.
                meta: dict or None
                    A dictionary with different metalayers.  One entry per metalayer:

                        key: bytes or str
                            The name of the metalayer.
                        value: object
                            The metalayer object that will be serialized using msgpack.

        Examples
        --------
        >>> import blosc2
        >>> storage = {"contiguous": True, "cparams": {}, "dparams": {}}
        >>> schunk = blosc2.SChunk(**storage)

        In the following, we will write and read a super-chunk to and from disk
        via memory-mapped files.

        >>> a = np.arange(3, dtype=np.int64)
        >>> chunksize = a.size * a.itemsize
        >>> n_chunks = 2
        >>> urlpath = getfixture('tmp_path') / "schunk.b2frame"

        Optional: we intend to write 2 chunks of 24 bytes each, and we expect
        the compressed size to be smaller than the original size. Hence, we
        (generously) set the initial size of the mapping to 48 bytes
        effectively avoiding remappings.

        >>> initial_mapping_size = chunksize * n_chunks
        >>> schunk_mmap = blosc2.SChunk(
        ...     chunksize=chunksize,
        ...     mmap_mode="w+",
        ...     initial_mapping_size=initial_mapping_size,
        ...     urlpath=urlpath,
        ... )
        >>> schunk_mmap.append_data(a)
        1
        >>> schunk_mmap.append_data(a * 2)
        2

        Optional: explicitly close the file and free the mapping.

        >>> del schunk_mmap

        Reading the data back again via memory-mapped files:

        >>> schunk_mmap = blosc2.open(urlpath, mmap_mode="r")
        >>> np.frombuffer(schunk_mmap.decompress_chunk(0), dtype=np.int64).tolist()
        [0, 1, 2]
        >>> np.frombuffer(schunk_mmap.decompress_chunk(1), dtype=np.int64).tolist()
        [0, 2, 4]
        """
        # Check only allowed kwarg are passed
        allowed_kwargs = [
            "urlpath",
            "contiguous",
            "cparams",
            "dparams",
            "_schunk",
            "meta",
            "mode",
            "mmap_mode",
            "initial_mapping_size",
            "_is_view",
        ]
        for kwarg in kwargs:
            if kwarg not in allowed_kwargs:
                raise ValueError(f"{kwarg} is not supported as keyword argument")
        urlpath = kwargs.get("urlpath")
        if "contiguous" not in kwargs:
            # Make contiguous true for disk, else sparse (for in-memory performance)
            kwargs["contiguous"] = urlpath is not None

        # This a private param to get an SChunk from a blosc2_schunk*
        sc = kwargs.pop("_schunk", None)

        # If not passed, set a sensible typesize
        itemsize = data.itemsize if data is not None and hasattr(data, "itemsize") else 1
        if "cparams" in kwargs:
            if "typesize" not in kwargs["cparams"]:
                cparams = kwargs.pop("cparams").copy()
                cparams["typesize"] = itemsize
                kwargs["cparams"] = cparams
        else:
            kwargs["cparams"] = {"typesize": itemsize}

        # chunksize handling
        if chunksize is None:
            chunksize = 2**24
            if data is not None:
                if hasattr(data, "itemsize"):
                    chunksize = data.size * data.itemsize
                    # Make that a multiple of typesize
                    chunksize = chunksize // data.itemsize * data.itemsize
                else:
                    chunksize = len(data)
            # Use a cap of 256 MB (modern boxes should all have this RAM available)
            if chunksize > 2**28:
                chunksize = 2**28

        super().__init__(_schunk=sc, chunksize=chunksize, data=data, **kwargs)
        self.vlmeta = vlmeta(
            super().c_schunk, self.urlpath, self.mode, self.mmap_mode, self.initial_mapping_size
        )
        self._cparams = super().get_cparams()
        self._dparams = super().get_dparams()

    @property
    def cparams(self):
        """
        Dictionary with the compression parameters.
        """
        return self._cparams

    @cparams.setter
    def cparams(self, value):
        super().update_cparams(value)
        self._cparams = super().get_cparams()

    @property
    def dparams(self):
        """
        Dictionary with the decompression parameters.
        """
        return self._dparams

    @dparams.setter
    def dparams(self, value):
        super().update_dparams(value)
        self._dparams = super().get_dparams()

    @property
    def meta(self):
        return Meta(self)

    def append_data(self, data):
        """Append a data buffer to the SChunk.

        The data buffer must be of size `chunksize` specified in
        :func:`SChunk.__init__ <blosc2.schunk.SChunk.__init__>`.

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
        >>> import numpy as np
        >>> schunk = blosc2.SChunk(chunksize=200*1000*4)
        >>> data = np.arange(200 * 1000, dtype='int32')
        >>> print(schunk.append_data(data))
        1
        """
        blosc2_ext.check_access_mode(self.urlpath, self.mode)
        return super().append_data(data)

    def fill_special(self, nitems, special_value):
        """Fill the SChunk with a special value.  SChunk must be empty.

        Parameters
        ----------
        nitems: int
            The number of items to fill with the special value.
        special_value: SpecialValue
            The special value to be used for filling the SChunk.

        Returns
        -------
        out: int
            The number of chunks in the SChunk.

        Raises
        ------
        RunTimeError
            If the SChunk could not be filled with the special value.

        Examples
        --------
        >>> import blosc2
        >>> import numpy as np
        >>> schunk = blosc2.SChunk(chunksize=200*1000*4)
        >>> # Fill the SChunk with the special value
        >>> nitems = 200 * 1000
        >>> print(f"Initial number of chunks: {len(schunk)}")
        Initial number of chunks: 0
        >>> special_value = blosc2.SpecialValue.ZERO
        >>> # Fill the SChunk with the special value
        >>> nchunks = schunk.fill_special(nitems, special_value)
        >>> print(f"Number of chunks filled: {nchunks}")
        Number of chunks filled: 1
        >>> print(f"Number of chunks after fill_special: {len(schunk)}")
        Number of chunks after fill_special: 200000
        """
        if not isinstance(special_value, SpecialValue):
            raise TypeError("special_value must be a SpecialValue instance")
        nchunks = super().fill_special(nitems, special_value.value)
        if nchunks < 0:
            raise RuntimeError("Unable to fill with special values")
        return nchunks

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
            :paramref:`dst` is `None`. Otherwise, it will return `None` because the
            result will already be in :paramref:`dst`.

        Raises
        ------
        RunTimeError
            If some problem was detected.

        Examples
        --------
        >>> import blosc2
        >>> schunk = blosc2.SChunk(chunksize=11, cparams={'typesize': 1})
        >>> buffer = b"wermqeoir23"
        >>> print(schunk.append_data(buffer))
        1
        >>> print(schunk.decompress_chunk(0))
        b'wermqeoir23'
        >>> # Construct a mutable bytearray object
        >>> bytes_obj = bytearray(len(buffer))
        >>> schunk.decompress_chunk(0, dst=bytes_obj)
        >>> print(bytes_obj == buffer)
        True
        """
        return super().decompress_chunk(nchunk, dst)

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

        Examples
        --------
        >>> import blosc2
        >>> import numpy as np
        >>> # Create an SChunk with 3 chunks
        >>> nchunks = 3
        >>> data = np.arange(200 * 1000 * nchunks, dtype=np.int32)
        >>> schunk = blosc2.SChunk(chunksize=200 * 1000 * 4, data=data, cparams={"typesize": 4})
        >>> # Retrieve the first chunk (index 0)
        >>> chunk = schunk.get_chunk(0)
        >>> # Check the type and length of the compressed chunk
        >>> print(type(chunk))
        <class 'bytes'>
        >>> print(len(chunk) > 0)
        True
        """
        return super().get_chunk(nchunk)

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

        Examples
        --------
        >>> import blosc2
        >>> import numpy as np
        >>> # Create an SChunk with 3 chunks
        >>> nchunks = 3
        >>> data = np.arange(200 * 1000 * nchunks, dtype=np.int32)
        >>> schunk = blosc2.SChunk(chunksize=200 * 1000 * 4, data=data, cparams={"typesize": 4})
        >>> # Check the number of chunks before deletion
        >>> print(schunk.nchunks)
        3
        >>>  # Delete the second chunk (index 1)
        >>> schunk.delete_chunk(1)
        >>>  # Check the number of chunks after deletion
        >>> print(schunk.nchunks)
        2
        """
        
        blosc2_ext.check_access_mode(self.urlpath, self.mode)
        return super().delete_chunk(nchunk)

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

        Examples
        --------
        >>> import blosc2
        >>> import numpy as np
        >>> # Create an SChunk with 2 chunks
        >>> data = np.arange(400 * 1000, dtype=np.int32)
        >>> schunk = blosc2.SChunk(chunksize=200*1000*4, data=data, cparams={"typesize": 4})
        >>> # Get a compressed chunk from the SChunk
        >>> chunk = schunk.get_chunk(0)
        >>> # Insert the chunk at a different position (in this case, at index 1)
        >>> schunk.insert_chunk(1, chunk)
        >>> # Verify the total number of chunks after insertion
        >>> print(schunk.nchunks)
        3
        """
        blosc2_ext.check_access_mode(self.urlpath, self.mode)
        return super().insert_chunk(nchunk, chunk)

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

        Examples
        --------
        >>> import blosc2
        >>> import numpy as np
        >>> # Create an SChunk with 2 chunks
        >>> data = np.arange(400 * 1000, dtype=np.int32)
        >>> schunk = blosc2.SChunk(chunksize=200*1000*4, data=data, cparams={"typesize": 4})
        >>> # New data; create a NumPy array containing 200,000 sequential integers, starting from 0 up to 199,999.
        >>> # Each element in the array is of type int32.
        >>> new_data = np.arange(200 * 1000, dtype=np.int32)
        >>> # Insert the new data at position 1, compressing it
        >>> schunk.insert_data(1, new_data, copy=True)
        >>> # Verify the total number of chunks after insertion
        >>> print(schunk.nchunks)
        3
        """
        blosc2_ext.check_access_mode(self.urlpath, self.mode)
        return super().insert_data(nchunk, data, copy)

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

        Examples
        --------
        >>> import blosc2
        >>> import numpy as np
        >>> nchunks = 5
        >>> chunk_size = 200 * 1000 * 4
        >>> data = np.arange(nchunks * chunk_size // 4, dtype=np.int32)
        >>> schunk = blosc2.SChunk(chunksize=chunk_size, data=data, cparams={"typesize": 4})
        >>> initial_nchunks = schunk.nchunks
        >>> print(f"Initial number of chunks: {initial_nchunks}")
        Initial number of chunks: 5
        >>> chunk_index = 1
        >>> new_data = np.full(chunk_size // 4, fill_value=chunk_index, dtype=np.int32).tobytes()
        >>> compressed_data = blosc2.compress(new_data, typesize=4)
        >>> # Update the 2nd chunk (index 1) with new data
        >>> nchunks = schunk.update_chunk(chunk_index, compressed_data)
        >>> print(f"Number of chunks after update: {nchunks}")
        Number of chunks after update: 5
        """
        blosc2_ext.check_access_mode(self.urlpath, self.mode)
        return super().update_chunk(nchunk, chunk)

    def update_data(self, nchunk, data, copy):
        """Update the chunk in the :paramref:`nchunk`-th position with the given data.

        Parameters
        ----------
        nchunk: int
            The position identifying the chunk that will be updated.
        data: bytes object
            The data that will be compressed and will replace the old one.
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

        Examples
        --------
        >>> import blosc2
        >>> import numpy as np
        >>> nchunks = 4
        >>> chunk_size = 200 * 1000 * 4
        >>> data = np.arange(nchunks * chunk_size // 4, dtype=np.int32)
        >>> schunk = blosc2.SChunk(chunksize=chunk_size, data=data, cparams={"typesize": 4})
        >>> initial_nchunks = schunk.nchunks
        >>> print(f"Initial number of chunks: {initial_nchunks}")
        Initial number of chunks: 4
        >>> chunk_index = 1 # Update the 2nd chunk (index 1)
        >>> new_data = np.full(chunk_size // 4, fill_value=chunk_index, dtype=np.int32).tobytes()
        >>> nchunks = schunk.update_data(chunk_index, new_data, copy=True)
        >>> final_nchunks = schunk.nchunks
        >>> print(f"Number of chunks after update: {final_nchunks}")
        Number of chunks after update: 4
        """
        blosc2_ext.check_access_mode(self.urlpath, self.mode)
        return super().update_data(nchunk, data, copy)

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

        Examples
        --------
        >>> import blosc2
        >>> import numpy as np
        >>> nchunks = 4
        >>> chunk_size = 200 * 1000 * 4
        >>> data = np.arange(nchunks * chunk_size // 4, dtype=np.int32)
        >>> schunk = blosc2.SChunk(chunksize=chunk_size, data=data, cparams={"typesize": 4})
        >>> # Define the slice parameters
        >>> start_index = 200 * 1000
        >>> stop_index = 2 * 200 * 1000
        >>> # Prepare an output buffer
        >>> slice_size = stop_index - start_index
        >>> out_buffer = bytearray(slice_size * 4)  # Ensure the buffer is large enough
        >>> result = schunk.get_slice(start=start_index, stop=stop_index, out=out_buffer)
        >>> # Check if `result` is None and if the output buffer was filled
        >>> if result is None:
        >>>     print(f"Data slice obtained successfully. Length of slice: {len(out_buffer)}")
        Data slice obtained successfully. Length of slice: 400000
        >>>     # Convert bytearray to NumPy array for easier inspection
        >>>     slice_array = np.frombuffer(out_buffer, dtype=np.int32)
        >>>     print(f"Slice data: {slice_array[:10]} ...")  # Print the first 10 elements
        Slice data: [100000 100001 100002 100003 100004 100005 100006 100007 100008 100009] ...
        >>> else:
        >>>     print("Data slice obtained successfully.")
        """
        return super().get_slice(start, stop, out)

    def __getitem__(self, item):
        """Get a slice from the SChunk.

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

        Examples
        --------
        >>> import blosc2
        >>> import numpy as np
        >>> nchunks = 4
        >>> chunk_size = 200 * 1000 * 4
        >>> data = np.arange(nchunks * chunk_size // 4, dtype=np.int32)
        >>> schunk = blosc2.SChunk(chunksize=chunk_size, data=data, cparams={"typesize": 4})
        >>> # Define a slice of the data array that we want to compare with the result from SChunk
        >>> sl = data[150:155]
        >>> # Use the get_slice method of SChunk to get the data from the same slice range
        >>> res = schunk.get_slice(150, 155)
        >>> # Check if the retrieved slice from SChunk matches the original slice
        >>> # Convert the original slice to bytes and compare with the result from SChunk
        >>> assert res == sl.tobytes()
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
            `Buffer Protocol <https://docs.python.org/3/c-api/buffer.html>`_ used to
            fill the slice.

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

        Examples
        --------
        >>> import blosc2
        >>> import numpy as np
        >>> nchunks = 4
        >>> chunk_size = 200 * 1000 * 4
        >>> data = np.arange(nchunks * chunk_size // 4, dtype=np.int32)
        >>> schunk = blosc2.SChunk(chunksize=chunk_size, data=data, cparams={"typesize": 4})
        >>> # Create a new array of values to update the slice (values from 1000 to 1999 multiplied by 2)
        >>> start_ = 1000
        >>> stop = 2000
        >>> new_values = np.arange(start_, stop, dtype=np.int32) * 2
        >>> schunk.__setitem__(slice(start_, stop), new_values)
        >>> sl = schunk[start_:stop]
        >>> res = schunk.get_slice(start_, stop)
        >>> assert res == sl
        >>> print("The slice comparison is successful!")
        The slice comparison is successful!
        """
        if key.step is not None and key.step != 1:
            raise IndexError("`step` must be 1")
        blosc2_ext.check_access_mode(self.urlpath, self.mode)
        return super().set_slice(start=key.start, stop=key.stop, value=value)

    def to_cframe(self):
        """Get a bytes object containing the serialized :ref:`SChunk` instance.

        Returns
        -------
        out: bytes
            The buffer containing the serialized :ref:`SChunk` instance.

        See Also
        --------
        :func:`~blosc2.schunk_from_cframe`

        Examples
        --------
        >>> import blosc2
        >>> import numpy as np
        >>> nchunks = 4
        >>> chunk_size = 200 * 1000 * 4
        >>> data = np.arange(nchunks * chunk_size // 4, dtype=np.int32)
        >>> schunk = blosc2.SChunk(chunksize=chunk_size, data=data, cparams={"typesize": 4})
        >>> # Serialize the SChunk instance to a bytes object
        >>> serialized_schunk = schunk.to_cframe()
        >>> print(f"Serialized SChunk length: {len(serialized_schunk)} bytes")
        Serialized SChunk length: 15545 bytes
        >>> # Create a new SChunk from the serialized data
        >>> deserialized_schunk = blosc2.schunk_from_cframe(serialized_schunk)
        >>> # Print a slice of the deserialized SChunk to verify
        >>> start = 1000
        >>> stop = 1005
        >>> sl = deserialized_schunk[start:stop]
        >>> res = schunk.get_slice(start, stop)
        >>> assert res == sl
        """
        return super().to_cframe()

    def iterchunks(self, dtype):
        """
        Iterate over :paramref:`self` chunks.

        Parameters
        ----------
        dtype: np.dtype
            The data type to use.

        Yields
        ------
        chunk: NumPy ndarray
           The decompressed chunk.

        Examples
        --------
        >>> import blosc2
        >>> import numpy as np
        >>> # Create sample data and an SChunk
        >>> nchunks = 2     # Total data for 2 chunks
        >>> data = np.arange(400 * 1000, dtype=np.int32)
        >>> schunk = blosc2.SChunk(chunksize=200*1000*4, data=data, cparams={"typesize": 4})
        >>> # Iterate over chunks using the iterchunks method
        >>> for chunk in schunk.iterchunks(dtype=np.int32):
        >>>     print("Chunk shape:", chunk.shape)
        >>>     print("First 5 elements of chunk:", chunk[:5])
        Chunk shape: (200000,)
        First 5 elements of chunk: [0 1 2 3 4]
        Chunk shape: (200000,)
        First 5 elements of chunk: [200000 200001 200002 200003 200004]
        """
        out = np.empty(self.chunkshape, dtype)
        for i in range(0, len(self), self.chunkshape):
            self.get_slice(i, i + self.chunkshape, out)
            yield out

    def iterchunks_info(self):
        """
        Iterate over :paramref:`self` chunks, providing info on index and special values.

        Yields
        ------
        info: namedtuple
            A namedtuple with the following fields:

                nchunk: int
                    The index of the chunk.
                cratio: float
                    The compression ratio of the chunk.
                special: :class:`~blosc2.SpecialValue`
                    The special value enum of the chunk; if 0, the chunk is not special.
                repeated_value: bytes or None
                    The repeated value for the chunk; if not SpecialValue.VALUE, it is None.
                lazychunk: bytes
                    A buffer with the complete lazy chunk.

        Examples
        --------
        >>> import blosc2
        >>> import numpy as np
        >>> # Create sample data and an SChunk
        >>> data = np.arange(400 * 1000, dtype=np.int32)
        >>> nchunks = 2 # Total data for 2 chunks
        >>> schunk = blosc2.SChunk(chunksize=200*1000*4, data=data, cparams={"typesize": 4})
        >>> # Iterate over chunks and print detailed information
        >>> for chunk_info in schunk.iterchunks_info():
        >>>     print(f"Chunk index: {chunk_info.nchunk}")
        >>>     print(f"Compression ratio: {chunk_info.cratio:.2f}")
        >>>     print(f"Special value: {chunk_info.special.name}")
        >>>     print(f"Repeated value: {chunk_info.repeated_value[:10] if chunk_info.repeated_value else None}")
        Chunk index: 0
        Compression ratio: 213.79
        Special value: NOT_SPECIAL
        Repeated value: None
        Chunk index: 1
        Compression ratio: 206.88
        Special value: NOT_SPECIAL
        Repeated value: None
        """
        ChunkInfo = namedtuple("ChunkInfo", ["nchunk", "cratio", "special", "repeated_value", "lazychunk"])
        for nchunk in range(self.nchunks):
            lazychunk = self.get_lazychunk(nchunk)
            # Blosc2 flags are encoded at the end of the header
            # (see https://github.com/Blosc/c-blosc2/blob/main/README_CHUNK_FORMAT.rst)
            is_special = (lazychunk[31] & 0x70) >> 4
            special = SpecialValue(is_special)
            # The special value is encoded at the end of the header
            repeated_value = lazychunk[32:] if special == SpecialValue.VALUE else None
            # Compression ratio (nbytes and cbytes are little-endian)
            cratio = (
                np.frombuffer(lazychunk[4:8], dtype="<i4")[0]
                / np.frombuffer(lazychunk[12:16], dtype="<i4")[0]
            )
            yield ChunkInfo(nchunk, cratio, special, repeated_value, lazychunk)

    def postfilter(self, input_dtype, output_dtype=None):
        """Decorator to set a function as a postfilter.

        The postfilter function will be executed each time after decompressing
        blocks of data. It will receive three parameters:

        * the input `ndarray` to be read from
        * the output `ndarray` to be filled out
        * the offset inside the `SChunk` instance where the corresponding block begins (see example below).

        Parameters
        ----------
        input_dtype: np.dtype
            Data type of the input that will receive the postfilter function.
        output_dtype: np.dtype
            Data type of the output that will receive and fill the postfilter function.
            If None (default) it will be set to :paramref:`input_dtype`.

        Returns
        -------
        out: None

        Notes
        -----
        * `nthreads` must be 1 when decompressing.

        * The :paramref:`input_dtype` itemsize must be the same as the
          :paramref:`output_dtype` itemsize.

        See Also
        --------
        :meth:`remove_postfilter`
        :meth:`prefilter`

        Examples
        --------
        .. code-block:: python

            # Create SChunk
            input_dtype = np.dtype(np.int64)
            cparams = {"typesize": input_dtype.itemsize}
            dparams = {"nthreads": 1}
            storage = {"cparams": cparams, "dparams": dparams}
            schunk = blosc2.SChunk(chunksize=20_000 * input_dtype.itemsize, **storage)

            # Create postfilter and associate it to the schunk
            @schunk.postfilter(input_dtype)
            def postfilter(input, output, offset):
                output[:] = offset + np.arange(input.size)

        """

        def initialize(func):
            super(SChunk, self)._set_postfilter(func, input_dtype, output_dtype)

            def exec_func(*args):
                func(*args)

            return exec_func

        return initialize

    def remove_postfilter(self, func_name, _new_ctx=True):
        """Remove the postfilter from the `SChunk` instance.

        Parameters
        ----------
        func_name: str
            Name of the postfilter func.

        Returns
        -------
        out: None

        Examples
        --------
        >>> import blosc2
        >>> import numpy as np
        >>> dtype = np.dtype(np.int32)
        >>> chunk_size = 20_000 * input_dtype.itemsize
        >>> storage = {"cparams": {"typesize": input_dtype.itemsize}, "dparams": {"nthreads": 1}}
        >>> data = np.arange(500, dtype=np.int32)
        >>> schunk = blosc2.SChunk(chunksize=chunk_size, data=data, **storage)
        >>> # Define the postfilter function
        >>> @schunk.postfilter(input_dtype)
        >>> def postfilter(input, output, offset):
        >>>     output[:] = input + offset + np.arange(input.size)
        >>> out = np.empty(data.size, dtype=dtype)
        >>> schunk.get_slice(out=out)
        >>> print("Data slice with postfilter applied (first 8 elements):", out[:8])
        Data slice with postfilter applied (first 8 elements): [ 0  2  4  6  8 10 12 14]
        >>> schunk.remove_postfilter('postfilter')
        >>> print("Original data (first 8 elements):", data[:8])
        Original data (first 8 elements): [0 1 2 3 4 5 6 7]
        """
        return super().remove_postfilter(func_name)

    def filler(self, inputs_tuple, schunk_dtype, nelem=None):
        """Decorator to set a filler function.

        This function will fill :paramref:`self` according to :paramref:`nelem`.
        It will receive three parameters: a tuple with the inputs as `ndarrays`
        from which to read, the `ndarray` to fill :paramref:`self` and the
        offset inside the `SChunk` instance where the corresponding block
        begins (see example below).

        Parameters
        ----------
        inputs_tuple: tuple of tuples
            Tuple which will contain a tuple for each argument that the function will
            receive with their corresponding np.dtype.
            The supported operand types are :ref:`SChunk`, `ndarray` and
            Python scalars.
        schunk_dtype: np.dtype
            The data type to use to fill :paramref:`self`.
        nelem: int
            Number of elements to append to :paramref:`self`. If None (default) it
            will be the number of elements from the operands.

        Returns
        -------
        out: None

        Notes
        -----
        * Compression `nthreads` must be 1 when using this.
        * This does not need to be removed from the created `SChunk` instance.

        See Also
        --------
        :meth:`prefilter`

        Examples
        --------
        .. code-block:: python

            # Set the compression and decompression parameters
            schunk_dtype = np.dtype(np.float64)
            cparams = {"typesize": schunk_dtype.itemsize, "nthreads": 1}
            storage = {"cparams": cparams}
            # Create empty SChunk
            schunk = blosc2.SChunk(chunksize=20_000 * schunk_dtype.itemsize, **storage)

            # Create operands
            op_dtype = np.dtype(np.int32)
            data = np.full(20_000 * 3, 12, dtype=op_dtype)
            schunk_op = blosc2.SChunk(chunksize=20_000 * op_dtype.itemsize, data=data)

            # Create filler
            @schunk.filler(((schunk_op, op_dtype), (np.e, np.float32)), schunk_dtype)
            def filler(inputs_tuple, output, offset):
                output[:] = inputs_tuple[0] - inputs_tuple[1]

        """

        def initialize(func):
            if self.nbytes != 0:
                raise ValueError("Cannot apply a filler to a non empty SChunk")
            nelem_ = blosc2_ext.nelem_from_inputs(inputs_tuple, nelem)
            super(SChunk, self)._set_filler(func, id(inputs_tuple), schunk_dtype)
            chunksize = self.chunksize
            written_nbytes = 0
            nbytes = nelem_ * self.typesize
            while written_nbytes < nbytes:
                chunk = np.zeros(chunksize // self.typesize, dtype=schunk_dtype)
                self.append_data(chunk)
                written_nbytes += chunksize
                if (nbytes - written_nbytes) < self.chunksize:
                    chunksize = nbytes - written_nbytes
            self.remove_prefilter(func.__name__)

            def exec_func(*args):
                func(*args)

            return exec_func

        return initialize

    def prefilter(self, input_dtype, output_dtype=None):
        """Decorator to set a function as a prefilter.

        This function will be executed each time before compressing the data. It will receive three parameters:

        * the actual data as a `ndarray` from which to read,
        * the `ndarray` to be filled
        * the offset inside the `SChunk` instance where the corresponding block begins (see example below).

        Parameters
        ----------
        input_dtype: np.dtype
            Data type of the input that will receive the prefilter function.
        output_dtype: np.dtype
            Data type of the output that will receive and fill the prefilter function.
            If None (default) it will be :paramref:`input_dtype`.

        Returns
        -------
        out: None

        Notes
        -----
        * `nthreads` must be 1 when compressing.

        * The :paramref:`input_dtype` itemsize must be the same as the
          :paramref:`output_dtype` itemsize.

        See Also
        --------
        :meth:`remove_prefilter`
        :meth:`postfilter`
        :meth:`filler`

        Examples
        --------
        .. code-block:: python

            # Set the compression and decompression parameters
            input_dtype = np.dtype(np.int32)
            output_dtype = np.dtype(np.float32)
            cparams = {"typesize": output_dtype.itemsize, "nthreads": 1}
            # Create schunk
            schunk = blosc2.SChunk(chunksize=200 * 1000 * input_dtype.itemsize,
                                   cparams=cparams)

            # Set prefilter with decorator
            @schunk.prefilter(input_dtype, output_dtype)
            def prefilter(input, output, offset):
                output[:] = input - np.pi

        """

        def initialize(func):
            super(SChunk, self)._set_prefilter(func, input_dtype, output_dtype)

            def exec_func(*args):
                func(*args)

            return exec_func

        return initialize

    def remove_prefilter(self, func_name, _new_ctx=True):
        """Remove the prefilter from the `SChunk` instance.

        Parameters
        ----------
        func_name: str
            Name of the prefilter function.

        Returns
        -------
        out: None

        Examples
        --------
        >>> import blosc2
        >>> import numpy as np
        >>> dtype = np.dtype(np.int32)
        >>> chunk_size = 20_000 * dtype.itemsize
        >>> cparams = {"typesize": dtype.itemsize, "nthreads": 1}
        >>> data = np.arange(1000, dtype=np.int32)
        >>> schunk = blosc2.SChunk(chunksize=chunk_size, cparams=cparams)
        >>> # Define the prefilter function
        >>> @schunk.prefilter(input_dtype, output_dtype)
        >>> def prefilter(input, output, offset):
        >>>     output[:] = input - np.pi
        >>> schunk[:1000] = data
        >>> # Retrieve compressed data with prefilter applied
        >>> compressed_data_with_filter = schunk[0:5]
        >>> # Convert the bytes to NumPy array for comparison
        >>> compressed_array_with_filter = np.frombuffer(compressed_data_with_filter, dtype=output_dtype)
        >>> print("Compressed data with prefilter applied:", compressed_array_with_filter)
        Compressed data with prefilter applied: [-3.1415927  -2.1415927  -1.1415927  -0.14159274  0.85840726]
        >>> schunk.remove_prefilter('prefilter')
        >>> schunk[:1000] = data
        >>> compressed_data_without_filter = schunk[0:5]
        >>> compressed_array_without_filter = np.frombuffer(schunk[0:5], dtype=output_dtype)
        >>> print("Compressed data without prefilter:", compressed_array_without_filter)
        Compressed data without prefilter: [0. 1. 2. 3. 4.]
        """
        return super().remove_prefilter(func_name)

    def __dealloc__(self):
        super().__dealloc__()


@_inherit_doc_parameter(SChunk.__init__, "mmap_mode:", {r"\* - 'w\+'[^*]+": ""})
@_inherit_doc_parameter(SChunk.__init__, "initial_mapping_size:", {r"r\+ w\+, or c": "r+ or c"})
def open(urlpath, mode="a", offset=0, **kwargs):
    """Open a persistent :ref:`SChunk` or :ref:`NDArray` or a remote :ref:`C2Array`
    or a :ref:`Proxy` (see the `Notes` section for more info on the latter case).

    Parameters
    ----------
    urlpath: str | pathlib.Path | :ref:`URLPath`
        The path where the :ref:`SChunk` (or :ref:`NDArray`)
        is stored. In case it is a remote array, a :ref:`URLPath` must be passed.
    mode: str, optional
        The open mode.
    offset: int, optional
        An offset in the file where super-chunk or array data is located
        (e.g. in a file containing several such objects).

    Other parameters
    ----------------
    kwargs: dict, optional
        Keyword arguments supported:
            mmap_mode:
            initial_mapping_size:
            cparams: dict
                A dictionary with the compression parameters, which are the same that can be
                used in the :func:`~blosc2.compress2` function.
                Typesize and blocksize cannot be changed.
            dparams: dict
                A dictionary with the decompression parameters, which are the same that can
                be used in the :func:`~blosc2.decompress2` function.

    Notes
    -----
    * This is just a 'logical' open, so there is not a `close()` counterpart because
      currently there is no need for it.

    * In case :paramref:`urlpath` is a :ref:`URLPath` instance, :paramref:`mode`
      must be 'r', :paramref:`offset` must be 0, and kwargs cannot be passed.

    * In case the original object saved in :paramref:`urlpath` was a :ref:`Proxy`, this function
      will only return a :ref:`Proxy` if its source is a local :ref:`SChunk`, :ref:`NDArray`
      or a remote :ref:`C2Array`. Otherwise, it will return the Python-Blosc2 container used to cache the data which
      can be a :ref:`SChunk` or a :ref:`NDArray` and may not have all the data initialized (e.g. if the user
      has not accessed it yet).

    * When opening a :ref:`LazyExpr` keep in mind the later note regarding the operands.

    Returns
    -------
    out: :ref:`SChunk`, :ref:`NDArray` or :ref:`C2Array`
        The SChunk or NDArray (in case there is a "b2nd" metalayer")
        or the C2Array if :paramref:`urlpath` is a :ref:`blosc2.URLPath <URLPath>` instance.

    Examples
    --------
    >>> import blosc2
    >>> import numpy as np
    >>> storage = {"contiguous": True, "urlpath": getfixture('tmp_path') / "b2frame", "mode": "w"}
    >>> nelem = 20 * 1000
    >>> nchunks = 5
    >>> chunksize = nelem * 4 // nchunks
    >>> data = np.arange(nelem, dtype="int32")
    >>> # Create SChunk and append data
    >>> schunk = blosc2.SChunk(chunksize=chunksize, data=data.tobytes(), **storage)
    >>> # Open SChunk
    >>> sc_open = blosc2.open(urlpath=storage["urlpath"])
    >>> for i in range(nchunks):
    ...     dest = np.empty(nelem // nchunks, dtype=data.dtype)
    ...     schunk.decompress_chunk(i, dest)
    ...     dest1 = np.empty(nelem // nchunks, dtype=data.dtype)
    ...     sc_open.decompress_chunk(i, dest1)
    ...     np.array_equal(dest, dest1)
    True
    True
    True
    True
    True

    To open the same schunk memory-mapped, we simply need to pass the `mmap_mode` parameter:

    >>> sc_open_mmap = blosc2.open(urlpath=storage["urlpath"], mmap_mode="r")
    >>> sc_open.nchunks == sc_open_mmap.nchunks
    True
    >>> all(sc_open.decompress_chunk(i, dest1) == sc_open_mmap.decompress_chunk(i, dest1) for i in range(nchunks))
    True
    """
    if isinstance(urlpath, blosc2.URLPath):
        if mode != "r" or offset != 0 or kwargs != {}:
            raise NotImplementedError(
                "Cannot open a C2Array with mode != 'r', or offset != 0 or some kwargs"
            )
        return blosc2.C2Array(urlpath.path, urlbase=urlpath.urlbase, auth_token=urlpath.auth_token)

    if isinstance(urlpath, pathlib.PurePath):
        urlpath = str(urlpath)
    if not os.path.exists(urlpath):
        raise FileNotFoundError(f"No such file or directory: {urlpath}")

    res = blosc2_ext.open(urlpath, mode, offset, **kwargs)

    meta = getattr(res, "schunk", res).meta
    if "proxy-source" in meta:
        proxy_src = meta["proxy-source"]
        if proxy_src["local_abspath"] is not None:
            src = blosc2.open(proxy_src["local_abspath"])
            return blosc2.Proxy(src, _cache=res)
        elif proxy_src["urlpath"] is not None:
            src = blosc2.C2Array(proxy_src["urlpath"][0], proxy_src["urlpath"][1], proxy_src["urlpath"][2])
            return blosc2.Proxy(src, _cache=res)
        elif not proxy_src["caterva2_env"]:
            raise RuntimeError("Could not find the source when opening a Proxy")

    if isinstance(res, blosc2.NDArray) and "LazyArray" in res.schunk.meta:
        return blosc2._open_lazyarray(res)
    else:
        return res
