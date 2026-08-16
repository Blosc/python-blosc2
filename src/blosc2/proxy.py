#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

import ast
import asyncio
import inspect
import os
import struct
import textwrap
from abc import ABC, abstractmethod
from collections.abc import Sequence

try:
    from numpy.typing import DTypeLike
except (ImportError, AttributeError):
    # fallback to internal module (use with caution)
    from numpy._typing import DTypeLike

import numpy as np

import blosc2
from blosc2.dsl_kernel import DSLKernel, DSLSyntaxError

# Default Proxy.afetch concurrency cap for remote sources (e.g. C2Array),
# where fetches are dominated by round-trip latency, not local CPU/IO.
REMOTE_MAX_CONCURRENCY = 8

# `jit` kwargs that tune *how* an expression is evaluated, not what container the
# result is stored in. Unlike storage kwargs (`cparams`, `chunks`, `urlpath`, ...),
# these must not by themselves flip the return type from a plain NumPy array to
# an NDArray -- wanting a faster JIT backend has nothing to do with wanting a
# compressed/persisted container back.
_JIT_EXECUTION_TUNING_KWARGS = frozenset({"jit", "jit_backend", "fp_accuracy"})


class ProxyNDSource(ABC):
    """
    Base interface for NDim sources in :ref:`Proxy`.
    """

    @property
    @abstractmethod
    def shape(self) -> tuple:
        """
        The shape of the source.
        """
        pass

    @property
    @abstractmethod
    def chunks(self) -> tuple:
        """
        The chunk shape of the source.
        """
        pass

    @property
    @abstractmethod
    def blocks(self) -> tuple:
        """
        The block shape of the source.
        """
        pass

    @property
    @abstractmethod
    def dtype(self) -> np.dtype:
        """
        The dtype of the source.
        """
        pass

    @property
    def cparams(self) -> blosc2.CParams:
        """
        The compression parameters of the source.

        This property is optional and can be overridden if the source has a
        different compression configuration.
        """
        return blosc2.CParams(typesize=self.dtype.itemsize)

    @abstractmethod
    def get_chunk(self, nchunk: int) -> bytes:
        """
        Return the compressed chunk in :paramref:`self`.

        Parameters
        ----------
        nchunk: int
            The unidimensional index of the chunk to retrieve.

        Returns
        -------
        out: bytes object
            The compressed chunk.
        """
        pass

    async def aget_chunk(self, nchunk: int) -> bytes:
        """
        Return the compressed chunk in :paramref:`self` asynchronously.

        Parameters
        ----------
        nchunk: int
            The index of the chunk to retrieve.

        Returns
        -------
        out: bytes object
            The compressed chunk.

        Notes
        -----
        This method is optional, and only available if the source has an async
        `aget_chunk` method.
        """
        raise NotImplementedError(
            "aget_chunk is only available if the source has an async aget_chunk method"
        )


class ProxySource(ABC):
    """
    Base interface for sources of :ref:`Proxy` that are not NDim objects.
    """

    @property
    @abstractmethod
    def nbytes(self) -> int:
        """
        The total number of bytes in the source.
        """
        pass

    @property
    @abstractmethod
    def chunksize(self) -> tuple:
        """
        The chunksize of the source.
        """
        pass

    @property
    @abstractmethod
    def typesize(self) -> int:
        """
        The typesize of the source.
        """
        pass

    @property
    def cparams(self) -> blosc2.CParams:
        """
        The compression parameters of the source.

        This property is optional and can be overridden if the source has a
        different compression configuration.
        """
        return blosc2.CParams(typesize=self.typesize)

    @abstractmethod
    def get_chunk(self, nchunk: int) -> bytes:
        """
        Return the compressed chunk in :paramref:`self`.

        Parameters
        ----------
        nchunk: int
            The index of the chunk to retrieve.

        Returns
        -------
        out: bytes object
            The compressed chunk.
        """
        pass

    async def aget_chunk(self, nchunk: int) -> bytes:
        """
        Return the compressed chunk in :paramref:`self` asynchronously.

        Parameters
        ----------
        nchunk: int
            The index of the chunk to retrieve.

        Returns
        -------
        out: bytes object
            The compressed chunk.

        Notes
        -----
        This method is optional and only available if the source has an async
        `aget_chunk` method.
        """
        raise NotImplementedError(
            "aget_chunk is only available if the source has an async aget_chunk method"
        )


class Proxy(blosc2.Operand):
    """Proxy (with cache support) for an object following the :ref:`ProxySource` interface.

    This can be used to cache chunks of a regular data container which follows the
    :ref:`ProxySource` or :ref:`ProxyNDSource` interfaces.
    """

    def __init__(
        self, src: ProxySource or ProxyNDSource, urlpath: str | None = None, mode="a", **kwargs: dict
    ):
        """
        Create a new :ref:`Proxy` to serve as a cache to save accessed chunks locally.

        Parameters
        ----------
        src: :ref:`ProxySource` or :ref:`ProxyNDSource`
            The original container.
        urlpath: str, optional
            The urlpath where to save the container that will work as a cache.
        mode: str, optional
            "a" means read/write (create if it doesn't exist); "w" means create
            (overwrite if it exists). Default is "a".

            With "a" and an existing :paramref:`urlpath`, the cache written by an
            earlier run is adopted as is, so whatever it already holds is not
            fetched from the source again. It must be a cache from a proxy over a
            source of the same shape and dtype; anything else raises rather than
            being silently reused or overwritten.
        kwargs: dict, optional
            Keyword arguments supported:

                vlmeta: dict or None
                    A dictionary with different variable length metalayers.  One entry per metalayer:
                        key: bytes or str
                            The name of the metalayer.
                        value: object
                            The metalayer object that will be serialized using msgpack.

            Any other keyword argument (e.g. ``contiguous``) is forwarded to the
            cache container constructor (:func:`blosc2.empty` or :ref:`SChunk`),
            so callers can request e.g. a sparse (non-contiguous) cache without
            resorting to the ``_cache=`` escape hatch.

        """
        self.src = src
        self.urlpath = urlpath
        if kwargs is None:
            kwargs = {}
        self._cache = kwargs.pop("_cache", None)
        vlmeta = kwargs.pop("vlmeta", None)

        if self._cache is None and mode == "a" and urlpath is not None and os.path.exists(urlpath):
            # Reuse the cache left by an earlier run: whatever was fetched then is
            # still in there, and the creation path below would refuse to build
            # over an existing container anyway
            self._cache = self._reopen_cache(urlpath)

        if self._cache is None:
            meta_val = {
                "local_abspath": None,
                "urlpath": None,
                "caterva2_env": kwargs.pop("caterva2_env", False),
            }
            container = getattr(self.src, "schunk", self.src)
            if hasattr(container, "urlpath"):
                meta_val["local_abspath"] = container.urlpath
            elif isinstance(self.src, blosc2.C2Array):
                meta_val["urlpath"] = (self.src.path, self.src.urlbase, self.src.auth_token)
            meta = {"proxy-source": meta_val}
            if hasattr(self.src, "shape"):
                self._cache = blosc2.empty(
                    self.src.shape,
                    self.src.dtype,
                    chunks=self.src.chunks,
                    blocks=self.src.blocks,
                    cparams=self.src.cparams,
                    urlpath=urlpath,
                    mode=mode,
                    meta=meta,
                    **kwargs,
                )
            else:
                self._cache = blosc2.SChunk(
                    chunksize=self.src.chunksize,
                    cparams=self.src.cparams,
                    urlpath=urlpath,
                    mode=mode,
                    meta=meta,
                    **kwargs,
                )
                self._cache.fill_special(self.src.nbytes // self.src.typesize, blosc2.SpecialValue.UNINIT)
        self._schunk_cache = getattr(self._cache, "schunk", self._cache)
        if self.urlpath is None:
            self.urlpath = getattr(self._schunk_cache, "urlpath", None)
        if vlmeta:
            for key in vlmeta:
                self._schunk_cache.vlmeta[key] = vlmeta[key]

    def __enter__(self) -> "Proxy":
        """Enter a context manager and return this proxy."""
        return self

    def _reopen_cache(self, urlpath: str):
        """Adopt the cache container stored at *urlpath*, checking it fits the source."""
        from blosc2.schunk import _set_default_dparams

        # Not blosc2.open(): that would rebuild the source we already hold, and
        # raise outright for sources it cannot reconstruct from the cache metadata
        kwargs = {}
        _set_default_dparams(kwargs)
        cached = blosc2.blosc2_ext.open(str(urlpath), "a", 0, **kwargs)
        schunk = getattr(cached, "schunk", cached)
        if "proxy-source" not in schunk.meta:
            raise ValueError(
                f"{urlpath} is not a proxy cache; pass mode='w' to overwrite it or choose another urlpath"
            )
        if hasattr(self.src, "shape") and (
            tuple(cached.shape) != tuple(self.src.shape) or cached.dtype != self.src.dtype
        ):
            raise ValueError(
                f"the cache at {urlpath} holds a {cached.shape} {cached.dtype} array, which "
                f"does not fit the {self.src.shape} {self.src.dtype} source"
            )
        return cached

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Exit a context manager.

        ``Proxy`` does not currently expose an explicit close operation; the
        underlying cache object manages its own lifetime.
        """
        return False

    def fetch(self, item: slice | list[slice] | None = ()) -> blosc2.NDArray | blosc2.schunk.SChunk:
        """
        Get the container used as cache with the requested data updated.

        Parameters
        ----------
        item: slice or list of slices, optional
            If not None, only the chunks that intersect with the slices
            in items will be retrieved if they have not been already.

        Returns
        -------
        out: :ref:`NDArray` or :ref:`SChunk`
            The local container used to cache the already requested data.

        Examples
        --------
        >>> import numpy as np
        >>> import blosc2
        >>> data = np.arange(20).reshape(10, 2)
        >>> ndarray = blosc2.asarray(data)
        >>> proxy = blosc2.Proxy(ndarray)
        >>> slice_data = proxy.fetch((slice(0, 3), slice(0, 2)))
        >>> slice_data[:3, :2]
        [[0 1]
        [2 3]
        [4 5]]
        """
        if item == ():
            # Full realization
            for info in self._schunk_cache.iterchunks_info():
                if info.special != blosc2.SpecialValue.NOT_SPECIAL:
                    chunk = self.src.get_chunk(info.nchunk)
                    self._schunk_cache.update_chunk(info.nchunk, chunk)
        else:
            # Get only a slice
            nchunks = blosc2.get_slice_nchunks(self._cache, item)
            for info in self._schunk_cache.iterchunks_info():
                if info.nchunk in nchunks and info.special != blosc2.SpecialValue.NOT_SPECIAL:
                    chunk = self.src.get_chunk(info.nchunk)
                    self._schunk_cache.update_chunk(info.nchunk, chunk)

        return self._cache

    async def afetch(
        self, item: slice | list[slice] | None = (), max_concurrency: int | None = None
    ) -> blosc2.NDArray | blosc2.schunk.SChunk:
        """
        Retrieve the cache container with the requested data updated asynchronously.

        Parameters
        ----------
        item: slice or list of slices, optional
            If provided, only the chunks intersecting with the specified slices
            will be retrieved if they have not been already.
        max_concurrency: int, optional
            Maximum number of `aget_chunk` calls to have in flight at once
            (semaphore-bounded, so a slice spanning thousands of chunks doesn't
            fire thousands of concurrent requests at the source). Defaults to 1
            (serial, as before) for most sources, and to a higher value for
            remote sources such as :ref:`C2Array` where concurrency turns
            `N x round-trip` latency into roughly `1 x round-trip`.

        Returns
        -------
        out: :ref:`NDArray` or :ref:`SChunk`
            The local container used to cache the already requested data.

        Notes
        -----
        This method is only available if the :ref:`ProxySource` or :ref:`ProxyNDSource`
        have an async `aget_chunk` method.

        Examples
        --------
        >>> import numpy as np
        >>> import blosc2
        >>> import asyncio
        >>> from blosc2 import ProxyNDSource
        >>> class MyProxySource(ProxyNDSource):
        >>>     def __init__(self, data):
        >>>         # If the next source is multidimensional, it must have the attributes:
        >>>         self.data = data
        >>>         f"Data shape: {self.shape}, Chunks: {self.chunks}"
        >>>         f"Blocks: {self.blocks}, Dtype: {self.dtype}"
        >>>     @property
        >>>     def shape(self):
        >>>         return self.data.shape
        >>>     @property
        >>>     def chunks(self):
        >>>         return self.data.chunks
        >>>     @property
        >>>     def blocks(self):
        >>>         return self.data.blocks
        >>>     @property
        >>>     def dtype(self):
        >>>         return self.data.dtype
        >>>     # This method must be present
        >>>     def get_chunk(self, nchunk):
        >>>         return self.data.get_chunk(nchunk)
        >>>     # This method is optional
        >>>     async def aget_chunk(self, nchunk):
        >>>         await asyncio.sleep(0.1) # Simulate an asynchronous operation
        >>>         return self.data.get_chunk(nchunk)
        >>> data = np.arange(20).reshape(4, 5)
        >>> chunks = [2, 5]
        >>> blocks = [1, 5]
        >>> data = blosc2.asarray(data, chunks=chunks, blocks=blocks)
        >>> source = MyProxySource(data)
        >>> proxy = blosc2.Proxy(source)
        >>> async def fetch_data():
        >>>     # Fetch a slice of the data from the proxy asynchronously
        >>>     slice_data = await proxy.afetch(slice(0, 2))
        >>>     # Note that only data fetched is shown, the rest is uninitialized
        >>>     slice_data[:]
        >>> asyncio.run(fetch_data())
        >>> # Using getitem to get a slice of the data
        >>> result = proxy[1:2, 1:3]
        >>> f"Proxy getitem: {result}"
        Data shape: (4, 5), Chunks: (2, 5)
        Blocks: (1, 5), Dtype: int64
        [[0 1 2 3 4]
        [5 6 7 8 9]
        [0 0 0 0 0]
        [0 0 0 0 0]]
        Proxy getitem: [[6 7]]
        """
        if not callable(getattr(self.src, "aget_chunk", None)):
            raise NotImplementedError("afetch is only available if the source has an aget_chunk method")

        if item == ():
            wanted = None  # every missing chunk
        else:
            wanted = set(blosc2.get_slice_nchunks(self._cache, item))
        to_fetch = [
            info.nchunk
            for info in self._schunk_cache.iterchunks_info()
            if info.special != blosc2.SpecialValue.NOT_SPECIAL and (wanted is None or info.nchunk in wanted)
        ]

        if max_concurrency is None:
            max_concurrency = REMOTE_MAX_CONCURRENCY if isinstance(self.src, blosc2.C2Array) else 1
        semaphore = asyncio.Semaphore(max(1, max_concurrency))

        async def _fetch_one(nchunk):
            async with semaphore:
                chunk = await self.src.aget_chunk(nchunk)
            # Runs to completion between awaits, so concurrent writers can't interleave.
            self._schunk_cache.update_chunk(nchunk, chunk)

        if to_fetch:
            await asyncio.gather(*(_fetch_one(nchunk) for nchunk in to_fetch))

        return self._cache

    def __getitem__(self, item: slice | list[slice]) -> np.ndarray:
        """
        Get a slice as a numpy.ndarray using the :ref:`Proxy`.

        Parameters
        ----------
        item: slice or list of slices
            The slice of the desired data.

        Returns
        -------
        out: numpy.ndarray
            An array with the data slice.

        Examples
        --------
        >>> import numpy as np
        >>> import blosc2
        >>> data = np.arange(25).reshape(5, 5)
        >>> ndarray = blosc2.asarray(data)
        >>> proxy = blosc2.Proxy(ndarray)
        >>> proxy[0:3, 0:3]
        [[ 0  1  2]
        [ 5  6  7]
        [10 11 12]
        [20 21 22]]
        >>> proxy[2:5, 2:5]
        [[12 13 14]
        [17 18 19]
        [22 23 24]]
        """
        # Populate the cache when possible.  Read-only reopens must remain
        # observational, so fall back to the source without mutating the cache.
        try:
            self.fetch(item)
        except ValueError as exc:
            if getattr(self._schunk_cache, "mode", None) != "r" or "reading mode" not in str(exc):
                raise
            return self.src[item]
        return self._cache[item]

    @property
    def dtype(self) -> np.dtype:
        """The dtype of :paramref:`self` or None if the data is unidimensional"""
        return self._cache.dtype if isinstance(self._cache, blosc2.NDArray) else None

    @property
    def shape(self) -> tuple[int]:
        """The shape of :paramref:`self`"""
        return self._cache.shape if isinstance(self._cache, blosc2.NDArray) else len(self._cache)

    @property
    def chunks(self) -> tuple[int]:  # cache should have same chunks as src
        """The chunks of :paramref:`self` or None if the data is not a Blosc2 NDArray"""
        return self._cache.chunks if isinstance(self._cache, blosc2.NDArray) else None

    @property
    def blocks(self) -> tuple[int]:  # cache should have same blocks as src
        """The blocks of :paramref:`self` or None if the data is not a Blosc2 NDArray"""
        return self._cache.blocks if isinstance(self._cache, blosc2.NDArray) else None

    @property
    def schunk(self) -> blosc2.schunk.SChunk:
        """The :ref:`SChunk` of the cache"""
        return self._schunk_cache

    @property
    def cparams(self) -> blosc2.CParams:
        """The compression parameters of the cache"""
        return self._cache.cparams

    @property
    def info(self) -> str:
        """The info of the cache"""
        if isinstance(self._cache, blosc2.NDArray):
            return self._cache.info
        raise NotImplementedError("info is only available if the source is a NDArray")

    def __str__(self):
        return f"Proxy({self.src}, urlpath={self.urlpath})"

    @property
    def vlmeta(self) -> blosc2.schunk.vlmeta:
        """
        Get the vlmeta of the cache.

        See Also
        --------
        :py:attr:`blosc2.schunk.SChunk.vlmeta`
        """
        return self._schunk_cache.vlmeta

    @property
    def fields(self) -> dict:
        """
        Dictionary with the fields of :paramref:`self`.

        Returns
        -------
        fields: dict
            A dictionary with the fields of the :ref:`Proxy`.

        See Also
        --------
        :ref:`NDField`

        Examples
        --------
        >>> import numpy as np
        >>> import blosc2
        >>> data = np.ones(16, dtype=[('field1', 'i4'), ('field2', 'f4')]).reshape(4, 4)
        >>> ndarray = blosc2.asarray(data)
        >>> proxy = blosc2.Proxy(ndarray)
        >>>  # Get a dictionary of fields from the proxy, where each field can be accessed individually
        >>> fields_dict = proxy.fields
        >>> for field_name, field_proxy in fields_dict.items():
        >>>     print(f"Field name: {field_name}, Field data: {field_proxy}")
        Field name: field1, Field data: <blosc2.proxy.ProxyNDField object at 0x114472d20>
        Field name: field2, Field data: <blosc2.proxy.ProxyNDField object at 0x10e215be0>
        >>> fields_dict['field2'][:]
        [[1. 1. 1. 1.]
         [1. 1. 1. 1.]
         [1. 1. 1. 1.]
         [1. 1. 1. 1.]]
        """
        _fields = getattr(self._cache, "fields", None)
        if _fields is None:
            return None
        return {key: ProxyNDField(self, key) for key in _fields}


_FRAME_MAGIC = b"b2frame\0"


def _read_frame_index(f) -> tuple[bytes, list, np.ndarray]:
    """Read the header and the chunk offsets of a contiguous frame from *f*.

    Returns the raw header bytes, the header decoded as the msgpack array it is,
    and the absolute position of every chunk.  A negative position is not a
    position at all: it encodes a run-length chunk that was never written to the
    file.  See ``README_CFRAME_FORMAT.rst`` in c-blosc2 for the layout.
    """
    import msgpack

    f.seek(0)
    prefix = f.read(24)
    if prefix[2:10] != _FRAME_MAGIC:
        raise ValueError("not a Blosc2 contiguous frame")
    # header_len is the one field that must be located by hand; everything after
    # it comes out of unpacking the header, which is plain msgpack
    header_len = struct.unpack(">i", prefix[11:15])[0]
    f.seek(0)
    raw = f.read(header_len)
    header = msgpack.unpackb(raw, raw=False, strict_map_key=False)

    # The offsets live in a Blosc2 chunk of their own, right after the data ones
    index_pos = header[1] + header[5]
    f.seek(index_pos)
    index_cbytes = struct.unpack("<i", f.read(16)[12:16])[0]
    f.seek(index_pos)
    offsets = np.frombuffer(blosc2.decompress2(f.read(index_cbytes)), dtype=np.int64)
    # Offsets are relative to the end of the header
    return raw, header, np.where(offsets >= 0, offsets + header_len, offsets)


def _frame_metalayer(raw: bytes, header: list, name: str):
    """Decode the *name* metalayer out of an already-read frame header."""
    offset = header[13][1][name]  # KeyError if the frame has no such metalayer
    nbytes = struct.unpack(">I", raw[offset + 1 : offset + 5])[0]  # msgpack bin32
    import msgpack

    return msgpack.unpackb(raw[offset + 5 : offset + 5 + nbytes], raw=False)


def _chunk_extents(offsets: np.ndarray, header: list) -> np.ndarray:
    """How many bytes to read at each chunk offset to be sure of covering it.

    A chunk carries its own compressed size in its header, but asking for that
    first would cost a second request per chunk.  The next thing stored after a
    chunk bounds it instead -- another chunk, or the offsets chunk -- capped by
    what a chunk can possibly weigh, so a hole left by an updated chunk cannot
    turn into an absurd read.  The caller truncates to the real size.
    """
    index_pos = header[1] + header[5]
    bounds = np.sort(np.append(offsets[offsets >= 0], index_pos))
    extents = bounds[np.searchsorted(bounds, offsets, side="right")] - offsets
    return np.minimum(extents, header[8] + blosc2.MAX_OVERHEAD)


class FsspecNDSource(ProxyNDSource):
    """A :ref:`Proxy` source that serves the chunks of a remote Blosc2 frame.

    The frame stays where it is: only its header, its chunk offsets, and the
    chunks a slice actually touches ever cross the network.  This is what
    ``blosc2.open(url, lazy=True)`` builds; wrap it in a :ref:`Proxy` by hand
    when the cache belongs at a path of your choosing rather than inside
    ``cache_storage``::

        src = blosc2.FsspecNDSource("s3://bucket/big.b2nd")
        a = blosc2.Proxy(src, urlpath="big-cache.b2nd", mode="a")

    Contiguous frames carrying a ``b2nd`` metalayer only, which is what
    :func:`blosc2.asarray` and friends write to a single file.  Sparse frames and
    ``.b2d`` stores are directories; open those with ``cache_storage=``.
    """

    def __init__(self, urlpath: str):
        from blosc2.core import _import_fsspec

        fsspec = _import_fsspec(urlpath)
        fs, path = fsspec.url_to_fs(urlpath)
        if fs.isdir(path):
            raise NotImplementedError(
                f"{urlpath} is a directory (a sparse frame or a store), which cannot be read "
                "chunk by chunk; open it with cache_storage= instead"
            )
        self.urlpath = urlpath
        self._fs, self._path = fs, path
        # Identifies the remote bytes, so a cache built against them can tell it
        # has gone stale -- and chunk offsets from a replaced frame are garbage.
        # fsspec's own token, rather than a tuple of the metadata fields we guess
        # a backend exposes: memory:// has no mtime, which left it size-only.
        self.stamp = fs.ukey(path)
        # The handle is only for reading the index: chunk reads are stateless, so
        # the source holds no file position that two threads could fight over
        with fs.open(path, "rb") as f:
            raw, header, self._offsets = _read_frame_index(f)
        self._chunksize = header[8]
        self._extents = _chunk_extents(self._offsets, header)
        try:
            _, _, shape, chunks, blocks, dtype_format, dtype = _frame_metalayer(raw, header, "b2nd")
        except KeyError:
            raise NotImplementedError(
                f"{urlpath} has no b2nd metalayer, so it is a plain SChunk rather than an "
                "NDArray; read it whole or with cache_storage= instead"
            ) from None
        if dtype_format != 0:
            raise NotImplementedError(f"unsupported dtype format {dtype_format} in {urlpath}")
        self._shape, self._chunks, self._blocks = tuple(shape), tuple(chunks), tuple(blocks)
        self._dtype = np.dtype(dtype)

    @property
    def shape(self) -> tuple:
        return self._shape

    @property
    def chunks(self) -> tuple:
        return self._chunks

    @property
    def blocks(self) -> tuple:
        return self._blocks

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    def get_chunk(self, nchunk: int) -> bytes:
        offset = int(self._offsets[nchunk])
        if offset < 0:
            return self._special_chunk(offset)
        data = self._fs.cat_file(self._path, start=offset, end=offset + int(self._extents[nchunk]))
        return data[: struct.unpack("<i", data[12:16])[0]]

    async def aget_chunk(self, nchunk: int) -> bytes:
        """Same as :meth:`get_chunk`, but letting several fetches overlap.

        This is what makes :meth:`Proxy.afetch` worth using against an object
        store, where a slice spanning many chunks is nearly all round-trip
        latency.  Backends without an async implementation fall back to the
        blocking path, which costs nothing but gains nothing either.
        """
        offset = int(self._offsets[nchunk])
        if offset < 0:
            return self._special_chunk(offset)
        if not getattr(self._fs, "async_impl", False):
            return self.get_chunk(nchunk)
        end = offset + int(self._extents[nchunk])
        data = await self._fs._cat_file(self._path, start=offset, end=end)
        return data[: struct.unpack("<i", data[12:16])[0]]

    def _special_chunk(self, offset: int) -> bytes:
        """Rebuild a run-length chunk, which lives in its offset instead of the file."""
        kind = ((offset & 0xFFFFFFFFFFFFFFFF) >> 56) & 0x7
        nitems = self._chunksize // self._dtype.itemsize
        if kind == 2:
            data = np.full(nitems, np.nan, dtype=self._dtype)
        else:
            # A run of zeros (1); uninitialized chunks (4) have no defined
            # content, and zeros is what reading them locally hands back too
            data = np.zeros(nitems, dtype=self._dtype)
        return blosc2.compress2(data, typesize=self._dtype.itemsize)


class ProxyNDField(blosc2.Operand):
    def __init__(self, proxy: Proxy, field: str):
        self.proxy = proxy
        self.field = field
        self._dtype = proxy.dtype[field]
        self._shape = proxy.shape

    @property
    def dtype(self) -> np.dtype:
        """
        Get the data type of the :class:`ProxyNDField`.

        Returns
        -------
        out: np.dtype
            The data type of the :class:`ProxyNDField`.
        """
        return self._dtype

    @property
    def shape(self) -> tuple[int]:
        """
        Get the shape of the :class:`ProxyNDField`.

        Returns
        -------
        out: tuple
            The shape of the :class:`ProxyNDField`.
        """
        return self._shape

    def __getitem__(self, item: slice | list[slice]) -> np.ndarray:
        """
        Get a slice as a numpy.ndarray using the `field` in `proxy`.

        Parameters
        ----------
        item: slice or list of slices
            The slice of the desired data.

        Returns
        -------
        out: numpy.ndarray
            An array with the data slice.
        """
        # Get the data and return the corresponding field
        nparr = self.proxy[item]
        return nparr[self.field]


def convert_dtype(dt: str | DTypeLike):
    """
    Attempts to convert to blosc2.dtype (i.e. numpy dtype)
    """
    if hasattr(dt, "as_numpy_dtype"):
        dt = dt.as_numpy_dtype
    try:
        return np.dtype(dt)
    except TypeError:  # likely passed e.g. a torch.float64
        return np.dtype(str(dt).split(".")[1])
    except Exception as e:
        raise TypeError(f"Could not parse dtype arg {dt}.") from e


class SimpleProxy(blosc2.Operand):
    """
    Simple proxy for any data container to be used with the compute engine.

    The source must have a `shape` and `dtype` attributes; if not,
    it will be converted to a NumPy array via the `np.asarray` function.
    It should also have a `__getitem__` method.

    This only supports the __getitem__ method. No caching is performed.

    Examples
    --------
    >>> import numpy as np
    >>> import blosc2
    >>> a = np.arange(20, dtype=np.float32).reshape(4, 5)
    >>> proxy = blosc2.SimpleProxy(a)
    >>> proxy[1:3, 2:4]
    [[ 7.  8.]
     [12. 13.]]
    """

    def __init__(self, src, chunks: tuple | None = None, blocks: tuple | None = None):
        from blosc2._utf8_array import UTF8Array

        if isinstance(src, UTF8Array):
            # The compute engine indexes chunk-wise into fixed-width elements,
            # which a variable-length utf8 array has not got, so widen it here.
            # (lazyexpr() routes utf8 operands to the span driver instead; this
            # is the fallback for the entry points that do not.)  Until this
            # array grew a .shape, the branch below did the same thing by
            # accident, via np.asarray.
            src = src.astype()
        elif not hasattr(src, "shape") or not hasattr(src, "dtype"):
            # If the source is not an array, convert it to NumPy
            src = np.asarray(src)
        if not hasattr(src, "__getitem__"):
            raise TypeError("The source must have a __getitem__ method")
        self._src = src
        self._dtype = convert_dtype(src.dtype)
        self._shape = src.shape if isinstance(src.shape, tuple) else tuple(src.shape)
        # Compute reasonable values for chunks and blocks
        cparams = blosc2.CParams(clevel=0)

        def is_ints_sequence(src, attr):
            seq = getattr(src, attr, None)
            if not isinstance(seq, Sequence) or isinstance(seq, str | bytes):
                return False
            return all(isinstance(x, int) for x in seq)

        chunks = src.chunks if chunks is None and is_ints_sequence(src, "chunks") else chunks
        blocks = src.blocks if blocks is None and is_ints_sequence(src, "blocks") else blocks
        self.chunks, self.blocks = blosc2.compute_chunks_blocks(
            self.shape, chunks, blocks, self.dtype, cparams=cparams
        )

    @property
    def src(self):
        """The source object that this proxy wraps."""
        return self._src

    @property
    def shape(self):
        """The shape of the source array."""
        return self._shape

    @property
    def dtype(self):
        """The data type of the source array."""
        return self._dtype

    @property
    def ndim(self):
        """The number of dimensions of the source array."""
        return len(self.shape)

    def __getitem__(self, item: slice | list[slice]) -> np.ndarray:
        """
        Get a slice as a numpy.ndarray (via this proxy).

        Parameters
        ----------
        item

        Returns
        -------
        out: numpy.ndarray
            An array with the data slice.
        """
        out = self._src[item]
        if not hasattr(out, "shape") or out.shape == ():
            return out
        else:
            # avoids copy for PyTorch (JAX/Tensorflow will always copy,
            # no easy way around it)
            return np.asarray(out)


def as_simpleproxy(*arrs: Sequence[blosc2.Array]) -> tuple[SimpleProxy | blosc2.Operand]:
    """
    Convert an Array object which fulfills Array protocol into SimpleProxy. If x is already a
    blosc2.Operand simply returns object.

    Parameters
    ----------
    arrs: Sequence[blosc2.Array]
        Objects fulfilling Array protocol.

    Returns
    -------
    out: tuple[blosc2.SimpleProxy | blosc2.Operand]
        Objects with minimal interface for blosc2 LazyExpr computations.
    """
    out = ()
    for x in arrs:
        if isinstance(x, blosc2.Operand):
            out += (x,)
        else:
            out += (SimpleProxy(x),)
    return out[0] if len(out) == 1 else out


def _is_pandas_string_series(col) -> bool:
    """True for a pandas string column.

    pandas 3's `str` dtype reports `kind == "O"`, so the kind is useless here
    and `pd.api.types.is_string_dtype` is the only reliable test.
    """
    try:
        import pandas as pd
    except ImportError:
        return False
    return pd.api.types.is_string_dtype(getattr(col, "dtype", None))


def _string_series_to_numpy(col, label=None):
    """A pandas string column as a fixed-width `<Un` array.

    `np.asarray` on a pandas 3 `str` column yields an object array of PyObject
    pointers (the `__array__` route destroys the Arrow layout), which miniexpr
    cannot take; going through `to_numpy(dtype=object)` and `.astype(str)` gives
    the fixed-width array the kernels want.

    Nulls are rejected rather than substituted.  A row kernel that touches a
    null raises in pandas too (`'p=' + row['x']` -> TypeError,
    `row['x'].lower()` -> AttributeError), so quietly turning one into `""`
    would invent a value pandas never produces.
    """
    if label is None:
        label = col.name
    if col.isna().any():
        raise ValueError(
            f"blosc2.jit: string column {label!r} contains nulls, and a row-wise kernel "
            "over a null raises in pandas too. Fill them first, e.g. "
            f"df[{label!r}] = df[{label!r}].fillna('')."
        )
    values = col.to_numpy(dtype=object)
    return values.astype(str)


class _PandasRowProxy(blosc2.Operand):
    """Row proxy for `PandasUdfEngine.apply`'s axis=1 route.

    Stands in for "the current row" the way the textbook `axis=1` idiom
    expects (`row["colname"]`), but is backed by whole *columns*: `row["a"]
    + row["b"]` traces to one fused expression over the whole column set in
    a single call, instead of looping over rows in Python. Columns are
    extracted lazily (and cached) from the original DataFrame, not from a
    whole-frame NumPy array, so per-column dtypes are preserved.
    """

    def __init__(self, df):
        self._df = df
        self._cache = {}

    def _raw_column(self, key):
        """The column as a plain array, for the DSL route.

        The tracing route wants a SimpleProxy operand; a DSL kernel wants the
        array itself, and accepts string columns the traced one does not.
        """
        col = self._df[key]
        if _is_pandas_string_series(col):
            return _string_series_to_numpy(col, key)
        return col.to_numpy()

    def __getitem__(self, key):
        if not isinstance(key, str):
            raise TypeError(
                f"row[{key!r}]: axis=1 row proxies only support column access by "
                "name (a string). Positional or iterable row access is not "
                "supported; for row-wise computations, call your @blosc2.jit "
                "function directly with the DataFrame columns as separate "
                "arguments instead, e.g. func(df['a'], df['b'])."
            )
        if key in self._cache:
            return self._cache[key]
        n_matches = int((self._df.columns == key).sum())
        if n_matches == 0:
            raise KeyError(f"row[{key!r}]: no such column in the DataFrame")
        if n_matches > 1:
            raise KeyError(
                f"row[{key!r}]: column label is duplicated ({n_matches} matches); "
                "axis=1 row proxies require unique column labels"
            )
        col = self._raw_column(key)
        if col.dtype.kind not in "biufcUS":
            raise ValueError(
                f"row[{key!r}]: column has dtype {col.dtype!r}, which the Blosc2 engine "
                "cannot vectorize. Numeric, boolean and string columns are supported."
            )
        proxy = SimpleProxy(col)
        self._cache[key] = proxy
        return proxy

    def __getattr__(self, name):
        raise AttributeError(
            f"row.{name}: axis=1 row proxies only support column access via "
            f"row[{name!r}]; attribute access, iteration and per-row methods "
            "(e.g. row.isna()) are not supported. For per-row computations that "
            "need more than combining columns (e.g. per-row branching), call "
            "your @blosc2.jit function directly with the columns as separate "
            "array arguments instead of through df.apply(..., axis=1)."
        )


def _undecorated(func):
    """The original function behind a @blosc2.jit wrapper, or *func* itself.

    Source inspection has to see what the user wrote, not the wrapper.
    """
    return getattr(func, "_blosc2_jit_wrapped", func)


def _decorate_once(func, decorator):
    """Apply *decorator* unless *func* is already a @blosc2.jit wrapper.

    Decorating twice used to break the DSL route: the outer (tracing) wrapper
    replaces array arguments with SimpleProxy operands, so the inner DSL kernel
    saw no array at all and failed asking for `shape=`.
    """
    return func if hasattr(func, "_blosc2_jit_wrapped") else decorator(func)


def _analyze_row_func(func) -> tuple[bool, bool]:
    """Inspect *func* for the two signals `PandasUdfEngine.apply`'s axis=1
    route needs to pick a dispatch strategy: whether it subscripts its first
    parameter with a string literal anywhere in its body (the `row["colname"]`
    idiom), and whether its body contains a `for`/`while` loop.

    Both default to False (the historical per-row loop) if the source can't
    be inspected, e.g. a dynamically built function.
    """
    try:
        sig = inspect.signature(func)
        params = list(sig.parameters.values())
        if not params:
            return False, False
        row_name = params[0].name
        source = textwrap.dedent(inspect.getsource(func))
        tree = ast.parse(source)
    except (OSError, TypeError, SyntaxError, ValueError):
        return False, False
    nodes = list(ast.walk(tree))
    uses_subscript = any(
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == row_name
        and isinstance(node.slice, ast.Constant)
        and isinstance(node.slice.value, str)
        for node in nodes
    )
    has_loop = any(isinstance(node, ast.For | ast.While) for node in nodes)
    return uses_subscript, has_loop


def _has_control_flow(source: str | None) -> bool:
    """Whether *source* (a DSL-extracted function source, or None) contains a
    branch or loop that tracing cannot observe."""
    if source is None:
        return False
    tree = ast.parse(source)
    return any(isinstance(node, ast.If | ast.For | ast.While) for node in ast.walk(tree))


def _wide_frame_hint(err: BaseException, func_name: str, params) -> str | None:
    """Guidance to append when a call gets a keyword the function doesn't take.

    The usual cause is the `kernel(**df)` idiom (see doc/guides/pandas_engine.md)
    against a frame carrying more columns than the kernel has parameters. Extra
    keywords are rejected rather than dropped, so that a keyword meant to do
    something -- a typo, a stale argument name -- never goes silently unused.
    """
    if not isinstance(err, TypeError) or "unexpected keyword argument" not in str(err):
        return None
    params = list(params)
    if not params:
        return None
    cols = ", ".join(repr(p) for p in params)
    return (
        f"If you are calling {func_name}(**df), subset the frame to the "
        f"kernel's parameters: {func_name}(**df[[{cols}]])"
    )


def _signature_params(func) -> list:
    """Parameter names of *func*, or an empty list if it cannot be introspected."""
    try:
        return list(inspect.signature(func).parameters)
    except (TypeError, ValueError):
        return []


def _row_column(row, label):
    """The raw column array behind *label*, from a row proxy or a DataFrame."""
    getter = getattr(row, "_raw_column", None)
    if getter is not None:
        return getter(label)
    col = row[label]
    if isinstance(col, np.ndarray | blosc2.NDArray):
        return col
    if _is_pandas_string_series(col):
        return _string_series_to_numpy(col, label)
    return np.asarray(col)


def _dsl_operand_values(kernel: DSLKernel, sig, args, func_kwargs) -> tuple:
    """The kernel's operands, one per DSL input name, in declaration order."""
    if kernel.row_columns and len(args) == 1 and not func_kwargs:
        # The `row["colname"]` kernel: its signature still says one row, but the
        # compiled kernel takes one parameter per referenced column.
        values = tuple(_row_column(args[0], label) for label in kernel.row_columns.values())
    else:
        try:
            bound = sig.bind(*args, **func_kwargs)
        except TypeError as e:
            # sig.bind's message names no function; prefix it, and point at the
            # subsetting fix when a wide DataFrame was unpacked into the call.
            hint = _wide_frame_hint(e, kernel.__name__, kernel.input_names or sig.parameters)
            raise TypeError(f"{kernel.__name__}() {e}" + (f"\n{hint}" if hint else "")) from None
        bound.apply_defaults()
        values = tuple(bound.arguments[name] for name in kernel.input_names)
    # Accept array-protocol operands (pandas Series, polars Series, ...) the same
    # way the tracing route already does; zero-copy when the source is numpy-backed.
    return tuple(
        np.asarray(v)
        if not isinstance(v, np.ndarray | blosc2.NDArray)
        and hasattr(v, "__array__")
        and getattr(v, "ndim", 0) > 0
        else v
        for v in values
    )


def _jit_dsl_wrapper(kernel: DSLKernel, out, decorator_kwargs: dict):
    """Build the call wrapper for the DSL (control-flow) dispatch route of `jit`.

    Unlike the tracing `wrapper` (which calls `func` once to record a single
    expression, losing any branch not taken on that one call), this calls
    `kernel` once per invocation through `blosc2.lazyudf`, so every branch and
    loop in the kernel body is compiled and actually runs, once per chunk.
    """

    def dsl_wrapper(*args, **func_kwargs):
        sig = kernel._sig
        if sig is None:
            raise TypeError(f"@blosc2.jit: cannot introspect the signature of {kernel.__name__!r}")
        values = _dsl_operand_values(kernel, sig, args, func_kwargs)

        array_shapes = {
            v.shape
            for v in values
            if isinstance(v, np.ndarray | blosc2.NDArray) and getattr(v, "ndim", 0) > 0
        }
        if not array_shapes:
            shape = decorator_kwargs.get("shape")
            if shape is None:
                raise TypeError(
                    "@blosc2.jit DSL kernels with only scalar inputs require `shape=` "
                    "(passed to the jit decorator) to determine the result shape."
                )
        elif len(array_shapes) > 1:
            raise TypeError(
                "blosc2.jit DSL kernels do not support broadcasting; all array arguments "
                f"must share one shape, got {sorted(array_shapes)}"
            )
        else:
            (shape,) = array_shapes

        # Execution-tuning kwargs (jit/jit_backend/fp_accuracy) are baked into the
        # LazyUDF at construction, so they take effect on *both* the getitem
        # (NumPy) and compute (NDArray) return paths below.  Storage kwargs
        # (cparams, chunks, urlpath, ...) are applied once, only at the return
        # step -- passing them here too would e.g. apply `urlpath=` twice and raise.
        exec_kwargs = {
            k: v for k, v in decorator_kwargs.items() if k in _JIT_EXECUTION_TUNING_KWARGS and v is not None
        }
        storage_kwargs = {k: v for k, v in decorator_kwargs.items() if k not in _JIT_EXECUTION_TUNING_KWARGS}
        lexpr = blosc2.lazyudf(kernel, values, dtype=None, shape=shape, **exec_kwargs)

        if out is not None:
            if isinstance(out, blosc2.NDArray):
                raise NotImplementedError(
                    "blosc2.jit does not support an NDArray `out` on the DSL (control-flow) "
                    "dispatch route; use lexpr.compute(urlpath=..., mode='w') to persist a "
                    "result chunk-by-chunk instead."
                )
            if not isinstance(out, np.ndarray):
                raise TypeError(f"blosc2.jit `out` must be a NumPy array or NDArray, got {type(out)!r}")
            if out.shape != shape:
                raise TypeError(f"`out` shape {out.shape} does not match operand shape {shape}")
            res = lexpr.compute(cparams=blosc2.CParams(clevel=0))
            if out.dtype != res.dtype:
                raise TypeError(
                    f"`out` dtype {out.dtype} does not match the inferred result dtype {res.dtype}"
                )
            if out.flags.c_contiguous:
                res.get_slice_numpy(out, (tuple(0 for _ in res.shape), tuple(res.shape)))
            else:
                np.copyto(out, res[()], casting="no")
            return out

        if storage_kwargs and any(v is not None for v in storage_kwargs.values()):
            # Execution-tuning kwargs go along too: compute() names all three,
            # while lazyudf() above only names jit/jit_backend, so fp_accuracy
            # would otherwise be dropped on this path.
            return lexpr.compute(**decorator_kwargs)
        return lexpr[()]

    return dsl_wrapper


def jit(func=None, *, out=None, disable=False, strict=None, **kwargs):  # noqa: C901
    """
    Prepare a function so that it can be used with the Blosc2 compute engine.

    The inputs of the function can be any combination of NumPy/NDArray arrays
    and scalars.  By default, the function is *traced*: it is called once with
    the NumPy arrays replaced by :ref:`SimpleProxy` objects (NDArray objects are
    used as is) to record a single expression, which is then what actually gets
    evaluated. Because tracing only calls the function once, an ``if``/``for``/
    ``while`` in the body only ever takes the one path that single call
    happened to follow — see `strict` below for when `jit` instead compiles the
    function whole, so every branch and loop genuinely runs.

    The returned value will be a NDArray if a *storage* kwarg is provided (e.g.
    `cparams=`, `chunks=`, `urlpath=` — anything that only makes sense for a
    compressed/persisted container). Else, the return value will be a NumPy
    array (if the function returns a NumPy array). Execution-tuning kwargs
    (`jit=`, `jit_backend=`, `fp_accuracy=`) do not by themselves trigger this —
    they take effect either way, without changing the return type. If `out` is
    provided, the result will be computed and stored in the `out` array.

    Parameters
    ----------
    func: callable
        The function to be prepared for the Blosc2 compute engine.
    out: np.ndarray, NDArray, optional
        The output array where the result will be stored.  On the DSL
        (control-flow) dispatch route, a NumPy `out` is filled in place
        (directly when C-contiguous, else via a copy); an NDArray `out` is not
        supported there — use ``compute(urlpath=..., mode="w")`` instead.
    disable: bool, optional
        If True, the decorator is disabled and the original function is returned unchanged.
        Default is False.
    strict: bool, optional
        Control which evaluation route is used:

        - ``None`` (default): if *func*'s body contains an ``if``/``for``/``while``
          and it compiles as a DSL kernel, dispatch to the DSL route (miniexpr
          runs the whole function, so branches/loops behave as written); a
          control-flow function that fails DSL extraction still falls back to
          tracing, but a subsequent tracing failure is annotated with the DSL
          extraction error.  Functions without control flow always trace, even
          if they happen to be DSL-valid (tracing is faster for pure elementwise
          expressions).
        - ``True``: always use the DSL route, raising
          :class:`~blosc2.dsl_kernel.DSLSyntaxError` at decoration time if
          *func*'s source cannot be **parsed** as a DSL kernel.  Note the
          guarantee is exactly that -- parsing -- and not that the kernel will
          compile: a function that is DSL-shaped but calls something miniexpr
          does not implement passes here and fails later, at call time, with a
          ``RuntimeError``.  See the DSL syntax reference for what the grammar
          accepts.  (Unrelated to :func:`blosc2.dsl_kernel`, which builds a
          :class:`DSLKernel` object rather than an evaluating wrapper.)

          This also works as a pandas engine, which is the only way to reach
          ``strict`` through that entry point:
          ``df.apply(f, engine=blosc2.jit(strict=True))``.
        - ``False``: always use the tracing route, even if *func* has control
          flow (this only works when branches/loops depend on plain Python
          values, not on traced arrays).
    **kwargs: dict, optional
        Additional keyword arguments supported by the :func:`empty` constructor.

    Returns
    -------
    wrapper

    Notes
    -----
    * Although many NumPy functions are supported, some may not be implemented yet.
      If you find a function that is not supported, please open an issue.
    * `out` and `kwargs` parameters are not supported for all expressions
      (e.g. when using a reduction as the last function).  In this case, you can
      still use the `out` parameter of the reduction function for some custom
      control over the output.
    * DSL-route kernels do not support broadcasting: every array argument must
      share the same shape.

    Examples
    --------
    >>> import numpy as np
    >>> import blosc2
    >>> @blosc2.jit
    ... def compute_expression(a, b, c):
    ...     return np.sum(((a ** 3 + np.sin(a * 2)) > 2 * c) & (b > 0), axis=1)
    >>> a = np.arange(20, dtype=np.float32).reshape(4, 5)
    >>> b = np.arange(20).reshape(4, 5)
    >>> c = np.arange(5)
    >>> compute_expression(a, b, c)
    array([3, 5, 5, 5])

    With ``strict=True`` the function is compiled as a DSL kernel, so a real
    per-element ``if`` runs as written -- only the matching arm is evaluated:

    >>> @blosc2.jit(strict=True)
    ... def clamp(x):
    ...     if x < 0.0:
    ...         out = 0.0
    ...     else:
    ...         out = x
    ...     return out
    >>> clamp(np.array([-1.5, 2.0, -0.5]))
    array([0., 2., 0.])

    The guarantee is that the source *parses* as DSL, checked at decoration
    time. A body the grammar does not accept is rejected right away, rather
    than silently falling back to tracing:

    >>> @blosc2.jit(strict=True)  # doctest: +IGNORE_EXCEPTION_DETAIL
    ... def not_dsl(x):
    ...     return np.where(x >= 0, x.mean(), x)
    Traceback (most recent call last):
        ...
    blosc2.dsl_kernel.DSLSyntaxError: Unsupported call target in DSL ...
    """

    def decorator(func):  # noqa: C901
        if disable:
            return func

        kernel = DSLKernel(func)
        has_cf = _has_control_flow(kernel.dsl_source)
        dsl_ok = kernel.dsl_source is not None and kernel.dsl_error is None
        if strict is True and not dsl_ok:
            # One condition, one exception type: DSLSyntaxError (a ValueError)
            # whether the source failed to parse as DSL or could not be read at
            # all (a lambda, a C function). The message avoids naming the
            # decorator spelling, since `strict=True` also arrives through
            # `df.apply(..., engine=blosc2.jit(strict=True))`.
            raise kernel.dsl_error or DSLSyntaxError(
                f"strict=True: could not extract a DSL kernel from {func.__name__!r}"
            )
        use_dsl = strict is True or (strict is None and has_cf and dsl_ok)

        if use_dsl:
            dsl_wrapper = _jit_dsl_wrapper(kernel, out, kwargs)
            dsl_wrapper._blosc2_jit_wrapped = func
            return dsl_wrapper

        _trace_hint = None
        if strict is None and has_cf and not dsl_ok:
            _trace_hint = (
                f"Note: {func.__name__!r} contains control flow (if/for/while) but could not be "
                f"compiled as a DSL kernel: {kernel.dsl_error or 'source unavailable'}. See "
                "doc/reference/dsl_syntax.md for the DSL syntax reference."
            )

        exec_kwargs = {
            k: v for k, v in kwargs.items() if k in _JIT_EXECUTION_TUNING_KWARGS and v is not None
        }
        storage_kwargs = {k: v for k, v in kwargs.items() if k not in _JIT_EXECUTION_TUNING_KWARGS}

        def wrapper(*args, **func_kwargs):
            # Get some kwargs in decorator for SimpleProxy constructor
            proxy_kwargs = {"chunks": kwargs.get("chunks"), "blocks": kwargs.get("blocks")}

            # Wrap the arguments in SimpleProxy objects if they are not NDArrays
            new_args = []
            for arg in args:
                if issubclass(type(arg), blosc2.Operand):
                    new_args.append(arg)
                else:
                    new_args.append(SimpleProxy(arg, **proxy_kwargs))
            # The same for the keyword arguments
            for key, value in func_kwargs.items():
                if issubclass(type(value), blosc2.Operand):
                    continue
                func_kwargs[key] = SimpleProxy(value, **proxy_kwargs)

            # Call function with the new arguments
            try:
                retval = func(*new_args, **func_kwargs)
            except Exception as e:
                # Notes rather than a re-raise: type(e)(msg) assumes a one-argument
                # constructor, and any exception needing more (or rejecting a bare
                # string) would surface as a TypeError instead of the real failure.
                for hint in (
                    _wide_frame_hint(e, getattr(func, "__name__", "the function"), _signature_params(func)),
                    _trace_hint,
                ):
                    if hint is not None:
                        e.add_note(hint)
                raise

            # Treat return value
            # If it is a numpy array, return it as is
            if isinstance(retval, np.ndarray):
                if storage_kwargs and any(v is not None for v in storage_kwargs.values()):
                    # But if storage kwargs are provided, return a NDArray instead.
                    # Only storage kwargs: asarray() rejects the execution-tuning
                    # ones, and there is nothing left to tune -- the function has
                    # already run.
                    return blosc2.asarray(retval, **storage_kwargs)
                return retval

            # In some instances, the return value is not a LazyExpr
            # (e.g. using a reduction as the last function, and using an `out` param)
            if not isinstance(retval, blosc2.LazyExpr):
                return retval

            # If the return value is a LazyExpr, compute it
            if out is not None:
                return retval.compute(out=out, **kwargs)
            if storage_kwargs and any(v is not None for v in storage_kwargs.values()):
                return retval.compute(**kwargs)
            # No storage kwargs: return a NumPy array (like retval[()]), but still
            # honor any execution-tuning kwargs (jit/jit_backend/fp_accuracy).
            return retval.compute(_getitem=True, **exec_kwargs)

        # Lets callers (notably the pandas engine below) tell an already-jitted
        # function from a plain one, so it is not decorated a second time.
        wrapper._blosc2_jit_wrapped = func
        return wrapper

    # Carry the engine on the decorator too, so a configured call such as
    # `blosc2.jit(strict=True)` is accepted by `df.apply(..., engine=...)`:
    # pandas gates on hasattr(engine, "__pandas_udf__") and then uses the engine
    # object itself as the decorator.
    decorator.__pandas_udf__ = PandasUdfEngine

    if func is None:
        return decorator
    else:
        return decorator(func)


class PandasUdfEngine:
    @staticmethod
    def _ensure_numpy_data(data):
        if not isinstance(data, np.ndarray):
            try:
                data = data.values
            except AttributeError as err:
                raise ValueError(
                    f"blosc2.jit received an object of type {type(data).__name__}, which is not "
                    "supported. Try casting your Series or DataFrame to a NumPy dtype."
                ) from err
        if data.dtype.kind not in "biufc":
            raise ValueError(
                f"blosc2.jit requires a numeric dtype, got {data.dtype!r}. The Blosc2 engine only "
                "supports vectorized numeric computations; cast non-numeric columns before using "
                "engine=blosc2.jit."
            )
        return data

    @classmethod
    def map(cls, data, func, args, kwargs, decorator, skip_na):
        """
        JIT a NumPy array element-wise. In the case of Blosc2, functions are
        expected to be vectorized NumPy operations, so the function is called
        once with the whole NumPy array, instead of calling the function once
        for each element.
        """
        if skip_na:
            raise NotImplementedError("The Blosc2 engine does not support na_action='ignore' in map.")
        values = cls._ensure_numpy_data(data)
        func = _decorate_once(func, decorator)
        return func(values, *args, **kwargs)

    @classmethod
    def apply(cls, data, func, args, kwargs, decorator, axis):
        """
        JIT a NumPy array by column or row. In the case of Blosc2, functions are
        expected to be vectorized NumPy operations, so the function is called
        with the NumPy array as the function parameter, instead of calling the
        function once for each column or row.
        """
        orig = data
        func_name = getattr(func, "__name__", "the function")
        uses_subscript, has_loop = (
            _analyze_row_func(_undecorated(func)) if hasattr(orig, "columns") else (False, False)
        )
        # The row-proxy route reads columns one at a time and never needs the
        # whole frame as one array, so a non-numeric column (a pandas string
        # column, say) is fine there and only `nrows` is wanted.
        if uses_subscript and axis in (1, "columns"):
            values = None
            nrows = len(orig)
        else:
            values = cls._ensure_numpy_data(data)
            nrows = values.shape[0]
        func = _decorate_once(func, decorator)
        if values is not None and (values.ndim == 1 or axis is None):
            # pandas Series.apply or pipe
            result = func(values, *args, **kwargs)
        elif axis in (0, "index"):
            # pandas apply(axis=0) column-wise
            result = [func(values[:, col_idx], *args, **kwargs) for col_idx in range(values.shape[1])]
            result = np.vstack(result).transpose()
        elif axis in (1, "columns"):
            if uses_subscript and has_loop:
                # row["colname"] combined with for/while: tracing would unroll
                # the loop eagerly at call time, growing the traced expression
                # with every iteration (a real per-row iteration count, like a
                # Newton-Raphson loop, blows this up well past practical). No
                # existing dispatch route can run this well; point at the one
                # that can instead of hanging or crashing confusingly.
                raise TypeError(
                    f"@blosc2.jit engine=... axis=1: {func_name!r} "
                    'combines row["colname"] access with a for/while loop, which cannot be '
                    "traced efficiently per-row. Call your @blosc2.jit function directly with "
                    "the DataFrame columns as separate array arguments instead: name its "
                    "parameters after the columns and call kernel(**df) -- see "
                    "doc/guides/pandas_engine.md."
                )
            if uses_subscript:
                # The `row["colname"]` idiom: replace the per-row Python loop
                # with one call over whole per-column arrays (row-proxy, see
                # `_PandasRowProxy`), extracted from the original DataFrame so
                # per-column dtypes survive.
                row_proxy = _PandasRowProxy(orig)
                result = func(row_proxy, *args, **kwargs)
                if not (isinstance(result, np.ndarray) and result.ndim == 1 and result.shape[0] == nrows):
                    raise TypeError(
                        '@blosc2.jit engine=... axis=1: functions using row["colname"] must '
                        f"return one scalar per row (shape ({nrows},)); got "
                        f"{result!r}. Returning multiple values per row is not supported here."
                    )
            else:
                # pandas apply(axis=1) row-wise: the historical per-row loop.
                # Fine for functions treating the row as a plain array (e.g.
                # `row + 1`); functions using row["colname"] are dispatched
                # above instead, since this loop hands each call a positional
                # ndarray row that does not support string subscripting.
                result = [func(values[row_idx, :], *args, **kwargs) for row_idx in range(values.shape[0])]
                result = np.vstack(result)
        else:
            raise NotImplementedError(f"Unknown axis '{axis}'. Use one of 0, 1 or None.")

        # pandas only reconstructs a DataFrame/Series for us when it called us
        # with `raw=True` data (a plain ndarray); when it handed us the
        # original DataFrame (`raw=False`, the default), we must return a
        # properly indexed pandas object ourselves, mirroring what pandas'
        # own raw=True code path does.
        if isinstance(result, np.ndarray) and hasattr(orig, "columns"):
            if result.ndim == 2:
                return orig.__class__(result, index=orig.index, columns=orig.columns)
            agg_axis = orig._get_agg_axis(orig._get_axis_number(axis))
            return orig._constructor_sliced(result, index=agg_axis)
        return result


jit.__pandas_udf__ = PandasUdfEngine
