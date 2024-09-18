#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################
from abc import ABC, abstractmethod

import blosc2
import numpy as np


class ProxySource(ABC):
    """
    Base interface for all supported sources in :ref:`Proxy`.

    In case the source is multidimensional, the attributes `shape`, `chunks`,
    `blocks` and `dtype` are also required when creating the :ref:`Proxy`.

    In case the source is unidimensional, the attributes `chunksize`, `typesize`
     and `nbytes` are required as well when creating the :ref:`Proxy`.
    These attributes do not need to be available when opening an already
     existing :ref:`Proxy`.
    """

    @abstractmethod
    def get_chunk(self, nchunk):
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


class Proxy(blosc2.Operand):
    """Proxy (with cache support) of an object following the :ref:`ProxySource` interface.

    This can be used to cache chunks of a regular data container
    which follows the :ref:`ProxySource` interface in an urlpath.
    """

    def __init__(self, src, urlpath=None, **kwargs):
        """
        Create a new :ref:`Proxy` to serve like a cache to save accessed
        chunks locally.

        Parameters
        ----------
        src: :ref:`ProxySource`
            The original container
        urlpath: str, optional
            The urlpath where to save the container that will work as a cache.

        Other parameters
        ----------------
        kwargs: dict, optional
            Keyword arguments supported:

                vlmeta: dict or None
                    A dictionary with different variable length metalayers.  One entry per metalayer:

                        key: bytes or str
                            The name of the metalayer.
                        value: object
                            The metalayer object that will be serialized using msgpack.

        """
        self.src = src
        self.urlpath = urlpath
        if kwargs is None:
            kwargs = {}
        self._cache = kwargs.pop("_cache", None)

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
                    urlpath=urlpath,
                    meta=meta,
                )
            else:
                self._cache = blosc2.SChunk(
                    chunksize=self.src.chunksize,
                    urlpath=urlpath,
                    cparams={"typesize": self.src.typesize},
                    meta=meta,
                )
                self._cache.fill_special(self.src.nbytes // self.src.typesize, blosc2.SpecialValue.UNINIT)
        self._schunk_cache = getattr(self._cache, "schunk", self._cache)
        vlmeta = kwargs.get("vlmeta", None)
        if vlmeta:
            for key in vlmeta:
                self._schunk_cache.vlmeta[key] = vlmeta[key]

    def fetch(self, item=None):
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
        >>> f"Slice data cache: {slice_data[:3, :2]}"
        Slice data cache:
                    [[0 1]
                    [2 3]
                    [4 5]]
        """
        if item is None:
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

    async def afetch(self, item=None):
        """
        Get the container used as cache with the requested data updated
        in an asynchronous way.

        Parameters
        ----------
        item: slice or list of slices, optional
            If not None, only the chunks that intersect with the slices
            in items will be retrieved if they have not been already.

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
        >>>           # If the next source is multidimensional, it must have the attributes:
        >>>           self.data = data
        >>>           f"Data shape: {self.shape}, Chunks: {self.chunks}"
        Data shape: (4, 5), Chunks: [2, 5]
        >>>           f"Blocks: {self.blocks}, Dtype: {self.dtype}"
        Blocks: [1, 5], Dtype: int64
        >>>     @property
        >>>     def shape(self):
        >>>           return self.data.shape
        >>>     @property
        >>>     def chunks(self):
        >>>           return self.data.chunks
        >>>     @property
        >>>     def blocks(self):
        >>>           return self.data.blocks
        >>>     @property
        >>>     def dtype(self):
        >>>           return self.data.dtype
        >>>     # This method must be present
        >>>     def get_chunk(self, nchunk):
        >>>           return self.data.get_chunk(nchunk)
        >>>     # This method is optional
        >>>     async def aget_chunk(self, nchunk):
        >>>           await asyncio.sleep(0.1) # Simulate an asynchronous operation
        >>>           return self.data.get_chunk(nchunk)
        >>> data = np.arange(20).reshape(4, 5)
        >>> chunks = [2, 5]
        >>> blocks = [1, 5]
        >>> data = blosc2.asarray(data, chunks=chunks, blocks=blocks)
        >>> source = MyProxySource(data)
        >>> proxy = blosc2.Proxy(source)
        >>> async def fetch_data():
        >>>        # Fetch a slice of the data from the proxy asynchronously
        >>>        slice_data = await proxy.afetch(slice(0, 2))
        >>>        # Note that only data fetched is shown, the rest is uninitialized
        >>>        f"Slice data cache: {slice_data[:]}"
        Slice data cache:
                [[0 1 2 3 4]
                [5 6 7 8 9]
                [0 0 0 0 0]
                [0 0 0 0 0]]
        >>>         # Fetch the full data from the proxy asynchronously
        >>>         full_data = await proxy.afetch()
        >>>         # Now, all data is shown, meaning the full data has been fetched
        >>>         f"Full data cache: {full_data[:]}"
        Full data cache:
                [[ 0  1  2  3  4]
                [ 5  6  7  8  9]
                [10 11 12 13 14]
                [15 16 17 18 19]]
        >>> asyncio.run(fetch_data())
        >>> # Using getitem to get a slice of the data
        >>> result = proxy[1:2, 1:3]
        [[6 7]]
        """
        if not callable(getattr(self.src, "aget_chunk", None)):
            raise NotImplementedError("afetch is only available if the source has an aget_chunk method")
        if item is None:
            # Full realization
            for info in self._schunk_cache.iterchunks_info():
                if info.special != blosc2.SpecialValue.NOT_SPECIAL:
                    chunk = await self.src.aget_chunk(info.nchunk)
                    self._schunk_cache.update_chunk(info.nchunk, chunk)
        else:
            # Get only a slice
            nchunks = blosc2.get_slice_nchunks(self._cache, item)
            for info in self._schunk_cache.iterchunks_info():
                if info.nchunk in nchunks and info.special != blosc2.SpecialValue.NOT_SPECIAL:
                    chunk = await self.src.aget_chunk(info.nchunk)
                    self._schunk_cache.update_chunk(info.nchunk, chunk)

        return self._cache

    def __getitem__(self, item):
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
        >>> slice_1 = proxy[0:3, 0:3]
        >>> f"Slice 1: {slice_1}"
        Slice 1:
        [[ 0  1  2]
        [ 5  6  7]
        [10 11 12]
        >>> slice_2 = proxy[2:5, 2:5]
        >>> f"Slice 2: {slice_2}"
        Slice 2:
        [[12 13 14]
        [17 18 19]
        [22 23 24]]
        """
        # Populate the cache
        self.fetch(item)
        return self._cache[item]

    @property
    def dtype(self):
        """The dtype of :paramref:`self` or None if the data is unidimensional
        """
        return self._cache.dtype if isinstance(self._cache, blosc2.NDArray) else None

    @property
    def shape(self):
        """The shape of :paramref:`self`
        """
        return self._cache.shape if isinstance(self._cache, blosc2.NDArray) else len(self._cache)

    def __str__(self):
        return f"Proxy({self.src}, urlpath={self.urlpath})"

    @property
    def vlmeta(self):
        """
        Get the vlmeta of the cache.

        See Also
        --------
        :ref:`SChunk.vlmeta`

        Examples
        --------
        >>> import numpy as np
        >>> import blosc2
        >>> data = np.arange(100).reshape(10, 10)
        >>> ndarray = blosc2.asarray(data)
        >>> proxy = blosc2.Proxy(ndarray)
        >>> f"VLMeta data: {proxy.vlmeta}"
        VLMeta data: <blosc2.schunk.vlmeta object at 0x1062cf250>
        """
        return self._schunk_cache.vlmeta

    @property
    def fields(self)-> dict:
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
        >>> data = np.zeros(16, dtype=[('field1', 'i4'), ('field2', 'f4')]).reshape(4, 4)
        >>> ndarray = blosc2.asarray(data)
        >>> proxy = blosc2.Proxy(ndarray)
        >>> # Get a dictionary of fields from the proxy, where each field can be accessed individually
        >>> fields_dict = proxy.fields
        >>> for field_name, field_proxy in fields_dict.items():
        >>>     f"Field name: {field_name}, Field data: {field_proxy}"
        Field name: field1, Field data: <blosc2.proxy.ProxyNDField object at 0x10c176c90>
        Field name: field2, Field data: <blosc2.proxy.ProxyNDField object at 0x103264bf0>
        >>> field1_data = fields_dict['field1'][:]
        >>> field1_data
        [[0 0 0 0]
        [0 0 0 0]
        [0 0 0 0]
        [0 0 0 0]]
        """
        _fields = getattr(self._cache, "fields", None)
        if _fields is None:
            return None
        return {key: ProxyNDField(self, key) for key in _fields}


class ProxyNDField(blosc2.Operand):
    def __init__(self, proxy: Proxy, field: str):
        self.proxy = proxy
        self.field = field
        self.shape = proxy.shape
        self.dtype = proxy.dtype

    def __getitem__(self, item: slice):
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
