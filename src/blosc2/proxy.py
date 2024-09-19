#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################
from abc import ABC, abstractmethod

import numpy as np

import blosc2


class ProxyNDSource(ABC):
    """
    Base interface for NDim sources in :ref:`Proxy`.
    """

    @property
    @abstractmethod
    def shape(self) -> tuple:
        """
        Return the shape of the source.
        """
        pass

    @shape.setter
    @abstractmethod
    def shape(self, shape: tuple):
        """
        Set the shape of the source.
        """
        pass

    @property
    @abstractmethod
    def chunks(self) -> tuple:
        """
        Return the chunks of the source.
        """
        pass

    @chunks.setter
    @abstractmethod
    def chunks(self, chunks: tuple):
        """
        Set the chunks of the source.
        """
        pass

    @property
    @abstractmethod
    def blocks(self) -> tuple:
        """
        Return the blocks of the source.
        """
        pass

    @blocks.setter
    @abstractmethod
    def blocks(self, blocks: tuple):
        """
        Set the blocks of the source.
        """
        pass

    @property
    @abstractmethod
    def dtype(self) -> np.dtype:
        """
        Return the dtype of the source.
        """
        pass

    @dtype.setter
    @abstractmethod
    def dtype(self, dtype: np.dtype):
        """
        Set the dtype of the source.
        """
        pass

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

    def aget_chunk(self, nchunk: int) -> bytes:
        """
        Return the compressed chunk in :paramref:`self` in an asynchronous way.

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
        This method is optional, and only available if the source has an async `aget_chunk` method.
        """
        raise NotImplementedError("aget_chunk is only available if the source has an aget_chunk method")


class ProxySource(ABC):
    """
    Base interface for sources of :ref:`Proxy` that are not NDim objects.
    """

    @property
    @abstractmethod
    def nbytes(self) -> int:
        """
        Return the nbytes in the source.
        """
        pass

    @nbytes.setter
    @abstractmethod
    def nbytes(self, nbytes: int):
        """
        Set the nbytes in the source.
        """
        pass

    @property
    @abstractmethod
    def chunksize(self) -> tuple:
        """
        Return the chunksize of the source.
        """
        pass

    @chunksize.setter
    @abstractmethod
    def chunksize(self, chunksize: tuple):
        """
        Set the chunksize of the source.
        """
        pass

    @property
    @abstractmethod
    def typesize(self) -> int:
        """
        Return the typesize of the source.
        """
        pass

    @typesize.setter
    @abstractmethod
    def typesize(self, typesize: int):
        """
        Set the typesize of the source.
        """
        pass

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

    def aget_chunk(self, nchunk: int) -> bytes:
        """
        Return the compressed chunk in :paramref:`self` in an asynchronous way.

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
        This method is optional, and only available if the source has an async `aget_chunk` method.
        """
        raise NotImplementedError("aget_chunk is only available if the source has an aget_chunk method")


class Proxy(blosc2.Operand):
    """Proxy (with cache support) of an object following the :ref:`ProxySource` interface.

    This can be used to cache chunks of a regular data container which follows the
    :ref:`ProxySource` or :ref:`ProxyNDSource` interfaces.
    """

    def __init__(self, src: ProxySource or ProxyNDSource, urlpath: str = None, **kwargs: dict):
        """
        Create a new :ref:`Proxy` to serve like a cache to save accessed chunks locally.

        Parameters
        ----------
        src: :ref:`ProxySource` or :ref:`ProxyNDSource`
            The original container.
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

    def fetch(self, item: slice | list[slice] = None) -> blosc2.NDArray | blosc2.schunk.SChunk:
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
        >>> full_data = proxy.fetch()
        >>> f"Full data cache: {full_data[:]}"
        Full data cache:
            [[ 0  1][ 2  3][ 4  5]
            [ 6  7][ 8  9][10 11]
            [12 13][14 15][16 17]
            [18 19]]
        >>> slice_data = proxy[0:2, :]
        >>> f"Slice data cache: {slice_data}"
        Slice data cache:
        [[0 1][2 3]]
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

    async def afetch(self, item: slice | list[slice] = None) -> blosc2.NDArray | blosc2.schunk.SChunk:
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
        >>> data = np.arange(100).reshape(10, 10)
        >>> ndarray = blosc2.asarray(data)
        >>> proxy = blosc2.Proxy(ndarray)
        >>> slice_1 = proxy[0:3, 0:3]
        >>> f"Slice 1: {slice_1}"
        Slice 1:
        [[ 0  1  2]
        [10 11 12]
        [20 21 22]]
        >>> slice_2 = proxy[5:8, 2:5]
        >>> f"Slice 2: {slice_2}"
        Slice 2:
        [[52 53 54]
        [62 63 64]
        [72 73 74]]
        """
        # Populate the cache
        self.fetch(item)
        return self._cache[item]

    @property
    def dtype(self) -> np.dtype:
        """The dtype of :paramref:`self` or None if the data is unidimensional"""
        return self._cache.dtype if isinstance(self._cache, blosc2.NDArray) else None

    @property
    def shape(self) -> tuple[int]:
        """The shape of :paramref:`self`"""
        return self._cache.shape if isinstance(self._cache, blosc2.NDArray) else len(self._cache)

    def __str__(self):
        return f"Proxy({self.src}, urlpath={self.urlpath})"

    @property
    def vlmeta(self) -> blosc2.schunk.vlmeta:
        """
        Get the vlmeta of the cache.

        See Also
        --------
        :ref:`SChunk.vlmeta`
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
