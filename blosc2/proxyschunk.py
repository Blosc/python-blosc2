#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################
import blosc2
from abc import ABC, abstractmethod


class ProxySource(ABC):
    """
    Base interface for all supported sources in :ref:`ProxySChunk`.

    In case the source is multidimensional, the attributes `shape`, `chunks`,
    `blocks` and `dtype` are also required when creating the :ref:`ProxySChunk`.

    In case the source is unidimensional, the attributes `chunksize`, `typesize`
     and `nbytes` are required as well when creating the :ref:`ProxySChunk`.
    These attributes do not need to be available when opening an already
     existing :ref:`ProxySChunk`.
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


class ProxySChunk:
    """Class that implements a proxy (with cache support) of an object which
    follows the :ref:`ProxySource` interface.

    This can be used to cache chunks of
    a regular data container which follows the :ref:`ProxySource` interface in an urlpath.
    """
    def __init__(self, src, urlpath=None, **kwargs):
        """
        Create a new :ref:`ProxySChunk` to serve like a cache to save accessed
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
        self._cache = kwargs.pop('_cache', None)

        if self._cache is None:
            meta_val = {'local_abspath': None, 'urlpath': None,
                        'caterva2_env': kwargs.pop('caterva2_env', False)}
            container = getattr(self.src, 'schunk', self.src)
            if hasattr(container, 'urlpath'):
                meta_val['local_abspath'] = container.urlpath
            elif isinstance(self.src, blosc2.C2Array):
                meta_val['urlpath'] = (self.src.path, self.src.urlbase, self.src.auth_token)
            meta = {'proxy-source': meta_val}
            if hasattr(self.src, "shape"):
                self._cache = blosc2.empty(self.src.shape, self.src.dtype, chunks=self.src.chunks,
                                           blocks=self.src.blocks, urlpath=urlpath, meta=meta)
            else:
                self._cache = blosc2.SChunk(chunksize=self.src.chunksize, urlpath=urlpath,
                                            cparams={'typesize': self.src.typesize}, meta=meta)
                self._cache.fill_special(self.src.nbytes // self.src.typesize, blosc2.SpecialValue.UNINIT)
        self._schunk_cache = getattr(self._cache, 'schunk', self._cache)
        vlmeta = kwargs.get('vlmeta', None)
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

    def __getitem__(self, item):
        """
        Get a slice as a numpy.ndarray using the :ref:`ProxySChunk`.

        Parameters
        ----------
        item: slice or list of slices
            The slice of the desired data.

        Returns
        -------
        out: numpy.ndarray
            An array with the data slice.
        """
        # Populate the cache
        self.fetch(item)
        return self._cache[item]

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
        This method is only available if the :ref:`ProxySource` has an
        async `aget_chunk` method.
        """
        if not callable(getattr(self.src, 'aget_chunk', None)):
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

    @property
    def dtype(self):
        """The dtype of :paramref:`self` or None if the data is unidimensional"""
        return self._cache.dtype if isinstance(self._cache, blosc2.NDArray) else None

    @property
    def shape(self):
        """The shape of :paramref:`self`"""
        return self._cache.shape if isinstance(self._cache, blosc2.NDArray) else len(self._cache)

    def __str__(self):
        return f"ProxySChunk({self.src}, urlpath={self.urlpath})"

    @property
    def vlmeta(self):
        """
        Get the vlmeta of the cache.

        See Also
        --------
        :ref:`SChunk.vlmeta`
        """
        return self._schunk_cache.vlmeta
