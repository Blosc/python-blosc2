#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################
import blosc2


class ProxySChunk:
    """Class that implements a proxy (with cache support) of a Python-Blosc2 container.

    This can be used to cache chunks of
    a regular SChunk, NDArray or C2Array in an urlpath.
    """
    def __init__(self, src, urlpath=None, **kwargs):
        """
        Create a new :ref:`ProxySChunk` to serve like a cache to save accessed
        chunks locally.

        Parameters
        ----------
        src: :ref:`SChunk`, :ref:`NDArray` or :ref:`C2Array`
            The original container
        urlpath: str, optional
            The urlpath where to save the container that will work as a cache.
        """
        self.src = src
        self.urlpath = urlpath
        if kwargs is None:
            kwargs = {}
        self._cache = kwargs.pop('_cache', None)

        if self._cache is None:
            # TODO: decide whether to keep caterva2_env or not, since it does not seem to be useful
            meta_val = {'local_abspath': None, 'urlpath': None, 'caterva2_env': kwargs.pop('caterva2_env', False)}
            container = getattr(self.src, 'schunk', self.src)
            if hasattr(container, 'urlpath'):
                meta_val['local_abspath'] = container.urlpath
            elif isinstance(self.src, blosc2.C2Array):
                meta_val['urlpath'] = (self.src.path, self.src.urlbase, self.src.auth_token)
            meta = {'proxy': meta_val}
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

    def eval(self, item=None):
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
        self.eval(item)
        return self._cache[item]

    @property
    def dtype(self):
        """The dtype of :paramref:`self` or None if it comes from a :ref:`SChunk`"""
        return self._cache.dtype if isinstance(self._cache, blosc2.NDArray) else None

    @property
    def shape(self):
        """The shape of :paramref:`self`"""
        return self._cache.shape if isinstance(self._cache, blosc2.NDArray) else len(self._cache)

    def __str__(self):
        return f"ProxySChunk({self.src}, urlpath={self.urlpath})"

    def vlmeta(self):
        return self._schunk_cache.vlmeta
