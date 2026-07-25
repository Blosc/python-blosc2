#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

import ast
import asyncio
import inspect
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
from blosc2.dsl_kernel import DSLKernel

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
        if not hasattr(src, "shape") or not hasattr(src, "dtype"):
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
        col = self._df[key].to_numpy()
        if col.dtype.kind not in "biufc":
            raise ValueError(
                f"row[{key!r}]: column has dtype {col.dtype!r}, which is not numeric. "
                "The Blosc2 engine only supports vectorized numeric computations."
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
        bound = sig.bind(*args, **func_kwargs)
        bound.apply_defaults()
        values = tuple(bound.arguments[name] for name in kernel.input_names)
        # Accept array-protocol operands (pandas Series, polars Series, ...) the
        # same way the tracing route already does; zero-copy when the source is
        # numpy-backed.
        values = tuple(
            np.asarray(v)
            if not isinstance(v, np.ndarray | blosc2.NDArray)
            and hasattr(v, "__array__")
            and getattr(v, "ndim", 0) > 0
            else v
            for v in values
        )

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
        - ``True``: always use the DSL route, raising at decoration time if
          *func* cannot be compiled as a DSL kernel.  Equivalent to
          ``blosc2.dsl_kernel``.
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
    >>> def compute_expression(a, b, c):
    >>>     return np.sum(((a ** 3 + np.sin(a * 2)) > 2 * c) & (b > 0), axis=1)
    >>> a = np.arange(20, dtype=np.float32).reshape(4, 5)
    >>> b = np.arange(20).reshape(4, 5)
    >>> c = np.arange(5)
    >>> compute_expression(a, b, c)
    [5 5 5 5]
    """

    def decorator(func):  # noqa: C901
        if disable:
            return func

        kernel = DSLKernel(func)
        has_cf = _has_control_flow(kernel.dsl_source)
        dsl_ok = kernel.dsl_source is not None and kernel.dsl_error is None
        if strict is True and not dsl_ok:
            raise kernel.dsl_error or TypeError(
                f"@blosc2.jit(strict=True): could not extract a DSL kernel from {func.__name__!r}"
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
                if _trace_hint is not None:
                    raise type(e)(f"{e}\n{_trace_hint}") from e
                raise

            # Treat return value
            # If it is a numpy array, return it as is
            if isinstance(retval, np.ndarray):
                if storage_kwargs and any(v is not None for v in storage_kwargs.values()):
                    # But if storage kwargs are provided, return a NDArray instead
                    return blosc2.asarray(retval, **kwargs)
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
        values = cls._ensure_numpy_data(data)
        func_name = getattr(func, "__name__", "the function")
        uses_subscript, has_loop = (
            _analyze_row_func(_undecorated(func)) if hasattr(orig, "columns") else (False, False)
        )
        func = _decorate_once(func, decorator)
        if values.ndim == 1 or axis is None:
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
                    "the DataFrame columns as separate array arguments instead, e.g. "
                    "kernel(df['a'], df['b']) -- see doc/guides/pandas_engine.md."
                )
            if uses_subscript:
                # The `row["colname"]` idiom: replace the per-row Python loop
                # with one call over whole per-column arrays (row-proxy, see
                # `_PandasRowProxy`), extracted from the original DataFrame so
                # per-column dtypes survive.
                row_proxy = _PandasRowProxy(orig)
                result = func(row_proxy, *args, **kwargs)
                if not (
                    isinstance(result, np.ndarray)
                    and result.ndim == 1
                    and result.shape[0] == values.shape[0]
                ):
                    raise TypeError(
                        '@blosc2.jit engine=... axis=1: functions using row["colname"] must '
                        f"return one scalar per row (shape ({values.shape[0]},)); got "
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
