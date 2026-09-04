#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""Persistable references to remote arrays."""

from __future__ import annotations

import math
import os
from urllib.parse import parse_qsl, urlsplit

import numpy as np

import blosc2
from blosc2.b2objects import make_b2object_carrier, write_b2object_payload
from blosc2.info import InfoReporter, format_nbytes_info

DEFAULT_MEMORY_CACHE_BYTES = 256 * 2**20


class _PolicyDefault:
    def __repr__(self) -> str:
        return "<policy default>"


_POLICY_DEFAULT = _PolicyDefault()
_SENSITIVE_QUERY_PARTS = (
    "credential",
    "signature",
    "signed",
    "token",
    "password",
    "secret",
    "key",
    "sig",
    "expires",
)


def _validate_persistable_url(url: str) -> None:
    """Reject URL features that would put credentials in a portable carrier."""
    if "::" in url:
        raise ValueError("RemoteProxy does not persist chained fsspec URLs")
    parsed = urlsplit(url)
    if not parsed.scheme:
        raise ValueError("RemoteProxy requires a remote URL or a Caterva2 URLPath")
    if parsed.scheme.lower() in {"file", "local"}:
        raise ValueError("RemoteProxy does not persist local filesystem URLs")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("RemoteProxy URLs cannot contain user information")
    if parsed.fragment:
        raise ValueError("RemoteProxy URLs cannot contain fragments")
    sensitive = [
        key
        for key, _ in parse_qsl(parsed.query, keep_blank_values=True)
        if any(part in key.lower() for part in _SENSITIVE_QUERY_PARTS)
    ]
    if sensitive:
        raise ValueError("RemoteProxy URLs cannot contain credential-like query parameters")


def _normalize_limit(policy, value):
    if policy is blosc2.CachePolicy.NONE:
        if value is not _POLICY_DEFAULT:
            raise ValueError("max_cache_bytes is not applicable to CachePolicy.NONE")
        return None
    if value is _POLICY_DEFAULT:
        return DEFAULT_MEMORY_CACHE_BYTES if policy is blosc2.CachePolicy.MEMORY else None
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("max_cache_bytes must be a positive integer or None")
    if value <= 0:
        raise ValueError("max_cache_bytes must be a positive integer or None")
    return value


def _validate_max_concurrency(value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("max_concurrency must be a positive integer")
    if value <= 0:
        raise ValueError("max_concurrency must be a positive integer")
    return value


class RemoteProxy(blosc2.Operand):
    """A persistable reference to a Caterva2 or fsspec remote array.

    Unlike :class:`Proxy`, the object serialized by :meth:`save` is never the
    data cache. It is a metadata-only carrier which reopens with
    :attr:`CachePolicy.NONE`, regardless of the live proxy's runtime policy.
    Memory and disk caches are runtime choices and are not serialized.

    Parameters
    ----------
    urlpath: str, URLPath, or C2Array
        A single-file B2ND URL opened through fsspec, or a Caterva2 array
        reference.
    cache_policy: CachePolicy
        ``NONE`` retains no array data between operations. ``MEMORY`` retains
        compressed data in memory. ``DISK`` retains it in ``cache_path`` or
        under ``cache_dir``.
    cache_path: str or path-like, optional
        Exact persistent cache filename. Only valid with ``DISK`` and mutually
        exclusive with ``cache_dir``.
    cache_dir: str or path-like, optional
        Directory in which a source-derived persistent cache filename is made.
        Only valid with ``DISK``.
    max_cache_bytes: int or None, optional
        Post-operation compressed-payload bound. It defaults to 256 MiB for
        ``MEMORY`` and unlimited for ``DISK``; explicit ``None`` means
        unlimited. It is not applicable to ``NONE``.
    max_concurrency: int, optional
        Maximum number of independent remote fetches in flight.
    """

    def __init__(
        self,
        urlpath,
        *,
        cache_policy=blosc2.CachePolicy.NONE,
        cache_path=None,
        cache_dir=None,
        max_cache_bytes=_POLICY_DEFAULT,
        max_concurrency: int | None = None,
    ):
        if not isinstance(cache_policy, blosc2.CachePolicy):
            raise TypeError("cache_policy must be a blosc2.CachePolicy instance")
        if cache_dir is not None and cache_path is not None:
            raise ValueError("cache_dir and cache_path are mutually exclusive")
        if cache_policy is not blosc2.CachePolicy.DISK and (cache_dir is not None or cache_path is not None):
            raise ValueError("cache_dir and cache_path require CachePolicy.DISK")
        if cache_policy is blosc2.CachePolicy.DISK and cache_dir is None and cache_path is None:
            raise ValueError("CachePolicy.DISK requires cache_dir or cache_path")

        self._cache_policy = cache_policy
        self._cache_limit = _normalize_limit(cache_policy, max_cache_bytes)
        max_concurrency = _validate_max_concurrency(max_concurrency)
        self.src, self._source = self._open_source(urlpath, max_concurrency)
        self._proxy = None

        if cache_policy is not blosc2.CachePolicy.NONE:
            identity = self._source_identity()
            self._proxy = blosc2.schunk._lazy_remote_proxy(
                self.src,
                identity,
                cache_dir,
                cache_path,
                source_fresh=True,
                max_cache_bytes=self._cache_limit,
            )

    @staticmethod
    def _open_source(urlpath, max_concurrency):
        if isinstance(urlpath, blosc2.C2Array):
            src = urlpath
            if src.urlbase is not None:
                _validate_persistable_url(src.urlbase)
            source = {
                "kind": "caterva2",
                "version": 1,
                "path": src.path,
                "urlbase": src.urlbase,
            }
        elif isinstance(urlpath, blosc2.URLPath):
            if urlpath.urlbase is not None:
                _validate_persistable_url(urlpath.urlbase)
            src = blosc2.C2Array(urlpath.path, urlbase=urlpath.urlbase, auth_token=urlpath.auth_token)
            source = {
                "kind": "caterva2",
                "version": 1,
                "path": src.path,
                "urlbase": src.urlbase,
            }
        elif isinstance(urlpath, str):
            _validate_persistable_url(urlpath)
            kwargs = {} if max_concurrency is None else {"max_concurrency": max_concurrency}
            src = blosc2.FsspecNDSource(urlpath, **kwargs)
            source = {"kind": "fsspec", "version": 1, "urlpath": urlpath}
        else:
            raise TypeError("RemoteProxy requires a URL string, URLPath, or C2Array")

        if max_concurrency is not None and isinstance(src, blosc2.C2Array):
            src.max_concurrency = max_concurrency
        return src, source

    def _source_identity(self) -> str:
        if self._source["kind"] == "fsspec":
            return self._source["urlpath"]
        return f"caterva2:{blosc2.c2array._server_url(self.src.urlbase, self.src.path)}"

    def _validate_geometry(self, expected) -> None:
        if expected is None:
            return
        actual = (
            tuple(self.src.shape),
            np.dtype(self.src.dtype),
            tuple(self.src.chunks),
            tuple(self.src.blocks),
        )
        normalized = (
            tuple(expected[0]),
            np.dtype(expected[1]),
            tuple(expected[2]),
            tuple(expected[3]),
        )
        if actual != normalized:
            raise ValueError(
                "RemoteProxy source geometry no longer matches its carrier: "
                f"carrier={normalized}, source={actual}"
            )

    @property
    def shape(self):
        return tuple(self.src.shape)

    @property
    def dtype(self):
        return np.dtype(self.src.dtype)

    @property
    def ndim(self) -> int:
        """The number of dimensions in the remote array."""
        return len(self.shape)

    @property
    def chunks(self):
        return tuple(self.src.chunks)

    @property
    def blocks(self):
        return tuple(self.src.blocks)

    @property
    def cache_policy(self) -> blosc2.CachePolicy:
        """The immutable runtime retention policy."""
        return self._cache_policy

    @property
    def max_cache_bytes(self) -> int | None:
        """The immutable post-operation retained-cache bound."""
        return self._cache_limit

    @property
    def cparams(self):
        return self.src.cparams

    @property
    def traffic(self):
        return getattr(self.src, "traffic", None)

    @property
    def nbytes(self) -> int:
        """The uncompressed size of the remote array."""
        value = getattr(self.src, "nbytes", None)
        return int(value) if value is not None else math.prod(self.shape) * self.dtype.itemsize

    @property
    def info(self) -> InfoReporter:
        """A printable summary of this remote reference."""
        return InfoReporter(self)

    @property
    def info_items(self) -> list[tuple[str, object]]:
        """The fields shown by :attr:`info`."""
        return [
            ("type", type(self).__name__),
            ("source", self.source),
            ("shape", self.shape),
            ("chunks", self.chunks),
            ("blocks", self.blocks),
            ("dtype", self.dtype),
            ("nbytes", format_nbytes_info(self.nbytes)),
            ("cache_policy", self.cache_policy.name),
            ("cache_bytes", format_nbytes_info(self.cache_bytes)),
        ]

    @property
    def source(self) -> dict:
        """A copy of the credential-free source descriptor."""
        return dict(self._source)

    @property
    def urlpath(self):
        """The remote fsspec URL or credential-free Caterva2 URLPath."""
        if self._source["kind"] == "fsspec":
            return self._source["urlpath"]
        return blosc2.URLPath(self._source["path"], urlbase=self._source["urlbase"])

    @property
    def cache_path(self):
        """The runtime disk cache path, or ``None`` for other policies."""
        if self._proxy is None or self.cache_policy is not blosc2.CachePolicy.DISK:
            return None
        return self._proxy.urlpath

    @property
    def cache_status(self):
        """How a persistent disk cache was handled, or ``None`` otherwise."""
        if self._proxy is None:
            return None
        return self._proxy.cache_status

    @property
    def cache_bytes(self) -> int:
        """Compressed bytes currently retained by the runtime cache."""
        if self._proxy is None:
            return 0
        if self._proxy._max_cache_bytes is None:
            return self._proxy.schunk.cbytes
        return self._proxy._retained_cache_bytes()

    def __getitem__(self, item):
        if self._proxy is not None:
            return self._proxy[item]
        if isinstance(self.src, blosc2.C2Array):
            # Caterva2 can evaluate slices and fancy indices server-side.  In
            # particular, do not turn a no-cache C2 read into a chunk-by-chunk
            # client assembly operation just to satisfy the fsspec backend.
            return self.src[item]
        # fsspec exposes chunk/range reads rather than NumPy indexing.  Use an
        # operation-scoped Proxy so its temporary assembly state is discarded
        # as soon as this result is returned.
        proxy = blosc2.Proxy(self.src, _refresh_source=False)
        return proxy[item]

    def __len__(self) -> int:
        """The length of the first dimension, like other array operands."""
        if not self.shape:
            raise TypeError("len() of unsized object")
        return self.shape[0]

    def _chunk_slice(self, nchunk: int):
        grid = tuple(math.ceil(size / chunk) for size, chunk in zip(self.shape, self.chunks, strict=True))
        total = math.prod(grid)
        if nchunk < 0 or nchunk >= total:
            raise IndexError(f"nchunk must be in range [0, {total}), got {nchunk}")
        coords = np.unravel_index(nchunk, grid)
        return tuple(
            slice(int(coord) * chunk, min((int(coord) + 1) * chunk, size))
            for coord, chunk, size in zip(coords, self.chunks, self.shape, strict=True)
        )

    def get_chunk(self, nchunk: int) -> bytes:
        if self._proxy is None:
            return self.src.get_chunk(nchunk)
        item = self._chunk_slice(nchunk)
        self._proxy.fetch(item)
        chunk = self._proxy.schunk.get_chunk(nchunk)
        self._proxy._enforce_cache_limit(item)
        return chunk

    async def aget_chunk(self, nchunk: int) -> bytes:
        if self._proxy is None:
            method = getattr(self.src, "aget_chunk", None)
            if method is None:
                raise NotImplementedError("the remote source does not provide asynchronous chunk reads")
            return await method(nchunk)
        item = self._chunk_slice(nchunk)
        await self._proxy.afetch(item)
        chunk = self._proxy.schunk.get_chunk(nchunk)
        self._proxy._enforce_cache_limit(item)
        return chunk

    def _payload(self):
        return {
            "kind": "remote_proxy",
            "version": 1,
            "source": dict(self._source),
            # Persistence is intentionally reference-only. Runtime cache state
            # and local cache paths never cross this boundary.
            "cache_policy": blosc2.CachePolicy.NONE.value,
        }

    def _to_b2object_carrier(self, **kwargs):
        array = make_b2object_carrier(
            "remote_proxy",
            self.shape,
            self.dtype,
            chunks=self.chunks,
            blocks=self.blocks,
            cparams=self.cparams,
            **kwargs,
        )
        write_b2object_payload(array, self._payload())
        return array

    def to_cframe(self) -> bytes:
        """Serialize this reference, without cached array data, as a CFrame."""
        return self._to_b2object_carrier().to_cframe()

    def save(self, urlpath: str | os.PathLike, contiguous: bool = True, **kwargs) -> None:
        """Persist this reference without its runtime cache or credentials."""
        urlpath = os.fspath(urlpath)
        blosc2.blosc2_ext.check_access_mode(urlpath, "w")
        kwargs.update(urlpath=urlpath, contiguous=contiguous, mode="w")
        self._to_b2object_carrier(**kwargs)

    @classmethod
    def _from_payload(cls, payload, carrier):
        if set(payload) != {"kind", "version", "source", "cache_policy"}:
            raise ValueError("persisted RemoteProxy payload contains unsupported fields")
        if payload.get("cache_policy") != blosc2.CachePolicy.NONE.value:
            raise ValueError("persisted RemoteProxy objects must use cache policy 'none'")
        source = payload.get("source")
        if not isinstance(source, dict) or source.get("version") != 1:
            raise ValueError("unsupported RemoteProxy source descriptor")
        source_kind = source.get("kind")
        if source_kind == "fsspec":
            if set(source) != {"kind", "version", "urlpath"}:
                raise ValueError("fsspec RemoteProxy source descriptors contain unsupported fields")
            urlpath = source.get("urlpath")
            if not isinstance(urlpath, str):
                raise TypeError("fsspec RemoteProxy sources require a string 'urlpath'")
        elif source_kind == "caterva2":
            if set(source) != {"kind", "version", "path", "urlbase"}:
                raise ValueError("Caterva2 RemoteProxy source descriptors contain unsupported fields")
            path = source.get("path")
            urlbase = source.get("urlbase")
            if not isinstance(path, str) or (urlbase is not None and not isinstance(urlbase, str)):
                raise TypeError("Caterva2 RemoteProxy sources require string 'path' and 'urlbase' fields")
            urlpath = blosc2.URLPath(path, urlbase=urlbase)
        else:
            raise ValueError(f"unsupported RemoteProxy source kind: {source_kind!r}")
        expected = (carrier.shape, carrier.dtype, carrier.chunks, carrier.blocks)
        obj = cls(urlpath, cache_policy=blosc2.CachePolicy.NONE)
        obj._validate_geometry(expected)
        return obj

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def __str__(self):
        return f"RemoteProxy({self._source_identity()!r}, cache_policy={self.cache_policy.name})"
