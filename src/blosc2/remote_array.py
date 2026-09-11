#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""Persistable references to remote arrays."""

from __future__ import annotations

import asyncio
import contextlib
import math
import os
import threading
import weakref
from collections.abc import Mapping
from contextlib import nullcontext
from functools import wraps
from types import SimpleNamespace
from typing import Any
from urllib.parse import parse_qsl, urlsplit, urlunsplit

import numpy as np

import blosc2
from blosc2.b2objects import (
    _B2OBJECT_USER_VLMETA_KEY,
    make_b2object_carrier,
    read_b2object_user_vlmeta,
    write_b2object_payload,
    write_b2object_user_vlmeta,
)
from blosc2.core import parse_container_url, storage_options_fingerprint
from blosc2.info import InfoReporter, format_nbytes_info

DEFAULT_DISK_CACHE_BYTES = 256 * 2**20


class RemoteMetadataMapping(Mapping):
    """Read-only dictionary-like mapping of remote array metadata."""

    def __init__(self, data: Mapping | None = None):
        self._data = dict(data) if data is not None else {}

    def __getitem__(self, key: str | slice) -> Any:
        if isinstance(key, slice):
            if key.start is None and key.stop is None and key.step is None:
                return self.getall()
            raise NotImplementedError("Slicing is not supported, unless [:]")
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def getall(self) -> dict[str, Any]:
        return self._data.copy()

    def copy(self) -> dict[str, Any]:
        return self._data.copy()

    def __repr__(self) -> str:
        return repr(self._data)

    def __str__(self) -> str:
        return str(self._data)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Mapping):
            return self._data == dict(other)
        return False


class _PolicyDefault:
    def __repr__(self) -> str:
        return "<policy default>"


CACHE_POLICY_DEFAULT = _PolicyDefault()
_INTERNAL_CARRIER_METALAYERS = frozenset({"b2o", "proxy", "proxy-source"})
_C2_INTERNAL_VLMETA_KEYS = frozenset({"fill_nonce", "fill_state"})
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


def _normalize_source_format(urlpath, source_format):
    if source_format not in {None, "blosc2", "zarr", "hdf5", "b2z"}:
        raise ValueError("source_format must be None, 'blosc2', 'zarr', 'hdf5', or 'b2z'")
    if source_format is not None:
        return source_format
    if isinstance(urlpath, blosc2.ZarrNDSource):
        return "zarr"
    if isinstance(getattr(blosc2, "HDF5NDSource", None), type) and isinstance(urlpath, blosc2.HDF5NDSource):
        return "hdf5"
    if isinstance(urlpath, str):
        _, _, hint = parse_container_url(urlpath)
        if hint is not None:
            return hint
    return "blosc2"


def _validate_assume_immutable(value, name="assume_immutable"):
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool")
    return value


def _serialized_operation(method):
    @wraps(method)
    def locked(self, *args, **kwargs):
        owner = getattr(self, "_store_owner", None)
        with owner.lock if owner is not None else nullcontext(), self._operation_lock:
            self._check_open()
            if owner is not None and getattr(owner, "shared", False):
                self._proxy = owner.get_cache(self.src)
            cache_lock = (
                self._runtime_cache.holding_lock()
                if self._shared_runtime_cache and self._runtime_cache is not None
                else nullcontext()
            )
            with cache_lock:
                if self._shared_runtime_cache and self._proxy is not None:
                    self._proxy._refresh_shared_cache()
                try:
                    return method(self, *args, **kwargs)
                finally:
                    if self._proxy is not None and getattr(self, "is_cache_mutable", True):
                        self._proxy._enforce_cache_limit(tuple(slice(0, 0) for _ in self.shape))

    return locked


def validate_persistable_url(url: str) -> None:
    """Reject URL features that would put credentials in a portable carrier."""
    if "::" in url:
        raise ValueError("RemoteArray does not persist chained fsspec URLs")
    parsed = urlsplit(url)
    if not parsed.scheme:
        raise ValueError("RemoteArray requires a remote URL or a Caterva2 URLPath")
    if parsed.scheme.lower() in {"file", "local"}:
        raise ValueError("RemoteArray does not persist local filesystem URLs")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("RemoteArray URLs cannot contain user information")
    if parsed.fragment:
        raise ValueError("RemoteArray URLs cannot contain fragments")
    sensitive = [
        key
        for key, _ in parse_qsl(parsed.query, keep_blank_values=True)
        if any(part in key.lower() for part in _SENSITIVE_QUERY_PARTS)
    ]
    if sensitive:
        raise ValueError("RemoteArray URLs cannot contain credential-like query parameters")


def normalize_cache_limit(policy, value):
    if policy is blosc2.CachePolicy.NONE:
        if value is not CACHE_POLICY_DEFAULT:
            raise ValueError("max_cache_bytes is not applicable to CachePolicy.NONE")
        return None
    if value is CACHE_POLICY_DEFAULT:
        return DEFAULT_DISK_CACHE_BYTES
    if value is None:
        if policy is blosc2.CachePolicy.DISK:
            return None
        raise TypeError("max_cache_bytes must be a positive integer")
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("max_cache_bytes must be a positive integer")
    if value <= 0:
        raise ValueError("max_cache_bytes must be a positive integer")
    return value


def _validate_max_concurrency(value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("max_concurrency must be a positive integer")
    if value <= 0:
        raise ValueError("max_concurrency must be a positive integer")
    return value


def _validate_payload_limit(policy: blosc2.CachePolicy, limit) -> None:
    if policy is blosc2.CachePolicy.NONE:
        if limit is not None:
            raise ValueError("persisted NONE RemoteArray cannot have max_cache_bytes")
    elif policy is blosc2.CachePolicy.DISK:
        if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0):
            raise ValueError("persisted DISK RemoteArray requires positive max_cache_bytes or None")
    elif isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError(f"persisted {policy.name} RemoteArray requires positive max_cache_bytes")


def _validate_authorized_source(urlpath, storage_options, source_descriptor, *, store_attachment=False):
    if storage_options is not None:
        raise ValueError("storage_options cannot be used with an authorized source")
    hdf5_cls = getattr(blosc2, "HDF5NDSource", ())
    if not isinstance(urlpath, (blosc2.FsspecNDSource, blosc2.ZarrNDSource, hdf5_cls, blosc2.B2ZNDSource)):
        raise TypeError(
            "source_descriptor requires an authorized FsspecNDSource, ZarrNDSource, HDF5NDSource, or B2ZNDSource"
        )
    assume_immutable = _validate_assume_immutable(
        source_descriptor.get("assume_immutable"), "source_descriptor assume_immutable"
    )
    expected = {
        "kind": (
            "b2z"
            if isinstance(urlpath, blosc2.B2ZNDSource)
            else "hdf5"
            if isinstance(urlpath, hdf5_cls)
            else "zarr"
            if isinstance(urlpath, blosc2.ZarrNDSource)
            else "fsspec"
        ),
        "version": 1,
        "urlpath": urlpath.urlpath,
        "assume_immutable": assume_immutable,
    }
    if isinstance(urlpath, (hdf5_cls, blosc2.B2ZNDSource)):
        expected["dataset"] = urlpath.dataset
    if isinstance(urlpath, blosc2.B2ZNDSource) and urlpath._archive.urlpath != urlpath.urlpath:
        raise ValueError("B2Z source URL does not match its archive")
    if source_descriptor != expected:
        raise ValueError("source_descriptor does not match the supplied source")
    validate_persistable_url(urlpath.urlpath)
    return urlpath, dict(expected)


def _open_url_source(
    urlpath: str,
    max_concurrency: int | None,
    *,
    traffic=None,
    persistable=True,
    storage_options=None,
    source_format=None,
    assume_immutable=True,
    dataset=None,
    refs=None,
    blocks=None,
    cparams=None,
):
    if persistable:
        validate_persistable_url(urlpath)
    kwargs = {} if max_concurrency is None else {"max_concurrency": max_concurrency}
    if storage_options is not None:
        kwargs["storage_options"] = storage_options
    source_format = _normalize_source_format(urlpath, source_format)
    if source_format == "zarr":
        if not assume_immutable:
            raise NotImplementedError("mutable Zarr sources are not supported")
        src = blosc2.ZarrNDSource(urlpath, _traffic=traffic, blocks=blocks, cparams=cparams, **kwargs)
        source = {
            "kind": "zarr",
            "version": 1,
            "urlpath": urlpath,
            "assume_immutable": assume_immutable,
        }
    elif source_format == "hdf5":
        if not assume_immutable:
            raise NotImplementedError("mutable HDF5 sources are not supported")
        if dataset is None:
            raise ValueError("HDF5 sources require a dataset path (e.g., dataset='d0/d1/a2')")
        src = blosc2.HDF5NDSource(
            urlpath,
            dataset,
            refs=refs,
            _traffic=traffic,
            blocks=blocks,
            cparams=cparams,
            **kwargs,
        )
        source = {
            "kind": "hdf5",
            "version": 1,
            "urlpath": urlpath,
            "dataset": src.dataset,
            "assume_immutable": assume_immutable,
        }
    elif source_format == "b2z":
        if not assume_immutable:
            raise NotImplementedError("mutable B2Z sources are not supported")
        src = blosc2.B2ZNDSource(urlpath, dataset, _traffic=traffic, **kwargs)
        source = {
            "kind": "b2z",
            "version": 1,
            "urlpath": urlpath,
            "dataset": src.dataset,
            "assume_immutable": True,
        }
    else:
        src = blosc2.FsspecNDSource(urlpath, _traffic=traffic, **kwargs)
        source = {
            "kind": "fsspec",
            "version": 1,
            "urlpath": urlpath,
            "assume_immutable": assume_immutable,
        }
    return src, source


def _validate_cache_locations(cache_policy, cache_dir, cache_path, carrier, runtime_cache_path):
    if cache_dir is not None and cache_path is not None:
        raise ValueError("cache_dir and cache_path are mutually exclusive")
    if cache_policy is not blosc2.CachePolicy.DISK and (cache_dir is not None or cache_path is not None):
        raise ValueError("cache_dir and cache_path require CachePolicy.DISK")
    if (
        cache_policy is blosc2.CachePolicy.DISK
        and cache_dir is None
        and cache_path is None
        and carrier is None
        and runtime_cache_path is None
    ):
        raise ValueError("CachePolicy.DISK requires cache_dir or cache_path")


def _validate_urlpath_source(source, expected_fields, kind_name):
    if set(source) != expected_fields:
        raise ValueError(f"{kind_name} RemoteArray source descriptors contain unsupported fields")
    urlpath = source.get("urlpath")
    if not isinstance(urlpath, str):
        raise TypeError(f"{kind_name} RemoteArray sources require a string 'urlpath'")
    return urlpath


def _parse_source_from_payload(source):
    if not isinstance(source, dict) or source.get("version") != 1:
        raise ValueError("unsupported RemoteArray source descriptor")
    source_kind = source.get("kind")
    if source_kind == "fsspec":
        urlpath = _validate_urlpath_source(
            source, {"kind", "version", "urlpath", "assume_immutable"}, "fsspec"
        )
    elif source_kind == "caterva2":
        if set(source) != {"kind", "version", "path", "urlbase", "assume_immutable"}:
            raise ValueError("Caterva2 RemoteArray source descriptors contain unsupported fields")
        path = source.get("path")
        urlbase = source.get("urlbase")
        if not isinstance(path, str) or (urlbase is not None and not isinstance(urlbase, str)):
            raise TypeError("Caterva2 RemoteArray sources require string 'path' and 'urlbase' fields")
        urlpath = blosc2.URLPath(path, urlbase=urlbase)
    elif source_kind == "zarr":
        urlpath = _validate_urlpath_source(
            source, {"kind", "version", "urlpath", "assume_immutable"}, "Zarr"
        )
    elif source_kind in {"hdf5", "b2z"}:
        urlpath = _validate_urlpath_source(
            source, {"kind", "version", "urlpath", "dataset", "assume_immutable"}, source_kind.upper()
        )
        dataset = source.get("dataset")
        if not isinstance(dataset, str):
            raise TypeError(f"{source_kind.upper()} RemoteArray sources require a string 'dataset'")
    else:
        raise ValueError(f"unsupported RemoteArray source kind: {source_kind!r}")
    _validate_assume_immutable(source.get("assume_immutable"), "source assume_immutable")
    return source_kind, urlpath


def _resolve_init_dataset_and_url(urlpath, dataset, source_format):
    if isinstance(urlpath, (blosc2.URLPath, blosc2.C2Array)) and source_format is not None:
        raise ValueError("source_format is not supported for Caterva2 inputs")
    urlpath, parsed_dataset, detected_format = parse_container_url(urlpath, dataset)
    if dataset is None:
        dataset = parsed_dataset
    if source_format is None:
        source_format = detected_format
    resolved_format = _normalize_source_format(urlpath, source_format)
    if dataset is not None and resolved_format not in {"hdf5", "zarr", "b2z"}:
        raise ValueError("dataset is only supported for HDF5 and Zarr sources or B2Z archives")

    if resolved_format == "zarr":
        if dataset is not None:
            resolved_dataset = dataset.strip("/")
            if isinstance(urlpath, str):
                parsed = urlsplit(urlpath)
                clean_path = parsed.path.rstrip("/")
                if not clean_path.endswith(f"/{resolved_dataset}"):
                    new_path = f"{clean_path}/{resolved_dataset}"
                    urlpath = urlunsplit(
                        (
                            parsed.scheme,
                            parsed.netloc,
                            new_path,
                            parsed.query,
                            parsed.fragment,
                        )
                    )
            elif not str(urlpath).rstrip("/").endswith(f"/{resolved_dataset}"):
                urlpath = f"{str(urlpath).rstrip('/')}/{resolved_dataset}"
        elif isinstance(urlpath, str):
            parsed = urlsplit(urlpath)
            clean_path = parsed.path.rstrip("/")
            if ".zarr/" in clean_path.lower():
                idx = clean_path.lower().find(".zarr/")
                resolved_dataset = clean_path[idx + 6 :].strip("/") or None
            else:
                resolved_dataset = None
        else:
            resolved_dataset = None
    elif resolved_format in {"hdf5", "b2z"}:
        resolved_dataset = dataset.strip("/") if dataset is not None else None
    else:
        resolved_dataset = None

    return urlpath, resolved_dataset, resolved_format


class RemoteArray(blosc2.Operand):
    """A persistable, optionally self-caching reference to a remote array.

    With :attr:`CachePolicy.DISK`, the public constructor uses the persisted
    B2ND carrier itself as the bounded cache.  Server code can instead use
    :meth:`with_sparse_cache` to keep a private directory-backed runtime cache
    beside a portable carrier. With :attr:`CachePolicy.MEMORY`, chunks are
    retained in process memory up to a bounded size. With
    :attr:`CachePolicy.NONE`, reads retain no data.

    Parameters
    ----------
    urlpath: str, URLPath, or C2Array
        A B2ND or Zarr array URL, an HDF5 or B2Z container URL with a
        ``dataset`` selection, or a Caterva2 array reference.
    cache_policy: CachePolicy
        ``NONE`` retains no array data. ``MEMORY`` retains compressed chunks
        in client process memory. ``DISK`` retains compressed chunks in
        the RemoteArray carrier at ``cache_path`` or under ``cache_dir``.
    cache_path: str or path-like, optional
        Exact persistent cache filename. Only valid with ``DISK`` and mutually
        exclusive with ``cache_dir``.
    cache_dir: str or path-like, optional
        Directory in which a source-derived persistent cache filename is made.
        Only valid with ``DISK``.
    max_cache_bytes: int or None, optional
        Post-operation compressed-payload bound. It defaults to 256 MiB for
        ``DISK`` and ``MEMORY``. Passing ``None`` with ``DISK`` disables cache
        eviction (unbounded cache). ``MEMORY`` requires a finite positive integer.
        It is not applicable to ``NONE``.
    max_concurrency: int, optional
        Maximum number of independent remote fetches in flight.
    storage_options: dict, optional
        Parameters passed to the underlying ``fsspec`` filesystem when opening
        an fsspec URL.
    source_format: {None, "blosc2", "zarr", "hdf5", "b2z"}, optional
        Format of a URL source, inferred from its container suffix when omitted.
    dataset: str, optional
        Array path within an HDF5, Zarr, or B2Z container. B2Z supports external
        NDArray leaves in immutable archives, e.g. ``dataset="d0/a3"``.
    assume_immutable: bool, optional
        Skip remote identity checks before reads. Defaults to ``True``. Set to
        ``False`` when the object at the URL may be replaced.
    """

    def __init__(
        self,
        urlpath,
        *,
        cache_policy=blosc2.CachePolicy.NONE,
        cache_path=None,
        cache_dir=None,
        max_cache_bytes=CACHE_POLICY_DEFAULT,
        max_concurrency: int | None = None,
        storage_options: dict | None = None,
        source_format: str | None = None,
        assume_immutable: bool = True,
        dataset: str | None = None,
        refs=None,
        _carrier=None,
        _runtime_cache_path=None,
        _source_descriptor=None,
        _source_blocks=None,
        _source_cparams=None,
        _store_owner=None,
        _runtime_is_mutable: bool = True,
    ):
        if not isinstance(cache_policy, blosc2.CachePolicy):
            raise TypeError("cache_policy must be a blosc2.CachePolicy instance")
        assume_immutable = _validate_assume_immutable(assume_immutable)
        if _store_owner is None:
            _validate_cache_locations(cache_policy, cache_dir, cache_path, _carrier, _runtime_cache_path)

        self._store_owner = _store_owner
        self._store_generation = _store_owner.generation if _store_owner is not None else None
        self._cache_policy = cache_policy
        self._cache_limit = normalize_cache_limit(cache_policy, max_cache_bytes)
        self._max_concurrency = _validate_max_concurrency(max_concurrency)
        urlpath, self._dataset, self._source_format = _resolve_init_dataset_and_url(
            urlpath, dataset, source_format
        )
        self._authorized_source = _source_descriptor is not None
        if self._authorized_source:
            self.src, self._source = _validate_authorized_source(
                urlpath, storage_options, _source_descriptor, store_attachment=_store_owner is not None
            )
        else:
            if refs is None and _carrier is not None:
                raw_refs = getattr(_carrier, "schunk", _carrier).vlmeta.get("hdf5-refs")
                if raw_refs is not None:
                    try:
                        import ujson as json_mod
                    except ImportError:
                        import json as json_mod
                    refs = json_mod.loads(blosc2.decompress(raw_refs).decode("utf-8"))
            self.src, self._source = self._open_source(
                urlpath,
                self._max_concurrency,
                persistable=cache_policy is not blosc2.CachePolicy.MEMORY,
                storage_options=storage_options,
                source_format=self._source_format,
                assume_immutable=assume_immutable,
                dataset=self._dataset,
                refs=refs,
                blocks=_source_blocks,
                cparams=_source_cparams,
            )
        self._assume_immutable = assume_immutable
        self._storage_options = storage_options
        self._runtime_urlpath = self._runtime_source(urlpath)
        self._expected_geometry = self._geometry(self.src)
        self._expected_cparams = self.src.cparams
        self._refresh_lock = threading.Lock()
        self._operation_lock = threading.RLock()
        self._proxy = None
        self._carrier = _carrier
        self._runtime_cache = _carrier if cache_policy is blosc2.CachePolicy.DISK else None
        self._shared_runtime_cache = _runtime_cache_path is not None
        self._cache_status = None
        self._cached_meta = None
        self._cached_vlmeta = None
        self._meta_mapping = None
        self._vlmeta_mapping = None
        self._runtime_is_mutable = _runtime_is_mutable
        self._mutable = False

        self._initialize_runtime_cache(cache_dir, cache_path, _runtime_cache_path)

        if self._carrier is not None:
            if self._cached_meta is None:
                self._cached_meta = self._meta_from_carrier(self._carrier)
            if self._cached_vlmeta is None:
                self._cached_vlmeta = read_b2object_user_vlmeta(self._carrier)

        if _store_owner is not None:
            _store_owner.acquire()
            self._store_owner = _store_owner
            self._store_finalizer = weakref.finalize(self, _store_owner.release)

    def _initialize_runtime_cache(self, cache_dir, cache_path, _runtime_cache_path):
        if self._store_owner is not None and self.cache_policy is not blosc2.CachePolicy.NONE:
            self._attach_carrier_cache()
        elif self.cache_policy is blosc2.CachePolicy.DISK:
            if _runtime_cache_path is not None:
                self._runtime_cache, self._cache_status = self._open_or_create_sparse_cache(
                    _runtime_cache_path
                )
            elif self._carrier is None:
                self._carrier, self._cache_status = self._open_or_create_carrier(cache_dir, cache_path)
                self._runtime_cache = self._carrier
            cache_lock = self._runtime_cache.holding_lock() if self._shared_runtime_cache else nullcontext()
            with cache_lock:
                self._attach_carrier_cache()
        elif self.cache_policy is blosc2.CachePolicy.MEMORY:
            self._attach_carrier_cache()

    def _check_open(self):
        finalizer = getattr(self, "_store_finalizer", None)
        if finalizer is not None and not finalizer.alive:
            raise RuntimeError("RemoteArray handle is closed")
        if self._store_owner is not None and self._store_generation != self._store_owner.generation:
            raise RuntimeError("RemoteArray handle is stale; look it up again after refresh")

    def close(self):
        """Release this store-derived handle; standalone handles retain their existing lifetime."""
        owner = getattr(self, "_store_owner", None)
        if owner is not None:
            with owner.lock, self._operation_lock:
                self._store_finalizer()
                self._proxy = None
                self._runtime_cache = None

    def _runtime_source(self, original):
        """Keep credentials in live process state, outside the descriptor."""
        if isinstance(self.src, blosc2.C2Array):
            return blosc2.URLPath(
                self.src.path,
                urlbase=self.src.urlbase,
                auth_token=self.src.auth_token,
            )
        return original

    @staticmethod
    def _geometry(src):
        return (
            tuple(src.shape),
            np.dtype(src.dtype),
            tuple(src.chunks),
            tuple(src.blocks),
        )

    def _open_or_create_carrier(self, cache_dir, cache_path):
        if cache_path is not None:
            path = os.fspath(cache_path)
            if os.path.isdir(path):
                raise ValueError("cache_path must name a file, not a directory")
        else:
            path = blosc2.schunk.fsspec_cache_path(self._cache_identity(), cache_dir, ".b2nd")
        if os.path.exists(path):
            kwargs = {"dparams": blosc2.DParams(nthreads=1)}
            carrier = blosc2.blosc2_ext.open(path, "a", 0, **kwargs)
            payload = carrier.schunk.vlmeta.get("b2o")
            if payload != self._payload(mutable=True):
                raise ValueError(
                    f"the RemoteArray carrier at {path} has a different specification; "
                    "open legacy Proxy caches directly with blosc2.open(cache_path), "
                    "or choose a new cache_path"
                )
            if self._source.get("kind") == "hdf5" and "hdf5-refs" not in carrier.schunk.vlmeta:
                refs = getattr(self.src, "_refs", None)
                if refs is not None:
                    try:
                        import ujson as json_mod
                    except ImportError:
                        import json as json_mod
                    carrier.schunk.vlmeta["hdf5-refs"] = blosc2.compress(
                        json_mod.dumps(refs).encode("utf-8"), typesize=1
                    )
            stored = carrier.schunk.vlmeta.get("proxy-stamp")
            current = getattr(self.src, "stamp", None)
            status = (
                "invalidated/rebuilt"
                if stored is not None and current is not None and stored != current
                else "reused"
            )
            if status == "reused":
                if self._cached_meta is None:
                    self._cached_meta = self._meta_from_carrier(carrier)
                if self._cached_vlmeta is None:
                    self._cached_vlmeta = read_b2object_user_vlmeta(carrier)
            return carrier, status
        carrier = self._to_b2object_carrier(urlpath=path, contiguous=True, mode="w", mutable=True)
        return carrier, "created"

    def _open_or_create_sparse_cache(self, cache_path):
        """Open a server-owned sparse runtime cache or create it cold.

        This is deliberately separate from ``cache_path`` in the public
        constructor: portable RemoteArray carriers remain contiguous files.
        """
        path = os.fspath(cache_path)
        if os.path.exists(path):
            if not os.path.isdir(path):
                raise ValueError("runtime_cache_path must name a sparse frame directory")
            runtime = blosc2.blosc2_ext.open(path, "a", 0, dparams=blosc2.DParams(nthreads=1), locking=True)
            if runtime.schunk.vlmeta.get("b2o") != self._payload(mutable=True):
                raise ValueError(f"the sparse runtime cache at {path} has a different specification")
            self._validate_geometry(
                (runtime.shape, runtime.dtype, runtime.chunks, runtime.blocks), src=self.src
            )
            stored = runtime.schunk.vlmeta.get("proxy-stamp")
            current = getattr(self.src, "stamp", None)
            status = (
                "invalidated/rebuilt"
                if stored is not None and current is not None and stored != current
                else "reused"
            )
            if status == "reused":
                if self._cached_meta is None:
                    self._cached_meta = self._meta_from_carrier(runtime)
                if self._cached_vlmeta is None:
                    self._cached_vlmeta = read_b2object_user_vlmeta(runtime)
            return runtime, status

        if self._carrier is not None:
            self._validate_warm_seed(self._carrier)

        runtime = self._to_b2object_carrier(
            urlpath=path, contiguous=False, mode="w", locking=True, mutable=True
        )
        if self._carrier is not None:
            self._import_warm_seed(self._carrier, runtime)
        return runtime, "created"

    def _import_warm_seed(self, seed, runtime) -> None:
        """Migrate valid warm chunks once into a newly-created runtime cache."""
        seed = self._validate_warm_seed(seed)
        seed_schunk = getattr(seed, "schunk", seed)
        if seed_schunk.vlmeta.get("proxy-dirty") is not None:
            return  # An interrupted seed has no trustworthy fetched bitmap.
        stamp = getattr(self.src, "stamp", None)
        if stamp is None or seed_schunk.vlmeta.get("proxy-stamp") != stamp:
            return

        bpc = seed_schunk.vlmeta.get("proxy-fetched-bpc", 1)
        if not isinstance(bpc, int) or bpc <= 0:
            return
        key = "proxy-fetched-blocks" if bpc > 1 else "proxy-fetched"
        fetched = seed_schunk.vlmeta.get(key)
        expected_size = (seed_schunk.nchunks * bpc + 7) // 8
        if not isinstance(fetched, bytes) or len(fetched) != expected_size:
            return
        for nchunk in range(seed_schunk.nchunks):
            start = nchunk * bpc
            if any(fetched[n // 8] >> (n % 8) & 1 for n in range(start, start + bpc)):
                runtime.schunk.update_chunk(nchunk, seed_schunk.get_chunk(nchunk))
        for name in (
            "proxy-cache-sizes",
            "proxy-fetched",
            "proxy-fetched-blocks",
            "proxy-fetched-bpc",
            "proxy-index",
            "proxy-stamp",
            _B2OBJECT_USER_VLMETA_KEY,
        ):
            value = seed_schunk.vlmeta.get(name)
            if value is not None:
                runtime.schunk.vlmeta[name] = value

    def _validate_warm_seed(self, seed):
        seed = getattr(seed, "cache", seed)
        seed_schunk = getattr(seed, "schunk", seed)
        self._validate_geometry((seed.shape, seed.dtype, seed.chunks, seed.blocks))
        seed_payload = seed_schunk.vlmeta.get("b2o")
        if (
            not isinstance(seed_payload, dict)
            or seed_payload.get("kind") != "remote_array"
            or seed_payload.get("source") != self._source
        ):
            raise ValueError("the warm carrier belongs to a different remote source")
        return seed

    @classmethod
    def with_sparse_cache(
        cls,
        urlpath,
        runtime_cache_path,
        *,
        carrier=None,
        source_descriptor=None,
        max_cache_bytes=CACHE_POLICY_DEFAULT,
        max_concurrency: int | None = None,
        assume_immutable: bool = True,
    ):
        """Attach an authorized remote source to a private sparse disk cache.

        This server-facing constructor keeps the portable carrier separate from
        the mutable directory-backed runtime cache.  All processes using the
        directory must construct it through this method so frame locking and
        interrupted-mutation recovery remain enabled.

        ``carrier`` is the portable RemoteArray carrier.  If it contains valid
        warm chunks when the sparse runtime cache is first created, those chunks
        are copied into the runtime cache.  Both copies continue to exist until
        the server replaces the portable carrier with a cold descriptor.  After
        migration, only the runtime cache is consulted for cached data, so an
        evicted chunk cannot be resurrected from the carrier.

        Pass an already-authorized ``FsspecNDSource`` and its credential-free
        ``source_descriptor`` to retain the caller's transport and source
        snapshot. The caller must authorize and refresh that snapshot before
        each attachment; this form never reopens its URL or refreshes its source.
        """
        return cls(
            urlpath,
            cache_policy=blosc2.CachePolicy.DISK,
            max_cache_bytes=max_cache_bytes,
            max_concurrency=max_concurrency,
            assume_immutable=assume_immutable,
            _carrier=carrier,
            _runtime_cache_path=runtime_cache_path,
            _source_descriptor=source_descriptor,
        )

    @_serialized_operation
    def read_cached(self, item=(), *, nchunk=None):
        """Return ``(hit, result)`` atomically, without fetching a missing block.

        The source must already have been authorized by the caller. A miss
        returns ``(False, None)``. This operation does not refresh the source.
        """
        if self._proxy is None:
            return False, None
        if nchunk is not None:
            item = self._chunk_slice(nchunk)
        if self._proxy._missing_blocks(item):
            return False, None
        result = self._proxy._cache[item] if nchunk is None else self.schunk.get_chunk(nchunk)
        if self.is_cache_mutable:
            self._proxy._enforce_cache_limit(item)
        return True, result

    @_serialized_operation
    def cache_contains(self, item=(), *, nchunk=None):
        """Check cached coverage; use ``read_cached`` for an atomic hit/read."""
        if nchunk is not None:
            item = self._chunk_slice(nchunk)
        return self._proxy is not None and not self._proxy._missing_blocks(item)

    @property
    def cached_payload_bytes(self):
        """Compressed resident payload accounting from the attached snapshot."""
        return 0 if self._proxy is None else sum(self._proxy._cache_sizes.values())

    @_serialized_operation
    def trim_cache(self, target_bytes, *, max_chunks=64):
        """Evict at most ``max_chunks`` LRU chunks toward a payload-byte target.

        Return the evicted logical chunk numbers. Allocated filesystem charge
        must be measured separately, including after a partially failed eviction.
        """
        for name, value in (("target_bytes", target_bytes), ("max_chunks", max_chunks)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not self.is_cache_mutable:
            raise RuntimeError("Cannot trim an immutable cache")
        if self._proxy is None:
            return ()
        return self._proxy._trim_cache(target_bytes, max_chunks=max_chunks)

    @staticmethod
    def trim_sparse_cache(runtime_cache_path, target_bytes, *, max_chunks=64):
        """Trim an offline private cache without constructing a remote source.

        The server must hold its generation lifecycle guard. Return
        ``(evicted_chunk_numbers, remaining_payload_bytes)``. A dirty cache is
        conservatively invalidated before trimming; filesystem charge still
        needs measurement because unreachable payload can remain on disk.
        """
        for value in (target_bytes, max_chunks):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("target_bytes and max_chunks must be non-negative integers")
        cache = blosc2.blosc2_ext.open(os.fspath(runtime_cache_path), "a", 0, locking=True)
        with cache.holding_lock():
            bpc = cache.schunk.vlmeta.get("proxy-fetched-bpc", 1)

            def unavailable(*args, **kwargs):
                raise RuntimeError("offline cache maintenance cannot fetch remote data")

            source = SimpleNamespace(
                shape=cache.shape,
                dtype=cache.dtype,
                chunks=cache.chunks,
                blocks=cache.blocks,
                cparams=cache.cparams,
                stamp=cache.schunk.vlmeta.get("proxy-stamp"),
                blocks_per_chunk=bpc,
                wants_blocks=unavailable,
                chunk_layout=unavailable,
                block_plan=unavailable,
                read_range=unavailable,
            )
            backend = blosc2.Proxy(source, _cache=cache, _refresh_source=False, _persistent_dirty=True)
            evicted = backend._trim_cache(target_bytes, max_chunks=max_chunks)
            return evicted, sum(backend._cache_sizes.values())

    def _attach_carrier_cache(self):
        if self._store_owner is not None and self.cache_policy is not blosc2.CachePolicy.NONE:
            self._proxy = self._store_owner.get_cache(self.src)
            return
        if self.cache_policy is blosc2.CachePolicy.DISK:
            if self._runtime_cache is None:
                self._proxy = None
                return
            if not getattr(self, "_runtime_is_mutable", True):
                self._runtime_cache.schunk.mode = "r"
                self._proxy = blosc2.Proxy(
                    self.src,
                    _cache=self._runtime_cache,
                    mode="r",
                    _refresh_source=False,
                    _max_cache_bytes=self._cache_limit,
                    _persistent_dirty=False,
                )
                return
            if getattr(self.src, "stamp", None) is None:
                # Without a stable validator, cached bytes cannot be trusted across
                # independent opens. Reads still work, but misses are not retained.
                self._proxy = None
                return
            self._proxy = blosc2.Proxy(
                self.src,
                _cache=self._runtime_cache,
                _refresh_source=False,
                _max_cache_bytes=self._cache_limit,
                _persistent_dirty=self._shared_runtime_cache,
            )
        elif self.cache_policy is blosc2.CachePolicy.MEMORY:
            self._proxy = blosc2.Proxy(
                self.src,
                _refresh_source=False,
                _max_cache_bytes=self._cache_limit,
            )
        else:
            self._proxy = None

    @staticmethod
    def _open_source(
        urlpath,
        max_concurrency,
        *,
        traffic=None,
        persistable=True,
        storage_options: dict | None = None,
        source_format: str | None = None,
        assume_immutable: bool = True,
        dataset: str | None = None,
        refs=None,
        blocks=None,
        cparams=None,
    ):
        if isinstance(urlpath, blosc2.C2Array):
            if source_format not in {None, "blosc2"}:
                raise ValueError("source_format is not supported for Caterva2 inputs")
            if storage_options is not None:
                raise ValueError("storage_options is only supported for fsspec URLs")
            src = urlpath
            if persistable and src.urlbase is not None:
                validate_persistable_url(src.urlbase)
            source = {
                "kind": "caterva2",
                "version": 1,
                "path": src.path,
                "urlbase": src.urlbase,
                "assume_immutable": assume_immutable,
            }
        elif isinstance(urlpath, blosc2.URLPath):
            if source_format not in {None, "blosc2"}:
                raise ValueError("source_format is not supported for Caterva2 URLPath inputs")
            if storage_options is not None:
                raise ValueError("storage_options is only supported for fsspec URLs")
            src = blosc2.C2Array(
                urlpath.path,
                urlbase=urlpath.urlbase,
                auth_token=urlpath.auth_token,
                _traffic=traffic,
            )
            source = {
                "kind": "caterva2",
                "version": 1,
                "path": src.path,
                "urlbase": src.urlbase,
                "assume_immutable": assume_immutable,
            }
        elif isinstance(urlpath, str):
            src, source = _open_url_source(
                urlpath,
                max_concurrency,
                traffic=traffic,
                persistable=persistable,
                storage_options=storage_options,
                source_format=source_format,
                assume_immutable=assume_immutable,
                dataset=dataset,
                refs=refs,
                blocks=blocks,
                cparams=cparams,
            )
        else:
            raise TypeError("RemoteArray requires a URL string, URLPath, or C2Array")

        if max_concurrency is not None and isinstance(src, blosc2.C2Array):
            src.max_concurrency = max_concurrency
        return src, source

    def _source_identity(self) -> str:
        if self._source["kind"] in {"hdf5", "b2z"}:
            return f"{self._source['urlpath']}::{self._source['dataset']}"
        if self._source["kind"] in {"fsspec", "zarr"}:
            return self._source["urlpath"]
        return f"caterva2:{blosc2.c2array._server_url(self.src.urlbase, self.src.path)}"

    def _cache_identity(self) -> str:
        """``_source_identity`` plus a fingerprint of the fsspec access options.

        Different endpoints or accounts can reach different bytes through the
        same URL, so source-derived cache names must not collide across them.
        """
        identity = self._source_identity()
        fingerprint = storage_options_fingerprint(self._storage_options)
        return f"{identity}::{fingerprint}" if fingerprint else identity

    def _validate_geometry(self, expected, *, src=None) -> None:
        if expected is None:
            return
        actual = self._geometry(self.src if src is None else src)
        normalized = (
            tuple(expected[0]),
            np.dtype(expected[1]),
            tuple(expected[2]),
            tuple(expected[3]),
        )
        if actual != normalized:
            raise ValueError(
                "RemoteArray source geometry no longer matches its carrier: "
                f"carrier={normalized}, source={actual}"
            )

    def _prepare_read(self):
        """Refresh source identity and return the backend for one operation."""
        self._check_open()
        if self._authorized_source or self._assume_immutable:
            return self._proxy if self._proxy is not None else self.src
        with self._refresh_lock:
            previous_stamp = getattr(self.src, "stamp", None)
            refresh = getattr(self.src, "refresh_identity", None)
            if refresh is None:
                refresh = getattr(self.src, "refresh_stamp", None)
            if refresh is not None:
                if isinstance(self.src, blosc2.C2Array):
                    refresh(force=True)
                else:
                    refresh()

            self._validate_geometry(self._expected_geometry)
            current_stamp = getattr(self.src, "stamp", None)
            source_changed = (
                previous_stamp is None or current_stamp is None or current_stamp != previous_stamp
            )
            if source_changed:
                fresh, _ = self._open_source(
                    self._runtime_urlpath,
                    self._max_concurrency,
                    traffic=self.traffic,
                    persistable=self.cache_policy is not blosc2.CachePolicy.MEMORY,
                    storage_options=self._storage_options,
                    source_format=self._source_format,
                    assume_immutable=self._assume_immutable,
                    dataset=self.dataset,
                    refs=getattr(self.src, "_refs", None),
                )
                if current_stamp is None and not isinstance(fresh, blosc2.C2Array):
                    # No stable validator means cached bytes cannot safely be
                    # carried from one independent operation to the next.
                    fresh.stamp = None
                self._validate_geometry(self._expected_geometry, src=fresh)
                self.src = fresh
                self._attach_carrier_cache()
                self._cached_meta = None
                self._cached_vlmeta = None
                self._meta_mapping = None
                self._vlmeta_mapping = None

            return self.src if self._proxy is None else self._proxy

    @property
    def shape(self):
        self._check_open()
        return self._expected_geometry[0]

    @property
    def dtype(self):
        self._check_open()
        return self._expected_geometry[1]

    @property
    def ndim(self) -> int:
        """The number of dimensions in the remote array."""
        return len(self.shape)

    @property
    def chunks(self):
        self._check_open()
        return self._expected_geometry[2]

    @property
    def blocks(self):
        self._check_open()
        return self._expected_geometry[3]

    @property
    def cache_policy(self) -> blosc2.CachePolicy:
        """The persisted retention policy."""
        self._check_open()
        return self._cache_policy

    @property
    def max_cache_bytes(self) -> int | None:
        """The persisted post-operation retained-cache bound."""
        self._check_open()
        return self._cache_limit

    @property
    def mutable(self) -> bool:
        """The export default mutability for future exports."""
        self._check_open()
        if self._store_owner is not None:
            return self._store_owner.mutable
        return getattr(self, "_mutable", False)

    @mutable.setter
    def mutable(self, value: bool) -> None:
        self._check_open()
        if not isinstance(value, bool):
            raise TypeError("mutable must be a boolean")
        if self._store_owner is not None:
            self._store_owner.mutable = value
        else:
            self._mutable = value

    @property
    def is_cache_mutable(self) -> bool:
        """Whether the currently opened cache is writable."""
        self._check_open()
        if self._store_owner is not None:
            return getattr(self._store_owner, "is_mutable", True)
        return getattr(self, "_runtime_is_mutable", True)

    @property
    def cparams(self):
        self._check_open()
        return self._expected_cparams

    @property
    def traffic(self):
        self._check_open()
        return getattr(self.src, "traffic", None)

    @property
    def nbytes(self) -> int:
        """The uncompressed size of the remote array."""
        self._check_open()
        value = getattr(self.src, "nbytes", None)
        return int(value) if value is not None else math.prod(self.shape) * self.dtype.itemsize

    @property
    def info(self) -> InfoReporter:
        """A printable summary of this remote reference."""
        self._check_open()
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
        self._payload()  # Runtime-only URLs must not escape as portable descriptors.
        return dict(self._source)

    @property
    def assume_immutable(self) -> bool:
        """Whether reads skip remote identity checks."""
        self._check_open()
        return self._assume_immutable

    @property
    def meta(self) -> RemoteMetadataMapping:
        """The fixed-length metalayers of the remote array."""
        self._prepare_read()
        if self._cached_meta is None:
            self._cached_meta = self._fetch_meta()
            self._meta_mapping = None
        if self._meta_mapping is None:
            self._meta_mapping = RemoteMetadataMapping(self._cached_meta)
        return self._meta_mapping

    @property
    def vlmeta(self) -> RemoteMetadataMapping:
        """The variable-length metadata of the remote array."""
        self._prepare_read()
        if self._cached_vlmeta is None:
            self._cached_vlmeta = self._fetch_vlmeta()
            self._vlmeta_mapping = None
        if self._vlmeta_mapping is None:
            self._vlmeta_mapping = RemoteMetadataMapping(self._cached_vlmeta)
        return self._vlmeta_mapping

    @property
    def attrs(self) -> RemoteMetadataMapping:
        """The read-only user attributes of the remote array."""
        return self.vlmeta

    @staticmethod
    def _meta_from_carrier(carrier) -> dict[str, Any]:
        carrier_schunk = getattr(carrier, "schunk", carrier)
        return {
            name: carrier_schunk.meta[name]
            for name in carrier_schunk.meta
            if name not in _INTERNAL_CARRIER_METALAYERS
        }

    def _fetch_meta(self) -> dict[str, Any]:
        if isinstance(self.src, blosc2.C2Array):
            # The Caterva2 REST API (api/info) does not carry Blosc2 fixed metalayers.
            return {}
        meta = getattr(self.src, "meta", None)
        if meta is not None and isinstance(meta, Mapping):
            return {name: meta[name] for name in meta if name not in _INTERNAL_CARRIER_METALAYERS}
        return {}

    def _fetch_vlmeta(self) -> dict[str, Any]:
        vlmeta = (
            self.src.attrs if isinstance(self.src, blosc2.C2Array) else getattr(self.src, "vlmeta", None)
        )
        if vlmeta is not None and isinstance(vlmeta, Mapping):
            if isinstance(self.src, blosc2.C2Array) and self.src.meta.get("attrs") is None:
                res = {k: v for k, v in vlmeta.items() if k not in _C2_INTERNAL_VLMETA_KEYS}
            elif isinstance(vlmeta, dict):
                res = vlmeta.copy()
            else:
                res = dict(vlmeta)
        else:
            res = {}
        if self._runtime_cache is not None:
            cache_lock = (
                self._runtime_cache.holding_lock()
                if self._shared_runtime_cache and hasattr(self._runtime_cache, "holding_lock")
                else nullcontext()
            )
            with self._operation_lock, cache_lock:
                with contextlib.suppress(Exception):
                    write_b2object_user_vlmeta(self._runtime_cache, res)
        return res

    @property
    def schunk(self):
        """The underlying carrier's or cache's :class:`SChunk`, or None if unattached."""
        self._check_open()
        if self._proxy is not None:
            return self._proxy.schunk
        if self._runtime_cache is not None:
            return getattr(self._runtime_cache, "schunk", self._runtime_cache)
        if self._carrier is not None:
            return getattr(self._carrier, "schunk", self._carrier)
        return None

    @property
    def cache(self):
        """The local container used as cache, or None if caching is disabled."""
        self._check_open()
        if self._proxy is not None:
            return getattr(self._proxy, "cache", getattr(self._proxy, "_cache", None))
        return self._runtime_cache

    @property
    def urlpath(self):
        """The remote fsspec URL or credential-free Caterva2 URLPath."""
        self._check_open()
        if self._source["kind"] in {"fsspec", "zarr", "hdf5", "b2z"}:
            return self._source["urlpath"]
        return blosc2.URLPath(self._source["path"], urlbase=self._source["urlbase"])

    @property
    def dataset(self) -> str | None:
        """The dataset path within a container source, or None."""
        self._check_open()
        return self._source.get("dataset", self._dataset)

    @property
    def cache_path(self):
        """The self-caching carrier path, or ``None`` for other policies."""
        self._check_open()
        if self._carrier is None or self.cache_policy is not blosc2.CachePolicy.DISK:
            return None
        return getattr(self._carrier.schunk, "urlpath", None)

    @property
    def runtime_cache_path(self):
        """The mutable sparse cache directory, when one is attached."""
        self._check_open()
        if not self._shared_runtime_cache or self._runtime_cache is None:
            return None
        return getattr(self._runtime_cache.schunk, "urlpath", None)

    @property
    def cache_status(self):
        """How a persistent disk cache was handled, or ``None`` otherwise."""
        self._check_open()
        return self._cache_status

    @property
    def cache_bytes(self) -> int:
        """Compressed bytes currently retained by the runtime cache."""
        self._check_open()
        if self._proxy is None:
            return 0
        if self._proxy._cache_coordinator is None:
            return self._proxy.schunk.cbytes
        return self._proxy._retained_cache_bytes()

    @_serialized_operation
    def __getitem__(self, item):
        backend = self._prepare_read()
        if isinstance(backend, blosc2.Proxy):
            if not self.is_cache_mutable:
                if backend._missing_blocks(item):
                    return blosc2.Proxy(self.src, _refresh_source=False)[item]
                return backend._cache[item]
            return backend[item]
        if isinstance(backend, blosc2.C2Array):
            # Caterva2 can evaluate slices and fancy indices server-side.  In
            # particular, do not turn a no-cache C2 read into a chunk-by-chunk
            # client assembly operation just to satisfy the fsspec backend.
            return backend[item]
        # fsspec exposes chunk/range reads rather than NumPy indexing.  Use an
        # operation-scoped Proxy so its temporary assembly state is discarded
        # as soon as this result is returned.
        proxy = blosc2.Proxy(backend, _refresh_source=False)
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

    @_serialized_operation
    def fetch(self, item=(), max_concurrency: int | None = None):
        """Fetch remote data into the cache container.

        Return this proxy, not a materialized array. Eviction may discard
        prefetched chunks. Use indexing for values or :meth:`materialize`
        for an independent NDArray. Requires MEMORY or DISK caching.
        """
        backend = self._prepare_read()
        if not isinstance(backend, blosc2.Proxy):
            raise NotImplementedError("fetch requires CachePolicy.DISK or CachePolicy.MEMORY")
        if not self.is_cache_mutable:
            raise RuntimeError("Cannot prefetch into an immutable cache; use indexing or get_chunk()")
        backend.fetch(item, max_concurrency=max_concurrency)
        backend._enforce_cache_limit(item)
        return self

    async def afetch(self, item=(), max_concurrency: int | None = None):
        """Prefetch in a worker thread and return this proxy, like :meth:`fetch`.

        Requires MEMORY or DISK. Cancelling the await does not interrupt an
        already running fetch, which retains the operation lock until done.
        """
        if self.cache_policy is blosc2.CachePolicy.NONE:
            raise NotImplementedError("afetch requires CachePolicy.DISK or CachePolicy.MEMORY")
        return await asyncio.to_thread(self.fetch, item, max_concurrency=max_concurrency)

    def materialize(self, item=(), **kwargs):
        """Return an independent NDArray containing the requested values.

        The output and temporary NumPy buffer are not bounded by max_cache_bytes.
        Keyword arguments are forwarded to blosc2.asarray.
        """
        return blosc2.asarray(self[item], **kwargs)

    @_serialized_operation
    def get_chunk(self, nchunk: int) -> bytes:
        backend = self._prepare_read()
        if not isinstance(backend, blosc2.Proxy):
            return backend.get_chunk(nchunk)
        item = self._chunk_slice(nchunk)
        if not self.is_cache_mutable:
            if backend._missing_blocks(item):
                return self.src.get_chunk(nchunk)
            return backend.schunk.get_chunk(nchunk)
        backend.fetch(item)
        chunk = backend.schunk.get_chunk(nchunk)
        backend._enforce_cache_limit(item)
        return chunk

    async def aget_chunk(self, nchunk: int) -> bytes:
        return await asyncio.to_thread(self.get_chunk, nchunk)

    def _payload(self, mutable=None):
        url = self._source.get("urlpath", self._source.get("urlbase"))
        if url is not None:
            validate_persistable_url(url)
        return {
            "kind": "remote_array",
            "version": 1,
            "source": dict(self._source),
            "cache_policy": self.cache_policy.value,
            "max_cache_bytes": self.max_cache_bytes,
            "mutable": self.mutable if mutable is None else mutable,
        }

    def _to_b2object_carrier(self, mutable=None, **kwargs):
        carrier_excluded = _INTERNAL_CARRIER_METALAYERS | {"b2nd"}
        if self._carrier is not None:
            kwargs.setdefault(
                "meta",
                {
                    name: self._carrier.schunk.meta[name]
                    for name in self._carrier.schunk.meta
                    if name not in carrier_excluded
                },
            )
        else:
            kwargs.setdefault(
                "meta",
                {name: self.meta[name] for name in self.meta if name not in carrier_excluded},
            )
        array = make_b2object_carrier(
            "remote_array",
            self.shape,
            self.dtype,
            chunks=self.chunks,
            blocks=self.blocks,
            cparams=self.cparams,
            **kwargs,
        )
        write_b2object_payload(array, self._payload(mutable=mutable))
        user_vlmeta = dict(self.vlmeta)
        if user_vlmeta:
            write_b2object_user_vlmeta(array, user_vlmeta)
        if self._source.get("kind") == "hdf5":
            refs = getattr(self.src, "_refs", None)
            if refs is not None:
                try:
                    import ujson as json_mod
                except ImportError:
                    import json as json_mod
                array.schunk.vlmeta["hdf5-refs"] = blosc2.compress(
                    json_mod.dumps(refs).encode("utf-8"), typesize=1
                )
            elif self._carrier is not None:
                carrier_schunk = getattr(self._carrier, "schunk", self._carrier)
                if "hdf5-refs" in carrier_schunk.vlmeta:
                    array.schunk.vlmeta["hdf5-refs"] = carrier_schunk.vlmeta["hdf5-refs"]
        return array

    def _export_carrier_with_policy(self, cache_policy, effective_mutable):
        carrier = self._to_b2object_carrier(mutable=effective_mutable)
        payload = self._payload(mutable=effective_mutable)
        payload["cache_policy"] = cache_policy.value
        if cache_policy is blosc2.CachePolicy.NONE:
            payload["max_cache_bytes"] = None
        elif cache_policy is blosc2.CachePolicy.DISK:
            payload["max_cache_bytes"] = (
                self.max_cache_bytes
                if self.cache_policy is not blosc2.CachePolicy.NONE
                else DEFAULT_DISK_CACHE_BYTES
            )
        else:
            payload["max_cache_bytes"] = self.max_cache_bytes or DEFAULT_DISK_CACHE_BYTES
        write_b2object_payload(carrier, payload)
        return carrier

    def _export_carrier(self, include_cache: bool, cache_policy=None, mutable=None):
        if not isinstance(include_cache, bool):
            raise TypeError("include_cache must be a boolean")
        effective_mutable = self.mutable if mutable is None else mutable
        if cache_policy is not None:
            if not isinstance(cache_policy, blosc2.CachePolicy):
                raise TypeError("cache_policy must be a blosc2.CachePolicy instance")
            return self._export_carrier_with_policy(cache_policy, effective_mutable)
        if include_cache and self._runtime_cache is not None:
            current_payload = getattr(self._runtime_cache, "schunk", self._runtime_cache).vlmeta.get(
                "b2o", {}
            )
            if current_payload.get("mutable") == effective_mutable:
                return self._runtime_cache
            carrier = self._to_b2object_carrier(mutable=effective_mutable)
            runtime_schunk = getattr(self._runtime_cache, "schunk", self._runtime_cache)
            for nchunk in range(runtime_schunk.nchunks):
                chunk = runtime_schunk.get_chunk(nchunk)
                if chunk is not None:
                    carrier.schunk.update_chunk(nchunk, chunk)
            for key in runtime_schunk.vlmeta:
                if key.startswith("proxy-") or key == _B2OBJECT_USER_VLMETA_KEY:
                    carrier.schunk.vlmeta[key] = runtime_schunk.vlmeta[key]
            return carrier
        if include_cache and self._store_owner is not None and self._proxy is not None:
            carrier = self._to_b2object_carrier(mutable=effective_mutable)
            for nchunk in self._proxy._cache_sizes:
                carrier.schunk.update_chunk(nchunk, self._proxy.schunk.get_chunk(nchunk))
            for key in self._proxy.schunk.vlmeta:
                if key.startswith("proxy-") or key == _B2OBJECT_USER_VLMETA_KEY:
                    carrier.schunk.vlmeta[key] = self._proxy.schunk.vlmeta[key]
            return carrier
        return self._to_b2object_carrier(mutable=effective_mutable)

    @_serialized_operation
    def to_cframe(
        self, *, include_cache: bool = True, cache_policy=None, mutable: bool | None = None
    ) -> bytes:
        """Export a carrier. Only DISK preserves warm chunks by default.

        An explicit cache_policy exports a cold carrier with that policy.
        """
        if mutable is not None and not isinstance(mutable, bool):
            raise TypeError("mutable must be a boolean")
        return self._export_carrier(include_cache, cache_policy, mutable=mutable).to_cframe()

    @_serialized_operation
    def save(
        self,
        urlpath: str | os.PathLike,
        contiguous: bool = True,
        *,
        include_cache: bool = True,
        cache_policy=None,
        mutable: bool | None = None,
        **kwargs,
    ) -> str:
        """Save a carrier; MEMORY exports are cold. See :meth:`to_cframe`.

        Return the written ``urlpath``.
        """
        if mutable is not None and not isinstance(mutable, bool):
            raise TypeError("mutable must be a boolean")
        urlpath = os.fspath(urlpath)
        if (cache_policy is not None or not include_cache) and any(
            path is not None and os.path.abspath(path) == os.path.abspath(urlpath)
            for path in (self.cache_path, self.runtime_cache_path)
        ):
            raise ValueError("cold or policy-changing export requires a different destination")
        carrier = self._export_carrier(include_cache, cache_policy, mutable=mutable)
        source_path = getattr(carrier.schunk, "urlpath", None)
        if source_path is not None and os.path.abspath(source_path) == os.path.abspath(urlpath):
            return urlpath
        blosc2.blosc2_ext.check_access_mode(urlpath, "w")
        carrier.save(urlpath, contiguous=contiguous, **kwargs)
        return urlpath

    @classmethod
    def _from_payload(cls, payload, carrier):
        allowed = {"kind", "version", "source", "cache_policy", "max_cache_bytes", "mutable"}
        if not set(payload).issubset(allowed) or not {
            "kind",
            "version",
            "source",
            "cache_policy",
            "max_cache_bytes",
        }.issubset(payload):
            raise ValueError("persisted RemoteArray payload contains unsupported fields")
        try:
            policy = blosc2.CachePolicy(payload.get("cache_policy"))
        except ValueError as exc:
            raise ValueError("persisted RemoteArray has an unsupported cache policy") from exc
        mutable = payload.get("mutable", False)
        if not isinstance(mutable, bool):
            raise ValueError("persisted RemoteArray mutable flag must be a boolean")
        limit = payload.get("max_cache_bytes")
        _validate_payload_limit(policy, limit)
        source = payload.get("source")
        source_kind, urlpath = _parse_source_from_payload(source)
        expected = (carrier.shape, carrier.dtype, carrier.chunks, carrier.blocks)
        kwargs = {} if policy is blosc2.CachePolicy.NONE else {"max_cache_bytes": limit}
        carrier_arg = carrier if policy is blosc2.CachePolicy.DISK else None
        refs = None
        if source_kind == "hdf5" and carrier is not None:
            raw_refs = getattr(carrier, "schunk", carrier).vlmeta.get("hdf5-refs")
            if raw_refs is not None:
                try:
                    import ujson as json_mod
                except ImportError:
                    import json as json_mod
                refs = json_mod.loads(blosc2.decompress(raw_refs).decode("utf-8"))
        carrier_mode = getattr(carrier.schunk, "mode", "r") if carrier is not None else "r"
        is_disk_file = carrier is not None and bool(getattr(carrier.schunk, "urlpath", None))
        is_runtime_mutable = mutable and (carrier_mode != "r" if is_disk_file else True)
        obj = cls(
            urlpath,
            cache_policy=policy,
            source_format=source_kind if source_kind in {"zarr", "hdf5", "b2z"} else None,
            dataset=source.get("dataset") if source_kind in {"hdf5", "b2z"} else None,
            refs=refs,
            assume_immutable=source["assume_immutable"],
            _carrier=carrier_arg,
            _source_blocks=carrier.blocks if source_kind in {"zarr", "hdf5"} else None,
            _source_cparams=carrier.cparams if source_kind in {"zarr", "hdf5"} else None,
            _runtime_is_mutable=is_runtime_mutable,
            **kwargs,
        )
        obj._mutable = False
        obj._validate_geometry(expected)
        if not obj.is_cache_mutable and limit is not None and obj.cache_bytes > limit:
            raise ValueError(
                f"Retained immutable payload ({obj.cache_bytes} bytes) exceeds max_cache_bytes ({limit}); "
                "use a larger allowance or produce a smaller/cold export"
            )
        if carrier is not None:
            if obj._cached_meta is None:
                obj._cached_meta = obj._meta_from_carrier(carrier)
            if obj._cached_vlmeta is None:
                obj._cached_vlmeta = read_b2object_user_vlmeta(carrier)
        return obj

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def __str__(self):
        return f"RemoteArray({self._source_identity()!r}, cache_policy={self.cache_policy.name})"
