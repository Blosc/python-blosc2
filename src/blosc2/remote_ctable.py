#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################
"""Read-only remote CTable access through fsspec-backed B2Z sources."""

from __future__ import annotations

import operator
import os  # noqa: TC003

from blosc2.ctable import CTable
from blosc2.ctable_storage import RemoteTableStorage
from blosc2.remote_array import CACHE_POLICY_DEFAULT, RemoteMetadataMapping
from blosc2.remote_object import RemoteObject


def _positive_integer(name, value):
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a positive integer")
    try:
        value = operator.index(value)
    except TypeError:
        raise TypeError(f"{name} must be a positive integer") from None
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _read_setting(name):
    def get(self):
        return getattr(self._remote_storage(), name)

    def set(self, value):
        value = _positive_integer(name, value)
        storage = self._remote_storage()
        with storage._owner.lock:
            storage._check_open()
            setattr(storage, name, value)

    return property(get, set)


class RemoteCTable(RemoteObject, CTable):
    """A read-only CTable whose columns are fetched on demand.

    Supported columns include fixed-width, UTF-8, batch-backed variable-length,
    batch-backed list, struct/object and dictionary columns. Batch reads transfer
    one whole compressed batch; dictionary decoding loads the full vocabulary on
    first use.

    Independent column requests overlap by default. ``max_concurrency`` defaults
    to 8; use 1 for serial reads. ``metadata_buffer_bytes`` (8 MiB) and
    ``row_buffer_bytes`` (64 MiB) bound temporary transport batches, not retained
    caches or total RAM. An indivisible oversized unit is read alone. These
    positive-integer settings can also be changed on an open table; views use
    their base table's settings. There is no automatic CPU/RAM-based tuning.

    ``blosc2.open`` accepts ``max_concurrency`` but not the table-specific buffer
    keywords. Use this constructor or the returned table's settings to tune
    buffers. Cache policies and ``max_cache_bytes`` remain independent.

    A RemoteCTable's cache policy applies to every column read through it,
    including columns backed by independent RemoteArray carriers. Those columns
    share the table owner's cache budget and traffic accounting; their persisted
    standalone policies are not used or modified by the table.

    ``hdf5_index`` accepts a native index dictionary or a local/remote JSON
    path for PyTables/HDF5 sources. Supplying one skips HDF5 discovery.

    ``path`` selects the table within the source. ``dataset`` remains a supported
    alias; when both are supplied they must agree after stripping outer slashes.
    None leaves selection unspecified; an empty string or slash selects the root.
    Selector keywords cannot be combined with a selector embedded in the URL.
    """

    def __new__(
        cls,
        urlpath=None,
        *,
        dataset=None,
        path=None,
        storage_options=None,
        cache_policy=CACHE_POLICY_DEFAULT,
        max_cache_bytes=CACHE_POLICY_DEFAULT,
        cache_dir=None,
        hdf5_index=None,
        max_concurrency=8,
        metadata_buffer_bytes=8 << 20,
        row_buffer_bytes=64 << 20,
        _filesystem=None,
    ):
        if urlpath is None:
            raise TypeError("RemoteCTable requires a remote B2Z URL")
        settings = {
            name: _positive_integer(name, value)
            for name, value in {
                "max_concurrency": max_concurrency,
                "metadata_buffer_bytes": metadata_buffer_bytes,
                "row_buffer_bytes": row_buffer_bytes,
            }.items()
        }

        from blosc2.remote_store import RemoteStore

        store = RemoteStore(
            urlpath,
            dataset=dataset,
            path=path,
            storage_options=storage_options,
            cache_policy=cache_policy,
            max_cache_bytes=max_cache_bytes,
            cache_dir=cache_dir,
            hdf5_index=hdf5_index,
            _allow_array_root=True,
            _filesystem=_filesystem,
        )
        try:
            _, full = store._resolve("")
            kind, diagnostic = store._owner.nodes[full]
            if kind != "ctable":
                if kind == "unsupported":
                    raise NotImplementedError(str(diagnostic))
                raise ValueError("RemoteCTable requires a CTable node")
            return cls._from_owner(store._owner, full, **settings)
        finally:
            store.close()

    def __init__(self, *args, **kwargs):
        # Construction is completed by CTable._open_from_storage() in __new__.
        pass

    @classmethod
    def with_sparse_cache(
        cls,
        urlpath,
        runtime_cache_path,
        *,
        dataset=None,
        path=None,
        manifest=None,
        max_cache_bytes=CACHE_POLICY_DEFAULT,
        carrier=None,
        storage_options=None,
        max_concurrency=8,
        metadata_buffer_bytes=8 << 20,
        row_buffer_bytes=64 << 20,
        _filesystem=None,
        _source_validator=None,
        _manifest_validator=None,
        _max_nodes=None,
    ):
        """Attach a remote CTable to a sparse disk cache shared across processes.

        ``path`` and ``dataset`` select the table as in the ordinary constructor.
        The aggregate compressed-payload budget defaults to 256 MiB; pass
        ``max_cache_bytes=None`` for unlimited retention. For ordinary shared
        caching, prefer ``blosc2.open(url, cache_dir=..., shared_cache=True)``.
        """
        settings = {
            name: _positive_integer(name, value)
            for name, value in {
                "max_concurrency": max_concurrency,
                "metadata_buffer_bytes": metadata_buffer_bytes,
                "row_buffer_bytes": row_buffer_bytes,
            }.items()
        }

        from blosc2.remote_store import RemoteStore

        store = RemoteStore.with_sparse_cache(
            urlpath,
            runtime_cache_path,
            dataset=dataset,
            path=path,
            manifest=manifest,
            max_cache_bytes=max_cache_bytes,
            carrier=carrier,
            storage_options=storage_options,
            _filesystem=_filesystem,
            _source_validator=_source_validator,
            _manifest_validator=_manifest_validator,
            _max_nodes=_max_nodes,
        )
        try:
            _, full = store._resolve("")
            kind, diagnostic = store._owner.nodes[full]
            if kind != "ctable":
                if kind == "unsupported":
                    raise NotImplementedError(str(diagnostic))
                raise ValueError("RemoteCTable requires a CTable node")
            return cls._from_owner(store._owner, full, **settings)
        finally:
            store.close()

    @classmethod
    def _from_owner(cls, owner, full_path, **settings):
        settings = {name: _positive_integer(name, value) for name, value in settings.items()}
        storage = RemoteTableStorage(owner, full_path, **settings)
        try:
            return cls._open_from_storage(storage)
        except BaseException:
            storage.close()
            raise

    max_concurrency = _read_setting("max_concurrency")
    metadata_buffer_bytes = _read_setting("metadata_buffer_bytes")
    row_buffer_bytes = _read_setting("row_buffer_bytes")

    def _remote_storage(self) -> RemoteTableStorage:
        storage = getattr(self, "_storage", None)
        if not isinstance(storage, RemoteTableStorage):
            raise RuntimeError("RemoteCTable handle is closed")
        storage._check_open()
        return storage

    def _check_open(self) -> None:
        self._remote_storage()

    def close(self) -> None:
        storage = getattr(self, "_storage", None)
        if isinstance(storage, RemoteTableStorage):
            storage.close()

    def refresh(self) -> None:
        """Reload a standalone table and invalidate its old columns and views.

        Preserve the table and its cache on discovery/initialization failure.
        For tables obtained from a RemoteStore, refresh the root store instead.
        """
        storage = self._remote_storage()
        owner = storage._owner
        with owner.lock:
            storage._check_open()
            if self.base is not None or owner.root != storage._root_key or owner.is_tree:
                raise ValueError("Refresh the root RemoteStore, then retrieve this table again")
            replacement = owner.prepare_refresh("ctable")
            replacement.acquire()  # Keep failed initialization from closing the borrowed disk cache.
            fresh = None
            try:
                fresh = type(self)._from_owner(
                    replacement,
                    replacement.root,
                    max_concurrency=storage.max_concurrency,
                    metadata_buffer_bytes=storage.metadata_buffer_bytes,
                    row_buffer_bytes=storage.row_buffer_bytes,
                )
                replacement.restoring = False
                replacement.save_manifest()
                replacement.publish_source_cache(refresh=True)
            except BaseException:
                replacement.disk = None
                if fresh is not None:
                    fresh.close()
                replacement.release()
                raise
            replacement.release()
            if getattr(replacement, "shared", False):
                from blosc2.remote_store_cache import SharedStoreOperation

                replacement.lock = SharedStoreOperation(replacement)
            replacement._cleanup_dir, owner._cleanup_dir = owner._cleanup_dir, None
            replacement.artifact_path = owner.artifact_path
            if not getattr(owner, "shared", False):
                owner.disk = None
            owner.generation = replacement.generation
            state = fresh.__dict__.copy()
            fresh._storage = None  # Ownership is transferred to this handle.
            self.__dict__ = state
            self._cols._table = self
            storage.close()
            owner.close()
            if replacement.disk is not None:
                replacement.disk.discard_old_generations(replacement.generation)

    @property
    def vlmeta(self):
        return RemoteMetadataMapping(self._remote_storage().load_user_attrs())

    @property
    def attrs(self):
        """Read-only user attributes."""
        return self.vlmeta

    @property
    def source(self):
        storage = self._remote_storage()
        return {
            "kind": storage._owner.format,
            "version": 1,
            "urlpath": storage._owner.urlpath,
            "dataset": storage._root_key,
            "assume_immutable": True,
        }

    @property
    def traffic(self):
        return self._remote_storage()._owner.traffic

    @property
    def cache_policy(self):
        return self._remote_storage()._owner.cache_policy

    @property
    def max_cache_bytes(self):
        return self._remote_storage()._owner.max_cache_bytes

    @property
    def mutable(self) -> bool:
        """The export default mutability for future reference exports."""
        return self._remote_storage()._owner.mutable

    @mutable.setter
    def mutable(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError("mutable must be a boolean")
        storage = self._remote_storage()
        with storage._owner.lock:
            storage._check_open()
            storage._owner.mutable = value

    @property
    def is_cache_mutable(self) -> bool:
        """Whether the current local cache is writable, not the remote table."""
        return self._remote_storage()._owner.is_mutable

    @property
    def cache_bytes(self):
        return self._remote_storage()._owner.cache_coordinator.cache_bytes

    @property
    def metadata_bytes(self):
        storage = self._remote_storage()
        storage._owner.save_manifest()
        return storage._owner.metadata_bytes

    def save(
        self,
        destination: str | os.PathLike | None = None,
        *,
        urlpath: str | os.PathLike | None = None,
        include_cache: bool = True,
        mutable: bool | None = None,
        overwrite: bool = False,
    ) -> str:
        """Export this table as a portable remote-reference archive."""
        if destination is None:
            if urlpath is None:
                raise TypeError("save() missing required destination")
            destination = urlpath
        elif urlpath is not None:
            raise TypeError("destination and urlpath cannot both be specified")

        storage = self._remote_storage()
        with storage._owner.lock:
            storage._check_open()
            return storage._owner.save_selection(
                storage._root_key,
                destination,
                include_cache=include_cache,
                mutable=mutable,
                overwrite=overwrite,
            )
