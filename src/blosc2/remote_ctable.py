#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################
"""Read-only remote CTable access through fsspec-backed B2Z sources."""

from __future__ import annotations

from blosc2.ctable import CTable
from blosc2.ctable_storage import RemoteTableStorage
from blosc2.remote_array import CACHE_POLICY_DEFAULT, RemoteMetadataMapping


class RemoteCTable(CTable):
    """A read-only CTable whose fixed-width and UTF-8 columns are fetched on demand."""

    def __new__(
        cls,
        urlpath=None,
        *,
        dataset=None,
        storage_options=None,
        cache_policy=CACHE_POLICY_DEFAULT,
        max_cache_bytes=CACHE_POLICY_DEFAULT,
        cache_dir=None,
        _filesystem=None,
    ):
        if urlpath is None:
            raise TypeError("RemoteCTable requires a remote B2Z URL")

        from blosc2.remote_store import RemoteStore

        store = RemoteStore(
            urlpath,
            dataset=dataset,
            storage_options=storage_options,
            cache_policy=cache_policy,
            max_cache_bytes=max_cache_bytes,
            cache_dir=cache_dir,
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
            return cls._from_owner(store._owner, full)
        finally:
            store.close()

    def __init__(self, *args, **kwargs):
        # Construction is completed by CTable._open_from_storage() in __new__.
        pass

    @classmethod
    def _from_owner(cls, owner, full_path):
        storage = RemoteTableStorage(owner, full_path)
        try:
            return cls._open_from_storage(storage)
        except BaseException:
            storage.close()
            raise

    def _remote_storage(self) -> RemoteTableStorage:
        storage = getattr(self, "_storage", None)
        if not isinstance(storage, RemoteTableStorage):
            raise RuntimeError("RemoteCTable handle is closed")
        storage._check_open()
        return storage

    def close(self) -> None:
        storage = getattr(self, "_storage", None)
        if isinstance(storage, RemoteTableStorage):
            storage.close()

    @property
    def vlmeta(self):
        return RemoteMetadataMapping(self._remote_storage().load_user_attrs())

    @property
    def source(self):
        storage = self._remote_storage()
        return {
            "kind": "b2z",
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
    def cache_bytes(self):
        return self._remote_storage()._owner.cache_coordinator.cache_bytes

    @property
    def metadata_bytes(self):
        storage = self._remote_storage()
        storage._owner.save_manifest()
        return storage._owner.metadata_bytes
