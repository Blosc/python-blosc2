"""Read-only remote hierarchy discovery and public store handles."""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import tempfile
import threading
import uuid
import weakref
import zipfile
from dataclasses import dataclass
from pathlib import PurePosixPath
from urllib.parse import urlsplit, urlunsplit

import blosc2
from blosc2.core import parse_container_url, storage_options_fingerprint
from blosc2.proxy import CacheCoordinator
from blosc2.proxy_source import Traffic
from blosc2.remote_array import (
    CACHE_POLICY_DEFAULT,
    RemoteMetadataMapping,
    normalize_cache_limit,
    validate_persistable_url,
)

RESERVED_NAMES = {"embed.b2e", "__vlmeta__"}


def get_zip_offsets(zip_path: str) -> dict[str, dict[str, int]]:
    """Get offset, length, and storage status of files in a .b2z archive."""
    offsets = {}
    with open(zip_path, "rb") as f, zipfile.ZipFile(f) as zf:
        for info in zf.infolist():
            name = info.filename.rstrip("/")
            if not name or ":" in name:
                raise ValueError("Invalid RemoteStore archive path")
            if name != "embed.b2e":
                RemoteDiscovery._validate(name)
            if info.filename in offsets or info.flag_bits & 1:
                raise ValueError("Invalid RemoteStore archive member")
            f.seek(info.header_offset)
            local_header = f.read(30)
            filename_len = int.from_bytes(local_header[26:28], "little")
            extra_len = int.from_bytes(local_header[28:30], "little")
            data_offset = info.header_offset + 30 + filename_len + extra_len
            offsets[info.filename] = {
                "offset": data_offset,
                "length": info.compress_size,
                "stored": info.compress_type == zipfile.ZIP_STORED,
            }
    return offsets


@dataclass(frozen=True)
class RemoteNode:
    """Discovery metadata; unknown attributes are None and require opening the array."""

    path: str
    kind: str
    attrs: RemoteMetadataMapping | None
    diagnostic: str | None = None


class RemoteDiscovery:
    """Shared metadata and source resources, independent of browser presentation."""

    def __init__(
        self,
        urlpath,
        storage_options=None,
        *,
        dataset=None,
        manifest=None,
        persist_metadata=False,
        _filesystem=None,
        _source_validator=None,
        _manifest_validator=None,
        _max_nodes=None,
    ):
        self.urlpath, dataset, self.format = parse_container_url(urlpath, dataset)
        self.root = (dataset or "").strip("/")
        self._validate(self.root)
        self.storage_options = storage_options or {}
        self.traffic = Traffic()
        self.nodes = {}
        self.attrs = {}
        self.listed = {}
        self.notice = None
        self.archive = None
        self.zstore = None
        self.sources = {}
        self.caches = {}
        self.disk = None
        self.generation = manifest["generation"] if manifest else uuid.uuid4().hex
        self.metadata = manifest["metadata"] if manifest else {}
        self.persist_metadata = persist_metadata
        self._external_filesystem = _filesystem
        self.source_validator = _source_validator
        self.manifest_validator = _manifest_validator
        self.max_nodes = _max_nodes
        self.metadata_bytes = 0
        self.restoring = False
        self.filesystem = None
        self.mutable = False
        self.is_mutable = True
        self.artifact_path = None
        self.artifact_offsets = None
        self._cleanup_dir = None
        self._users = 0
        self._closed = False
        # ponytail: serialize store operations; finer locks if multi-leaf throughput matters.
        self.lock = threading.RLock()
        try:
            import fsspec

            options = {**self.storage_options, "skip_instance_cache": True}
            self.filesystem = _filesystem
            if self.filesystem is None:
                self.filesystem, _ = fsspec.core.url_to_fs(self.urlpath, **options)
            if manifest:
                self._restore_manifest(manifest)
            elif self.format == "b2z":
                self._open_b2z()
            elif self.format == "hdf5":
                self._open_hdf5()
            else:
                self._open_zarr()
            self._check_node_limit()
            if self.root not in self.nodes:
                raise KeyError("Requested node does not exist in the container")
            self.is_tree = self.nodes[self.root][0] == "group"
        except BaseException:
            self.close()
            raise

    def _restore_manifest(self, manifest):
        for path, entry in manifest["nodes"].items():
            self._validate(path)
            if (
                not isinstance(entry, (list, tuple))
                or len(entry) != 2
                or entry[0] not in {"group", "ndarray", "unsupported"}
            ):
                raise ValueError("Invalid RemoteStore node")
            self.nodes[path] = tuple(entry)
        for mapping in (manifest["attrs"], manifest["listed"]):
            for path in mapping:
                self._validate(path)
                if path not in self.nodes:
                    raise ValueError("Invalid RemoteStore metadata path")
        for parent, children in manifest["listed"].items():
            if not isinstance(children, list) or any(
                not isinstance(child, str) or child not in self.nodes or child.rpartition("/")[0] != parent
                for child in children
            ):
                raise ValueError("Invalid RemoteStore child list")
        self.attrs = manifest["attrs"]
        self.listed = manifest["listed"]
        self.notice = manifest.get("notice")
        if self.format == "b2z" and self.archive is None:
            from blosc2.b2z_source import B2ZArchive

            self.archive = B2ZArchive(
                self.urlpath,
                storage_options=self.storage_options,
                _traffic=self.traffic,
                _metadata=self.metadata,
                _filesystem=self.filesystem,
            )
        elif self.format == "hdf5":
            self.refs = self.metadata
            self._validate_refs()
        elif self.format == "zarr" and self.zstore is None:
            self._open_zarr()
        self._check_node_limit()

    def _validate_refs(self):
        refs = self.refs.get("refs", self.refs)
        if not isinstance(refs, dict) or self.refs.get("templates"):
            raise ValueError("Invalid HDF5 manifest references")
        for value in refs.values():
            if isinstance(value, list):
                if (
                    len(value) != 3
                    or value[0] != self.urlpath
                    or any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in value[1:])
                ):
                    raise ValueError("HDF5 manifest contains an unsafe reference")
                validate_persistable_url(value[0])

    def save_manifest(self):
        if self.disk is None or self.restoring or not self.is_mutable:
            return
        # ponytail: publish whole discovery snapshots; add dirty tracking if large maps make this costly.
        if self.format == "hdf5":
            self._validate_refs()
            self.metadata = self.refs
        elif self.format == "b2z":
            self.metadata = self.archive.metadata
        nodes = {
            path: (kind, value if kind == "unsupported" else None)
            for path, (kind, value) in self.nodes.items()
        }
        manifest = {
            "version": 1,
            "source": self.disk.source,
            "generation": self.generation,
            "nodes": nodes,
            "attrs": self.attrs,
            "listed": self.listed,
            "notice": self.notice,
            "metadata": self.metadata,
            "caches": sorted(self.caches),
            "cache_policy": getattr(self, "cache_policy", blosc2.CachePolicy.DISK).value,
            "max_cache_bytes": getattr(self, "max_cache_bytes", None),
            "mutable": getattr(self, "mutable", False),
        }
        if self.manifest_validator is not None:
            self.manifest_validator(manifest)
        self.metadata_bytes = self.disk.publish(manifest)

    @staticmethod
    def _validate(path):
        if not isinstance(path, str):
            raise ValueError("Remote container paths must be strings")
        parts = path.split("/")
        if path and (
            any(p in {"", ".", ".."} | RESERVED_NAMES for p in parts)
            or any(c in path for c in ":\\\0\n\r\t")
        ):
            raise ValueError("Unsafe path in remote container")

    def _add(self, path, kind, value=None):
        self._validate(path)
        if path in self.nodes and self.nodes[path][0] != "group":
            raise ValueError("Ambiguous duplicate logical key in remote container")
        if path in self.nodes and kind != "group":
            raise ValueError("Leaf/group collision in remote container")
        self.nodes[path] = (kind, value)
        parent = path
        while parent:
            parent = parent.rpartition("/")[0]
            if parent in self.nodes and self.nodes[parent][0] != "group":
                raise ValueError("Leaf/group collision in remote container")
            self.nodes.setdefault(parent, ("group", None))
        self._check_node_limit()

    def _check_node_limit(self):
        if self.max_nodes is not None and len(self.nodes) > self.max_nodes:
            raise ValueError("RemoteStore discovery exceeds the node limit")

    def _find_b2z_ctable_roots(self, members, embedded, registry):
        from blosc2.b2z_source import B2ZEmbeddedMetadata, member_vlmeta

        roots = {key.strip("/") for key in registry}
        embedded_reader = None
        # Match TreeStore's legacy CTable manifest check using metadata only.
        for name, info in members.items():
            if (name == "_meta.b2f" or name.endswith("/_meta.b2f")) and member_vlmeta(
                self.archive, info
            ).get("kind") in {"ctable", b"ctable"}:
                roots.add(name.rpartition("/")[0])
        for key, entry in embedded.items():
            if key.endswith("/_meta"):
                try:
                    if embedded_reader is None:
                        embedded_reader = B2ZEmbeddedMetadata(self.archive, members["embed.b2e"])
                    if embedded_reader.attrs(entry).get("kind") in {"ctable", b"ctable"}:
                        roots.add(key.rpartition("/")[0].strip("/"))
                except NotImplementedError as exc:
                    roots.add(key.rpartition("/")[0].strip("/"))
                    self.notice = f"Partial B2Z metadata: object boundary cannot be verified: {exc}."
        return roots, embedded_reader

    def _process_b2z_members(self, members, roots):
        from blosc2.b2z_source import member_vlmeta
        from blosc2.dict_store import DictStore

        for name, info in members.items():
            if info.is_dir():
                key = name.rstrip("/")
                if not any(key == root or key.startswith(root + "/") for root in roots):
                    self._add(key, "group")
                continue
            if name == "embed.b2e":
                continue
            if PurePosixPath(name).suffix not in {".b2nd", ".b2f", ".b2b"}:
                continue
            key = DictStore._logical_key_from_relpath(name).lstrip("/")
            if any(key == root or key.startswith(root + "/") for root in roots):
                continue
            if key == "__vlmeta__" or key.endswith("/__vlmeta__"):
                group = key.rpartition("/")[0]
                self._add(group, "group")
                self.attrs[group] = member_vlmeta(self.archive, info)
                continue
            supported = name.endswith(".b2nd") and not info.flag_bits & 1 and info.compress_type == 0
            self._add(
                key,
                "ndarray" if supported else "unsupported",
                info
                if supported
                else "B2Z array access requires unencrypted ZIP_STORED external NDArray members",
            )

    def _process_b2z_embedded(self, embedded, roots, embedded_reader, members):
        from blosc2.b2z_source import B2ZEmbeddedMetadata

        for key in embedded:
            path = key.strip("/")
            if any(path == root or path.startswith(root + "/") for root in roots):
                continue
            if path == "__vlmeta__" or path.endswith("/__vlmeta__"):
                group = path.rpartition("/")[0]
                self._add(group, "group")
                try:
                    if embedded_reader is None:
                        embedded_reader = B2ZEmbeddedMetadata(self.archive, members["embed.b2e"])
                    self.attrs[group] = embedded_reader.attrs(embedded[key])
                except NotImplementedError as exc:
                    self.attrs[group] = None
                    self.notice = f"Partial B2Z metadata: {exc}."
            else:
                self._add(path, "unsupported", "Embedded B2Z array access is unavailable")

    def _open_b2z(self):
        from blosc2.b2z_source import B2ZArchive, member_vlmeta

        self.archive = B2ZArchive(
            self.urlpath,
            storage_options=self.storage_options,
            _traffic=self.traffic,
            _metadata=self.metadata if self.persist_metadata else None,
            _filesystem=self.filesystem,
        )
        self.archive.capture_metadata = True
        members = {}
        for info in self.archive.members:
            if info.filename == "embed.b2e":
                if info.filename in members:
                    raise ValueError("Duplicate member in B2Z archive")
                members[info.filename] = info
                continue
            self._validate(info.filename.rstrip("/"))
            if info.filename in members:
                raise ValueError("Duplicate member in B2Z archive")
            members[info.filename] = info
        embedded = {}
        registry = {}
        if "embed.b2e" in members:
            meta = member_vlmeta(self.archive, members["embed.b2e"])
            embedded = meta.get("estore_metadata", {}).get("embed_map", {})
            registry = meta.get("_object_registry", {})

        roots, embedded_reader = self._find_b2z_ctable_roots(members, embedded, registry)
        if "" in roots:
            self.nodes[""] = ("unsupported", "CTable access is unavailable for remote B2Z hierarchies")
            return
        self._add("", "group")
        for root in sorted(roots, key=len):
            if not any(root.startswith(other + "/") for other in roots if other != root):
                self._add(root, "unsupported", "CTable access is unavailable for remote B2Z hierarchies")
        self._process_b2z_members(members, roots)
        self._process_b2z_embedded(embedded, roots, embedded_reader, members)
        self.archive._opening_ranges.clear()
        self.archive.capture_metadata = False

    def _open_hdf5(self):
        from blosc2.hdf5_source import scan_hdf5_refs

        unsupported = {}
        self.refs = scan_hdf5_refs(
            self.urlpath,
            self.storage_options,
            unsupported=unsupported,
            traffic=self.traffic,
            _filesystem=self.filesystem,
        )
        refs = self.refs.get("refs", self.refs)
        for key, value in refs.items():
            name = key.rsplit("/", 1)[-1]
            path = key.rpartition("/")[0]
            if name in {".zgroup", ".zarray"}:
                self._add(path, "group" if name == ".zgroup" else "ndarray", json.loads(value))
            elif name == ".zattrs":
                self.attrs[path] = {k: v for k, v in json.loads(value).items() if k != "_ARRAY_DIMENSIONS"}
        for path, message in unsupported.items():
            self._validate(path)
            self.nodes[path] = ("unsupported", message)
        self.notice = (
            "HDF5 view includes objects represented by Kerchunk; external and group links are omitted."
        )

    def _open_zarr(self):
        import zarr

        from blosc2.zarr_source import counting_store, owned_fsspec_store

        self.zstore = counting_store(
            zarr,
            owned_fsspec_store(zarr, self.filesystem, self.filesystem._strip_protocol(self.urlpath)),
            self.traffic,
            metadata=self.metadata if self.persist_metadata else None,
        )
        # Open the requested node directly; array-only readers need no parent LIST.
        try:
            node = zarr.open(store=self.zstore, path=self.root, mode="r")
        except (ValueError, TypeError, NotImplementedError):
            # Invalid codec metadata in a consolidated sibling must not block
            # the group root. Ordinary discovery can isolate that child.
            node = zarr.open(store=self.zstore, path=self.root, mode="r", use_consolidated=False)
        self.nodes[self.root] = ("group" if isinstance(node, zarr.Group) else "ndarray", node)
        self.attrs[self.root] = dict(node.attrs)

    def _path(self, path):
        if not isinstance(path, str):
            raise TypeError("RemoteStore paths must be strings")
        relative = path.strip("/")
        self._validate(relative)
        return "/".join(p for p in (self.root, relative) if p)

    def resolve(self, path):
        """Resolve a relative path without listing a Zarr parent."""
        full = self._path(path)
        if self.format == "zarr" and (
            full not in self.nodes or (self.nodes[full][0] != "unsupported" and self.nodes[full][1] is None)
        ):
            import zarr

            try:
                node = zarr.open(store=self.zstore, path=full, mode="r", use_consolidated=False)
            except FileNotFoundError as exc:
                raise KeyError(path) from exc
            except (ValueError, TypeError, NotImplementedError) as exc:
                self._add(full, "unsupported", f"{type(exc).__name__}: {exc}")
            else:
                self.nodes.pop(full, None)
                self._add(full, "group" if isinstance(node, zarr.Group) else "ndarray", node)
                self.attrs[full] = dict(node.attrs)
        if full not in self.nodes:
            raise KeyError(path)
        return full

    def kind(self, path):
        return self.nodes[self._path(path)][0]

    def list_children(self, path):
        full = self._path(path)
        if self.nodes[full][0] != "group":
            return []
        if full not in self.listed:
            if self.format == "zarr":
                import zarr

                group = self.nodes[full][1]
                try:
                    children = group.members(max_depth=0)
                except (ValueError, TypeError, NotImplementedError):
                    # One unknown codec must not hide supported siblings. Use the
                    # store's immediate-directory API only on this failure path.
                    from zarr.core.sync import sync

                    async def names():
                        return [name async for name in self.zstore.list_dir(full)]

                    try:
                        names = sync(names())
                    except Exception as exc:
                        raise OSError(
                            "Cannot list Zarr group; check LIST permission and backend support"
                        ) from exc
                    children = []
                    for name in names:
                        if name in {".zattrs", ".zarray", ".zgroup", ".zmetadata", "zarr.json"}:
                            continue
                        try:
                            children.append((name, group[name]))
                        except KeyError:
                            continue  # unrelated storage object
                        except (ValueError, TypeError, NotImplementedError) as exc:
                            key = "/".join(p for p in (full, name) if p)
                            self._validate(key)
                            self.nodes[key] = ("unsupported", f"{type(exc).__name__}: {exc}")
                except Exception as exc:
                    raise OSError(
                        "Cannot list Zarr group; check LIST permission and backend support"
                    ) from exc
                for name, node in children:
                    key = "/".join(p for p in (full, name) if p)
                    self._validate(key)
                    self.nodes[key] = ("group" if isinstance(node, zarr.Group) else "ndarray", node)
                    self.attrs[key] = dict(node.attrs)
            self.listed[full] = sorted(
                key for key in self.nodes if key != full and key.rpartition("/")[0] == full
            )
        self._check_node_limit()
        return self.listed[full].copy()

    def open_source(self, path):
        full = self.resolve(path)
        if full in self.sources:
            return self.sources[full]
        kind, value = self.nodes[full]
        if kind != "ndarray":
            raise NotImplementedError(
                value if isinstance(value, str) else "Array access is unavailable for this node"
            )
        if self.format == "b2z":
            from blosc2.b2z_source import B2ZNDSource

            source = B2ZNDSource(self.urlpath, full, _archive=self.archive)
        elif self.format == "hdf5":
            from blosc2.hdf5_source import HDF5NDSource

            source = HDF5NDSource(
                self.urlpath,
                full,
                refs=self.refs,
                storage_options=self.storage_options,
                _traffic=self.traffic,
                _filesystem=self.filesystem,
            )
        else:
            from blosc2.zarr_source import ZarrNDSource

            url = urlsplit(self.urlpath)
            leaf_url = urlunsplit(url._replace(path=url.path.rstrip("/") + "/" + full))
            source = ZarrNDSource(self.zstore, _path=full, _urlpath=leaf_url, _traffic=self.traffic)
        if self.source_validator is not None:
            self.source_validator(source)
        self.sources[full] = source
        return source

    def acquire(self):
        with self.lock:
            if self._closed:
                raise RuntimeError("RemoteStore resources are closed")
            self._users += 1

    def release(self):
        with self.lock:
            self._users -= 1
            if not self._users:
                self.close()

    def restore_caches(self, manifest):
        if manifest:
            self.restoring = True
            for path in manifest["caches"]:
                self._validate(path)
                if path not in self.nodes or self.nodes[path][0] != "ndarray":
                    raise ValueError("Invalid cached RemoteStore leaf")
                relative = path[len(self.root) + 1 :] if self.root else path
                self.resolve(relative)
                self.get_cache(self.open_source(relative))
            self.cache_coordinator.enforce()
            self.restoring = False

    def get_cache(self, source, *, seed=None):
        key = next(path for path, value in self.sources.items() if value is source)
        if key not in self.caches:
            if getattr(self, "shared", False):
                descriptor = {
                    "kind": self.format,
                    "version": 1,
                    "urlpath": source.urlpath,
                    "assume_immutable": True,
                }
                if self.format in {"b2z", "hdf5"}:
                    descriptor["dataset"] = key
                runtime = blosc2.RemoteArray.with_sparse_cache(
                    source,
                    self.disk.payload_path(self.generation, key),
                    source_descriptor=descriptor,
                    max_cache_bytes=None,
                    carrier=seed,
                )
                proxy = runtime._proxy
                if proxy is None:
                    raise ValueError("Shared store caching requires a stable source identity")
                proxy._cache_coordinator = self.cache_coordinator
                proxy._cache_key = key
                self.cache_coordinator.register(proxy)
                self.caches[key] = proxy
                self.save_manifest()
                return proxy
            if not self.is_mutable:
                carrier = None
                if self.artifact_offsets is not None:
                    leaf_name = f"{key}.b2nd"
                    if leaf_name in self.artifact_offsets:
                        offset = self.artifact_offsets[leaf_name]["offset"]
                        carrier = blosc2.blosc2_ext.open(self.artifact_path, mode="r", offset=offset)
                elif self.artifact_path is not None:
                    leaf_path = os.path.join(self.artifact_path, f"{key}.b2nd")
                    if os.path.exists(leaf_path):
                        carrier = blosc2.blosc2_ext.open(leaf_path, mode="r", offset=0)
                if carrier is None:
                    carrier = blosc2.empty(
                        source.shape,
                        source.dtype,
                        chunks=source.chunks,
                        blocks=source.blocks,
                        cparams=source.cparams,
                    )
                    carrier.schunk.mode = "r"
                self.caches[key] = blosc2.Proxy(
                    source,
                    _cache=carrier,
                    mode="r",
                    _refresh_source=False,
                    _cache_coordinator=self.cache_coordinator,
                    _cache_key=key,
                    _persistent_dirty=False,
                )
                return self.caches[key]

            path = None if self.disk is None else str(self.disk.payload_path(self.generation, key))
            identity = {"generation": self.generation, "dataset": key}
            exists = path is not None and os.path.exists(path)
            self.caches[key] = blosc2.Proxy(
                source,
                urlpath=path,
                mode="a",
                _refresh_source=False,
                _cache_coordinator=self.cache_coordinator,
                _cache_key=key,
                _persistent_dirty=self.disk is not None,
                meta={"remote-store": identity} if path is not None and not exists else None,
            )
            if exists and self.caches[key].schunk.meta.get("remote-store") != identity:
                self.caches.pop(key)
                raise ValueError("RemoteStore payload generation or dataset mismatch")
            self.save_manifest()
        return self.caches[key]

    def close(self):
        if self._closed:
            return
        try:
            self.save_manifest()
        finally:
            try:
                self._close_resources()
            finally:
                if self.disk is not None:
                    self.disk.close()
                if self._cleanup_dir is not None:
                    self._cleanup_dir.cleanup()
                    self._cleanup_dir = None

    def _close_resources(self):
        self._closed = True
        if self.archive is not None:
            self.archive.close()
        if self.zstore is not None:
            self.zstore.close()
        for source in self.sources.values():
            if isinstance(source, blosc2.HDF5NDSource):
                source.array.store.close()
        self.sources.clear()
        self.caches.clear()
        self.nodes.clear()
        self.attrs.clear()
        self.listed.clear()
        self.refs = None
        if self.filesystem is not None and self._external_filesystem is None:
            # fsspec's HTTP and S3 clients expose their own synchronous close hook.
            close = getattr(self.filesystem, "close_session", None)
            session = getattr(self.filesystem, "_s3creator", None) or getattr(
                self.filesystem, "_session", None
            )
            if close is not None and session is not None:
                close(self.filesystem.loop, session)
            self.filesystem = None


class RemoteStore:
    """Read-only remote B2Z, Zarr or HDF5 hierarchy.

    Discovery and returned array handles share source resources and traffic.
    MEMORY shares one bounded cache across all leaves; NONE retains no payload.
    DISK retains payload and discovery under an exclusively owned cache directory.
    Sources must be immutable until an explicit root refresh.

    ``keys()`` lists immediate children; ``get_info()`` inspects metadata without
    creating an array cache. Paths are relative to this group. Closing a handle
    leaves its previously returned arrays and group handles usable.
    """

    @classmethod
    def _try_open_artifact(
        cls, urlpath, dataset, storage_options, cache_policy, max_cache_bytes, cache_dir, allow_array_root
    ):
        if not os.path.exists(urlpath):
            return None
        from blosc2.schunk import _meta_from_store

        meta = _meta_from_store(urlpath, 0)
        if meta is None or "b2remote_store" not in meta:
            return None
        kwargs = {
            "dataset": dataset,
            "storage_options": storage_options,
            "cache_policy": None if cache_policy is CACHE_POLICY_DEFAULT else cache_policy,
            "max_cache_bytes": None if max_cache_bytes is CACHE_POLICY_DEFAULT else max_cache_bytes,
            "cache_dir": cache_dir,
            "_allow_array_root": allow_array_root,
        }
        return cls._open_artifact(urlpath, **kwargs)

    @staticmethod
    def _validate_cache_config(cache_policy, max_cache_bytes, cache_dir):
        if cache_policy is CACHE_POLICY_DEFAULT:
            cache_policy = blosc2.CachePolicy.DISK if cache_dir is not None else blosc2.CachePolicy.MEMORY
        if not isinstance(cache_policy, blosc2.CachePolicy):
            raise TypeError("cache_policy must be a blosc2.CachePolicy instance")
        if (cache_policy is blosc2.CachePolicy.DISK) != (cache_dir is not None):
            raise ValueError("CachePolicy.DISK requires cache_dir; other policies reject it")
        limit = normalize_cache_limit(cache_policy, max_cache_bytes)
        return cache_policy, limit

    def __init__(
        self,
        urlpath,
        *,
        dataset=None,
        storage_options=None,
        cache_policy=CACHE_POLICY_DEFAULT,
        max_cache_bytes=CACHE_POLICY_DEFAULT,
        cache_dir=None,
        _allow_array_root=False,
        _filesystem=None,
        _manifest=None,
        _source_validator=None,
        _manifest_validator=None,
        _max_nodes=None,
    ):
        if isinstance(urlpath, os.PathLike):
            urlpath = os.fspath(urlpath)
        if not isinstance(urlpath, str):
            raise TypeError("RemoteStore requires a remote URL string")
        artifact = self._try_open_artifact(
            urlpath, dataset, storage_options, cache_policy, max_cache_bytes, cache_dir, _allow_array_root
        )
        if artifact is not None:
            self._attach(artifact._owner, artifact._path)
            return
        if dataset is not None and not isinstance(dataset, str):
            raise TypeError("dataset must be a string")
        cache_policy, limit = self._validate_cache_config(cache_policy, max_cache_bytes, cache_dir)
        base_url, _, _ = parse_container_url(urlpath, dataset)
        validate_persistable_url(base_url)
        disk = None
        manifest = _manifest
        if cache_policy is blosc2.CachePolicy.DISK:
            from blosc2.remote_store_cache import StoreDiskCache

            base_url, root, kind = parse_container_url(urlpath, dataset)
            source = {"urlpath": base_url, "dataset": (root or "").strip("/"), "kind": kind}
            fingerprint = storage_options_fingerprint(storage_options)
            if fingerprint:
                # The same URL through another endpoint or account must not
                # reuse this manifest and its leaf payloads.
                source["storage_options"] = fingerprint
            disk = StoreDiskCache(cache_dir, source)
        try:
            manifest = disk.load() if disk is not None else manifest
            owner = RemoteDiscovery(
                urlpath,
                storage_options,
                dataset=dataset,
                manifest=manifest,
                persist_metadata=disk is not None,
                _filesystem=_filesystem,
                _source_validator=_source_validator,
                _manifest_validator=_manifest_validator,
                _max_nodes=_max_nodes,
            )
        except BaseException:
            if disk is not None:
                disk.close()
            raise
        owner.disk = disk
        if not owner.is_tree and not _allow_array_root:
            kind, diagnostic = owner.nodes[owner.root]
            owner.close()
            if kind == "unsupported":
                raise NotImplementedError(str(diagnostic))
            raise ValueError("RemoteStore requires a group; use RemoteArray for an array")
        owner.cache_policy = cache_policy
        owner.max_cache_bytes = limit
        owner.cache_coordinator = CacheCoordinator(limit)
        try:
            owner.restore_caches(manifest)
            owner.save_manifest()
            if disk is not None:
                disk.discard_old_generations(owner.generation)
        except BaseException:
            owner.close()
            raise
        self._attach(owner, "")

    def _attach(self, owner, path):
        owner.acquire()
        self._owner = owner
        self._path = path
        self._generation = owner.generation
        self._finalizer = weakref.finalize(self, owner.release)

    @classmethod
    def with_sparse_cache(
        cls,
        urlpath,
        runtime_cache_path,
        *,
        dataset=None,
        manifest=None,
        max_cache_bytes=None,
        carrier=None,
        _filesystem=None,
        _source_validator=None,
        _manifest_validator=None,
        _max_nodes=None,
    ):
        """Attach an immutable remote hierarchy to a cache shared across processes.

        All users of this private cache must use this constructor. Operations
        serialize per store, reload discovery, and enforce one aggregate payload
        allowance. The caller authorizes the supplied filesystem and manifest;
        no credentials or filesystem objects are persisted. Portable artifacts
        are exported with ``save`` rather than opened as mutable runtime storage.
        """
        from blosc2.remote_store_cache import SharedStoreCache, SharedStoreOperation

        limit = normalize_cache_limit(blosc2.CachePolicy.DISK, max_cache_bytes)
        base, root, kind = parse_container_url(urlpath, dataset)
        validate_persistable_url(base)
        source = {"urlpath": base, "dataset": (root or "").strip("/"), "kind": kind}
        disk = SharedStoreCache(runtime_cache_path, source)
        with disk.guard():
            current = disk.load()
            seed_manifest = None
            if current is None and carrier is not None:
                seed_manifest, seed_offsets = cls._load_artifact_manifest(os.fspath(carrier))
                if seed_manifest["source"] != source:
                    raise ValueError("RemoteStore seed source mismatch")
            if current is None and manifest is not None:
                cls._validate_artifact_manifest(manifest)
                if manifest["source"] != source:
                    raise ValueError("RemoteStore manifest source mismatch")
                current = dict(manifest, caches=[], generation=uuid.uuid4().hex)
            owner = RemoteDiscovery(
                base,
                dataset=root,
                manifest=current,
                persist_metadata=True,
                _filesystem=_filesystem,
                _source_validator=_source_validator,
                _manifest_validator=_manifest_validator,
                _max_nodes=_max_nodes,
            )
            owner.disk = disk
            owner.shared = True
            owner.cache_policy = blosc2.CachePolicy.DISK
            owner.max_cache_bytes = limit
            owner.cache_coordinator = CacheCoordinator(limit)
            try:
                owner.restore_caches(current)
                if seed_manifest is not None:
                    for key in seed_manifest["caches"]:
                        relative = key[len(owner.root) + 1 :] if owner.root else key
                        src = owner.open_source(relative)
                        name = key + ".b2nd"
                        seed = blosc2.blosc2_ext.open(
                            os.fspath(carrier),
                            "r",
                            seed_offsets[name]["offset"],
                        )
                        seed = blosc2.ndarray_from_cframe(seed.to_cframe(), copy=True)
                        descriptor = {
                            "kind": owner.format,
                            "version": 1,
                            "urlpath": src.urlpath,
                            "assume_immutable": True,
                        }
                        if owner.format in {"b2z", "hdf5"}:
                            descriptor["dataset"] = key
                        seed.schunk.vlmeta["b2o"] = {"kind": "remote_array", "source": descriptor}
                        owner.get_cache(src, seed=seed)
                    owner.cache_coordinator.enforce()
                owner.save_manifest()
                obj = object.__new__(cls)
                obj._attach(owner, "")
                owner.lock = SharedStoreOperation(owner)
                return obj
            except BaseException:
                owner.close()
                raise

    @staticmethod
    def trim_sparse_cache(runtime_cache_path, source, target_bytes, *, max_chunks=64):
        """Trim shared leaf payload without opening or contacting the source."""
        from blosc2.remote_store_cache import SharedStoreCache

        for value in (target_bytes, max_chunks):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("Cache trim limits must be non-negative integers")
        if not os.path.isdir(runtime_cache_path):
            return (), 0
        disk = SharedStoreCache(runtime_cache_path, source)
        with disk.guard():
            manifest = disk.load()
            if manifest is None:
                return (), 0
            leaves = []
            for key in manifest["caches"]:
                path = disk.payload_path(manifest["generation"], key)
                evicted, size = blosc2.RemoteArray.trim_sparse_cache(path, 1 << 63, max_chunks=0)
                leaves.append((key, path, size))
            total = sum(size for _, _, size in leaves)
            removed = []
            # ponytail: leaf-order eviction; persist global recency if workloads need exact LRU.
            for key, path, size in leaves:
                if total <= target_bytes or len(removed) >= max_chunks:
                    break
                evicted, remaining = blosc2.RemoteArray.trim_sparse_cache(
                    path,
                    max(0, size - (total - target_bytes)),
                    max_chunks=max_chunks - len(removed),
                )
                removed.extend((key, chunk) for chunk in evicted)
                total += remaining - size
            return tuple(removed), total

    def read_cached(self, path, item=(), *, nchunk=None):
        """Return ``(hit, result)`` atomically without fetching missing payload."""
        with self._owner.lock:
            _, full = self._resolve(path)
            if full not in self._owner.caches:
                return False, None
            with self[path] as array:
                return array.read_cached(item, nchunk=nchunk)

    def _resolve(self, path):
        if not self._finalizer.alive:
            raise RuntimeError("RemoteStore handle is closed")
        if self._generation != self._owner.generation:
            raise RuntimeError("RemoteStore handle is stale; look it up again after refresh")
        if not isinstance(path, str):
            raise TypeError("RemoteStore paths must be strings")
        relative = path.strip("/")
        self._owner._validate(relative)
        joined = "/".join(part for part in (self._path, relative) if part)
        return joined, self._owner.resolve(joined)

    def keys(self):
        """Return sorted immediate child names, using only discovery metadata."""
        with self._owner.lock:
            path, _ = self._resolve("")
            result = [key.rsplit("/", 1)[-1] for key in self._owner.list_children(path)]
            self._owner.save_manifest()
            return result

    def __iter__(self):
        return iter(self.keys())

    def __getitem__(self, path):
        with self._owner.lock:
            relative, full = self._resolve(path)
            self._owner.save_manifest()
            kind, value = self._owner.nodes[full]
            if kind == "group":
                group = object.__new__(type(self))
                group._attach(self._owner, relative)
                return group
            if kind == "unsupported":
                raise NotImplementedError(f"{path!r}: {value}")
            source = self._owner.open_source(relative)
            descriptor = {
                "kind": self._owner.format,
                "version": 1,
                "urlpath": source.urlpath,
                "assume_immutable": True,
            }
            if self._owner.format in {"b2z", "hdf5"}:
                descriptor["dataset"] = full
            return blosc2.RemoteArray(
                source,
                _source_descriptor=descriptor,
                _store_owner=self._owner,
                cache_policy=self.cache_policy,
                max_cache_bytes=(
                    self.max_cache_bytes
                    if self.cache_policy is not blosc2.CachePolicy.NONE
                    else CACHE_POLICY_DEFAULT
                ),
            )

    def get_info(self, path=""):
        """Return node kind, known attributes and unsupported-node diagnostics."""
        with self._owner.lock:
            _, full = self._resolve(path)
            kind, value = self._owner.nodes[full]
            attrs = self._owner.attrs.get(full, {} if kind == "group" else None)
            diagnostic = str(value) if kind == "unsupported" else self._owner.notice
            return RemoteNode(
                path.strip("/"),
                kind,
                None if attrs is None else RemoteMetadataMapping(attrs),
                diagnostic,
            )

    def kind(self, path=""):
        """Return 'group', 'ndarray' or 'unsupported'."""
        return self.get_info(path).kind

    @property
    def attrs(self):
        """Read-only group attributes, or None when discovery cannot decode them."""
        return self.get_info().attrs

    @property
    def source(self):
        """Credential-free source descriptor, including this group's full path."""
        with self._owner.lock:
            _, full = self._resolve("")
            return {
                "kind": self._owner.format,
                "version": 1,
                "urlpath": self._owner.urlpath,
                "dataset": full,
                "assume_immutable": True,
            }

    @property
    def traffic(self):
        """Shared source counter, including discovery; aliases expose the same counter.

        Source reads are counted, not TCP connections or all HTTP requests.
        HEAD requests and failed Zarr probes are outside this counter.
        """
        self._resolve("")
        return self._owner.traffic

    @property
    def cache_policy(self):
        self._resolve("")
        return self._owner.cache_policy

    @property
    def max_cache_bytes(self):
        self._resolve("")
        return self._owner.max_cache_bytes

    @property
    def cache_bytes(self):
        """Retained payload bytes; NONE retains no payload between reads."""
        with self._owner.lock:
            self._resolve("")
            return self._owner.cache_coordinator.cache_bytes

    @property
    def metadata_bytes(self):
        """Encoded discovery manifest size, separate from retained payload."""
        with self._owner.lock:
            self._resolve("")
            self._owner.save_manifest()
            return self._owner.metadata_bytes

    @property
    def mutable(self) -> bool:
        """The export default mutability for future exports."""
        self._resolve("")
        return self._owner.mutable

    @mutable.setter
    def mutable(self, value: bool) -> None:
        self._resolve("")
        if not isinstance(value, bool):
            raise TypeError("mutable must be a boolean")
        self._owner.mutable = value

    @property
    def is_cache_mutable(self) -> bool:
        """Whether the currently opened cache is writable."""
        self._resolve("")
        return getattr(self._owner, "is_mutable", True)

    def refresh(self):
        """Rebuild root discovery atomically; existing child handles become stale."""
        with self._owner.lock:
            self._resolve("")
            if not getattr(self._owner, "is_mutable", True):
                raise ValueError(
                    "Cannot refresh an immutable RemoteStore artifact; use a new live session or a writable artifact"
                )
            if self._path:
                raise ValueError("refresh must be called on the root store handle")
            owner = self._owner
            replacement = RemoteDiscovery(
                owner.urlpath,
                owner.storage_options,
                dataset=owner.root,
                persist_metadata=owner.disk is not None,
                _filesystem=owner._external_filesystem,
                _source_validator=owner.source_validator,
                _manifest_validator=owner.manifest_validator,
                _max_nodes=owner.max_nodes,
            )
            try:
                if not replacement.is_tree:
                    raise ValueError("Refreshed source is no longer a group")
                replacement.disk = owner.disk
                replacement.cache_policy = owner.cache_policy
                replacement.max_cache_bytes = owner.max_cache_bytes
                replacement.shared = getattr(owner, "shared", False)
                replacement.mutable = owner.mutable
                replacement.save_manifest()
                if replacement.disk is not None:
                    replacement.disk.discard_old_generations(replacement.generation)
            except BaseException:
                replacement.disk = None
                replacement.close()
                raise
            # Keep the owner and lifetime lock stable for dependent finalizers.
            users, lock = owner._users, owner.lock
            replacement._cleanup_dir = owner._cleanup_dir
            replacement.artifact_path = owner.artifact_path
            owner._close_resources()
            owner.__dict__.update(replacement.__dict__)
            owner._users, owner.lock = users, lock
            owner.cache_coordinator = CacheCoordinator(owner.max_cache_bytes)
            self._generation = owner.generation

    def close(self):
        """Release this handle; the last dependent handle closes shared resources."""
        with self._owner.lock:
            self._finalizer()

    def __enter__(self):
        self._resolve("")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def save(
        self,
        destination: str | os.PathLike,
        *,
        include_cache: bool = True,
        mutable: bool | None = None,
        overwrite: bool = False,
    ) -> str:
        """Export the current store or subtree to a portable .b2z reference archive."""
        if not isinstance(include_cache, bool):
            raise TypeError("include_cache must be a boolean")
        if mutable is not None and not isinstance(mutable, bool):
            raise TypeError("mutable must be a boolean")
        dest_abs, dest_dir = self._validate_save_destination(destination, overwrite)

        with self._owner.lock:
            self._resolve("")
            effective_mutable = self.mutable if mutable is None else mutable
            if include_cache:
                retained = self.cache_bytes
                if self.max_cache_bytes is not None and retained > self.max_cache_bytes:
                    raise ValueError(
                        f"Retained cache ({retained} bytes) exceeds max_cache_bytes ({self.max_cache_bytes})"
                    )

            if self._owner.format == "hdf5":
                self._owner._validate_refs()
                metadata = self._owner.refs
            elif self._owner.format == "b2z":
                metadata = self._owner.archive.metadata
            else:
                metadata = self._owner.metadata

            src_desc, nodes, attrs, listed, candidates = self._collect_export_nodes(include_cache)

            staging_dir = tempfile.mkdtemp(prefix="b2z-export-", dir=dest_dir)
            fd, tmp_zip = tempfile.mkstemp(prefix="export-", suffix=".b2z.tmp", dir=dest_dir)
            os.close(fd)
            try:
                exported_caches = []
                for orig_key in candidates:
                    proxy = self._owner.caches.get(orig_key)
                    if proxy is None or not proxy._cache_sizes:
                        continue
                    self._copy_leaf_carrier(orig_key, proxy, staging_dir)
                    exported_caches.append(orig_key)

                exported_manifest = {
                    "version": 1,
                    "source": src_desc,
                    "generation": self._owner.generation,
                    "nodes": nodes,
                    "attrs": attrs,
                    "listed": listed,
                    "notice": self._owner.notice,
                    "metadata": metadata,
                    "caches": sorted(exported_caches),
                    "cache_policy": self.cache_policy.value,
                    "max_cache_bytes": self.max_cache_bytes,
                    "mutable": effective_mutable,
                }

                embed_dst = os.path.join(staging_dir, "embed.b2e")
                st = blosc2.Storage(contiguous=True, urlpath=embed_dst, mode="w")
                st.meta = {"b2tree": {"version": 1}, "b2remote_store": {"version": 1}}
                embed = blosc2.SChunk(chunksize=2**13, data=None, storage=st)
                embed.vlmeta["b2remote_manifest"] = exported_manifest
                del embed

                filepaths = []
                for root, _, files in os.walk(staging_dir):
                    for file in files:
                        fp = os.path.join(root, file)
                        if os.path.abspath(fp) != os.path.abspath(embed_dst):
                            filepaths.append(fp)
                filepaths.sort(key=os.path.getsize, reverse=True)

                with zipfile.ZipFile(tmp_zip, "w", zipfile.ZIP_STORED) as zf:
                    for fp in filepaths:
                        arcname = os.path.relpath(fp, staging_dir)
                        zf.write(fp, arcname)
                    zf.write(embed_dst, "embed.b2e")

                os.replace(tmp_zip, dest_abs)
                return dest_abs
            finally:
                if os.path.exists(tmp_zip):
                    with contextlib.suppress(OSError):
                        os.unlink(tmp_zip)
                shutil.rmtree(staging_dir, ignore_errors=True)

    def _validate_save_destination(self, destination, overwrite):
        destination = os.fspath(destination)
        if not destination.endswith(".b2z"):
            raise ValueError("destination must have a .b2z extension")
        dest_abs = os.path.abspath(destination)
        if os.path.isdir(dest_abs):
            raise ValueError("destination must name a file, not a directory")
        if os.path.exists(dest_abs) and not overwrite:
            raise FileExistsError(f"'{dest_abs}' already exists. Use overwrite=True to overwrite.")
        dest_dir = os.path.dirname(dest_abs)
        if not os.path.exists(dest_dir):
            raise FileNotFoundError(f"Destination directory '{dest_dir}' does not exist")
        if self._owner.disk is not None:
            live_dir = os.path.abspath(str(self._owner.disk.path))
            cache_root = os.path.abspath(str(self._owner.disk.path.parent))
            if (
                dest_abs in (live_dir, cache_root)
                or dest_abs.startswith(live_dir + os.sep)
                or dest_abs.startswith(cache_root + os.sep)
            ):
                raise ValueError("destination cannot be inside live cache storage")
        if self._owner.artifact_path is not None and dest_abs == os.path.abspath(self._owner.artifact_path):
            raise ValueError("destination cannot be the source artifact")
        return dest_abs, dest_dir

    def _collect_export_nodes(self, include_cache):
        group_full = "/".join(p for p in (self._owner.root, self._path.strip("/")) if p)
        prefix = (group_full + "/") if group_full else ""
        exported_source = {
            "urlpath": self._owner.urlpath,
            "dataset": group_full,
            "kind": self._owner.format,
        }
        fingerprint = storage_options_fingerprint(getattr(self._owner, "storage_options", None))
        if fingerprint:
            exported_source["storage_options"] = fingerprint
        if prefix:
            exported_nodes = {
                k: (v[0], v[1] if v[0] == "unsupported" else None)
                for k, v in self._owner.nodes.items()
                if k == group_full or k.startswith(prefix)
            }
            exported_attrs = {
                k: v for k, v in self._owner.attrs.items() if k == group_full or k.startswith(prefix)
            }
            exported_listed = {
                k: list(v) for k, v in self._owner.listed.items() if k == group_full or k.startswith(prefix)
            }
            candidate_caches = (
                [k for k in self._owner.caches if k.startswith(prefix)] if include_cache else []
            )
        else:
            exported_nodes = {
                k: (v[0], v[1] if v[0] == "unsupported" else None) for k, v in self._owner.nodes.items()
            }
            exported_attrs = dict(self._owner.attrs)
            exported_listed = {k: list(v) for k, v in self._owner.listed.items()}
            candidate_caches = list(self._owner.caches) if include_cache else []
        return exported_source, exported_nodes, exported_attrs, exported_listed, candidate_caches

    def _copy_leaf_carrier(self, orig_key, proxy, staging_dir):
        leaf_filename = f"{orig_key}.b2nd"
        leaf_dst = os.path.join(staging_dir, leaf_filename)
        os.makedirs(os.path.dirname(leaf_dst), exist_ok=True)
        if self._owner.artifact_offsets is not None:
            with zipfile.ZipFile(self._owner.artifact_path, "r") as zf:
                zf.extract(leaf_filename, staging_dir)
        elif self._owner.disk is not None and not getattr(self._owner, "shared", False):
            src_file = self._owner.disk.payload_path(self._owner.generation, orig_key)
            if src_file.exists():
                shutil.copy2(src_file, leaf_dst)
        elif self._owner.artifact_path is not None and os.path.isdir(self._owner.artifact_path):
            src_file = os.path.join(self._owner.artifact_path, leaf_filename)
            if os.path.exists(src_file):
                shutil.copy2(src_file, leaf_dst)
        else:
            # An in-memory proxy cache is an NDArray, so the leaf must be built the
            # same way a disk leaf is: a matching NDArray carrier, not a bare SChunk.
            # The ``remote-store`` identity and ``proxy-source`` metalayers are what
            # the mutable-reopen path validates when it adopts the extracted leaf.
            meta = {name: proxy._schunk_cache.meta[name] for name in proxy._schunk_cache.meta}
            meta.pop("b2nd", None)
            meta["remote-store"] = {"generation": self._owner.generation, "dataset": orig_key}
            cache = proxy._cache
            if isinstance(cache, blosc2.NDArray):
                leaf = blosc2.empty(
                    cache.shape,
                    cache.dtype,
                    chunks=cache.chunks,
                    blocks=cache.blocks,
                    cparams=cache.cparams,
                    urlpath=leaf_dst,
                    mode="w",
                    meta=meta,
                )
                leaf_schunk = leaf.schunk
            else:
                st = blosc2.Storage(contiguous=True, urlpath=leaf_dst, mode="w")
                st.meta = meta
                leaf = blosc2.SChunk(chunksize=proxy._schunk_cache.chunksize, storage=st)
                leaf_schunk = leaf
            for nchunk in sorted(proxy._cache_sizes):
                chunk = proxy._schunk_cache.get_chunk(nchunk)
                if chunk is not None:
                    leaf_schunk.update_chunk(nchunk, chunk)
            for key, value in proxy._schunk_cache.vlmeta.items():
                leaf_schunk.vlmeta[key] = value
            del leaf

    @classmethod
    def _load_artifact_manifest(cls, urlpath):
        if os.path.isdir(urlpath):
            for directory, dirs, files in os.walk(urlpath):
                if any(os.path.islink(os.path.join(directory, name)) for name in dirs + files):
                    raise ValueError("RemoteStore artifacts cannot contain symbolic links")
            embed_path = os.path.join(urlpath, "embed.b2e")
            if not os.path.exists(embed_path):
                raise ValueError("Invalid RemoteStore artifact: missing embed.b2e")
            embed = blosc2.blosc2_ext.open(embed_path, "r", 0)
            manifest = embed.vlmeta.get("b2remote_manifest")
            del embed
            artifact_offsets = None
        else:
            artifact_offsets = get_zip_offsets(urlpath)
            if "embed.b2e" not in artifact_offsets:
                raise ValueError("Invalid RemoteStore artifact: missing embed.b2e")
            compressed = [name for name, info in artifact_offsets.items() if not info["stored"]]
            if compressed:
                raise ValueError(
                    "Invalid RemoteStore artifact: ZIP members must be ZIP_STORED, "
                    f"found compressed {sorted(compressed)}"
                )
            embed_offset = artifact_offsets["embed.b2e"]["offset"]
            embed = blosc2.blosc2_ext.open(urlpath, "r", embed_offset)
            manifest = embed.vlmeta.get("b2remote_manifest")
            del embed

        if not isinstance(manifest, dict) or manifest.get("version") != 1 or "source" not in manifest:
            raise ValueError("Invalid RemoteStore manifest in artifact")
        if not isinstance(manifest.get("mutable", False), bool):
            raise ValueError("Invalid RemoteStore manifest mutable flag")
        source = manifest["source"]
        if not isinstance(source, dict) or not isinstance(source.get("urlpath"), str):
            raise ValueError("Invalid RemoteStore manifest source descriptor")
        validate_persistable_url(source["urlpath"])
        cls._validate_artifact_manifest(manifest)
        return manifest, artifact_offsets

    @staticmethod
    def _validate_artifact_manifest(manifest):
        from blosc2.remote_store_cache import validate_generation

        validate_generation(manifest.get("generation"))
        source = manifest["source"]
        root = source.get("dataset", "")
        RemoteDiscovery._validate(root)
        if source.get("kind") not in {"b2z", "hdf5", "zarr"}:
            raise ValueError("Invalid RemoteStore source kind")
        for field in ("nodes", "attrs", "listed", "metadata"):
            if not isinstance(manifest.get(field), dict):
                raise ValueError(f"Invalid RemoteStore manifest {field}")
        nodes = manifest["nodes"]
        for path, entry in nodes.items():
            RemoteDiscovery._validate(path)
            if (
                not isinstance(entry, (list, tuple))
                or len(entry) != 2
                or entry[0] not in {"group", "ndarray", "unsupported"}
            ):
                raise ValueError("Invalid RemoteStore node")
        if root not in nodes:
            raise ValueError("Missing RemoteStore root")
        if any(path not in nodes for field in ("attrs", "listed") for path in manifest[field]):
            raise ValueError("Invalid RemoteStore metadata path")
        for parent, children in manifest["listed"].items():
            if not isinstance(children, list) or any(
                not isinstance(child, str) or child not in nodes or child.rpartition("/")[0] != parent
                for child in children
            ):
                raise ValueError("Invalid RemoteStore child list")
        caches = manifest.get("caches")
        if not isinstance(caches, list):
            raise ValueError("Invalid RemoteStore caches")
        for path in caches:
            RemoteDiscovery._validate(path)
            if (
                path not in nodes
                or nodes[path][0] != "ndarray"
                or (root and not path.startswith(root + "/"))
            ):
                raise ValueError("Invalid cached RemoteStore leaf")

    @classmethod
    def _open_mutable_artifact(cls, urlpath, manifest, storage_options, cache_policy, limit, cache_dir):
        if cache_policy is not blosc2.CachePolicy.DISK:
            raise ValueError(
                "Mutable RemoteStore artifacts require CachePolicy.DISK: their payload is staged "
                "into writable runtime storage. Use the default policy or supply cache_dir."
            )
        cleanup_dir = None
        if cache_dir is not None:
            staged_cache_dir = cache_dir
        else:
            cleanup_dir = tempfile.TemporaryDirectory(prefix="remote-store-staged-")
            staged_cache_dir = cleanup_dir.name

        from blosc2.remote_store_cache import StoreDiskCache

        source_desc = manifest["source"]
        try:
            owner = RemoteDiscovery(
                source_desc["urlpath"],
                storage_options,
                dataset=source_desc.get("dataset"),
                manifest=manifest,
                persist_metadata=True,
            )
        except BaseException:
            if cleanup_dir is not None:
                cleanup_dir.cleanup()
            raise

        owner.is_mutable = True
        owner.mutable = False
        owner.artifact_path = os.path.abspath(urlpath)
        owner.cache_policy = cache_policy
        owner.max_cache_bytes = limit
        owner.cache_coordinator = CacheCoordinator(limit)
        owner._cleanup_dir = cleanup_dir
        owner.restoring = True
        try:
            disk = StoreDiskCache(staged_cache_dir, source_desc)
            owner.disk = disk
            gen = manifest["generation"]
            b2d_dir = disk.path / f"{gen}.b2d"
            b2d_dir.mkdir(parents=True, exist_ok=True)
            if os.path.isdir(urlpath):
                shutil.copytree(urlpath, b2d_dir, dirs_exist_ok=True)
            else:
                with zipfile.ZipFile(urlpath, "r") as zf:
                    zf.extractall(b2d_dir)
            owner.restore_caches(manifest)
            owner.cache_coordinator.enforce()
            owner.save_manifest()
            disk.discard_old_generations(gen)
        except BaseException:
            owner.close()
            raise
        return owner

    @classmethod
    def _open_immutable_artifact(
        cls, urlpath, manifest, artifact_offsets, storage_options, cache_policy, limit, cache_dir
    ):
        if cache_dir is not None:
            raise ValueError("cache_dir cannot be specified for an immutable RemoteStore artifact")
        source_desc = manifest["source"]
        owner = RemoteDiscovery(
            source_desc["urlpath"],
            storage_options,
            dataset=source_desc.get("dataset"),
            manifest=manifest,
            persist_metadata=False,
        )
        owner.disk = None
        owner.is_mutable = False
        owner.mutable = False
        owner.artifact_path = os.path.abspath(urlpath)
        owner.artifact_offsets = artifact_offsets
        owner.cache_policy = cache_policy
        owner.max_cache_bytes = limit
        owner.cache_coordinator = CacheCoordinator(None)
        try:
            owner.restore_caches(manifest)
            retained = owner.cache_coordinator.cache_bytes
            if limit is not None and retained > limit:
                raise ValueError(
                    f"Requested max_cache_bytes ({limit}) is smaller than retained immutable "
                    f"payload ({retained} bytes); an immutable snapshot cannot be trimmed in place. "
                    f"Use a larger allowance or produce a smaller/cold export."
                )
            owner.cache_coordinator.max_cache_bytes = limit
        except BaseException:
            owner.close()
            raise
        return owner

    @classmethod
    def _open_artifact(cls, urlpath, mode="r", **kwargs):
        urlpath = os.fspath(urlpath)
        if not os.path.exists(urlpath):
            raise FileNotFoundError(f"Artifact {urlpath} does not exist")
        if mode not in ("r", "a"):
            raise ValueError(f"RemoteStore artifacts only support modes 'r' and 'a', not {mode!r}")
        manifest, artifact_offsets = cls._load_artifact_manifest(urlpath)
        if mode == "a" and not manifest.get("mutable", False):
            raise ValueError(
                "Immutable RemoteStore artifacts are read-only; export a mutable artifact to open for writes"
            )

        storage_options = kwargs.get("storage_options")
        cache_policy = kwargs.get("cache_policy")
        max_cache_bytes = kwargs.get("max_cache_bytes")
        cache_dir = kwargs.get("cache_dir")

        if cache_policy is not None:
            if not isinstance(cache_policy, blosc2.CachePolicy):
                raise TypeError("cache_policy must be a blosc2.CachePolicy instance")
            if cache_policy is blosc2.CachePolicy.NONE and manifest.get("caches"):
                raise ValueError(
                    "Cannot reopen a warm RemoteStore artifact with CachePolicy.NONE; "
                    "use a cold export or a cache policy that permits retained payload."
                )
        else:
            cache_policy = blosc2.CachePolicy.DISK

        if cache_policy is blosc2.CachePolicy.NONE:
            if max_cache_bytes is not None:
                raise ValueError("max_cache_bytes is not applicable to CachePolicy.NONE")
            limit = None
        elif max_cache_bytes is not None:
            limit = normalize_cache_limit(cache_policy, max_cache_bytes)
        else:
            # A cold artifact may still carry the exporting store's allowance; NONE
            # retains nothing, so its saved limit must not be treated as a conflict.
            saved_limit = manifest.get("max_cache_bytes")
            limit = normalize_cache_limit(cache_policy, saved_limit) if saved_limit is not None else None

        if manifest.get("mutable", False):
            owner = cls._open_mutable_artifact(
                urlpath, manifest, storage_options, cache_policy, limit, cache_dir
            )
        else:
            owner = cls._open_immutable_artifact(
                urlpath, manifest, artifact_offsets, storage_options, cache_policy, limit, cache_dir
            )

        store = cls.__new__(cls)
        store._attach(owner, "")
        req_dataset = kwargs.get("dataset")
        if req_dataset:
            return store[req_dataset]
        return store
