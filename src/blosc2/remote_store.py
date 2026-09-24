"""Read-only remote hierarchy discovery and public store handles."""

from __future__ import annotations

import contextlib
import hashlib
import os
import shutil
import tempfile
import threading
import uuid
import weakref
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
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
from blosc2.remote_object import RemoteObject

RESERVED_NAMES = {"embed.b2e", "__vlmeta__"}


def validate_remote_store_reference(descriptor):
    """Validate and normalize a persisted RemoteStore reference."""
    if not isinstance(descriptor, dict):
        raise ValueError("RemoteStore reference must be a mapping")
    if descriptor.get("version") != 1:
        raise ValueError("Unsupported RemoteStore reference version")
    kind = descriptor.get("kind")
    if kind not in {"b2z", "hdf5", "zarr"}:
        raise ValueError("Invalid RemoteStore reference kind")
    urlpath = descriptor.get("urlpath")
    if not isinstance(urlpath, str):
        raise ValueError("RemoteStore reference urlpath must be a string")
    validate_persistable_url(urlpath)
    dataset = descriptor.get("dataset", "")
    if not isinstance(dataset, str):
        raise ValueError("RemoteStore reference dataset must be a string")
    dataset = dataset.strip("/")
    RemoteDiscovery._validate(dataset)
    policy = descriptor.get("cache_policy", blosc2.CachePolicy.MEMORY.value)
    try:
        policy = blosc2.CachePolicy(policy)
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid RemoteStore reference cache policy") from exc
    if policy is blosc2.CachePolicy.DISK:
        policy = blosc2.CachePolicy.MEMORY
    limit = descriptor.get("max_cache_bytes")
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 0):
        raise ValueError("Invalid RemoteStore reference cache limit")
    return {
        "kind": kind,
        "version": 1,
        "urlpath": urlpath,
        "dataset": dataset,
        "assume_immutable": True,
        "cache_policy": policy.value,
        "max_cache_bytes": limit,
    }


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


def _resolve_hdf5_options(hdf5_index, private_index, source_format):
    if hdf5_index is not None and private_index is not None:
        raise TypeError("hdf5_index was supplied twice")
    index = hdf5_index if hdf5_index is not None else private_index
    return index, "hdf5" if index is not None and source_format is None else source_format


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
        _source_format=None,
        _hdf5_index=None,
        _hdf5_blob=None,
        _traffic=None,
        _source_cache_dir=None,
        _refresh_source=False,
        _b2z_blob=None,
        _local_source=False,
    ):
        self.urlpath, dataset, self.format = parse_container_url(urlpath, dataset)
        if _source_format is not None:
            self.format = _source_format
        elif manifest is not None:
            self.format = manifest["source"]["kind"]
        self.root = (dataset or "").strip("/")
        self._validate(self.root)
        self.storage_options = storage_options or {}
        self.local_source = _local_source
        self.source_cache_dir = _source_cache_dir
        self.refresh_source = _refresh_source
        self.b2z_source_cache = (None, None, None)
        if self.format == "b2z":
            from blosc2.b2z_source import SMALL_REMOTE_FILE, b2z_source_cache
            from blosc2.remote_source_cache import source_version

            self.b2z_source_cache = b2z_source_cache(
                self.urlpath, _source_cache_dir, self.storage_options, refresh=_refresh_source
            )
            marker = self.b2z_source_cache[1]
            if (
                self.b2z_source_cache[2] is None
                and _b2z_blob is not None
                and (marker is None or hashlib.sha256(_b2z_blob).hexdigest() == marker["token"])
            ):
                self.b2z_source_cache = (*self.b2z_source_cache[:2], _b2z_blob)
            if manifest and marker and source_version(manifest["metadata"]) != marker["token"]:
                manifest = None
            if (
                manifest
                and _source_cache_dir is not None
                and not source_version(manifest["metadata"])
                and SMALL_REMOTE_FILE
                and manifest["metadata"].get("object_info", {}).get("size", 0) <= SMALL_REMOTE_FILE
            ):
                manifest = None  # Rebind legacy metadata/payload to the source digest once.
        self.traffic = _traffic if _traffic is not None else Traffic()
        self.nodes = {}
        self.attrs = {}
        self.listed = {}
        self.notice = None
        self.archive = None
        self.zstore = None
        self.hdf5_blob = None
        self.hdf5_source_cache_path = None
        if _hdf5_blob is not None:
            self.hdf5_blob = _hdf5_blob
        self.sources = {}
        self.source_descriptors = {}
        self.caches = {}
        self.batch_caches = {}
        self.linked_stores = {}
        self.linked_artifacts = {}
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
        self.cache_namespace = ""
        self.nested_storage_options = None
        # ponytail: serialize store operations; finer locks if multi-leaf throughput matters.
        self.lock = threading.RLock()
        try:
            self.filesystem = _filesystem
            if self.filesystem is None and not (self.format == "hdf5" and os.path.isfile(self.urlpath)):
                import fsspec

                options = {**self.storage_options, "skip_instance_cache": True}
                self.filesystem, _ = fsspec.core.url_to_fs(self.urlpath, **options)
            self.restored_manifest = manifest
            if manifest:
                self._restore_manifest(manifest)
            elif self.format == "b2z":
                self._open_b2z()
            elif self.format == "hdf5":
                self._open_hdf5(_hdf5_index)
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
                or entry[0] not in {"group", "ndarray", "ctable", "remote_store", "unsupported"}
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
                _source_cache=self.b2z_source_cache,
                _refresh=self.refresh_source,
            )
        elif self.format == "hdf5":
            self.hdf5_index = self.metadata
            self._validate_hdf5_index()
            # JSON round trips lose the shared table metadata used by lazy index discovery.
            for path, (kind, _) in self.nodes.items():
                if kind == "ctable":
                    self.nodes[path] = (kind, self.hdf5_index["datasets"][path])
        elif self.format == "zarr" and self.zstore is None:
            self._open_zarr()
        self._check_node_limit()

    def _validate_hdf5_index(self):
        from blosc2.hdf5_source import validate_hdf5_index

        validate_hdf5_index(self.hdf5_index, self.urlpath, dataset=self.root or None)

    def save_manifest(self):
        if self.disk is None or self.restoring or not self.is_mutable:
            return
        # ponytail: publish whole discovery snapshots; add dirty tracking if large maps make this costly.
        if self.format == "hdf5":
            self._validate_hdf5_index()
            self.metadata = self.hdf5_index
        elif self.format == "b2z":
            self.metadata = self.archive.metadata
        nodes = {
            path: (kind, value if kind in {"ctable", "remote_store", "unsupported"} else None)
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
            "batch_caches": sorted(self.batch_caches),
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

    def _find_b2z_object_roots(self, members, embedded, registry):
        from blosc2.b2z_source import B2ZEmbeddedMetadata, member_vlmeta

        roots = {
            key.strip("/"): None
            for key, value in registry.items()
            if isinstance(value, dict) and value.get("kind") == "ctable"
        }
        embedded_reader = None
        # Match TreeStore's legacy CTable manifest check using metadata only.
        for name, info in members.items():
            if name == "_meta.b2f" or name.endswith("/_meta.b2f"):
                metadata = member_vlmeta(self.archive, info)
                if metadata.get("kind") in {"ctable", b"ctable"}:
                    roots[name.rpartition("/")[0]] = metadata
        for key, entry in embedded.items():
            if key.endswith("/_meta"):
                try:
                    if embedded_reader is None:
                        embedded_reader = B2ZEmbeddedMetadata(self.archive, members["embed.b2e"])
                    attrs = embedded_reader.attrs(entry)
                    if attrs.get("kind") in {"ctable", b"ctable"}:
                        roots[key.rpartition("/")[0].strip("/")] = attrs
                except NotImplementedError as exc:
                    roots.setdefault(key.rpartition("/")[0].strip("/"), None)
                    self.notice = f"Partial B2Z metadata: object boundary cannot be verified: {exc}."
        references = {}
        for key, value in registry.items():
            if not isinstance(value, dict) or value.get("kind") != "remote_store":
                continue
            path = key.strip("/")
            try:
                if value.get("version") != 1 or value.get("layout") != "reference":
                    raise ValueError("unsupported registry entry")
                references[path] = ("remote_store", validate_remote_store_reference(value.get("source")))
            except ValueError as exc:
                references[path] = ("unsupported", f"Invalid RemoteStore reference: {exc}")
        return roots, references, embedded_reader

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
            _source_cache=self.b2z_source_cache,
            _refresh=self.refresh_source,
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

        roots, references, embedded_reader = self._find_b2z_object_roots(members, embedded, registry)
        if "" in roots:
            metadata = roots[""]
            self.nodes[""] = (
                ("ctable", metadata)
                if metadata is not None
                else ("unsupported", "CTable metadata is unavailable")
            )
            self.archive._opening_ranges.clear()
            self.archive.capture_metadata = False
            return
        self._add("", "group")
        for root in sorted(roots, key=len):
            if not any(root.startswith(other + "/") for other in roots if other != root):
                metadata = roots[root]
                if metadata is None:
                    self._add(root, "unsupported", "CTable metadata is unavailable")
                else:
                    self._add(root, "ctable", metadata)
        object_roots = {*roots, *references}
        for path, (kind, value) in references.items():
            if not any(path.startswith(other + "/") for other in object_roots if other != path):
                self._add(path, kind, value)
        self._process_b2z_members(members, object_roots)
        self._process_b2z_embedded(embedded, object_roots, embedded_reader, members)
        self.archive._opening_ranges.clear()
        self.archive.capture_metadata = False

    def _open_hdf5(self, hdf5_index=None):
        from blosc2.hdf5_source import decode_hdf5_value, load_hdf5_index, scan_hdf5_index

        unsupported = {}
        if hdf5_index is None:
            self.hdf5_index, self.hdf5_blob = scan_hdf5_index(
                self.urlpath,
                self.storage_options,
                dataset=self.root or None,
                unsupported=unsupported,
                traffic=self.traffic,
                _filesystem=self.filesystem,
                _lazy_allocations=True,
                _return_blob=True,
                _blob=self.hdf5_blob,
            )
        else:
            self.hdf5_index = load_hdf5_index(
                hdf5_index,
                self.urlpath,
                self.storage_options,
                filesystem=self.filesystem,
                dataset=self.root or None,
            )
        for path, metadata in self.hdf5_index["groups"].items():
            self._add(path, "group")
            self.attrs[path] = {
                key: decode_hdf5_value(value) for key, value in metadata.get("attrs", {}).items()
            }
        for path, metadata in self.hdf5_index["datasets"].items():
            self._add(path, "ctable" if metadata.get("kind") == "ctable" else "ndarray", metadata)
            self.attrs[path] = {
                key: decode_hdf5_value(value) for key, value in metadata.get("attrs", {}).items()
            }
        for path, message in unsupported.items():
            self._validate(path)
            self.nodes[path] = ("unsupported", message)
        self.notice = "HDF5 view includes indexed groups and datasets; external and soft links are omitted."

    def ensure_hdf5_allocations(self, path):
        """Populate one deferred HDF5 allocation map and return its metadata."""
        from blosc2.hdf5_source import scan_hdf5_allocations

        with self.lock:
            metadata = self.hdf5_index["datasets"][path]
            if metadata["allocated"] is None:
                metadata["allocated"] = scan_hdf5_allocations(
                    self.urlpath,
                    path,
                    self.storage_options,
                    traffic=self.traffic,
                    _filesystem=self.filesystem,
                    _blob=self.hdf5_blob,
                )
                self.save_manifest()
            return metadata

    def ensure_hdf5_allocations_many(self, paths):
        """Populate deferred HDF5 allocation maps in one scan."""
        from blosc2.hdf5_source import scan_hdf5_allocations_many

        with self.lock:
            missing = [path for path in paths if self.hdf5_index["datasets"][path]["allocated"] is None]
            if missing:
                allocations = scan_hdf5_allocations_many(
                    self.urlpath,
                    missing,
                    self.storage_options,
                    traffic=self.traffic,
                    _filesystem=self.filesystem,
                    _blob=self.hdf5_blob,
                )
                for path, allocated in allocations.items():
                    self.hdf5_index["datasets"][path]["allocated"] = allocated
                self.save_manifest()

    def attach_hdf5_source_cache(self, path, marker):
        self.hdf5_source_cache_path = path
        self.hdf5_source_cache_marker = marker
        if path is not None and self.hdf5_blob is not None:
            self.hdf5_index = dict(self.hdf5_index)
            self.hdf5_index["source_sha256"] = hashlib.sha256(self.hdf5_blob).hexdigest()
        elif marker is not None:
            self.hdf5_index["source_cache_version"] = marker["token"]

    def publish_source_cache(self, *, refresh=False):
        if self.format == "b2z" and self.archive is not None:
            self.archive.publish_source(refresh=refresh)
        if self.hdf5_source_cache_path is not None:
            from blosc2.hdf5_source import publish_hdf5_source_cache

            publish_hdf5_source_cache(
                self.hdf5_source_cache_path,
                self.hdf5_blob,
                self.hdf5_index,
                expected=getattr(self, "hdf5_source_cache_marker", None),
                refresh=refresh,
            )

    def ensure_pytables_indexes(self, table_path):
        """Discover one table's hidden PyTables index nodes on first use."""
        from blosc2.hdf5_source import decode_hdf5_value, scan_pytables_indexes

        with self.lock:
            metadata = self.hdf5_index["datasets"][table_path]
            if "pytables_indexes" in metadata:
                return
            groups, datasets, indexes = scan_pytables_indexes(
                self.urlpath,
                table_path,
                metadata,
                self.storage_options,
                traffic=self.traffic,
                _filesystem=self.filesystem,
                _blob=self.hdf5_blob,
            )
            self.hdf5_index["groups"].update(groups)
            self.hdf5_index["datasets"].update(datasets)
            metadata["pytables_indexes"] = indexes
            for path, item in groups.items():
                if path not in self.nodes:
                    self._add(path, "group")
                self.attrs[path] = {
                    key: decode_hdf5_value(value) for key, value in item.get("attrs", {}).items()
                }
            for path, item in datasets.items():
                if path not in self.nodes:
                    self._add(path, "ndarray")
                self.attrs[path] = {
                    key: decode_hdf5_value(value) for key, value in item.get("attrs", {}).items()
                }
            self.save_manifest()

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
        if kind != "ndarray" and not (kind == "ctable" and self.format == "hdf5"):
            raise NotImplementedError(
                value if isinstance(value, str) else "Array access is unavailable for this node"
            )
        if self.format == "b2z":
            source = self._open_b2z_source(full)
        elif self.format == "hdf5":
            from blosc2.hdf5_source import HDF5NDSource

            source = HDF5NDSource(
                self.urlpath,
                full,
                hdf5_index=self.hdf5_index,
                storage_options=self.storage_options,
                _traffic=self.traffic,
                _filesystem=self.filesystem,
                _blob=self.hdf5_blob,
                _ensure_allocations=self.ensure_hdf5_allocations,
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

    def open_ctable_array(self, table_path, logical_key):
        """Open one external NDArray member hidden below a CTable node."""
        if self.format != "b2z":
            raise NotImplementedError("Remote CTable access currently requires a B2Z source")
        full = "/".join(part.strip("/") for part in (table_path, logical_key) if part.strip("/"))
        self._validate(full)
        if full in self.sources:
            return self.sources[full]
        matches = [info for info in self.archive.members if info.filename == full + ".b2nd"]
        if len(matches) != 1:
            raise NotImplementedError(f"Remote CTable array {full!r} is unavailable or not external")
        info = matches[0]
        if info.flag_bits & 1 or info.compress_type != 0:
            raise NotImplementedError(
                f"Remote CTable array {full!r} requires an unencrypted ZIP_STORED member"
            )
        source = self._open_b2z_source(full)
        if self.source_validator is not None:
            self.source_validator(source)
        self.nodes[full] = ("ndarray", None)
        self.sources[full] = source
        return source

    def open_ctable_carrier(self, table_path, logical_key):
        """Open a RemoteArray carrier stored as a CTable column."""
        if self.format != "b2z":
            raise NotImplementedError("Remote CTable access currently requires a B2Z source")
        full = "/".join(part.strip("/") for part in (table_path, logical_key) if part.strip("/"))
        self._validate(full)
        matches = [info for info in self.archive.members if info.filename == full + ".b2nd"]
        if len(matches) != 1:
            raise NotImplementedError(f"Remote CTable source carrier {full!r} is unavailable")
        offset, length = self.archive.member_window(matches[0])
        carrier = blosc2.ndarray_from_cframe(self.archive._read_archive(offset, length), copy=True)
        return blosc2.RemoteArray._from_carrier_with_owner(carrier, self, full + ".source")

    def open_ctable_batch(self, full):
        """Open one external BatchArray member hidden below a CTable node."""
        if full in self.batch_caches:
            return self.batch_caches[full]
        if self.format != "b2z":
            raise NotImplementedError("Remote CTable access currently requires a B2Z source")
        self._validate(full)
        from blosc2.b2z_source import B2ZBatchSource

        generation = self.generation

        def check_open():
            if self._closed:
                raise RuntimeError("RemoteCTable handle is closed")
            if self.generation != generation:
                raise RuntimeError("RemoteCTable handle is stale; look it up again after refresh")

        self.archive.capture_metadata = True
        try:
            source = B2ZBatchSource(self.archive, full, check_open)
        finally:
            self.archive.capture_metadata = False
            self.archive._opening_ranges.clear()
        if self.cache_policy is blosc2.CachePolicy.NONE:
            return source
        from blosc2.remote_batch import _RemoteBatchCache

        path = None
        artifact = None
        if self.disk is not None:
            path = self.disk.batch_payload_path(self.generation, full)
        elif not self.is_mutable:
            if self.artifact_offsets is not None:
                prefix = f"{full}.b2b.cache/"
                offsets = {}
                for name, info in self.artifact_offsets.items():
                    if name.startswith(prefix):
                        suffix = name[len(prefix) :]
                        if not suffix.endswith(".chunk") or not suffix[:-6].isdigit():
                            raise ValueError("Invalid cached batch member")
                        offsets[int(suffix[:-6])] = info
                artifact = self.artifact_path, offsets
            elif self.artifact_path is not None:
                path = os.path.join(self.artifact_path, f"{full}.b2b.cache")
        key = f"{self.cache_namespace}:{full}" if self.cache_namespace else full
        self.batch_caches[full] = _RemoteBatchCache(
            source, key, self.cache_coordinator, path, read_only=not self.is_mutable, artifact=artifact
        )
        return self.batch_caches[full]

    def load_ctable_attrs(self, table_path):
        """Load one table's user attributes without opening its data arrays."""
        if table_path in self.attrs:
            return dict(self.attrs[table_path])
        member = "/".join(part for part in (table_path, "_vlmeta.b2f") if part)
        matches = [info for info in self.archive.members if info.filename == member]
        if len(matches) > 1:
            raise ValueError(f"Duplicate Remote CTable metadata member {member!r}")
        if matches:
            from blosc2.b2z_source import member_vlmeta

            self.attrs[table_path] = dict(member_vlmeta(self.archive, matches[0]))
        else:
            self.attrs[table_path] = {}
        self.save_manifest()
        return dict(self.attrs[table_path])

    def _open_b2z_source(self, full):
        from blosc2.b2z_source import B2ZNDSource

        # Immutable archives trust their persisted identity when restoring leaves.
        seeds = self.archive.metadata.get("ctable_seeds", {})
        if full in seeds and self.archive.blob is None:
            return B2ZNDSource(
                self.urlpath,
                full,
                _seed=seeds[full],
                storage_options=self.storage_options,
                _filesystem=self.filesystem,
                _traffic=self.traffic,
            )
        source = B2ZNDSource(self.urlpath, full, _archive=self.archive)
        if self.persist_metadata and any(
            kind == "ctable" and full.startswith(root + "/" if root else "")
            for root, (kind, _) in self.nodes.items()
        ):
            self.archive.metadata.setdefault("ctable_seeds", {})[full] = source._seed
        return source

    def remote_array(self, full):
        """Return a RemoteArray sharing this discovery owner's resources."""
        if self.format == "hdf5" and full in self.nodes:
            from blosc2.hdf5_source import HDF5NDSource

            source = self.sources.get(full)
            if source is None:
                source = HDF5NDSource(
                    self.urlpath,
                    full,
                    hdf5_index=self.hdf5_index,
                    storage_options=self.storage_options,
                    _traffic=self.traffic,
                    _filesystem=self.filesystem,
                    _blob=self.hdf5_blob,
                    _ensure_allocations=self.ensure_hdf5_allocations,
                )
                if self.source_validator is not None:
                    self.source_validator(source)
                self.sources[full] = source
        else:
            relative = full[len(self.root) + 1 :] if self.root else full
            source = self.open_source(relative) if full in self.nodes else None
        if source is None:
            raise KeyError(full)
        descriptor = {
            "kind": self.format,
            "version": 1,
            "urlpath": source.urlpath,
            "assume_immutable": True,
        }
        if self.format in {"b2z", "hdf5"}:
            descriptor["dataset"] = full
        return blosc2.RemoteArray(
            source,
            _source_descriptor=descriptor,
            _store_owner=self,
            cache_policy=self.cache_policy,
            max_cache_bytes=(
                self.max_cache_bytes
                if self.cache_policy is not blosc2.CachePolicy.NONE
                else CACHE_POLICY_DEFAULT
            ),
        )

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
                if path.endswith(".source") and path not in self.nodes:
                    # External CTable columns are registered lazily when their
                    # persisted carrier is opened.  get_cache() adopts this
                    # artifact leaf after that source has been authorized.
                    continue
                if path not in self.nodes or (
                    self.nodes[path][0] != "ndarray"
                    and not (self.format == "hdf5" and self.nodes[path][0] == "ctable")
                ):
                    raise ValueError("Invalid cached RemoteStore leaf")
                if self.format == "hdf5":
                    from blosc2.hdf5_source import HDF5NDSource

                    source = self.sources.get(path)
                    if source is None:
                        source = HDF5NDSource(
                            self.urlpath,
                            path,
                            hdf5_index=self.hdf5_index,
                            storage_options=self.storage_options,
                            _traffic=self.traffic,
                            _filesystem=self.filesystem,
                            _blob=self.hdf5_blob,
                            _ensure_allocations=self.ensure_hdf5_allocations,
                        )
                        self.sources[path] = source
                    self.get_cache(source)
                    continue
                relative = path[len(self.root) + 1 :] if self.root else path
                self.resolve(relative)
                self.get_cache(self.open_source(relative))
            for path in manifest.get("batch_caches", []):
                self.open_ctable_batch(path)
            self.cache_coordinator.enforce()
            self.restoring = False

    def get_cache(self, source, *, seed=None):
        key = next(path for path, value in self.sources.items() if value is source)
        coordinator_key = f"{self.cache_namespace}:{key}" if self.cache_namespace else key
        if key not in self.caches:
            if getattr(self, "shared", False):
                descriptor = self.source_descriptors.get(key)
                if descriptor is None:
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
                proxy._cache_key = coordinator_key
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
                    _cache_key=coordinator_key,
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
                _cache_key=coordinator_key,
                _persistent_dirty=self.disk is not None,
                meta={"remote-store": identity} if path is not None and not exists else None,
            )
            if exists and self.caches[key].schunk.meta.get("remote-store") != identity:
                self.caches.pop(key)
                raise ValueError("RemoteStore payload generation or dataset mismatch")
            self.save_manifest()
        return self.caches[key]

    def prepare_refresh(self, kind):
        """Prepare fresh discovery without publishing or retiring the current generation."""
        if not self.is_mutable:
            raise ValueError("Cannot refresh an immutable remote artifact; use a writable cache")
        replacement = RemoteDiscovery(
            self.urlpath,
            self.storage_options,
            dataset=self.root,
            persist_metadata=self.disk is not None,
            _filesystem=self._external_filesystem,
            _source_validator=self.source_validator,
            _manifest_validator=self.manifest_validator,
            _max_nodes=self.max_nodes,
            _source_format=self.format,
            _source_cache_dir=self.source_cache_dir,
            _refresh_source=True,
            _local_source=self.local_source,
        )
        try:
            if replacement.nodes[replacement.root][0] != kind:
                raise ValueError(f"Refreshed source is no longer a {kind}")
            replacement.cache_policy = self.cache_policy
            replacement.nested_storage_options = self.nested_storage_options
            replacement.max_cache_bytes = self.max_cache_bytes
            replacement.cache_coordinator = CacheCoordinator(self.max_cache_bytes)
            replacement.shared = getattr(self, "shared", False)
            replacement.mutable = self.mutable
            replacement.restoring = True
            replacement.disk = self.disk
            replacement.hdf5_source_cache_path = self.hdf5_source_cache_path
            if self.hdf5_source_cache_path is not None and replacement.hdf5_blob is None:
                # A large replacement has no retained bytes, but must invalidate
                # sibling scopes that still describe the previous small source.
                replacement.hdf5_index["source_cache_version"] = hashlib.sha256(
                    uuid.uuid4().bytes
                ).hexdigest()
            return replacement
        except BaseException:
            replacement.close()
            raise

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
        for store in getattr(self, "linked_stores", {}).values():
            store.close()
        getattr(self, "linked_stores", {}).clear()
        if self.archive is not None:
            self.archive.close()
            self.archive = None
        if self.zstore is not None:
            self.zstore.close()
            self.zstore = None
        for source in self.sources.values():
            if isinstance(source, blosc2.HDF5NDSource):
                source.close()
        self.sources.clear()
        getattr(self, "source_descriptors", {}).clear()
        self.caches.clear()
        getattr(self, "batch_caches", {}).clear()
        self.nodes.clear()
        self.attrs.clear()
        self.listed.clear()
        self.hdf5_index = None
        if self.filesystem is not None and self._external_filesystem is None:
            # s3fs already registered this exact close through weakref.finalize,
            # and its aiobotocore cleanup is not idempotent.  Other fsspec
            # clients (notably HTTP) still need deterministic closure here.
            session = getattr(self.filesystem, "_session", None)
            close = getattr(self.filesystem, "close_session", None)
            if close is not None and session is not None and not hasattr(self.filesystem, "_s3creator"):
                close(self.filesystem.loop, session)
        self.filesystem = None

    def _export_metadata(self):
        if self.format == "hdf5":
            self._validate_hdf5_index()
            return self.hdf5_index
        if self.format != "b2z":
            return self.metadata
        metadata = dict(self.archive.metadata)
        ranges = list(metadata.get("ranges", ()))
        for offset, data in (
            *self.archive._captured_ranges,
            *self.archive._opening_ranges,
            *self.archive._batch_ranges,
        ):
            if not any(
                start <= offset and offset + len(data) <= start + len(saved) for start, saved in ranges
            ):
                ranges.append((offset, data))
        metadata["ranges"] = ranges
        return metadata

    def save_selection(
        self,
        full_path,
        destination: str | os.PathLike,
        *,
        include_cache: bool = True,
        mutable: bool | None = None,
        overwrite: bool = False,
    ) -> str:
        """Export the current store or subtree to a portable .b2z reference archive."""
        if self.local_source:
            raise ValueError("local-source caches cannot be exported as portable RemoteStore references")
        if not isinstance(include_cache, bool):
            raise TypeError("include_cache must be a boolean")
        if mutable is not None and not isinstance(mutable, bool):
            raise TypeError("mutable must be a boolean")
        dest_abs, dest_dir = self._validate_save_destination(destination, overwrite)

        with self.lock:
            effective_mutable = self.mutable if mutable is None else mutable
            if include_cache:
                retained = self.cache_coordinator.cache_bytes
                if self.max_cache_bytes is not None and retained > self.max_cache_bytes:
                    raise ValueError(
                        f"Retained cache ({retained} bytes) exceeds max_cache_bytes ({self.max_cache_bytes})"
                    )

            metadata = self._export_metadata()

            src_desc, nodes, attrs, listed, candidates = self._collect_export_nodes(full_path, include_cache)

            staging_dir = tempfile.mkdtemp(prefix="b2z-export-", dir=dest_dir)
            fd, tmp_zip = tempfile.mkstemp(prefix="export-", suffix=".b2z.tmp", dir=dest_dir)
            os.close(fd)
            try:
                exported_caches = []
                for orig_key in candidates:
                    proxy = self.caches.get(orig_key)
                    if proxy is None or not proxy._cache_sizes:
                        continue
                    self._copy_leaf_carrier(orig_key, proxy, staging_dir)
                    exported_caches.append(orig_key)

                exported_batches = self._export_batch_caches(full_path, staging_dir) if include_cache else []

                linked_exports = self._export_linked_stores(full_path, staging_dir) if include_cache else {}

                exported_manifest = {
                    "version": 1,
                    "source": src_desc,
                    "generation": self.generation,
                    "nodes": nodes,
                    "attrs": attrs,
                    "listed": listed,
                    "notice": self.notice,
                    "metadata": metadata,
                    "caches": sorted(exported_caches),
                    "batch_caches": sorted(exported_batches),
                    "cache_policy": self.cache_policy.value,
                    "max_cache_bytes": self.max_cache_bytes,
                    "mutable": effective_mutable,
                    "linked": linked_exports,
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

    def _export_batch_caches(self, full_path, staging_dir):
        exported = []
        for key, cache in self.batch_caches.items():
            if full_path and not key.startswith(full_path + "/"):
                continue
            if not cache._cache_sizes:
                continue
            folder = Path(staging_dir) / f"{key}.b2b.cache"
            folder.mkdir(parents=True, exist_ok=True)
            for index in cache._cache_sizes:
                (folder / f"{index}.chunk").write_bytes(cache.read_cached_chunk(index))
            exported.append(key)
        return exported

    def _export_linked_stores(self, full_path, staging_dir):
        exports = {}
        for mount, store in self.linked_stores.items():
            if full_path and mount != full_path and not mount.startswith(full_path + "/"):
                continue
            nested_file = os.path.join(staging_dir, f"linked-{len(exports)}.b2z")
            store.save(nested_file, include_cache=True, overwrite=True)
            manifest, offsets = RemoteStore._load_artifact_manifest(nested_file)
            prefix = f"__remote_links__/{hashlib.sha256(mount.encode()).hexdigest()}"
            with zipfile.ZipFile(nested_file) as nested_zip:
                for name in offsets:
                    if name == "embed.b2e":
                        continue
                    destination = os.path.join(staging_dir, prefix, name)
                    os.makedirs(os.path.dirname(destination), exist_ok=True)
                    with nested_zip.open(name) as src, open(destination, "wb") as dst:
                        shutil.copyfileobj(src, dst)
            os.unlink(nested_file)
            exports[mount] = {"manifest": manifest, "prefix": prefix}
        return exports

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
        if self.disk is not None:
            live_dir = os.path.abspath(str(self.disk.path))
            cache_root = os.path.abspath(str(self.disk.path.parent))
            if (
                dest_abs in (live_dir, cache_root)
                or dest_abs.startswith(live_dir + os.sep)
                or dest_abs.startswith(cache_root + os.sep)
            ):
                raise ValueError("destination cannot be inside live cache storage")
        if self.artifact_path is not None and dest_abs == os.path.abspath(self.artifact_path):
            raise ValueError("destination cannot be the source artifact")
        return dest_abs, dest_dir

    def _collect_export_nodes(self, group_full, include_cache):
        prefix = (group_full + "/") if group_full else ""
        for path, (kind, _) in self.nodes.items():
            if kind == "ctable" and (path == group_full or not prefix or path.startswith(prefix)):
                self.load_ctable_attrs(path)
        exported_source = {
            "urlpath": self.urlpath,
            "dataset": group_full,
            "kind": self.format,
        }
        fingerprint = storage_options_fingerprint(getattr(self, "storage_options", None))
        if fingerprint:
            exported_source["storage_options"] = fingerprint
        if prefix:
            exported_nodes = {
                k: (v[0], v[1] if v[0] in {"ctable", "remote_store", "unsupported"} else None)
                for k, v in self.nodes.items()
                if k == group_full or k.startswith(prefix)
            }
            exported_attrs = {k: v for k, v in self.attrs.items() if k == group_full or k.startswith(prefix)}
            exported_listed = {
                k: list(v) for k, v in self.listed.items() if k == group_full or k.startswith(prefix)
            }
            candidate_caches = [k for k in self.caches if k.startswith(prefix)] if include_cache else []
        else:
            exported_nodes = {
                k: (v[0], v[1] if v[0] in {"ctable", "remote_store", "unsupported"} else None)
                for k, v in self.nodes.items()
            }
            exported_attrs = dict(self.attrs)
            exported_listed = {k: list(v) for k, v in self.listed.items()}
            candidate_caches = list(self.caches) if include_cache else []
        return exported_source, exported_nodes, exported_attrs, exported_listed, candidate_caches

    def _copy_leaf_carrier(self, orig_key, proxy, staging_dir):
        leaf_filename = f"{orig_key}.b2nd"
        leaf_dst = os.path.join(staging_dir, leaf_filename)
        os.makedirs(os.path.dirname(leaf_dst), exist_ok=True)
        if self.artifact_offsets is not None:
            with zipfile.ZipFile(self.artifact_path, "r") as zf:
                zf.extract(leaf_filename, staging_dir)
        elif self.disk is not None and not getattr(self, "shared", False):
            src_file = self.disk.payload_path(self.generation, orig_key)
            if src_file.exists():
                shutil.copy2(src_file, leaf_dst)
        elif self.artifact_path is not None and os.path.isdir(self.artifact_path):
            src_file = os.path.join(self.artifact_path, leaf_filename)
            if os.path.exists(src_file):
                shutil.copy2(src_file, leaf_dst)
        else:
            # An in-memory proxy cache is an NDArray, so the leaf must be built the
            # same way a disk leaf is: a matching NDArray carrier, not a bare SChunk.
            # The ``remote-store`` identity and ``proxy-source`` metalayers are what
            # the mutable-reopen path validates when it adopts the extracted leaf.
            meta = {name: proxy._schunk_cache.meta[name] for name in proxy._schunk_cache.meta}
            meta.pop("b2nd", None)
            meta["remote-store"] = {"generation": self.generation, "dataset": orig_key}
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


class RemoteStore(RemoteObject):
    """Read-only remote B2Z, Zarr or HDF5 hierarchy.

    Also used for local hierarchies opened with ``blosc2.open(..., cache_dir=...)``.

    Discovery and returned array handles share source resources and traffic.
    MEMORY shares one bounded cache across all leaves; NONE retains no payload.
    DISK retains payload and discovery under an exclusively owned cache directory.
    Sources must be immutable until an explicit root refresh.

    ``keys()`` lists immediate children; ``get_info()`` inspects metadata without
    creating an array cache. Paths are relative to this group. Closing a handle
    leaves its previously returned arrays and group handles usable.

    ``hdf5_index`` accepts a native index dictionary or a local/remote JSON path
    for HDF5 sources. Supplying one skips HDF5 discovery.

    ``path`` selects a group within the source. ``dataset`` remains a supported
    alias; when both are supplied they must agree after stripping outer slashes.
    None leaves selection unspecified; an empty string or slash selects the root.
    Selector keywords cannot be combined with a selector embedded in the URL.
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
    def _cache_source(urlpath, dataset, source_format, storage_options):
        base_url, root, kind = parse_container_url(urlpath, dataset)
        local = not urlsplit(base_url).scheme or bool(os.path.splitdrive(base_url)[0])
        if local:
            base_url = os.path.abspath(base_url)
        source = {
            "urlpath": base_url,
            "dataset": (root or "").strip("/"),
            "kind": source_format or kind,
        }
        if local:
            source["local"] = True
        fingerprint = storage_options_fingerprint(storage_options)
        if fingerprint:
            source["storage_options"] = fingerprint
        return source

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

    @staticmethod
    def _validate_nested_storage_options(value):
        if value is not None and not callable(value) and not isinstance(value, dict):
            raise TypeError("nested_storage_options must be a mapping or callable")

    def __init__(
        self,
        urlpath,
        *,
        dataset=None,
        path=None,
        storage_options=None,
        cache_policy=CACHE_POLICY_DEFAULT,
        max_cache_bytes=CACHE_POLICY_DEFAULT,
        cache_dir=None,
        hdf5_index=None,
        _allow_array_root=False,
        _filesystem=None,
        _manifest=None,
        _source_validator=None,
        _manifest_validator=None,
        _max_nodes=None,
        _source_format=None,
        _hdf5_index=None,
        _hdf5_blob=None,
        _traffic=None,
        nested_storage_options=None,
        _b2z_blob=None,
        _allow_local_source=False,
    ):
        dataset = blosc2.core.resolve_dataset_path(dataset, path)
        if not isinstance(urlpath, (str, os.PathLike)):
            raise TypeError("RemoteStore requires a remote URL string")
        urlpath = os.fspath(urlpath)
        hdf5_index, _source_format = _resolve_hdf5_options(hdf5_index, _hdf5_index, _source_format)
        artifact = self._try_open_artifact(
            urlpath, dataset, storage_options, cache_policy, max_cache_bytes, cache_dir, _allow_array_root
        )
        if artifact is not None:
            self._attach(artifact._owner, artifact._path)
            return
        if dataset is not None and not isinstance(dataset, str):
            raise TypeError("dataset must be a string")
        self._validate_nested_storage_options(nested_storage_options)
        cache_policy, limit = self._validate_cache_config(cache_policy, max_cache_bytes, cache_dir)
        base_url, dataset, _ = parse_container_url(urlpath, dataset)
        urlpath = base_url
        local_source = bool(
            _allow_local_source
            and (not urlsplit(base_url).scheme or os.path.splitdrive(base_url)[0])
            and os.path.exists(base_url)
        )
        if not local_source:
            validate_persistable_url(base_url)
        else:
            base_url = os.path.abspath(base_url)
            urlpath = base_url
        local_hdf5 = local_source and _source_format == "hdf5"
        disk = None
        source_cache_path = source_cache_marker = None
        manifest = _manifest
        if cache_policy is blosc2.CachePolicy.DISK:
            from blosc2.remote_store_cache import StoreDiskCache

            source = self._cache_source(urlpath, dataset, _source_format, storage_options)
            disk = StoreDiskCache(cache_dir, source)
        try:
            manifest = disk.load() if disk is not None else manifest
            if disk is not None and source["kind"] == "hdf5" and not local_hdf5:
                from blosc2.hdf5_source import prepare_hdf5_source_cache

                source_cache_path, source_cache_marker, _hdf5_blob, hdf5_index, manifest = (
                    prepare_hdf5_source_cache(
                        base_url,
                        source["dataset"] or None,
                        cache_dir,
                        storage_options,
                        hdf5_index,
                        manifest,
                        _hdf5_blob,
                        explicit=_hdf5_index is None,
                    )
                )
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
                _source_format=_source_format,
                _hdf5_index=hdf5_index,
                _hdf5_blob=_hdf5_blob,
                _traffic=_traffic,
                _source_cache_dir=cache_dir if disk is not None else None,
                _b2z_blob=_b2z_blob,
                _local_source=local_source,
            )
            manifest = owner.restored_manifest
            owner.attach_hdf5_source_cache(source_cache_path, source_cache_marker)
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
            if kind == "ctable":
                raise ValueError("RemoteStore requires a group; use RemoteCTable for a table")
            raise ValueError("RemoteStore requires a group; use RemoteArray for an array")
        owner.cache_policy = cache_policy
        owner.max_cache_bytes = limit
        owner.cache_coordinator = CacheCoordinator(limit)
        owner.nested_storage_options = nested_storage_options
        try:
            owner.restore_caches(manifest)
            owner.save_manifest()
            owner.publish_source_cache()
            if disk is not None:
                disk.discard_old_generations(owner.generation)
        except BaseException:
            owner.close()
            raise
        self._attach(owner, "")

    @classmethod
    def _from_reference(cls, descriptor, **runtime):
        obj = object.__new__(cls)
        obj._deferred_reference = validate_remote_store_reference(descriptor)
        obj._deferred_runtime = runtime
        obj._deferred_closed = False
        obj._owner = None
        parent = runtime.get("_parent_owner")
        obj._reference_parent = parent
        obj._reference_generation = None if parent is None else parent.generation
        obj._reference_parent_finalizer = None
        if parent is not None:
            parent.acquire()
            obj._reference_parent_finalizer = weakref.finalize(obj, parent.release)
        return obj

    def _ensure_open(self):  # noqa: C901
        if getattr(self, "_deferred_closed", False):
            raise RuntimeError("RemoteStore handle is closed")
        parent = getattr(self, "_reference_parent", None)
        if parent is not None and self._reference_generation != parent.generation:
            raise RuntimeError("RemoteStore handle is stale; look it up again after refresh")
        descriptor = getattr(self, "_deferred_reference", None)
        if descriptor is None:
            return
        runtime = dict(getattr(self, "_deferred_runtime", {}))
        parent = runtime.pop("_parent_owner", None)
        namespace = runtime.pop("_cache_namespace", "")
        mount = runtime.pop("_mount", None)
        artifact = runtime.pop("_artifact", None)
        resolver = runtime.pop("_storage_options_resolver", None)
        if "storage_options" not in runtime:
            if callable(resolver):
                runtime["storage_options"] = resolver(dict(descriptor))
            elif resolver is not None:
                runtime["storage_options"] = resolver.get(descriptor["urlpath"])
            else:
                runtime["storage_options"] = None
        runtime.setdefault("cache_policy", blosc2.CachePolicy(descriptor["cache_policy"]))
        if runtime["cache_policy"] is not blosc2.CachePolicy.NONE:
            runtime.setdefault("max_cache_bytes", descriptor["max_cache_bytes"] or CACHE_POLICY_DEFAULT)
        else:
            runtime.pop("max_cache_bytes", None)
        requested_policy = runtime["cache_policy"]
        if artifact is not None and parent is not None:
            prefix = artifact["prefix"].rstrip("/") + "/"
            offsets = {
                name[len(prefix) :]: info
                for name, info in parent.artifact_offsets.items()
                if name.startswith(prefix)
            }
            opened_owner = type(self)._open_immutable_artifact(
                parent.artifact_path,
                artifact["manifest"],
                offsets,
                runtime["storage_options"],
                requested_policy,
                parent.max_cache_bytes,
                None,
                _traffic=parent.traffic,
            )
            opened = object.__new__(type(self))
            opened._attach(opened_owner, "")
        elif (
            requested_policy is blosc2.CachePolicy.DISK and "cache_dir" not in runtime and parent is not None
        ):
            identity = hashlib.sha256(
                f"{namespace}\0{descriptor['kind']}\0{descriptor['urlpath']}\0{descriptor['dataset']}".encode()
            ).hexdigest()
            if getattr(parent, "shared", False):
                opened = type(self).with_sparse_cache(
                    descriptor["urlpath"],
                    parent.disk.parent / "nested" / identity,
                    dataset=descriptor["dataset"] or None,
                    max_cache_bytes=parent.max_cache_bytes,
                    storage_options=runtime["storage_options"],
                    _traffic=parent.traffic,
                )
            elif parent.disk is not None:
                runtime["cache_dir"] = parent.disk.path / "nested" / identity
                opened = type(self)(
                    descriptor["urlpath"],
                    dataset=descriptor["dataset"] or None,
                    _source_format=descriptor["kind"],
                    _traffic=parent.traffic,
                    **runtime,
                )
            else:
                runtime["cache_policy"] = blosc2.CachePolicy.MEMORY
                opened = type(self)(
                    descriptor["urlpath"],
                    dataset=descriptor["dataset"] or None,
                    _source_format=descriptor["kind"],
                    _traffic=parent.traffic,
                    **runtime,
                )
        else:
            opened = type(self)(
                descriptor["urlpath"],
                dataset=descriptor["dataset"] or None,
                _source_format=descriptor["kind"],
                _traffic=None if parent is None else parent.traffic,
                **runtime,
            )
        owner, path = opened._owner, opened._path
        if parent is not None:
            owner.cache_policy = requested_policy
            owner.max_cache_bytes = parent.max_cache_bytes
            owner.cache_coordinator = parent.cache_coordinator
            owner.cache_namespace = namespace
            for key, cache in (*owner.caches.items(), *owner.batch_caches.items()):
                cache._cache_coordinator = parent.cache_coordinator
                cache._cache_key = f"{namespace}:{key}"
                parent.cache_coordinator.register(cache)
            owner.nested_storage_options = parent.nested_storage_options
            if mount is not None and mount not in parent.linked_stores:
                anchor = object.__new__(type(self))
                anchor._reference_parent = None
                anchor._reference_parent_finalizer = None
                anchor._deferred_reference = None
                anchor._deferred_closed = False
                anchor._attach(owner, path)
                parent.linked_stores[mount] = anchor
        self._attach(owner, path)
        opened.close()
        self._deferred_reference = None

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
        path=None,
        manifest=None,
        max_cache_bytes=CACHE_POLICY_DEFAULT,
        carrier=None,
        storage_options=None,
        _filesystem=None,
        _source_validator=None,
        _manifest_validator=None,
        _max_nodes=None,
        _traffic=None,
        _source_format=None,
        _hdf5_index=None,
    ):
        """Attach an immutable remote hierarchy to a cache shared across processes.

        All users of this private cache must use this constructor. Operations
        serialize per store, reload discovery, and enforce one aggregate payload
        allowance. The caller authorizes the supplied filesystem and manifest;
        no credentials or filesystem objects are persisted. Portable artifacts
        are exported with ``save`` rather than opened as mutable runtime storage.
        ``path`` and ``dataset`` select the group as in the ordinary constructor.
        The aggregate compressed-payload budget defaults to 256 MiB; pass
        ``max_cache_bytes=None`` for unlimited retention. For ordinary shared
        caching, prefer ``blosc2.open(url, cache_dir=..., shared_cache=True)``.
        """
        from blosc2.remote_store_cache import SharedStoreCache, SharedStoreOperation

        dataset = blosc2.core.resolve_dataset_path(dataset, path)
        limit = normalize_cache_limit(blosc2.CachePolicy.DISK, max_cache_bytes)
        base, root, kind = parse_container_url(urlpath, dataset)
        if _source_format is not None:
            kind = _source_format
        validate_persistable_url(base)
        source = {"urlpath": base, "dataset": (root or "").strip("/"), "kind": kind}
        fingerprint = storage_options_fingerprint(storage_options)
        if fingerprint:
            source["storage_options"] = fingerprint
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
                storage_options,
                dataset=root,
                manifest=current,
                persist_metadata=True,
                _filesystem=_filesystem,
                _source_validator=_source_validator,
                _manifest_validator=_manifest_validator,
                _max_nodes=_max_nodes,
                _traffic=_traffic,
                _source_format=kind,
                _hdf5_index=_hdf5_index,
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
        self._ensure_open()
        with self._owner.lock:
            _, full = self._resolve(path)
            if full not in self._owner.caches:
                return False, None
            with self[path] as array:
                return array.read_cached(item, nchunk=nchunk)

    def _resolve(self, path):
        self._ensure_open()
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

    def _linked_path(self, path):
        if not isinstance(path, str):
            raise TypeError("RemoteStore paths must be strings")
        relative = path.strip("/")
        self._owner._validate(relative)
        joined = "/".join(part for part in (self._path, relative) if part)
        parts = joined.split("/") if joined else []
        for end in range(len(parts), 0, -1):
            mount = "/".join(parts[:end])
            node = self._owner.nodes.get(mount)
            if node is not None and node[0] == "remote_store":
                return mount, "/".join(parts[end:]), node[1]
        return None

    def _linked_store(self, mount, descriptor, **overrides):
        existing = self._owner.linked_stores.get(mount)
        if existing is not None:
            store = object.__new__(type(self))
            store._reference_parent = self._owner
            store._reference_generation = self._owner.generation
            self._owner.acquire()
            store._reference_parent_finalizer = weakref.finalize(store, self._owner.release)
            store._deferred_reference = None
            store._deferred_closed = False
            store._attach(existing._owner, existing._path)
            return store
        resolver = self._owner.nested_storage_options
        runtime = {
            "_storage_options_resolver": resolver,
            "cache_policy": self._owner.cache_policy,
            "_parent_owner": self._owner,
            "_cache_namespace": f"{self._owner.generation}:{mount}",
            "_mount": mount,
        }
        if self._owner.cache_policy is not blosc2.CachePolicy.NONE:
            runtime["max_cache_bytes"] = self._owner.max_cache_bytes
        runtime.update(overrides)
        artifact = self._owner.linked_artifacts.get(mount)
        if artifact is not None:
            runtime["_artifact"] = artifact
        return type(self)._from_reference(descriptor, **runtime)

    def keys(self):
        """Return sorted immediate child names, using only discovery metadata."""
        self._ensure_open()
        with self._owner.lock:
            path, _ = self._resolve("")
            result = [key.rsplit("/", 1)[-1] for key in self._owner.list_children(path)]
            self._owner.save_manifest()
            return result

    def __iter__(self):
        return iter(self.keys())

    def __getitem__(self, path):
        self._ensure_open()
        with self._owner.lock:
            linked = self._linked_path(path)
            if linked is not None:
                _, suffix, descriptor = linked
                store = self._linked_store(linked[0], descriptor)
                return store if not suffix else store[suffix]
            relative, full = self._resolve(path)
            self._owner.save_manifest()
            kind, value = self._owner.nodes[full]
            if kind == "group":
                group = object.__new__(type(self))
                group._attach(self._owner, relative)
                return group
            if kind == "ctable":
                return blosc2.RemoteCTable._from_owner(self._owner, full)
            if kind == "unsupported":
                raise NotImplementedError(f"{path!r}: {value}")
            self._owner.open_source(relative)
            return self._owner.remote_array(full)

    def get_info(self, path=""):
        """Return node kind, known attributes and unsupported-node diagnostics."""
        self._ensure_open()
        with self._owner.lock:
            linked = self._linked_path(path)
            if linked is not None:
                mount, suffix, descriptor = linked
                if not suffix:
                    return RemoteNode(path.strip("/"), "remote_store", None, None)
                with self._linked_store(mount, descriptor) as store:
                    info = store.get_info(suffix)
                    return RemoteNode(path.strip("/"), info.kind, info.attrs, info.diagnostic)
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
        """Return the discovered node kind."""
        return self.get_info(path).kind

    @property
    def attrs(self):
        """Read-only group attributes, or None when discovery cannot decode them."""
        return self.get_info().attrs

    @property
    def source(self):
        """Credential-free source descriptor, including this group's full path."""
        descriptor = getattr(self, "_deferred_reference", None)
        if descriptor is not None:
            return {k: descriptor[k] for k in ("kind", "version", "urlpath", "dataset", "assume_immutable")}
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
        self._ensure_open()
        with self._owner.lock:
            self._resolve("")
            return self._owner.cache_coordinator.cache_bytes

    @property
    def metadata_bytes(self):
        """Encoded discovery manifest size, separate from retained payload."""
        self._ensure_open()
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
        self._ensure_open()
        with self._owner.lock:
            self._resolve("")
            if not getattr(self._owner, "is_mutable", True):
                raise ValueError(
                    "Cannot refresh an immutable RemoteStore artifact; use a new live session or a writable artifact"
                )
            if self._path:
                raise ValueError("refresh must be called on the root store handle")
            owner = self._owner
            replacement = owner.prepare_refresh("group")
            try:
                replacement.restoring = False
                replacement.save_manifest()
                replacement.publish_source_cache(refresh=True)
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
        if getattr(self, "_deferred_reference", None) is not None:
            self._deferred_closed = True
            finalizer = getattr(self, "_reference_parent_finalizer", None)
            if finalizer is not None:
                finalizer()
            return
        with self._owner.lock:
            self._finalizer()
        finalizer = getattr(self, "_reference_parent_finalizer", None)
        if finalizer is not None:
            finalizer()

    def _check_open(self):
        if getattr(self, "_deferred_reference", None) is not None:
            if self._deferred_closed:
                raise RuntimeError("RemoteStore handle is closed")
            return
        self._resolve("")

    def save(
        self,
        destination: str | os.PathLike,
        *,
        include_cache: bool = True,
        mutable: bool | None = None,
        overwrite: bool = False,
    ) -> str:
        """Export the current store or subtree to a portable .b2z reference archive."""
        self._ensure_open()
        with self._owner.lock:
            _, full = self._resolve("")
            return self._owner.save_selection(
                full, destination, include_cache=include_cache, mutable=mutable, overwrite=overwrite
            )

    def materialize(self, destination, *, overwrite=False):
        """Write this hierarchy and reachable store references as one local TreeStore."""
        from blosc2.store_materialize import materialize_store

        self._ensure_open()
        return materialize_store(self, destination, overwrite=overwrite)

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
    def _validate_artifact_manifest(manifest):  # noqa: C901
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
                or entry[0] not in {"group", "ndarray", "ctable", "remote_store", "unsupported"}
            ):
                raise ValueError("Invalid RemoteStore node")
            if entry[0] == "ctable":
                metadata = entry[1]
                if (
                    source.get("kind") not in {"b2z", "hdf5"}
                    or not isinstance(metadata, dict)
                    or metadata.get("kind") not in {"ctable", b"ctable"}
                    or not isinstance(metadata.get("schema"), (str, bytes))
                    or (
                        source.get("kind") == "hdf5"
                        and (
                            not isinstance(metadata.get("shape"), (list, tuple))
                            or len(metadata["shape"]) != 1
                            or not isinstance(metadata.get("dtype"), dict)
                        )
                    )
                ):
                    raise ValueError("Invalid RemoteStore CTable node")
            if entry[0] == "remote_store":
                validate_remote_store_reference(entry[1])
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
            if path.endswith(".source") and path not in nodes:
                continue
            if (
                path not in nodes
                or (
                    nodes[path][0] != "ndarray"
                    and not (source.get("kind") == "hdf5" and nodes[path][0] == "ctable")
                )
                or (source.get("kind") != "hdf5" and root and not path.startswith(root + "/"))
            ):
                raise ValueError("Invalid cached RemoteStore leaf")
        batches = manifest.get("batch_caches", [])
        if not isinstance(batches, list):
            raise ValueError("Invalid RemoteStore batch caches")
        for path in batches:
            RemoteDiscovery._validate(path)
            if (
                source.get("kind") != "b2z"
                or (root and not path.startswith(root + "/"))
                or not any(
                    kind == "ctable" and (not table or path.startswith(table + "/"))
                    for table, (kind, _) in nodes.items()
                )
            ):
                raise ValueError("Invalid cached RemoteStore batch")
        linked = manifest.get("linked", {})
        if not isinstance(linked, dict):
            raise ValueError("Invalid nested RemoteStore artifacts")
        for mount, entry in linked.items():
            RemoteDiscovery._validate(mount)
            if (
                mount not in nodes
                or nodes[mount][0] != "remote_store"
                or not isinstance(entry, dict)
                or not isinstance(entry.get("prefix"), str)
                or not isinstance(entry.get("manifest"), dict)
            ):
                raise ValueError("Invalid nested RemoteStore artifact")
            RemoteDiscovery._validate(entry["prefix"])
            if not entry["prefix"].startswith("__remote_links__/"):
                raise ValueError("Invalid nested RemoteStore artifact prefix")
            RemoteStore._validate_artifact_manifest(entry["manifest"])

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
        cls,
        urlpath,
        manifest,
        artifact_offsets,
        storage_options,
        cache_policy,
        limit,
        cache_dir,
        *,
        _traffic=None,
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
            _traffic=_traffic,
        )
        owner.disk = None
        owner.is_mutable = False
        owner.mutable = False
        owner.artifact_path = os.path.abspath(urlpath)
        owner.artifact_offsets = artifact_offsets
        owner.linked_artifacts = manifest.get("linked", {})
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
            if cache_policy is blosc2.CachePolicy.NONE and (
                manifest.get("caches") or manifest.get("batch_caches")
            ):
                raise ValueError(
                    "Cannot reopen a warm RemoteStore artifact with CachePolicy.NONE; "
                    "use a cold export or a cache policy that permits retained payload."
                )
        else:
            try:
                cache_policy = (
                    blosc2.CachePolicy.DISK
                    if manifest.get("mutable", False)
                    else blosc2.CachePolicy(manifest.get("cache_policy", "disk"))
                )
            except ValueError as exc:
                raise ValueError("RemoteStore artifact has an unsupported cache policy") from exc

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

        return cls._select_artifact(owner, kwargs.get("dataset"), kwargs.get("max_concurrency"))

    @classmethod
    def _select_artifact(cls, owner, req_dataset, max_concurrency):
        if owner.nodes[owner.root][0] == "ctable":
            if req_dataset:
                owner.close()
                raise ValueError("dataset cannot select below a RemoteCTable artifact root")
            result = blosc2.RemoteCTable._from_owner(owner, owner.root)
        else:
            result = cls.__new__(cls)
            result._attach(owner, "")
            if req_dataset:
                store = result
                try:
                    result = store[req_dataset]
                finally:
                    store.close()
        try:
            if max_concurrency is not None:
                if not isinstance(result, blosc2.RemoteCTable):
                    raise NotImplementedError(
                        "max_concurrency is only supported for table artifact selections"
                    )
                result.max_concurrency = max_concurrency
            return result
        except BaseException:
            result.close()
            raise
