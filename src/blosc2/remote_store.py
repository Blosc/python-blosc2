"""Read-only remote hierarchy discovery and public store handles."""

from __future__ import annotations

import json
import threading
import weakref
from dataclasses import dataclass
from pathlib import PurePosixPath
from urllib.parse import urlsplit, urlunsplit

import blosc2
from blosc2.core import parse_container_url
from blosc2.proxy import CacheCoordinator
from blosc2.proxy_source import Traffic
from blosc2.remote_array import (
    CACHE_POLICY_DEFAULT,
    RemoteMetadataMapping,
    normalize_cache_limit,
    validate_persistable_url,
)


@dataclass(frozen=True)
class RemoteNode:
    """Discovery metadata; unknown attributes are None and require opening the array."""

    path: str
    kind: str
    attrs: RemoteMetadataMapping | None
    diagnostic: str | None = None


class RemoteDiscovery:
    """Shared metadata and source resources, independent of browser presentation."""

    def __init__(self, urlpath, storage_options=None, *, dataset=None):
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
        self._users = 0
        self._closed = False
        # ponytail: serialize store operations; finer locks if multi-leaf throughput matters.
        self.lock = threading.RLock()
        try:
            if self.format == "b2z":
                self._open_b2z()
            elif self.format == "hdf5":
                self._open_hdf5()
            elif self.format == "zarr":
                self._open_zarr()
            else:
                raise ValueError("Unsupported remote hierarchy format")
            if self.root not in self.nodes:
                raise KeyError("Requested node does not exist in the container")
            self.is_tree = self.nodes[self.root][0] == "group"
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _validate(path):
        if path and (
            any(p in {"", ".", ".."} for p in path.split("/")) or any(c in path for c in "\\\0\n\r\t")
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

        self.archive = B2ZArchive(self.urlpath, storage_options=self.storage_options, _traffic=self.traffic)
        members = {}
        for info in self.archive.members:
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

    def _open_hdf5(self):
        from blosc2.hdf5_source import scan_hdf5_refs

        unsupported = {}
        self.refs = scan_hdf5_refs(
            self.urlpath, self.storage_options, unsupported=unsupported, traffic=self.traffic
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
        import fsspec
        import zarr

        from blosc2.zarr_source import counting_store

        self.zstore = counting_store(
            zarr,
            zarr.storage.FsspecStore.from_mapper(
                fsspec.get_mapper(self.urlpath, **self.storage_options), read_only=True
            ),
            self.traffic,
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
        if self.format == "zarr" and (full not in self.nodes or self.nodes[full] == ("group", None)):
            import zarr

            try:
                node = zarr.open(store=self.zstore, path=full, mode="r", use_consolidated=False)
            except FileNotFoundError as exc:
                raise KeyError(path) from exc
            except (ValueError, TypeError, NotImplementedError) as exc:
                self._add(full, "unsupported", f"{type(exc).__name__}: {exc}")
            else:
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
        return self.listed[full].copy()

    def open_source(self, path):
        full = self._path(path)
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
            )
        else:
            from blosc2.zarr_source import ZarrNDSource

            url = urlsplit(self.urlpath)
            leaf_url = urlunsplit(url._replace(path=url.path.rstrip("/") + "/" + full))
            source = ZarrNDSource(self.zstore, _path=full, _urlpath=leaf_url, _traffic=self.traffic)
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

    def get_cache(self, source):
        key = next(path for path, value in self.sources.items() if value is source)
        if key not in self.caches:
            self.caches[key] = blosc2.Proxy(
                source, _refresh_source=False, _cache_coordinator=self.cache_coordinator, _cache_key=key
            )
        return self.caches[key]

    def close(self):
        if self._closed:
            return
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


class RemoteStore:
    """Read-only remote B2Z, Zarr or HDF5 hierarchy.

    Discovery and returned array handles share source resources and traffic.
    MEMORY shares one bounded cache across all leaves; NONE retains no payload.
    DISK retention will follow. Sources must be immutable.

    ``keys()`` lists immediate children; ``get_info()`` inspects metadata without
    creating an array cache. Paths are relative to this group. Closing a handle
    leaves its previously returned arrays and group handles usable.
    """

    def __init__(
        self,
        urlpath,
        *,
        dataset=None,
        storage_options=None,
        cache_policy=blosc2.CachePolicy.MEMORY,
        max_cache_bytes=CACHE_POLICY_DEFAULT,
    ):
        if not isinstance(urlpath, str):
            raise TypeError("RemoteStore requires a remote URL string")
        if dataset is not None and not isinstance(dataset, str):
            raise TypeError("dataset must be a string")
        if not isinstance(cache_policy, blosc2.CachePolicy):
            raise TypeError("cache_policy must be a blosc2.CachePolicy instance")
        if cache_policy is blosc2.CachePolicy.DISK:
            raise NotImplementedError("RemoteStore DISK caching is not implemented")
        limit = normalize_cache_limit(cache_policy, max_cache_bytes)
        base_url, _, _ = parse_container_url(urlpath, dataset)
        validate_persistable_url(base_url)
        owner = RemoteDiscovery(urlpath, storage_options, dataset=dataset)
        if not owner.is_tree:
            kind, diagnostic = owner.nodes[owner.root]
            owner.close()
            if kind == "unsupported":
                raise NotImplementedError(str(diagnostic))
            raise ValueError("RemoteStore requires a group; use RemoteArray for an array")
        owner.cache_policy = cache_policy
        owner.max_cache_bytes = limit
        owner.cache_coordinator = CacheCoordinator(limit)
        self._attach(owner, "")

    def _attach(self, owner, path):
        owner.acquire()
        self._owner = owner
        self._path = path
        self._finalizer = weakref.finalize(self, owner.release)

    def _resolve(self, path):
        if not self._finalizer.alive:
            raise RuntimeError("RemoteStore handle is closed")
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
            return [key.rsplit("/", 1)[-1] for key in self._owner.list_children(path)]

    def __iter__(self):
        return iter(self.keys())

    def __getitem__(self, path):
        with self._owner.lock:
            relative, full = self._resolve(path)
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
                    if self.cache_policy is blosc2.CachePolicy.MEMORY
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

    def close(self):
        """Release this handle; the last dependent handle closes shared resources."""
        with self._owner.lock:
            self._finalizer()

    def __enter__(self):
        self._resolve("")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
