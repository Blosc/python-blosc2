"""Internal, read-only remote hierarchy discovery for b2view."""

from __future__ import annotations

import json
from pathlib import PurePosixPath

import blosc2
from blosc2.core import parse_container_url
from blosc2.proxy_source import Traffic


class RemoteHierarchy:
    """Metadata for one immutable container, retaining only the current leaf cache.

    Paths exposed to the browser are relative to the requested group. No child
    URL is constructed: sources receive the container and dataset separately.
    """

    def __init__(self, urlpath, storage_options=None):
        self.urlpath, dataset, self.format = parse_container_url(urlpath)
        self.root = (dataset or "").strip("/")
        self.storage_options = storage_options or {}
        self.traffic = Traffic()
        self.nodes = {}
        self.attrs = {}
        self.listed = {}
        self.notice = None
        self.archive = None
        self.zstore = None
        self.leaf_path = None
        self.leaf = None
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
            if name == "embed.b2e" or info.is_dir():
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
                else "B2Z previews require unencrypted ZIP_STORED external NDArray members",
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
                self._add(path, "unsupported", "Embedded B2Z previews are unavailable")

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
            self.nodes[""] = ("unsupported", "CTable previews are unavailable for remote B2Z hierarchies")
            return
        self._add("", "group")
        for root in sorted(roots, key=len):
            if not any(root.startswith(other + "/") for other in roots if other != root):
                self._add(root, "unsupported", "CTable previews are unavailable for remote B2Z hierarchies")
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
        relative = path.strip("/")
        self._validate(relative)
        return "/".join(p for p in (self.root, relative) if p)

    def kind(self, path):
        return self.nodes[self._path(path)][0]

    def list_children(self, path):
        from blosc2.b2view.model import NodeInfo

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
                            self.nodes[key] = ("unsupported", f"{type(exc).__name__}: {exc}")
                except Exception as exc:
                    raise OSError(
                        "Cannot list Zarr group; check LIST permission and backend support"
                    ) from exc
                for name, node in children:
                    key = "/".join(p for p in (full, name) if p)
                    self.nodes[key] = ("group" if isinstance(node, zarr.Group) else "ndarray", node)
                    self.attrs[key] = dict(node.attrs)
            self.listed[full] = sorted(
                key for key in self.nodes if key != full and key.rpartition("/")[0] == full
            )
        return [
            NodeInfo(
                path="/" + key[len(self.root) :].strip("/"),
                name=key.rsplit("/", 1)[-1],
                kind=self.nodes[key][0],
                has_children=self.nodes[key][0] == "group",
            )
            for key in self.listed[full]
        ]

    def get_info(self, path):
        from blosc2.b2view.model import ObjectInfo, StoreBrowser, object_metadata

        full = self._path(path)
        kind, value = self.nodes[full]
        metadata = {"type": f"{self.format.upper()} {kind}"}
        attrs = self.attrs.get(full, {} if kind == "group" else None)
        if kind == "group":
            self.release_leaf()
            if full in self.listed:
                metadata["children"] = len(self.listed[full])
            if self.notice:
                metadata["notice"] = self.notice
        elif kind == "unsupported":
            self.release_leaf()
            metadata["preview"] = value if isinstance(value, str) else "Unsupported remote B2Z member"
        else:
            obj = self.open_leaf(path)
            metadata.update(object_metadata(obj))
            attrs_obj = getattr(obj.src, "attrs", getattr(obj.src, "vlmeta", None))
            attrs = StoreBrowser._attrs_dict(attrs_obj)
        return ObjectInfo(path, kind, metadata, attrs)

    def open_leaf(self, path):
        full = self._path(path)
        if self.leaf_path == full:
            return self.leaf
        self.release_leaf()
        kind, value = self.nodes[full]
        if kind != "ndarray":
            raise NotImplementedError(
                value if isinstance(value, str) else "Preview unavailable for this node"
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

            source = ZarrNDSource(self.zstore, _path=full, _urlpath=self.urlpath, _traffic=self.traffic)
        self.leaf = blosc2.Proxy(source, _max_cache_bytes=64 * 1024 * 1024)
        self.leaf_path = full
        return self.leaf

    def release_leaf(self):
        self.leaf = None
        self.leaf_path = None

    def close(self):
        self.release_leaf()
        if self.archive is not None:
            self.archive.close()
        if self.zstore is not None:
            self.zstore.close()
