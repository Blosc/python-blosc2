"""Browser presentation adapter for library remote discovery."""

import blosc2
from blosc2.remote_store import RemoteDiscovery


class RemoteHierarchy(RemoteDiscovery):
    """Retain the browser's current-leaf cache until its RemoteStore migration."""

    def __init__(self, urlpath, storage_options=None):
        self.leaf_path = None
        self.leaf = None
        super().__init__(urlpath, storage_options)

    def list_children(self, path):
        from blosc2.b2view.model import NodeInfo

        return [
            NodeInfo(
                path="/" + key[len(self.root) :].strip("/"),
                name=key.rsplit("/", 1)[-1],
                kind=self.nodes[key][0],
                has_children=self.nodes[key][0] == "group",
            )
            for key in super().list_children(path)
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
        if self.leaf_path != full:
            self.release_leaf()
            self.leaf = blosc2.Proxy(self.open_source(path), _max_cache_bytes=64 * 1024 * 1024)
            self.leaf_path = full
        return self.leaf

    def release_leaf(self):
        self.leaf = None
        self.leaf_path = None

    def close(self):
        self.release_leaf()
        super().close()
