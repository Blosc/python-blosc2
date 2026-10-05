"""Lazy, read-only directory roots with dataset containers as mounted subtrees."""

import os
from pathlib import Path, PurePosixPath

from blosc2.b2view.ordinary_file import NATIVE_SUFFIXES, local_path, ordinary_filesystem
from blosc2.core import find_url_separator, is_fsspec_url

CONTAINER_SUFFIXES = {".b2d", ".b2z", ".h5", ".hdf5", ".zarr"}


class DirectoryStore:
    """List only immediate children; open dataset mounts only on selection/expansion."""

    def __init__(self, source, *, storage_options=None, **options):
        self.source = source
        self.storage_options = storage_options
        self.options = options
        self.fs = None
        if is_fsspec_url(source):
            self.fs, self.root = ordinary_filesystem(source, storage_options)
        else:
            self.root = local_path(source).resolve()
        self._browsers = {}
        self._counts = {}

    def _resolve(self, path):
        """Return (physical path, mount browser or None, internal mount path)."""
        from blosc2.b2view.model import StoreBrowser

        parts = PurePosixPath(path).parts
        parts = [part for part in parts if part != "/"]
        if ".." in parts:
            raise ValueError("Directory navigation cannot escape the root")
        physical = self.root
        for index, part in enumerate(parts):
            physical = physical / part if self.fs is None else physical.rstrip("/") + "/" + part
            if self.fs is None and physical.is_symlink():
                raise ValueError("Directory symlinks are not followed")
            suffix = PurePosixPath(part).suffix.lower()
            is_dir = physical.is_dir() if self.fs is None else self.fs.isdir(physical)
            if suffix in NATIVE_SUFFIXES | CONTAINER_SUFFIXES or not is_dir:
                key = "/" + "/".join(parts[: index + 1])
                if key not in self._browsers:
                    source = os.fspath(physical) if self.fs is None else self.fs.unstrip_protocol(physical)
                    separator = find_url_separator(self.source)
                    if self.fs is not None and separator != -1:
                        source += self.source[separator:]
                    options = dict(self.options)
                    if options.get("cache_dir") is not None:
                        import hashlib

                        # Mounted datasets need independent snapshot/lock roots.
                        cache_key = hashlib.sha256(source.encode()).hexdigest()
                        options["cache_dir"] = str(Path(options["cache_dir"]) / cache_key)
                    self._browsers[key] = StoreBrowser(
                        source, storage_options=self.storage_options, **options
                    )
                return physical, self._browsers[key], "/" + "/".join(parts[index + 1 :])
        return physical, None, "/"

    def list_children(self, path):
        from blosc2.b2view.model import NodeInfo

        physical, browser, inner = self._resolve(path)
        prefix = path.rstrip("/")
        if browser is not None:
            return [
                NodeInfo(prefix + "/" + node.name, node.name, node.kind, node.has_children)
                for node in browser.list_children(inner)
            ]
        if self.fs is None:
            entries = [(entry.name, entry.is_dir(), entry.is_symlink()) for entry in physical.iterdir()]
        else:
            entries = [
                (PurePosixPath(entry["name"].rstrip("/")).name, entry["type"] == "directory", False)
                for entry in self.fs.ls(physical, detail=True)
            ]
        nodes = []
        for name, is_dir, symlink in sorted(entries):
            suffix = PurePosixPath(name).suffix.lower()
            mount = suffix in CONTAINER_SUFFIXES
            directory = is_dir and suffix not in NATIVE_SUFFIXES
            kind = (
                "unsupported"
                if symlink
                else "group"
                if directory or mount
                else {".b2nd": "ndarray", ".parquet": "ctable", ".b2": "schunk", ".b2frame": "schunk"}.get(
                    suffix, "unknown" if suffix in NATIVE_SUFFIXES else "file"
                )
            )
            nodes.append(NodeInfo(prefix + "/" + name, name, kind, not symlink and (directory or mount)))
        self._counts[path] = len(nodes)
        return nodes

    def get_info(self, path):
        from blosc2.b2view.model import ObjectInfo

        physical, browser, inner = self._resolve(path)
        if browser is not None:
            info = browser.get_info(inner)
            return ObjectInfo(path, info.kind, info.metadata, info.user_attrs, path)
        metadata = {"type": "Local directory" if self.fs is None else "FSSPEC directory"}
        if path in self._counts:
            metadata["children"] = self._counts[path]
        return ObjectInfo(path, "group", metadata, display_path=path)

    def __getitem__(self, path):
        _, browser, inner = self._resolve(path)
        if browser is None:
            raise TypeError("Directory groups have no array/file payload")
        return browser._get_object(inner)

    def close(self):
        error = None
        try:
            for browser in self._browsers.values():
                try:
                    browser.close()
                except Exception as exc:
                    error = exc
        finally:
            self._browsers.clear()
        if error is not None:
            raise error


def is_directory(source, storage_options=None):
    """Check only the root, not its descendants."""
    if is_fsspec_url(source):
        fs, path = ordinary_filesystem(source, storage_options)
        return fs.isdir(path)
    return local_path(source).is_dir()
