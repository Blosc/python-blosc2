"""Read-only browsing helpers for b2view."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

import numpy as np

import blosc2


@dataclass(frozen=True)
class NodeInfo:
    """Lightweight description of one TreeStore child."""

    path: str
    name: str
    kind: str
    has_children: bool


@dataclass(frozen=True)
class ObjectInfo:
    """Metadata for a TreeStore object or group."""

    path: str
    kind: str
    metadata: dict[str, Any]
    user_attrs: dict[str, Any] | None = None


class StoreBrowser:
    """Small, read-only adapter used by the b2view UI.

    The adapter intentionally exposes a narrow API so the TUI does not depend
    on TreeStore internals.  It accepts either a TreeStore hierarchy or a
    single top-level Blosc2 object (for example a standalone CTable).  It
    performs bounded previews only; callers must explicitly request pages or
    slices.
    """

    def __init__(self, urlpath: str):
        self.urlpath = urlpath
        self.store = blosc2.open(urlpath, mode="r")
        self.is_tree = isinstance(self.store, blosc2.TreeStore)

    def close(self) -> None:
        close = getattr(self.store, "close", None)
        if close is not None:
            close()

    def __enter__(self) -> StoreBrowser:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @staticmethod
    def normalize_path(path: str) -> str:
        """Return an absolute TreeStore path."""
        if not path:
            return "/"
        if not path.startswith("/"):
            path = "/" + path
        normalized = str(PurePosixPath(path))
        return "/" if normalized == "." else normalized

    def list_children(self, path: str = "/") -> list[NodeInfo]:
        """Return direct children for *path*."""
        path = self.normalize_path(path)
        if not self.is_tree:
            self._check_root_path(path)
            return []

        children = []
        for child_path in self.store.get_children(path):
            descendants = self.store.get_descendants(child_path)
            has_children = bool(descendants)
            kind = "group" if has_children else self.kind(child_path)
            children.append(
                NodeInfo(
                    path=child_path,
                    name=child_path.rsplit("/", 1)[-1] or "/",
                    kind=kind,
                    has_children=has_children,
                )
            )
        return children

    def kind(self, path: str) -> str:
        """Classify a browser path."""
        path = self.normalize_path(path)
        if not self.is_tree:
            self._check_root_path(path)
            return object_kind(self.store)
        if path == "/" or self.store.get_descendants(path):
            return "group"
        obj = self.store[path]
        return object_kind(obj)

    def get_info(self, path: str) -> ObjectInfo:
        """Return metadata for *path*."""
        path = self.normalize_path(path)
        kind = self.kind(path)
        if kind == "group":
            metadata: dict[str, Any] = {
                "type": "TreeStore group",
                "children": len(self.store.get_children(path)),
                "descendants": len(self.store.get_descendants(path)),
            }
            user_attrs = self._vlmeta_dict(self.store.vlmeta) if path == "/" else None
            return ObjectInfo(path=path, kind=kind, metadata=metadata, user_attrs=user_attrs)

        obj = self._get_object(path)
        metadata = object_metadata(obj)
        metadata.setdefault("type", type(obj).__name__)
        user_attrs = self._vlmeta_dict(getattr(obj, "vlmeta", None))
        return ObjectInfo(path=path, kind=kind, metadata=metadata, user_attrs=user_attrs)

    def preview(
        self,
        path: str,
        *,
        start: int = 0,
        stop: int | None = None,
        columns: list[str] | None = None,
        slices: tuple[Any, ...] | None = None,
        max_rows: int = 20,
        max_cols: int = 10,
        col_start: int = 0,
    ) -> Any:
        """Return a bounded data preview for *path*."""
        path = self.normalize_path(path)
        obj = self._get_object(path)
        kind = object_kind(obj)
        if kind in {"ndarray", "c2array"}:
            shape = tuple(getattr(obj, "shape", ()) or ())
            if slices is None and len(shape) == 2:
                stop = min(start + max_rows, shape[0]) if stop is None else stop
                return preview_array_2d(obj, start=start, stop=stop, col_start=col_start, max_cols=max_cols)
            return preview_array(obj, slices=slices, max_rows=max_rows, max_cols=max_cols)
        if kind == "ctable":
            stop = min(start + max_rows, len(obj)) if stop is None else stop
            return preview_ctable(obj, start=start, stop=stop, columns=columns, max_cols=max_cols)
        if kind == "schunk":
            return {"message": "SChunk byte preview is not implemented yet."}
        return {"message": f"Preview is not supported for {kind!r} objects."}

    def _get_object(self, path: str) -> Any:
        """Return the object represented by *path*."""
        path = self.normalize_path(path)
        if self.is_tree:
            return self.store[path]
        self._check_root_path(path)
        return self.store

    @staticmethod
    def _check_root_path(path: str) -> None:
        if path != "/":
            raise KeyError(f"Standalone objects only expose the root path '/', got {path!r}")

    @staticmethod
    def _vlmeta_dict(vlmeta) -> dict[str, Any] | None:
        if vlmeta is None:
            return None
        try:
            data = vlmeta[:]
        except Exception:
            try:
                data = {name: vlmeta[name] for name in vlmeta}
            except Exception:
                return None
        return data or None


def object_kind(obj: Any) -> str:
    """Return a stable b2view kind string for *obj*."""
    if isinstance(obj, blosc2.TreeStore):
        return "group"
    if isinstance(obj, blosc2.NDArray):
        return "ndarray"
    if isinstance(obj, blosc2.CTable):
        return "ctable"
    if hasattr(blosc2, "C2Array") and isinstance(obj, blosc2.C2Array):
        return "c2array"
    if isinstance(obj, blosc2.SChunk):
        return "schunk"
    return "unknown"


def object_metadata(obj: Any) -> dict[str, Any]:
    """Extract lightweight metadata from a supported object."""
    kind = object_kind(obj)
    if kind in {"ndarray", "c2array"}:
        return {
            "shape": getattr(obj, "shape", None),
            "ndim": len(getattr(obj, "shape", ()) or ()),
            "dtype": str(getattr(obj, "dtype", None)),
            "chunks": getattr(obj, "chunks", None),
            "blocks": getattr(obj, "blocks", None),
            "nbytes": getattr(obj, "nbytes", None),
            "cbytes": getattr(obj, "cbytes", None),
        }
    if kind == "ctable":
        try:
            return dict(obj.info_items)
        except Exception:
            return {
                "rows": getattr(obj, "nrows", len(obj)),
                "columns": getattr(obj, "ncols", len(getattr(obj, "col_names", []))),
                "schema": {
                    name: str(getattr(obj[name], "dtype", None)) for name in getattr(obj, "col_names", [])
                },
            }
    if kind == "schunk":
        return {
            "chunks": getattr(obj, "nchunks", None),
            "nbytes": getattr(obj, "nbytes", None),
            "cbytes": getattr(obj, "cbytes", None),
        }
    return {"repr": repr(obj)}


def preview_array_2d(
    obj: Any, *, start: int = 0, stop: int = 20, col_start: int = 0, max_cols: int = 10
) -> dict[str, Any]:
    """Return a bounded row/column preview for a 2-D array."""
    shape = tuple(getattr(obj, "shape", ()) or ())
    if len(shape) != 2:
        raise ValueError(f"Expected a 2-D array, got shape {shape!r}")
    nrows, ncols = shape
    start = max(0, min(start, nrows))
    stop = min(max(start, stop), nrows)
    col_start = max(0, min(col_start, ncols))
    col_stop = min(col_start + max_cols, ncols)
    columns = [str(i) for i in range(col_start, col_stop)]
    values = np.asarray(obj[(slice(start, stop), slice(col_start, col_stop))])
    data = {str(col): values[:, i] for i, col in enumerate(range(col_start, col_stop))}
    return {
        "start": start,
        "stop": stop,
        "nrows": nrows,
        "columns": columns,
        "hidden_columns": max(0, ncols - (col_stop - col_start)),
        "data": data,
        "source_kind": "ndarray2d",
        "shape": shape,
        "col_start": col_start,
        "col_stop": col_stop,
        "ncols": ncols,
    }


def preview_array(
    obj: Any, *, slices: tuple[Any, ...] | None = None, max_rows: int = 20, max_cols: int = 10
):
    """Return a small NumPy preview from an NDArray/C2Array-like object."""
    shape = tuple(getattr(obj, "shape", ()) or ())
    if slices is None:
        if len(shape) == 0:
            slices = ()
        elif len(shape) == 1:
            slices = (slice(0, min(shape[0], max_rows)),)
        elif len(shape) == 2:
            slices = (slice(0, min(shape[0], max_rows)), slice(0, min(shape[1], max_cols)))
        else:
            leading = tuple(0 for _ in shape[:-2])
            slices = leading + (
                slice(0, min(shape[-2], max_rows)),
                slice(0, min(shape[-1], max_cols)),
            )
    return np.asarray(obj[slices])


def preview_ctable(
    obj: Any,
    *,
    start: int = 0,
    stop: int = 20,
    columns: list[str] | None = None,
    max_cols: int = 10,
    include_expensive: bool = False,
) -> dict[str, Any]:
    """Return a bounded column-oriented preview from a CTable.

    Complex nested/list/object columns may require one variable-length block
    read per row.  By default, keep table navigation responsive by showing a
    placeholder for those columns instead of decoding them eagerly.
    """
    all_columns = list(getattr(obj, "col_names", []))
    visible_columns = all_columns if columns is None else [name for name in columns if name in all_columns]
    hidden_columns = max(0, len(visible_columns) - max_cols)
    visible_columns = visible_columns[:max_cols]
    start = max(0, start)
    stop = min(max(start, stop), len(obj))
    data = {}
    skipped_columns = {}
    nrows = stop - start
    for name in visible_columns:
        if not include_expensive and is_expensive_ctable_column(obj, name):
            label = ctable_column_label(obj, name)
            placeholder = f"<{label}; skipped>"
            data[name] = np.full(nrows, placeholder, dtype=object)
            skipped_columns[name] = label
        else:
            data[name] = safe_asarray(obj[name][start:stop])
    return {
        "start": start,
        "stop": stop,
        "nrows": len(obj),
        "columns": visible_columns,
        "hidden_columns": hidden_columns,
        "skipped_columns": skipped_columns,
        "data": data,
    }


def is_expensive_ctable_column(obj: Any, name: str) -> bool:
    """Return whether previewing a CTable column is likely row-by-row expensive."""
    try:
        schema = obj.schema_dict()
    except Exception:
        return False
    for column in schema.get("columns", []):
        if column.get("name") != name:
            continue
        return column.get("kind") in {"list", "struct", "object", "ndarray"}
    return False


def ctable_column_label(obj: Any, name: str) -> str:
    """Return a compact schema label for *name*."""
    try:
        schema = dict(obj.info_items).get("schema", {})
        label = schema.get(name)
        if label is not None:
            return str(label)
    except Exception:
        pass
    try:
        for column in obj.schema_dict().get("columns", []):
            if column.get("name") == name:
                return str(column.get("kind", "complex"))
    except Exception:
        pass
    return "complex"


def safe_asarray(values: Any) -> np.ndarray:
    """Convert preview values to an array, preserving ragged/nested values.

    NumPy 2 raises for ragged nested sequences unless ``dtype=object`` is
    requested explicitly.  CTable columns can legitimately contain list/struct
    values, so previews must keep those as object cells instead of failing.
    """
    try:
        return np.asarray(values)
    except ValueError:
        return np.asarray(values, dtype=object)
