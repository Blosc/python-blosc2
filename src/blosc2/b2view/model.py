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


@dataclass
class DataSliceLayout:
    """Describes the fixed/navigable state for slicing an N-D array into a 2-D table view.

    At most 2 dimensions can be navigable (shown as table rows/columns).
    All other dimensions must be fixed at a specific index value.
    """

    shape: tuple[int, ...]
    fixed_values: dict[int, int]  # dim_index → fixed index value
    navigable_dims: list[int]  # sorted list of up to 2 navigable dim indices

    # Current scroll positions for navigable dims
    # (index 0 → rows, index 1 → cols if present)
    row_start: int = 0
    row_stop: int = 0
    col_start: int = 0
    col_stop: int = 0

    @classmethod
    def from_shape(cls, shape: tuple[int, ...]) -> DataSliceLayout:
        """Create a default layout: leading dims fixed at 0, last up-to-2 dims navigable."""
        ndim = len(shape)
        if ndim <= 2:
            navigable = list(range(ndim))
            fixed: dict[int, int] = {}
        else:
            navigable = list(range(ndim - 2, ndim))
            fixed = dict.fromkeys(range(ndim - 2), 0)
        return cls(
            shape=shape,
            fixed_values=fixed,
            navigable_dims=navigable,
        )

    def make_slices(self, max_rows: int = 20, max_cols: int = 10) -> tuple[int | slice, ...]:
        """Build the tuple of index expressions for slicing into the array.

        Uses *max_rows* and *max_cols* to size the navigable dimensions when
        ``row_stop <= row_start`` (i.e. no explicit stop was set).
        """
        slices: list[int | slice] = []
        for i in range(len(self.shape)):
            if i in self.fixed_values:
                slices.append(self.fixed_values[i])
            elif self.navigable_dims and i == self.navigable_dims[0]:
                start = max(0, min(self.row_start, self.shape[i]))
                if self.row_stop > self.row_start:
                    stop = min(self.row_stop, self.shape[i])
                else:
                    stop = min(start + max_rows, self.shape[i])
                slices.append(slice(start, stop))
            elif len(self.navigable_dims) > 1 and i == self.navigable_dims[1]:
                start = max(0, min(self.col_start, self.shape[i]))
                if self.col_stop > self.col_start:
                    stop = min(self.col_stop, self.shape[i])
                else:
                    stop = min(start + max_cols, self.shape[i])
                slices.append(slice(start, stop))
            else:
                slices.append(slice(0, self.shape[i]))
        return tuple(slices)

    def copy_with(
        self,
        *,
        fixed_values: dict[int, int] | None = None,
        navigable_dims: list[int] | None = None,
        row_start: int | None = None,
        row_stop: int | None = None,
        col_start: int | None = None,
        col_stop: int | None = None,
    ) -> DataSliceLayout:
        """Return a new layout with specified fields overridden."""
        return DataSliceLayout(
            shape=self.shape,
            fixed_values=self.fixed_values if fixed_values is None else fixed_values,
            navigable_dims=list(self.navigable_dims) if navigable_dims is None else navigable_dims,
            row_start=self.row_start if row_start is None else row_start,
            row_stop=self.row_stop if row_stop is None else row_stop,
            col_start=self.col_start if col_start is None else col_start,
            col_stop=self.col_stop if col_stop is None else col_stop,
        )

    def total_for_dim(self, dim: int) -> int:
        """Return the total size of *dim*."""
        if 0 <= dim < len(self.shape):
            return self.shape[dim]
        return 0


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
            user_attrs = self._vlmeta_dict(self.store.vlmeta)
            return ObjectInfo(path=path, kind=kind, metadata=metadata, user_attrs=user_attrs)

        obj = self._get_object(path)
        metadata = object_metadata(obj)
        metadata.setdefault("type", type(obj).__name__)
        user_attrs = self._vlmeta_dict(getattr(obj, "vlmeta", None))
        if user_attrs is None and self.is_tree:
            user_attrs = self._vlmeta_dict(self.store.vlmeta)
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
        slice_indices: list[int] | None = None,
        layout: DataSliceLayout | None = None,
    ) -> Any:
        """Return a bounded data preview for *path*.

        For N-D arrays (N >= 3) a *layout* may be provided instead of the
        legacy *slice_indices*, *start*/*stop*, *col_start* parameters.
        """
        path = self.normalize_path(path)
        obj = self._get_object(path)
        kind = object_kind(obj)
        if kind in {"ndarray", "c2array"}:
            shape = tuple(getattr(obj, "shape", ()) or ())
            if slices is None:
                if layout is not None:
                    return preview_array_from_layout(
                        obj, layout=layout, max_rows=max_rows, max_cols=max_cols
                    )
                if len(shape) >= 3:
                    return preview_array_nd_slice(
                        obj,
                        slice_indices=slice_indices,
                        start=start,
                        stop=stop,
                        col_start=col_start,
                        max_cols=max_cols,
                    )
                if len(shape) == 2:
                    stop = min(start + max_rows, shape[0]) if stop is None else stop
                    return preview_array_2d(
                        obj, start=start, stop=stop, col_start=col_start, max_cols=max_cols
                    )
                if len(shape) == 1:
                    stop = min(start + max_rows, shape[0]) if stop is None else stop
                    return preview_array_1d(obj, start=start, stop=stop)
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

    _INTERNAL_VLMETA_KEYS = frozenset(
        {
            "kind",
            "version",
            "schema",
            "n_rows",
            "value_epoch",
            "computed_columns",
            "materialized_columns",
        }
    )

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
        if data is None:
            return None
        # Filter out internal blosc2 metadata keys (schema, version, etc.)
        return {k: v for k, v in data.items() if k not in StoreBrowser._INTERNAL_VLMETA_KEYS}


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


def preview_array_from_layout(
    obj: Any,
    *,
    layout: DataSliceLayout,
    max_rows: int = 20,
    max_cols: int = 10,
) -> dict[str, Any]:
    """Return a bounded preview for an N-D array using a *layout*.

    The layout describes which dimensions are fixed (slider) vs navigable
    (table rows/columns).  At most 2 navigable dimensions are allowed.
    """
    shape = tuple(getattr(obj, "shape", ()) or ())
    if len(shape) != len(layout.shape):
        raise ValueError(f"Layout shape {layout.shape} does not match object shape {shape}")
    ndim = len(shape)
    navigable = layout.navigable_dims

    # Determine row and col navigable dims
    row_dim = navigable[0] if len(navigable) >= 1 else None
    col_dim = navigable[1] if len(navigable) >= 2 else None

    # Page sizes
    nrows = shape[row_dim] if row_dim is not None else 1
    ncols = shape[col_dim] if col_dim is not None else 1

    # Clamp fixed values
    fixed_values = {}
    for d, val in layout.fixed_values.items():
        total = shape[d]
        fixed_values[d] = max(0, min(val, total - 1)) if total > 0 else 0

    # Ensure every non-navigable dim is fixed at 0 (safety catch)
    for i in range(ndim):
        if i not in fixed_values and (row_dim is None or i != row_dim) and (col_dim is None or i != col_dim):
            fixed_values[i] = 0

    # Build slicing tuple
    idx: list[int | slice] = []
    for i in range(ndim):
        if i in fixed_values:
            idx.append(fixed_values[i])
        elif row_dim is not None and i == row_dim:
            start = max(0, min(layout.row_start, nrows))
            stop = min(max(start, start + max_rows), nrows)
            idx.append(slice(start, stop))
        elif col_dim is not None and i == col_dim:
            col_start = max(0, min(layout.col_start, ncols))
            col_stop = min(col_start + max_cols, ncols)
            idx.append(slice(col_start, col_stop))
        else:
            # Shouldn't happen: non-navigable dims are caught above
            idx.append(slice(0, shape[i]))

    values = np.asarray(obj[tuple(idx)])

    # Build column labels — match data keys below
    if col_dim is not None:
        col_start = max(0, min(layout.col_start, ncols))
        col_stop = min(col_start + max_cols, ncols)
        columns = [str(i) for i in range(col_start, col_stop)]
    elif row_dim is not None:
        columns = ["value"]
    else:
        columns = ["value"]

    # Extract 2-D data from result
    data: dict[str, Any] = {}
    if row_dim is not None and col_dim is not None:
        # 2-D navigable → 2-D table
        col_start = max(0, min(layout.col_start, ncols))
        col_stop = min(col_start + max_cols, ncols)
        for i, c in enumerate(range(col_start, col_stop)):
            data[str(c)] = values[:, i]
    elif row_dim is not None:
        # Only rows navigable → 1-D view
        data["value"] = values
    else:
        # 0 navigable → scalar
        data["value"] = np.asarray([values.item()]) if np.ndim(values) == 0 else np.asarray([values])

    row_start_val = max(0, min(layout.row_start, nrows)) if row_dim is not None else 0
    row_stop_val = min(row_start_val + max_rows, nrows) if row_dim is not None else 1
    col_start_val = max(0, min(layout.col_start, ncols)) if col_dim is not None else 0
    col_stop_val = min(col_start_val + max_cols, ncols) if col_dim is not None else 1

    result: dict[str, Any] = {
        "start": row_start_val,
        "stop": row_stop_val,
        "nrows": nrows,
        "columns": columns,
        "hidden_columns": max(0, ncols - (col_stop_val - col_start_val)),
        "data": data,
        "source_kind": "ndarray_slice",
        "shape": shape,
        "col_start": col_start_val,
        "col_stop": col_stop_val,
        "ncols": ncols,
        "layout": layout,
        "slice_indices": [fixed_values.get(i, 0) for i in range(min(ndim - 2, ndim))],
        "n_slices_per_dim": [shape[i] for i in range(ndim) if i in fixed_values],
    }
    # Keep legacy fields for backward compat
    result["slice_indices"] = [fixed_values.get(i, 0) for i in range(ndim) if i in fixed_values]
    result["n_slices_per_dim"] = [shape[i] for i in range(ndim) if i in fixed_values]
    return result


def preview_array_nd_slice(
    obj: Any,
    *,
    slice_indices: list[int] | None = None,
    start: int = 0,
    stop: int = 20,
    col_start: int = 0,
    max_cols: int = 10,
) -> dict[str, Any]:
    """Return a bounded 2-D slice preview for N-D arrays (N >= 3)."""
    shape = tuple(getattr(obj, "shape", ()) or ())
    ndim = len(shape)
    if ndim < 3:
        raise ValueError(f"Expected an N-D array with N >= 3, got shape {shape!r}")
    n_leading = ndim - 2
    n_slices_per_dim = list(shape[:n_leading])
    if slice_indices is None or len(slice_indices) != n_leading:
        slice_indices = [0] * n_leading
    # Clamp
    slice_indices = [
        min(max(0, idx), n_slices_per_dim[i] - 1) if n_slices_per_dim[i] > 0 else 0
        for i, idx in enumerate(slice_indices)
    ]
    nrows, ncols = shape[-2], shape[-1]
    if stop is None:
        stop = min(start + 20, nrows)
    start = max(0, min(start, nrows))
    stop = min(max(start, stop), nrows)
    col_start = max(0, min(col_start, ncols))
    col_stop = min(col_start + max_cols, ncols)
    columns = [str(i) for i in range(col_start, col_stop)]
    idx = tuple(slice_indices) + (slice(start, stop), slice(col_start, col_stop))
    values = np.asarray(obj[idx])
    data = {str(col): values[:, i] for i, col in enumerate(range(col_start, col_stop))}
    return {
        "start": start,
        "stop": stop,
        "nrows": nrows,
        "columns": columns,
        "hidden_columns": max(0, ncols - (col_stop - col_start)),
        "data": data,
        "source_kind": "ndarray_slice",
        "shape": shape,
        "col_start": col_start,
        "col_stop": col_stop,
        "ncols": ncols,
        "slice_indices": slice_indices,
        "n_slices_per_dim": n_slices_per_dim,
    }


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


def preview_array_1d(obj: Any, *, start: int = 0, stop: int = 20, **kwargs) -> dict[str, Any]:
    """Return a bounded row preview for a 1-D array."""
    shape = tuple(getattr(obj, "shape", ()) or ())
    if len(shape) != 1:
        raise ValueError(f"Expected a 1-D array, got shape {shape!r}")
    nrows = shape[0]
    start = max(0, min(start, nrows))
    stop = min(max(start, stop), nrows)
    data = {
        "value": np.asarray(obj[start:stop]),
    }
    return {
        "start": start,
        "stop": stop,
        "nrows": nrows,
        "columns": ["value"],
        "hidden_columns": 0,
        "data": data,
        "source_kind": "ndarray1d",
        "shape": shape,
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
