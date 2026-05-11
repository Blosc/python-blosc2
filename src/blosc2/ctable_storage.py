#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

"""Storage backends for CTable.

Two concrete backends:

* :class:`InMemoryTableStorage` — all arrays live in RAM (default when
  ``urlpath`` is not provided).
* :class:`FileTableStorage` — arrays are stored inside a :class:`blosc2.TreeStore`
  rooted at ``urlpath``; logical object metadata lives in ``/_meta`` and table
  data lives under ``/_valid_rows`` and ``/_cols/<name>``.
"""

from __future__ import annotations

import copy
import json
import os
from typing import TYPE_CHECKING, Any

import numpy as np

import blosc2
from blosc2.batch_array import BatchArray
from blosc2.dictionary_column import DictionaryColumn
from blosc2.list_array import ListArray
from blosc2.scalar_array import (
    _make_persistent_backend,
    _open_persistent_backend,
    _ScalarVarLenArray,
    _validate_role_metadata,
)
from blosc2.schunk import process_opened_object

if TYPE_CHECKING:
    from blosc2.schema import ListSpec

# Directory inside the table root that holds per-column index sidecar files.
_INDEXES_DIR = "_indexes"


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class TableStorage:
    """Interface that CTable uses to create/open its backing arrays."""

    def create_column(
        self,
        name: str,
        *,
        dtype: np.dtype,
        shape: tuple[int, ...],
        chunks: tuple[int, ...],
        blocks: tuple[int, ...],
        cparams: dict[str, Any] | None,
        dparams: dict[str, Any] | None,
    ) -> blosc2.NDArray:
        raise NotImplementedError

    def open_column(self, name: str) -> blosc2.NDArray:
        raise NotImplementedError

    def create_list_column(
        self,
        name: str,
        *,
        spec: ListSpec,
        cparams: dict[str, Any] | None,
        dparams: dict[str, Any] | None,
    ) -> ListArray:
        raise NotImplementedError

    def open_list_column(self, name: str) -> ListArray:
        raise NotImplementedError

    def create_varlen_scalar_column(
        self,
        name: str,
        *,
        spec,
        cparams=None,
        dparams=None,
    ) -> _ScalarVarLenArray:
        raise NotImplementedError

    def open_varlen_scalar_column(self, name: str, spec) -> _ScalarVarLenArray:
        raise NotImplementedError

    def create_dictionary_column(
        self,
        name: str,
        *,
        spec,
        cparams: dict[str, Any] | None = None,
        dparams: dict[str, Any] | None = None,
    ) -> DictionaryColumn:
        raise NotImplementedError

    def open_dictionary_column(self, name: str, spec) -> DictionaryColumn:
        raise NotImplementedError

    def create_valid_rows(
        self,
        *,
        shape: tuple[int, ...],
        chunks: tuple[int, ...],
        blocks: tuple[int, ...],
    ) -> blosc2.NDArray:
        raise NotImplementedError

    def open_valid_rows(self) -> blosc2.NDArray:
        raise NotImplementedError

    def save_schema(self, schema_dict: dict[str, Any]) -> None:
        raise NotImplementedError

    def load_schema(self) -> dict[str, Any] | None:
        raise NotImplementedError

    def table_exists(self) -> bool:
        raise NotImplementedError

    def is_read_only(self) -> bool:
        raise NotImplementedError

    def open_mode(self) -> str | None:
        raise NotImplementedError

    def delete_column(self, name: str) -> None:
        raise NotImplementedError

    def rename_column(self, old: str, new: str) -> blosc2.NDArray:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

    def discard(self) -> None:
        """Clean up resources without persisting changes back to the archive."""
        self.close()

    # -- Index catalog and epoch helpers -------------------------------------

    def load_index_catalog(self) -> dict:
        """Return the current index catalog (column_name → descriptor dict)."""
        raise NotImplementedError

    def save_index_catalog(self, catalog: dict) -> None:
        """Persist *catalog* (column_name → descriptor dict)."""
        raise NotImplementedError

    def get_epoch_counters(self) -> tuple[int, int]:
        """Return ``(value_epoch, visibility_epoch)``."""
        raise NotImplementedError

    def bump_value_epoch(self) -> int:
        """Increment and return the value epoch (data values changed)."""
        raise NotImplementedError

    def bump_visibility_epoch(self) -> int:
        """Increment and return the visibility epoch (row set changed by delete)."""
        raise NotImplementedError

    def index_anchor_path(self, col_name: str) -> str | None:
        """Return the urlpath used as the anchor for index sidecar naming.

        Returns *None* for in-memory storage.  For file-backed storage returns
        a path of the form ``<root>/_indexes/<col_name>/_anchor``.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# In-memory backend
# ---------------------------------------------------------------------------


class InMemoryTableStorage(TableStorage):
    """All arrays are plain in-memory blosc2.NDArray objects."""

    def __init__(self) -> None:
        self._index_catalog: dict = {}
        self._value_epoch: int = 0
        self._visibility_epoch: int = 0

    def create_column(self, name, *, dtype, shape, chunks, blocks, cparams, dparams):
        kwargs: dict[str, Any] = {"chunks": chunks, "blocks": blocks}
        if cparams is not None:
            kwargs["cparams"] = cparams
        if dparams is not None:
            kwargs["dparams"] = dparams
        return blosc2.zeros(shape, dtype=dtype, **kwargs)

    def open_column(self, name):
        raise RuntimeError("In-memory tables have no on-disk representation to open.")

    def create_list_column(self, name, *, spec, cparams, dparams):
        kwargs = {}
        if cparams is not None:
            kwargs["cparams"] = cparams
        if dparams is not None:
            kwargs["dparams"] = dparams
        return ListArray(spec=spec, **kwargs)

    def open_list_column(self, name):
        raise RuntimeError("In-memory tables have no on-disk representation to open.")

    def create_varlen_scalar_column(self, name, *, spec, cparams=None, dparams=None):
        return _ScalarVarLenArray(spec)

    def open_varlen_scalar_column(self, name, spec):
        raise RuntimeError("In-memory tables have no on-disk representation to open.")

    def create_dictionary_column(self, name, *, spec, cparams=None, dparams=None):
        from blosc2.schema import VLStringSpec

        chunks, blocks = (4096,), (256,)
        codes = blosc2.zeros((4096,), dtype=np.int32, chunks=chunks, blocks=blocks)
        dict_store = _ScalarVarLenArray(VLStringSpec(nullable=False))
        return DictionaryColumn(spec, codes, dict_store)

    def open_dictionary_column(self, name, spec):
        raise RuntimeError("In-memory tables have no on-disk representation to open.")

    def create_valid_rows(self, *, shape, chunks, blocks):
        return blosc2.zeros(shape, dtype=np.bool_, chunks=chunks, blocks=blocks)

    def open_valid_rows(self):
        raise RuntimeError("In-memory tables have no on-disk representation to open.")

    def save_schema(self, schema_dict):
        pass  # nothing to persist

    def load_schema(self):
        return None

    def table_exists(self):
        return False

    def is_read_only(self):
        return False

    def open_mode(self) -> str | None:
        return None

    def delete_column(self, name):
        raise RuntimeError("In-memory tables have no on-disk representation to mutate.")

    def rename_column(self, old: str, new: str):
        raise RuntimeError("In-memory tables have no on-disk representation to mutate.")

    def close(self):
        pass

    # -- Index catalog and epoch helpers -------------------------------------

    def load_index_catalog(self) -> dict:
        return copy.deepcopy(self._index_catalog)

    def save_index_catalog(self, catalog: dict) -> None:
        self._index_catalog = copy.deepcopy(catalog)

    def get_epoch_counters(self) -> tuple[int, int]:
        return self._value_epoch, self._visibility_epoch

    def bump_value_epoch(self) -> int:
        self._value_epoch += 1
        return self._value_epoch

    def bump_visibility_epoch(self) -> int:
        self._visibility_epoch += 1
        return self._visibility_epoch

    def index_anchor_path(self, col_name: str) -> str | None:
        return None


# ---------------------------------------------------------------------------
# File-backed backend
# ---------------------------------------------------------------------------

_META_KEY = "/_meta"
_VALID_ROWS_KEY = "/_valid_rows"
_COLS_DIR = "_cols"


class FileTableStorage(TableStorage):
    """Arrays stored as TreeStore leaves inside *urlpath*.

    Parameters
    ----------
    urlpath:
        Path to the backing TreeStore (typically ``.b2d`` or ``.b2z``).
    mode:
        ``'w'`` — create (overwrite existing files).
        ``'a'`` — open existing or create new.
        ``'r'`` — open existing read-only.
    """

    def __init__(self, urlpath: str, mode: str) -> None:
        if mode not in ("r", "a", "w"):
            raise ValueError(f"mode must be 'r', 'a', or 'w'; got {mode!r}")
        self._root = urlpath
        self._mode = mode
        self._meta: blosc2.SChunk | None = None
        self._store: blosc2.TreeStore | None = None

    # ------------------------------------------------------------------
    # Key helpers
    # ------------------------------------------------------------------

    @property
    def _meta_path(self) -> str:
        return self._key_to_path(_META_KEY)

    @property
    def _valid_rows_path(self) -> str:
        return self._key_to_path(_VALID_ROWS_KEY)

    def _col_path(self, name: str) -> str:
        return self._key_to_path(self._col_key(name))

    def _list_col_path(self, name: str) -> str:
        rel_key = self._col_key(name).lstrip("/")
        # Use working_dir so .b2z stores write into their temp dir (gets zipped on close).
        # For .b2d, working_dir == self._root, so behaviour is unchanged.
        return os.path.join(self._open_store().working_dir, rel_key + ".b2b")

    def _dict_col_path(self, name: str) -> str:
        """Path for the dictionary values store of a dictionary column."""
        rel_key = self._col_key(name).lstrip("/")
        return os.path.join(self._open_store().working_dir, rel_key + "_dict.b2b")

    def _col_key(self, name: str) -> str:
        return f"/{_COLS_DIR}/{name}"

    def _key_to_path(self, key: str) -> str:
        rel_key = key.lstrip("/")
        suffix = ".b2f" if key == _META_KEY else ".b2nd"
        if self._root.endswith(".b2d"):
            return os.path.join(self._root, rel_key + suffix)
        return os.path.join(self._root, rel_key + suffix)

    def _open_store(self) -> blosc2.TreeStore:
        if self._store is None:
            kwargs: dict[str, Any] = {"mode": self._mode}
            if self._mode != "r":
                # Force table internals to be stored as proper external leaves so
                # reopened arrays stay live and mutable through the TreeStore.
                kwargs["threshold"] = 0
            self._store = blosc2.TreeStore(self._root, **kwargs)
        return self._store

    # ------------------------------------------------------------------
    # TableStorage interface
    # ------------------------------------------------------------------

    def table_exists(self) -> bool:
        return os.path.exists(self._root)

    def is_read_only(self) -> bool:
        return self._mode == "r"

    def open_mode(self) -> str | None:
        return self._mode

    def create_column(self, name, *, dtype, shape, chunks, blocks, cparams, dparams):
        kwargs: dict[str, Any] = {
            "chunks": chunks,
            "blocks": blocks,
        }
        if cparams is not None:
            kwargs["cparams"] = cparams
        if dparams is not None:
            kwargs["dparams"] = dparams
        col = blosc2.zeros(shape, dtype=dtype, **kwargs)
        store = self._open_store()
        store[self._col_key(name)] = col
        return store[self._col_key(name)]

    def open_column(self, name: str) -> blosc2.NDArray:
        return self._open_store()[self._col_key(name)]

    def create_list_column(self, name, *, spec, cparams, dparams):
        kwargs: dict[str, Any] = {"urlpath": self._list_col_path(name), "mode": "w", "contiguous": True}
        if cparams is not None:
            kwargs["cparams"] = cparams
        if dparams is not None:
            kwargs["dparams"] = dparams
        os.makedirs(os.path.dirname(self._list_col_path(name)), exist_ok=True)
        return ListArray(spec=spec, **kwargs)

    def open_list_column(self, name: str) -> ListArray:
        store = self._open_store()
        if store.is_zip_store and self._mode == "r":
            # In read mode, .b2z is never extracted — read the member at its zip offset directly.
            rel = f"{_COLS_DIR}/{name}.b2b"
            if rel not in store.offsets:
                raise KeyError(f"List column {name!r} not found in {self._root!r}")
            opened = blosc2.blosc2_ext.open(store.b2z_path, mode="r", offset=store.offsets[rel]["offset"])
            return process_opened_object(opened)
        return blosc2.open(self._list_col_path(name), mode=self._mode)

    def create_varlen_scalar_column(self, name, *, spec, cparams=None, dparams=None) -> _ScalarVarLenArray:
        urlpath = self._list_col_path(name)
        backend = _make_persistent_backend(spec, urlpath, "w", cparams=cparams, dparams=dparams)
        return _ScalarVarLenArray(spec, backend)

    def open_varlen_scalar_column(self, name: str, spec) -> _ScalarVarLenArray:
        store = self._open_store()
        path = self._list_col_path(name)
        if store.is_zip_store and self._mode == "r":
            rel = f"{_COLS_DIR}/{name}.b2b"
            if rel not in store.offsets:
                raise KeyError(f"Varlen scalar column {name!r} not found in {self._root!r}")
            backend = BatchArray(
                _from_schunk=blosc2.blosc2_ext.open(
                    store.b2z_path, mode="r", offset=store.offsets[rel]["offset"]
                )
            )
        else:
            backend = _open_persistent_backend(path, self._mode, spec=spec)
        _validate_role_metadata(backend, spec)
        return _ScalarVarLenArray(spec, backend)

    def create_dictionary_column(self, name, *, spec, cparams=None, dparams=None) -> DictionaryColumn:
        from blosc2.schema import VLStringSpec

        # Codes: stored as a regular NDArray under _cols/name
        codes = self.create_column(
            name,
            dtype=np.int32,
            shape=(4096,),
            chunks=(4096,),
            blocks=(256,),
            cparams=cparams,
            dparams=dparams,
        )
        # Dictionary values: stored as a varlen scalar (vlstring) at name_dict.b2b
        dict_spec = VLStringSpec(nullable=False)
        dict_path = self._dict_col_path(name)
        dict_backend = _make_persistent_backend(dict_spec, dict_path, "w")
        dict_store = _ScalarVarLenArray(dict_spec, dict_backend)
        return DictionaryColumn(spec, codes, dict_store)

    def open_dictionary_column(self, name: str, spec) -> DictionaryColumn:
        from blosc2.schema import VLStringSpec

        codes = self.open_column(name)
        dict_spec = VLStringSpec(nullable=False)
        store = self._open_store()
        dict_path = self._dict_col_path(name)
        if store.is_zip_store and self._mode == "r":
            rel = f"{_COLS_DIR}/{name}_dict.b2b"
            if rel not in store.offsets:
                raise KeyError(f"Dictionary column dict store {name!r} not found in {self._root!r}")
            dict_backend = BatchArray(
                _from_schunk=blosc2.blosc2_ext.open(
                    store.b2z_path, mode="r", offset=store.offsets[rel]["offset"]
                )
            )
        else:
            dict_backend = _open_persistent_backend(dict_path, self._mode, spec=dict_spec)
        dict_store = _ScalarVarLenArray(dict_spec, dict_backend)
        return DictionaryColumn(spec, codes, dict_store)

    def create_valid_rows(self, *, shape, chunks, blocks):
        valid_rows = blosc2.zeros(
            shape,
            dtype=np.bool_,
            chunks=chunks,
            blocks=blocks,
        )
        store = self._open_store()
        store[_VALID_ROWS_KEY] = valid_rows
        return store[_VALID_ROWS_KEY]

    def open_valid_rows(self) -> blosc2.NDArray:
        return self._open_store()[_VALID_ROWS_KEY]

    def save_schema(self, schema_dict: dict[str, Any]) -> None:
        """Write *schema_dict* (plus kind/version markers) to ``/_meta``."""
        meta = blosc2.SChunk()
        meta.vlmeta["kind"] = "ctable"
        meta.vlmeta["version"] = 1
        meta.vlmeta["schema"] = json.dumps(schema_dict)
        store = self._open_store()
        store[_META_KEY] = meta
        opened = store[_META_KEY]
        if not isinstance(opened, blosc2.SChunk):
            raise ValueError("CTable manifest '/_meta' must materialize as an SChunk.")
        self._meta = opened

    def _open_meta(self) -> blosc2.SChunk:
        """Open (or return cached) the ``/_meta`` SChunk."""
        if self._meta is None:
            try:
                opened = self._open_store()[_META_KEY]
            except KeyError as exc:
                raise FileNotFoundError(f"No CTable manifest found at {self._root!r}") from exc
            if not isinstance(opened, blosc2.SChunk):
                raise ValueError(f"CTable manifest at {self._root!r} must be an SChunk.")
            self._meta = opened
        return self._meta

    def load_schema(self) -> dict[str, Any]:
        """Read and return the schema dict stored in ``/_meta``."""
        raw = self._open_meta().vlmeta["schema"]
        if isinstance(raw, bytes):
            raw = raw.decode()
        return json.loads(raw)

    def check_kind(self) -> None:
        """Raise :exc:`ValueError` if ``_meta`` does not identify a CTable."""
        kind = self._open_meta().vlmeta["kind"]
        if isinstance(kind, bytes):
            kind = kind.decode()
        if kind != "ctable":
            raise ValueError(f"Path {self._root!r} does not contain a CTable (kind={kind!r}).")

    def column_names_from_schema(self) -> list[str]:
        d = self.load_schema()
        return [c["name"] for c in d["columns"]]

    def delete_column(self, name: str) -> None:
        key = self._col_key(name)
        if key in self._open_store():
            del self._open_store()[key]
            return
        list_path = self._list_col_path(name)
        if os.path.exists(list_path):
            blosc2.remove_urlpath(list_path)
            return
        raise KeyError(name)

    def rename_column(self, old: str, new: str):
        store = self._open_store()
        old_key = self._col_key(old)
        new_key = self._col_key(new)
        if old_key in store:
            store[new_key] = store[old_key]
            del store[old_key]
            return store[new_key]
        old_path = self._list_col_path(old)
        new_path = self._list_col_path(new)
        if os.path.exists(old_path):
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            os.replace(old_path, new_path)
            return blosc2.open(new_path, mode=self._mode)
        raise KeyError(old)

    def close(self) -> None:
        if self._store is not None:
            self._store.close()
            self._store = None
        self._meta = None

    def discard(self) -> None:
        """Clean up without repacking the .b2z archive."""
        if self._store is not None:
            self._store.discard()
            self._store = None
        self._meta = None

    # -- Index catalog and epoch helpers -------------------------------------

    @staticmethod
    def _walk_descriptor_paths(descriptor: dict):
        """Yield (obj, key) for every string value that looks like a file path."""
        stack = [descriptor]
        while stack:
            obj = stack.pop()
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if (k == "path" or k.endswith("_path")) and isinstance(v, str):
                        yield obj, k
                    elif isinstance(v, (dict, list)):
                        stack.append(v)
            elif isinstance(obj, list):
                for item in obj:
                    if isinstance(item, (dict, list)):
                        stack.append(item)

    @staticmethod
    def _relativize_descriptor(descriptor: dict, working_dir: str) -> dict:
        """Replace absolute paths inside *working_dir* with working-dir relative paths."""
        prefix = working_dir.rstrip(os.sep) + os.sep
        d = copy.deepcopy(descriptor)
        for obj, key in FileTableStorage._walk_descriptor_paths(d):
            v = obj[key]
            if os.path.isabs(v) and v.startswith(prefix):
                obj[key] = v[len(prefix) :].replace(os.sep, "/")
        return d

    @staticmethod
    def _absolutize_descriptor(descriptor: dict, working_dir: str) -> dict:
        """Expand working-dir relative paths back to absolute paths."""
        d = copy.deepcopy(descriptor)
        for obj, key in FileTableStorage._walk_descriptor_paths(d):
            v = obj[key]
            if not os.path.isabs(v):
                obj[key] = os.path.join(working_dir, v)
        return d

    def _ensure_index_files_extracted(self, store, rel_paths: list[str]) -> None:
        """Extract *rel_paths* from the zip into the working_dir (read mode only)."""
        import zipfile

        for rel in rel_paths:
            dest = os.path.join(store.working_dir, rel)
            if os.path.exists(dest):
                continue
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            info = store.offsets.get(rel)
            if info is None:
                continue
            with zipfile.ZipFile(store.b2z_path, "r") as zf, zf.open(rel) as src, open(dest, "wb") as dst:
                dst.write(src.read())

    def load_index_catalog(self) -> dict:
        meta = self._open_meta()
        raw = meta.vlmeta.get("index_catalog")
        if not isinstance(raw, dict):
            return {}
        catalog = copy.deepcopy(raw)
        store = self._open_store()
        working_dir = store.working_dir
        # Expand relative paths and, for b2z read mode, extract sidecar files.
        rel_paths_needed = []
        for col_name, descriptor in catalog.items():
            catalog[col_name] = self._absolutize_descriptor(descriptor, working_dir)
            if store.is_zip_store and self._mode == "r":
                for obj, key in self._walk_descriptor_paths(catalog[col_name]):
                    v = obj[key]
                    rel = os.path.relpath(v, working_dir)
                    if not os.path.exists(v):
                        rel_paths_needed.append(rel.replace(os.sep, "/"))
        if rel_paths_needed and store.is_zip_store and self._mode == "r":
            self._ensure_index_files_extracted(store, rel_paths_needed)
        return catalog

    def save_index_catalog(self, catalog: dict) -> None:
        meta = self._open_meta()
        working_dir = self._open_store().working_dir
        relativized = {col: self._relativize_descriptor(desc, working_dir) for col, desc in catalog.items()}
        meta.vlmeta["index_catalog"] = relativized

    def get_epoch_counters(self) -> tuple[int, int]:
        meta = self._open_meta()
        ve = int(meta.vlmeta.get("value_epoch", 0) or 0)
        vis_e = int(meta.vlmeta.get("visibility_epoch", 0) or 0)
        return ve, vis_e

    def bump_value_epoch(self) -> int:
        meta = self._open_meta()
        ve = int(meta.vlmeta.get("value_epoch", 0) or 0) + 1
        meta.vlmeta["value_epoch"] = ve
        return ve

    def bump_visibility_epoch(self) -> int:
        meta = self._open_meta()
        vis_e = int(meta.vlmeta.get("visibility_epoch", 0) or 0) + 1
        meta.vlmeta["visibility_epoch"] = vis_e
        return vis_e

    def index_anchor_path(self, col_name: str) -> str | None:
        return os.path.join(self._open_store().working_dir, _INDEXES_DIR, col_name, "_anchor")


# ---------------------------------------------------------------------------
# TreeStore-backed backend (inline subtree layout)
# ---------------------------------------------------------------------------


class TreeStoreTableStorage(TableStorage):
    """TableStorage backend that stores a CTable inline inside an outer TreeStore.

    All CTable components are written as normal external leaves under *root_key*
    inside *store*'s working directory.  This avoids nested ZIP files and allows
    the entire bundle to be packed as a flat ``.b2z`` archive.

    Parameters
    ----------
    store:
        The outer :class:`blosc2.TreeStore` (or its subtree view) that will
        hold the CTable internals.
    root_key:
        Full absolute key where the CTable lives, e.g. ``"/table"``.
    mode:
        Open mode (``'r'``, ``'a'``, or ``'w'``).  Should match ``store.mode``.
    owns_store:
        If ``True``, ``close()`` / ``discard()`` will also close / discard
        *store*.  Use ``False`` (default) when *store* is owned by the caller.
    """

    def __init__(
        self,
        store: blosc2.TreeStore,
        root_key: str,
        mode: str,
        owns_store: bool = False,
    ) -> None:
        self._store = store
        self._root_key = root_key.rstrip("/")
        self._mode = mode
        self._owns_store = owns_store
        self._meta: blosc2.SChunk | None = None

    # ------------------------------------------------------------------
    # Key / path helpers
    # ------------------------------------------------------------------

    def _table_key(self, logical_key: str) -> str:
        """Translate a CTable-internal logical key to an outer-store absolute key.

        For example, if *root_key* is ``"/table"`` and *logical_key* is
        ``"/_meta"``, the result is ``"/table/_meta"``.
        """
        return self._root_key + logical_key

    def _working_dir(self) -> str:
        return self._store.working_dir

    def _dest_path(self, logical_key: str, ext: str) -> str:
        """Absolute filesystem path for the external leaf file."""
        rel = self._table_key(logical_key).lstrip("/")
        return os.path.join(self._working_dir(), rel + ext)

    def _write_leaf(self, logical_key: str, value: Any, ext: str) -> None:
        """Write *value* as a raw cframe file and register it in the outer
        store's map_tree so DictStore can find it again on open."""
        dest_path = self._dest_path(logical_key, ext)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        if isinstance(value, blosc2.SChunk) or hasattr(value, "to_cframe"):
            with open(dest_path, "wb") as f:
                f.write(value.to_cframe())
        else:
            value.save(urlpath=dest_path, mode="w")
        rel_path = os.path.relpath(dest_path, self._working_dir()).replace(os.sep, "/")
        full_key = self._table_key(logical_key)
        self._store.map_tree[full_key] = rel_path
        self._store._modified = True

    def _open_leaf(self, logical_key: str) -> Any:
        """Open a leaf via the outer store's map_tree / estore logic."""
        from blosc2.dict_store import DictStore

        full_key = self._table_key(logical_key)
        return DictStore.__getitem__(self._store, full_key)

    def _list_col_path(self, name: str) -> str:
        """Filesystem path for a list-style column (``.b2b``)."""
        return self._dest_path(f"/_cols/{name}", ".b2b")

    # ------------------------------------------------------------------
    # TableStorage interface — lifecycle
    # ------------------------------------------------------------------

    def table_exists(self) -> bool:
        full_key = self._table_key("/_meta")
        return full_key in self._store.map_tree or full_key in self._store._estore

    def is_read_only(self) -> bool:
        return self._mode == "r"

    def open_mode(self) -> str | None:
        return self._mode

    def close(self) -> None:
        if self._owns_store and self._store is not None:
            self._store.close()
            self._store = None
        self._meta = None

    def discard(self) -> None:
        if self._owns_store and self._store is not None:
            self._store.discard()
            self._store = None
        self._meta = None

    # ------------------------------------------------------------------
    # TableStorage interface — columns and valid_rows
    # ------------------------------------------------------------------

    def create_column(
        self,
        name: str,
        *,
        dtype: np.dtype,
        shape: tuple[int, ...],
        chunks: tuple[int, ...],
        blocks: tuple[int, ...],
        cparams: dict[str, Any] | None,
        dparams: dict[str, Any] | None,
    ) -> blosc2.NDArray:
        kwargs: dict[str, Any] = {"chunks": chunks, "blocks": blocks}
        if cparams is not None:
            kwargs["cparams"] = cparams
        if dparams is not None:
            kwargs["dparams"] = dparams
        dest_path = self._dest_path(f"/_cols/{name}", ".b2nd")
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        col = blosc2.zeros(shape, dtype=dtype, urlpath=dest_path, mode="w", **kwargs)
        rel_path = os.path.relpath(dest_path, self._working_dir()).replace(os.sep, "/")
        self._store.map_tree[self._table_key(f"/_cols/{name}")] = rel_path
        self._store._modified = True
        return col

    def open_column(self, name: str) -> blosc2.NDArray:
        return self._open_leaf(f"/_cols/{name}")

    def create_list_column(
        self,
        name: str,
        *,
        spec: ListSpec,
        cparams: dict[str, Any] | None,
        dparams: dict[str, Any] | None,
    ) -> ListArray:
        kwargs: dict[str, Any] = {
            "urlpath": self._list_col_path(name),
            "mode": "w",
            "contiguous": True,
        }
        if cparams is not None:
            kwargs["cparams"] = cparams
        if dparams is not None:
            kwargs["dparams"] = dparams
        os.makedirs(os.path.dirname(self._list_col_path(name)), exist_ok=True)
        return ListArray(spec=spec, **kwargs)

    def open_list_column(self, name: str) -> ListArray:
        if self._store.is_zip_store and self._mode == "r":
            rel = self._table_key(f"/_cols/{name}").lstrip("/") + ".b2b"
            if rel not in self._store.offsets:
                raise KeyError(f"List column {name!r} not found in {self._store.localpath!r}")
            opened = blosc2.blosc2_ext.open(
                self._store.b2z_path,
                mode="r",
                offset=self._store.offsets[rel]["offset"],
            )
            return process_opened_object(opened)
        return blosc2.open(self._list_col_path(name), mode=self._mode)

    def create_varlen_scalar_column(
        self,
        name: str,
        *,
        spec,
        cparams=None,
        dparams=None,
    ) -> _ScalarVarLenArray:
        urlpath = self._list_col_path(name)
        os.makedirs(os.path.dirname(urlpath), exist_ok=True)
        return _make_persistent_backend(spec, urlpath, "w", cparams=cparams, dparams=dparams)

    def open_varlen_scalar_column(self, name: str, spec) -> _ScalarVarLenArray:
        if self._store.is_zip_store and self._mode == "r":
            rel = self._table_key(f"/_cols/{name}").lstrip("/") + ".b2b"
            if rel not in self._store.offsets:
                raise KeyError(f"Varlen scalar column {name!r} not found in {self._store.localpath!r}")
            backend = BatchArray(
                _from_schunk=blosc2.blosc2_ext.open(
                    self._store.b2z_path,
                    mode="r",
                    offset=self._store.offsets[rel]["offset"],
                )
            )
        else:
            backend = _open_persistent_backend(self._list_col_path(name), self._mode, spec=spec)
        _validate_role_metadata(backend, spec)
        return _ScalarVarLenArray(spec, backend)

    def _dict_col_path(self, name: str) -> str:
        """Path for the dictionary values store of a dictionary column."""
        return self._dest_path(f"/_cols/{name}", "_dict.b2b")

    def create_dictionary_column(
        self,
        name: str,
        *,
        spec,
        cparams=None,
        dparams=None,
    ) -> DictionaryColumn:
        from blosc2.schema import VLStringSpec

        codes = self.create_column(
            name,
            dtype=np.int32,
            shape=(4096,),
            chunks=(4096,),
            blocks=(256,),
            cparams=cparams,
            dparams=dparams,
        )
        dict_spec = VLStringSpec(nullable=False)
        dict_path = self._dict_col_path(name)
        os.makedirs(os.path.dirname(dict_path), exist_ok=True)
        dict_backend = _make_persistent_backend(dict_spec, dict_path, "w")
        dict_store = _ScalarVarLenArray(dict_spec, dict_backend)
        return DictionaryColumn(spec, codes, dict_store)

    def open_dictionary_column(self, name: str, spec) -> DictionaryColumn:
        from blosc2.schema import VLStringSpec

        codes = self.open_column(name)
        dict_spec = VLStringSpec(nullable=False)
        if self._store.is_zip_store and self._mode == "r":
            rel = self._table_key(f"/_cols/{name}").lstrip("/") + "_dict.b2b"
            if rel not in self._store.offsets:
                raise KeyError(
                    f"Dictionary column dict store {name!r} not found in {self._store.localpath!r}"
                )
            dict_backend = BatchArray(
                _from_schunk=blosc2.blosc2_ext.open(
                    self._store.b2z_path,
                    mode="r",
                    offset=self._store.offsets[rel]["offset"],
                )
            )
        else:
            dict_backend = _open_persistent_backend(self._dict_col_path(name), self._mode, spec=dict_spec)
        dict_store = _ScalarVarLenArray(dict_spec, dict_backend)
        return DictionaryColumn(spec, codes, dict_store)

    def create_valid_rows(
        self,
        *,
        shape: tuple[int, ...],
        chunks: tuple[int, ...],
        blocks: tuple[int, ...],
    ) -> blosc2.NDArray:
        dest_path = self._dest_path("/_valid_rows", ".b2nd")
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        valid_rows = blosc2.zeros(
            shape,
            dtype=np.bool_,
            chunks=chunks,
            blocks=blocks,
            urlpath=dest_path,
            mode="w",
        )
        rel_path = os.path.relpath(dest_path, self._working_dir()).replace(os.sep, "/")
        self._store.map_tree[self._table_key("/_valid_rows")] = rel_path
        self._store._modified = True
        return valid_rows

    def open_valid_rows(self) -> blosc2.NDArray:
        return self._open_leaf("/_valid_rows")

    # ------------------------------------------------------------------
    # TableStorage interface — schema and manifest
    # ------------------------------------------------------------------

    def save_schema(self, schema_dict: dict[str, Any]) -> None:
        meta = blosc2.SChunk()
        meta.vlmeta["kind"] = "ctable"
        meta.vlmeta["version"] = 1
        meta.vlmeta["schema"] = json.dumps(schema_dict)
        self._write_leaf("/_meta", meta, ".b2f")
        opened = self._open_leaf("/_meta")
        if not isinstance(opened, blosc2.SChunk):
            raise ValueError("CTable manifest '/_meta' must materialise as an SChunk.")
        self._meta = opened

    def _open_meta(self) -> blosc2.SChunk:
        if self._meta is None:
            try:
                opened = self._open_leaf("/_meta")
            except KeyError as exc:
                raise FileNotFoundError(f"No CTable manifest found at {self._root_key!r}") from exc
            if not isinstance(opened, blosc2.SChunk):
                raise ValueError(f"CTable manifest at {self._root_key!r} must be an SChunk.")
            self._meta = opened
        return self._meta

    def load_schema(self) -> dict[str, Any]:
        raw = self._open_meta().vlmeta["schema"]
        if isinstance(raw, bytes):
            raw = raw.decode()
        return json.loads(raw)

    def check_kind(self) -> None:
        kind = self._open_meta().vlmeta["kind"]
        if isinstance(kind, bytes):
            kind = kind.decode()
        if kind != "ctable":
            raise ValueError(f"Object at {self._root_key!r} is not a CTable (kind={kind!r})")

    def column_names_from_schema(self) -> list[str]:
        return [c["name"] for c in self.load_schema()["columns"]]

    def delete_column(self, name: str) -> None:
        full_key = self._table_key(f"/_cols/{name}")
        if full_key in self._store.map_tree:
            filepath = self._store.map_tree.pop(full_key)
            full_path = os.path.join(self._working_dir(), filepath)
            if os.path.exists(full_path):
                os.remove(full_path)
            return
        list_path = self._list_col_path(name)
        if os.path.exists(list_path):
            blosc2.remove_urlpath(list_path)
            return
        raise KeyError(name)

    def rename_column(self, old: str, new: str) -> blosc2.NDArray:
        old_key = self._table_key(f"/_cols/{old}")
        new_key = self._table_key(f"/_cols/{new}")
        if old_key in self._store.map_tree:
            new_dest = self._dest_path(f"/_cols/{new}", ".b2nd")
            old_dest = os.path.join(self._working_dir(), self._store.map_tree[old_key])
            os.makedirs(os.path.dirname(new_dest), exist_ok=True)
            os.replace(old_dest, new_dest)
            del self._store.map_tree[old_key]
            self._store.map_tree[new_key] = os.path.relpath(new_dest, self._working_dir()).replace(
                os.sep, "/"
            )
            self._store._modified = True
            return blosc2.open(new_dest, mode=self._mode)
        old_path = self._list_col_path(old)
        new_path = self._list_col_path(new)
        if os.path.exists(old_path):
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            os.replace(old_path, new_path)
            return blosc2.open(new_path, mode=self._mode)
        raise KeyError(old)

    # ------------------------------------------------------------------
    # TableStorage interface — index catalog and epoch counters
    # ------------------------------------------------------------------

    def load_index_catalog(self) -> dict:
        meta = self._open_meta()
        raw = meta.vlmeta.get("index_catalog")
        if not isinstance(raw, dict):
            return {}
        catalog = copy.deepcopy(raw)
        working_dir = self._working_dir()
        store = self._store
        rel_paths_needed = []
        for col_name, descriptor in catalog.items():
            catalog[col_name] = FileTableStorage._absolutize_descriptor(descriptor, working_dir)
            if store.is_zip_store and self._mode == "r":
                for obj, key in FileTableStorage._walk_descriptor_paths(catalog[col_name]):
                    v = obj[key]
                    if not os.path.exists(v):
                        rel_paths_needed.append(os.path.relpath(v, working_dir).replace(os.sep, "/"))
        if rel_paths_needed:
            self._ensure_index_files_extracted(rel_paths_needed)
        return catalog

    def _ensure_index_files_extracted(self, rel_paths: list[str]) -> None:
        import zipfile

        store = self._store
        for rel in rel_paths:
            dest = os.path.join(self._working_dir(), rel)
            if os.path.exists(dest):
                continue
            info = store.offsets.get(rel)
            if info is None:
                continue
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with zipfile.ZipFile(store.b2z_path, "r") as zf:
                with zf.open(rel) as src, open(dest, "wb") as dst:
                    dst.write(src.read())

    def save_index_catalog(self, catalog: dict) -> None:
        meta = self._open_meta()
        working_dir = self._working_dir()
        relativized = {
            col: FileTableStorage._relativize_descriptor(desc, working_dir) for col, desc in catalog.items()
        }
        meta.vlmeta["index_catalog"] = relativized

    def get_epoch_counters(self) -> tuple[int, int]:
        meta = self._open_meta()
        ve = int(meta.vlmeta.get("value_epoch", 0) or 0)
        vis_e = int(meta.vlmeta.get("visibility_epoch", 0) or 0)
        return ve, vis_e

    def bump_value_epoch(self) -> int:
        meta = self._open_meta()
        ve = int(meta.vlmeta.get("value_epoch", 0) or 0) + 1
        meta.vlmeta["value_epoch"] = ve
        return ve

    def bump_visibility_epoch(self) -> int:
        meta = self._open_meta()
        vis_e = int(meta.vlmeta.get("visibility_epoch", 0) or 0) + 1
        meta.vlmeta["visibility_epoch"] = vis_e
        return vis_e

    def index_anchor_path(self, col_name: str) -> str | None:
        table_rel = self._root_key.lstrip("/")
        return os.path.join(self._working_dir(), table_rel, _INDEXES_DIR, col_name, "_anchor")
