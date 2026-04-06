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
* :class:`FileTableStorage` — arrays are stored as individual Blosc2 files
  under a table root directory; schema and kind metadata live in a small
  :class:`blosc2.SChunk` whose ``vlmeta`` is the source of truth.

Layout produced by :class:`FileTableStorage`::

    <urlpath>/
        _meta.b2frame       ← SChunk with vlmeta: kind, version, schema JSON
        _valid_rows.b2nd    ← boolean NDArray (tombstone mask)
        _cols/
            <name>.b2nd     ← one NDArray per column
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

import blosc2

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


# ---------------------------------------------------------------------------
# In-memory backend
# ---------------------------------------------------------------------------


class InMemoryTableStorage(TableStorage):
    """All arrays are plain in-memory blosc2.NDArray objects."""

    def create_column(self, name, *, dtype, shape, chunks, blocks, cparams, dparams):
        kwargs: dict[str, Any] = {"chunks": chunks, "blocks": blocks}
        if cparams is not None:
            kwargs["cparams"] = cparams
        if dparams is not None:
            kwargs["dparams"] = dparams
        return blosc2.zeros(shape, dtype=dtype, **kwargs)

    def open_column(self, name):
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


# ---------------------------------------------------------------------------
# File-backed backend
# ---------------------------------------------------------------------------

_META_FILE = "_meta.b2frame"
_VALID_ROWS_FILE = "_valid_rows.b2nd"
_COLS_DIR = "_cols"


class FileTableStorage(TableStorage):
    """Arrays stored as individual Blosc2 files inside *urlpath* directory.

    Parameters
    ----------
    urlpath:
        Path to the table root directory.
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

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    @property
    def _meta_path(self) -> str:
        return os.path.join(self._root, _META_FILE)

    @property
    def _valid_rows_path(self) -> str:
        return os.path.join(self._root, _VALID_ROWS_FILE)

    def _col_path(self, name: str) -> str:
        return os.path.join(self._root, _COLS_DIR, f"{name}.b2nd")

    def _ensure_dirs(self) -> None:
        os.makedirs(os.path.join(self._root, _COLS_DIR), exist_ok=True)

    # ------------------------------------------------------------------
    # TableStorage interface
    # ------------------------------------------------------------------

    def table_exists(self) -> bool:
        return os.path.exists(self._meta_path)

    def is_read_only(self) -> bool:
        return self._mode == "r"

    def create_column(self, name, *, dtype, shape, chunks, blocks, cparams, dparams):
        self._ensure_dirs()
        kwargs: dict[str, Any] = {
            "chunks": chunks,
            "blocks": blocks,
            "urlpath": self._col_path(name),
            "mode": "w",
        }
        if cparams is not None:
            kwargs["cparams"] = cparams
        if dparams is not None:
            kwargs["dparams"] = dparams
        return blosc2.zeros(shape, dtype=dtype, **kwargs)

    def open_column(self, name: str) -> blosc2.NDArray:
        b2_mode = "r" if self._mode == "r" else "a"
        return blosc2.open(self._col_path(name), mode=b2_mode)

    def create_valid_rows(self, *, shape, chunks, blocks):
        self._ensure_dirs()
        return blosc2.zeros(
            shape,
            dtype=np.bool_,
            chunks=chunks,
            blocks=blocks,
            urlpath=self._valid_rows_path,
            mode="w",
        )

    def open_valid_rows(self) -> blosc2.NDArray:
        b2_mode = "r" if self._mode == "r" else "a"
        return blosc2.open(self._valid_rows_path, mode=b2_mode)

    def save_schema(self, schema_dict: dict[str, Any]) -> None:
        """Write *schema_dict* (plus kind/version markers) to ``_meta.b2frame``."""
        self._ensure_dirs()
        # Always overwrite: save_schema is only called at table-creation time.
        self._meta = blosc2.SChunk(urlpath=self._meta_path, mode="w")
        self._meta.vlmeta["kind"] = "ctable"
        self._meta.vlmeta["version"] = 1
        self._meta.vlmeta["schema"] = json.dumps(schema_dict)

    def _open_meta(self) -> blosc2.SChunk:
        """Open (or return cached) the ``_meta.b2frame`` SChunk."""
        if self._meta is None:
            b2_mode = "r" if self._mode == "r" else "a"
            self._meta = blosc2.open(self._meta_path, mode=b2_mode)
        return self._meta

    def load_schema(self) -> dict[str, Any]:
        """Read and return the schema dict stored in ``_meta.b2frame``."""
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
