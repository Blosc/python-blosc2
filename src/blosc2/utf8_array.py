#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

"""Internal variable-length UTF-8 string column backed by offsets + bytes.

This module is *not* part of the public API.  It provides row-wise string
semantics (one ``str`` value per row) stored Arrow-style as two companion
:class:`blosc2.NDArray` objects:

* **offsets** — ``int64``, length ``n + 1`` where ``n`` is the number of
  persisted rows.  ``offsets[0]`` is always ``0`` and ``offsets[i+1]`` is the
  end byte position of row ``i``.
* **data** — ``uint8``, the concatenated UTF-8 encoding of all row values.
  Its length is at least ``offsets[n]`` (one slack byte is kept when empty
  because zero-length NDArrays cannot be created).

Reading rows ``[a, b)`` needs ``offsets[a : b + 1]`` plus
``bytes[offsets[a] : offsets[b]]`` — both plain NDArray slice reads.

Nulls are represented with a per-column sentinel string (like every other
scalar CTable column); ``None`` written to a nullable column is converted to
the sentinel, and reads return the sentinel verbatim.

Bulk reads return :class:`numpy.dtypes.StringDType` arrays (NumPy >= 2.0),
which support vectorized comparison, ordering, and ``np.strings`` functions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

# Pending rows are flushed to the backing arrays once either bound is hit.
_FLUSH_ROWS = 4096
_FLUSH_CHARS = 1 << 22  # ~4 Mi characters (>= 4 MiB encoded)

# Storage grids for freshly created backing arrays.  Both arrays are created
# tiny (shape (1,)) and grown by resize; the chunk shape is fixed at creation
# time, so it must be sized for the eventual data, not the initial shape.
_OFFSETS_CHUNKS = (2**17,)  # 1 MiB chunks of int64 row offsets
_DATA_CHUNKS = (2**21,)  # 2 MiB chunks of UTF-8 bytes

# Sparse gathers read the persisted region in clusters; a new cluster starts
# when the gap between consecutive row indices exceeds this many rows.
_GATHER_GAP = 1024


def string_dtype():
    """Return a ``numpy.dtypes.StringDType`` instance, or raise if unavailable."""
    try:
        return np.dtypes.StringDType()
    except AttributeError:  # pragma: no cover - only on numpy < 2.0
        raise TypeError(
            "utf8 columns require NumPy >= 2.0 (numpy.dtypes.StringDType); "
            f"installed version is {np.__version__}. Use blosc2.vlstring() or "
            "blosc2.string(max_length=...) instead."
        ) from None


def _new_backend_arrays(cparams=None, dparams=None, *, offsets_urlpath=None, data_urlpath=None):
    """Create fresh (offsets, data) NDArrays for an empty utf8 column."""
    import blosc2

    kwargs: dict[str, Any] = {}
    if cparams is not None:
        kwargs["cparams"] = cparams
    if dparams is not None:
        kwargs["dparams"] = dparams
    off_kwargs = dict(kwargs)
    data_kwargs = dict(kwargs)
    if offsets_urlpath is not None:
        off_kwargs["urlpath"] = offsets_urlpath
        off_kwargs["mode"] = "w"
    if data_urlpath is not None:
        data_kwargs["urlpath"] = data_urlpath
        data_kwargs["mode"] = "w"
    offsets = blosc2.zeros((1,), dtype=np.int64, chunks=_OFFSETS_CHUNKS, **off_kwargs)
    data = blosc2.zeros((1,), dtype=np.uint8, chunks=_DATA_CHUNKS, **data_kwargs)
    return offsets, data


class Utf8Array:
    """Row-wise variable-length UTF-8 string array over offsets + bytes NDArrays.

    Provides the row-oriented interface expected by CTable columns:
    ``append``, ``extend``, ``flush``, ``__len__``, ``__getitem__``, and
    ``__setitem__``.  Bulk reads return ``StringDType`` NumPy arrays; single
    reads return ``str``.

    In-place assignment to row ``i`` rewrites the byte blob and offsets of
    every row after ``i`` (a new value usually has a different byte length,
    which shifts all subsequent offsets), so ``__setitem__`` costs O(n - i).

    This class is internal; obtain instances via
    ``storage.create_varlen_scalar_column()`` or
    ``storage.open_varlen_scalar_column()`` with a ``Utf8Spec``.

    Parameters
    ----------
    spec:
        The :class:`~blosc2.schema.Utf8Spec` describing this column.
    offsets:
        ``int64`` NDArray of row offsets (length ``n + 1``).  Created fresh
        (in memory) when ``None``.
    data:
        ``uint8`` NDArray with the concatenated UTF-8 bytes.  Created fresh
        (in memory) when ``None``.
    """

    def __init__(self, spec, offsets=None, data=None) -> None:
        from blosc2.schema import Utf8Spec

        if not isinstance(spec, Utf8Spec):
            raise TypeError(f"Utf8Array requires a Utf8Spec, got {type(spec)!r}")
        self._dtype = string_dtype()
        self._spec = spec
        if (offsets is None) != (data is None):
            raise ValueError("offsets and data must be provided together")
        if offsets is None:
            offsets, data = _new_backend_arrays()
        self._offsets = offsets
        self._data = data
        self._persisted_rows: int = int(offsets.shape[0]) - 1
        # End byte position of the persisted region; resolved lazily because it
        # needs a chunk read from the offsets array.
        self._bytes_used_cache: int | None = 0 if self._persisted_rows == 0 else None
        # Rows not yet flushed to the backing arrays (list of str).
        self._pending: list[str] = []
        self._pending_chars: int = 0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @property
    def _bytes_used(self) -> int:
        if self._bytes_used_cache is None:
            self._bytes_used_cache = int(self._offsets[self._persisted_rows])
        return self._bytes_used_cache

    def _coerce(self, value: Any) -> str:
        """Coerce *value* to ``str``, mapping ``None`` to the null sentinel."""
        if value is None:
            null_value = getattr(self._spec, "null_value", None)
            if null_value is None:
                raise TypeError("Column of utf8 strings is not nullable; received None.")
            return null_value
        if isinstance(value, str):
            return str(value)  # np.str_ instances are normalized to plain str
        raise TypeError(f"Expected str for utf8 column, got {type(value).__name__!r}.")

    def _flush_if_needed(self) -> None:
        if len(self._pending) >= _FLUSH_ROWS or self._pending_chars >= _FLUSH_CHARS:
            self.flush()

    def _read_persisted_span(self, a: int, b: int) -> np.ndarray:
        """Return persisted rows ``[a, b)`` as a StringDType array."""
        n = b - a
        if n <= 0:
            return np.empty(0, dtype=self._dtype)
        offs = np.asarray(self._offsets[a : b + 1], dtype=np.int64)
        start, end = int(offs[0]), int(offs[-1])
        blob = np.asarray(self._data[start:end]).tobytes() if end > start else b""
        rel = offs - start
        out = np.empty(n, dtype=self._dtype)
        for i in range(n):
            out[i] = blob[rel[i] : rel[i + 1]].decode("utf-8")
        return out

    def _gather_persisted(self, indices: np.ndarray) -> np.ndarray:
        """Gather persisted rows at *indices* (any order) as a StringDType array.

        Indices are read in sorted clusters of nearby rows so that a sparse
        gather (e.g. a few head and tail rows for display) does not read the
        whole column.
        """
        out = np.empty(len(indices), dtype=self._dtype)
        if len(indices) == 0:
            return out
        order = np.argsort(indices, kind="stable")
        sorted_idx = indices[order]
        cluster_starts = np.flatnonzero(np.diff(sorted_idx) > _GATHER_GAP) + 1
        pos = 0
        for cluster in np.split(sorted_idx, cluster_starts):
            lo, hi = int(cluster[0]), int(cluster[-1])
            span = self._read_persisted_span(lo, hi + 1)
            out[order[pos : pos + len(cluster)]] = span[cluster - lo]
            pos += len(cluster)
        return out

    def _read_span(self, a: int, b: int) -> np.ndarray:
        """Return rows ``[a, b)`` (persisted + pending) as a StringDType array."""
        np_rows = self._persisted_rows
        if b <= np_rows:
            return self._read_persisted_span(a, b)
        persisted = self._read_persisted_span(a, min(b, np_rows)) if a < np_rows else None
        pending = np.array(self._pending[max(0, a - np_rows) : b - np_rows], dtype=self._dtype)
        if persisted is None:
            return pending
        return np.concatenate([persisted, pending])

    def _get_many(self, indices: np.ndarray) -> np.ndarray:
        indices = np.asarray(indices)
        if indices.ndim != 1:
            indices = indices.ravel()
        n = len(self)
        indices = np.where(indices < 0, indices + n, indices).astype(np.int64, copy=False)
        if len(indices) and (indices.min() < 0 or indices.max() >= n):
            raise IndexError("Utf8Array index out of range")
        out = np.empty(len(indices), dtype=self._dtype)
        np_rows = self._persisted_rows
        pending_mask = indices >= np_rows
        if pending_mask.any():
            out[pending_mask] = [self._pending[i - np_rows] for i in indices[pending_mask]]
        if not pending_mask.all():
            persisted_mask = ~pending_mask
            out[persisted_mask] = self._gather_persisted(indices[persisted_mask])
        return out

    def _rewrite_from(self, pos: int, values: list[str]) -> None:
        """Replace persisted rows ``pos ..`` with *values*, shifting offsets.

        ``pos + len(values)`` becomes the new persisted row count; the byte
        blob and offsets after ``pos`` are rewritten.
        """
        encoded = [v.encode("utf-8") for v in values]
        blob = b"".join(encoded)
        if pos == 0:
            start = 0
        elif pos == self._persisted_rows:
            start = self._bytes_used
        else:
            start = int(self._offsets[pos])
        new_used = start + len(blob)
        new_rows = pos + len(encoded)
        if int(self._data.shape[0]) != max(new_used, 1):
            self._data.resize((max(new_used, 1),))
        if blob:
            self._data[start:new_used] = np.frombuffer(blob, dtype=np.uint8)
        if int(self._offsets.shape[0]) != new_rows + 1:
            self._offsets.resize((new_rows + 1,))
        if encoded:
            lengths = np.fromiter((len(e) for e in encoded), dtype=np.int64, count=len(encoded))
            self._offsets[pos + 1 : new_rows + 1] = start + np.cumsum(lengths)
        self._persisted_rows = new_rows
        self._bytes_used_cache = new_used

    # ------------------------------------------------------------------
    # Public write interface
    # ------------------------------------------------------------------

    def append(self, value: Any) -> None:
        """Append one string row (``None`` maps to the null sentinel)."""
        value = self._coerce(value)
        self._pending.append(value)
        self._pending_chars += len(value)
        self._flush_if_needed()

    def extend(self, values: Iterable[Any]) -> None:
        """Append many string rows."""
        for v in values:
            v = self._coerce(v)
            self._pending.append(v)
            self._pending_chars += len(v)
            self._flush_if_needed()

    def flush(self) -> None:
        """Write pending rows to the backing offsets/data NDArrays."""
        if not self._pending:
            return
        values, self._pending = self._pending, []
        self._pending_chars = 0
        self._rewrite_from(self._persisted_rows, values)

    # ------------------------------------------------------------------
    # Public read interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._persisted_rows + len(self._pending)

    def __iter__(self) -> Iterator[str]:
        yield from self[:]

    def __getitem__(self, index: int | slice | list | tuple | np.ndarray):
        if isinstance(index, (int, np.integer)):
            n = len(self)
            index = int(index)
            if index < 0:
                index += n
            if not (0 <= index < n):
                raise IndexError("Utf8Array index out of range")
            if index >= self._persisted_rows:
                return self._pending[index - self._persisted_rows]
            return str(self._read_persisted_span(index, index + 1)[0])

        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            if step == 1:
                return self._read_span(start, stop)
            return self._get_many(np.arange(start, stop, step, dtype=np.int64))

        if isinstance(index, np.ndarray) and index.dtype == np.bool_:
            if len(index) != len(self):
                raise IndexError(f"Boolean mask length {len(index)} does not match array length {len(self)}")
            return self._get_many(np.flatnonzero(index))

        if isinstance(index, (list, tuple, np.ndarray)):
            return self._get_many(np.asarray(index, dtype=np.int64))

        raise TypeError(f"Utf8Array indices must be int, slice, or array; got {type(index)!r}")

    def __setitem__(self, index: int, value: Any) -> None:
        """Overwrite the value at *index*.

        Because row values have variable byte lengths, overwriting a persisted
        row rewrites the byte blob and offsets of all subsequent rows —
        an O(n - index) operation.
        """
        if not isinstance(index, (int, np.integer)):
            raise TypeError(f"Utf8Array assignment index must be int, got {type(index)!r}")
        value = self._coerce(value)
        n = len(self)
        index = int(index)
        if index < 0:
            index += n
        if not (0 <= index < n):
            raise IndexError("Utf8Array index out of range")
        if index >= self._persisted_rows:
            self._pending[index - self._persisted_rows] = value
            return
        tail = self._read_persisted_span(index + 1, self._persisted_rows)
        self._rewrite_from(index, [value, *tail.tolist()])

    # ------------------------------------------------------------------
    # Properties mirroring the interface expected by CTable
    # ------------------------------------------------------------------

    @property
    def spec(self):
        return self._spec

    @property
    def dtype(self):
        """The ``StringDType`` used for materialized reads."""
        return self._dtype

    @property
    def offsets(self):
        """The underlying ``int64`` NDArray of row offsets (length ``n + 1``)."""
        return self._offsets

    @property
    def data(self):
        """The underlying ``uint8`` NDArray with the concatenated UTF-8 bytes."""
        return self._data

    @property
    def schunk(self):
        return self._offsets.schunk

    @property
    def urlpath(self) -> str | None:
        return getattr(self._offsets, "urlpath", None)

    @property
    def nbytes(self) -> int:
        return self._offsets.schunk.nbytes + self._data.schunk.nbytes

    @property
    def cbytes(self) -> int:
        return self._offsets.schunk.cbytes + self._data.schunk.cbytes

    @property
    def cratio(self) -> float:
        cb = self.cbytes
        if cb == 0:
            return float("inf")
        return self.nbytes / cb

    def copy(self, spec=None, **kwargs: Any) -> Utf8Array:
        """Return an in-memory copy."""
        if spec is None:
            spec = self._spec
        out = Utf8Array(spec)
        out.extend(self)
        out.flush()
        return out
