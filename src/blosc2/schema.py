#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

"""Schema spec primitives and field helper for CTable."""

from __future__ import annotations

import dataclasses
from dataclasses import MISSING
from typing import Any

import numpy as np

BLOSC2_FIELD_METADATA_KEY = "blosc2"

# Aliases so we can still use the builtins inside this module
# after our spec classes shadow them.
_builtin_bool = bool
_builtin_bytes = bytes


# ---------------------------------------------------------------------------
# Base spec class
# ---------------------------------------------------------------------------


class SchemaSpec:
    """Base class for all Blosc2 column schema descriptors.

    Subclasses carry the logical type, storage dtype, and optional
    validation constraints for one column.
    """

    dtype: np.dtype
    python_type: type

    def to_pydantic_kwargs(self) -> dict[str, Any]:
        """Return kwargs for building a Pydantic field annotation."""
        raise NotImplementedError

    def to_metadata_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dict for schema serialization."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Numeric spec classes
# ---------------------------------------------------------------------------

# Internal helper to avoid repeating the constraint boilerplate for every
# integer and float spec.  Subclasses only need to set `dtype`, `python_type`,
# and `_kind` as class attributes.


class _NumericSpec(SchemaSpec):
    """Mixin for numeric specs that support ge / gt / le / lt constraints."""

    _kind: str  # set by each concrete subclass

    def __init__(self, *, ge=None, gt=None, le=None, lt=None):
        self.ge = ge
        self.gt = gt
        self.le = le
        self.lt = lt

    def to_pydantic_kwargs(self) -> dict[str, Any]:
        return {
            k: v
            for k, v in {"ge": self.ge, "gt": self.gt, "le": self.le, "lt": self.lt}.items()
            if v is not None
        }

    def to_metadata_dict(self) -> dict[str, Any]:
        return {"kind": self._kind, **self.to_pydantic_kwargs()}


# ── Signed integers ──────────────────────────────────────────────────────────


class int8(_NumericSpec):
    """8-bit signed integer column  (−128 … 127)."""

    dtype = np.dtype(np.int8)
    python_type = int
    _kind = "int8"


class int16(_NumericSpec):
    """16-bit signed integer column  (−32 768 … 32 767)."""

    dtype = np.dtype(np.int16)
    python_type = int
    _kind = "int16"


class int32(_NumericSpec):
    """32-bit signed integer column  (−2 147 483 648 … 2 147 483 647)."""

    dtype = np.dtype(np.int32)
    python_type = int
    _kind = "int32"


class int64(_NumericSpec):
    """64-bit signed integer column."""

    dtype = np.dtype(np.int64)
    python_type = int
    _kind = "int64"


# ── Unsigned integers ────────────────────────────────────────────────────────


class uint8(_NumericSpec):
    """8-bit unsigned integer column  (0 … 255)."""

    dtype = np.dtype(np.uint8)
    python_type = int
    _kind = "uint8"


class uint16(_NumericSpec):
    """16-bit unsigned integer column  (0 … 65 535)."""

    dtype = np.dtype(np.uint16)
    python_type = int
    _kind = "uint16"


class uint32(_NumericSpec):
    """32-bit unsigned integer column  (0 … 4 294 967 295)."""

    dtype = np.dtype(np.uint32)
    python_type = int
    _kind = "uint32"


class uint64(_NumericSpec):
    """64-bit unsigned integer column."""

    dtype = np.dtype(np.uint64)
    python_type = int
    _kind = "uint64"


# ── Floating point ───────────────────────────────────────────────────────────


class float32(_NumericSpec):
    """32-bit floating-point column (single precision)."""

    dtype = np.dtype(np.float32)
    python_type = float
    _kind = "float32"


class float64(_NumericSpec):
    """64-bit floating-point column (double precision)."""

    dtype = np.dtype(np.float64)
    python_type = float
    _kind = "float64"


class complex64(SchemaSpec):
    """64-bit complex number column (two 32-bit floats)."""

    dtype = np.dtype(np.complex64)
    python_type = complex

    def __init__(self):
        pass

    def to_pydantic_kwargs(self) -> dict[str, Any]:
        return {}

    def to_metadata_dict(self) -> dict[str, Any]:
        return {"kind": "complex64"}


class complex128(SchemaSpec):
    """128-bit complex number column (two 64-bit floats)."""

    dtype = np.dtype(np.complex128)
    python_type = complex

    def __init__(self):
        pass

    def to_pydantic_kwargs(self) -> dict[str, Any]:
        return {}

    def to_metadata_dict(self) -> dict[str, Any]:
        return {"kind": "complex128"}


class bool(SchemaSpec):
    """Boolean column."""

    dtype = np.dtype(np.bool_)
    python_type = _builtin_bool

    def __init__(self):
        pass

    def to_pydantic_kwargs(self) -> dict[str, Any]:
        return {}

    def to_metadata_dict(self) -> dict[str, Any]:
        return {"kind": "bool"}


# ---------------------------------------------------------------------------
# String / bytes spec classes
# ---------------------------------------------------------------------------


class string(SchemaSpec):
    """Fixed-width Unicode string column.

    Parameters
    ----------
    max_length:
        Maximum number of characters.  Determines the NumPy ``U<n>`` dtype.
        Defaults to 32 if not specified.
    min_length:
        Minimum number of characters (validation only, no effect on dtype).
    pattern:
        Regex pattern the value must match (validation only).
    """

    python_type = str
    _DEFAULT_MAX_LENGTH = 32

    def __init__(self, *, min_length=None, max_length=None, pattern=None):
        self.min_length = min_length
        self.max_length = max_length if max_length is not None else self._DEFAULT_MAX_LENGTH
        self.pattern = pattern
        self.dtype = np.dtype(f"U{self.max_length}")

    def to_pydantic_kwargs(self) -> dict[str, Any]:
        d = {}
        if self.min_length is not None:
            d["min_length"] = self.min_length
        if self.max_length is not None:
            d["max_length"] = self.max_length
        if self.pattern is not None:
            d["pattern"] = self.pattern
        return d

    def to_metadata_dict(self) -> dict[str, Any]:
        return {"kind": "string", **self.to_pydantic_kwargs()}


class bytes(SchemaSpec):
    """Fixed-width bytes column.

    Parameters
    ----------
    max_length:
        Maximum number of bytes.  Determines the NumPy ``S<n>`` dtype.
        Defaults to 32 if not specified.
    min_length:
        Minimum number of bytes (validation only, no effect on dtype).
    """

    python_type = _builtin_bytes
    _DEFAULT_MAX_LENGTH = 32

    def __init__(self, *, min_length=None, max_length=None):
        self.min_length = min_length
        self.max_length = max_length if max_length is not None else self._DEFAULT_MAX_LENGTH
        self.dtype = np.dtype(f"S{self.max_length}")

    def to_pydantic_kwargs(self) -> dict[str, Any]:
        d = {}
        if self.min_length is not None:
            d["min_length"] = self.min_length
        if self.max_length is not None:
            d["max_length"] = self.max_length
        return d

    def to_metadata_dict(self) -> dict[str, Any]:
        return {"kind": "bytes", **self.to_pydantic_kwargs()}


# ---------------------------------------------------------------------------
# Field helper
# ---------------------------------------------------------------------------


def field(
    spec: SchemaSpec,
    *,
    default=MISSING,
    cparams: dict[str, Any] | None = None,
    dparams: dict[str, Any] | None = None,
    chunks: tuple[int, ...] | None = None,
    blocks: tuple[int, ...] | None = None,
) -> dataclasses.Field:
    """Attach a Blosc2 schema spec and per-column storage options to a dataclass field.

    Parameters
    ----------
    spec:
        A schema descriptor such as ``b2.int64(ge=0)`` or ``b2.float64()``.
    default:
        Default value for the field.  Omit for required fields.
    cparams:
        Compression parameters for this column's NDArray.
    dparams:
        Decompression parameters for this column's NDArray.
    chunks:
        Chunk shape for this column's NDArray.
    blocks:
        Block shape for this column's NDArray.

    Examples
    --------
    >>> from dataclasses import dataclass
    >>> import blosc2 as b2
    >>> @dataclass
    ... class Row:
    ...     id: int = b2.field(b2.int64(ge=0))
    ...     score: float = b2.field(b2.float64(ge=0, le=100))
    ...     active: bool = b2.field(b2.bool(), default=True)
    """
    if not isinstance(spec, SchemaSpec):
        raise TypeError(f"field() requires a SchemaSpec as its first argument, got {type(spec)!r}.")

    metadata = {
        BLOSC2_FIELD_METADATA_KEY: {
            "spec": spec,
            "cparams": cparams,
            "dparams": dparams,
            "chunks": chunks,
            "blocks": blocks,
        }
    }
    if default is MISSING:
        return dataclasses.field(metadata=metadata)
    return dataclasses.field(default=default, metadata=metadata)
