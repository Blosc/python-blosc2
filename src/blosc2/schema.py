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
_builtin_list = list
_builtin_object = object


# ---------------------------------------------------------------------------
# Base spec class
# ---------------------------------------------------------------------------


class SchemaSpec:
    """Base class for all Blosc2 column schema descriptors.

    Subclasses carry the logical type, storage dtype, and optional
    validation constraints for one column.

    Numpy dtype attributes (``itemsize``, ``kind``, ``type``, ``str``,
    ``name``) are mirrored at class level so that schema spec classes can
    be used anywhere blosc2 internals expect a dtype-like object.
    """

    dtype: np.dtype
    python_type: type

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Mirror numpy dtype attributes at class level for duck-typing.
        _np_dtype = cls.__dict__.get("dtype")
        if isinstance(_np_dtype, np.dtype):
            cls.itemsize = _np_dtype.itemsize
            cls.kind = _np_dtype.kind
            cls.type = _np_dtype.type
            cls.str = _np_dtype.str
            cls.name = _np_dtype.name

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
    """Mixin for numeric specs that support constraints and null sentinels.

    ``nullable=True`` asks CTable to choose a null sentinel from the current
    null policy when the schema is compiled.  An explicit ``null_value`` takes
    precedence.
    """

    _kind: str  # set by each concrete subclass

    def __init__(self, *, ge=None, gt=None, le=None, lt=None, nullable: bool = False, null_value=None):
        self.ge = ge
        self.gt = gt
        self.le = le
        self.lt = lt
        self.nullable = nullable or null_value is not None
        self.null_value = null_value

    def to_pydantic_kwargs(self) -> dict[str, Any]:
        # null_value is not a Pydantic constraint — exclude it from Pydantic kwargs.
        return {
            k: v
            for k, v in {"ge": self.ge, "gt": self.gt, "le": self.le, "lt": self.lt}.items()
            if v is not None
        }

    def to_metadata_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"kind": self._kind, **self.to_pydantic_kwargs()}
        if self.nullable:
            d["nullable"] = True
        if self.null_value is not None:
            d["null_value"] = self.null_value
        return d


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


class timestamp(SchemaSpec):
    """Timestamp column stored as signed 64-bit epoch offsets.

    The physical storage dtype is ``int64``.  ``unit`` follows Arrow/NumPy
    datetime units: ``"s"``, ``"ms"``, ``"us"`` or ``"ns"``.  ``timezone``
    is metadata preserved for Arrow/Parquet roundtrips.
    """

    dtype = np.dtype(np.int64)
    python_type = _builtin_object

    def __init__(
        self, *, unit: str = "us", timezone: str | None = None, nullable: bool = False, null_value=None
    ):
        if unit not in {"s", "ms", "us", "ns"}:
            raise ValueError("timestamp unit must be one of: 's', 'ms', 'us', 'ns'")
        self.unit = unit
        self.timezone = timezone
        self.nullable = nullable or null_value is not None
        self.null_value = null_value

    def to_pydantic_kwargs(self) -> dict[str, Any]:
        return {}

    def to_metadata_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"kind": "timestamp", "unit": self.unit}
        if self.timezone is not None:
            d["timezone"] = self.timezone
        if self.nullable:
            d["nullable"] = True
        if self.null_value is not None:
            d["null_value"] = self.null_value
        return d


class bool(SchemaSpec):
    """Boolean column.

    Nullable bool columns use uint8 physical storage with values
    ``0`` (false), ``1`` (true), and ``255`` (null).
    """

    dtype = np.dtype(np.bool_)
    python_type = _builtin_bool

    def __init__(self, *, nullable: bool = False, null_value=None):
        if null_value is not None and null_value != 255:
            raise ValueError("Nullable bool null_value must be 255")
        self.nullable = nullable or null_value is not None
        self.null_value = null_value
        self.dtype = np.dtype(np.uint8) if self.nullable else np.dtype(np.bool_)

    def to_pydantic_kwargs(self) -> dict[str, Any]:
        return {}

    def to_metadata_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"kind": "bool"}
        if self.nullable:
            d["nullable"] = True
            d["null_value"] = self.null_value
        return d


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
    nullable:
        If ``True`` and ``null_value`` is not set, choose a null sentinel from
        the current CTable null policy when the schema is compiled.
    null_value:
        Explicit null sentinel.  Takes precedence over ``nullable=True``.
    """

    python_type = str
    _DEFAULT_MAX_LENGTH = 32

    def __init__(
        self, *, min_length=None, max_length=None, pattern=None, nullable: bool = False, null_value=None
    ):
        self.min_length = min_length
        self.max_length = max_length if max_length is not None else self._DEFAULT_MAX_LENGTH
        self.pattern = pattern
        self.nullable = nullable or null_value is not None
        self.null_value = null_value
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
        d: dict[str, Any] = {"kind": "string", **self.to_pydantic_kwargs()}
        if self.nullable:
            d["nullable"] = True
        if self.null_value is not None:
            d["null_value"] = self.null_value
        return d


class bytes(SchemaSpec):
    """Fixed-width bytes column.

    Parameters
    ----------
    max_length:
        Maximum number of bytes.  Determines the NumPy ``S<n>`` dtype.
        Defaults to 32 if not specified.
    min_length:
        Minimum number of bytes (validation only, no effect on dtype).
    nullable:
        If ``True`` and ``null_value`` is not set, choose a null sentinel from
        the current CTable null policy when the schema is compiled.
    null_value:
        Explicit null sentinel.  Takes precedence over ``nullable=True``.
    """

    python_type = _builtin_bytes
    _DEFAULT_MAX_LENGTH = 32

    def __init__(self, *, min_length=None, max_length=None, nullable: bool = False, null_value=None):
        self.min_length = min_length
        self.max_length = max_length if max_length is not None else self._DEFAULT_MAX_LENGTH
        self.nullable = nullable or null_value is not None
        self.null_value = null_value
        self.dtype = np.dtype(f"S{self.max_length}")

    def to_pydantic_kwargs(self) -> dict[str, Any]:
        d = {}
        if self.min_length is not None:
            d["min_length"] = self.min_length
        if self.max_length is not None:
            d["max_length"] = self.max_length
        return d

    def to_metadata_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"kind": "bytes", **self.to_pydantic_kwargs()}
        if self.nullable:
            d["nullable"] = True
        if self.null_value is not None:
            d["null_value"] = self.null_value
        return d


# ---------------------------------------------------------------------------
# List spec
# ---------------------------------------------------------------------------


class StructSpec(SchemaSpec):
    """Logical schema descriptor for dict-like structured values.

    Top-level CTable struct columns are stored as row-wise dictionaries in a
    batched variable-length backend.  Struct specs can also be used as
    :func:`list` item specs for Arrow ``list<struct<...>>`` columns.
    """

    python_type = dict
    dtype = None

    def __init__(self, fields: dict[str, SchemaSpec], *, nullable: bool = False):
        if not isinstance(fields, dict) or not fields:
            raise TypeError("StructSpec fields must be a non-empty dict")
        for name, spec in fields.items():
            if not isinstance(name, str):
                raise TypeError("StructSpec field names must be strings")
            if not isinstance(spec, SchemaSpec):
                raise TypeError("StructSpec field values must be SchemaSpec instances")
        self.fields = dict(fields)
        self.nullable = nullable

    def to_pydantic_kwargs(self) -> dict[str, Any]:
        return {}

    def to_metadata_dict(self) -> dict[str, Any]:
        return {
            "kind": "struct",
            "fields": [{"name": name, **spec.to_metadata_dict()} for name, spec in self.fields.items()],
            "nullable": self.nullable,
        }

    def display_label(self) -> str:
        return "struct[" + ", ".join(self.fields) + "]"

    @classmethod
    def from_metadata_dict(cls, data: dict[str, Any]) -> StructSpec:
        from blosc2.schema_compiler import spec_from_metadata_dict

        fields = {}
        for field in data["fields"]:
            field = dict(field)
            name = field.pop("name")
            fields[name] = spec_from_metadata_dict(field)
        return cls(fields, nullable=data.get("nullable", False))


class ListSpec(SchemaSpec):
    """Logical schema descriptor for a list-valued column."""

    python_type = _builtin_list
    dtype = None

    def __init__(
        self,
        item_spec: SchemaSpec,
        *,
        nullable: bool = False,
        storage: str = "batch",
        serializer: str = "msgpack",
        batch_rows: int | None = None,
        items_per_block: int | None = None,
    ):
        if not isinstance(item_spec, SchemaSpec):
            raise TypeError("ListSpec item_spec must be a SchemaSpec instance")
        if isinstance(item_spec, ListSpec):
            raise TypeError("Nested list item specs are not supported in V1")
        if storage not in {"batch", "vl"}:
            raise ValueError("storage must be 'batch' or 'vl'")
        if serializer not in {"msgpack", "arrow"}:
            raise ValueError("serializer must be 'msgpack' or 'arrow'")
        if storage == "vl" and serializer != "msgpack":
            raise ValueError("storage='vl' only supports serializer='msgpack'")
        if serializer == "arrow" and storage != "batch":
            raise ValueError("serializer='arrow' requires storage='batch'")
        self.item_spec = item_spec
        self.nullable = nullable
        self.storage = storage
        self.serializer = serializer
        self.batch_rows = batch_rows
        self.items_per_block = items_per_block

    def to_pydantic_kwargs(self) -> dict[str, Any]:
        return {}

    def to_metadata_dict(self) -> dict[str, Any]:
        d = {
            "kind": "list",
            "item": self.item_spec.to_metadata_dict(),
            "nullable": self.nullable,
            "storage": self.storage,
            "serializer": self.serializer,
        }
        if self.batch_rows is not None:
            d["batch_rows"] = self.batch_rows
        if self.items_per_block is not None:
            d["items_per_block"] = self.items_per_block
        return d

    def to_listarray_metadata(self) -> dict[str, Any]:
        d = {"version": 1, **self.to_metadata_dict()}
        d["backend"] = d.pop("storage")
        return d

    def display_label(self) -> str:
        item_kind = self.item_spec.to_metadata_dict().get("kind", type(self.item_spec).__name__)
        return f"list[{item_kind}]"

    @classmethod
    def from_metadata_dict(cls, data: dict[str, Any]) -> ListSpec:
        from blosc2.schema_compiler import spec_from_metadata_dict

        backend = data.get("backend")
        return cls(
            spec_from_metadata_dict(data["item_spec"] if "item_spec" in data else data["item"]),
            nullable=data.get("nullable", False),
            storage=backend if backend is not None else data.get("storage", "batch"),
            serializer=data.get("serializer", "msgpack"),
            batch_rows=data.get("batch_rows"),
            items_per_block=data.get("items_per_block"),
        )


# ---------------------------------------------------------------------------
# Variable-length scalar spec classes
# ---------------------------------------------------------------------------


class VLStringSpec(SchemaSpec):
    """Variable-length scalar string column backed by batched object storage.

    Unlike :class:`string`, this spec does not use a fixed-width NumPy dtype.
    Each row value is a plain Python ``str`` (or ``None`` when nullable).
    Physical storage uses batched msgpack serialization via
    :class:`blosc2.BatchArray` internally.

    Parameters
    ----------
    nullable:
        If ``True``, ``None`` is a valid row value representing a missing entry.
        Nullability is represented natively — no sentinel value is used.
    serializer:
        Serialization backend.  Currently only ``"msgpack"`` is supported.
    batch_rows:
        Target number of rows per storage batch.  Defaults to 2048.
    items_per_block:
        Optional items-per-block hint passed to the underlying BatchArray.
    """

    python_type = str
    dtype = None

    def __init__(
        self,
        *,
        nullable: bool = False,
        serializer: str = "msgpack",
        batch_rows: int | None = 2048,
        items_per_block: int | None = None,
    ):
        if serializer != "msgpack":
            raise ValueError("vlstring currently only supports serializer='msgpack'")
        if batch_rows is not None and batch_rows <= 0:
            raise ValueError("batch_rows must be positive or None")
        if items_per_block is not None and items_per_block <= 0:
            raise ValueError("items_per_block must be positive or None")
        self.nullable = nullable
        self.serializer = serializer
        self.batch_rows = batch_rows
        self.items_per_block = items_per_block

    def to_pydantic_kwargs(self) -> dict[str, Any]:
        return {}

    def to_metadata_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "kind": "vlstring",
            "nullable": self.nullable,
            "serializer": self.serializer,
        }
        if self.batch_rows is not None:
            d["batch_rows"] = self.batch_rows
        if self.items_per_block is not None:
            d["items_per_block"] = self.items_per_block
        return d


class ObjectSpec(SchemaSpec):
    """Schema-less Python object column backed by batched msgpack storage.

    Each row value can be any msgpack-serializable Python object, or ``None``
    when *nullable* is true.  Use this for heterogeneous per-row payloads when
    a typed :func:`struct`, :func:`list`, :func:`vlstring`, or :func:`vlbytes`
    schema would not describe the data.
    """

    python_type = _builtin_object
    dtype = None

    def __init__(
        self,
        *,
        nullable: bool = False,
        serializer: str = "msgpack",
        batch_rows: int | None = 2048,
        items_per_block: int | None = None,
    ):
        if serializer != "msgpack":
            raise ValueError("object currently only supports serializer='msgpack'")
        if batch_rows is not None and batch_rows <= 0:
            raise ValueError("batch_rows must be positive or None")
        if items_per_block is not None and items_per_block <= 0:
            raise ValueError("items_per_block must be positive or None")
        self.nullable = nullable
        self.serializer = serializer
        self.batch_rows = batch_rows
        self.items_per_block = items_per_block

    def to_pydantic_kwargs(self) -> dict[str, Any]:
        return {}

    def to_metadata_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "kind": "object",
            "nullable": self.nullable,
            "serializer": self.serializer,
        }
        if self.batch_rows is not None:
            d["batch_rows"] = self.batch_rows
        if self.items_per_block is not None:
            d["items_per_block"] = self.items_per_block
        return d

    def display_label(self) -> str:
        return "object"


class VLBytesSpec(SchemaSpec):
    """Variable-length scalar bytes column backed by batched object storage.

    Unlike :class:`bytes`, this spec does not use a fixed-width NumPy dtype.
    Each row value is a plain Python ``bytes`` (or ``None`` when nullable).
    Physical storage uses batched msgpack serialization via
    :class:`blosc2.BatchArray` internally.

    Parameters
    ----------
    nullable:
        If ``True``, ``None`` is a valid row value representing a missing entry.
        Nullability is represented natively — no sentinel value is used.
    serializer:
        Serialization backend.  Currently only ``"msgpack"`` is supported.
    batch_rows:
        Target number of rows per storage batch.  Defaults to 2048.
    items_per_block:
        Optional items-per-block hint passed to the underlying BatchArray.
    """

    python_type = _builtin_bytes
    dtype = None

    def __init__(
        self,
        *,
        nullable: bool = False,
        serializer: str = "msgpack",
        batch_rows: int | None = 2048,
        items_per_block: int | None = None,
    ):
        if serializer != "msgpack":
            raise ValueError("vlbytes currently only supports serializer='msgpack'")
        if batch_rows is not None and batch_rows <= 0:
            raise ValueError("batch_rows must be positive or None")
        if items_per_block is not None and items_per_block <= 0:
            raise ValueError("items_per_block must be positive or None")
        self.nullable = nullable
        self.serializer = serializer
        self.batch_rows = batch_rows
        self.items_per_block = items_per_block

    def to_pydantic_kwargs(self) -> dict[str, Any]:
        return {}

    def to_metadata_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "kind": "vlbytes",
            "nullable": self.nullable,
            "serializer": self.serializer,
        }
        if self.batch_rows is not None:
            d["batch_rows"] = self.batch_rows
        if self.items_per_block is not None:
            d["items_per_block"] = self.items_per_block
        return d


def vlstring(
    *,
    nullable: bool = False,
    serializer: str = "msgpack",
    batch_rows: int | None = 2048,
    items_per_block: int | None = None,
) -> VLStringSpec:
    """Build a variable-length scalar string schema descriptor.

    Use this as an explicit opt-in when a CTable column holds long or
    wildly variable-length strings that would waste space in a fixed-width
    ``string(max_length=N)`` column.  Must be requested via
    ``blosc2.field(blosc2.vlstring())`` — it is never inferred automatically
    from plain ``str`` annotations.
    """
    return VLStringSpec(
        nullable=nullable,
        serializer=serializer,
        batch_rows=batch_rows,
        items_per_block=items_per_block,
    )


def object(
    *,
    nullable: bool = False,
    serializer: str = "msgpack",
    batch_rows: int | None = 2048,
    items_per_block: int | None = None,
) -> ObjectSpec:
    """Build a schema-less Python object column descriptor for CTable.

    Values are stored via batched msgpack serialization.  Prefer typed specs
    such as :func:`struct`, :func:`list`, :func:`vlstring`, or :func:`vlbytes`
    when the data has a stable schema; use ``object`` for heterogeneous per-row
    payloads.
    """
    return ObjectSpec(
        nullable=nullable,
        serializer=serializer,
        batch_rows=batch_rows,
        items_per_block=items_per_block,
    )


def vlbytes(
    *,
    nullable: bool = False,
    serializer: str = "msgpack",
    batch_rows: int | None = 2048,
    items_per_block: int | None = None,
) -> VLBytesSpec:
    """Build a variable-length scalar bytes schema descriptor.

    Use this as an explicit opt-in when a CTable column holds long or
    wildly variable-length byte strings.  Must be requested via
    ``blosc2.field(blosc2.vlbytes())`` — it is never inferred automatically
    from plain ``bytes`` annotations.
    """
    return VLBytesSpec(
        nullable=nullable,
        serializer=serializer,
        batch_rows=batch_rows,
        items_per_block=items_per_block,
    )


def struct(fields: dict[str, SchemaSpec], *, nullable: bool = False) -> StructSpec:
    """Build a structured schema descriptor for dict-like CTable values.

    Top-level struct columns store one dictionary (or ``None`` when nullable)
    per row.  Struct specs may also be nested as list item specs.
    """
    return StructSpec(fields, nullable=nullable)


def list(
    item_spec: SchemaSpec,
    *,
    nullable: bool = False,
    storage: str = "batch",
    serializer: str = "msgpack",
    batch_rows: int | None = None,
    items_per_block: int | None = None,
) -> ListSpec:
    """Build a list-valued schema descriptor for CTable and ListArray."""
    return ListSpec(
        item_spec,
        nullable=nullable,
        storage=storage,
        serializer=serializer,
        batch_rows=batch_rows,
        items_per_block=items_per_block,
    )


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
