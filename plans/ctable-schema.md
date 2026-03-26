# CTable Schema Redesign

## Motivation

The current `CTable` prototype in PR #598 uses `pydantic.BaseModel` plus
`Annotated[...]` metadata to define table schemas.  That works, but it is not the
best long-term API for a columnar container in `python-blosc2`.

The main issues with the current shape are:

* It mixes row validation concerns with physical storage concerns.
* It relies on custom metadata objects (`NumpyDtype`, `MaxLen`) embedded in
  Pydantic annotations.
* It is verbose for simple schemas.
* It does not provide an obvious place for NDArray-specific per-column options
  such as `cparams`, `dparams`, `chunks`, `blocks`, or future indexing hints.

What we want instead is:

* A schema API that is easy to read and write.
* A place to attach Blosc2-specific per-column configuration.
* A way to express logical constraints such as `ge=0`, `le=100`, `max_length=10`.
* Internal validation without forcing the public API to be Pydantic-shaped.
* A clean distinction between:
  * logical field type and constraints
  * physical storage type
  * per-column storage options

The proposed solution is a **dataclass-first schema API** with **declarative field
spec objects** and **optional internal Pydantic-backed validation**.

The intended usage style is:

* canonical form for constrained or storage-tuned columns:
  `id: int = b2.field(b2.int64(ge=0))`
* shorthand for simple inferred columns:
  `id: int`
* not preferred as a primary style:
  `id = b2.field(b2.int64(ge=0))`

The reason is that the canonical form preserves normal Python type annotations,
which are valuable for readability, static tooling, and schema inspection.

---

## Proposed public API

### Schema declaration

The intended schema declaration style is:

```python
from dataclasses import dataclass

import blosc2 as b2


@dataclass
class Row:
    id: int = b2.field(b2.int64(ge=0))
    score: float = b2.field(
        b2.float64(ge=0, le=100),
        cparams={"codec": b2.Codec.LZ4, "clevel": 5},
    )
    active: bool = b2.field(b2.bool(), default=True)
```

This is the target user-facing API for `CTable`.

This should be documented as the **canonical** schema declaration style.

For simple unconstrained cases, `CTable` may support an inferred shorthand:

```python
@dataclass
class Row:
    id: int
    score: float
    active: bool = True
```

which is interpreted approximately as:

```python
@dataclass
class Row:
    id: int = b2.field(b2.int64())
    score: float = b2.field(b2.float64())
    active: bool = b2.field(b2.bool(), default=True)
```

This shorthand should be limited to simple built-in Python types where the
mapping is obvious.

### Naming convention

Use **lowercase names** for schema descriptor objects:

* `b2.int64`
* `b2.float64`
* `b2.bool`
* later: `b2.string(max_length=...)`, `b2.bytes(max_length=...)`, `b2.complex128`

Reason:

* `b2.int64(...)` is not just a dtype; it is a schema descriptor with constraints.
* The lowercase form keeps the API closer in spirit to NumPy and PyTorch.
* If plain NumPy dtypes are needed, callers can use `np.int64`, `np.float64`,
  `np.bool_`, etc.
* `b2.bool(...)` is preferred over `b2.bool_(...)` for readability, even though
  NumPy uses `bool_`.  This is closer to PyTorch style and fits better for a
  schema-builder API.

### Field helper

`b2.field(...)` should be the standard way to attach schema metadata to a
dataclass field.

Expected shape:

```python
b2.field(
    b2.float64(ge=0, le=100),
    default=...,
    default_factory=...,
    cparams=...,
    dparams=...,
    chunks=...,
    blocks=...,
    title=...,
    description=...,
    nullable=...,
)
```

At minimum for the first version:

* `spec`
* `default`
* `default_factory`
* `cparams`
* `dparams`
* `chunks`
* `blocks`

The implementation should store these in `dataclasses.field(metadata=...)`.

The unannotated form:

```python
id = b2.field(b2.int64(ge=0))
```

should not be the primary API.  It may be supported later only if there is a
strong reason, but the preferred style should retain:

* a Python type annotation in the annotation slot
* `b2.field(...)` in the field/default slot

That keeps the schema aligned with normal dataclass usage.

---

## Core design

### 1. Dataclass is the schema carrier

The dataclass defines:

* field names
* Python-level row shape
* user-visible defaults

Example:

```python
@dataclass
class Row:
    id: int = b2.field(b2.int64(ge=0))
    score: float = b2.field(b2.float64(ge=0, le=100))
    active: bool = b2.field(b2.bool(), default=True)
```

This keeps the declaration small and idiomatic.

The Python annotation should remain part of the design, not be replaced by
`b2.field(...)` alone.  The annotation provides value independently of the
Blosc2 schema descriptor.

### 2. Schema spec objects are the source of truth

Each lowercase builder object is a lightweight immutable schema descriptor.

Examples:

```python
b2.int64(ge=0)
b2.float64(ge=0, le=100)
b2.bool()
b2.string(max_length=32)
b2.bytes(max_length=64)
```

Each spec object should carry only schema-level metadata, for example:

* logical kind
* storage dtype
* numeric constraints (`ge`, `gt`, `le`, `lt`, `multiple_of`)
* string constraints (`max_length`, `min_length`, `pattern`)
* nullability
* maybe logical annotations later (`categorical`, `timezone`, `unit`)

They should **not** directly carry per-column NDArray instance settings such as
`cparams` or `chunks`; those belong in `b2.field(...)`.

### 3. Column field metadata carries NDArray-specific configuration

`b2.field(...)` metadata should be the place for:

* column storage options
* per-column compression settings
* chunk/block tuning
* persistence options in future versions

This keeps the separation clean:

* `b2.float64(ge=0, le=100)` answers: "what values are valid?"
* `b2.field(..., cparams=..., chunks=...)` answers: "how is this column stored?"

### 4. Schema compilation step inside CTable

`CTable` should not consume raw dataclass fields repeatedly.  On construction, it
should compile the row class into an internal schema representation.

For example:

```python
compiled = CompiledSchema(
    row_cls=Row,
    columns=[
        CompiledColumn(
            name="id",
            py_type=int,
            spec=b2.int64(ge=0),
            dtype=np.int64,
            default=MISSING,
            cparams=...,
            dparams=...,
            chunks=...,
            blocks=...,
            validator_info=...,
        ),
        ...,
    ],
    validator_model=...,
)
```

This compiled form should drive:

* NDArray creation
* row validation
* bulk validation
* introspection and future serialization

---

## Validation strategy

### Use Pydantic internally, but do not make it the public schema API

Pydantic is a good fit for validation because it is:

* mature
* well-tested
* expressive
* fast enough for row-level operations

However, it should be an **implementation detail**, not the public schema surface.

The public schema should remain:

* dataclass-based
* Blosc2-specific
* independent of any one validation library

### Why not use Pydantic as the schema source directly?

Because storage and validation are overlapping but not identical concerns.

Examples:

* `dtype=np.int16` is both logical and physical.
* `cparams`, `chunks`, `blocks`, `dparams` are not Pydantic concepts.
* a future column index, bloom filter, or codec hint is not a validation concept.

Therefore, the internal architecture should be:

* user declares a dataclass + `b2.field(...)`
* `CTable` compiles it into:
  * storage schema
  * validation schema

### Row-level validation

For `append(row)` and other row-wise inserts:

* compile a cached internal Pydantic model once per schema
* validate incoming rows against that model
* convert the validated row into column values

This is the simplest and safest path.

Expected behavior:

* `table.append(Row(...))`
* `table.append({"id": 1, "score": 2.0, "active": True})`
* `table.append((1, 2.0, True))`

All may be accepted, but internally normalized through one validator path.

### Bulk validation

For `extend(...)`, row-by-row Pydantic validation may be too expensive for large
batches.  Bulk inserts need a separate strategy.

Recommended modes:

* `validate=True`
  Full validation.  May use row-wise Pydantic validation for smaller inputs and
  vectorized checks where available.
* `validate=False`
  Trust caller, perform dtype coercion only.
* optional later: `validate="sample"` or `validate="vectorized"`

For numeric and simple string constraints, vectorized checks are preferable when
possible:

* `ge`, `gt`, `le`, `lt`
* `max_length`, `min_length`
* null checks
* dtype coercion checks

This means the architecture should support both:

* Pydantic row validation
* vectorized array validation

The compiled schema should expose enough information for both.

### Performance stance

Pydantic should be treated as:

* a strong default for correctness
* fast enough for row-wise validation
* not necessarily the fastest choice for large batch validation

This is important because the performance bottleneck for `extend()` is more about
per-row Python overhead than about Pydantic specifically.

---

## Detailed API proposal

### Schema spec classes

Add schema descriptor classes under `blosc2`, for example:

* `int8`, `int16`, `int32`, `int64`
* `uint8`, `uint16`, `uint32`, `uint64`
* `float32`, `float64`
* `bool`
* `complex64`, `complex128`
* `string`
* `bytes`

Minimal constructor examples:

```python
b2.int64(ge=0)
b2.float64(ge=0, le=100)
b2.string(max_length=32)
b2.bytes(max_length=64)
b2.bool()
```

Internal common fields:

* `dtype`
* `nullable`
* `constraints`
* `python_type`

### Field helper

`b2.field(spec, **kwargs)` should return a `dataclasses.field(...)` object with
Blosc2 metadata attached.

Example metadata layout:

```python
{
    "blosc2": {
        "spec": ...,
        "cparams": ...,
        "dparams": ...,
        "chunks": ...,
        "blocks": ...,
    }
}
```

This metadata key should be stable and reserved.

### CTable constructor

The desired constructor remains:

```python
table = b2.CTable(Row)
```

Optional overrides:

```python
table = b2.CTable(
    Row,
    expected_size=1_000_000,
    compact=False,
    validate=True,
)
```

`CTable` should detect that `Row` is a dataclass schema and compile it.

### Possible compatibility layer

If needed temporarily, `CTable` may continue accepting the old Pydantic model
style during a transition period:

```python
table = b2.CTable(LegacyPydanticRow)
```

But that should be documented as legacy or transitional once the dataclass API
lands.

---

## Internal compilation pipeline

### Step 1. Inspect dataclass fields

For each dataclass field:

* field name
* Python annotation
* default or default factory
* Blosc2 metadata from `b2.field(...)`

Reject invalid shapes early:

* missing `b2.field(...)`
* missing schema spec
* incompatible Python annotation vs schema spec
* unsupported defaults

If inferred shorthand is supported, refine the first two rules to:

* either a supported plain annotation, or an explicit `b2.field(...)`
* if `b2.field(...)` is present, it must contain a schema spec

### Step 2. Build compiled column descriptors

For each field, produce a `CompiledColumn` object containing:

* `name`
* `py_type`
* `spec`
* `dtype`
* `default`
* `default_factory`
* `nullable`
* `cparams`
* `dparams`
* `chunks`
* `blocks`
* validation constraints

### Step 3. Derive physical NDArray creation arguments

From the compiled column descriptor, derive:

* `dtype`
* shape
* chunks
* blocks
* `cparams`
* `dparams`

This should happen once during table initialization.

### Step 4. Derive validation model

Translate each schema spec into a Pydantic field definition.

Examples:

* `int64(ge=0)` -> integer field with `ge=0`
* `float64(ge=0, le=100)` -> float field with `ge=0`, `le=100`
* `string(max_length=32)` -> string field with `max_length=32`

Cache the compiled Pydantic model class per row schema.

### Step 5. Expose introspection hooks

Expose enough metadata for:

* debugging
* `table.info()`
* future schema serialization
* future schema-driven docs and reprs

Possible user-facing hooks later:

* `table.schema`
* `table.schema.columns`
* `table.schema.as_dict()`

---

## Handling defaults

Defaults should follow dataclass semantics as closely as possible.

Examples:

```python
active: bool = b2.field(b2.bool(), default=True)
tags: list[str] = b2.field(..., default_factory=list)
```

For the first implementation, keep this conservative:

* support scalar defaults
* support `default_factory` only if there is a clear use case
* reject mutable defaults directly

On insert:

* omitted values should be filled from defaults
* explicit `None` should be accepted only if the field is nullable

---

## Insert semantics

### append()

`append()` should accept a small set of normalized shapes:

* dataclass row instance
* dict-like row
* tuple/list in schema order

Recommended internal path:

1. normalize the input to a field mapping
2. validate with cached validator model
3. coerce to final column values
4. append into underlying NDArrays

### extend()

`extend()` should accept:

* iterable of row objects
* dict-of-arrays
* structured NumPy array
* maybe another `CTable`

Recommended internal path:

1. normalize to column batches where possible
2. validate according to `validate=` mode
3. coerce dtypes
4. write in bulk

For `dict-of-arrays` and structured arrays, vectorized validation should be the
preferred long-term path.

---

## Per-column NDArray options

One of the main reasons for `b2.field(...)` is that different columns may want
different storage settings.

Examples:

* a boolean column may want different compression parameters from a float column
* a high-cardinality string column may need different chunk sizes
* a metric column may use a specific codec or filter tuning

So the schema system must allow:

```python
@dataclass
class Row:
    id: int = b2.field(b2.int64(ge=0), cparams={"codec": b2.Codec.ZSTD, "clevel": 1})
    score: float = b2.field(
        b2.float64(ge=0, le=100), cparams={"codec": b2.Codec.LZ4HC, "clevel": 9}
    )
    active: bool = b2.field(b2.bool(), cparams={"codec": b2.Codec.LZ4})
```

The implementation should define precedence rules clearly:

* column-level options override table defaults
* table-level options fill in unspecified values

This implies `CTable(...)` may also take default storage options:

```python
table = b2.CTable(Row, cparams=..., dparams=...)
```

Column-level overrides should merge against those defaults, not replace them
blindly.

---

## Compatibility and migration

### Goal

Move toward the dataclass-based schema API without locking the project into the
current Pydantic-shaped declaration model.

### Migration path

Phase 1:

* introduce schema spec classes and `b2.field(...)`
* support dataclass schemas in `CTable`
* keep existing prototype behavior separate

Phase 2:

* add row validation via cached internal Pydantic model
* add bulk validation modes
* document the dataclass schema API as preferred

Phase 3:

* optionally add a compatibility adapter for existing Pydantic models
* deprecate ad hoc `Annotated[...]` metadata conventions if they remain exposed

### Non-goal

Do not make the first implementation solve every possible schema feature.  The
first goal is to get the schema shape and internal architecture right.

---

## Serialization implications

Even if `save()` / `load()` are not implemented yet, this schema design should
anticipate persistence.

Eventually a persisted `CTable` will need to store:

* column names
* logical schema descriptors
* per-column defaults
* per-column NDArray storage options
* maybe validation constraints

That argues strongly for having a stable compiled schema representation early.

The compiled schema should be serializable to:

* JSON-compatible metadata
* or a small msgpack payload

The public dataclass itself does not need to be serialized directly.  Only the
compiled schema matters for persistence.

---

## Open questions

### 1. Should Python annotations be required to match the schema spec?

Example:

```python
id: int = b2.field(b2.int64(ge=0))
```

Recommended answer: yes, broadly, with sensible compatibility rules.

Allowed:

* `int` with `int64`
* `float` with `float64`
* `bool` with `bool`

Potentially allowed later:

* `str` with `string`
* `bytes` with `bytes`

Reject obviously inconsistent declarations early.

In other words:

* `id: int = b2.field(b2.int64(ge=0))` is good
* `id: int` is acceptable shorthand for inferred `b2.int64()`
* `id = b2.field(b2.int64(ge=0))` is not the preferred style because it drops
  the Python annotation

### 2. Where should nullability live?

Recommended answer: on the schema spec.

Example:

```python
name: str | None = b2.field(b2.string(max_length=32, nullable=True))
```

The Python annotation and schema spec should agree.

### 3. Should `b2.field()` require a spec?

Recommended answer: yes for the first version.

Allowing `b2.field(default=True)` without a spec means we must infer too much
from the Python annotation and lose clarity.

This still allows fully inferred fields that do not use `b2.field(...)` at all:

```python
active: bool = True
```

but once `b2.field(...)` is used, it should carry an explicit schema spec.

### 4. How much should Pydantic-specific behavior leak?

Recommended answer: as little as possible.

Users should not need to know whether validation is backed by Pydantic,
vectorized NumPy checks, or another mechanism.

---

## Concrete implementation sequence

This section turns the design into a proposed execution order with concrete
files, class names, and function signatures.

### Step 1: add schema descriptor primitives

Create a new module:

* `src/blosc2/schema.py`

Primary contents:

```python
from __future__ import annotations

from dataclasses import MISSING, Field as DataclassField, field as dc_field
from typing import Any

import numpy as np
```

Proposed public classes and functions:

```python
class SchemaSpec:
    dtype: np.dtype
    python_type: type[Any]
    nullable: bool

    def to_pydantic_kwargs(self) -> dict[str, Any]: ...
    def to_metadata_dict(self) -> dict[str, Any]: ...


class int64(SchemaSpec):
    def __init__(
        self, *, ge=None, gt=None, le=None, lt=None, nullable: bool = False
    ): ...


class float64(SchemaSpec):
    def __init__(
        self, *, ge=None, gt=None, le=None, lt=None, nullable: bool = False
    ): ...


class bool(SchemaSpec):
    def __init__(self, *, nullable: bool = False): ...


class string(SchemaSpec):
    def __init__(
        self, *, min_length=None, max_length=None, pattern=None, nullable: bool = False
    ): ...


class bytes(SchemaSpec):
    def __init__(self, *, min_length=None, max_length=None, nullable: bool = False): ...


def field(
    spec: SchemaSpec,
    *,
    default=MISSING,
    default_factory=MISSING,
    cparams: dict[str, Any] | None = None,
    dparams: dict[str, Any] | None = None,
    chunks: tuple[int, ...] | None = None,
    blocks: tuple[int, ...] | None = None,
    title: str | None = None,
    description: str | None = None,
) -> DataclassField: ...
```

Internal helper constants:

```python
BLOSC2_FIELD_METADATA_KEY = "blosc2"
```

Notes:

* Start with only the spec classes needed for the first `CTable` iteration:
  `int64`, `float64`, `bool`.
* Add `string` and `bytes` only if needed in the same slice of work.
* Avoid over-generalizing the first implementation.

### Step 2: add schema compiler and compiled representations

Create a new module:

* `src/blosc2/schema_compiler.py`

Primary internal dataclasses:

```python
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ColumnConfig:
    cparams: dict[str, Any] | None
    dparams: dict[str, Any] | None
    chunks: tuple[int, ...] | None
    blocks: tuple[int, ...] | None
    title: str | None
    description: str | None


@dataclass(slots=True)
class CompiledColumn:
    name: str
    py_type: Any
    spec: Any
    dtype: np.dtype
    default: Any
    default_factory: Any
    config: ColumnConfig


@dataclass(slots=True)
class CompiledSchema:
    row_cls: type[Any]
    columns: list[CompiledColumn]
    columns_by_name: dict[str, CompiledColumn]
    validator_model: type[Any] | None = None
```

Primary internal functions:

```python
def compile_schema(row_cls: type[Any]) -> CompiledSchema: ...
def infer_spec_from_annotation(annotation: Any, default: Any = MISSING) -> Any: ...
def validate_annotation_matches_spec(annotation: Any, spec: Any) -> None: ...
def get_blosc2_field_metadata(dc_field) -> dict[str, Any] | None: ...
```

Behavior:

* accept a dataclass type only
* for explicit `b2.field(...)`, read the spec from metadata
* for inferred fields like `id: int`, derive `b2.int64()`
* reject unsupported annotations early
* normalize all defaults/config into `CompiledSchema`

### Step 3: export the schema API from `blosc2`

Update:

* `src/blosc2/__init__.py`

Exports to add:

```python
from .schema import bool, bytes, field, float64, int64, string
```

And in `__all__`:

```python
"bool",
"bytes",
"field",
"float64",
"int64",
"string",
```

Notes:

* Be careful with `bool` and `bytes` in `__init__.py` because they shadow
  builtins within the module namespace.  That is acceptable if done deliberately,
  but it should be reviewed explicitly.
* If shadowing proves too awkward internally, keep the implementation names
  private and re-export the public names only.

### Step 4: refactor `CTable` to consume compiled schemas

Update:

* `src/blosc2/ctable.py`

Primary constructor signature:

```python
class CTable(Generic[RowT]):
    def __init__(
        self,
        row_type: type[RowT],
        new_data=None,
        *,
        expected_size: int = 1_048_576,
        compact: bool = False,
        validate: bool = True,
        cparams: dict[str, Any] | None = None,
        dparams: dict[str, Any] | None = None,
    ) -> None: ...
```

New internal state:

```python
self._schema: CompiledSchema
self._validate: bool
self._table_cparams: dict[str, Any] | None
self._table_dparams: dict[str, Any] | None
```

New internal helper methods:

```python
def _init_columns(self, expected_size: int) -> None: ...
def _resolve_column_storage(self, col: CompiledColumn) -> dict[str, Any]: ...
def _normalize_row_input(self, data: Any) -> dict[str, Any]: ...
def _coerce_row_to_storage(self, row: dict[str, Any]) -> dict[str, Any]: ...
```

Behavior changes:

* replace direct inspection of `row_type.model_fields`
* build columns from `self._schema.columns`
* derive column dtypes from compiled schema
* merge table-level and field-level storage settings

### Step 5: implement row validation adapter

Create a new internal module:

* `src/blosc2/schema_validation.py`

Primary functions:

```python
from typing import Any


def build_validator_model(schema: CompiledSchema) -> type[Any]: ...
def validate_row(schema: CompiledSchema, row: dict[str, Any]) -> dict[str, Any]: ...
def validate_rows_rowwise(
    schema: CompiledSchema, rows: list[dict[str, Any]]
) -> list[dict[str, Any]]: ...
```

Behavior:

* build and cache a Pydantic model per compiled schema
* map `SchemaSpec` constraints into Pydantic field definitions
* return normalized Python values ready for storage coercion

Implementation note:

* Cache the generated validator model on `CompiledSchema.validator_model`.
* Keep all Pydantic-specific logic isolated in this module.

### Step 6: wire validation into `append()`

Update:

* `src/blosc2/ctable.py`

Target signatures:

```python
def append(self, data: Any) -> None: ...
def _append_validated_row(self, row: dict[str, Any]) -> None: ...
```

Concrete behavior:

1. normalize incoming row shape
2. if `self._validate` is true, validate via `schema_validation.validate_row`
3. coerce to storage values
4. append into column NDArrays

Inputs to support in the first cut:

* dataclass row instance
* dict
* tuple/list in schema order

Inputs that can wait until later if needed:

* structured NumPy scalar
* Pydantic model instance

### Step 7: add `extend(..., validate=...)`

Update:

* `src/blosc2/ctable.py`

Proposed signature:

```python
def extend(self, data: Any, *, validate: bool | None = None) -> None: ...
```

Supporting internal helpers:

```python
def _normalize_rows_input(
    self, data: Any
) -> tuple[list[dict[str, Any]] | None, dict[str, Any] | None]: ...
def _extend_rowwise(self, rows: list[dict[str, Any]], *, validate: bool) -> None: ...
def _extend_columnwise(self, columns: dict[str, Any], *, validate: bool) -> None: ...
```

First implementation target:

* support iterable of rows via `_extend_rowwise`
* preserve correctness first, optimize later

Second implementation target:

* add `_extend_columnwise` for structured arrays and dict-of-arrays
* add vectorized validation for simple constraints

### Step 8: add vectorized validation helpers

Create a new internal module:

* `src/blosc2/schema_vectorized.py`

Primary functions:

```python
from typing import Any


def validate_column_values(col: CompiledColumn, values: Any) -> None: ...
def validate_column_batch(schema: CompiledSchema, columns: dict[str, Any]) -> None: ...
```

Initial checks to support:

* numeric `ge`, `gt`, `le`, `lt`
* string and bytes `min_length`, `max_length`
* nullability
* dtype compatibility after coercion

This module should remain optional in the first PR if the rowwise path is enough
to land the architecture cleanly.

### Step 9: add schema introspection to `CTable`

Update:

* `src/blosc2/ctable.py`

Proposed property:

```python
@property
def schema(self) -> CompiledSchema: ...
```

Optional helper methods:

```python
def schema_dict(self) -> dict[str, Any]: ...
def column_schema(self, name: str) -> CompiledColumn: ...
```

Goal:

* make the new schema layer visible and debuggable
* provide a stable base for future save/load work

### Step 10: add tests in focused modules

Add:

* `tests/ctable/test_schema_specs.py`
* `tests/ctable/test_schema_compiler.py`
* `tests/ctable/test_schema_validation.py`
* `tests/ctable/test_ctable_dataclass_schema.py`

Test scope by file:

`tests/ctable/test_schema_specs.py`

* spec construction
* dtype mapping
* metadata export

`tests/ctable/test_schema_compiler.py`

* explicit `b2.field(...)`
* inferred shorthand from plain annotations
* annotation/spec mismatch rejection
* defaults handling

`tests/ctable/test_schema_validation.py`

* Pydantic validator generation
* constraint enforcement
* nullable vs non-nullable behavior

`tests/ctable/test_ctable_dataclass_schema.py`

* `CTable(Row)` construction
* append with dataclass/dict/tuple
* extend with iterable of rows
* per-column `cparams` override plumbing

### Step 11: keep the legacy prototype isolated during transition

Short-term implementation choice:

* if the current `ctable.py` prototype is still in active flux, prefer landing
  the schema/compiler modules first and then refactoring `CTable` over them
* do not expand the old Pydantic-specific schema path further

Possible follow-up helper:

```python
def compile_legacy_pydantic_schema(row_cls: type[Any]) -> CompiledSchema: ...
```

But only add that if compatibility becomes necessary.

### Step 12: persistence groundwork

No need to implement `save()` / `load()` immediately, but define serialization
hooks on the schema side now.

Add to `CompiledSchema` or a related helper:

```python
def schema_to_dict(schema: CompiledSchema) -> dict[str, Any]: ...
def schema_from_dict(data: dict[str, Any]) -> CompiledSchema: ...
```

This should remain internal until the persisted format is stable.

### Step 13: delivery order across PRs

Recommended PR slicing:

PR 1:

* `src/blosc2/schema.py`
* `src/blosc2/schema_compiler.py`
* exports in `src/blosc2/__init__.py`
* tests for schema specs and compiler

PR 2:

* `CTable` constructor refactor to use compiled schema
* `append()` row normalization
* row-wise validation module
* `tests/ctable/test_ctable_dataclass_schema.py`

PR 3:

* `extend(..., validate=...)`
* vectorized validation helpers
* schema introspection property
* more tests for batch validation and overrides

PR 4:

* persistence groundwork
* optional compatibility adapter for legacy Pydantic model declarations

### Step 14: concrete first-PR checklist

The smallest coherent first implementation should be:

1. add `src/blosc2/schema.py`
2. add `src/blosc2/schema_compiler.py`
3. export `field`, `int64`, `float64`, `bool`
4. add tests for:
   * explicit field specs
   * inferred shorthand
   * mismatch rejection
5. stop there

That first PR gives the project:

* the public schema vocabulary
* the internal compiled representation
* confidence in the canonical API shape

before touching too much `CTable` mutation logic.

---

## Recommendation

The recommended direction is:

1. Make **dataclasses** the public schema declaration mechanism for `CTable`.
2. Introduce **lowercase schema spec objects** such as `b2.int64(...)`.
3. Use **`b2.field(...)`** to carry both the schema spec and per-column NDArray
   configuration.
4. Compile the schema once into an internal representation.
5. Use **Pydantic internally for row validation**, but keep it hidden behind the
   Blosc2 schema API.
6. Add a separate **bulk validation path** for large inserts so `extend()` does
   not depend entirely on per-row Pydantic validation.

This design gives the project:

* a cleaner user API
* a better place for columnar storage configuration
* a clear boundary between schema, validation, and storage
* flexibility to evolve validation internals later
* a strong base for future persistence and schema introspection
