# CTable Implementation Log

This document records what was implemented as part of the `ctable-schema.md` redesign.
It covers every new file, every significant change, and the reasoning behind each decision.

---

## Overview

The goal was to replace the original Pydantic-`BaseModel`-based schema API with a
**dataclass-first schema API** using declarative spec objects (`b2.int64()`,
`b2.float64()`, etc.) and to wire in full constraint validation on insert.

---

## New files

### `src/blosc2/schema.py`

Defines the public schema vocabulary.

**Contents:**

- `SchemaSpec` — abstract base class for all column type descriptors.
- `int64`, `float64`, `bool`, `complex64`, `complex128`, `string`, `bytes` —
  concrete spec classes. Each carries:
  - `dtype` — the NumPy storage dtype
  - `python_type` — the corresponding Python type
  - Constraint attributes: `ge`, `gt`, `le`, `lt` (numeric); `min_length`,
    `max_length`, `pattern` (string/bytes)
  - `to_pydantic_kwargs()` — returns only the non-`None` constraints as a dict,
    used internally to build Pydantic validator models
  - `to_metadata_dict()` — returns a JSON-compatible dict used for serialization
- `field(spec, *, default, cparams, dparams, chunks, blocks)` — attaches a spec
  and per-column storage options to a dataclass field via
  `dataclasses.field(metadata={"blosc2": {...}})`.
- `BLOSC2_FIELD_METADATA_KEY = "blosc2"` — stable key for the metadata dict.

**Key design note:** `bool` and `bytes` shadow Python builtins inside this module.
Private aliases `_builtin_bool` and `_builtin_bytes` are used where the originals
are needed.

---

### `src/blosc2/schema_compiler.py`

Compiles a dataclass row definition into an internal `CompiledSchema`.

**Contents:**

- `ColumnConfig(slots=True)` — holds per-column NDArray storage options:
  `cparams`, `dparams`, `chunks`, `blocks`.
- `CompiledColumn(slots=True)` — holds everything about one column:
  `name`, `py_type`, `spec`, `dtype`, `default`, `config`, `display_width`.
- `CompiledSchema(slots=True)` — holds the full compiled schema:
  `row_cls`, `columns`, `columns_by_name`, `validator_model` (filled lazily by
  `schema_validation.py`).
- `compile_schema(row_cls)` — main entry point. Walks `dataclasses.fields()`,
  reads `blosc2` metadata from each field, infers specs from plain annotations
  where no `b2.field()` is present, validates annotation/spec compatibility, and
  returns a `CompiledSchema`.
- `get_blosc2_field_metadata(dc_field)` — extracts the `"blosc2"` metadata dict
  from a dataclass field, or returns `None`.
- `infer_spec_from_annotation(annotation)` — builds a default spec from a plain
  Python type (`int` → `int64()`, `float` → `float64()`, etc.). Used for inferred
  shorthand fields like `id: int` (no `b2.field()`).
- `validate_annotation_matches_spec(name, annotation, spec)` — rejects
  declarations where the Python annotation is incompatible with the spec (e.g.
  `id: str = b2.field(b2.int64())`).
- `compute_display_width(spec)` — returns a sensible terminal column width based
  on dtype kind.
- `schema_to_dict(schema)` — serializes a `CompiledSchema` to a JSON-compatible
  dict. Handles `MISSING` defaults (→ `None`), complex defaults
  (→ `{"__complex__": True, "real": ..., "imag": ...}`), and optional per-column
  storage config fields.
- `schema_from_dict(data)` — reconstructs a `CompiledSchema` from a serialized
  dict. Does not require the original Python dataclass. Returns `row_cls=None`.
  Raises `ValueError` on unknown `kind` or unsupported `version`.

---

### `src/blosc2/schema_validation.py`

Row-level constraint validation backed by Pydantic. All Pydantic imports are
isolated here so the rest of the codebase never touches Pydantic directly.

**Contents:**

- `build_validator_model(schema)` — builds a `pydantic.create_model(...)` class
  from the compiled schema. Each column's `to_pydantic_kwargs()` result is passed
  to `pydantic.Field(...)`. The result is cached on `schema.validator_model` so it
  is built only once per schema.
- `validate_row(schema, row_dict)` — validates one `{col_name: value}` dict.
  Calls the cached Pydantic model, catches `ValidationError`, and re-raises as a
  plain `ValueError` so callers never need to import Pydantic.
- `validate_rows_rowwise(schema, rows)` — validates a list of row dicts. Raises
  `ValueError` on the first violation, including the row index.

**When used:** called by `CTable.append()` when `self._validate` is `True`.

---

### `src/blosc2/schema_vectorized.py`

NumPy-based constraint validation for bulk inserts. Used by `CTable.extend()` to
check entire column arrays at once without per-row Python overhead.

**Contents:**

- `validate_column_values(col, values)` — checks all constraint attributes
  present on `col.spec` against a NumPy array of values. Uses `np.any(arr < ge)`
  style checks. For string/bytes, uses `np.vectorize(len)` to check lengths.
  Reports the first offending value in the error message.
- `validate_column_batch(schema, columns)` — calls `validate_column_values` for
  every column present in the `columns` dict.

**Why separate from Pydantic validation:** `extend()` can receive millions of
rows. Row-by-row Pydantic validation would be unacceptably slow for large batches.
NumPy operations run in C with no per-element Python overhead.

---

## Changes to existing files

### `src/blosc2/ctable.py`

**Schema detection at construction:**

```python
if dataclasses.is_dataclass(row_type) and isinstance(row_type, type):
    self._schema = compile_schema(row_type)
else:
    self._schema = _compile_pydantic_schema(row_type)  # legacy path
```

**New constructor parameters:** `validate=True`, `cparams=None`, `dparams=None`.
Stored as `self._validate`, `self._table_cparams`, `self._table_dparams`.

**`_init_columns`:** builds NDArrays from `self._schema.columns` instead of
iterating `row_type.model_fields`.

**`_resolve_column_storage`:** merges column-level and table-level storage
settings. Column-level wins.

**`_normalize_row_input`:** normalizes list/tuple/dict/dataclass instance/
`np.void` to a `{col_name: value}` dict.

**`_coerce_row_to_storage`:** coerces each value to the column's NumPy dtype
using `np.array(val, dtype=col.dtype).item()`.

**`append()` new flow:**
1. `_normalize_row_input(data)` → dict
2. `validate_row(schema, row)` if `self._validate` (Pydantic row validation)
3. `_coerce_row_to_storage(row)` → storage dict
4. Find write position, resize if needed, write column by column.

**`extend()` new signature:** `extend(data, *, validate=None)`.
- `validate=None` uses `self._validate` (table default).
- `validate=True/False` overrides for this call only.
- Vectorized validation runs on raw column arrays before `blosc2.asarray` conversion.

**Schema introspection (new):**
- `table.schema` property — returns `self._schema`.
- `table.column_schema(name)` — returns `CompiledColumn` for a given column name.
- `table.schema_dict()` — delegates to `schema_to_dict(self._schema)`.

**Legacy Pydantic adapter kept:**
- `NumpyDtype`, `MaxLen`, `_resolve_field_dtype`, `_LegacySpec`,
  `_compile_pydantic_schema` all remain so existing Pydantic-`BaseModel`-based
  schemas continue to work during the transition.

### `src/blosc2/__init__.py`

Added to delayed imports:

```python
from .schema import bool, bytes, complex64, complex128, field, float64, int64, string
```

Added to `__all__`:
`"bool"`, `"bytes"`, `"complex64"`, `"complex128"`, `"field"`, `"float64"`,
`"int64"`, `"string"`.

---

## Tests

All tests live in `tests/ctable/`.

| File | Covers |
|---|---|
| `test_schema_specs.py` | Spec dtypes, python types, constraint storage, `to_pydantic_kwargs`, `to_metadata_dict`, `blosc2` namespace exports |
| `test_schema_compiler.py` | `compile_schema` with explicit `b2.field()`, inferred shorthand, mismatch rejection, defaults, cparams; `schema_to_dict` / `schema_from_dict` roundtrip |
| `test_schema_validation.py` | `append` and `extend` constraint enforcement; boundary values; `validate=False` bypass; `gt`/`lt` exclusive bounds; NumPy structured array path |
| `test_ctable_dataclass_schema.py` | End-to-end CTable construction, `append` with tuple/list/dict, `extend` with iterable and structured array, per-call `validate=` override, schema introspection |
| `test_construct.py` | Construction variants, `append`/`extend`/resize, column integrity, `_valid_rows` |
| `test_column.py` | Column indexing, slicing, iteration, `to_numpy()`, mask independence |
| `test_compact.py` | Manual and auto compaction |
| `test_delete_rows.py` | Single/list/slice deletion, out-of-bounds, edge cases, stress |
| `test_extend_delete.py` | Interleaved extend/delete cycles, mask correctness, resize behavior |
| `test_row_logic.py` | Row indexer (int/slice/list), views, chained views |

Total: **127 tests, all passing**.

---

## Design decisions

**Why two validation paths?**
`append()` handles one row at a time — Pydantic is fast enough and also performs
type coercion and default filling. `extend()` handles bulk data — vectorized NumPy
checks are orders of magnitude faster for large batches.

**Why `validate=None` as the default on `extend()`?**
`None` means "inherit the table-level flag". `True`/`False` are explicit overrides.
This avoids a boolean that accidentally silences the table-level setting.

**Why keep the Pydantic adapter?**
Existing code using `class RowModel(BaseModel)` continues to work without
modification. The adapter is not on the critical path for new code.

**Why `schema_to_dict` / `schema_from_dict` now?**
Persistence (`save()`/`load()`) requires a self-contained schema representation.
Establishing the serialization format early means the format will be stable before
anything depends on it.
