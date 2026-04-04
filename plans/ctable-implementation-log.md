# CTable Implementation Log

This document records everything implemented across the CTable feature:
the `ctable-schema.md` redesign (schema, validation, serialization, optimizations)
and the `ctable-persistency.md` phase (file-backed storage, `open()`, read-only mode).

---

## Phase 1 — Schema redesign (`ctable-schema.md`)

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

Total: **135 tests, all passing** (after Phase 1 + optimizations).

---

## Phase 1 design decisions

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
Persistence requires a self-contained schema representation that survives without
the original Python dataclass. Establishing the serialization format before
persistence was built ensured the format was stable before anything depended on it.

---

## Phase 1 optimizations (post-schema)

Several performance improvements were made after the schema work was complete:

**`_last_pos` cache**
Added `_last_pos: int | None` to `CTable`. Tracks the physical index of the next
write slot so that `append()` and `extend()` no longer need to scan backward through
chunk metadata on every call. Set to `None` after any deletion (triggers one lazy
recalculation on the next write). Set to `_n_rows` after `compact()`. Eliminated a
backward O(n_chunks) scan per insert.

**`_grow()` helper**
Extracted the capacity-doubling logic into `_grow()`. Removes duplication between
`append()` and `extend()`.

**In-place delete**
`delete()` now writes the updated boolean array back with `self._valid_rows[:] =
valid_rows_np` (in-place slice assignment) instead of creating a new NDArray.
Avoids a full allocation on each delete.

**`head()` / `tail()` refactored**
Both methods now reuse `_find_physical_index()` instead of containing their own
chunk-walk loops.

**`_make_view()` classmethod**
Added to construct view CTables without going through `__init__`. Avoids
allocating and immediately discarding NDArrays that were never used.

**`_NumericSpec` mixin + new spec types**
All numeric specs (`int8` through `uint64`, `float32`, `float64`) share a common
`_NumericSpec` mixin for `ge`/`gt`/`le`/`lt` constraint handling, eliminating
boilerplate. New specs added: `int8`, `int16`, `int32`, `uint8`, `uint16`,
`uint32`, `uint64`, `float32`.

**String vectorized validation**
`validate_column_values` uses `np.char.str_len()` (true C-level) for `U`/`S` dtype
arrays instead of `np.vectorize(len)` (Python loop in disguise). The check also
extracted `_validate_string_lengths()` to reduce cyclomatic complexity.

**Column name validation**
`compile_schema` now calls `_validate_column_name()` on every field. Rejects names
that are empty, start with `_`, or contain `/` — rules that apply equally to
in-memory and persistent tables.

---

## Phase 2 — Persistency (`ctable-persistency.md`)

### New file: `src/blosc2/ctable_storage.py`

A storage-backend abstraction that keeps all file I/O out of `ctable.py`.

**`TableStorage`** — interface class defining:
`create_column`, `open_column`, `create_valid_rows`, `open_valid_rows`,
`save_schema`, `load_schema`, `table_exists`, `is_read_only`.

**`InMemoryTableStorage`** — trivial implementation that creates plain in-memory
`blosc2.NDArray` objects and is a no-op for `save_schema`. Used when `urlpath` is
not provided (existing default behaviour, unchanged).

**`FileTableStorage`** — file-backed implementation.

Disk layout:

```
<urlpath>/
    _meta.b2frame       ← blosc2.SChunk; vlmeta holds kind, version, schema JSON
    _valid_rows.b2nd    ← file-backed boolean NDArray (tombstone mask)
    _cols/
        <name>.b2nd     ← one file-backed NDArray per column
```

Key implementation notes:
- `save_schema` always opens `_meta.b2frame` with `mode="w"` (create path only).
- `load_schema` / `check_kind` use `blosc2.open()` (not `blosc2.SChunk(...,
  mode="a")`), which is the correct API for reopening an existing SChunk file.
- File-backed NDArrays (`urlpath=..., mode="w"`) support in-place writes
  (`col[pos] = value`, `col[start:end] = arr`) that persist immediately. This is
  why resize (`_grow()`), append, extend, and delete all work transparently on
  persistent tables.
- `_n_rows` on reopen is reconstructed as `blosc2.count_nonzero(valid_rows)` —
  always correct because unwritten slots are `False`, same as deleted slots.
- `_last_pos` is set to `None` on reopen and resolved lazily by `_resolve_last_pos()`
  on the first write.

### Changes to `src/blosc2/ctable.py`

**Constructor**

New parameters: `urlpath: str | None = None`, `mode: str = "a"`.

Logic:
- `urlpath=None` → `InMemoryTableStorage` → existing behaviour unchanged.
- `urlpath` + existing table + `mode != "w"` → open existing (load schema from
  disk, open file-backed arrays, reconstruct state).
- `urlpath` + `mode="w"` or no existing table → create new (compile schema,
  save to disk, create file-backed arrays).
- Passing `new_data` when opening an existing table raises `ValueError`.

**`CTable.open(cls, urlpath, *, mode="r")`**

New classmethod for ergonomic read-only access. Opens the table, verifies
`kind="ctable"` in vlmeta, reconstructs schema from JSON (no dataclass needed),
returns a fully usable `CTable`.

**Read-only enforcement**

`_read_only: bool` flag set from `storage.is_read_only()`. Guards added to the top
of `append()`, `extend()`, `delete()`, `compact()` — each raises
`ValueError("Table is read-only (opened with mode='r').")`.

**`_make_view(cls, parent, new_valid_rows)`**

New classmethod that constructs a view `CTable` directly via `cls.__new__` without
calling `__init__`. Replaces the old `CTable(self._row_type, expected_size=...)` +
`retval._cols = self._cols` pattern, which was wasteful (allocated NDArrays then
discarded them) and broke when `_row_type` is `None` (tables opened via `open()`).

**`schema_dict()`**

No longer needs a local import of `schema_to_dict` — now imported at the module top.

### New test file: `tests/ctable/test_persistency.py`

23 tests covering:

| Test group | What it checks |
|---|---|
| Layout | `_meta.b2frame`, `_valid_rows.b2nd`, `_cols/<name>.b2nd` all exist after creation |
| Metadata | `kind`, `version`, `schema` in vlmeta; column names and order in schema JSON |
| Round-trips | Data survives reopen via both `CTable(Row, urlpath=..., mode="a")` and `CTable.open()` |
| Column order | Preserved exactly from schema JSON, not from filesystem order |
| Constraints | Validation re-enabled after reopen (schema reconstructed from disk) |
| Append/extend/delete after reopen | Mutations visible in subsequent opens |
| `_valid_rows` on disk | Tombstone mask correctly stored and loaded |
| `mode="w"` | Overwrites existing table; subsequent open sees empty table |
| Read-only | `append`, `extend`, `delete`, `compact` all raise on `mode="r"` |
| Read-only reads | `row[]`, column access, `head()`, `tail()`, `where()` all work |
| Error cases | `FileNotFoundError` for missing path; `ValueError` for wrong kind |
| Column name validation | Empty, `_`-prefixed, `/`-containing names rejected |
| `new_data` guard | `ValueError` when `new_data` passed to open-existing path |
| Capacity growth | `_grow()` (resize) works on file-backed arrays and survives reopen |

Total: **158 tests, all passing**.

### New benchmark: `bench/ctable/bench_persistency.py`

Four sections:

1. **`extend()` bulk insert** — in-memory vs file-backed at 1k–1M rows.
   Overhead converges to ~1x at 1M rows (compression dominates, not I/O).
2. **`open()` / reopen time** — ~4–10 ms regardless of table size. Fixed cost:
   open 3 files (meta, valid_rows, one column) + parse schema JSON.
3. **`append()` single-row** — file-backed is ~6x slower per row (~3 ms vs ~0.5 ms).
   Recommendation: batch inserts via `extend()` for persistent tables.
4. **Column `to_numpy()`** — essentially identical between backends (≤1.06x ratio).
   Decompression dominates; file I/O is negligible once data is loaded.

---

## Phase 2 design decisions

**Why direct files instead of TreeStore?**
TreeStore stores snapshots of in-memory arrays. In-place writes to a
TreeStore-retrieved NDArray do not persist after reopen. File-backed NDArrays
created with `urlpath=...` support in-place writes natively. Using direct `.b2nd`
files aligns with how the rest of blosc2 handles persistent arrays.

**Why `blosc2.SChunk` vlmeta for metadata, not JSON files?**
`vlmeta` is compressed and is already part of the blosc2 ecosystem.
`blosc2.open()` works on `.b2frame` files the same way it works on `.b2nd` files,
keeping the open path uniform.

**Why not store `_last_pos` in metadata?**
`_resolve_last_pos()` reconstructs it in O(n_chunks) with no full decompression.
Storing it would create a write on every `append()` just to update a counter in the
SChunk — not worth the extra I/O.

**Why `_make_view()` instead of calling `__init__`?**
`__init__` now has storage-routing logic and would try to create new NDArrays even
for views (which immediately get thrown away). `_make_view()` via `__new__` is
explicit and zero-waste.

**Why `CTable.open()` defaults to `mode="r"`?**
The most common read-back scenario is inspection or analysis, not modification.
Defaulting to read-only prevents accidental mutations on shared or archived tables.
