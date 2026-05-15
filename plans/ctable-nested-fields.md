# CTable nested fields via physical leaf columns

## Summary

Add first-class support for nested schemas in `CTable` by **physically flattening leaf fields** into real persisted columns, while preserving logical nested structure for row I/O and Arrow/Parquet roundtrips.

Key idea:

- Logical path: `trip.begin.lon`
- Physical storage path in container: `/_cols/trip/begin/lon`
- Canonical root field name: `""` (empty string)
- Display alias for root (optional): `/`

This keeps analytics/indexing fast (leaf = ordinary column), and matches `.b2d` / `.b2z` container layout naturally.

**Status: core implementation complete.**  All acceptance criteria are met.
Remaining work is captured in the [Future work](#future-work) section below.

---

## Goals

1. Support nested struct/list schemas without storing struct leaves as opaque varlen/object blobs.
2. Enable columnar analytics on scalar leaves using existing `CTable` machinery:
   - filters (`where`)
   - lazy expressions
   - aggregates (`sum/min/max/mean/std`)
   - indexes
   - sorting/grouping paths already supported for scalar columns
3. Preserve nested logical row interface (dict/list reconstruction on read).
4. Keep backward compatibility for existing flat tables and existing nested-as-varlen tables.

## Non-goals (phase 1)

1. Full list-element relational semantics (`explode`, SQL-like unnests) for query planner.
2. Indexing directly on list-valued paths.
3. Breaking on-disk compatibility of existing tables.

---

## Proposed model

## 1) Path model

Define a canonical logical field-path type:

- Root: `""`
- Path segments: `("trip", "begin", "lon")`
- Dotted display key: `trip.begin.lon`

Add helpers:

- `split_field_path(str) -> tuple[str, ...]`  ✅ implemented (`ctable_storage.py`; backslash-escape aware)
- `join_field_path(tuple[str, ...]) -> str`  ✅ implemented (`ctable_storage.py`; escapes literal `.`, `/`, and `\\`)
- escaping/unescaping for literal `.` and `/` in field names  ✅ implemented for logical names via backslash escaping and for physical storage via percent-encoded path segments

Recommendation:

- Canonical internal identity: tuple segments
- Dotted names only as user syntax
- Physical storage path built from escaped segments  ✅ literal `.`, `/`, `%`, and `\\` inside segments are percent-encoded

## 2) Physical layout  ✅ implemented

Persist scalar leaves as standard column arrays under `_cols` hierarchy:

- `/_cols/trip/begin/lon`
- `/_cols/trip/begin/lat`
- `/_cols/trip/begin/time`
- `/_cols/payment/fare`

Intermediate nodes are namespaces only (no data arrays).

For lists:

- Keep existing `ListArray` physical representation for list leaves.  ✅
- For `list<struct<...>>`, phase 1 keeps list cell storage as list payload (no explode).  ✅

## 3) Schema metadata  ✅ implemented

Extend schema serialization with nested mapping metadata, e.g.:

- logical path -> physical column token/path  ✅ (`schema.metadata["nested"]` dict)
- physical column -> storage path  ✅ (`schema.metadata["nested"]["physical_to_storage"]`)
- root logical alias metadata when needed  ✅
- row reconstruction flag when nested Arrow structs were flattened  ✅

Leaf spec details such as kind, dtype, nullability, and scalar/list/dictionary behavior remain in the standard schema column specs rather than being duplicated in `metadata["nested"]`.  ✅

Keep `CompiledSchema.columns` as the ordered list of **physically stored leaf columns**. `CompiledSchema.columns_by_name` may additionally contain virtual logical aliases, such as top-level `StructSpec` entries used for Arrow/Parquet schema roundtrips; these aliases are not stored columns and do not appear in `CTable.col_names`.  ✅

---

## API behavior

## Column access  ✅ implemented

Allow both:

- `t["trip.begin.lon"]`  ✅
- `t.trip.begin.lon` (via lightweight namespace proxy objects)  ✅ (`_NestedColumnNamespace`; `_StructPathColumn` is used for struct-prefix virtual access such as `t["trip"]`)

`Column` operations on scalar leaves behave exactly like current top-level scalar columns.  ✅

## Row materialization  ✅ implemented

- `t[i]` reconstructs nested dict/list shape from leaves and list payload columns.  ✅
- Top-level unnamed field (`""`) is handled as root container.  ✅

## Select/projection  ✅ implemented

`select([...])` accepts:

- leaf paths (`"trip.begin.lon"`)  ✅
- struct prefix (`"trip.begin"`) that expands to descendant leaves  ✅

## Expressions  ✅ implemented

`where("trip.begin.lon > -87.7 and payment.fare > 10")` supported by path rewriting to operand IDs or canonical flat leaf names.  ✅

---

## Implementation plan

## Phase 0 — design/compat scaffolding

1. ✅ Path splitting/joining helpers (`_column_name_to_relpath` + inverse in schema metadata).
2. ✅ New schema metadata version (`schema.metadata["nested"]` with `version` key; backward-compatible read of old flat schemas).
3. ⚠️ Feature flag (internal) to enable nested physical layout for new tables — not a separate flag; nested layout is activated implicitly when the input schema contains struct fields.

## Phase 1 — schema compilation flattening

1. ✅ Schema compiler flattens nested structs into physical leaf columns (`schema_compiler.py`, `_flatten_arrow_struct_schema`).
2. ✅ Nested path mapping kept for reconstruction/export (`logical_to_physical`, `physical_to_storage`, optional root alias, and `reconstruct_rows` in nested metadata). Leaf type details remain in normal schema column specs.
3. ✅ Deterministic flat column keys — canonical dotted form used throughout.
4. ✅ Nullable propagation rules explicit (propagated from parent struct nullability).

## Phase 2 — storage backend

1. ✅ `ctable_storage` create/open accept hierarchical column paths.
2. ✅ Arrays stored in `/_cols/<seg>/<seg>/...` hierarchy.
3. ✅ Reopen logic uses stored schema column names and maps dotted names back to hierarchical `_cols/...` paths.
4. ~~Migration-safe fallback for legacy flat `_cols/<name>` tables~~ — **skipped**: no code ever shipped writing dotted names as flat paths, so no migration is needed.

## Phase 3 — read/write data paths

1. ✅ `append`/`extend` flatten input nested dicts into leaf columns (`_flatten_nested_dict`, updated `_normalize_row_input` and `extend`).
2. ✅ `__getitem__(int)` and row iterators reconstruct nested rows (`_materialize_row`, `reconstruct_rows` flag).
3. ✅ Fast-path for already-flat rows preserved.

## Phase 4 — column resolution and expression engine

1. ✅ Column resolver from dotted path string → physical leaf column.
2. ✅ Attribute path proxy `t.trip.begin.lon` via `_StructPathColumn`.
3. ✅ Expression parsing includes nested leaves (`_where_expression_operands`).
4. ✅ List/object leaf expressions restricted appropriately in phase 1.

## Phase 5 — indexes and analytics

1. ✅ `create_index(col_name="trip.begin.lon")` works on scalar leaves.
2. ✅ Index catalog uses canonical dotted target path.
3. ✅ Aggregates (`mean`, `sum`, `min`, `max`, `std`) and `sort_by` work on resolved leaf NDArrays.

## Phase 6 — Arrow/Parquet import/export

1. ✅ Import: nested Arrow schema flattened into leaf storage + nested metadata (`from_arrow`, `_flatten_arrow_struct_*`).
2. ✅ Export: Arrow nested schema rebuilt from leaves (`to_arrow`, `to_parquet` reconstruct struct hierarchy).
3. ✅ Dictionary/timestamp/null semantics unchanged.

## Phase 7 — docs/tests/perf

1. Tests:
   - ✅ Append/reopen/roundtrip for nested rows (`tests/ctable/test_nested_append.py`, `test_nested_access_storage.py`).
   - ✅ `where`/`select`/`index`/`aggregate` on nested scalar leaves (covered in existing ctable test suite).
   - ✅ Compatibility: legacy flat tables still pass all tests.
   - ✅ Path parsing and escaping tests for literal `.` and `/` in nested Arrow field names (`tests/ctable/test_nested_access_storage.py`).
2. Docs:
   - ✅ Nested path syntax, column access, filtering, Arrow/Parquet roundtrip (`doc/reference/ctable.rst`, "Nested fields" section).
   - ✅ Method-level docstrings updated: `append`, `extend`, `__getitem__`, `where`, `select`, `rename_column`, `create_index`, `sort_by`, `from_arrow`, `from_parquet`.
3. Benchmarks:
   - ✅ Nested leaf filter/index performance vs flat columns (`bench/ctable/bench_nested_filter_index.py`); overhead is negligible.

---

## Compatibility and migration

1. ✅ Existing tables remain readable/writable as-is.
2. ✅ Nested layout activated automatically when schema contains struct fields.
3. Optional utility later: migrate legacy nested-varlen columns to flattened-leaf layout (see Future work).

---

## Acceptance criteria  ✅ all met

1. ✅ Can ingest taxi-like schema and persist leaves under hierarchical `_cols/...` paths.
2. ✅ `t["trip.begin.lon"].mean()` works and matches Arrow/Awkward reference.
3. ✅ `t.where("payment.fare > 20").nrows` works.
4. ✅ `t.create_index(col_name="trip.begin.time")` works for scalar leaf.
5. ✅ `t[i]` returns nested row shape equivalent to input schema.
6. ✅ Existing non-nested/legacy tables keep current behavior unchanged.

---

## Future work

### FW-1 — Field-name escaping for literal `.` and `/`

**Status**: implemented.

Logical nested paths use unescaped `.` as the separator. Literal `.`, `/`, and `\\`
inside a field-name segment are represented with backslash escaping in the logical
column name, e.g. Arrow path segments `("trip.info", "begin/point", "lon.deg")`
become `trip\\.info.begin\\/point.lon\\.deg`.

Physical storage percent-encodes structural characters inside each path segment before
joining segments under `_cols`, e.g. the same leaf is stored at
`_cols/trip%2Einfo/begin%2Fpoint/lon%2Edeg`.

### FW-2 — List-struct analytics (explode / unnest)

**Status**: deferred (non-goal for phase 1).

`list<struct<...>>` fields are currently stored as opaque list-payload columns.  Future
work would:

- Define an `explode` operation that creates a row-per-element view.
- Enable `where` / `create_index` on paths inside list elements.
- Design SQL-style unnest semantics.

### FW-3 — Migration utility for legacy nested-varlen tables

**Status**: deferred; likely unnecessary unless user demand appears.

Because `CTable` is newly released, few if any production tables are expected to exist
with top-level Arrow `struct<...>` columns imported as opaque `blosc2.struct` varlen
columns. Existing tables remain readable as-is, but they will not automatically gain
nested-leaf analytics.

Recommended path: re-import the original Arrow/Parquet source with a python-blosc2
version that supports nested-leaf flattening. This creates the new physical leaf layout
and nested metadata directly.

A future `CTable.migrate_nested_columns()` utility could still be considered if users
have important legacy tables without access to the original source data. Such a utility
would need to:

- Detect columns whose schema spec is `struct` with a known logical type.
- Re-import/materialize them as flattened leaf columns.
- Update schema metadata and physical layout atomically.
- Leave `list<struct<...>>` migration out of scope until list-struct analytics are
  designed separately.
