# Materialized Computed Column Plan

## Goal

Add a **minimal** way to turn an existing virtual/computed `CTable` column into a
regular stored column that can then be indexed, persisted, exported, and managed
using the current stored-column machinery.

The intended first public API is:

```python
t.materialize_computed_column("total", new_name="total_stored")
```

This plan intentionally prefers a small, low-risk feature over a broader
redesign of computed-column indexing.

---

## Why this approach

The current implementation already has:

- stored `CTable` columns backed by physical 1-D `NDArray` objects
- virtual/computed columns backed by `LazyExpr` metadata in `CTable._computed_cols`
- table-owned indexing for **stored** columns
- persistent schema support for stored columns and computed-column metadata

What is missing is direct indexing for computed columns.  Implementing that for
virtual columns would require table-level expression indexing and planner work.
By contrast, materializing a computed column is much simpler:

1. evaluate the computed expression once over the table's physical row space
2. create a normal stored column with the resulting dtype and values
3. leave all indexing, persistence, and query planning unchanged

This makes materialized computed columns the easiest route to:

- `create_index()` support, including `FULL`
- persistence with no extra index semantics
- clear user mental model

---

## Non-goals for the first iteration

This plan does **not** aim to add:

- automatic synchronization between source columns and the materialized result
- dependency-aware stale tracking for materialized columns
- direct indexing of virtual/computed columns
- support for materializing arbitrary `LazyExpr` objects not already registered
  as computed columns
- replacement/in-place conversion of a computed column into a stored one
- special storage outside the table for materialized data

The first version should be a **snapshot materialization** feature only.

---

## Current implementation constraints

Relevant existing behavior:

- Computed columns live in `CTable._computed_cols` and are persisted only as
  metadata in the schema dict:
  - `src/blosc2/ctable.py:_schema_dict_with_computed`
  - `src/blosc2/ctable.py:_load_computed_cols_from_schema`
  - `src/blosc2/ctable.py:add_computed_column`
- Computed columns are exposed in `col_names` but are not stored in
  `self._schema.columns`
- Stored columns live in `self._cols` and participate in current table-owned
  indexing
- `CTable.add_column()` already creates a new physical stored column and updates:
  - storage
  - schema
  - `self._cols`
  - `self.col_names`
  - `self._col_widths`
- `CTable.create_index()` currently rejects computed columns because they have no
  physical storage

This means the cleanest design is:

- compute values from the existing computed column definition
- create a **new stored column** using the same path as any other added column
- write the computed values into that new stored column

---

## Proposed public API

## `CTable.materialize_computed_column`

```python
def materialize_computed_column(
    self,
    name: str,
    *,
    new_name: str | None = None,
    dtype: np.dtype | None = None,
    cparams: dict | blosc2.CParams | None = None,
) -> None:
```

### Semantics

- `name` must be the name of an existing computed column
- `new_name` defaults to `f"{name}_stored"`
- the method evaluates the computed column over the table's **physical row
  space**, not just currently live rows
- the result is written into a newly created stored column
- the original computed column is kept unchanged
- the new stored column becomes part of the table schema and, for persistent
  tables, part of the persistent table itself

### Why evaluate over physical rows

This matches the current `CTable` storage model best:

- stored columns are sized to table capacity / physical row space
- deleted rows still occupy physical positions until compaction
- indexes are built on physical stored arrays and filtered via `_valid_rows`
- `delete()` should not force special handling here

So materialization should not try to create a "live rows only" column.  It
should create a regular stored column aligned with the rest of the physical
columns.

### Snapshot semantics

The materialized column is a snapshot of the computed expression at the time of
materialization.

If any dependency column later changes:

- the materialized stored column does **not** update automatically
- users may explicitly recompute by dropping/recreating it, or by a future
  `refresh_materialized_column()` feature if one is ever added

This is intentional and keeps the feature minimal.

---

## Recommended docstring behavior

Suggested user-facing behavior:

```python
>>> t.add_computed_column("total", "price * qty")
>>> t.materialize_computed_column("total", new_name="total_stored")
>>> t.create_index("total_stored")
```

Errors:

- `ValueError` if called on a view
- `ValueError` if the table is read-only
- `KeyError` if `name` is not a computed column
- `ValueError` if `new_name` collides with an existing stored or computed column
- `TypeError` if `dtype` is incompatible with the computed values

---

## Storage and persistence model

For persistent tables, the materialized column should be stored as a **normal
persistent stored column inside the table**.

That means:

- it lives under the normal `/_cols/<new_name>` subtree
- it is represented in the normal stored-column schema
- it survives close/reopen like any other stored column
- it can use the existing `CTable.create_index()` implementation unchanged

This avoids introducing any sidecar-only or "cached virtual column" storage.

### Important consequence

After materialization, the new column is operationally just a stored column.
Any provenance metadata is optional and informational only.

---

## Internal design

### High-level implementation path

The implementation should be split into three conceptual steps:

1. validate the request and resolve the computed-column definition
2. create an empty stored column using the same mechanics as `add_column`
3. fill that stored column by evaluating the computed expression in slices over
   physical positions

### Suggested helpers

To keep `CTable` maintainable, factor the work into private helpers.

#### 1. Resolve metadata

```python
def _require_computed_column(self, name: str) -> dict:
    ...
```

Responsibilities:

- check that `name` exists in `self._computed_cols`
- return the computed-column metadata dict

#### 2. Create a stored output column

```python
def _create_empty_stored_column(
    self,
    name: str,
    dtype: np.dtype,
    *,
    cparams: dict | blosc2.CParams | None = None,
):
    ...
```

Responsibilities:

- validate `name`
- create the physical column in the same way as `add_column`
- update:
  - `self._cols`
  - `self.col_names`
  - `self._col_widths`
  - `self._schema`
- persist schema updates for file-backed tables

This helper may be a refactor extracted from existing `add_column()` logic.

#### 3. Fill from a computed expression

```python
def _fill_stored_column_from_computed(
    self,
    target_name: str,
    computed_name: str,
    *,
    dtype: np.dtype,
) -> None:
    ...
```

Responsibilities:

- evaluate the computed column over physical row positions
- write slices into `self._cols[target_name]`
- cast/coerce to the target dtype if requested

---

## Detailed algorithm

### 1. Validation

Inside `materialize_computed_column(...)`:

1. reject views:
   - `if self.base is not None: raise ValueError(...)`
2. reject read-only tables:
   - `if self._read_only: raise ValueError(...)`
3. resolve computed metadata from `self._computed_cols[name]`
4. choose `target_name = new_name or f"{name}_stored"`
5. validate `target_name`:
   - must satisfy `_validate_column_name`
   - must not exist in `self._cols`
   - must not exist in `self._computed_cols`
6. choose `target_dtype`:
   - default to computed-column dtype
   - or use explicit `dtype`

### 2. Create the destination column

Use table capacity / physical row length:

```python
capacity = len(self._valid_rows)
```

Create the new stored column with that shape.  Prefer to reuse the same default
chunk/block heuristics already used by `add_column()`:

```python
default_chunks, default_blocks = compute_chunks_blocks((capacity,))
```

This keeps the new column aligned with the existing stored-column behavior.

### 3. Evaluate and write values in slices

Reconstruct the raw `LazyExpr` for the computed column from the current table
state.  Avoid relying on any iteration path that returns only live rows.

Two valid options:

#### Option A: use the stored `lazy`

The computed definition already stores:

```python
{
    "expression": ...,
    "col_deps": ...,
    "lazy": ...,
    "dtype": ...,
}
```

In many cases, `cc["lazy"][start:stop]` should work directly.

#### Option B: reconstruct lazily from metadata

Safer and more explicit:

```python
operands = {f"o{i}": self._cols[dep] for i, dep in enumerate(cc["col_deps"])}
lazy = blosc2.lazyexpr(cc["expression"], operands)
```

This ensures the materialization always uses current column objects.

Recommended for clarity: **Option B**.

### 4. Slice loop

Use a write loop over physical positions:

```python
capacity = len(self._valid_rows)
step = self._valid_rows.chunks[0] if self._valid_rows.chunks else 65536
for start in range(0, capacity, step):
    stop = min(start + step, capacity)
    values = lazy[start:stop]
    values = np.asarray(values, dtype=target_dtype)
    self._cols[target_name][start:stop] = values
```

Notes:

- this writes all physical rows, including deleted ones
- chunk-sized slicing should limit memory overhead
- if `lazy[start:stop]` can return an NDArray in some code path, normalize via
  `values[:]` or `np.asarray(values)` as needed

### 5. Failure handling

A partial failure while filling the destination column could leave a half-written
stored column behind.

Recommended first-iteration behavior:

- best-effort cleanup if the fill step fails after destination creation:
  - drop the newly created column
  - restore schema / name registries
- if cleanup is difficult in one patch, document that the implementation should
  at least avoid leaving inconsistent in-memory metadata

A private helper that only registers the new column **after** successful fill
would be ideal, but may require more refactoring of current `add_column()` code.

---

## Interaction with existing API

### `create_index()`

No changes needed for the new materialized column:

```python
t.materialize_computed_column("total", new_name="total_stored")
t.create_index("total_stored")
```

Because `total_stored` is just a normal stored column, all current index kinds
can work unchanged.

### `drop_column()`

The materialized stored column should be droppable like any other stored column.
No special logic is needed.

### `rename_column()`

The materialized stored column should be renamable like any other stored column.
No special logic is needed.

### `drop_computed_column()`

Dropping the original computed column should not affect the stored materialized
column.  They are intentionally independent after materialization.

### `compact()`

Since the stored materialized column is just another physical stored column,
existing compaction logic should naturally include it.

---

## Schema considerations

The materialized column should be added to the normal stored schema as a normal
column entry.

The first version should **not** try to encode special "this was derived from a
computed column" semantics into the schema.

### Optional future metadata

If desired later, a small provenance block could be added to schema metadata,
for example:

```python
{
    "materialized_from": {
        "computed_column": "total",
        "expression": "o0 * o1",
        "col_deps": ["price", "qty"]
    }
}
```

But this is not required for the first implementation and should not change the
runtime semantics.

---

## Dtype policy

### Default

By default, use the computed column's declared dtype:

```python
cc["dtype"]
```

### Override

If `dtype=` is supplied, coerce each slice to that dtype during write.

If coercion fails, raise `TypeError`.

### Validation

The implementation should verify that the computed expression is 1-D and aligned
with the table's physical row space.  This is likely already guaranteed by the
current computed-column construction, but materialization should still fail
cleanly if a malformed definition slips through.

---

## Compression/layout policy

### First version

Keep this simple.

Recommended behavior:

- accept optional `cparams=`
- otherwise use the existing default `add_column()` path and chunk/block
  heuristics

Do **not** add extra knobs for:

- `chunks`
- `blocks`
- `dparams`
- target storage placement

unless they are already natural in the surrounding `CTable` API.

The first goal is to create a usable stored column, not to solve all physical
layout tuning questions.

---

## Persistence behavior

### In-memory tables

- the new column is added to `self._cols` and `self._schema`
- no special persistence work is needed

### File-backed tables

- create the new persistent column under the table's normal column storage
- update schema via existing file-backed schema save logic
- after reopen, the materialized column appears as a normal stored column

This should require no additional reopen logic beyond what already exists for
stored columns.

---

## Suggested implementation steps

### Phase 1: API and core helper refactor

1. add `CTable.materialize_computed_column(...)`
2. extract or factor stored-column creation logic from `add_column()` into a
   reusable helper
3. implement slice-based computed-expression evaluation into a stored column

### Phase 2: tests

Add tests for:

1. **basic in-memory materialization**
   - create table
   - add computed column
   - materialize to a new stored column
   - verify values match computed column on live rows

2. **deleted rows / physical positions**
   - create rows
   - delete some rows
   - materialize
   - verify visible rows match expectations
   - verify no shape inconsistencies are introduced

3. **persistent round-trip**
   - create persistent table
   - add computed column
   - materialize
   - close/reopen
   - verify new stored column exists and values persist

4. **indexing workflow**
   - materialize computed column
   - create `BUCKET` and `FULL` indexes on the new stored column
   - run representative queries

5. **error cases**
   - missing computed column
   - name collision
   - read-only table
   - view

### Phase 3: documentation

Update docs to state that computed columns themselves are still not indexable,
while showing the recommended path:

```python
t.add_computed_column("total", "price * qty")
t.materialize_computed_column("total", new_name="total_stored")
t.create_index("total_stored")
```

Suggested doc touch points:

- `doc/reference/ctable.rst`
- `doc/getting_started/tutorials/15.indexing-ctables.ipynb`
- computed-column tutorial/reference material if present

---

## Open questions

### 1. Default name

Should the default be:

- `f"{name}_stored"`
- `f"{name}_mat"`
- require `new_name`

Recommendation: default to `f"{name}_stored"` for convenience and clarity.

### 2. Reuse `add_column()` internally or not

If `add_column()` is currently too specialized around schema-field specs and
defaults, it may be cleaner to extract lower-level stored-column creation logic
instead of forcing `materialize_computed_column()` through the full public
`add_column()` path.

Recommendation: extract a small private helper for physical stored-column
creation.

### 3. Provenance metadata

Should we record where the stored materialized column came from?

Recommendation: not in the first version.  Keep semantics simple.

### 4. Refresh behavior

Should a later version support recomputation of a materialized column?

Recommendation: out of scope for now.

---

## Recommended first-patch scope

Keep the first patch intentionally narrow:

- add `CTable.materialize_computed_column(...)`
- create a new stored column from an existing computed column
- support in-memory and persistent tables
- document snapshot semantics
- add tests proving the new stored column can be indexed with existing code

Do **not** expand the feature in the first patch to:

- arbitrary source expressions
- auto-refresh
- provenance tracking
- replacement of the original computed column
- generalized `add_column(name, t.total)` support

That broader surface can be added later if this minimal feature proves useful.

---

## Summary

The simplest and most robust way to make computed-column results indexable is to
materialize them into a **new stored column** that becomes part of the table.

This plan keeps the design minimal by:

- adding one explicit method
- using snapshot semantics
- storing the result inside the normal table column layout
- reusing all existing stored-column indexing machinery unchanged

That gives users a clean workflow without taking on the much larger complexity
of true virtual computed-column indexing.
