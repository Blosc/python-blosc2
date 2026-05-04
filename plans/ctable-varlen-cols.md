# CTable Variable-Length Columns Implementation Plan

## Summary

Add support for variable-length list columns to `CTable` via a new logical list type:

- public schema API: `b2.list(...)`
- physical row-oriented container: `blosc2.ListArray`
- internal storage backends:
  - `VLArray` for row-oriented point updates
  - `BatchArray` for append/read efficiency

The design goal is to let users declare typed list columns in a `CTable` schema without exposing backend details unless they want to tune them.

`ListArray` should also be treated as a first-class public container for row-oriented list-valued data, not merely as an internal `CTable` helper.  At the same time, it should not be positioned as a replacement for `VLArray` or `BatchArray`; those remain the lower-level variable-length building blocks for more generic or explicitly batch-oriented workloads.

Example target API:

```python
from dataclasses import dataclass
import blosc2 as b2


@dataclass
class Product:
    code: str = b2.field(b2.string(max_length=32))
    ingredients: list[str] = b2.field(b2.list(b2.string(), nullable=True))
    allergens: list[str] = b2.field(
        b2.list(b2.string(), storage="batch", serializer="msgpack")
    )
```

---

## Final decisions already made

### Public API

- Use `b2.list(...)`, not `b2.list_(...)`.
- Use `cell` only as a documentation/design term when helpful, not as a formal API term.
- `ListArray` is the public row-oriented container abstraction for variable-length list columns.
- `ListArray` should be documented as a first-class container in its own right, useful both inside and outside `CTable`.
- `ListArray` should not be presented as replacing `VLArray` or `BatchArray`; instead, it should be positioned as the natural high-level container for row-oriented list data, while `VLArray` and `BatchArray` remain the lower-level building blocks.

### Defaults

- default `storage="batch"`
- default `serializer="msgpack"`
- default `nullable=False`
- `serializer="arrow"` remains optional and must not introduce a hard `pyarrow` dependency
- `serializer="arrow"` is only allowed with `storage="batch"`

### Null semantics

For V1, distinguish:

- `None` → null list cell
- `[]` → empty list cell

Do not support nullable items inside the list by default.

So V1 supports:

- `nullable=True|False` for the whole list cell
- no `item_nullable=True` behavior yet

### Update semantics

Support **explicit whole-cell replacement only**:

```python
t.ingredients[5] = ["salt", "sugar"]
```

Do not support implicit write-through mutation of returned Python objects:

```python
x = t.ingredients[5]
x.append("salt")  # local only
# user must reassign
```

### Batch layout policy

- default `batch_rows` should follow the column chunk size
- batch-backed list columns should use an internal append buffer in `ListArray`
- buffering lives in `ListArray`, not in `BatchArray`
- flushes occur:
  - when buffer reaches `batch_rows`
  - on explicit `flush()`
  - on persistence boundaries such as `save()` / `close()`
  - before exports that must observe all rows (e.g. `to_arrow()`)

### V1 scope

Support in V1:

- schema declaration via `b2.list(...)`
- append / extend
- row reads
- whole-cell replacement
- persistence (`save`, `open`, `load`)
- standalone `ListArray` reopen through `blosc2.open()` / `blosc2.from_cframe()`
- `head`, `tail`, `select`, and scalar-driven `where()`/view operations
- `compact()`
- `to_arrow()` / `from_arrow()`
- display / info support

Explicitly out of scope for V1:

- indexes on list columns
- computed columns over list columns
- sorting by list columns
- list-aware predicates such as contains / overlaps
- nullable items inside a list
- nested list-of-list / struct / map types
- standalone insert/delete API on `ListArray`

---

## Main design principle

Do **not** force list columns into the current scalar `np.dtype` model.

Today the schema/compiler/storage path assumes that every column:

- has a scalar `np.dtype`
- is physically stored as an `NDArray`
- can be coerced with scalar NumPy conversion rules

That is not true for list columns.

Instead, the refactor should distinguish:

- logical scalar columns
- logical list columns

and separately distinguish their physical storage:

- scalar column → `NDArray`
- list column → `ListArray`

This keeps the scalar path fast and clean while adding a first-class path for variable-length lists.

---

## High-level architecture

### Logical layer

Add a new schema descriptor:

- `ListSpec`

Keep existing scalar specs, but conceptually move toward:

- `ColumnSpec`
  - `ScalarSpec`
  - `ListSpec`

This can be implemented either by introducing explicit base classes or by broadening the meaning of the current spec system.  The important part is that `CompiledColumn` and `CTable` must stop assuming every spec has a scalar `dtype`.

### Physical layer

Add a new container:

- `blosc2.ListArray`

`ListArray` is cell-oriented:

- `arr[i]` returns one list cell or `None`
- `arr[i:j]` returns a Python `list` of cells
- `arr[i] = value` replaces one cell
- `append(value)` appends one cell
- `extend(values)` appends many cells

Internally it wraps one of:

- `VLArray`
- `BatchArray`

### CTable layer

Teach `CTable` to manage two families of physical columns:

- scalar columns backed by `NDArray`
- list columns backed by `ListArray`

`CTable` should understand list columns at the schema and storage levels, but should not need backend-specific logic beyond creation/open/flush/update hooks.

---

## Phase 1: Schema system changes

## 1.1 Add `b2.list(...)`

Add a new public builder in `src/blosc2/schema.py`:

```python
def list(
    item_spec,
    *,
    nullable=False,
    storage="batch",
    serializer="msgpack",
    batch_rows=None,
    items_per_block=None,
): ...
```

Initial accepted parameters:

- `item_spec`
  - typically a scalar spec such as `b2.string()` or `b2.int32()`
- `nullable`
  - whether the whole list cell may be `None`
- `storage`
  - `"batch"`, `"vl"`
- `serializer`
  - `"msgpack"`, `"arrow"`
- `batch_rows`
  - optional row count per persisted batch for batch backend
- `items_per_block`
  - forwarded to `BatchArray` when backend is batch

Validation rules for V1:

- `storage` must be `"batch"` or `"vl"`
- `serializer` must be `"msgpack"` or `"arrow"`
- if `storage == "vl"`, serializer must be `"msgpack"`
- if `serializer == "arrow"`, storage must be `"batch"`
- `item_spec` should initially be restricted to scalar specs

## 1.2 Introduce `ListSpec`

Add a new schema descriptor class with at least:

- `python_type = list`
- `item_spec`
- `nullable`
- `storage`
- `serializer`
- `batch_rows`
- `items_per_block`

Methods analogous to existing specs:

- `to_metadata_dict()`
- optional `display_label()` helper or equivalent

Suggested serialized form:

```json
{
  "kind": "list",
  "item": {"kind": "string", "max_length": 64},
  "nullable": true,
  "storage": "batch",
  "serializer": "msgpack",
  "batch_rows": 65536,
  "items_per_block": 256
}
```

## 1.3 Broaden `field()` acceptance

`b2.field(...)` should accept any valid column spec, not just scalar specs.

The implementation contract becomes:

- scalar field spec allowed
- list field spec allowed

No user-visible API change beyond this.

---

## Phase 2: Schema compiler changes

## 2.1 Relax `CompiledColumn`

`CompiledColumn` currently assumes a scalar `dtype`.  Refactor it so list columns are first-class.

Target shape:

- `name`
- `py_type`
- `spec`
- `default`
- `config`
- `display_width`
- optional scalar dtype information only when applicable

There are two acceptable implementation styles:

### Option A: minimal change

Keep `dtype` on `CompiledColumn`, but allow it to be `None` for non-scalar columns.

### Option B: cleaner long-term change

Replace mandatory `dtype` with something like:

- `storage_dtype: np.dtype | None`
- `logical_kind`
- convenience properties such as `is_scalar`, `is_list`

Recommendation: choose the smallest refactor that avoids fake object dtypes.

## 2.2 Update annotation validation

Current annotation validation is scalar-oriented.  Extend it to support:

```python
ingredients: list[str] = b2.field(b2.list(b2.string()))
```

Compiler responsibilities:

- inspect `typing.get_origin(annotation)`
- inspect `typing.get_args(annotation)`
- validate that `list[...]` annotations match `ListSpec`
- validate that the item annotation matches `item_spec`

For V1, support:

- built-in `list[T]`
- likely `typing.List[T]` as a compatibility path if desired

Restrict V1 item annotations to scalar item types.

## 2.3 Schema serialization/deserialization

Extend `schema_to_dict()` / `schema_from_dict()` so list specs round-trip through stored schema metadata.

This includes:

- emitting `kind="list"`
- recursively serializing `item_spec`
- restoring `ListSpec` on reopen/load

---

## Phase 3: Add `ListArray`

## 3.1 Public role

Create a new file:

- `src/blosc2/list_array.py`

And export it from:

- `src/blosc2/__init__.py`

`ListArray` should be the row-oriented facade used by `CTable` and also a standalone public container for users working with row-oriented list-valued data.

It should not expose `BatchArray`'s native batch-oriented semantics.

Documentation should encourage `ListArray` for typed, row-oriented list data, while still keeping `VLArray` and `BatchArray` visible as the lower-level containers for arbitrary object payloads and explicitly batch-oriented workflows.

## 3.2 Core API

Initial API:

- constructor taking list spec / backend hints / storage kwargs
- `append(value)`
- `extend(values)`
- `flush()`
- `close()`
- `__enter__()` / `__exit__()`
- `__getitem__(index|slice)`
- `__setitem__(index, value)`
- `__len__()`
- `__iter__()`
- `copy(**kwargs)`
- `info`
- `to_arrow()` when possible
- `from_arrow()` if useful as a constructor helper

V1 read behavior:

- `arr[i]` → `list | None`
- `arr[i:j]` → `list[list | None]`

No item-level API like `arr.items` is required for V1.

## 3.3 Validation/coercion inside `ListArray`

`ListArray` should validate cell values against the provided `ListSpec`.

Rules for V1:

- `None` allowed only if `nullable=True`
- otherwise value must be list-like
- strings / bytes are not accepted as list-like cells
- each item must satisfy `item_spec`
- `None` items rejected for V1

The goal is that `ListArray` can be safely used both inside and outside `CTable`.

## 3.4 Backend selection

Selection policy:

- `storage="batch"` → batch backend
- `storage="vl"` → VL backend
- `serializer="arrow"` only valid with batch backend
- default backend = batch
- default serializer = msgpack

Implementation note: backend choice should be explicit in metadata and persisted schema, not inferred later from heuristics.

---

## Phase 4: `ListArray` backend implementation

## 4.1 VL backend

Map one logical cell to one VLArray entry.

Properties:

- simplest implementation
- natural row-level replacement
- no internal buffer needed

Implementation behavior:

- `append(cell)` → `VLArray.append(cell)`
- `extend(cells)` → `VLArray.extend(cells)`
- `__getitem__(i)` → `VLArray[i]`
- `__setitem__(i, cell)` → `VLArray[i] = cell`

This backend is the easiest one and should be implemented first to stabilize list semantics.

## 4.2 Batch backend

Map many logical cells to one persisted batch in `BatchArray`.

Persisted representation for V1 msgpack path:

- one batch = list of cells
- each cell = `None` or Python list of scalar items

Arrow-serializer path, implemented after the msgpack path but within the same overall design:

- one batch = Arrow array where each slot corresponds to one list cell
- type = `list<item_type>`

## 4.3 Internal append buffer

`ListArray(storage="batch")` maintains:

- persisted batches in a `BatchArray`
- a pending in-memory Python list of cells not yet flushed

Suggested internal state:

- `_store`: `BatchArray`
- `_pending_cells: list`
- `_batch_rows: int`
- mapping helpers derived from persisted batch lengths

Append flow:

- validate cell
- append to `_pending_cells`
- if `len(_pending_cells) >= batch_rows`, flush full batches

Extend flow:

- validate each cell
- fill pending buffer
- flush full batches as needed

Flush flow:

- write full `batch_rows` groups to `BatchArray.append(...)`
- keep any tail cells pending unless caller requested a full final flush
- on explicit final flush / close / save, write tail as last batch

## 4.4 Logical indexing in batch backend

`ListArray` must expose row-level indexing across:

- persisted cells in `BatchArray`
- pending cells in `_pending_cells`

This requires row → batch lookup for persisted rows.

Suggested approach:

- rely on `BatchArray`'s stored batch lengths and prefix sums for persisted portion
- append pending tail logically after persisted rows

Point update behavior:

- if target row is in pending cells: replace in memory
- if target row is persisted: load batch, replace cell, rewrite entire batch

This should be supported but documented as more expensive than VL-backed updates.

## 4.5 Flush semantics

`ListArray.flush()` should:

- write all pending cells, including the tail
- leave `_pending_cells` empty

Automatic flushes should occur on:

- buffer full
- explicit `flush()`
- `close()` / context-manager exit
- `CTable.save()`
- `CTable.to_arrow()`
- any persistence-sensitive operation that must see all rows on disk

---

## Phase 5: `CTable` storage abstraction changes

## 5.1 Broaden `TableStorage`

Current `TableStorage.create_column()` / `open_column()` assume `NDArray`.  Extend storage abstraction to support list columns.

Recommended shape:

- `create_scalar_column(...)`
- `open_scalar_column(...)`
- `create_list_column(...)`
- `open_list_column(...)`

Alternative minimal path:

- keep `create_column(...)` but branch on compiled spec kind

Recommendation: use explicit separate methods if the diff stays manageable; it makes the abstraction cleaner.

## 5.2 File-backed layout

Current scalar layout is under:

- `/_cols/<name>`

Keep that logical namespace for list columns too.

Possible physical forms:

- `/_cols/<name>` points directly to the underlying backend object (`VLArray` or `BatchArray`), tagged so it reopens logically as `ListArray`
- `/_cols/<name>` plus optional side metadata stored in schema

For V1, prefer storing backend configuration in the schema metadata and keeping the on-disk object itself as the concrete backend container.

## 5.3 In-memory layout

In-memory `CTable` should keep list columns as live `ListArray` objects.

No persistence is needed there beyond normal `save()` behavior.

---

## Phase 6: `CTable` core changes

## 6.1 Column creation/opening

During table creation/open/open-from-schema:

- scalar compiled columns create/open `NDArray`
- list compiled columns create/open `ListArray`

`self._cols` may continue to map names to physical column objects, but code using `self._cols[name]` must stop assuming every entry is an `NDArray`.

### 6.1.1 Physical length vs capacity

Scalar columns remain capacity-based and grow with `_valid_rows`.

List columns should instead be treated as append-sized physical stores:

- their physical length tracks the number of written physical rows
- they are not preallocated out to `len(_valid_rows)`
- `_grow()` should continue to resize scalar columns and `_valid_rows`, but should not pad list columns with placeholder cells

This means `CTable` internals must stop assuming every stored column has physical length equal to `len(_valid_rows)`.

Logical row resolution should always go through physical row positions that are actually written.

## 6.2 Row coercion path

Current `_coerce_row_to_storage()` is scalar-only.  Refactor it to branch on column kind.

For scalar columns:

- keep current NumPy scalar coercion

For list columns:

- validate and normalize through list-spec logic
- store Python list / `None` as-is for handoff to `ListArray`

## 6.3 Append path

Current `append()` loops over columns and assigns directly into scalar arrays.  Refactor:

- scalar column: `ndarray[pos] = scalar`
- list column: `list_array.append(cell)`

Important invariants:

- all stored columns must stay logically aligned by row index
- list columns append one physical cell per newly written physical row position
- `_last_pos` remains the source of truth for the next physical row id

This means `append()` must ensure each list column receives exactly one new cell per appended row.

### Write coordination / partial-failure safety

This deserves explicit care because mixed scalar/list writes are no longer a single homogeneous NDArray operation.

V1 should guarantee at least:

- validate and normalize the full incoming row or batch before mutating storage
- mark `_valid_rows` only after all column writes for the row(s) succeed
- avoid leaving logically visible partial rows after a failure

For list columns in particular, the implementation should prefer staging writes in memory where possible, or otherwise provide best-effort rollback for per-row appends that fail after some columns were already updated.

## 6.4 Extend path

Current `extend()` materializes column arrays and writes slices into NDArrays.  For list columns, this should become backend-aware.

Suggested behavior:

- scalar columns: keep vectorized path
- list columns: collect Python list of cells and call `ListArray.extend(cells)`

This may be less vectorized than the scalar path, which is acceptable for V1.

## 6.5 Delete behavior

No special physical delete support is required for V1.

`CTable.delete()` already uses `_valid_rows`, so list columns can simply remain append-only physically while logical row deletion is controlled by the validity mask.

## 6.6 Row reads

`_Row.__getitem__` and column reads must support list columns.

Expected behavior:

- row access on a list column returns the corresponding list cell or `None`
- sliced column access on a list column returns a Python `list` of cells

## 6.7 Column wrapper behavior

Current `Column` logic is NumPy/NDArray-centric.  For V1, update it carefully for list columns.

Key points:

- `Column.dtype` is not meaningful for list columns and should return `None` rather than a fake NumPy dtype
- many scalar comparison/aggregate methods should reject list columns cleanly
- `Column.__getitem__` must work for logical row indexing on list columns
- `Column.__setitem__` should support whole-cell replacement for list columns

Recommendation:

- branch internally on compiled spec kind
- do not try to make list columns mimic NumPy arrays

## 6.8 String representation and info

Update display and info methods so list columns show a logical type label like:

- `list[string]`
- `list[int32]`

For previews, show Python repr-style cells with truncation.

Statistics in `describe()` for list columns can be omitted in V1 or show a message like:

- `(stats not available for list columns)`

---

## Phase 7: Arrow interoperability

## 7.1 Keep Arrow optional

`pyarrow` must remain optional.

Rules:

- importing / using list columns with msgpack default must not require `pyarrow`
- `serializer="arrow"` should raise a clear `ImportError` when `pyarrow` is absent
- `CTable.to_arrow()` / `from_arrow()` may already be optional-import based and should stay that way

## 7.2 `CTable.to_arrow()`

Extend `to_arrow()` so list columns export to Arrow list arrays.

For msgpack-backed list columns:

- materialize Python cells
- build `pa.array(values)` with appropriate list type when possible

For Arrow-backed batch list columns later:

- a faster path may reuse Arrow-native data more directly if feasible

V1 success criterion:

- correct Arrow output, even if implemented through Python materialization

## 7.3 `CTable.from_arrow()`

Extend Arrow import to recognize Arrow list fields and create `ListSpec` columns.

Initial supported Arrow list cases for V1:

- `list<string>`
- possibly `large_list<string>` if trivial to normalize
- optionally numeric item types if easy to map consistently

Import policy:

- create `b2.list(item_spec, storage="batch", serializer="msgpack")` by default unless caller later gains a way to override
- append row cells into `ListArray`

The important part for V1 is round-tripping list columns through `CTable`.

---

## Phase 8: Persistence and reopen behavior

## 8.1 Schema metadata

Persist list-specific metadata in the table schema.

Each list column should carry at least:

- `kind="list"`
- serialized `item_spec`
- `nullable`
- `storage`
- `serializer`
- `batch_rows` if relevant
- `items_per_block` if relevant

## 8.2 Container reopen behavior

When a table is reopened:

- compiled schema reconstructs `ListSpec`
- storage layer opens the concrete list column backend
- `CTable` re-wraps it as `ListArray`

For standalone use, persistent `ListArray` containers should also reopen as `ListArray` through the generic dispatch path.

### 8.2.1 Layered metadata tagging

`ListArray` should have its own explicit container tag in fixed metadata, while preserving the underlying backend tag.

Examples:

- batch-backed `ListArray`:
  - `meta["batcharray"] = {...}`
  - `meta["listarray"] = {...}`
- VL-backed `ListArray`:
  - `meta["vlarray"] = {...}`
  - `meta["listarray"] = {...}`

This keeps both identities available:

- physical identity: the underlying storage container (`BatchArray` or `VLArray`)
- logical identity: the object should reopen as `ListArray`

Suggested `meta["listarray"]` payload:

```json
{
  "version": 1,
  "backend": "batch",
  "serializer": "msgpack",
  "nullable": false,
  "item_spec": {"kind": "string", "max_length": 64},
  "batch_rows": 65536,
  "items_per_block": 256
}
```

The exact payload can be trimmed, but it should at least record:

- format version
- backend kind
- serializer
- nullability
- serialized item spec
- backend-specific layout hints when relevant

Recommendation: store this in `schunk.meta`, not only in `vlmeta`, because it defines the container kind used for reopen dispatch.

### 8.2.2 Generic reopen dispatch

`blosc2.open()` and `blosc2.from_cframe()` should prefer `ListArray` when `meta["listarray"]` is present.

Suggested dispatch priority:

1. `listarray`
2. `batcharray`
3. `vlarray`
4. existing fallback behavior

This makes the generic open path return the logical container type rather than exposing the raw backend by default.

Advanced users can still reach the lower-level backend explicitly if needed.

### 8.2.3 `ListArray` reopen constructor

`ListArray` should support an internal reopen hook such as:

```python
ListArray(_from_schunk=schunk)
```

This path should:

- validate the `listarray` tag
- validate consistency with the backend tag (`batcharray` or `vlarray`)
- reconstruct the correct backend wrapper
- return a row-oriented `ListArray` object

## 8.3 Save/load and flush coordination

Ensure all list columns are flushed before:

- saving to disk
- serializing table data for export if needed
- closing persistent stores

Add a helper on `CTable` such as an internal `_flush_varlen_columns()` used by:

- `save()`
- `to_arrow()`
- close/discard paths

---

## Phase 9: Testing plan

Add focused tests rather than trying to cover the entire matrix at once.

## 9.1 Schema/compiler tests

New tests for:

- `b2.list(...)` construction
- invalid storage/serializer combinations
- annotation matching for `list[str]`
- schema serialization/deserialization of `ListSpec`

## 9.2 `ListArray` tests

Standalone tests covering both backends:

- append / extend
- `None` vs `[]`
- reject nullable items in V1
- row reads
- slice reads
- whole-cell replacement
- negative indexing
- flush behavior, including `close()` / context-manager flush-on-exit
- reopen behavior for persistent stores
- `blosc2.open()` / `blosc2.from_cframe()` dispatch returning `ListArray`
- read-only handling where applicable

Batch backend specific tests:

- pending-buffer reads before flush
- automatic flush on buffer full
- update in pending region
- update in persisted region
- correct length across persisted + pending regions

VL backend specific tests:

- direct row replacement

## 9.3 `CTable` tests

Add `CTable` tests for:

- schema with one list column
- schema with scalar + list columns mixed
- append row with list column
- extend rows with list column
- read rows back
- `head`, `tail`, `select`
- scalar-driven `where()` / view operations with list columns carried through correctly
- `compact()` with list columns
- row deletion via `_valid_rows`
- reopen persistent table
- `save()` / `load()` round-trip
- `to_arrow()` / `from_arrow()` with list columns

## 9.4 Non-goal tests for V1

Do not add tests for unsupported features such as:

- list column indexes
- sorting by list column
- computed expressions over lists
- nullable items inside lists

Instead, add clear failure-path tests where appropriate.

---

## Phase 10: Documentation plan

Update documentation incrementally.

## 10.1 Reference docs

Add docs for:

- `b2.list(...)`
- `ListArray`
- `CTable` list column support

## 10.2 Tutorial/examples

Add at least one example such as:

- products with `ingredients: list[str]`

Show:

- schema declaration
- append / extend
- distinction between `None` and `[]`
- whole-cell replacement
- save/reopen
- Arrow export if `pyarrow` is installed

## 10.3 Design notes in docs

Explain briefly:

- `cell` as a descriptive concept only
- batch vs VL backend tradeoffs
- why `msgpack` is the default
- why returned Python lists must be reassigned after mutation

---

## Recommended implementation order

To reduce risk, implement in this order:

1. **Schema groundwork**
   - add `ListSpec`
   - add `b2.list(...)`
   - update schema serialization and compiler

2. **Standalone `ListArray` with VL backend first**
   - easiest path to stabilize list semantics
   - validates `None` vs `[]`, whole-cell replacement, persistence
   - includes layered tagging and standalone reopen through `blosc2.open()` / `from_cframe()`

3. **`CTable` integration for VL-backed list columns**
   - proves scalar/list coexistence in compiler and core table paths

4. **Batch-backed `ListArray` with pending buffer**
   - implement `batch_rows` buffering
   - add flush semantics and persisted/pending indexing

5. **`CTable` integration for batch-backed list columns**
   - update save/open/load and flush coordination

6. **Arrow import/export support for list columns**
   - keep optional
   - start with materialization-based path

7. **Docs and broader test coverage**

This staged rollout makes it easier to separate:

- logical list semantics
- `CTable` schema/compiler changes
- batch buffering complexity

---

## Practical code touch points

Expected Python files to update or add:

### New files

- `src/blosc2/list_array.py`
- `tests/test_list_array.py`
- `tests/test_ctable_varlen.py` or equivalent
- `doc/reference/list_array.rst` or a combined varlen/list reference page
- example file under `examples/ctable/`

### Existing files likely to change

- `src/blosc2/__init__.py`
- `src/blosc2/schema.py`
- `src/blosc2/schema_compiler.py`
- `src/blosc2/ctable.py`
- `src/blosc2/ctable_storage.py`
- likely `src/blosc2/schema_validation.py`
- likely `src/blosc2/schema_vectorized.py`
- `src/blosc2/schunk.py` for standalone reopen dispatch
- `src/blosc2/core.py` for generic `open()` / `from_cframe()` dispatch for `ListArray`
- docs under `doc/reference/ctable.rst` and related tutorial pages

---

## Open follow-up items after V1

These are intentionally postponed, not rejected:

- `item_nullable=True`
- nested list-of-list
- struct / map item types
- list-aware query predicates
- indexing for membership tests
- sorting/grouping semantics for list columns
- optimized Arrow-native batch representation
- convenience APIs for explicitly opening the raw backend (`VLArray` / `BatchArray`) behind a `ListArray` when advanced users want to bypass the logical container

---

## Short rationale for the chosen defaults

### Why `b2.list(...)`

It reads naturally next to `list[str]` annotations and fits the rest of the schema-builder API.

### Why `ListArray` should be first-class

It gives users a clear row-oriented abstraction for list-valued data that is useful both on its own and as the natural physical model for list columns inside `CTable`, while still preserving `VLArray` and `BatchArray` as lower-level containers.

### Why `msgpack` default

It avoids a hard `pyarrow` dependency and works uniformly with both `VLArray` and `BatchArray`.

### Why `storage="batch"` default

It better matches append/scan-oriented table workloads and Parquet-like usage, while `VLArray` remains available for update-heavy cases.

### Why whole-cell replacement only

It keeps behavior explicit and avoids surprising write-through mutation of returned Python lists.

### Why `None` vs `[]`

The distinction is valuable and common, while item-level nullability can be added later without breaking the model.

---

## Outcome expected from V1

After this work:

- `CTable` should be able to host list columns in a way that is:
  - typed at the schema level
  - persistent
  - row-addressable
  - backend-tunable
  - install-light by default
  - compatible with optional Arrow export/import

- `ListArray` should stand as a public reusable container for row-oriented list-valued data.

This should happen without compromising the current scalar-column fast path or displacing `VLArray` / `BatchArray` from their lower-level roles.
