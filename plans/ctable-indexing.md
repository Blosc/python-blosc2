# CTable Indexing Integration Plan

## Goal

Add persistent, table-owned indexing to `CTable` so that:

- indexes can be created on `CTable` columns
- persistent indexes live inside the `TreeStore` that backs the table
- `CTable.where(...)` can reuse the existing index machinery as directly as possible
- index management feels aligned with the current `NDArray` indexing API

This plan is for design and implementation guidance only. It does not assume
that all pieces must land in one patch.

> Note: indexing true virtual computed columns is a larger design problem. For
> the minimal alternative based on first materializing a computed column into a
> normal stored one, see
> [plans/materialized-computed-column.md](/Users/faltet/blosc/python-blosc2.proves/plans/materialized-computed-column.md).

## Current Situation

### What already exists

`CTable` already supports persistent storage on top of `TreeStore`:

- `/_meta`
- `/_valid_rows`
- `/_cols/<name>`

This is implemented in [src/blosc2/ctable_storage.py](/Users/faltet/blosc/python-blosc2/src/blosc2/ctable_storage.py)
and used by [src/blosc2/ctable.py](/Users/faltet/blosc/python-blosc2/src/blosc2/ctable.py).

The generic indexing engine already exists for 1-D `NDArray` targets:

- summary / bucket / partial / full indexes
- persistent descriptors in `array.schunk.vlmeta`
- sidecar arrays stored next to the indexed array
- query planning via `plan_query(...)`
- ordered reuse via `plan_ordered_query(...)`

This lives in [src/blosc2/indexing.py](/Users/faltet/blosc/python-blosc2/src/blosc2/indexing.py)
and is exposed through `NDArray.create_index()`, `drop_index()`, `rebuild_index()`,
`compact_index()`, `index()`, and `indexes`.

### What is missing

`CTable` cannot currently reuse that machinery cleanly because:

1. `CTable.where(...)` eagerly computes a boolean filter and never gives the
   planner a table-aware lazy query shape.
2. the current index engine assumes that one index belongs to one `NDArray`
   and stores its descriptor in that array's `vlmeta`.
3. persistent sidecar path derivation is based on `array.urlpath`, which places
   index files next to the array file rather than inside a table-owned subtree.
4. `CTable` has row visibility semantics through `_valid_rows`, which means
   "row still exists" and "row currently matches" are distinct concerns.

## Design Principles

The implementation should follow these rules:

- indexes are table-managed, not column-autonomous
- column indexes are still built from and logically targeted at individual column arrays
- persistent index artifacts must be part of the table store layout
- the public API should mirror existing `NDArray` indexing names where possible
- delete visibility should not force index rebuilds when it can be handled by
  post-filtering with `_valid_rows`
- planner and evaluator logic should be reused, not reimplemented from scratch
- unsupported queries must keep a correct scan fallback

## Proposed Storage Layout

Extend the persistent `CTable` layout with a reserved index subtree:

- `/_meta`
- `/_valid_rows`
- `/_cols/<name>`
- `/_indexes/<token>/...`

Recommended concrete shape:

- `/_indexes/<token>/_meta`
- `/_indexes/<token>/summary.chunk`
- `/_indexes/<token>/summary.block`
- `/_indexes/<token>/bucket.values`
- `/_indexes/<token>/bucket.bucket_positions`
- `/_indexes/<token>/bucket.offsets`
- `/_indexes/<token>/bucket_nav.l1`
- `/_indexes/<token>/bucket_nav.l2`
- `/_indexes/<token>/partial.values`
- `/_indexes/<token>/partial.positions`
- `/_indexes/<token>/partial.offsets`
- `/_indexes/<token>/partial_nav.l1`
- `/_indexes/<token>/partial_nav.l2`
- `/_indexes/<token>/full.values`
- `/_indexes/<token>/full.positions`
- `/_indexes/<token>/full_nav.l1`
- `/_indexes/<token>/full_nav.l2`
- `/_indexes/<token>/full_run.<id>.values`
- `/_indexes/<token>/full_run.<id>.positions`

Notes:

- `token` should match the current indexing token model:
  - field token for column indexes
  - normalized expression token for expression indexes
- all index payloads should stay under `/_indexes/<token>/`
- query-cache payloads, if reused for `CTable`, should also be table-owned and
  not emitted as sibling files outside the table root

## Metadata Placement

The top-level table manifest in `/_meta.vlmeta` should gain index catalog
entries and epoch counters.

Recommended fields:

```
{
    "kind": "ctable",
    "version": 1,
    "schema": {...},
    "index_catalog_version": 1,
    "value_epoch": 0,
    "visibility_epoch": 0,
    "indexes": {
        "id": {
            "name": "id",
            "token": "id",
            "target": {"source": "column", "column": "id"},
            "kind": "full",
            "version": 1,
            "persistent": True,
            "stale": False,
            "built_value_epoch": 3,
            ...
        }
    }
}
```

Notes:

- do not keep a historical list of epochs
- overwrite descriptor metadata on rebuild
- descriptors remain small; large payloads stay in `/_indexes/...`
- index catalog ownership remains at the table level, not per-column

## Public API

The `CTable` surface should mirror `NDArray` as closely as possible:

```python
table.create_index("id", kind=blosc2.IndexKind.FULL)
table.drop_index("id")
table.rebuild_index("id")
table.compact_index("id")
table.index("id")
table.indexes
```

### Initial target support

Phase 1 should support column indexes only:

- `table.create_index("id", kind=...)`
- `table.create_index(field="id", kind=...)`

Phase 2 can add expression indexes:

- `table.create_index(expression="abs(score - baseline)", operands=...)`

but only when all operands resolve to columns from the same `CTable`.

### Descriptor identity

Use one active index per target, matching current `NDArray` behavior:

- one index per column token
- one index per normalized expression token
- optional `name=` remains a label, not identity

## Query Integration Model

### Current `CTable` behavior

Today, `CTable` column comparisons produce `NDArray` or `LazyExpr` results over
physical rows, and `CTable.where(...)` then:

1. computes the filter
2. pads or trims it
3. intersects it with `_valid_rows`
4. returns a view

This is correct but fully scan-based.

### Proposed behavior

Teach `CTable.where(...)` to detect when the incoming predicate is a `LazyExpr`
that can be interpreted as a table query over table-owned columns.

For such predicates:

1. normalize the expression into a table-query descriptor
2. ask a new `CTable` planner for candidate physical row positions
3. intersect candidates with `_valid_rows`
4. evaluate any residual predicate only on surviving candidates
5. produce the final boolean mask or direct row-position set
6. return the usual `CTable` view

If any step is unsupported, fall back to the current eager full-filter path.

## Planner Strategy

Do not build a second independent indexing engine for `CTable`.

Instead, refactor the current engine into:

- reusable target normalization
- reusable index build logic
- reusable query plan primitives
- storage backends:
  - `NDArrayIndexStorage`
  - `CTableIndexStorage`

### Reusable concepts from current `indexing.py`

The following should be kept conceptually unchanged:

- index kinds: summary / bucket / partial / full
- descriptor structure where practical
- target token resolution
- exact and segment planning
- ordered full-index reuse
- full-index compaction model

### New `CTable` planner layer

Add a thin planner layer that:

- maps expression operands back to `CTable` columns
- resolves which indexed columns can participate
- requests index plans from the underlying column index implementation
- intersects or combines candidate physical positions
- reports a reason when indexed planning is not possible

For v1:

- single-column predicates should be first-class
- multi-column conjunctions should be supported when each term can be planned independently
- disjunctions can initially fall back to scan if they complicate correctness

## Row Visibility Semantics

`CTable` indexes should be defined over physical row positions, not over the
current live-row numbering.

That means:

- index payloads refer to physical positions in the backing arrays
- `_valid_rows` remains the source of truth for row visibility
- deleted rows are filtered at query execution time

This is important because deletes in `CTable` do not rewrite columns; they only
flip visibility bits.

## Epoch Model

The epoch model is intentionally small.

### Table-level counters

Store only:

- `value_epoch`
- `visibility_epoch`

Both are monotonically increasing integers in top-level table metadata.

### Per-index metadata

Each descriptor stores:

- `built_value_epoch`

Optionally later:

- `built_visibility_epoch`

but this is not required in the first implementation.

### Why this is enough

- if indexed values or row order change, the index may be invalid:
  bump `value_epoch`
- if only `_valid_rows` changes, the index still points to correct physical
  rows; execution can intersect with current visibility:
  bump `visibility_epoch`

No epoch history is retained. There is no cleanup problem because only current
scalar values are stored.

## Mutation Rules

### Mutations that should bump `value_epoch`

- `append(...)`
- `extend(...)`
- column writes through `Column.__setitem__`
- `Column.assign(...)`
- `compact()`
- `sort_by(inplace=True)`
- any future row rewrite / reorder operation
- add / drop / rename column for affected targets

### Mutations that should bump `visibility_epoch` only

- `delete(...)`

### Initial stale policy

For a first implementation, keep rebuild behavior conservative:

- if a mutation changes indexed values or row positions:
  - set affected indexes stale
  - bump `value_epoch`
- if only visibility changes:
  - do not set indexes stale
  - bump `visibility_epoch`

This is simpler than trying to preserve append-compatible incremental
maintenance on day one.

## Incremental Maintenance Policy

The current `NDArray` engine supports limited append maintenance for some index
types. `CTable` does not need to replicate all of that immediately.

Recommended rollout:

### Phase 1

- create / drop / rebuild / compact indexes
- mark value-changing mutations stale
- keep deletes valid via `_valid_rows`

### Phase 2

- optimize append / extend maintenance for column indexes
- reuse full-index append-run logic where practical
- decide whether summary / bucket / partial can be refreshed incrementally for
  appended ranges without rebuilding everything

The plan should prefer correctness and clear ownership before maintenance
optimizations.

## Ordered Queries

The smoothest integration with current `CTable` querying is:

- filtering remains `table.where(predicate)`
- ordered access is added later in a table-appropriate way

Possible later APIs:

- `table.where(expr).sort_by("id")` with index reuse
- `table.where(expr).argsort(order="id")` on a row-index result abstraction
- dedicated row-position helpers for internal use

For the first version, the main target should be indexed filtering, not full
ordered traversal.

However, the storage format should not block future ordered reuse, so `full`
indexes should still store enough information to support:

- ordered filtered row positions
- stable tie handling
- secondary refinement

## Refactoring Needed in `indexing.py`

The current implementation mixes three concerns:

1. planner / evaluator logic
2. metadata ownership
3. sidecar path naming and opening

To support `CTable`, split these concerns.

### Step A: storage abstraction

Introduce an internal storage protocol with responsibilities like:

- load/save index catalog
- derive payload location for a component
- open/store/remove sidecar arrays
- load/save query-cache catalog and payloads

Concrete implementations:

- `NDArrayIndexStorage`
- `CTableIndexStorage`

### Step B: generic target abstraction

Introduce an internal target wrapper that represents:

- base length
- dtype
- chunks / blocks
- slice access for the indexed value stream
- optional block-read helpers
- identity for query cache keys

For `CTable`, the target for a column index is the column `NDArray`, but
descriptor ownership and sidecar storage are table-owned.

### Step C: planner entry points

Keep the existing `NDArray` public entry points intact, but allow internal
planner functions to accept the new abstractions rather than hard-coded raw
`NDArray` ownership assumptions.

## `CTable` Internal Changes

### New helpers on `CTable`

Add private helpers for:

- resolving the root table from a view
- checking whether a `LazyExpr` is table-plannable
- mapping operands back to column names
- building a physical-position result into a boolean mask
- reading and writing index metadata via storage

### New helpers on `FileTableStorage`

Add persistent helpers for:

- `index_root(token)`
- `index_component_key(token, component_name)`
- create/open/delete index sidecars under `/_indexes/...`
- load/save index catalog in `/_meta`
- load/save table epoch counters

### View behavior

Views should not own indexes.

Rules:

- creating or dropping indexes on a view should raise
- querying a view may reuse root-table indexes
- planner must always combine indexed matches with the view's current mask

## Expression Index Scope

Expression indexes are valuable but should not be part of the first patch
unless the column-index path is already stable.

Recommended sequence:

1. column indexes only
2. exact-match multi-column filtering using multiple column indexes
3. expression indexes over same-table columns
4. ordered reuse

When expression indexes are added, require:

- all operands belong to the same base `CTable`
- expression normalization produces a stable token
- dependencies are stored by column name, not transient operand aliases

## Query Cache Scope

The existing query cache in `indexing.py` is array-owned.

For `CTable`, if reused, it should be table-owned as well:

- cache identity should include the table root plus query descriptor
- cache invalidation should happen on `value_epoch` changes
- visibility-only changes can either:
  - invalidate conservatively in v1, or
  - be ignored if cached results are always post-filtered through current `_valid_rows`

To keep the first version smaller, query-cache reuse can be deferred entirely.

## Validation and Reserved Names

Extend reserved internal names for persistent `CTable` layout:

- `_meta`
- `_valid_rows`
- `_cols`
- `_indexes`

If the schema compiler already blocks these, document it. If not, extend the
reserved-name validation explicitly.

## Error Handling

Recommended behavior:

- creating an index on a view: `ValueError`
- creating an index on a missing column: `KeyError`
- creating an unsupported index target: `TypeError` or `ValueError`
- querying with a non-plannable expression: silent scan fallback
- querying with malformed index metadata: clear error on open/use
- compacting a non-`full` index: same semantics as current engine

## Testing Plan

### Storage and metadata

Add tests for:

- create persistent `CTable` column index
- reopen table and see the index catalog
- verify index payloads are stored under `/_indexes/...`
- verify no sidecar siblings are emitted outside the table root layout
- drop index removes `/_indexes/<token>/...`

### Query correctness

Add tests for:

- equality and range predicates on indexed columns
- same queries on reopened persistent tables
- results match scan-based filtering
- deleted rows are excluded without rebuilding the index
- appending after index creation follows the chosen stale policy

### View semantics

Add tests for:

- view queries can reuse parent indexes
- creating indexes on views is rejected
- view mask and `_valid_rows` are both respected

### Mutation semantics

Add tests for:

- delete bumps visibility only and keeps index query correctness
- overwrite of indexed column marks index stale
- compact marks index stale
- inplace sort marks index stale
- rebuild refreshes `built_value_epoch`

### Multi-column planning

Add tests for:

- one indexed term + one unindexed residual term
- two indexed conjunctive terms
- unsupported disjunction falls back correctly

## Documentation Plan

The feature should not land with code and tests only. It needs user-facing
documentation from the start.

### Examples

Add runnable examples under `examples/ctable` covering at least:

- creating a `CTable` index on one column
- querying a `CTable` with an indexed predicate
- reopening a persistent table and reusing the index
- basic index management such as `indexes`, `index(...)`, `drop_index(...)`,
  and `rebuild_index(...)`

### Tutorial

Add a dedicated tutorial notebook at:

- `doc/getting_started/tutorials/15.indexing-ctables.ipynb`

The tutorial should explain:

- what a `CTable` index is
- how indexes relate to columns and to the table as a whole
- how persistence works for indexed tables
- what kinds of queries benefit from indexes
- what happens after deletes and other mutations
- how to inspect and maintain indexes

### API docstrings and Sphinx integration

Do not treat docstrings as optional follow-up work.

For every new public `CTable` indexing API entry point, add fully descriptive
docstrings with small examples, following the style already used elsewhere in
the codebase.

This includes, as applicable:

- `CTable.create_index(...)`
- `CTable.drop_index(...)`
- `CTable.rebuild_index(...)`
- `CTable.compact_index(...)`
- `CTable.index(...)`
- `CTable.indexes`

The docstrings should cover:

- parameters
- return values
- persistence behavior
- mutation / stale behavior where relevant
- short examples that show the intended usage

These APIs should also be integrated into the Sphinx docs so they are reachable
from the generated documentation, not only from source docstrings.

## Recommended Implementation Order

### Phase 1: storage foundations

1. add `/_indexes` reserved subtree conventions
2. extend `FileTableStorage` with index catalog and sidecar helpers
3. add table-level epoch metadata

### Phase 2: API skeleton

4. add `CTable.create_index`, `drop_index`, `rebuild_index`, `compact_index`,
   `index`, and `indexes`
5. implement build/drop/rebuild against column targets only
6. keep query path unchanged initially

### Phase 3: planner integration

7. refactor `indexing.py` storage ownership assumptions
8. add `CTable` query planner shim
9. teach `CTable.where(...)` to use indexed planning when possible
10. keep scan fallback for everything else

### Phase 4: mutation policy

11. wire `value_epoch` / `visibility_epoch`
12. mark affected indexes stale on value-changing mutations
13. keep delete visibility index-safe without rebuild

### Phase 5: follow-up optimizations

14. consider append-aware maintenance
15. consider expression indexes
16. consider ordered reuse for table queries
17. consider query-cache reuse

### Phase 6: documentation

18. add `examples/ctable` indexing examples
19. add `doc/getting_started/tutorials/15.indexing-ctables.ipynb`
20. add full public docstrings with examples for the `CTable` indexing API
21. integrate the new API and tutorial into Sphinx documentation

## Non-Goals for the First Implementation

Do not include these in the first patch unless they come almost for free:

- full expression-index support
- ordered query reuse for `CTable`
- disjunction planning across multiple indexes
- aggressive incremental maintenance for all index kinds
- index-aware query caching
- cross-table expression operands

## Future Work

One possible future storage evolution would be to make each persisted column a
subtree root instead of a single leaf object.

That would allow a layout more like:

- `/_cols/id/data`
- `/_cols/id/indexes/...`
- `/_cols/id/missing/...`
- `/_cols/id/sidecars/...`
- `/_cols/score/data`
- `/_cols/score/indexes/...`

Potential benefits:

- stronger locality between a column and its derived artifacts
- easier `rename_column()` and `drop_column()` handling
- a natural home for future per-column sidecars beyond indexes
- room for explicit missing-value bitmaps, nullability metadata, sketches, or
  other derived column structures

Potential costs:

- this would be a real `CTable` storage-schema change, not just an indexing feature
- current persisted tables would need migration or dual-layout support
- `FileTableStorage` and open/materialization logic would become more complex
- the benefit is broader than indexing, so it is better considered as part of a
  larger storage-layout revision

For that reason, this plan does not assume that redesign. It keeps the current
column-leaf layout and places indexes in a table-owned `/_indexes` subtree.

## Summary

The right model is:

- indexes are table-managed, not column-autonomous
- column indexes are still built from and logically targeted at individual
  column arrays
- persistent index artifacts live under `/_indexes`
- existing `indexing.py` logic is reused through refactoring, not duplicated
- deletes remain cheap by treating indexes as physical-row structures and
  applying `_valid_rows` at execution time
- epoch tracking stays minimal: a small number of table-level counters, not a
  growing history

This keeps the user model coherent with current `CTable` persistence and as
close as possible to the existing `NDArray` indexing API.
