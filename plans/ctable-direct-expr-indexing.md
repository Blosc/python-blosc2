# CTable Direct Expression Indexing Plan

## Goal

Add **direct expression indexes** to `CTable` so users can index expressions over
stored table columns **without** first creating a computed column or
materializing one into a stored snapshot column.

The intended end-state API is:

```python
t.create_index(expression="price * qty", kind=blosc2.IndexKind.FULL, name="total")
t.create_index(
    expression="abs(price - baseline)", kind=blosc2.IndexKind.BUCKET, name="abs_delta"
)
```

This should work for both in-memory and persistent tables, and should integrate
with the existing `CTable.where(...)` and ordered-query machinery so that table
queries can reuse the same index engine concepts already available for
`NDArray` expression indexes.

This plan is intentionally separate from
[plans/materialized-computed-column.md](/Users/faltet/blosc/python-blosc2.proves/plans/materialized-computed-column.md).
That plan is about **making a virtual result physical**.  This plan is about
**keeping the expression virtual while making the indexing/query planner aware
of it**.

---

## Why this is a separate effort

The current materialization plan has a very different spirit:

- evaluate once
- store a normal physical column
- reuse existing stored-column indexing unchanged

By contrast, direct table expression indexing requires the indexing engine to
understand:

- table-owned expression targets
- dependencies across multiple stored columns from the same `CTable`
- table row visibility via `_valid_rows`
- table-owned persistence/catalog state
- planner integration for expression predicates and orderings

So this plan is **not** a small extension of materialized computed columns.
It is a planner/index-engine feature.

---

## User-facing motivation

Users should be able to accelerate derived predicates and orderings without
polluting the logical table schema with temporary computed-column names.

Examples:

```python
# Filter acceleration
t.create_index(expression="price * qty", kind=blosc2.IndexKind.PARTIAL, name="total")
view = t.where((t["price"] * t["qty"]) > 100)

# Ordered reuse
t.create_index(
    expression="abs(score - baseline)", kind=blosc2.IndexKind.FULL, name="abs_delta"
)
result = t.where(
    (abs(t["score"] - t["baseline"]) >= 5) & (abs(t["score"] - t["baseline"]) < 20)
)
```

This is analogous to current `NDArray` support such as:

```python
arr.create_index(expression="abs(x)", kind=blosc2.IndexKind.FULL, name="abs_x")
```

but generalized from:

- expression over fields of the same structured `NDArray`

into:

- expression over columns of the same `CTable`

---

## Non-goals for the first iteration

This plan does **not** aim to add all possible expression-index features at
once.

The first iteration should **not** try to add:

- indexing expressions whose operands come from multiple different `CTable`
  objects
- creating direct indexes on table views
- automatic indexing of arbitrary computed columns just because they exist
- support for expressions that depend on computed columns
- support for expressions whose dependencies include non-column external arrays
- expression indexes as independent schema columns
- replacement of computed columns or automatic conversion from computed columns
  into direct expression indexes
- cross-table joins or multi-table query planning
- full general-purpose optimizer rewrites for every `LazyExpr`

For v1, direct expression indexes should be restricted to expressions whose
operands resolve to **stored columns from the same base table**.

---

## Current implementation constraints

### What already exists

The current generic index engine in `src/blosc2/indexing.py` already supports:

- field indexes on `NDArray`
- expression indexes on `NDArray`
- normalized expression target descriptors
- one active index per normalized target token
- persistent and in-memory descriptors
- planner support for predicates/orderings over indexed expression targets

The current `CTable` indexing support in `src/blosc2/ctable.py` currently adds:

- table-owned index catalogs
- table-owned persistent sidecar placement
- view rejection for index management
- column-only indexing based on physical stored columns

### What blocks direct table expression indexing today

The current `NDArray` expression-index implementation assumes:

1. indexes are only supported on `NDArray`
2. the indexed object is 1-D
3. expression operands resolve to the **same base array passed to
   `create_index()`**
4. the values for the target can be read from that one base array

This matches structured-array expressions like `abs(x)` where `x` is a field of
one array, but it does not match `CTable` expressions like:

```python
price * qty
```

because:

- `price` and `qty` are separate physical `NDArray` columns
- the shared owner is the **table**, not a single `NDArray`
- table queries also have `_valid_rows` visibility semantics layered over the
  physical row space

The current `CTable.create_index(...)` explicitly rejects computed columns
because they have no physical storage.  That remains correct for the current
column-only API, but direct expression indexing would provide a separate path
that targets the **expression itself**, not a virtual column object.

---

## Design principles

The implementation should follow these rules:

- do **not** build a second independent indexing engine for `CTable`
- reuse the current target descriptor model where possible
- generalize expression-target normalization from “same `NDArray`” to “same
  container”
- keep direct expression indexes **table-owned**, not column-owned
- index build should operate over the table's **physical row space**
- `_valid_rows` should remain a query-time visibility filter, not something that
  forces a rebuild on every delete
- unsupported expressions must keep a correct full-scan fallback
- direct expression indexes and computed columns should remain distinct features

---

## Conceptual model

### Distinguish three things

These should remain separate concepts:

1. **stored column**
   - physical data in `self._cols[name]`
   - part of the row schema
   - directly indexable today

2. **computed column**
   - user-visible virtual column in `self._computed_cols`
   - persisted as metadata
   - reusable in display/select/sort/aggregates
   - not inherently an index target in v1

3. **direct expression index**
   - optimizer structure for an expression over stored columns
   - does not create a new logical column
   - may have a human-readable `name=` label
   - identity is still the normalized expression token, not the label

This separation is important.  Users should not need to create computed columns
only to gain expression indexing.

---

## Proposed public API

## `CTable.create_index(...)`

Extend the existing method from column-only indexing to accept expression
targets too.

Suggested signature:

```text
def create_index(
    self,
    col_name: str | None = None,
    *,
    field: str | None = None,
    expression: str | None = None,
    operands: dict | None = None,
    kind: blosc2.IndexKind = blosc2.IndexKind.BUCKET,
    optlevel: int = 5,
    name: str | None = None,
    build: str = "auto",
    tmpdir: str | None = None,
    **kwargs,
) -> CTableIndex:
```

### Target-selection semantics

- `col_name` and `field` are aliases for the stored-column target
- `expression` selects a direct expression target
- `operands` is only valid together with `expression`
- `col_name`/`field` and `expression` are mutually exclusive

### Default operands behavior

For expression indexes on a table, if `operands` is omitted, default to the
stored columns of the table:

```python
operands = self._cols
```

or a more explicit mapping of column name -> column object.

This should mean that:

```python
t.create_index(expression="price * qty")
```

works naturally when `price` and `qty` are stored table columns.

### Optional explicit operands

Explicit `operands=` should be accepted when all operands resolve to columns
from the same table.  For example:

```python
t.create_index(
    expression="abs(a - b)",
    operands={"a": t._cols["price"], "b": t._cols["baseline"]},
)
```

but any operand set that resolves to a different table, a view, or an external
array should be rejected in v1.

---

## Expression-target identity

Expression indexes should follow the same identity model as `NDArray`:

- one active index per normalized target token
- optional `name=` is a label, **not** identity

So these should coexist:

```python
t.create_index(expression="price * qty", kind=blosc2.IndexKind.FULL, name="total")
t.create_index(
    expression="abs(price - baseline)", kind=blosc2.IndexKind.BUCKET, name="abs_delta"
)
```

but repeated creation of the same normalized expression target should raise an
"index already exists" error unless the existing one is dropped or rebuilt.

### Why token identity matters

Using the normalized expression token as identity ensures:

- exact target lookup is deterministic
- rebuild/drop/compact can stay aligned with current `NDArray` behavior
- user-facing `name=` can remain descriptive without affecting correctness

---

## Target normalization refactor

### Current shape

Today, the index engine normalizes expression targets under an implicit rule:

> all operands must resolve to the same `NDArray` passed to `create_index()`

This is the logic that must be generalized.

### Proposed minimal generalization

Refactor the target normalization path into two layers:

1. **shared expression canonicalization**
   - parse expression AST
   - canonicalize operand names
   - collect dependencies
   - produce normalized expression key / target descriptor

2. **container-specific operand resolution and validation**
   - for `NDArray`: all operands must resolve to the same base array
   - for `CTable`: all operands must resolve to columns from the same base table

### Recommended shared helper split

A minimal refactor can keep implementation simple by introducing helper families
rather than a large class hierarchy.

For example:

```python
_normalize_create_index_target_ndarray(array, field, expression, operands)
_normalize_create_index_target_ctable(table, field, expression, operands)

_normalize_expression_target_with_resolver(expression, operands, resolver)
```

where the resolver is responsible for turning each operand into a canonical
container+dependency pair.

### Container-specific invariant

The new table-specific invariant should be:

> expression index operands must resolve to columns from the same `CTable`
> passed to `create_index()`

This is the exact analogue of the current `NDArray` rule.

---

## Operand resolution model

### NDArray case

Keep current semantics:

- field `x` in a structured array resolves to `(array, "x")`
- scalar array target resolves to `(array, None)`

### CTable case

For a table, operands should resolve to canonical column dependencies such as:

- `price` -> `(table, "price")`
- `qty` -> `(table, "qty")`

### Allowed operand forms in v1

To keep this tight, allow only operand values that can be unambiguously mapped
back to one stored column in the same table, for example:

- `table._cols[name]`
- `table.cols[name]` / `table[name]` if those resolve to an identifiable column
  wrapper with a stable backing name
- default implicit names from stored columns when `operands` is omitted

### Disallowed operand forms in v1

Reject operands that resolve to:

- computed columns
- columns from a different table
- any table view
- arbitrary external `NDArray` objects
- temporary arrays detached from the table

This avoids ambiguities around persistence, invalidation, and query ownership.

---

## Target descriptor format

The current `NDArray` expression target descriptor shape should be reused where
possible.

Recommended shape for a table expression target:

```python
{
    "source": "expression",
    "expression": "price * qty",
    "expression_key": "o0 * o1",  # or whatever canonical form the shared normalizer uses
    "dependencies": ["price", "qty"],
}
```

For stored columns, keep a column target shape analogous to the existing table
column descriptor:

```python
{
    "source": "column",
    "column": "price",
}
```

### Why reuse this shape

Reusing the target descriptor model minimizes churn in:

- token generation
n- descriptor serialization
- rebuild/drop/compact lookup logic
- planner integration based on target identity

---

## Table-owned catalog and storage

Expression indexes should remain table-owned, exactly like current column
indexes on `CTable`.

### Persistent storage layout

Keep using the table-owned index subtree:

- `/_indexes/<token>/...`

No separate storage layout is needed just because the target is an expression.
The only difference is the `target` metadata in the descriptor.

### Catalog entries

The table index catalog should be able to store entries like:

```text
{
    "token": "expr:<normalized-token>",
    "target": {
        "source": "expression",
        "expression": "price * qty",
        "expression_key": "o0 * o1",
        "dependencies": ["price", "qty"],
    },
    "name": "total",
    "kind": "full",
    "persistent": True,
    "stale": False,
    "built_value_epoch": 7,
    ...
}
```

### Catalog identity

The catalog key should remain the token, not the user label.  This allows:

- multiple different expression indexes
- deterministic lookup
- rebuild/drop by target or by label when unambiguous

The `CTableIndex` handle may still expose `name`, `kind`, and `target`
information as today.

---

## Value extraction for table targets

The index build engine needs a way to obtain target values for both stored
column and expression targets.

### Column target

For a stored-column target, use the physical stored array directly:

```python
table._cols[col_name]
```

### Expression target

For an expression target, reconstruct a lazy expression using the current table
columns:

```python
operands = {dep: table._cols[dep] for dep in target["dependencies"]}
lazy = blosc2.lazyexpr(target["expression"], operands)
```

Then evaluate it over **physical row positions**, not only live rows.

### Why physical rows matter

This matches current table storage/index semantics:

- stored columns are aligned by physical position
- deleted rows remain present physically until compaction
- query evaluation already intersects with `_valid_rows`
- full index positions should therefore refer to physical row offsets

### Implementation guidance

Do not require whole-array eager materialization for large tables.
The preferred path is to support chunked iteration or slice evaluation over the
physical row space using the same chunk/block heuristics already used in the
index builder.

---

## Planner/query integration

Direct expression indexes only become useful if `CTable` queries can reuse them.

### Current `CTable.where(...)` behavior

Today, predicates are evaluated eagerly into a filter over physical rows and
then intersected with `_valid_rows`.

This must remain the fallback path.

### Proposed indexed path

When `CTable.where(...)` receives a predicate that can be normalized into a
supported table query descriptor:

1. extract/normalize the predicate expression against the same base table
2. identify indexed targets for subexpressions or ordering requirements
3. ask the shared planner for candidate physical positions using table-owned
   descriptors
4. intersect candidates with `_valid_rows`
5. evaluate any residual predicate only for the candidate rows
6. build the resulting view as usual

### Scope of v1 planner support

For a first implementation, the planner does **not** need to optimize every
possible `LazyExpr` tree.  It is enough to support the same subset already used
successfully by the `NDArray` index planner, adapted to table dependencies.

Unsupported shapes should simply fall back to scanning.

---

## Ordered-query integration

Expression `FULL` indexes are especially valuable for ordered reuse.

Examples:

```python
t.create_index(
    expression="abs(score - baseline)", kind=blosc2.IndexKind.FULL, name="abs_delta"
)
```

This should eventually allow table ordering/query paths to reuse that sorted
payload when ordering by the same target expression.

### Minimal expectation

For v1, the plan should aim for:

- range filters using expression indexes
- exact target matching for ordering when a `FULL` expression index exists

If ordered integration is too large for the first patch, it can be staged after
basic expression-filter acceleration.  But the descriptor format and storage
layout should be designed with ordered reuse in mind from the start.

---

## Visibility and delete semantics

Direct expression indexes should be built over physical rows and remain valid
across deletes without immediate rebuild.

### Delete behavior

When rows are deleted:

- `_valid_rows` changes
- expression index payloads remain aligned with physical row positions
- queries must intersect any candidate positions with `_valid_rows`

This mirrors the current strategy for stored-column indexes and avoids making
visibility changes force value-index rebuilds.

### Compaction behavior

Compaction rewrites physical row layout, so any index whose positions refer to
physical rows must be rebuilt or compacted accordingly.

The direct expression-index plan should follow the same table-level epoch and
staleness rules as current column indexes.

---

## Mutation and staleness policy

Expression indexes depend on one or more stored columns.

### Any dependency value mutation should stale the index

For a direct expression index depending on `price` and `qty`, any operation that
changes values or physical layout of either dependency should stale or rebuild
that index according to current table index policy.

That includes at least:

- append that extends dependent columns
- overwrite/assignment into dependent columns
- delete/compact when physical layout changes
- any future in-place column-transform operation

### Dependency tracking

The descriptor should store explicit dependency names:

```python
"dependencies": ["price", "qty"]
```

so mutation hooks can efficiently identify affected expression indexes.

### Conservative first version

A conservative v1 policy is acceptable:

- if **any** stored column changes, mark all expression indexes stale

This is simpler but more aggressive than necessary.  A better version narrows
staleness to indexes whose dependency lists intersect the mutated columns.

---

## Interaction with computed columns

Direct expression indexes should be designed as a separate feature from computed
columns.

### What should work independently

A user should be able to do:

```python
t.create_index(expression="price * qty", kind=blosc2.IndexKind.FULL, name="total")
```

without first doing:

```python
t.add_computed_column("total", "price * qty")
```

### What should not happen automatically

Adding a computed column should **not** automatically create a direct expression
index.

### Computed columns as future sugar

A later enhancement could allow:

- `create_index(expression="total")` where `total` is a computed column name,
  lowered to its stored expression
- `create_index(on_computed="total")`

But that should remain out of scope for the first direct-expression-index patch.
For v1, restrict direct expression indexes to stored-column dependencies only.

---

## API details for drop/rebuild/index lookup

Once expression targets are supported, methods such as:

- `drop_index(...)`
- `rebuild_index(...)`
- `compact_index(...)`
- `index(...)`

need a target-resolution story.

### Minimal target-resolution options

A reasonable v1 policy is:

- existing column-name APIs keep working for column indexes
- add `expression=` and optional `name=` parameters for expression indexes

Examples:

```python
t.drop_index(expression="price * qty")
t.rebuild_index(expression="abs(price - baseline)")
t.index(name="abs_delta")
```

### Ambiguity rules

If lookup by `name=` would match multiple indexes, raise `ValueError` and ask
for an explicit `expression=` or column target.

### Why not overload plain positional strings too much

Because a table may have both:

- a real column named `total`
- an expression index with `name="total"`

so plain string lookup should remain conservative and unsurprising.

---

## Suggested internal refactor steps

## Phase 1: shared target normalization helpers

1. extract the `NDArray` expression-target canonicalization logic into shared
   helpers
2. separate AST normalization from container-specific operand resolution
3. add a table-specific normalization path that validates “same table” instead
   of “same array”

Deliverable:

- ability to build normalized table expression target descriptors and tokens,
  even before query integration lands

## Phase 2: table target value provider

1. add helper(s) for obtaining target values from a table target descriptor
2. support:
   - stored-column target
   - direct expression target over stored columns
3. ensure build works in slices/chunks over physical rows

Deliverable:

- ability to build table-owned expression index descriptors and payloads

## Phase 3: catalog and management API extension

1. extend `CTable.create_index(...)` to accept `expression=` / `operands=`
2. extend table catalog logic to store expression-target descriptors
3. extend drop/rebuild/compact/index lookup to resolve expression targets

Deliverable:

- user can create, inspect, drop, and rebuild expression indexes on a table

## Phase 4: query planner integration

1. teach `CTable.where(...)` to normalize supported predicates against the base
   table
2. route supported predicates through the shared planner using the table catalog
3. intersect candidates with `_valid_rows`
4. evaluate residuals only for candidates

Deliverable:

- expression indexes can accelerate `where(...)` and related query paths

## Phase 5: ordered reuse integration

1. connect `FULL` expression indexes to table ordering paths
2. reuse sorted payloads where exact target match exists

Deliverable:

- ordered table queries can reuse direct expression indexes, not just filters

---

## Error handling and validation

Recommended v1 behavior:

- `ValueError` if called on a view
- `ValueError` if `field`/`col_name` and `expression` are both provided
- `ValueError` if `operands` is provided without `expression`
- `KeyError` if a column target does not exist
- `ValueError` if an expression dependency does not resolve to a stored column
  from the same table
- `ValueError` if an expression refers to a computed column in v1
- `TypeError` if the resolved expression dtype is unsupported by the current
  index engine
- `ValueError` if an index already exists for the same normalized expression
  target

The error messages should mirror current `NDArray` phrasing as closely as makes
sense, but mention “same table” rather than “same array” for table expressions.

---

## Testing plan

Add tests in `tests/ctable/` covering at least the following.

### 1. Basic in-memory expression index

- create a table with stored columns `price` and `qty`
- call `create_index(expression="price * qty", kind=...)`
- verify the catalog contains an expression target descriptor
- verify `index(...)` lookup returns the expected handle

### 2. Multiple expression indexes coexist

- create indexes for `price * qty` and `abs(price - baseline)`
- verify both exist simultaneously
- verify labels do not affect identity

### 3. Duplicate target rejection

- create an expression index
- attempt to create the same normalized target again with another `name=`
- verify it raises “already exists”

### 4. Same-table validation

- create two tables
- attempt to create an index whose operands mix columns from both tables
- verify it raises a same-table validation error

### 5. Computed-column dependency rejection (v1)

- add a computed column
- attempt to create an expression index that references it
- verify a clean error

### 6. Persistent round-trip

- create a persistent table
- build a direct expression index
- close/reopen
- verify the descriptor and payloads are restored correctly

### 7. Delete visibility semantics

- build an expression index
- delete some rows
- verify indexed queries still return only live rows after `_valid_rows`
  intersection

### 8. Staleness on dependency mutation

- build an expression index on `price * qty`
- mutate `price`
- verify the index becomes stale or is rebuilt according to policy

### 9. Query acceleration path

- build an expression index
- run a supported `where(...)` predicate over the same expression
- verify the planner reports indexed use, or otherwise assert behavior through
  explain/debug hooks if available

### 10. Ordered reuse path

- build a `FULL` expression index
- run a matching ordered query
- verify sorted reuse if explain/debug hooks expose this

### 11. Lookup/drop/rebuild by expression and by name

- exercise `index(expression=...)`, `drop_index(expression=...)`,
  `rebuild_index(expression=...)`
- test ambiguity behavior for `name=` lookup

---

## Documentation updates

If implemented, update docs to clearly separate:

- stored-column indexes
- computed columns
- direct expression indexes
- materialized computed columns

Suggested touch points:

- `doc/reference/ctable.rst`
- `doc/getting_started/tutorials/15.indexing-ctables.ipynb`
- any indexing overview docs that currently mention only stored columns

### Suggested doc examples

```python
# direct expression index on a table
t.create_index(expression="price * qty", kind=blosc2.IndexKind.FULL, name="total")

# query reuses the expression index when supported
view = t.where((t["price"] * t["qty"]) > 100)
```

and, separately:

```python
# computed column remains a logical schema feature
t.add_computed_column("total", "price * qty")
```

The docs should explain that these are related but distinct capabilities.

---

## Open questions

### 1. API spelling for target selection

Should the table API prefer:

- `create_index("price")`
- `create_index(field="price")`
- `create_index(expression="price * qty")`

Recommendation: support all of the above, mirroring `NDArray`, while keeping the
column target as the simple positional form.

### 2. Should `operands=` be publicly encouraged?

For tables, implicit column-name resolution may be enough for most users.
Explicit `operands=` is powerful but easier to misuse.

Recommendation: support it for parity, but document the simplest form first:

```python
t.create_index(expression="price * qty")
```

### 3. Should computed columns be allowed as expression dependencies?

This is tempting, but it blurs the line between direct expression indexing and
virtual-column indexing.

Recommendation: reject in v1.  Later, if needed, lower computed-column names to
stored-column expressions explicitly.

### 4. How much planner matching is required in v1?

Do we require exact normalized-expression matches only, or some algebraic
rewriting too?

Recommendation: exact normalized target matching first.  Do not attempt broad
symbolic equivalence in the first patch.

### 5. Name-based lookup ergonomics

Should `drop_index("total")` ever refer to an expression index label?

Recommendation: no.  Keep positional string lookup for real columns only.
Require `name=` or `expression=` for expression-target lookup.

---

## Recommended first-patch scope

Keep the first direct-expression-index patch intentionally focused:

- extend target normalization to support same-table expression targets
- allow `CTable.create_index(expression=...)` for expressions over stored
  columns of the same table
- persist and manage those descriptors in the existing table-owned catalog
- document that computed columns are still a separate feature
- add tests for build, persistence, duplicate rejection, same-table validation,
  and at least one indexed-query path

Do **not** expand the first patch to:

- computed-column dependency support
- arbitrary external operands
- view-local expression indexes
- broad optimizer rewrites
- automatic creation from computed columns
- schema-level expression aliases

---

## Summary

Direct table expression indexing is a natural generalization of the existing
`NDArray` expression-index machinery, but it is a distinct feature from
materialized computed columns.

The central refactor is to generalize target normalization from:

- all operands resolve to the same `NDArray`

into:

- all operands resolve to stored columns from the same `CTable`

Once that exists, `CTable` can support expression indexes as table-owned
optimizer structures with no need for explicit computed columns.

That would let users create multiple named indexes for different expressions,
for example:

```python
t.create_index(expression="price * qty", kind=blosc2.IndexKind.FULL, name="total")
t.create_index(
    expression="abs(price - baseline)", kind=blosc2.IndexKind.BUCKET, name="abs_delta"
)
```

while keeping computed columns and materialized columns as separate, optional
higher-level features.
