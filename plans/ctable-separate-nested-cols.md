# CTable separate nested columns for list-struct data

## Summary

Extend CTable nested storage so Arrow/Parquet datasets that are physically stored
as an unnamed top-level `list<struct<...>>` can be imported as a normal CTable
whose rows are the **elements of that root list** and whose struct leaves are
ordinary nested CTable columns.

This is especially important for Awkward-style Parquet files such as Chicago
taxi, whose top-level schema is effectively:

```text
"": list<struct<
  trip: struct<...>,
  payment: struct<...>,
  company: ...
>>
```

For this case, the outer unnamed list is treated as a physical/chunking artifact
of the Parquet encoding, not as a semantic CTable column.  The imported table
should look and behave like:

```python
ct["trip.begin.lon"]
ct["payment.fare"]
ct.where("payment.fare > 20")
ct.nrows == total_number_of_root_list_elements
```

No user-facing `column_0` and no required `ct.explode()` for this root-list case.

The mental model is:

```text
unnamed list<struct<...>>  ->  root record stream  ->  regular nested CTable rows
```

Named `list<struct<...>>` fields inside an otherwise normal parent table remain
typed `ListArray` columns by default.  Future `explode()` support can expose
those named repeated fields as element-row views when parent/child analytics are
needed.

---

## Relationship to existing nested-field work

`plans/ctable-nested-fields.md` already covers:

- logical dotted paths;
- escaping literal `.`, `/`, and `\\`;
- physical hierarchical `_cols/...` storage paths;
- top-level `struct<...>` flattening into leaf columns;
- nested row reconstruction for scalar struct leaves;
- Arrow/Parquet schema roundtrips for top-level structs.

This plan extends that machinery to the special and common case where the whole
Parquet file is an unnamed top-level `list<struct<...>>` record stream.

---

## Goals

1. Import a single unnamed top-level `list<struct<...>>` as a regular CTable row
   stream, with the list elements becoming CTable rows.
2. Physically store scalar leaves of the element struct as separate CTable
   columns, typically NDArrays or existing typed CTable column kinds.
3. Preserve nested logical field paths, e.g. `trip.begin.lon`, `payment.fare`.
4. Avoid `column_0` in the user-facing API for unnamed root-list datasets.
5. Keep named `list<struct<...>>` fields as typed `ListArray` columns by default.
6. Store enough provenance metadata to explain that an unnamed root list was
   flattened, without requiring exact original Parquet row grouping roundtrip.
7. Make separated nested-column import the default for Parquet inputs that qualify,
   with explicit opt-out for schema-fidelity workflows.

## Non-goals for first implementation

1. Exact reconstruction of the original Parquet row grouping for unnamed root
   lists.
2. In-place migration of existing opaque `ListArray` list-struct columns.
3. Full `explode()` / SQL-style unnesting for named repeated fields.
4. Recursive flattening of nested lists inside element structs.
5. Making Awkward Array a dependency.

---

## Core distinction: root record stream vs named repeated field

### Case 1: single unnamed top-level `list<struct<...>>`

Input schema:

```text
"": list<struct<trip: struct<...>, payment: struct<...>, company: ...>>
```

Interpretation:

- The unnamed top-level list is a physical container/chunking artifact.
- Its elements are the logical records.
- The element struct is the logical root schema.
- The imported CTable row count is the total number of list elements, not the
  number of original Parquet rows.

User-facing result:

```text
trip.sec
trip.begin.lon
trip.begin.lat
trip.begin.time
trip.end.lon
trip.end.lat
trip.end.time
trip.path          # nested list inside element; kept as a ListArray initially
payment.fare
payment.tips
payment.total
payment.type
company
```

Example:

```python
ct = blosc2.CTable.from_parquet("chicago-taxi.parquet", separate_nested_cols=True)
ct["trip.begin.lon"].mean()
ct.where("payment.fare > 20")
```

No `ct.explode()` is needed because `ct` is already in the element row space.

### Case 2: named `list<struct<...>>` inside a parent table

Input schema:

```text
user_id: int64
events: list<struct<time: timestamp, amount: float64>>
```

Interpretation:

- Parent rows are semantically meaningful.
- `user_id` has one value per parent row.
- `events` has one list per parent row.

Default representation:

```text
user_id: NDArray
events: ListArray(list(struct(...)))
```

This requires no separate parent-offset metadata for ordinary CTable use:

```python
ct["user_id"]
ct["events"]
```

Offsets only become important if/when a future `ct.explode("events")` view is
implemented and needs to map event elements back to parent rows.

---

## Proposed metadata

For unnamed-root flattening, store provenance metadata.  This is proposed shape,
not final schema:

```json
{
  "nested": {
    "version": 2,
    "original_root": {
      "kind": "unnamed_list_struct",
      "field_name": "",
      "preserve_grouping": false
    }
  }
}
```

Meaning:

- `kind = "unnamed_list_struct"`: source had an unnamed top-level list of struct.
- `field_name = ""`: canonical Arrow root field name.
- `preserve_grouping = false`: original Parquet row/list grouping is not part of
  the logical CTable model and is not guaranteed to roundtrip exactly.

Future optional metadata if exact grouping is requested:

```json
{
  "original_root": {
    "kind": "unnamed_list_struct",
    "field_name": "",
    "preserve_grouping": true,
    "offsets": "_root._offsets",
    "valid": "_root._valid"
  }
}
```

But first implementation should not store original offsets by default.

---

## Physical storage model for unnamed root list

Given:

```text
"": list<struct<
  trip: struct<
    sec: float,
    begin: struct<lon: double, lat: double, time: timestamp[ms]>,
    path: list<struct<londiff: float, latdiff: float>>
  >,
  payment: struct<fare: float, tips: float, total: float>,
  company: dictionary<string>
>>
```

Store scalar struct leaves as ordinary CTable physical columns:

```text
/_cols/trip/sec
/_cols/trip/begin/lon
/_cols/trip/begin/lat
/_cols/trip/begin/time
/_cols/payment/fare
/_cols/payment/tips
/_cols/payment/total
/_cols/company
```

Nested list fields inside the element struct remain typed list columns in phase
1:

```text
/_cols/trip/path
```

where `trip.path` is a `ListArray` with one cell per logical trip row.

All visible columns in the imported CTable have the same row count:

```text
nrows == total number of elements in the unnamed root list
```

Leaf types may be:

- fixed-width numeric/bool/timestamp NDArrays;
- dictionary columns;
- variable-length scalar columns (`vlstring`, `vlbytes`);
- typed `ListArray` columns for nested list fields;
- `ObjectArray` only as fallback for unsupported/heterogeneous data.

---

## Named list-struct fields: ListArray vs ObjectArray

For named `list<struct<...>>` fields, prefer typed `ListArray` by default:

```text
events: ListArray(spec=list(struct({"time": timestamp(...), "amount": float64()})))
```

Reasons:

- Preserves Arrow logical type better than schema-less objects.
- Keeps field/type metadata available for future `explode()`.
- Roundtrips to Arrow/Parquet more naturally.
- Supports both `serializer="msgpack"` and `serializer="arrow"` tradeoffs.

Use `ObjectArray` only as fallback when:

- the Arrow type is unsupported by typed `ListArray`;
- the list contents are heterogeneous;
- item schema cannot be represented by `ListSpec`;
- the user explicitly requests object fallback.

---

## Import behavior

### Phase A: default import with opt-out

The feature started as opt-in, but is now enabled by default for
`CTable.from_parquet()` and `parquet-to-blosc2` when the Parquet schema qualifies
as a single unnamed root `list<struct<...>>`.  The same `separate_nested_cols`
default also lets ordinary top-level Arrow/Parquet `struct<...>` fields follow
`CTable.from_arrow()` semantics and flatten recursively into dotted leaf columns
without changing row cardinality:

```text
CTable.from_parquet(...)
parquet-to-blosc2 input.parquet output.b2d
```

Opt out when closer fidelity to the original Parquet row/schema shape is desired:

```text
CTable.from_parquet(..., separate_nested_cols=False)
parquet-to-blosc2 ... --no-separate-nested-cols
```

`CTable.from_arrow(..., separate_nested_cols=True)` remains available for direct
Arrow inputs.  Named list fields, including named `list<struct<...>>`, remain
typed `ListArray` columns by default.

### Phase B: eligibility for root flattening

Root flattening applies when:

1. the Arrow schema has exactly one top-level field;
2. the top-level field name is `""` or is otherwise known to be the canonical
   unnamed root;
3. the top-level field type is `list<struct<...>>` or `large_list<struct<...>>`.

When all conditions hold, flatten `array.values` (the struct element array) into
CTable columns and use `len(array.values)` as the CTable row count.

### Phase C: import algorithm for unnamed root

1. Read Arrow list array/chunked array.
2. For each batch/chunk, access the flattened element struct array via
   `list_array.values`.
3. Recursively flatten struct fields into leaf arrays.
4. Create/append CTable columns for each leaf.
5. For nested list fields inside the element struct, create/append typed
   `ListArray` columns with one list cell per element row.
6. Avoid `to_pylist()` for scalar leaves whenever possible.
7. Store `original_root` provenance metadata.

The original top-level list offsets do not need to be stored by default.

---

## Row access and logical API

For unnamed-root flattening, `ct[i]` returns a row representing one element of
the original root list:

```python
row = ct[i]
row.trip["begin"]["lon"]
row.payment["fare"]
```

Column access is ordinary nested CTable access:

```python
ct["trip.begin.lon"]
ct.trip.begin.lon
ct["payment.fare"]
```

Filtering and analytics operate directly:

```python
ct.where("payment.fare > 20")
ct["trip.begin.lon"].mean()
ct.select(["trip.begin", "payment.fare"])
```

No `column_0` and no required `explode()` for this case.

---

## Arrow/Parquet export behavior

Exact reproduction of the original unnamed `list<struct<...>>` Parquet row
layout is not a goal by default. Blosc2 and Parquet have different storage
models; import/export should preserve the logical data decently rather than
promise byte- or schema-shape-exact Parquet roundtrips.

Default export may write the clean logical table:

```text
trip: struct<...>
payment: struct<...>
company: ...
```

rather than wrapping rows back into an unnamed top-level `list<struct<...>>`.

A future compatibility option could preserve and re-emit the original root-list
row grouping, but only if a concrete user need appears. If added, original
offsets/validity would need to be stored at import time.

---

## Future `explode()` semantics for named repeated fields

`explode()` remains useful for named list fields inside parent tables, but is not
required for unnamed-root record streams.

Example future API:

```python
events = ct.explode("events")
events["time"]
events["amount"]
events["_parent"]  # optional parent row index
events["_ordinal"]  # optional position inside parent list
```

This is a logical view over a repeated field and changes row granularity from
parent rows to element rows.  It may require offsets or a generated parent-index
array.  This is deferred until after root record stream flattening is working.

---

## Storage and CTable integration

### TreeStore / nested CTable compatibility

A CTable with separated nested columns must remain self-contained when stored as
an object/subtree inside a `TreeStore`, including compact `.b2z` stores. All
physical leaves, indexes, and metadata must live under the CTable root and be
addressed relative to that root:

```text
/some_table/_meta
/some_table/_valid_rows
/some_table/_cols/trip/sec
/some_table/_cols/trip/begin/lon
/some_table/_cols/payment/fare
```

Opening `/some_table` as a regular CTable should reconstruct the same logical
schema and expose the same APIs (`ct[i]`, `ct.where(...)`, `to_arrow()`) without
requiring state outside the CTable subtree. Reopen logic should continue to rely
on the CTable schema/manifest rather than scanning arbitrary outer TreeStore
children.

For `.b2z`, direct-offset/open behavior must work for all separated nested
leaves, just like current hierarchical `_cols/...` CTable leaves.

### Schema representation

Recommended for unnamed-root flattening:

- `CompiledSchema.columns` contains the physical, user-visible element-row leaf
  columns.
- `CTable.col_names` contains logical nested paths such as `trip.begin.lon` and
  `payment.fare`.
- `metadata["nested"]["original_root"]` records that these columns came from an
  unnamed top-level list of struct.
- There are no user-visible `_offsets` / `_valid` columns by default.

---

## Indexing

For unnamed-root flattened tables, indexes work like normal CTable indexes:

```python
ct.create_index("payment.fare")
ct.where("payment.fare > 20")
ct.create_index("trip.begin.time")
```

For named repeated fields, element-level indexes should be deferred until
`explode()` semantics are implemented.

---

## Implementation phases

### Phase 0 — design scaffolding

- [x] Define `original_root` provenance metadata.
- [x] Add helpers to detect a single unnamed top-level `list<struct<...>>` schema.
- [x] Add helpers to flatten Arrow `ListArray.values` struct arrays into leaf arrays.

### Phase 1 — unnamed-root record stream import

- [x] Implement `separate_nested_cols=True` support for single unnamed top-level
  `list<struct<...>>`; make it the default for `CTable.from_parquet()` and the CLI.
- [x] Import element struct leaves as normal nested CTable columns.
- [x] Keep nested list fields inside the element struct as typed `ListArray` columns.
- [x] Avoid `to_pylist()` for scalar leaves; fixed-width leaves use the Arrow → NumPy path.
- [x] Set `ct.nrows` to the total element count.
- [x] Store `original_root` provenance metadata.
- [x] Add `CTable.from_parquet(max_rows=...)`; for unnamed-root imports the limit
  applies to flattened element rows.

Acceptance tests:

- [x] Simple unnamed `list<struct<scalar leaves>>` imports to dotted CTable columns.
- [x] Chicago taxi-style sample imports without `column_0` via `CTable.from_parquet()`
  and `parquet-to-blosc2`.
- [x] `CTable.from_parquet(..., max_rows=N)` limits ordinary rows and flattened
  unnamed-root element rows.
- [x] `ct.where("payment.fare > 20")` works directly.
- [x] `ct["trip.begin.lon"].mean()` works directly.
- [x] Reopen persistent `.b2d` / `.b2z`.
- [x] `to_arrow()` emits a clean logical nested table.
- [x] CLI `--no-separate-nested-cols` preserves ordinary top-level structs as
  singleton-list columns for closer schema fidelity.
- [x] CLI default `--separate-nested-cols` flattens ordinary top-level structs into
  dotted columns consistently with `CTable.from_arrow()`.

### Phase 2 — nested list children inside root elements

- [x] Ensure fields like `trip.path: list<struct<...>>` become typed `ListArray`
  columns with one cell per element row.
- [x] Support `serializer="msgpack"` and `serializer="arrow"` for these list
  columns.
- [x] Add fast Arrow import path for Arrow-serialized list columns via
  `ListArray.extend_arrow()`, avoiding Python object materialization.
- [x] Make Arrow the default list serializer for Parquet imports in both
  `CTable.from_parquet()` and `parquet-to-blosc2`; msgpack remains available for
  read-time PyArrow independence.
- [x] Add serializer-aware batching defaults for the CLI: Arrow uses the sampled
  flattened Parquet-batch scale, while msgpack uses
  `compute_chunks_blocks(estimated_nrows).blocks[0]` to avoid giant Python object
  payloads.
- [x] Expose `items_per_block` in `BatchArray.info` and `ListArray.info` so the
  internal block-size heuristic is visible when tuning compression/random access.
- [x] Retune `BatchArray._guess_blocksize()` cache-budget tiers so default
  `clevel=5` uses `L2 / 2` instead of L1-sized blocks, improving compression for
  Arrow IPC payloads while keeping blocks smaller than full-batch `clevel=6+`
  behavior.
- [ ] Add regression tests for `items_per_block` appearing in `.info` output.
- [ ] Add compression/lookup microbenchmarks for Arrow `ListArray` block-size
  tuning on Chicago taxi-style list-struct payloads.

### Phase 3 — named repeated field explode (future)

- [ ] Add `ct.explode("events")` for named list fields if needed.
- [ ] Expose element leaf columns and optional `_parent`, `_ordinal`.
- [ ] Support `where`, aggregates, and sorting on exploded scalar leaves.

### Phase 4 — parent predicates (future)

- [ ] Add `where_any()` and `where_all()` for named repeated fields if there is user
  demand.
- [ ] Map element masks back to parent masks using offsets/parent-index arrays.

### Phase 5 — recursive repeated groups (future)

- [ ] Consider recursively flattening nested repeated fields inside element structs.
- [ ] Example: `trip.path.londiff` in Chicago taxi.
- [ ] This requires nested row-space semantics and should be designed separately.

---

## Profiling and tuning notes

Recent profiling on:

```bash
parquet-to-blosc2 chicago-taxi.parquet chicago-taxi.b2d \
  --overwrite --separate-nested-cols --max-rows 200_000
```

showed that the old msgpack list serializer spends most of its time in the
list-column conversion path:

- `CTable._write_arrow_batch()` dominated the import path.
- Inside that function, `arrow_col.to_pylist()` for the nested list column took
  about 88% of the function time for the profiled Chicago taxi import.
- Fixed-width scalar leaves were already using the Arrow → NumPy path via
  `_arrow_column_to_numpy()`, so the main Python-object materialization issue was
  the nested `ListArray` column, not all columns.

Using Arrow serialization for nested list columns avoids this conversion.  This
is now the default for Parquet imports; pass `--list-serializer msgpack` only when
read-time PyArrow independence is more important than import speed:

```bash
parquet-to-blosc2 chicago-taxi.parquet chicago-taxi.b2d \
  --overwrite --separate-nested-cols --max-rows 200_000
```

Observed result on the 200k-row sample:

- msgpack list serializer: about 6.1 s import time, 12.5 MB output.
- arrow list serializer: about 0.6 s import time, 14.7 MB output.

Arrow-serialized `ListArray`/`BatchArray` payloads are still compressed by Blosc2
as serialized byte payloads, so `BatchArray` keeps `typesize=1` by default.
Experiments with this Chicago taxi `trip.path` payload showed `typesize=1` was
also the best choice empirically.

The more important tuning parameter was internal `items_per_block`.  The old
`clevel=5` heuristic used an L1-sized budget and produced small blocks (for this
case, around 804 items/block), which compressed poorly.  Retuning the heuristic
to use `L2 / 2` for `clevel` 4–6 produced much larger but still sub-batch blocks
(for this case, around 51k items/block), improving the `trip.path` cratio from
about 4.95 to about 12.0 with only a small copy-time increase.

Current `BatchArray._guess_blocksize()` policy:

- `clevel` 1–3: L1 data-cache budget.
- `clevel` 4–6: half the L2 cache budget.
- `clevel` 7–8: full L2 cache budget.
- `clevel` 9: full batch.

Open follow-ups:

- Add tests around the new `.info` fields and block-size heuristic.
- Benchmark random lookup latency versus compression ratio for different
  `items_per_block` values on Arrow list-struct payloads.
- Keep the read-time PyArrow requirement for Arrow-serialized list columns documented
  in the `CTable.from_parquet()` docstring and CLI `--list-serializer` help.

---

## Resolved design decisions

1. Use the name `separate_nested_cols` for this behavior/API surface. It better
   describes the general physical goal: nested fields become separate physical
   CTable columns where possible.
2. For qualifying schemas, unnamed-root list flattening is automatic by default:
   - exactly one top-level field;
   - field name is the canonical unnamed root `""`;
   - field type is `list<struct<...>>` or `large_list<struct<...>>`.

   Rationale: for these files, the outer list is a physical Parquet encoding
   artifact rather than a meaningful user column. Separating the element struct
   leaves produces a more natural CTable, improves analytics, and should usually
   improve compression for scalar leaves because each leaf is compressed with its
   own dtype/codec path. Users can opt out with `separate_nested_cols=False` or
   `--no-separate-nested-cols` when closer fidelity to the original Parquet schema
   is desired.
3. Store provenance metadata by default, but do not store original root offsets
   by default. Exact original Parquet root grouping is considered a low-priority
   compatibility feature, not part of the normal CTable/Parquet interchange contract.
4. `to_parquet()` should emit a clean logical nested table by default, e.g.
   `trip: struct<...>`, `payment: struct<...>`, `company: ...`, not a re-wrapped
   unnamed `list<struct>` with arbitrary grouping.
5. Do not silently fall back to `ObjectArray` for unsupported nested children.
   Raise by default; use `object_fallback=True` for explicit ObjectArray fallback.

---

## Current status and remaining work

The first milestone is implemented: unnamed-root record stream flattening for one
top-level `list<struct<...>>` column supports:

```python
ct = blosc2.CTable.from_parquet(
    "chicago-taxi.parquet",
    separate_nested_cols=True,
)

ct["payment.fare"].mean()
ct.where("payment.fare > 20")
ct["trip.begin.lon"].mean()
```

This is now the default for `CTable.from_parquet()` and `parquet-to-blosc2` for
qualifying unnamed-root `list<struct<...>>` Parquet files. Pass
`separate_nested_cols=False` in the library API, or `--no-separate-nested-cols`
in the CLI, when preserving the original Parquet row/schema shape is more
important than the separated column layout.

Implemented beyond the original first milestone:

- ordinary top-level structs flatten into dotted columns by default in the CLI;
- `parquet-to-blosc2 --progress` is opt-in and reports ETA for unnamed-root
  imports;
- unnamed-root CLI imports write one flattened Parquet batch at a time, capped by
  `MAX_ELEMENT_WRITE_BATCH`;
- CLI summary output distinguishes unnamed-root row flattening from general
  nested-column separation and reports serializer-aware batching choices;
- Arrow is the default list serializer for Parquet imports, with msgpack still
  available explicitly;
- Arrow/msgpack use different default BatchArray sizes to match their memory
  behavior.

Remaining work:

- `ct.explode()` and parent/element mapping for named repeated fields;
- recursive flattening of nested repeated fields such as `trip.path.londiff`;
- tests and benchmarks for `.info` block-size fields, `items_per_block` tuning,
  compression ratio, and random lookup latency.
