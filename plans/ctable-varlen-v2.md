# CTable variable-length columns: ListArray V2

Status: implemented and verified on 2026-09-20.

## Objective and scope

Extend ListArray and list-valued CTable columns with four features, available
locally and through read-only RemoteCTable:

1. Nullable elements inside a list, independently of whole-cell nullability.
2. Nested lists with recursive schema validation and serialization.
3. List-aware `contains` and `overlaps` predicates.
4. Optional membership indexes that avoid scanning list payloads for selective
   queries, including remote queries.

Build on [remote-ctable-batches.md](remote-ctable-batches.md). Reuse ListArray,
BatchArray, the existing CTable query/index lifecycle and remote transport/cache
machinery. No new public remote array class or required dependency is intended.
V2 here names this feature proposal; persist a new format version only where
needed to prevent older readers from misinterpreting new semantics.

Remote coverage in this proposal is `storage="batch"`, with MessagePack and
optional Arrow serializers. Local `storage="vl"` must support the new logical
values and scan predicates too. Remote ObjectArray transport is a separate
extension: it is not required to deliver these four features remotely.

Sorting/grouping of list values, arbitrary list expressions, automatic byte-based
batch tuning, and a general Arrow-native execution engine remain follow-ups.

## Current implementation and boundaries

- `schema.py`: ListSpec rejects nested ListSpec items and carries storage,
  serializer and batch settings. `schema_compiler.py` validates annotations;
  recursive list annotations need corresponding handling.
- `list_array.py`: scalar coercion rejects `None` items; schema validation rejects
  nesting. Arrow type conversion is not recursive for lists. Reads, copy and
  mutation must preserve the new values through both existing backends.
- `batch_array.py` and `msgpack_utils.py`: reuse serializers and safe remote
  decoding. MessagePack already represents nested lists and null values; schema
  support and validation are still required.
- `ctable.py`: Column/query integration, physical-row selection, Arrow/Parquet
  conversion, copies, deletes and compaction are shared local/remote boundaries.
- `ctable_indexing.py`: index descriptors currently accept summary, bucket,
  partial, full and opsi kinds. Membership needs its own descriptor and lookup
  semantics, while reusing lifecycle and storage facilities.
- `ctable_storage.py`: RemoteTableStorage currently returns an empty index
  catalog. Selectively expose supported membership descriptors and resolve their
  companions inside the archive; do not enable existing local-path index openers.
- `remote_batch.py`, `ctable_remote_read.py`, `remote_array.py`, `b2z_source.py`
  and the owner cache coordinator provide range reads, scheduling and caching.

Trace shared callers before editing. Avoid a general storage or query framework
refactor. The index transport work is the largest part of the remote extension;
nullable and nested values mainly extend schema and decoding behavior.

## Schema and compatibility

Use the child spec's `nullable` flag for element nullability and ListSpec's own
`nullable` flag for the list value at that level. Proposed examples:

```python
# None, [], [1, None, 3]
blosc2.list(blosc2.int32(nullable=True), nullable=True, batch_rows=2048)

# None, [], [None, [], [1, None, 3]]
blosc2.list(
    blosc2.list(blosc2.int32(nullable=True), nullable=True),
    nullable=True,
    batch_rows=2048,
)
```

Inner ListSpec nodes describe values, not independently stored containers. Only
the outer column's storage, serializer, batch_rows and items_per_block control
physical storage. Reject nondefault inner storage options with an actionable
error rather than silently ignoring an attempted layout configuration.

Validate recursively, including nullable struct fields inside list items, with
errors identifying the column and nested item position. Keep `None`, `[]`,
`[None]`, `[[]]` and `[None, []]` distinct. Update annotation checking, schema
metadata round trips, display labels and validation bypass paths used by trusted
imports. Do not let table-wide sentinel null policies replace null list items.

Read existing V1 data unchanged. Audit existing child-nullable metadata before
deciding how to normalize it: V1's validator rejected null items even where a
child spec carried a nullable flag. V2 writers must mark schemas requiring new
semantics explicitly; unknown versions/features must fail clearly. Verify actual
V1 reader rejection of the chosen marker, rather than assuming a version bump
alone enforces it. Never silently rewrite an existing archive on read.

## Serialization and batch geometry

MessagePack retains recursive values as passive data. Extend Arrow schema
conversion recursively and preserve list value-field nullability, nested list
nullability and struct field nullability during import/export. Handle regular and
large list inputs explicitly; reject unsupported Arrow layouts with a useful
error. Arrow remains optional, and MessagePack must not import it.

Keep one persisted batch per compressed BatchArray chunk. Nested child lists
remain inside that chunk. Remote scalar reads therefore still fetch the entire
containing batch; reading one member inside a list does not imply partial-list
transport. `storage="batch"` does not imply Arrow: MessagePack is the current
default serializer.

Change the default to `batch_rows=2048` for newly constructed ListSpec,
`blosc2.list()` and ListArray values, including `ListArray.from_arrow` and CTable
construction paths. Keep Arrow/Parquet import defaults consistent, documenting
any intentional import-specific override. Inner ListSpec defaults do not create
separate batches; only the outer storage configuration applies. This option has
no batching effect on local `storage="vl"` columns.

Retain explicit `batch_rows=None` as an opt-in to caller-managed boundaries:
ordinary append/extend buffers rows until a flush, and the flush persists all
pending rows as one batch. Explain the resulting memory and remote overfetch
costs. With the finite default, full batches flush automatically, but remaining
rows still need `flush()`, `close()` or context-manager exit before reopening a
standalone ListArray. Do not imply that automatic batching bounds the temporary
input allocation of a large `extend()` call.

Keep construction defaults separate from metadata decoding: reopening an old
schema with no batch_rows field must preserve its previous None semantics and
existing batch boundaries. Preserve explicit None through metadata round trips
(encode it explicitly or use an equally unambiguous version-aware rule). Do not
silently apply 2048 when decoding legacy metadata or copying an existing spec.

Audit `extend_arrow`, import and rewrite paths so a configured row target is
honored consistently, splitting incoming batches where necessary. Flush may
produce a smaller batch; persist actual lengths rather than deriving row lookup
from the target. A single huge list can still produce a huge chunk. No byte or
decoded-memory bound follows from a row-count target.

## Required documentation updates

Update existing documentation as part of implementation, not just this plan:

- `doc/reference/list_array.rst` and public docstrings in `list_array.py` and
  `schema.py`: document the 2048-row default, positive overrides, explicit None,
  automatic full-batch flushing and persistence of the partial final batch.
  Repair the reference example to flush/close or exit a context manager before
  reopening the file. Explain that the unit is list cells, not child elements
  or bytes, and that batch_rows does not control VL storage.
- `doc/tutorials/11.containers.ipynb`: explain the default alongside the existing
  small explicit batch example. Demonstrate safe reopening with a partial final
  batch and explain when caller-managed None is useful. Execute the changed
  example and refresh its outputs.
- `doc/reference/ctable.rst` and relevant existing CTable/import tutorials:
  document list-column defaults, overrides and table-managed persistence;
  distinguish input/import batch size from persisted list batch size. Audit
  existing examples for assumptions about None or reopen without flushing.
- `doc/reference/remotectable.rst`: explain whole-batch transfer, the finite
  default, the smaller-batch versus throughput tradeoff, and why one very large
  list can still require a large download. Link to the ListArray batching guide.
- Add a release/migration note for the changed construction default and preserved
  legacy metadata behavior. Keep signatures, generated API docs and examples
  consistent; old archives do not acquire new physical boundaries on reopen.

## Predicate contract

The following is proposed API, not currently implemented syntax:

```python
has_python = table["tags"].contains("python")
has_either = table["tags"].overlaps(["python", "numpy"])
selected = table[has_either]
```

Provide equivalent behavior on ListArray and CTable Column. Column predicates
must enter the existing table expression/selection machinery lazily and compose
with `&`, `|` and `~`, including predicates on scalar columns. Reuse existing
expression classes where possible; string-expression syntax is not required.

Define membership over immediate children, with no implicit recursive flattening.
For `[[1, 2], [3]]`, `contains([1, 2])` is true and `contains([1])` is false;
`contains(1)` is a type error. `overlaps([[3], [4]])` is true. Structural equality
is recursive, ordered and exact for list children, including null positions.
Scalar literals must be validated against the child type without lossy coercions.
Define float NaN as unequal, consistent with ordinary numeric equality, and
normalize signed zero for indexing. Validate literals even on empty inputs.

Use explicit two-valued membership results: a null outer list or an empty list
matches neither operation; null child values match an explicit `None` literal
when the child schema permits it. `overlaps([])` is false for every row.
Repeated child values or query literals never duplicate result rows. Negation is
ordinary Boolean negation, so negating a false result includes a null outer row.
Document this behavior separately from scalar-column null comparisons and test
mixed expressions. Existing whole-cell null predicates remain available.

Implement the scan baseline first. Iterate batches once per operation, preserve
physical row positions internally, and apply live-row/view selection correctly.
On remote tables, a scan downloads all relevant uncached list batches; it is
client-side execution, not a server-side filter. Bound transient decoded batches
and avoid materializing the entire list column as Python values.

## Membership indexes

Add an opt-in membership index kind through the existing index API. Proposed use:

```python
table.create_index("tags", kind="membership")
```

Initially index flat lists of supported scalar child types: Boolean, integers,
floats, strings and bytes, including nullable items. Finalize a typed canonical
key encoding shared by scans and indexes; never use Python's randomized hash as
a persisted key. Exclude NaN from postings because it never matches. Keep a
distinct null-item key. Struct and nested-list membership uses the scan path in
V2; index creation for those types reports the limitation explicitly. All four
features are available remotely, but indexing every possible nested value is
not a V2 requirement.

Store each key's sorted, unique physical row IDs as a posting list. Repeated
values within one cell generate one posting. `contains` looks up one posting;
`overlaps` unions postings. Apply the table's live-row mask and the current view
selection. Complements use the live-row universe, never unused capacity.

Use existing compressed storage primitives for index companions. Persist a
versioned descriptor recording the stable typed key encoding, source revision,
row extent and companion identity. The initial layout keeps the sorted typed key
directory in CTable metadata and stores one compressed BatchArray posting batch
per distinct value. Exact key comparison is required; hashes alone never
establish membership. This makes selective remote lookup one metadata read plus
the requested posting batches. Paging the key directory is deferred until real
high-cardinality measurements justify the extra format and lookup machinery.

The initial builder accumulates unique physical row IDs per typed key in memory,
then publishes the complete posting store before activating its descriptor.
Reuse create/drop/rebuild, packaging and stale-index machinery.
Initially mark indexes stale after changes rather than adding incremental
posting updates. Audit append, assignment (including raw column access), schema
changes, deletes, compaction, copy and save. If a mutation path cannot reliably
invalidate a descriptor, fix that before enabling indexed queries. Compaction
changes physical row IDs and must invalidate or rebuild postings.

Missing or stale indexes use a scan. Malformed advertised indexes must raise an
actionable integrity error rather than return incomplete results. Offer the
existing scan-forcing/debug idiom where applicable so parity is testable.

## Efficient RemoteCTable execution

Expose valid supported membership descriptors lazily through RemoteTableStorage.
Resolve companions relative to the table's archive namespace, including tables
nested in RemoteStore. Validate member uniqueness, role, bounds, versions,
schema/revision agreement, key ordering, posting offsets and row-ID ranges.
Do not interpret remote metadata as local paths, URLs or executable objects.
Other persisted index kinds remain disabled unless separately implemented.

The selective query path is:

1. Read the descriptor and bounded directory metadata.
2. Fetch only key pages needed for the literals and chunks covering their postings.
3. Merge/intersect positions with live rows and any other query candidates.
4. Fetch only requested output columns, grouping selected rows by physical batch
   or NDArray block and restoring requested order/duplicates at the output.

An exact index-only membership selection must not fetch list data payloads merely
to recheck membership. If the result projects the list column, fetch its matching
batches. A query projecting only scalar columns need not fetch list batches at
all. Missing terms need no posting or list payload reads. Counts may be derived
from postings after live-row/view filtering, without materializing list cells.

Compose candidates conservatively: intersections can narrow an AND scan, but OR
must include both branches and NOT requires the correct live-row universe.
Provide a simple cost decision using posting lengths and candidate batch coverage
to avoid expensive index traversal for broad predicates. Index availability is
not a promise of lower transfer: scattered matches may touch every output batch.

Reuse the owner's NONE/MEMORY/DISK caches, identity/generation keys, eviction,
traffic accounting and bounded scheduler for key pages and postings. No separate
unbounded index cache or per-column payload budget. Group repeated literals and
chunk reads within one operation even under NONE. Decode and mutate caches on
the owning thread. Large postings must be streamable; scheduling limits do not
bound a caller-requested full result mask or decoded Python result.

Preserve read-only enforcement, leases, close/refresh invalidation and safe
MessagePack decoding at all nesting levels. Cached predicates and index handles
must check lifetime too. No implicit full-archive localization, remote index
creation or remote writes. Local materialization produces ordinary writable
tables with valid rebuilt indexes or no index, never stale source descriptors.

## Implementation sequence

1. **Define recursive schemas and compatibility.** Implement recursive ListSpec
   validation, child nullability, annotation handling and metadata feature/version
   checks. Add V1 fixtures and rejection tests before emitting V2 metadata.

2. **Deliver nullable and nested values locally.** Extend coercion and both
   backends; update recursive Arrow conversion and import/export. Audit batching,
   mutation, copy/save and null handling. Verify distinct null/empty forms and
   configured batch geometry across Python and Arrow writes. Implement the 2048
   construction default and explicit None behavior, preserving legacy metadata.
   Update batching docstrings and the reference/tutorial examples with this step.

3. **Deliver nullable and nested remote reads.** Reuse the remote batch reader
   for both serializers. Verify sparse reads, metadata-only opening, safe recursive
   decoding, wrapper lifetime and local exports against the same local archive.
   This milestone delivers the first two features remotely.

4. **Add shared list predicate scans.** Implement the stated truth table and
   nested structural comparisons, lazy Column integration and Boolean composition.
   Use batch-grouped scans locally and remotely, with live-row/view correctness.
   This supplies the correctness oracle for every indexed path.

5. **Build and persist local membership indexes.** Add the descriptor kind, typed
   keys, paged catalog, chunked postings and bounded builder. Integrate lifecycle,
   invalidation and archive packaging; verify indexed/scan parity, including null
   items, duplicates and updates. Record the exact on-disk layout in developer docs.

6. **Prove selective remote index lookup.** Resolve membership companions through
   the archive and expose only supported descriptors. On a large instrumented
   fixture, look up distant terms and fetch their postings without downloading
   the catalog, list column or archive in full. Integrate aggregate cache budgets,
   scheduling, safe metadata validation and generation checks.

7. **Integrate remote indexed query planning.** Connect positions to projections,
   mixed predicates, views, counts and local materialization. Handle AND/OR/NOT,
   scan fallback and broad-query cost decisions. Demonstrate an indexed selection
   projecting scalar columns with zero list payload reads, then list projection
   fetching only required batches.

8. **Document and validate the complete feature set.** Update ListArray/CTable
   schema, predicate, index and remote support documentation, completing every
   item in Required documentation updates. Add a runnable local
   and remote example, state nesting/index restrictions and batch costs, and record
   the cold/warm measurements below. Update related plans only to reflect features
   actually delivered. Run focused checks and the default suite.

## Verification and acceptance

Use the `blosc2` conda environment for Python, tests and builds. Extend existing
list, schema, Arrow/Parquet, CTable indexing and remote test modules; reuse the
instrumented fsspec memory filesystem and deterministic HTTP range server.

- Round-trip null outer lists, null scalar items, null inner lists, empty lists
  at every level, nullable struct children, multilingual strings and bytes.
  Cover both serializers, local VL, persistence, reopen and V1 compatibility.
- Verify default batch lengths using 4097 rows: two automatic 2048-row batches
  plus a final one-row batch after flushing. Cover explicit positive overrides,
  explicit None, legacy missing-field decoding, new metadata round trips, and
  Python/Arrow/CTable creation paths. Confirm schema copy preserves the setting.
  Execute updated tutorial examples and check that reopened data includes the
  partial final batch; review generated docs for consistent defaults and links.
- Test invalid depths/types, child constraints, unsupported versions, malformed
  metadata/payloads, unsafe MessagePack extensions and missing optional Arrow.
  Bound schema recursion and validate decoded structure before trusting offsets
  or lengths; document the supported depth limit.
- Compare scans and indexes with a simple reference implementation for empty,
  duplicate-heavy and randomized data. Include integer limits, signed zero, NaN,
  null literals, structural comparisons and invalid literals on empty tables.
- Cover deleted rows, capacity, scalar/slice/strided/reverse/fancy selections,
  filtered views, mixed expressions, projections, counts and detached exports.
  Test each mutation's index invalidation and interrupted index publication.
- Check lazy open, NONE/MEMORY/DISK, mixed index/column eviction, disk reopen,
  source identity isolation, stale descriptors, missing companions, malformed
  postings and close/refresh behavior even on warm cached results.
- Use payloads larger than metadata prefetch thresholds. Assert selective index
  lookup reads only needed pages/posting chunks plus bounded metadata; scalar-only
  projection reads no list payload, list projection reads only matching batches,
  and a warm query fitting cache needs no new payload requests. Include absent
  terms, huge postings and matches scattered across most batches.
- Record dataset/archive/index sizes, cardinality, list length distribution,
  null rate, batch geometry, selectivity, cold/warm requests and bytes, decoded
  memory and elapsed time. Compare scan and index locally and remotely, separating
  index traffic from output-column traffic. No universal latency target is needed.
- Run focused list/schema/serialization/index/remote tests, then the default
  suite. Run Ruff for changed Python and build affected documentation. Record
  any pre-existing failures separately.

Completion means the four features meet their documented type coverage locally
and remotely, indexed and scan semantics agree, and selective remote membership
queries demonstrably avoid list scans. Remote VL access, indexes on structural
children, sorting/grouping and finer-than-batch list transport remain explicit
follow-ups.

## Implementation results

The four requested features were delivered as separate commits:

1. `fd4f3bcc` — nullable list elements, plus the 2048-row construction default
   and backward-compatible explicit/legacy None handling.
2. `e49753d5` — recursive nested lists, recursive Arrow types and remote reads for
   MessagePack and Arrow batches.
3. `1ce260b7` — immediate-child `contains` and `overlaps` predicates shared by
   standalone ListArray, CTable and RemoteCTable.
4. `1caa94c7` — stable typed membership keys, compressed posting batches,
   mutation invalidation and selective RemoteCTable posting reads.

The implementation reuses the existing remote BatchArray source and aggregate
NONE/MEMORY/DISK cache for posting payloads. A remote indexed predicate that
projects only scalar columns does not open or transfer the source list column.
Absent terms return an empty posting result without scanning list batches.

The final implementation deliberately keeps the sorted key directory in table
metadata and builds postings in memory. This is the smallest format that provides
selective remote payload reads and exact typed-key semantics. Directory paging
and spill/merge construction remain follow-ups for measured high-cardinality
indexes; they do not change the public predicate or index API.

Focused ListArray, schema, Arrow/Parquet, CTable indexing and RemoteCTable suites
passed with 400 tests after one explicit-None import regression was repaired.
The default suite passed with 10,330 tests and 36 skips. Ruff and pre-commit
passed, the container tutorial executed successfully, and the normal Sphinx HTML
build completed with the repository's existing warnings.
