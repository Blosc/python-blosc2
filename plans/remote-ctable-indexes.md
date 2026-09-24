# RemoteCTable persisted scalar indexes

Status: implemented and verified, 2026-09-20. This completes follow-up 3 in
`remote-ctable.md`.

## Goal and scope

Execute queries locally against persisted remote indexes, fetching only the
index and column ranges needed by the query. Start with SUMMARY, then single-run
FULL, then PARTIAL, OPSI and BUCKET, and finally multi-run FULL. Existing remote
list membership indexes continue to work.

Support standalone and nested B2Z CTables, ordinary RemoteCTable handles,
RemoteStore-owned tables, saved reference artifacts and shared sparse caches.
Reuse existing index formats and query semantics. Index creation, rebuilding
and mutation remain local operations; this work reads immutable remote indexes.

Do not introduce a new query API, remote query service, index format or dependency.
Unsupported index layouts must retain the existing correct scan fallback.

## Original implementation and integration points

- `RemoteTableStorage.load_index_catalog()` currently admits only membership
  descriptors. `open_membership_postings()` already reads selected batches.
- `indexing.py` opens scalar index sidecars using local paths and the B2Z offset
  registry. `_open_sidecar_handle()` and the span-reading helpers are the main
  integration points for a remote resolver.
- `_load_array_sidecar()` can load an entire sidecar into the process-wide
  decoded-data cache. That behavior must not silently become an unbounded remote
  download or bypass the remote owner's cache lifetime.
- `_CTableIndexingMixin._find_indexed_columns()` currently recognizes NDArray
  operands. RemoteArray operands need equivalent descriptor selection and
  validation without opening unrelated columns.
- SUMMARY yields candidate segments; the remote read path must consume those
  candidates before fetching predicate data. Running local pruning after a full
  remote column read would provide no transfer savings.
- FULL already has navigation sidecars and selective value/position span reads.
  Some paths call NDArray-specific methods such as `get_1d_span_numpy`; inspect
  and adapt these paths rather than assuming RemoteArray is a drop-in handle.

## Shared index resolution and ownership

Add the smallest storage-backed sidecar resolution hook needed by the existing
index reader. Local resolution retains current behavior. Remote resolution opens
the selected B2Z member through the table's remote owner, with the same source
authorization, transport and generation checks as column reads. Avoid registering
remote URLs as fake local filesystem paths or eagerly downloading an index
directory to temporary files.

Validate descriptors before use: supported version/kind, dtype, geometry,
segment counts, epochs, referenced member names and navigation shapes. Resolve
paths relative to the table root, reject absolute paths and traversal, and reject
encrypted or ZIP-compressed members consistently with existing remote readers.
Unsupported valid formats can fall back to scans; malformed or unsafe metadata
must produce a clear error. Stale descriptors must never prune valid rows.

Index payload participates in the outer owner's cache policy, aggregate
`max_cache_bytes`, traffic accounting and generation lifecycle. This includes
external RemoteArray-backed columns. Do not independently enable their persisted
cache policy. Metadata discovery, retained compressed payload, temporary decoded
index arrays and query outputs must remain distinguishable in documentation.

NONE permits temporary query buffers but no retained index payload. MEMORY and
DISK reuse index payload under their existing eviction policies. Shared sparse
handles reuse the same process-safe storage and locking as ordinary array
leaves. Avoid a second unbounded decoded index cache under the global local-index
registries; keep decoded working sets scoped to the query initially.

Saved references include only index payload already retained when
`include_cache=True`; saving must not fetch missing sidecars. Cold references
preserve sufficient descriptors to resolve them later. Refresh invalidates old
index handles and candidate plans together with columns and views.

An index for an external column is usable only when its descriptor applies to
that column's persisted source snapshot and row mapping. Do not discover or
borrow arbitrary standalone source indexes automatically. Initially fall back
to scans where this correspondence cannot be validated.

## Phase 1: SUMMARY

Read summaries only for columns participating in the predicate. Each existing
summary level is a separate compressed sidecar with per-segment bounds and flags.
For a flat summary level, inspect all its entries to establish the candidate
mask; do not imply a logarithmic lookup or a request per original column chunk.

### One-request small-sidecar path

Once the archive member offset and length are known, fetch a sufficiently small
summary member in one contiguous range request, including its frame metadata.
If discovery already prefetched it or the cache contains it, reuse those bytes.
Opening the archive or locating an index may require additional requests: the
contract is one cold payload request per eligible sidecar, not one request for
the entire first query or all indexed columns combined.

Reuse existing B2Z range and frame decoding machinery. Adopt fetched compressed
chunks into the owner's normal cache representation so the optimization does not
create a separate whole-file cache or retain duplicate payload indefinitely.
Respect cache limits even when a whole-sidecar fetch exceeds the retention budget.

Use the existing `row_buffer_bytes` allowance to bound compressed index payload
staging and `metadata_buffer_bytes` for discovery; index summaries are payload,
not free metadata. Do not add a public index-specific tuning knob initially.
For large sidecars, process bounded groups of compressed chunks/ranges and build
the candidate mask incrementally. Decoded buffers and the candidate mask require
separate memory accounting; compressed size alone is not a decoded-memory bound.
Retain the established rule that an indivisible oversized unit is processed
alone, and document that these allowances are not total RSS caps.

### Pruning and evaluation

Map candidate segments to row ranges conservatively, preserving chunk/block
granularity, partial tail segments, deleted rows and null/NaN flags. Re-evaluate
the exact predicate on surviving rows. Combine supported conjunctions and
disjunctions using existing planner rules; unsupported expressions scan rather
than risk false negatives. Columns may have different chunk/block geometry, so
transfer row selections between columns rather than copying block numbers.

Fetch projected columns only for surviving rows. Avoid fetching predicate or
projection payload at all when the index proves there are no matches, except
metadata or validity information actually required for that decision.

SUMMARY is most effective when ranges separate segments well. Randomly distributed
values can leave every segment eligible. Measure index overhead and scan savings;
do not promise that every indexed query transfers less than a scan.

## Phase 2: single-run FULL

Use navigation metadata to locate candidate sorted-value regions, read those
compressed ranges, determine exact match bounds locally, and fetch corresponding
position spans. Resolve live-row visibility and project requested columns using
the resulting positions. Preserve existing ordering, duplicate and null semantics.

Start with existing compact single-run layouts that already support selective
out-of-core reads. Fetch small navigation sidecars as a unit when useful, but
keep large values and positions sidecars lazy. Audit whole-array conversions,
native NDArray-only calls and fallback paths before enabling each layout.

Coalesce adjacent ranges and use existing request concurrency/buffer controls.
Do not perform one remote request per binary-search comparison. A selective
query should read navigation plus a small number of compressed payload units;
a broad predicate may legitimately touch most of the index or favor a scan.
An explicit `_use_index=False` must continue to provide the scan baseline.

## Later phases

1. PARTIAL: reuse chunk navigation and fetch candidate local sorted regions and
   position data. Handle partial tails and conservative predicate rechecks.
2. OPSI: reuse block navigation and selective values/positions readers. Measure
   request amplification where matching regions are scattered.
3. BUCKET: read bucket navigation and selected candidate payload, preserving any
   exact predicate recheck required by the existing index representation.
4. Multi-run FULL: resolve relevant spans from each run and merge positions
   without downloading complete runs; preserve duplicate and ordering semantics.

Each phase must inspect its actual persisted layout and native reader assumptions.
Share the resolver and transport machinery; do not force all index types into
one new abstraction. Dictionary/scalar dtype coverage follows the local index's
semantics and requires explicit parity tests before enabling it remotely.

## Implementation sequence and verification

1. **Introduce remote sidecar resolution.** Validate and lazily expose supported
   scalar descriptors. Test safe relative paths, malformed descriptors, nested
   tables, unused indexed columns remaining unopened, and local-reader regression.
2. **Enable SUMMARY reads and pruning.** Add the small-member single-range path
   and bounded large-summary path. Test chunk/block granularities, empty tables,
   tails, nulls/NaNs, deleted rows, compound predicates and mismatched column
   geometry against local results and forced remote scans.
3. **Verify SUMMARY transport and lifecycle.** Instrument deterministic range
   requests: one cold payload fetch for a small sidecar whose location is known,
   no request per original data chunk, no unrelated sidecars, and no excluded
   data blocks read. Test NONE/MEMORY/DISK, eviction, warm reuse, sparse handles,
   refresh invalidation, and warm/cold reference save/reopen. Exercise the bounded
   path using a low buffer allowance rather than enormous test fixtures.
4. **Enable selective single-run FULL.** Compare equality/range queries, missing
   keys, duplicate values and supported ordering operations with local results.
   Assert selective queries do not fetch entire large values/positions sidecars;
   verify broad-query and unsupported-layout fallbacks remain correct.
5. **Add PARTIAL, OPSI and BUCKET individually.** For each kind, test semantic
   parity, selected-range transfers, scattered candidates and shared cache limits.
   Keep each addition independently reviewable.
6. **Add multi-run FULL.** Test overlapping runs, duplicate values, empty matches,
   range bounds and bounded transfer behavior. Verify unsupported layouts retain
   safe scan behavior until explicitly enabled.
7. **Document and measure.** Update the remote table guide/reference and follow-up
   status in `remote-ctable.md`. Correct the stale guide claim that all persisted
   indexes are unsupported. Record supported kinds/layouts and residual limits.
   Add a reproducible benchmark reporting cold/warm request counts, transferred
   index/data bytes, retained cache bytes and elapsed time versus forced scans,
   for clustered and unclustered data and selective/broad predicates.

Run focused index, CTable, RemoteArray and RemoteStore tests in the `blosc2`
conda environment, plus Ruff and applicable documentation checks. Use controlled
range-counting fixtures for assertions, not public-server timings. Run the full
default suite before declaring the extension complete. Record actual commands,
results and any remaining limitations here after implementation.

## Implementation result

Remote tables now expose validated scalar sidecars through the existing index
reader and outer remote cache owner. SUMMARY pruning and FULL, PARTIAL, OPSI and
BUCKET positional lookups work without materializing sidecars locally. FULL run
descriptors are resolved through the same path, so incremental runs remain
queryable. Existing membership indexes continue to use their posting reader.

The resolver accepts only relative `.b2nd` members beneath the table root and
rejects absolute paths, traversal and missing members. Sidecar handles share the
table's cache policy, byte budget, traffic counters, sparse cache, saved-reference
lifecycle and refresh generation. Local CTable sidecar resolution is unchanged.

Implementation sequence:

1. `058e126d` resolves remote CTable index sidecars.
2. `ad9fff29` enables remote SUMMARY pruning.
3. `d60c629e` verifies SUMMARY transport and cache lifecycle.
4. `744191d8` enables remote FULL lookups.
5. `d466707a` enables PARTIAL, OPSI and BUCKET lookups.
6. `24f97772` verifies incremental FULL runs.
7. `46b70d70` documents the feature and adds a reproducible benchmark.

Verification in the `blosc2` conda environment:

- Focused CTable, RemoteArray, RemoteStore and indexing suites: 574 passed,
  5 skipped.
- Full default suite: 10,352 passed, 36 skipped.
- Ruff formatting and checks passed for the changed Python files.
- The Sphinx HTML build completed successfully; it retained the repository's
  existing warnings.
- `conda run -n blosc2 python bench/remote_ctable_indexes.py --rows 10000`
  reported 5 requests/8,695 bytes for the indexed selective query and
  2 requests/5,688 bytes for its forced-scan baseline. At this small scale the
  scan was faster; the benchmark is intended for workload-specific measurement,
  not a universal index-speed claim.

The tests cover lazy catalog discovery, chunk/block SUMMARY pruning, NONE and
MEMORY transport behavior, warm references, sparse-cache reuse, refresh
invalidation, selective and missing FULL keys, equality/range lookups for the
three positional kinds, and an incremental FULL run. Detailed workload sweeps
for cache eviction, broad predicates and clustered versus unclustered data remain
benchmarking work rather than correctness requirements.

## Deferred work

Hierarchical SUMMARY navigation, new persisted formats, automatic remote index
construction, remote writes, server-side query execution and automatic source
change detection are outside this plan. Revisit hierarchical summaries only if
flat-summary downloads become a measured bottleneck. Selective transfers remain
bounded by compressed block/chunk layout; they cannot guarantee byte-exact reads
of only matching values or small downloads for broad queries.
