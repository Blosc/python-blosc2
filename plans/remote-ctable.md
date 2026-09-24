# RemoteCTable implementation plan

Status: initial fixed-width, read-only implementation completed on 2026-09-17;
UTF-8 support was added in the v2 extension (see `remote-ctable-v2.md`);
batch-backed columns were added in the batch extension (see
`remote-ctable-batches.md`). Portable references and RemoteStore artifact
inclusion are implemented (see `remote-ctable-save.md`). Remote list membership
and scalar persisted indexes are supported (see `remote-ctable-indexes.md`).
Status reviewed on 2026-09-20.

The implementation notes and first-release decisions below describe the initial
scope. The follow-up status at the end reflects subsequent extensions.

## Objective and architecture

Add `blosc2.RemoteCTable` for read-only, on-demand access to CTable objects in
immutable remote `.b2z` archives through fsspec. Keep RemoteArray, RemoteStore and
RemoteCTable as distinct public objects: arrays, hierarchies and tables have
different interfaces. Do not introduce a public RemoteSource or automatic URL
factory in this work.

Use a thin CTable subclass and a read-only `RemoteTableStorage` backend. Reuse
`CTable._open_from_storage()`, schema reconstruction, lazy column opening and
existing table query logic. Share the existing remote discovery owner, B2Z range
reader, RemoteArray leaves, cache coordinator and traffic accounting. No new
dependency or on-disk table format is planned.

The ponytail principle for this implementation is to adapt the existing storage
boundary and repair shared read paths, without copying table algorithms into a
second implementation or building a general remote-object framework.

## Implemented in this branch

- Added the public `blosc2.RemoteCTable` class and exported it from the package.
  It accepts `dataset`, `storage_options`, `cache_policy`, `max_cache_bytes` and
  `cache_dir`, and exposes the shared source, traffic and cache metrics.
- Added `RemoteTableStorage`, backed by the existing remote discovery owner and
  `RemoteArray` leaves. It opens table schema, fixed-width columns, row validity,
  nullable-column masks and user `attrs` lazily and rejects mutation.
- Extended B2Z discovery and `RemoteStore` so CTable roots and nested CTable
  nodes are recognized, remain opaque during hierarchy traversal, and open as
  `RemoteCTable` objects.
- Added slice-based shared CTable fallbacks needed by remote operands, preserving
  local optimized paths. Scalar and sliced rows, iteration, filtering,
  reductions, fixed-width strings and mask-backed nulls work remotely.
- Added remote scalar sidecar resolution and query support for SUMMARY, FULL,
  PARTIAL, OPSI and BUCKET indexes. Existing list membership indexes remain
  supported.
- Fixed owned `s3fs` cleanup so its registered finalizer closes the aiobotocore
  session exactly once, while HTTP fsspec sessions retain deterministic closing.
- Added `examples/ctable/remote_handling.py`. `--write FILE.b2z` generates a
  one-million-row fixed-width table with three null masks and representative
  integer, float, string and list-valued attributes. Passing a remote URL opens
  it, reports schema and attributes, reads sample rows twice, and reports time,
  requests and transferred bytes for metadata, cold and warm access.
- Updated Python examples to promote the public `attrs` alias instead of
  `vlmeta`; compatibility tests continue to cover the older name.

## Evidence from the current implementation

- `ctable.py`: `_open_from_storage()` reconstructs a table without the original
  Python row class; `_LazyColumnDict` opens physical columns on demand.
- `ctable_storage.py`: fixed-width columns, `_valid_rows` and null masks are
  NDArrays. The schema and table metadata live in `_meta` SChunk vlmeta.
- `b2z_source.py`: `B2ZNDSource` reads external, unencrypted ZIP_STORED `.b2nd`
  members. `member_vlmeta()` and `B2ZEmbeddedMetadata` provide metadata reads.
- `remote_store.py`: discovery recognizes CTable roots and nested table
  boundaries, opens supported table nodes as RemoteCTable objects, and keeps
  their internals opaque during hierarchy traversal.
- Persisted scalar index descriptors use the remote table owner to resolve
  validated B2Z sidecars. Batch-backed columns use local `.b2b` opening paths.

A disposable probe in the `blosc2` environment created a numeric CTable archive,
uploaded it to fsspec `memory://`, and opened it through a minimal storage adapter
returning RemoteArray leaves. Row count, column slicing and a `where()` query
worked. Scalar access and `sum()` failed because CTable calls
`iterchunks_info()`, which RemoteArray does not provide. This is evidence of
integration feasibility, not complete correctness or network performance.

## Decisions resolved for the first release

### Public API

Available usage:

```python
with blosc2.RemoteCTable("https://example.org/measurements.b2z") as table:
    values = table["temperature"][:10]
    selected = table.where(table["temperature"] > 20)
    values = selected["temperature"][:]

with blosc2.RemoteCTable(
    "s3://bucket/archive.b2z",
    dataset="experiments/run1",
    storage_options={...},
    cache_dir="table-cache",
) as table:
    values = table["temperature"][:10]

with blosc2.RemoteStore("s3://bucket/archive.b2z") as store:
    with store["experiments/run1"] as table:
        values = table["temperature"][:10]
```

- Constructor keywords: `dataset`, `storage_options`, `cache_policy`,
  `max_cache_bytes`, `cache_dir`. Reuse existing container URL parsing and
  validation; no separate source-format or writable-mode option.
- Match RemoteStore defaults: bounded MEMORY caching without `cache_dir`, DISK
  when `cache_dir` is supplied, and explicit NONE support. Reuse its validation
  and default limit rather than duplicate constants. No per-column cache budget.
- Expose `source`, `traffic`, `cache_policy`, `max_cache_bytes`, `cache_bytes`
  and `metadata_bytes` using owner accounting. Document that metrics are shared
  with sibling objects when obtained through a RemoteStore.
- `RemoteStore.kind()`/`get_info()` gain a `ctable` kind; indexing a supported
  table node returns RemoteCTable. Preserve the table as an opaque hierarchy
  node: `_cols`, `_meta` and index files are not ordinary public children.
- A standalone table archive is opened with RemoteCTable. RemoteStore retains
  its group-root requirement and gives a diagnostic directing users to it.
- The initial release did not change `blosc2.open()` or `CTable.open()` URL
  dispatch. A subsequent extension adds remote table/group dispatch to
  `blosc2.open()`; `CTable.open()` remains the local table opener.

### Supported data and operations

- First-release data: external fixed-width NDArray columns, including supported
  fixed-width strings, timestamps, booleans and per-row NDArray values; row
  validity and nullable-column masks; nested schema paths whose leaves qualify.
- First-release behavior: metadata and schema inspection, logical row count,
  scalar/slice/fancy reads, row iteration, column iteration, filtering and
  reductions that local CTable supports for those types. Preserve deleted-row,
  spare-capacity and null semantics, including on filtered views.
- Computed columns may reuse the existing safe expression machinery when all
  dependencies are supported. Do not execute arbitrary serialized callables or
  import code named by remote metadata. Give a precise error for unsupported
  expression forms or dependencies.
- Unsupported columns do not prevent schema inspection or reading supported
  siblings. Opening such a column raises `NotImplementedError` naming its path,
  storage/type and limitation. Whole-table operations requiring it also fail
  explicitly; never silently omit a column or download the whole archive.
- UTF-8 offsets/data arrays are a follow-up, despite fitting RemoteArray well;
  their wrapper and query paths need their own compatibility checks. Lists,
  other batch-backed variable-length values and dictionary value stores are
  also deferred. Dictionary codes alone do not constitute dictionary support.
- Embedded array payloads, encrypted members and ZIP-compressed members retain
  existing remote-reader limitations. Reuse supported embedded metadata access
  where possible; reject unsupported manifests explicitly.

### Queries, writes and persistence

- Initial queries scan required columns. The remote storage backend returns no
  usable index catalog, so persisted indexes cannot accidentally enter local
  sidecar opening paths. Document this even when the source contains indexes.
- Preserve local optimized paths. Where CTable assumes native chunk metadata,
  add a bounded slice-based fallback in the shared helper after tracing callers.
  Do not fabricate special-chunk metadata or make ordinary reads depend on a
  local cache SChunk: NONE must work too.
- Opening must not read all column payloads or scan the validity mask when the
  saved row count is available. Missing row counts and deleted-row position
  resolution may require bounded mask scans; document their cost.
- Reject data/schema mutation, index creation and remote metadata writes before
  changing any state. Source read-only status is separate from cache mutability.
  Audit inherited methods and raw column/metadata access, not only assignment.
- Materializing supported data into a new local CTable through existing copy or
  save operations is allowed and must produce an ordinary local table. It must
  not accidentally construct a RemoteCTable with missing owner state.
- Remote-reference artifact serialization, automatic b2object registration,
  sparse-cache convenience APIs and remote writes are out of scope. Distinguish
  local data export from saving a portable remote reference.
- RemoteStore artifact export encountering a table must fail explicitly until
  table-reference serialization is implemented; never silently drop that node.

### Lifetime and source identity

- Acquire/release the existing shared discovery owner. Closing a RemoteStore
  leaves an already returned RemoteCTable usable. Closing a table releases its
  resources; independent raw RemoteArray handles keep their own owner leases.
- Table Column objects and table views are borrowed from their root table and
  require it to remain open. Close on a view must not close the root. All access
  after root close must fail consistently, including opening an untouched column.
- Audit CTable view/copy constructors, some of which use `cls.__new__` and others
  `CTable.__new__`. Read-only views may be ordinary CTable objects, provided root
  ownership and generation checks are enforced; detached copies are local CTable.
- The remote archive is immutable for the session. Reopen a directly constructed
  RemoteCTable to observe replacement; no table-level `refresh()` in version one.
- Existing RemoteStore root `refresh()` invalidates previously returned tables,
  their views and columns, including metadata-only operations. Preserve atomic
  refresh failure behavior and check generations before opening new leaves.
- Cache identity includes the archive identity, table/leaf path and existing
  storage-options fingerprint. Do not persist credentials in source descriptors.

## Implementation sequence

1. Extend B2Z discovery with table descriptors and internal member lookup.
   Validate table kind, supported schema version, paths and required metadata.
   Reuse archive/metadata readers; retain table boundaries and existing limits.
   Update cached discovery manifest validation for the new node kind, with safe
   rejection or rediscovery of incompatible cached metadata.

2. Add `RemoteTableStorage` in `ctable_storage.py` and a thin public class in
   `remote_ctable.py`, exported from `__init__.py`. Implement schema, validity,
   null-mask and column opening, epoch reads, empty index catalog, read-only
   metadata and cleanup. Construct through `_open_from_storage()`. Extract only
   the owner-creation code needed to avoid duplicating RemoteStore initialization.

3. Repair shared CTable reads for remote operands. Start with
   `_find_physical_index()` and `Column.iter_chunks()`, then trace null handling,
   reductions, fancy indexing, expressions and materialization. Audit NDArray
   type checks, `.schunk`, `.urlpath`, native-extension calls and local sidecar
   opening. Reuse slice-based traversal where it already exists. Keep local
   special-chunk optimizations and add no broad array protocol refactor.

4. Integrate `RemoteStore[table_path]`, shared cache accounting, close behavior,
   generation invalidation and root diagnostics. Ensure view construction and
   local copy/save behavior follow the decisions above. Guard unsupported remote
   reference export paths.

5. Add focused regression coverage and API documentation. Describe supported
   columns, borrowed views, immutable sources, scan costs, cache accounting and
   unsupported indexes. Add one small local/remote usage example. No C/Cython
   changes are anticipated; establish a concrete need before expanding scope.

## Verification and acceptance

Use the `blosc2` conda environment for all Python and test commands. Add tests
under `tests/ctable/` and extend `tests/test_remote_store.py` as appropriate.

- Compare remote results with a local read-only CTable from the exact same
  archive: empty and nonempty tables, chunk edges, deleted rows, spare capacity,
  masks/sentinels, scalar/fancy/strided reads, filtering and reductions. Include
  fixed-width string, timestamp, nested and NDArray-valued columns.
- Cover both standalone and nested tables, unsupported-column isolation,
  malformed schema/metadata and missing members. Validate read-only enforcement
  and that local exports have the intended class and values.
- Exercise NONE, MEMORY and DISK using an instrumented fsspec memory filesystem.
  Check lazy opening, bounded range requests, reuse of warmed chunks, aggregate
  cache limits and disk reopen. Account for existing bounded small-member
  prefetch; do not assert that no payload byte ever arrives with metadata.
- Use a deterministic local HTTP range server to verify actual range transport
  without cloud credentials. Keep optional external-service tests network-marked.
- Test parent-store close, root-table close, borrowed views, raw-array leases,
  refresh invalidation and failed refresh. Ensure cleanup on partial open failure.
- Re-run relevant existing CTable, remote array/store and B2Z tests, followed by
  the default suite before declaring implementation complete. Warnings are errors.
- Record a small cold/warm read experiment on an archive much larger than the
  requested slice: report transferred bytes, requests and retained payload.
No universal latency target; the acceptance criterion is bounded on-demand I/O
and correct results, not a claim that scan queries avoid reading their operands.

### Results recorded for the initial implementation

- The default suite passed with 10,209 tests and 36 skips after the core
  implementation. Ruff passed for the changed source and test files.
- The focused RemoteStore and RemoteCTable suites passed with 157 tests and five
  skips after the S3 lifecycle fix.
- The generated example archive round-tripped through local CTable and fsspec
  `memory://`, preserving all three null masks and all four user attributes.
- A one-million-row `readings.b2z` uploaded to Backblaze B2 was opened from
  `s3://blosc2/readings.b2z` through its S3-compatible endpoint. Metadata and
  sample rows were read on demand; the repeated sample read issued zero requests
  and transferred zero bytes from the warm memory cache.

### Results recorded for the batch-backed extension

- The default suite passed with 10,315 tests and 36 skips. The focused remote
  table suite passed with 90 tests, and Ruff passed for all changed Python files.
- An instrumented `memory://` experiment used a 55,026,931-byte archive with
  100,000 rows, 1,024 rows per variable-length batch, and a 20,000-value
  dictionary. A distant five-row read used 5 requests and 575,310 bytes, retained
  558,335 compressed bytes, and repeated with 0 requests and 0 bytes.
- The dictionary code slice used 2 requests and 4,612 bytes. First decode then
  loaded the vocabulary with 10 requests and 95,494 bytes. The resulting Python
  strings and lookup maps occupied approximately 4,028,168 bytes by
  `sys.getsizeof`, outside the compressed transport cache budget. Metadata opening
  used 2 requests and 9,297 bytes.

## Follow-ups, separately scoped

1. UTF-8 columns through remote offsets and bytes, with null/query/size reporting:
   implemented in the v2 extension described in `remote-ctable-v2.md`.
2. Remote batch reads for lists, variable-length values and dictionary stores:
   implemented in the extension described in `remote-ctable-batches.md`.
3. Persisted indexes through a remote-aware sidecar resolver: implemented for
   SUMMARY, FULL, PARTIAL, OPSI and BUCKET scalar indexes as described in
   `remote-ctable-indexes.md`. List membership indexes continue to use selective
   posting reads.
4. Portable RemoteCTable references, RemoteStore artifact inclusion and sparse
   runtime-cache APIs if needed by actual consumers: reference saving and artifact
   inclusion are implemented. `RemoteCTable.save()` delegates to the shared
   exporter; `blosc2.open()` reconstructs table-root artifacts and nested tables.
   Root/nested round-trip tests live in `tests/ctable/test_remote_ctable.py`.
   `RemoteStore.with_sparse_cache()` and `RemoteCTable.with_sparse_cache()` supply
   shared sparse runtime-cache APIs. Table tests cover concurrent handles,
   cross-handle cache reuse, refresh invalidation and referenced RemoteArray
   column reuse.

Additional extensions include standalone `refresh()`, bounded parallel column
reads (`remote-ctable-parallel.md`), and `sources=` bindings for NDArray and
RemoteArray columns (`ctable-remote-cols.md`). External column reads use the
outer RemoteCTable cache owner and budget.

There are no blocking API questions or planned feature gaps left for the
read-only RemoteCTable scope described here.
