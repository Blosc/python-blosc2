# RemoteArray columns in CTable and RemoteCTable

Status: implemented for fixed-width NDArray and RemoteArray columns.

## Objective

Allow a CTable column to use an existing RemoteArray without copying its data.
Keep the row schema independent of source locations and bind arrays through a
new keyword-only `sources=` constructor argument. Support Caterva2 and fsspec
through the existing RemoteArray interface. Persist these references using the
same carriers already supported by TreeStore leaves.

The first implementation is a read-only table with fixed-width columns. It must
support slices, row access, projections, filtering, expressions, save/reopen and
explicit materialization. It does not add remote writes or a new transport.

## Public API

```python
from dataclasses import dataclass
import blosc2 as b2


@dataclass
class Measurement:
    station_id: int = b2.field(b2.int32())
    temperature: float = b2.field(b2.float32())


table = b2.CTable(
    Measurement,
    sources={
        "station_id": b2.RemoteArray("s3://weather/station_id.b2nd"),
        "temperature": b2.RemoteArray(
            b2.URLPath(
                "weather/temperature.b2nd",
                urlbase="https://caterva.example.org",
            ),
            cache_policy=b2.CachePolicy.MEMORY,
        ),
    },
)

table["temperature"][:10]
```

This is proposed syntax. `RemoteArray` and `URLPath` already exist; `sources=`
does not. Do not add `source=` to `field()` or a remote schema type.

Define `sources` as a mapping from compiled column names to `RemoteArray` or
`NDArray`. Allow NDArray values so local and remote columns can coexist without
another ingestion API. Require every stored schema column exactly once; reject
unknown or missing names. For nested schemas, use existing flattened leaf names.
Do not accept bare URL strings, arbitrary Proxy objects or NumPy arrays initially.
Callers can construct RemoteArray or use `blosc2.asarray` explicitly.

`sources=` is a creation path and is mutually exclusive with `new_data`. Reject
it when reopening an existing table. Treat an empty mapping as an explicit
request, not as omission. Preserve the existing constructor behavior when
`sources is None`.

## Source and cache contract

- Caterva2: accept RemoteArray constructed from an explicit URLPath. A plain
  HTTPS string retains its existing fsspec meaning.
- fsspec: support the formats RemoteArray already supports: B2ND, B2Z array
  datasets, HDF5 datasets and Zarr arrays. Protocol drivers and range-access
  requirements remain those of the existing readers.
- Direct `RemoteArray(...)` defaults to NONE. `blosc2.open(..., lazy=True)`
  defaults to MEMORY without an explicit disk cache location. A local CTable
  preserves supplied arrays' policies; it has no table-wide cache policy.
- MEMORY retains compressed data in process memory, with its existing default
  limit of 256 MiB per independent array. Its policy survives serialization;
  its warm payload does not. DISK and NONE keep existing RemoteArray semantics.
- RemoteCTable controls caching for everything read through it, including
  external columns. Its policy, cache location and shared `max_cache_bytes`
  budget override column carrier settings for table-owned runtime handles.
  Standalone arrays retain their own settings. Do not mutate supplied handles,
  original carriers or independently opened arrays when applying this override.
- Outer NONE retains no fetched column payload, MEMORY uses one shared memory
  budget, and DISK uses the table's local cache directory and one shared disk
  budget. These are retained compressed-payload bounds, not total process RAM
  or limits on temporary read buffers and results. Preserve existing metadata
  accounting and read-only artifact rules.
- External columns must use the table's cache coordinator rather than allocate
  independent per-array caches. Aggregate their requests into table traffic
  accounting without double counting. Cache keys must distinguish source
  identity/version, dataset and array geometry; keep existing credential-scope
  isolation and source validation. Equal chunk numbers in different sources
  must never collide.
- A DISK column needs table-owned writable local cache storage; a carrier inside
  a remote archive cannot be updated in place. Ignore serialized runtime cache
  locations and reuse existing owner-managed cache allocation. Do not create
  disk caches under outer NONE or MEMORY because a carrier requests DISK.
- Credentials remain runtime configuration. Reuse existing carrier URL checks
  and credential-free serialization; do not add secrets to table metadata.
  Persisted private sources use the existing RemoteArray authentication mechanisms.
  Do not forward outer archive credentials automatically to another host.

## Construction and validation

Add the sources branch before ordinary capacity allocation and column creation
in `CTable.__init__`. Compile the normal schema first, then validate the entire
mapping before creating or overwriting destination storage.

1. Accept fixed-width scalar columns and fixed-shape NDArraySpec columns.
   Reject UTF-8, variable-length, list, dictionary and object column bindings in
   this first version; their storage spans multiple coordinated components.
2. Require exact storage dtype compatibility. Scalars require shape `(N,)`;
   NDArraySpec columns require `(N, *item_shape)`. Do not silently cast, fetch,
   rechunk or recompress a reference to satisfy the schema.
3. Infer N from the sources and require equal first-axis lengths. Accept N=0.
   Metadata checks must not scan payloads. Schema value constraints such as
   `ge` and `le` cannot be verified without reads: reject such constrained source
   bindings with `validate=True`, explaining that `validate=False` skips value
   validation but never dtype/shape checks.
4. Initially reject nullable bindings requiring a separate validity mask. Permit
   supported explicit sentinel null representations using existing decoding.
   Add remote validity-sidecar binding in a subsequent extension.
5. Create the normal valid-row array with N live entries, set row-count/last-pos
   metadata consistently, and install the source objects as columns. Any minimum
   backing capacity required for an empty validity array must stay marked false.
   Do not apply the usual default million-row allocation to source columns.
6. Source geometry is authoritative. Reject conflicting per-column chunks,
   blocks or compression settings rather than silently ignoring them. Table
   storage settings may still configure table-owned metadata arrays.

Source row i represents table physical row i. Matching lengths do not prove that
two datasets describe corresponding observations; alignment is the caller's
responsibility. No joins or implicit key matching occur.

Treat supplied handles as borrowed: closing the table must not close a caller's
RemoteArray. Handles opened from persisted carriers are table-owned. Document
that independently mutating a supplied local NDArray is outside this read-only
table contract. Reuse existing source identity checks; independently changing
remote datasets do not form a transactional snapshot.

## Storage and persistence

Reuse TableStorage and its existing column paths. Add only the small column
installation operation needed by the supported storage backends: in-memory
storage retains a handle, persistent storage writes a RemoteArray carrier or
copies a local NDArray through existing store facilities. Never persist
`remote.cache` as though it were a complete data column.

Record which columns are external references in table storage metadata. Reuse
the carrier as the authoritative source descriptor, including its standalone
cache policy. RemoteCTable's runtime override does not replace that descriptor;
do not duplicate its URL/authentication/format schema in table metadata. Add a
versioned table capability marker, with readers rejecting unknown versions,
so older code cannot interpret an incomplete carrier as ordinary column data.
Inspect both schema and table-kind version checks before choosing the marker.

Audit all backends: InMemoryTableStorage, FileTableStorage,
EmbedStoreTableStorage and TreeStoreTableStorage. Use their normal leaf open
paths and existing object reconstruction. A persistent constructor with
`sources=` must write references and all required metadata before exposing the
table as read-only. Reopen restores that restriction even with mode='a'.

Keep existing materializing export behavior explicit:

- Add `preserve_sources=False` to CTable.save/to_b2z; their existing defaults
  continue writing independent local data. `preserve_sources=True` writes
  carriers and the table capability marker without scanning source columns.
- TreeStore assignment of an unfiltered reference-backed root table preserves
  its references, as ordinary RemoteArray leaf assignment already does.
- Preserve references only for the identity row mapping, including column-only
  projections. Reject reference-preserving exports of filtered/reordered views
  or requests to rechunk referenced columns. Their ordinary materializing
  exports remain available. Do not serialize a view as an unfiltered source.
- `copy()` continues to produce independent materialized columns, removing the
  external-reference marker. Document its read and storage cost.
- RemoteCTable.save already exports a remote-reference artifact. Extend that
  path to preserve nested column carriers; do not replace its existing contract
  with the local CTable.save default. Persist the effective outer policy and
  shared budget in the owner manifest, retaining nested standalone descriptors.
  Reopening the artifact applies the outer policy to all columns again. Export
  retained external payload through the owner cache using existing include_cache
  semantics; MEMORY exports reopen cold. Never serialize runtime cache paths.

In particular, `_save_to_storage()` currently compacts live rows and allocates
new arrays. Reference preservation needs an explicit branch before that copying
path. Cover `_save_to_treestore`, `to_cframe` and embedded storage too; no export
path may accidentally serialize only missing-cache placeholders as local data.

## RemoteCTable opening and reads

RemoteTableStorage currently opens each column as an array member of one B2Z
archive. Extend discovery/opening to distinguish an ordinary array member from
a RemoteArray carrier. Resolve the latter through existing carrier decoding,
so its target may be Caterva2 or a different fsspec object. Do not treat the
carrier's placeholder chunks as the column's actual values.

Construct external runtime handles under the outer owner before any cache is
allocated. Extend the existing cache coordinator/owner integration in
remote_store.py and remote_array.py to cover their source types, including
Caterva2. Resolve identity and seed-cache validity through existing source
checks. Warm carrier payload may be reused only under the outer policy and
shared budget; it must not enable hidden retention under NONE. Avoid constructing
an independently cached RemoteArray and changing its policy after the fact.

`ctable_remote_read.py` currently assumes B2Z member offsets, an archive transport
and a shared store owner. Keep that optimized path for ordinary archive columns.
For independent external sources, use a table-owned array's public indexing path
with the shared cache coordinator and existing concurrency behavior. Ensure this
fallback participates in eviction, traffic accounting and owner lifecycle checks.
Start with a correct fallback; do not generalize
the B2Z scheduler into a new transport framework. Preserve column order and
physical row selections when combining local/archive/external reads.

Audit Column access, `_fetch_col_at_positions_uncached`, expression construction,
reductions, null handling and NDArray-only type checks. Broaden only checks where
RemoteArray satisfies the actual read contract. Differently partitioned sources
must use existing general evaluation paths rather than assuming aligned chunks.
Do not access raw incomplete caches to satisfy an NDArray-only fast path.

External arrays opened by RemoteCTable must participate in close/stale-handle
checks. Refreshing the outer table discards owned external handles and reopens
them lazily. Refresh must not claim atomic version alignment across sources.
An unavailable external column should fail when accessed without preventing an
unrelated column from being read where metadata permits lazy opening.

## Mutation and indexing

Make the whole reference-backed table logically read-only in this first version,
including mixed local/remote tables. Reuse the existing read-only guards, and
audit append/extend, assignment, deletion, resize, compact, add/drop columns and
materialized-column maintenance for bypasses. Raise before any partial mutation.
Changing cache contents remains allowed according to the effective cache policy:
per-array for local CTable, outer-owner policy for RemoteCTable.

Disable automatic SUMMARY-index creation for source-backed construction and
reopen: closing a handle must not download columns. Initially reject explicit
persistent index creation on these tables and ignore/reject stale index metadata
for external columns. Read-only scans and transient expression evaluation remain
supported. Source-version-aware persisted indexes are a later feature.

## Implementation sequence and verification

1. [x] Implement constructor binding, validation, read-only behavior and fixed-width
   read/query paths in ctable.py and ctable_storage.py. Preserve ordinary-table
   behavior. Start tests with deterministic fsspec memory sources.
2. [x] Add versioned metadata and carrier-preserving local/store exports and reopen.
   Reuse dict_store.py/RemoteArray serialization; avoid a parallel Ref resolver.
3. [x] Extend RemoteStore discovery, RemoteTableStorage and the remote read fallback
   to open and query external column carriers inside remote B2Z tables. Integrate
   all external source types with the outer owner's cache budget and traffic
   accounting; this is required functionality, not a later optimization.
4. [x] Document the constructor, source immutability/alignment, local per-array versus
   RemoteCTable-wide policies and budgets,
   credential handling and reference versus materialized exports. Add a small
   example under examples/ctable with both Caterva2 and fsspec bindings.

The implementation landed in four focused commits:

- ``7a53f6e1`` adds source-bound construction, validation, reads, queries and
  read-only behavior.
- ``b2b57c9a`` adds versioned metadata, materializing and reference-preserving
  exports, and TreeStore/CFrame round trips.
- ``d362c3e7`` opens external carriers through RemoteCTable and applies the
  outer cache owner, budget, traffic accounting and artifact policy.
- ``d3a828f5`` documents the API, cache ownership and mixed remote sources.

Verification used focused pytest coverage under the ``blosc2`` conda environment:

- Same-length and empty sources; mixed NDArray/RemoteArray columns; scalar and
  fixed-shape cells; mismatched names, dtype, shape and conflicting options.
- Slices, stepped/fancy selections, rows, projection, filtering and expressions
  across columns with different chunk/block grids, compared with NumPy values.
- Mutation rejection before writes, and borrowed versus owned handle lifetimes.
- Local CTable NONE and MEMORY policy preservation; a warm MEMORY source reopens cold;
  DISK retains only the cache allowed by existing export semantics.
- Parameterize outer NONE/MEMORY/DISK against each persisted column policy.
  Verify the outer policy wins without changing standalone handles/descriptors.
  Under outer NONE/MEMORY, an inner DISK policy must not create disk caches.
- Read multiple ordinary and external columns past one shared budget and verify
  cross-column eviction. Test distinct sources with matching geometry/chunk
  numbers for cache isolation, plus changed-source invalidation. Temporary
  working buffers remain outside the retained-cache budget.
- Verify aggregate table traffic includes external requests exactly once, warm
  reads reuse the owner cache, DISK stays under the owner cache directory, and
  saved remote-reference artifacts restore the outer policy/shared budget.
- Construction and reference exports make no payload reads after source metadata
  discovery. Repeated MEMORY reads reuse fetched data; NONE does not retain it.
  Account for archive metadata prefetch separately from full-column reads.
- Local save/open, TreeStore B2Z and embedded round trips, explicit materializing
  export/copy, and rejection of nonidentity reference-preserving views.
- A remote B2Z table whose columns reference independent fsspec and mocked
  Caterva2 sources; close/refresh invalidation and one unavailable source.
- Serialized artifacts contain no credentials or runtime cache paths; legacy
  tables continue to open without a new marker.

The new focused suite passes with all three outer cache policies. The existing
RemoteCTable, RemoteStore and RemoteArray suites pass (350 passed, 5 skipped),
as do the persistence/CFrame suites (82 passed). Ruff passes for every changed
Python file. The Sphinx build reaches completion but ``-W`` remains non-zero
because the repository already emits unrelated autosummary, duplicate-target,
toctree and theme warnings.

## Deferred work

Remote writes, mutable hybrid tables, row remapping of references, nullable mask
bindings, composite/variable-length remote columns, persisted remote indexes,
and a unified parallel transport scheduler are
outside the first implementation. No new dependency is required.
