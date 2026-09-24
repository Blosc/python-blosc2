# Remote PyTables tables through RemoteCTable

## Objective and agreed scope

Make PyTables `Table` datasets in remote HDF5 files accessible as read-only
`RemoteCTable` objects through the existing fsspec infrastructure and, subsequently,
Caterva2 integration.

The remote file remains in its original PyTables/HDF5 format. Local cached table
data and imported indexes use native Blosc2 representations. PyTables format
knowledge belongs in discovery and conversion code, not in the query planner or
index-search implementation.

Table data is fetched and cached on demand; opening a table must not require
downloading all its records. The initial index importer eagerly converts each
selected supported index into reusable native sidecars. This does not imply
importing every index automatically at table open.

Supported index imports are limited to **clean PyTables `full` indexes covering
every table row**, including both partially sorted full indexes and completely
sorted indexes (CSI). An incomplete final sorted slice is normal and supported.
Dirty indexes and indexes with incomplete row coverage are outside the initial
import scope.

## Fallback behavior

PyTables `medium`, `light`, and `ultralight` indexes are ignored. Their presence
does not prevent table access.

Queries without a usable imported index use normal Blosc2 predicate evaluation
over the remotely backed table. Required HDF5 record chunks are fetched on demand
and their data cached locally in native Blosc2 format. Later queries can reuse
that cache. Another supported index in the same query may still narrow the rows
that need scanning, where the existing planner supports that predicate.

Do not automatically build replacement indexes for unsupported index kinds:
doing so would require a column scan and additional local storage. Encountering
a dirty or incompletely covering index likewise leaves the table readable through
the scan path. Malformed metadata is a separate validation concern and must not
be treated as a trustworthy index.

## Table representation and access

A PyTables Table is a one-dimensional HDF5 compound dataset identified by
`CLASS="TABLE"`. Its dtype describes fields, while attributes supply additional
table and field metadata.

Extend HDF5 discovery to recognize table nodes and derive a CTable schema. Expose
fields as lazy columns over a shared structured-record source, provide the
all-valid row representation required by CTable, and preserve appropriate user
attributes. Distinguish user metadata from PyTables implementation metadata and
keep index datasets available internally without presenting them as table columns.

Start with scalar numeric, Boolean, and fixed-width byte-string fields. Define
explicit behavior for nested records, shaped fields, enums, and PyTables time
types before claiming support. In particular, PyTables `Time64` has conversion
semantics that cannot be inferred by treating its storage as an ordinary number.

Fixed-width strings are exposed as native CTable `bytes(max_length=N)` columns
backed by NumPy `S<N>` fields. Values remain bytes, comparisons use byte literals,
and no UTF-8 decoding is implied. This preserves the HDF5 fixed-width storage
contract, including its maximum width, while avoiding h5py's separate string
metadata from leaking into the CTable schema.

Reuse HDF5NDSource transport, supported-filter decoding, cache accounting, and
RemoteStore ownership where possible. RemoteTableStorage and its batched reader
currently assume B2Z members and separate column arrays; recognizing a table node
alone does not remove those assumptions.

### Shared physical reads

PyTables stores records together. Selecting a field from a compressed dataset
generally requires fetching the compressed chunk containing all fields. Multiple
column accesses must share physical fetches and decoding rather than independently
downloading the same HDF5 chunk.

Keep transient record buffers bounded, group row gathers by physical chunk, and
account for shared cache usage and concurrent requests. Persisted cache data must
remain Blosc2-native. Preserve read-only, close, refresh, and stale-handle behavior.

## Native OPSI index conversion

PyTables indexes reside in groups such as `/group/_i_table/column`. Full indexes
retain every indexed value and its absolute 64-bit row position; their sorted
values are not reduced. Consequently, the sorting and OPSI optimization already
performed by PyTables can be reused.

| PyTables representation | Native Blosc2 representation |
| --- | --- |
| `sorted`, a two-dimensional array of sorted slices | One-dimensional OPSI values sidecar |
| Corresponding `indices` | One-dimensional `int64` positions sidecar |
| Valid prefix of `sortedLR` | Final entries of the values sidecar |
| Valid prefix of `indicesLR` | Final entries of the positions sidecar |
| `abounds` / `zbounds`, or imported block endpoints | OPSI minimum/maximum navigation sidecars |
| Index geometry and table binding | Native index descriptor |

Validate supported format versions, clean state, full row coverage, dataset
shapes, entry counts, dtypes, and row-position range before publishing the native
descriptor. Normalize byte order and validate unsigned positions before converting
them to signed `int64`. Trim last-slice arrays using their valid lengths; exclude
padding and the bounds stored after the valid `sortedLR` payload. Older index
versions encode last-slice lengths differently and need explicit handling or an
unsupported-version diagnostic.

### Preserve sorted-run boundaries

Every native OPSI chunk must be sorted. Do not let a native chunk straddle two
independently sorted PyTables slices: Blosc2 can binary-search adjacent blocks as
one span, so independently sorted blocks alone are insufficient.

A simple initial geometry maps one PyTables index chunk to one native Blosc2
chunk with one compression block. Larger chunks are possible when their boundaries
respect sorted slices. The final incomplete slice follows the same rule.

Flatten and repack the existing value/position pairs without re-sorting, merging,
or running OPSI optimization cycles. Generate navigation bounds for the chosen
native geometry, reusing stored bounds only when their semantics and geometry
match. The imported index retains the source ordering quality, although query
performance can change with navigation and compression geometry.

Register the result as an ordinary native OPSI index. The planner and query
executor must not dispatch to a PyTables-specific search implementation. CSI is
included in this path; a separate conversion to native FULL is not required for
the initial implementation.

### Original-row summaries

Blosc2's summaries over original row segments are distinct from bounds over
sorted index blocks. Do not substitute PyTables sorted bounds for these summaries.

Derive required summaries from imported value/position pairs, or explicitly
support their absence in the relevant native paths. Avoid accidentally invoking
the normal index builder and scanning the table payload merely to populate
summaries. Any omitted summaries must leave fallback planning correct.

### Cost and persistence

Eager conversion is a streaming O(N) repack: read/decompress the existing index,
normalize and reshape it, then compress native sidecars. It avoids table-record
reads and index sorting, but still transfers the selected index's entire payload.
For scale, one billion float64 values plus 64-bit positions occupy approximately
16 GB before compression.

Reuse converted sidecars across queries and opens. Bind them to the remote source
identity/version, dataset, index identity, and conversion format version. Publish
them only after a successful conversion; partial or stale imports must not become
queryable. The original HDF5 file remains unchanged.

Compressed-byte copying is not the general conversion path: PyTables indexes can
use zlib/shuffle or other HDF5 filters. Any future compatible Blosc2 chunk-copy
optimization requires separate validation.

## Why lighter indexes are not imported initially

Medium, light, and ultralight indexes discard exact within-bucket row positions.
Some also retain only sampled sorted values. Metadata translation cannot recover
the missing information needed by native exact OPSI indexes.

Native BUCKET indexes are not a direct substitute: they use a different layout
organized around source chunks and their own value-reduction scheme. Converting
approximate indexes into that representation is a separate investigation, not a
requirement for this plan. The agreed fallback is scanning, not reconstruction.

## fsspec and Caterva2

Implement and validate the storage/conversion path with fsspec first. For Caterva2,
reuse the same native representation and keep PyTables interpretation at the
source/preparation boundary. Server-side preparation could convert indexes once
near the HDF5 file and serve reusable native sidecars to clients.

Caterva2 table discovery, metadata, and serving integration need explicit work;
existing HDF5 array support does not establish end-to-end table support. A second
PyTables predicate engine on the server is not part of this proposal.

## Implementation sequence

1. [x] Fix HDF5 metadata compatibility prerequisites and add representative fixtures.
2. [x] Recognize PyTables tables and implement lazy shared-record access with native
   local caching; verify scan queries and read-only lifecycle behavior.
3. [x] Import clean, fully covering full/CSI indexes into native OPSI sidecars and
   register them with the existing planner. Keep unsupported-index scan fallback.
4. [x] Add conversion reuse, explicit-refresh invalidation, and interrupted-import checks.
5. [x] Integrate the same representation into Caterva2 and measure cold/warm behavior.

## Implementation status

Implemented on 2026-09-21 in these sequence commits:

1. `3de1681f` — fixed HDF5 metadata decoding for h5py fixed strings and empty
   fixed-width scalar attributes.
2. `3dab45c6` — added PyTables table discovery and lazy shared-record
   `RemoteCTable` access.
3. `d5c1ffab` — imported clean, fully covering 64-bit full indexes into native
   OPSI sidecars, including fixed-width byte-string indexes.
4. `27bd0ef0` — persisted converted sidecars and added completion-marker recovery.
   Cached generations now remain valid until explicit refresh, matching the
   immutable-source contract.
5. `36602d09` in Python-Blosc2 and `501c108` in Caterva2 — enabled portable
   HDF5 CTable stores and Caterva2 metadata/filter/fetch handling through
   `RemoteCTable`.

The table fields share one remote structured-record array, so projecting several
columns reuses the same HDF5 payload cache. Imported indexes are ordinary native
OPSI descriptors; neither the planner nor Caterva2 contains a PyTables-specific
search engine. Disk-cache imports write each Blosc2 sidecar atomically and publish
`complete.json` last. A missing, corrupt, or incomplete publication is rebuilt.

Caterva2 integration uses its portable `RemoteStore` path: a remote HDF5 table is
reported as a CTable, filtered through `RemoteCTable.where()`, and returned as a
normal CTable cframe. The direct local-HDF5 adapter remains an array-oriented
`HDF5Proxy`; converting that separate upload/unfold path was not required for
remote first-class access.

## Validation and findings from the feasibility analysis

Read-only probes were run in the `blosc2` conda environment using h5py and
PyTables' checked-in fixtures. PyTables itself was not installed in that environment.

- RemoteArray successfully read structured records from `bug-idx.h5` through
  fsspec's memory filesystem and matched h5py.
- Remote reads of `sorted`, `indices`, and `sortedLR` from `indexes_2_1.h5`
  matched h5py.
- HDF5 table discovery exposed an empty-byte-string attribute decoding failure
  (`itemsize cannot be zero in type`).
- A compound dtype containing h5py string metadata failed reconstruction
  (`invalid shape in fixed-type tuple`).
- An in-memory conversion of the full `var4` index in `indexes_2_1.h5` passed
  924 range cases through the native OPSI reader and public indexed expressions,
  including inclusive/exclusive bounds and the incomplete final slice. It did
  not sort or call `create_index()`. This small CSI fixture demonstrates the
  mapping, not broad compatibility or performance.
- Tiny multi-block native arrays exposed an incorrect span read in the current
  environment. The successful conversion probe used single-block chunks. Resolve
  this separately before claiming arbitrary native geometry support.

Implemented automated checks cover lazy shared record access, scan fallback,
numeric and fixed-width byte-string OPSI queries, rejection of light indexes,
disk conversion reuse, incomplete-publication rebuild, explicit source refresh,
portable-store validation, Caterva2 metadata/fetch integration, and sliced CTable
materialization. The Caterva2 cold/warm check records HDF5 chunk reads for the
first indexed request and verifies that repeating the request adds zero HDF5
chunk reads.

Further compatibility checks should include non-CSI full indexes with overlapping
sorted slices, nontrivial row permutations, duplicate values, numeric boundaries,
NaNs, supported strings, byte order, empty tables, and incomplete final slices.
Compare imported-index queries with forced scans and, where available, PyTables
results. Verify that unsupported index kinds remain readable through scanning.

Broader scale checks should confirm that conversion does not read table payload to rebuild the index, that
multi-column selections share physical HDF5 reads, and that warm caches avoid
unnecessary refetches. Exercise cache limits, close/refresh behavior, interrupted
conversion, and source changes.

Benchmark converted-index size, conversion time, transferred bytes, request
count, and cold/warm query latency. Include selective predicates and wide-record,
narrow-projection scans: indexing can reduce record reads, but cannot make the
remote row-oriented layout columnar.

## Deferred work

- Lazy conversion of index payload chunks behind native sidecar arrays. This
  could reduce first-query traffic while keeping the query planner format-agnostic,
  but retains PyTables layout knowledge in a source adapter.
- Import of lighter index kinds, dirty-index recovery, and indexed-prefix plus
  unindexed-tail execution.
- Additional field types and geometry/compressed-copy optimizations beyond the
  validated initial scope.

## References and implementation entry points

- [OPSI paper](https://blosc.org/docs/OPSI-indexes.pdf), also available from
  [PyTables](https://www.pytables.org/docs/OPSI-indexes.pdf).
- `src/blosc2/hdf5_source.py`: metadata, transport, and HDF5 chunk decoding.
- `src/blosc2/remote_store.py`: discovery and shared remote ownership.
- `src/blosc2/remote_ctable.py`, `ctable_storage.py`, and `ctable_remote_read.py`:
  table construction, storage assumptions, and batched reads.
- `src/blosc2/indexing.py`: native OPSI sidecars, navigation, and query planning.
- `/Users/faltet/blosc/PyTables/tables/index.py` and `idxutils.py`: source index
  layout, last-slice handling, position encoding, and reduction rules.
