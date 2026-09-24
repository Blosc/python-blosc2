# Remote HDF5 Cold-Open Optimizations and Explicit Index Sidecars

Status: implemented and verified on 2026-09-22.

## Objective

Reduce first-open latency for remote HDF5 and PyTables sources while preserving
bounded memory use and on-demand payload reads. The motivating cold open is a
1.49 MiB indexed PyTables file on Backblaze B2:

| Operation | Time | Data requests | Bytes |
| --- | ---: | ---: | ---: |
| Open indexed table | 10.38 s | 51 | 38 KiB |
| Render its head/tail | 1.18 s | 2 | 255 KiB |
| Load/convert its PyTables index | about 10 s | 392 | 554 KiB |
| Fetch the complete HDF5 file | 1.61 s | 1 | 1.49 MiB |

The current scanner opens the remote object with `block_size=1` and
`cache_type="none"`, then asks h5py to walk every node and enumerate every
allocated chunk. HDF5 metadata is pointer-rich, so this translates small,
sequential h5py reads into many network round trips. PyTables makes the effect
larger by adding hidden `_i_<table>` groups and their chunked index datasets.

Implement the following four improvements together:

1. Fetch and retain small remote HDF5 files in one request.
2. Discover an explicitly selected dataset directly and defer its PyTables index.
3. Build allocated-chunk maps only when a dataset needs direct reads.
4. Complete the existing explicit `hdf5_index=` sidecar workflow, including
   RemoteCTable/RemoteStore support and public documentation.

This is an HDF5 optimization. B2Z already uses its ZIP tail and bounded member
prefixes; Zarr has a different metadata-object/listing problem and is outside
this plan.

## Constraints and invariants

- Keep remote sources immutable. Replacing an HDF5 object at the same URL still
  requires cache invalidation/refresh or publication at a new URL.
- Preserve exact `Traffic` accounting at the transport boundary. A one-request
  prefetch is one request carrying the complete object, rather than dozens of
  logical h5py reads against an in-memory buffer.
- Keep the memory fast path bounded by one internal constant. Do not add a public
  tuning option until measurements show that one is needed.
- A selected dataset must retain the same public type dispatch: arrays produce
  `RemoteArray`, PyTables tables produce `RemoteCTable`, and group roots produce
  `RemoteStore`.
- Root/group discovery must still provide a complete hierarchy. Dataset-targeted
  opens may remain partial and must not claim completeness.
- A missing PyTables index, unsupported index kind, dirty index, or malformed
  sidecar must retain the existing safe fallback behavior.
- Do not change the source HDF5 file. Native indexes and cache manifests remain
  Blosc2 metadata external to the HDF5 object.

## Phase 1: retained small-file bootstrap

Add a private small-object threshold in `src/blosc2/hdf5_source.py`, initially
8 MiB. During an uncached remote scan:

1. Obtain the source size from the existing filesystem object.
2. If the size is at or below the threshold, fetch the object once with
   `cat_file()`, charge that transfer once, and open h5py over `io.BytesIO`.
3. Retain the immutable bytes on the shared `RemoteDiscovery` owner for the
   lifetime of that source session.
4. Let every `HDF5NDSource` created by that owner slice compressed chunk ranges
   from the retained bytes. Its h5py fallback reader must use the same bytes.

The retained object is a source-level read buffer, not cache payload. It is
bounded independently of `max_cache_bytes`, just like existing B2Z opening
buffers. A DISK cache persists the generated HDF5 index and ordinary converted
chunks; it need not duplicate the complete source object. A warm reopen that
already has discovery metadata must not download the full HDF5 file merely
because it is small.

Files above the threshold keep exact range reads initially. Do not introduce an
unbounded fsspec block cache as part of this phase. Targeted discovery and lazy
allocation maps below reduce their request count without speculative payload
downloads.

Failure handling remains simple: a short or failed full-object response aborts
the open and releases the owned filesystem session. Local files and explicit
indexes do not enter the bootstrap path.

## Phase 2: targeted dataset discovery

Extend the internal scan path with an optional normalized dataset scope. When
the caller supplies `dataset=` or `::dataset`, open that HDF5 object directly
instead of calling `h5file.visititems()` across the container.

For an ordinary array, record only the selected dataset and the ancestor groups
needed to represent its path. For a PyTables table, inspect the selected compound
dataset's `CLASS`, dtype and attributes and build its existing CTable schema.
Do not inspect `_i_<table>` during this initial open.

Update `_open_remote_hdf5()` so it performs one targeted discovery and uses the
result for type dispatch. Avoid the current pattern in which a temporary
`RemoteArray` performs a full-container scan before handing the same index to a
`RemoteStore`. The partial discovery object must be shared by the returned leaf,
including its filesystem, retained small-file bytes, traffic counters and cache
owner.

Root or group opens still need complete hierarchy discovery and may use
`visititems()`. Keep these two contracts explicit:

- **Targeted index:** sufficient for one selected dataset/table and its deferred
  PyTables index metadata.
- **Container index:** sufficient for hierarchy listing and arbitrary leaf
  resolution.

When `RemoteTableStorage.load_index_catalog()` is first called for a PyTables
table whose index metadata has not been discovered, inspect only the predictable
`_i_<table>/<column>` paths for that table. Check the four required leaves
(`sorted`, `indices`, `sortedLR`, `indicesLR`) and the existing `DIRTY`, row-count,
dtype and tail invariants. Merge valid metadata into the shared discovery index
under its owner lock and persist the updated manifest when a DISK cache exists.
Repeated catalog access and warm reopens must not repeat this metadata scan.

This keeps `print(table)` independent of PyTables indexes. `table.indexes`,
`table.info`, or an index-planned query may pay the deferred index cost.

## Phase 3: lazy allocated-chunk maps

Separate cheap dataset metadata from the allocated-chunk table currently built
inside `_dataset_metadata()`. Shape, dtype, chunk geometry, fill value,
attributes and filter descriptions are cheap enough for discovery. The loop over
`get_num_chunks()` / `get_chunk_info()` is deferred unless the dataset is the
selected read target.

Introduce HDF5 index format version 2 with an unambiguous allocation state:

- `allocated: null` means the allocation map has not been scanned.
- `allocated: []` means it was scanned and the dataset has no allocated chunks.
- A list retains the current validated records.
- A top-level scope/completeness field distinguishes a targeted index from a
  complete container index.

Continue accepting version-1 indexes, where `allocated` is always a list.
Do not rewrite user-supplied v1 dictionaries or JSON files on disk. New public
full-index generation should produce a complete v2 index by default; internal
targeted discovery and container browsing may leave non-selected maps deferred.

Add one shared `ensure_hdf5_allocations(path)` operation on the discovery owner.
It opens only the named dataset, builds and validates its records, updates the
in-memory index, and persists the manifest when appropriate. Guard it with the
existing owner lock so concurrent first reads do the work once. A direct
`HDF5NDSource` without a discovery owner uses the same targeted helper locally.

Call this operation immediately before constructing `_chunk_records` for a
direct-range source. Do not build maps for unsupported filter pipelines, because
they use the retained h5py fallback. PyTables index discovery records its four
leaf datasets cheaply; their maps are populated only when the index is actually
converted or queried.

Validation must reject malformed allocation states, invalid byte ranges and a
targeted sidecar used outside its declared scope. Existing sparse-fill behavior
must remain unchanged after a deferred map is installed.

## Phase 4: complete explicit `hdf5_index=` sidecars

The native index is a Blosc2 JSON-compatible catalog, not an HDF5, h5py or
fsspec standard. Keep sidecars explicit: do not probe an adjacent filename and
do not establish a mandatory `.b2index.json` naming convention.

### Public API

- Export `scan_hdf5_index` and `validate_hdf5_index` from `blosc2` so users do
  not have to import an internal module to create a supported sidecar.
- Keep `hdf5_index=` accepting a dictionary, local path, or remote fsspec URL.
- Centralize dictionary/path/URL loading and validation so `HDF5NDSource`,
  `RemoteArray`, `RemoteStore`, `RemoteCTable`, and `blosc2.open()` behave alike
  and fetch a remote sidecar only once.
- Add public `hdf5_index=` parameters to `RemoteStore` and `RemoteCTable` and
  pass them through `blosc2.open()`.
- Remove the dispatch shortcut that forces an explicit indexed PyTables dataset
  into `RemoteArray`. Inspect the supplied catalog and return `RemoteCTable` or
  `RemoteStore` where its metadata requires it.
- Use the source's `storage_options` for the sidecar URL. Separate credentials
  for source and sidecar are outside this plan.

The public creation workflow should be:

```python
import json

import blosc2

source = "https://example.com/readings.h5"
index = blosc2.scan_hdf5_index(source)
with open("readings.h5.b2index.json", "w") as file:
    json.dump(index, file)

table = blosc2.open(
    source + "::readings",
    hdf5_index="https://example.com/readings.h5.b2index.json",
)
```

The example filename is a user convention only. Document that generation scans
the source once, that the recorded source URL must exactly match the opened HDF5
URL, and that replacing the source requires regenerating the sidecar. Supplying
the sidecar skips discovery; it does not alter or embed data in the HDF5 file.

### Documentation

Update:

- `doc/guides/remote_arrays.md` with generation, serialization and remote URL
  examples, the immutable-source contract, scope semantics and table support.
- `doc/reference/remotearray.rst`, `doc/reference/remotestore.rst`,
  `doc/reference/remotectable.rst`, and `doc/reference/hdf5ndsource.rst` with the
  accepted input forms and ownership/lifetime behavior.
- The `blosc2.open`, `RemoteArray`, `RemoteStore`, `RemoteCTable`,
  `scan_hdf5_index`, and `validate_hdf5_index` API documentation.

State clearly that automatic adjacent-sidecar discovery is intentionally absent,
because a failed probe would add latency to every source without a sidecar.

## Implementation order

1. Extract shared HDF5 index loading/validation and add v1/v2 compatibility.
2. Add targeted metadata and allocation-map helpers with local/memory filesystem
   tests before changing dispatch.
3. Route selected datasets through targeted discovery and lazy PyTables index
   discovery.
4. Add the retained small-file bootstrap and share it with every leaf source.
5. Expose explicit sidecars through tables/stores and fix type dispatch.
6. Export and document the public index-generation workflow.
7. Measure the motivating HTTP case and retain the results in this plan.

This order keeps each behavioral change testable. The small-file fast path comes
after targeted helpers so its retained buffer plugs into one read abstraction
instead of creating a second scanner.

## Verification results

The motivating indexed Backblaze table now opens with one request for the
1,489,409-byte source. On the same URL, a cold run measured 2.386 s to open,
effectively 0 s for ``t.info``, and 0.019 s for ``str(t)``. The affected test
set passes with 618 tests and 5 skips; Ruff and the normal Sphinx build also
pass. Sphinx ``-W`` remains blocked by pre-existing repository-wide warnings.

## Tests

Use the `blosc2` conda environment. Extend the existing HDF5, fsspec,
RemoteStore and PyTables interoperability tests rather than adding a separate
framework.

### Small-file bootstrap

- Use the deterministic ranged HTTP server to verify that a file below the
  threshold crosses the wire once, is charged once, and serves discovery,
  head/tail reads and PyTables index conversion from retained bytes.
- Verify exact-threshold behavior, short responses, cleanup after failures, and
  that a larger file is not downloaded wholesale.
- Verify that a warm DISK reopen uses its persisted index without repeating the
  full-file bootstrap.

### Targeted discovery and lazy allocation

- Create an HDF5 container with unrelated groups, many sibling datasets and a
  PyTables index. Opening one selected dataset must omit sibling and hidden-index
  metadata work.
- Confirm array/table/group dispatch, attributes, ancestor paths, unsupported
  sibling isolation and missing-dataset errors.
- Confirm `print(table)` does not discover or load `_i_<table>`, while
  `table.indexes` discovers it once and a warm reopen performs no new scan.
- Verify allocation maps are built for only the first accessed leaf, are shared
  by sibling handles, survive DISK reopen, and preserve sparse fill chunks.
- Exercise concurrent first access to the same map and failure cleanup.

### Explicit sidecars

- Cover dictionaries, local JSON paths, `memory://` URLs and the HTTP test
  server for arrays, PyTables tables and group stores.
- Monkeypatch `scan_hdf5_index` to prove a valid supplied sidecar performs no
  source discovery.
- Reject URL mismatches, out-of-scope targeted indexes, malformed JSON, unknown
  versions, bad ranges and legacy reference maps with actionable errors.
- Verify version-1 compatibility and version-2 round trips through JSON, array
  carriers, RemoteStore manifests and exported remote-reference artifacts.
- Verify a remote sidecar is fetched once and shares the source filesystem
  session and traffic accounting where applicable.

### Regression and quality checks

- Run focused `tests/test_hdf5_source.py`, `tests/test_fsspec.py`,
  `tests/test_remote_array.py`, `tests/test_remote_store.py`, and
  `tests/ctable/test_remote_pytables_interop.py` coverage, followed by the
  default suite.
- Run Ruff on changed Python files and build the documentation with warnings as
  errors.
- No C/Cython change or new dependency is expected.

## Acceptance criteria

- A cold open plus `print(table)` for the motivating 1.49 MiB file requires one
  source data request on the small-file path; incidental identity requests are
  reported separately.
- `table.indexes` against that retained file adds no network request, while
  producing the same OPSI results as today.
- A selected dataset in a large multi-dataset HDF5 file does not traverse
  unrelated siblings or build their allocation maps.
- Full RemoteStore hierarchy browsing remains correct, with leaf allocation maps
  populated only on first read.
- An explicit remote JSON index opens arrays, PyTables tables and HDF5 stores
  without scanning the source and is fully documented as a public workflow.
- Warm DISK behavior remains zero-traffic for persisted discovery metadata and
  cached payloads, subject to the existing cache limits.

## Non-goals

- Automatic sidecar filename probing or publication/upload APIs.
- Mutable remote HDF5/SWMR semantics.
- A general HDF5 metadata server or Kerchunk/reference-map compatibility.
- Changing B2Z or Zarr discovery.
- Coalescing large PyTables index payload ranges beyond what the retained
  small-file path provides. That remains a follow-up if large indexed files are
  still request-bound after these changes.
