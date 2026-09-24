# Consistent caching for local and remote sources

## Objective and agreed contract

An explicit `cache_dir=` requests a real persistent cache, whether the source is
local or remote and whether its format is native Blosc2, HDF5, or Zarr. A local
source may live on slower storage than the cache. The library must honor the
request without deciding whether caching is worthwhile on the user's machine.

`cache_path=` follows the same caching and validity model. Its distinction is
placement: it names the exact cache file for one array. `cache_dir=` names a
parent directory under which source-specific array, table, or store caches are
managed. Do not introduce a single-file cache representation for tables/stores.

Implemented using the existing array carriers and store caches. Local runtime
descriptors remain separate from portable remote references. The behavior below
is the agreed contract; the next section records the original starting point.

## Original behavior to replace

Verified against the current working tree:

| Local source | Current behavior with `cache_dir=` |
| --- | --- |
| Selected PyTables table | Persistent chunks and imported index sidecars; file-stat invalidation |
| Ordinary HDF5 dataset | Rejected by remote URL validation |
| Standalone or nested Zarr array | Rejected by remote URL validation |
| Native `.b2nd` | Returns the native array and silently ignores caching |
| Native `.b2z` root | Uncommitted patch accepts and ignores caching |
| Array selected inside `.b2z` | Rejected by remote URL validation |

Replace the uncommitted `.b2z` no-op code, its documentation, and its test that
expects no cache directory. Remove local PyTables automatic file-stat
invalidation in favor of the immutable-source contract below.

## Public behavior

### Cache placement and returned objects

| Selected object | `cache_dir=` | `cache_path=` | Cached handle |
| --- | --- | --- | --- |
| Standalone `.b2nd` array | Supported | Supported | `RemoteArray` |
| Array inside `.b2z` | Supported | Supported | `RemoteArray` |
| HDF5 dataset | Supported | Supported | `RemoteArray` |
| Zarr array, standalone or nested | Supported | Supported | `RemoteArray` |
| Native B2Z or PyTables table | Supported | Reject; direct users to `cache_dir` | `RemoteCTable` |
| Supported container group | Supported | Reject; direct users to `cache_dir` | `RemoteStore` |

The existing wrapper names also apply to local sources. Document that choosing
caching can change a native `NDArray`/`CTable` into its read-only cached wrapper.
Without cache options, preserve current dispatch and native writable access.
Discover a selected node's kind before applying array-only cache options;
`cache_path` must not accidentally turn a PyTables table into a record array.

Both placement options imply `CachePolicy.DISK` when policy is omitted. They are
mutually exclusive. An explicit incompatible policy raises a clear error.
Retain the existing 256 MiB default compressed-payload budget and
`max_cache_bytes=None` behavior. Table/store leaves share their owner's budget.
Metadata and derived index sidecars retain their existing budget treatment;
do not describe the payload limit as a bound on total disk consumption.

Omitted or `None` `lazy` selects on-demand access for the new local caching
paths. Opening may create cache metadata; reads populate missing payload.
Reuse compressed native chunks where possible. Convert HDF5/Zarr data through
existing source adapters. Cache indexes using the existing table machinery.
Do not add decoded-data or query-result caching.

Initially support local cached opens in `mode="r"`. Reject write modes and
incompatible `mmap_mode`, embedded-frame offsets, and explicit `lazy=False`
with actionable messages. Preserve existing remote eager-localization behavior
where already supported; do not silently reinterpret `lazy=False`.

Keep the deprecated `cache_storage` alias on the same normalization path, with
its existing warning and mutual-exclusion checks.

### Source validity

Default to `assume_immutable=True` for local cached sources, as for remote
sources. Reopening an unchanged source identity reuses the cache; it does not
check modification times, hash payload, scan directories, or revalidate cached
chunks against the source.

Changing a source at the same identity violates that assumption. Cached and
uncached reads can otherwise mix versions. Users must close other handles and
explicitly refresh/rebuild or clear the cache before reading a changed source.
Do not promise offline access merely because some chunks have been cached.

Refresh/rebuild must discard stale source metadata, payload, and imported index
sidecars together, then publish a fresh generation. Audit the existing array,
table, and store refresh APIs and document the supported procedure for each;
reuse them rather than adding a competing invalidation API.

Reject `assume_immutable=False` for new local paths until change detection is
implemented. Preserve remote cases that already support it. Removing local
PyTables stat invalidation must not remove structural, geometry, descriptor,
cache-identity, or corruption checks.

### Identity and persistence

Local runtime cache identities must include a normalized absolute source path,
the selected node, source format, and relevant representation settings. Resolve
relative paths at open time so another working directory cannot reuse an
unrelated source's cache. Test equivalent relative/absolute paths and `file://`
inputs, and distinguish local sources from remote URLs in a shared cache parent.
Do not include file timestamps or content digests in the immutable identity.

An explicit `cache_path` already associated with another source or incompatible
representation must fail rather than overwrite or reuse that cache. Reject a
cache destination that would overwrite its source.

Treat local runtime cache descriptors separately from portable remote
references. Do not globally relax `validate_persistable_url()`: a loaded remote
reference must not acquire arbitrary local-file access. Ensure supported local
cache reopening retains an unambiguous absolute source identity. Keep portable
remote export restrictions unless local-reference export is explicitly designed.

## Implementation sequence

1. **Unify dispatch and option validation in `src/blosc2/schunk.py`.**
   Detect an explicit cache request before the native local fast path. Route
   local arrays/tables/groups to the existing cached readers. Remove the B2Z
   no-op and ensure `cache_path` follows the same node discovery as `cache_dir`.
   Keep ordinary uncached native opens on their current path.

2. **Generalize local source attachment.**
   Reuse `remote_array.py`, `remote_store.py`, and the existing source adapters.
   Replace the HDF5-table-only local exception with narrowly scoped support for
   authorized local runtime sources. Keep direct h5py reads for local HDF5 and
   preserve their lack of a required fsspec dependency. Support native contiguous
   arrays first through the existing chunk reader; explicitly check sparse
   `.b2nd` layouts and reject unsupported layouts clearly rather than ignoring
   caching. Use existing B2Z and Zarr discovery/readers for those formats.

3. **Reuse persistent cache storage.**
   Attach local arrays to the existing carrier/proxy cache and local tables and
   groups to `StoreDiskCache`. Preserve resource ownership, cache budgets,
   incomplete-cache recovery, and source separation. Retain HDF5 converted
   indexes and native B2Z index access through the current table reader.
   Avoid a second local-only cache backend or full-file copies on every open.

4. **Apply the immutable-source rule end to end.**
   Remove `_local_hdf5_stat`, `_reuse_local_hdf5_manifest`, and their automatic
   invalidation behavior. Treat old local stat metadata as obsolete rather than
   requiring it for reuse. Audit adapter stamps, small-file bootstrap caches,
   and cache reopening for implicit checks that would undermine this contract.
   Verify explicit refresh rebuilds all relevant metadata and payload together.

5. **Document and verify the public contract.**
   Update `open()` and wrapper documentation plus the remote array/table guides.
   Explain local cache placement, wrapper return types, the array-only meaning
   of `cache_path`, immutable sources, and refresh procedures. Replace statements
   that local native caches are ignored or PyTables caches automatically detect
   file changes.

## Sharing boundary

Reuse existing ownership and locking rules. An ordinary cache is not implicitly
safe for concurrent processes. `shared_cache=True` remains an explicit opt-in.
This implementation must not silently ignore it for local sources. Keep the
current clear rejection until local shared attachment is implemented and tested
using the existing sparse shared backend. Do not extend `cache_path` to shared
table/store caches as part of this work.

## Validation and acceptance

Use the `blosc2` conda environment for all checks. Add focused tests in the
existing HDF5, Zarr, B2Z, remote array/store, and CTable suites:

- Each supported local format creates real reusable payload after a read;
  directory existence alone is insufficient evidence.
- A fresh process reuses cached payload and PyTables index sidecars without
  rereading/reconverting those source payloads. Metadata access may still occur.
- Array `cache_dir` and `cache_path` produce equivalent values and cache policy.
  Cover standalone arrays and nested selectors using `path=` and `::`.
- Native uncached opens retain their existing types and behavior. Cached native
  table queries preserve null semantics and index correctness.
- Mutating a source does not automatically invalidate cached data under the
  immutable contract. The documented explicit rebuild procedure observes the
  new data and indexes without mixing generations.
- Test path identity, mismatched explicit cache files, invalid option
  combinations, unsupported mutable access, and table/group `cache_path` errors.
- Preserve local HDF5 operation without fsspec, and run remote cache regression
  tests to catch accidental changes to remote descriptors or dispatch.

Run the relevant existing suites, Ruff, and diff checks. The moto S3 HDF5 tests
previously stalled in this environment; report any exclusion explicitly.
Benchmark the user's large B2Z and indexed HDF5 examples in fresh processes with
cold and warm caches. Report timing without requiring a speedup over uncached
native reads: honoring explicit cache placement is the acceptance criterion.
