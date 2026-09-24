# Persist Small Remote HDF5 Source Objects

Status: implemented on 2026-09-22; measured results below.

## Objective

When a remote HDF5 object is at most 8 MiB, discovery already downloads the
complete object in one request and retains it in memory. With ``cache_dir=`` the
generated HDF5 index and converted Blosc2 chunks survive process exit, but the
downloaded source bytes do not. A later process can therefore restore metadata
without network traffic and still need remote range requests for table previews
or uncached dataset chunks.

Persist that already-downloaded small HDF5 object once per source URL so all
dataset-scoped caches and later processes can reuse it.

The motivating sequence currently behaves as follows:

| Operation | Requests | Bytes |
| --- | ---: | ---: |
| Cold ``print(t.info)`` | 1 | complete 886 KiB HDF5 file |
| Next-process ``print(t)`` | 2 | about 255 KiB of head/tail data |

After this change, the second operation should issue no source requests.

## Scope

Implement this for remote HDF5 sources using ``CachePolicy.DISK`` with an
explicit ``cache_dir=``. Cover ``RemoteArray``,
``RemoteCTable``, ``RemoteStore``, and ``blosc2.open()``.

Use that explicit directory as the shared source-cache root. Store-owned leaves
inherit their owner's root. A standalone ``cache_path=`` or a restored reference
artifact with only an internal runtime cache has no explicit shared root: keep
its current behavior in this first version. Do not infer a root from a carrier's
parent directory or persist machine-local source-cache paths in portable
artifacts. Restored references may participate when their supported opening API
explicitly supplies a runtime ``cache_dir``; otherwise defer this integration.

Do not change MEMORY or NONE cache behavior. Do not fetch a complete object when
an explicit ``hdf5_index=`` avoids source discovery, and do not expand the
existing 8 MiB threshold.

B2Z may reuse the source-object mechanism later if benchmarks justify complete
prefetch for small archives. Zarr remains outside this design because it is a
multi-object store rather than one source object.

## Cache model

Keep three cache categories separate:

1. **Source object**: the immutable original HDF5 bytes, bounded by the existing
   8 MiB bootstrap threshold and shared across dataset scopes.
2. **Discovery metadata**: native HDF5 indexes, hierarchy manifests, and
   PyTables index descriptions.
3. **Converted payload**: cached Blosc2 chunks and generated PyTables index
   sidecars, governed by ``max_cache_bytes``.

The source object is not a converted payload and must not be stored as a
``.b2nd`` leaf. It is cache infrastructure like the discovery manifest. It does
not count against ``max_cache_bytes``. The 8 MiB ceiling applies to each source
object, not the total cache directory: many distinct sources can accumulate
unbounded disk usage. Document this accounting policy and manual cleanup
explicitly; ``max_cache_bytes`` is not a total disk-space guarantee.

## Source-level identity and layout

Reuse ``fsspec_cache_path()`` so identity is based on the normalized base URL
and the non-reversible ``storage_options`` fingerprint. Do not include the HDF5
dataset scope.

For example:

```text
cache_dir/
├── hdf5-sources/
│   └── pt-readings.h5--<source-hash>/
│       ├── pt-readings.h5.hdf5-source
│       ├── pt-readings.h5.hdf5-source.json
│       └── pt-readings.h5.hdf5-source.lock
└── ... existing array and dataset-scoped store caches ...
```

The dedicated namespace avoids collisions with existing dataset-scoped cache
directories; their layout does not change.

The exact suffix is private. The important invariant is that these calls resolve
to the same source-object path:

```python
blosc2.open(url + "::readings", cache_dir=cache_dir)
blosc2.open(url + "::stations", cache_dir=cache_dir)
blosc2.open(url, cache_dir=cache_dir)
```

Calls using another URL, storage-options fingerprint, or cache directory remain
isolated. Repeating an identical call reuses the same file.

Store a small JSON marker beside the source object:

```json
{
  "version": 1,
  "size": 906738,
  "token": "...same SHA-256 digest...",
  "sha256": "..."
}
```

The URL and credentials fingerprint are already part of the parent-directory
identity and need not be duplicated in the marker.

## Publication and validation

Use the existing ``atomic_write()`` helper for both files:

1. Compute the digest from the downloaded bytes and attach it to generated
   discovery metadata before that metadata is persisted.
2. Initialize the normal carrier/store cache successfully.
3. Write the source object atomically, then write its marker atomically.

The source artifact is optional, so carriers/manifests may be published before
it. A crash between those steps leaves an ordinary metadata/chunk cache that
can still use remote reads. Use this ordering consistently for arrays and stores.

Concurrent dataset opens may both download the same immutable source and race to
publish it. Each replacement is atomic, but the two files are not one atomic
transaction. Readers must treat missing or mismatched pairs as cache misses.
Publication reuses the existing platform file-lock helper for a short
source-specific lock around the version check and replacement. It holds no lock
during downloads, and acquires no store/array locks while holding this lock.
An opener whose expected version has changed fails with a retry message instead
of overwriting a newer refresh. Readers still validate the two-file pair.

On reuse:

1. Read and validate the marker schema.
2. Validate that the marker size is a positive integer at most 8 MiB, and check
   the opened file's size before allocating its contents. Bound the read as well
   so a concurrent file change cannot bypass the limit.
3. Require the actual size to match the marker, and the index ``size`` when an
   index is available. A new dataset scope may have no index yet.
4. Verify SHA-256 and check index/source compatibility as specified below.
5. Pass the verified bytes as ``_blob``/``_hdf5_blob``.

An absent, truncated, corrupt, or unrecognized source cache is disposable. Ignore
it and retain the existing metadata-index plus remote-range fallback; do not make
a valid discovery cache unusable. A later complete bootstrap may replace the bad
source artifact.

A valid artifact whose digest disagrees with a generated index is a different
case: it signals inconsistent source versions, not simple artifact corruption.
Never use that index's byte offsets against either the replacement blob or
remote ranges without rebuilding discovery and invalidating its derived caches.

Do not count reads from the local source-object cache in ``Traffic``. Traffic
continues to represent remote transport only.

## Integration

### Shared helpers

Add minimal private helpers near the existing HDF5 cache/bootstrap code:

- Resolve the source-object and marker paths from base URL, ``cache_dir``, and
  ``storage_options``.
- Load and validate a cached object, returning ``bytes`` or ``None``.
- Atomically publish a newly fetched object.

Extend the internal ``scan_hdf5_index()`` path to accept an already validated
``_blob``. In that case derive the source size from ``len(_blob)`` and open h5py
over those bytes without calling ``filesystem.info()`` or ``cat_file()``. This
is required when a new dataset scope has no cached index yet but the shared
source object is already present.

Reuse ``fsspec_cache_path()``, ``atomic_write()``, ``hashlib``, and ``json``.
Do not introduce a general cache class or a new dependency.

### RemoteArray path

``RemoteArray`` already computes a source-level shared HDF5 index path before
opening the source. Extend that setup for ``CachePolicy.DISK``:

1. Try the cached source object before constructing ``HDF5NDSource``, including
   when an explicit index is supplied. Lookup alone must not fetch the source.
2. Pass valid bytes into ``HDF5NDSource`` and its index scanner so discovery and
   reads use local bytes without contacting the remote filesystem.
3. If scanning instead downloads a small object, publish ``src._blob`` after the
   disk carrier has been initialized successfully.
4. Persist the source digest with generated indexes and enforce compatibility
   before reusing converted chunks.

The source-object lookup is independent of the selected dataset, whereas the
carrier remains dataset-specific.

### RemoteStore and RemoteCTable path

``StoreDiskCache`` is constructed before ``RemoteDiscovery``, but currently the
owner receives ``owner.disk`` only after discovery. Avoid a broad lifecycle
refactor:

1. Resolve/load the source object from ``cache_dir`` before creating
   ``RemoteDiscovery`` and pass it as ``_hdf5_blob``.
2. On a cold discovery, attach the source digest to generated metadata. After
   normal cache initialization and manifest publication succeed, publish the
   retained ``owner.hdf5_blob`` as an optional source artifact.
3. For the table dispatch path, reuse the blob transferred from the temporary
   targeted ``RemoteArray`` instead of downloading or writing it twice.

This is a post-discovery persistence step, not a new owner responsibility.

### Explicit indexes

If ``hdf5_index=`` is supplied and no complete source bytes are fetched, do not
download the HDF5 file merely to populate the source cache. A previously cached
source object may still be used after it passes size and integrity validation.
If the explicit index records a source digest, require it to match; reject a
mismatch with an actionable error asking for a regenerated sidecar. Do not
silently replace a user-supplied index. Legacy explicit sidecars without a
digest retain their existing immutable-URL trust contract; size alone cannot
prove they describe the same source version.

## Source versions and refresh

Record an optional ``source_sha256`` in native indexes generated from complete
source bytes, retaining compatibility with existing indexes without that field.
The marker's checksum establishes file integrity; matching the index's digest
establishes that its offsets describe those bytes. Validate the new field when
present and preserve it through carriers, manifests, and JSON round trips.

``RemoteStore.refresh()`` and ``RemoteCTable.refresh()`` already exist. Integrate
with their current refresh lifecycle in this change:

- Explicit refresh bypasses the retained source artifact and performs fresh
  discovery. Keep the old live generation usable if preparing refresh fails.
- On successful refresh, publish the replacement metadata and source artifact,
  invalidating derived column chunks and converted PyTables indexes through the
  existing generation replacement mechanism.
- Other dataset scopes compare their recorded digest with the shared artifact
  on their next open. On mismatch, rebuild their scoped metadata from the new
  bytes and discard their derived caches before serving data. Existing live
  handles can retain their old immutable snapshot until refreshed or reopened.
- A pre-existing generated cache without a digest cannot establish compatibility
  with a newly shared source artifact. Rebuild its metadata and invalidate its
  derived payload once before binding it to the artifact. Keep older caches
  working as before when no source artifact is available.

If refresh produces a file above the threshold, publish a marker with its new
size, a random version token, and null SHA-256, then remove the previous small
artifact. Generated indexes record this token as ``source_cache_version``.
Other scopes compare tokens before reusing metadata/payload; a missing blob
alone is not evidence that old metadata remains valid.

Concurrent refresh and opens must either observe compatible metadata and bytes
or retry/fail clearly. Atomic writes alone do not establish this compatibility.
New URLs remain the preferred way to publish a new immutable source version.

## Cache lifecycle

The source object is shared across dataset-specific cache entries, so it cannot
belong to one dataset generation or be removed by
``discard_old_generations()``. It remains under the source-level directory until
the user removes that source cache or the entire ``cache_dir``.

This matches the current immutable-URL model. Automatic source-object eviction,
TTL handling, and a global cache-size budget are outside this change. Add them
only if accumulated small sources become a demonstrated problem.

The explicit refresh behavior above is in scope even though ordinary reads
assume immutable URLs. Automatic remote freshness checks remain outside scope.

## Tests

Use the ``blosc2`` conda environment and the existing ranged HTTP test server.

1. Cold open of a sub-threshold HDF5 source performs one complete GET and writes
   the source object plus marker.
2. Run an actual subprocess regression: process 1 formats ``t.info`` and exits;
   process 2 formats ``t`` using the same ``cache_dir``. The parent HTTP server
   must observe zero requests from process 2, including HEAD requests. Also
   verify array slicing and PyTables index discovery with new owners. Use the
   active blosc2 environment's interpreter for subprocesses.
3. Different dataset scopes reuse the same source-object path and do not create
   duplicate raw files.
4. Dataset-specific carriers/manifests remain distinct and contain the correct
   scoped metadata.
5. Files above 8 MiB are not stored as source objects and retain range reads.
6. Explicit ``hdf5_index=`` does not trigger a complete source download.
7. Missing source artifact falls back to remote ranges while retained metadata
   remains usable.
8. Truncated data, wrong size, oversized files/markers, malformed marker, unknown
   marker version, and SHA-256 corruption all fall back safely. Oversized cached
   files must be rejected before a whole-file allocation.
9. Concurrent publication of identical bytes leaves a valid file/marker pair.
   Missing or mismatched pairs during publication produce safe cache misses.
10. ``Traffic`` counts the initial remote transfer once and counts no local
    source-cache reads.
11. Existing DISK cache reopening, cache limits, source isolation, and cleanup
    tests remain unchanged.
12. Replace an HDF5 source with different contents/layout but exactly the same
    size, then explicitly refresh. Verify correct rows, index conversion, and
    invalidation of another scope's cached metadata and chunks on reopen.
13. Cover refresh failure, interrupted publication, concurrent refresh/open,
    and a refresh that crosses above the 8 MiB threshold.
14. Cover one-time migration of generated indexes without digests and explicit
    sidecars with matching, mismatched, and absent digests.
15. Verify cache_path-only and artifact-only opens preserve their documented
    behavior, and that eligible store-owned leaves share the explicit root.

Run the focused HDF5/fsspec/remote-table/store suites, Ruff, and the normal
Sphinx build, followed by the default test suite.

## Acceptance benchmark

With a new ``cache_dir``:

```text
process 1: print(t.info)  -> one complete-file request
process 2: print(t)       -> zero requests
process 3: table slice    -> zero requests
```

Opening another dataset from the same HDF5 URL must also use zero source
requests once its targeted metadata can be derived from the shared cached file.
Record wall time, request count, and transferred bytes for the motivating
Backblaze files before marking the plan implemented.

## Deliberate non-goals

- No public threshold or source-cache switch.
- No complete download for files above 8 MiB.
- No adjacent-sidecar probing.
- No B2Z implementation without measurements.
- No whole-store Zarr cache.
- No shared-cache garbage collector or new lock manager.

## Implementation results (2026-09-22)

Implemented for explicit DISK cache directories, reusing existing cache identity,
atomic writes, and platform file locking; no new dependencies or general cache
framework. Also fixed restored HDF5 table nodes to share their canonical index
metadata, so lazy PyTables index discovery remains visible after reopening.

Reproduce with:

```sh
conda run -n blosc2 python bench/remote_hdf5_source_cache.py --repeats 3
```

The baseline disables only source-object publication to reproduce the previous
metadata-only cache policy. Each sequence uses a fresh temporary cache and three
separate Python processes; modes alternate order. These are medians of three
sequences per mode against the public Backblaze files, without concurrent tests
or builds. Process times include interpreter startup/imports; operation times
measure opening and rendering only.

| File / phase | Baseline process | Persisted process | Baseline operation | Persisted operation | Requests before → after |
| --- | ---: | ---: | ---: | ---: | ---: |
| `pt-readings.h5`: cold info | 2.086 s | 2.082 s | 1.936 s | 1.930 s | 1 → 1 |
| next-process preview | 1.427 s | 0.283 s | 1.259 s | 0.138 s | 2 → 0 |
| warm preview | 0.277 s | 0.278 s | 0.129 s | 0.133 s | 0 → 0 |
| `pt-readings-idx.h5`: cold info | 2.321 s | 2.354 s | 2.171 s | 2.208 s | 1 → 1 |
| next-process preview | 1.518 s | 0.293 s | 1.357 s | 0.148 s | 2 → 0 |
| warm preview | 0.284 s | 0.286 s | 0.138 s | 0.141 s | 0 → 0 |

The next-process preview is **5.0–5.2× faster including startup**, or about
**9.2× faster for open/render**. Both files avoid 261,074 transferred bytes and
two source requests on that preview. Cold info still transfers the complete
906,738-byte or 1,489,409-byte file once; warm previews remain essentially
unchanged. Small cold/warm timing differences are not evidence of a speedup or
regression with only three network runs.

Validation: default suite **10,427 passed, 36 skipped**; Ruff checks and formatting
pass; Sphinx builds successfully with existing cross-reference warnings.
Regression coverage includes a second process with socket connections disabled,
shared dataset scopes, corruption and interrupted publication, explicit indexes,
refresh failure, stale publishers, same-size replacements, and growth above 8 MiB.
