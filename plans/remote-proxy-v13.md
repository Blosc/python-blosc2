# RemoteArray and RemoteStore v13: shared caching and public hierarchies

Status: implementation in progress, 2026-09-10. Steps 1–2 and measurement baseline
complete; steps 3–6 pending. RemoteStore currently supports NONE only.

## Implementation progress

- Step 1: renamed the class/module to `RemoteArray`/`remote_array`, the public
  export, maintained references, tests and reference page. Persisted object tags
  now use `remote_array`, including carrier dispatch and warm-cache validation.
  No compatibility alias was added. Existing Proxy tutorial filenames remain
  unchanged because they describe the original Proxy API.
- Validation in the `blosc2` conda environment: 497 focused tests passed across
  RemoteArray, Proxy, expressions, object serialization, B2Z/Zarr/HDF5/fsspec,
  browser model and Caterva2 blocks. Local HTTP/mock S3 tests initially hit
  sandbox socket restrictions; all affected tests passed outside that restriction.
  Ruff check and format check passed for all 21 changed Python files;
  `git diff --check` passed.
- Additional core/ndarray/schunk doctests: 72 passed, 2 skipped, 48 failed.
  Failures include unchanged examples with ambiguous array truth values and
  an incomplete `if` block. These are outside the rename milestone; the full
  default suite, marked network tests and Textual UI tests have not been run.
- Step 2: added public `RemoteStore` and `RemoteNode`, with immediate-child
  iteration, relative-path lookup, read-only attributes, unsupported-node
  diagnostics, source descriptors and shared discovery traffic. B2Z archives,
  HDF5 reference maps and Zarr stores/readers are reused across leaves and aliases.
  Enumeration constructs no leaf readers or payload caches. B2Z attachment checks
  archive/source identity; the existing sparse-attachment restriction remains.
- Discovery now lives in `remote_store.py`; `b2view/hierarchy.py` is a presentation
  adapter over that implementation, retaining the browser's existing current-leaf
  Proxy cache until step 5. Independent store/array handles own resource lifetime:
  closing a parent leaves children usable, and the last close or garbage collection
  closes owned archive/store wrappers. Fsspec still manages transport connection
  pools; store-owned transport exclusion/lifetime work remains with step 4.
- The step-2 API deliberately defaults to and supports `CachePolicy.NONE` only.
  MEMORY/DISK raise `NotImplementedError`; limits and disk locations are not yet
  constructor options. The public behavior below describes the completed v13
  target, including its eventual MEMORY default. Standalone RemoteArray caching
  and self-contained exports retain their existing behavior.
- Step-2 validation in `blosc2`: 437 focused array/source/serialization/browser
  tests passed, with 2 Zarr-specific cases skipped for other formats. Another
  23 Textual UI tests passed. Coverage includes Zarr v2/v3, one HDF5 translation,
  no-cache repeat reads, expressions, standalone exports and independent close/GC.
  Ruff check/format passed for all 8 affected Python files; whitespace checks passed.
  No new live-network benchmark or full default-suite run was performed for step 2.
- Next: implement aggregate MEMORY accounting and eviction (step 3), then DISK
  persistence, exclusion, manifest restoration and refresh (step 4). The final
  b2view migration to public handles and store-wide caching remains step 5.
- Measurement follow-up: added `bench/remote_array_traffic.py` with its report
  and raw JSONL results. It measures cold/warm requests, connections and response
  body bytes for B2Z, Zarr and HDF5 over S3 and HTTPS. Fixed HTTPS HDF5 discovery
  seekability in `scan_hdf5_refs` and added a regression test. This is a standalone
  RemoteArray baseline; it does not measure RemoteStore behavior.

## Measured remote baseline (2026-09-09)

Live Backblaze B2 measurements cover all three formats over S3 and direct HTTPS,
with three fresh-process trials per case (18 trials total). Each opens `d0/a3`
from `hierarchy.{b2z,zarr,h5}`, reads `[0, :100, :100]`, then repeats that slice
on the same handle. All sources have shape `(10, 1000, 1000)`, dtype `int32` and
chunks `(2, 500, 500)`; the returned slice is 40,000 bytes. The MEMORY allowance
is 64 MiB, and each handle retains 15,598 compressed cache bytes.

Cold totals include opening/discovery and the first slice. Times are medians;
variable request counts are ranges. Connections are newly established client
connections, separate from HTTP request attempts. Downloaded bytes count response
bodies, including metadata and error responses, but exclude HTTP headers and
TLS/TCP overhead.

| Format | Transport | Cold requests | New connections | Downloaded bytes | Cold time | Warm time |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `.b2z` | S3 | 5 | 1 | 40,817 | 1.683 s | 0.989 ms |
| `.b2z` | HTTPS | 5 | 1 | 40,817 | 1.419 s | 1.004 ms |
| `.zarr` | S3 | 8 | 5 | 13,433 | 1.918 s | 0.364 ms |
| `.zarr` | HTTPS | 4–5 | 3 | 12,875 | 0.904 s | 0.392 ms |
| `.h5` | S3 | 58 | 1 | 60,731 | 11.839 s | 0.595 ms |
| `.h5` | HTTPS | 58 | 1 | 60,731 | 11.829 s | 0.541 ms |

Every warm read required **zero requests, zero new connections and zero downloaded
bytes**. All values and warm-cache assertions passed. These are same-handle MEMORY
hits, not disk reopen or RemoteStore manifest measurements. Service/CDN and OS DNS
caches were not cleared. Latency outliers and the extra Zarr HTTPS GET remain in
the raw results; this small, regular fixture does not establish general throughput
or compression rankings.

Implications for the remaining implementation:

- HDF5 discovery alone costs 57 requests and 47,332 bytes, followed by one data
  request of 13,399 bytes. Reusing its reference map across leaves and disk reopens
  is the main opportunity demonstrated by this baseline. Validate that manifest
  restoration removes repeated translation, rather than merely retaining payloads.
- S3 and HTTPS have similar HDF5 costs here, but are not interchangeable performance
  baselines. Zarr's S3 route adds HEAD requests; keep both transports in subsequent
  comparisons and separate discovery from slice traffic.
- Built-in `array.traffic` is not a complete transport tally: standalone HDF5
  opening reports 392 bytes versus 47,332 received body bytes because its Kerchunk
  scan is outside that counter. HEAD requests and failed Zarr metadata probes are
  also absent. During shared-source extraction, include discovery in root accounting
  once and document counter semantics. Use transport instrumentation to verify
  manifest reuse rather than treating the existing counter as total HTTP traffic.

The benchmark exposed and fixed HTTPS HDF5 scanning: `block_size=0` selected a
non-seekable HTTP stream. `scan_hdf5_refs` now uses
`block_size=1, cache_type="none"`, preserving seekability without read-ahead.
The 155 existing focused fsspec/HDF5 tests and the new HTTP scan/warm-slice regression
test passed; Ruff and whitespace checks passed. Live network behavior was exercised
by the 18 benchmark trials; the marked network pytest suite, full default suite and
Textual UI suite remain unrun for this follow-up.

See the [full measurement report](../bench/remote_array_traffic.md),
[reproducible benchmark](../bench/remote_array_traffic.py), and
[raw trial results](../bench/remote_array_traffic.jsonl).

## Goal

Rename the unreleased `RemoteProxy` to `RemoteArray` and introduce a read-only
`RemoteStore` for remote B2Z, Zarr, and HDF5 hierarchies. Arrays selected from a
store share one cache budget. Persist discovery metadata so reopening a disk
cache avoids repeating archive discovery or HDF5 translation. Migrate b2view to
these public APIs and remove its internal hierarchy implementation.

Keep this a library and browser change. Caterva2 integration, customer quotas,
and concurrent processes sharing a writable disk cache are out of scope.

## Target public behavior

```python
store = blosc2.RemoteStore(
    "https://host/data.h5",
    cache_policy=blosc2.CachePolicy.MEMORY,
    max_cache_bytes=256 << 20,
)
store.keys()  # immediate children; metadata only
group = store["experiment"]  # RemoteStore view
arr = group["temperature"]  # RemoteArray
values = arr[:100]  # fetch and cache on demand
result = (arr + 273.15).compute()  # existing array expressions

disk_store = blosc2.RemoteStore(
    "https://host/data.h5",
    cache_dir="./cache",  # selects DISK when policy is omitted
    max_cache_bytes=256 << 20,  # total retained payload for this store
)

direct = blosc2.RemoteArray(
    "https://host/data.h5",
    dataset="experiment/temperature",
    cache_policy=blosc2.CachePolicy.MEMORY,
)
```

- `RemoteArray` retains current array operations, format support, serialization,
  and standalone cache defaults. Rename the class, module, exports, references,
  tests, and documentation without a compatibility alias. Historical plans need
  not be renamed. No migration support for unreleased RemoteProxy artifacts is
  required; update current persisted-type dispatch consistently.
- `RemoteStore` defaults to MEMORY, with a 256 MiB store-wide allowance. An omitted
  policy plus `cache_dir` selects DISK; contradictory explicit options raise.
  Reuse array validation: MEMORY requires a finite positive limit, DISK permits
  `None` for unbounded retention, and NONE takes no payload limit or disk cache.
- Group views and arrays share the root store's policy, budget, and resources.
  No independent per-leaf limit overrides in v13. A separately constructed
  RemoteArray remains independent.
- Expose immediate-child iteration/`keys()`, string-path lookup, group attributes,
  node kind and unsupported-node diagnostics, source identity, and cache/traffic
  information. Keep UI rendering types out of the library API. Do not implement
  mutation methods or subclass local TreeStore merely to reuse its interface.
- Paths are relative to the selected group. Both `store["a/b"]` and
  `store["a"]["b"]` resolve the same source and cache entry. Missing paths raise
  KeyError; unsupported objects remain discoverable and raise a useful error
  when opened as arrays.
- Keep `blosc2.open(..., lazy=True, dataset=...)` array behavior, now returning
  RemoteArray. The explicit RemoteStore constructor is the new hierarchy entry
  point; do not broaden generic root-opening dispatch or alter existing eager
  localization behavior as an incidental part of this change.

## Responsibilities and reuse

RemoteArray selects and fetches array data through existing source readers and
Proxy machinery. RemoteStore discovers hierarchy metadata and supplies shared
source resources and cache ownership. A small internal cache coordinator owns
aggregate accounting and eviction; it does not decode formats or fetch arrays.

Start by tracing all cache mutation and access paths in `proxy.py` and
`remote_proxy.py`, including partial blocks, hits, fill operations, expressions,
serialization, and trimming. Reuse their fetched maps, compressed caches,
mutation recovery, and eviction primitives. Do not add a second payload cache,
generic backend plugin framework, or replacement chunk reader.

Extract `b2view/hierarchy.py` discovery into `remote_store.py`. Retain helpers in
the existing B2Z, Zarr, and HDF5 source modules when used by both discovery and
array readers. Replace imports of browser NodeInfo/ObjectInfo and rendering
helpers with library metadata that StoreBrowser translates for the UI.

Reuse one B2Z archive/index, one HDF5 reference map, and the appropriate Zarr
store per root session. Add the smallest internal source-attachment path needed
for store arrays. In particular, do not reopen every leaf from its URL or bypass
the current B2Z attachment restriction without validating source identity.

## Shared payload cache

Use existing per-array backing storage with common ownership and eviction.
Different array geometries do not need to fit into a single B2ND carrier.

- Identify entries by canonical dataset identity and native cache chunk number.
  Preserve partial-block fetched maps and charge the actual retained compressed
  representation. Replacing a partial chunk updates its charge rather than
  counting both versions. Whole-chunk eviction is sufficient for v13.
- Maintain one LRU across array caches, touching entries on hits and publication.
  Repeated handles and group aliases reuse a dataset's cache state. Do not create
  a fresh Proxy/cache every time the same array is looked up.
- Enforce the aggregate limit after operations, including failures that retain
  data, following the current post-operation limit contract. An oversized chunk
  may be fetched to serve a read and then evicted. The budget does not cap output
  arrays, decompression buffers, metadata, or transient peak memory.
- Retain warm payloads when the browser changes selection. Cache ownership must
  not depend on retaining the selected UI handle. Conversely, enumerating a
  hierarchy must not allocate carriers/fetched maps for every array.
- Keep standalone behavior on the same code path with a private cache owner.
  Avoid duplicating accounting and applying both a per-array and store limit.
- Store `cache_bytes` reports the aggregate; array `cache_bytes` reports that
  leaf's contribution. Clearly document that a store-derived array's configured
  limit belongs to the store. Shared transport traffic is reported once at the
  root; do not sum multiple views of the same counter.

Initially serialize store cache operations within one process using a shared
reentrant lock if needed. Preserve concurrency inside existing remote fetch
operations. Establish one lock order before adding hooks to avoid cross-array
eviction deadlocks. Mark any deliberate coarse locking with a ponytail comment
describing its throughput ceiling and possible later refinement.

## Disk layout, ownership, and reopening

Treat `cache_dir` as a parent containing source-derived store directories, not
as a single shared cache for all URLs. A store directory contains one manifest
and lazily created per-array payload caches. Reuse current carrier/storage
primitives; a packed store cache format is unnecessary.

The payload bound is per store, not per parent directory, customer, or physical
filesystem. File overhead and stale physical allocation are not compressed
payload accounting. Do not describe this bound as a strict disk quota.

Only one independently opened owner may use a given store cache directory at
a time. Group views and arrays share that owner. Use a lifetime exclusive lock
to reject conflicting opens clearly, including another process; this is an
exclusion guard, not concurrent shared-cache support. Reuse an existing suitable
locking facility where possible. Verify crash release and supported platforms;
do not implement a stale PID-file lock or silently run without exclusion.
Different source-derived directories under the same parent remain independent.

On reopening, include retained payload from all leaf caches in accounting, even
if those leaves have not been selected in this session. Restore sizes from
existing bookkeeping and trim locally if a smaller limit was requested. Exact
LRU order need not survive a restart; a deterministic cold ordering is sufficient.
No remote payload fetch is needed merely to account for or evict local data.

## Persistent discovery manifest

Keep discovery metadata separate from evictable payloads:

| Format | Persisted discovery information |
| --- | --- |
| B2Z | Logical hierarchy, attributes, unsupported boundaries, ZIP member offsets/lengths and metadata needed to reopen bounded members |
| HDF5 | One Kerchunk reference map, hierarchy, attributes, unsupported diagnostics |
| Zarr | Discovered groups/arrays, attributes and decoding/layout metadata, including consolidated metadata where available |

Store a versioned, validated source descriptor and serializable discovery data;
never pickle live filesystem, archive, Zarr, or browser objects. Reuse current
serialization dependencies and URL portability validation. Credentials and
storage options remain runtime inputs; audit reference-map URLs as well as the
top-level URL so secrets are not copied into manifests.

Restore source readers from persisted locators through explicit internal hooks.
Saving offsets without teaching readers to reuse them does not satisfy this
milestone. HDF5 arrays share the manifest's map rather than embedding a duplicate
map in every store-owned carrier. Keep standalone array export self-contained
under its existing contract; store-owned backing files are implementation
details, not automatically portable standalone exports.

Zarr discovery remains lazy. Persist what is known and which groups have been
listed; absence from a partial manifest is not evidence that a child is missing.
Do not force a recursive scan on open or close to produce a complete manifest.

Publish manifests atomically with a temporary file and replacement. Tie locator
metadata and payloads to the same source identity/generation. Validate manifest
schema, paths, offsets and cache identities before using them. An invalid or
interrupted manifest must never cause old locators to be used with new payloads;
rebuild disposable state safely or report an actionable error.

Keep manifests when payloads are evicted. Expose `metadata_bytes` as the encoded
manifest size, distinct from payload bytes and Python memory consumption. Large
HDF5 reference maps remain possible and are outside `max_cache_bytes`.

## Source assumptions and lifetime

Support immutable sources, retaining current format restrictions. Do not claim
automatic change detection for Zarr/HDF5 or derive a content version solely from
matching shape/dtype. Reuse reliable validators where already available.

Provide an explicit root refresh that rebuilds discovery and invalidates dependent
payloads as one generation change. Existing child handles become stale and raise
a clear error requiring lookup again; do not silently mix source generations or
reinterpret an old array handle with a different shape. Build replacement
metadata before publishing it so a failed refresh does not publish a half-state.

Provide context-manager/close behavior. Closing a handle releases its ownership;
arrays and group views already returned remain usable until individually closed
or released. The last dependent handle releases resources and the disk ownership
lock. Ensure owner/cache references do not create a cycle that keeps the lock
alive indefinitely. Operations on an explicitly closed handle raise clearly.

Persistent caching does not initially promise fully offline reopening. Metadata
validation or source initialization may still contact the remote service; state
that limitation while proving that a valid manifest avoids repeated discovery.

## b2view migration

- Use RemoteStore for remote roots/groups and RemoteArray for direct leaves.
- Set one 64 MiB MEMORY budget for a browsing store, preserving the current
  preview allowance while allowing warm revisits across arrays.
- Translate public metadata into browser presentation types in StoreBrowser.
  Preserve empty groups, subtree-relative navigation, attributes, unsupported
  object messages, source headers, traffic reporting and standalone presentation.
- Preserve direct-array opening without listing its parent; preserve codec
  initialization required before Textual captures stderr.
- Remove the old internal RemoteHierarchy implementation after migration. Keep
  local TreeStore, CTable, plotting and query behavior intact.

## Implementation sequence

1. Rename RemoteProxy throughout maintained source/tests/docs and update current
   serialization dispatch. Run focused array/proxy/expression checks.
2. Extract public RemoteStore discovery and resource ownership. Return RemoteArray
   leaves through shared source readers; verify discovery does not fetch arrays.
3. Introduce minimal shared accounting/eviction hooks in existing cache machinery.
   Deliver and validate NONE and aggregate MEMORY behavior first.
4. Add store DISK layout, exclusive ownership, persistent manifest restoration,
   aggregate reopening/eviction and explicit refresh. Preserve standalone exports.
5. Migrate b2view, remove its internal implementation, and update reference docs,
   remote-array guide and examples with the two-type model and cache semantics.
6. Run focused and default repository checks; record actual results and limitations
   before marking this plan implemented.

## Validation and acceptance

Use the `blosc2` conda environment for every Python/test/build command. Extend
existing source, Proxy, serialization, expression and b2view tests rather than
creating another test framework. Prefer memory-backed fsspec and local fixtures;
network validation remains separately marked.

Required behavioral coverage:

- All three formats: root/subgroup lookup, aliases, attributes, empty groups,
  supported arrays, unsupported siblings, slicing and array expressions.
- Existing B2Z ZIP64/range checks, partial-block reads and restrictions; existing
  Zarr layouts/codecs and HDF5 filter/Kerchunk behavior remain intact.
- Read A, read B, revisit A under generous and restrictive shared budgets. Verify
  both returned values and transport hits/misses, aggregate bounds and cross-array
  eviction. Cover duplicate handles, partial chunk replacement and oversized reads.
- NONE retains no payload; enumeration creates no per-leaf payload caches.
- DISK reopen reuses discovery and warm chunks, counts unopened leaf caches, and
  applies a smaller budget without remote payload reads. HDF5 translation happens
  once initially and is not repeated on a valid manifest reopen or leaf selection.
- Manifest-only retention after payload eviction, partial Zarr discovery, malformed
  manifests, interrupted publication and generation refresh with stale handles.
- Conflicting owner opens fail in one process and in a subprocess. Lock release
  after normal close and process termination permits a later open.
- Returned arrays survive closing their originating store handle; final resource
  release closes transport handles and releases the disk lock.
- Standalone RemoteArray persistence/export/expression behavior and b2view's local
  and remote navigation/preview behavior pass their existing regression checks.

Run the focused suites after each relevant phase, then the default pytest suite
and Ruff checks for changed files. Do not mark network or offline behavior tested
unless explicitly exercised. No implementation or tests are run by writing this
plan alone.

## Deferred

Caterva2 integration; customer quotas; multiple processes concurrently using the
same writable cache; distributed locking; cache services; strict physical-disk
limits; automatic mutable-source invalidation; fully offline reopening; remote
writes; embedded B2Z array/CTable payload support; per-array reservations; public
cache-backend plugins. None is a prerequisite for this version.
