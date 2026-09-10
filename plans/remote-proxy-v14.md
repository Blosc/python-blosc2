# RemoteStore v14: TreeStore-backed caches and portable exports

Status: implemented and locally validated, 2026-09-10. All implementation steps
and acceptance criteria complete. TreeStore-backed DISK caches, portable .b2z
reference exports (`RemoteStore.save()`), root marker dispatch via `blosc2.open()`,
immutable (`mutable=False`) and mutable (`mutable=True`) cache modes, allowance
validation, and single-HDF5-map reuse are fully implemented and verified.

## Implementation progress

- Step 1 (Root schema, mutability model, and TreeStore leaf alignment):
  - Updated `DictStore._external_ext` and `DictStore._is_external_value` in `src/blosc2/dict_store.py`
    so store-owned `RemoteArray` leaves externalize with `.b2nd` instead of `.b2f`.
  - Defined the versioned `"b2remote_store"` root marker in `embed.b2e.meta` alongside `"b2tree"`,
    and the `b2remote_manifest` dictionary stored in `embed.b2e.vlmeta`.
  - Added boolean `.mutable` property (defaulting to `False` for future exports) and `.is_cache_mutable`
    property to `RemoteStore` and `RemoteArray`.
  - Group views inherit and propagate their owner's `.mutable` setting; independent arrays own theirs.
  - Guarded `_enforce_cache_limit` in `src/blosc2/proxy.py` when `_schunk_cache.mode == 'r'`, avoiding
    any cache-touch or eviction writes to read-only carriers.
  - In `src/blosc2/remote_array.py`, updated `__getitem__` and `_serialized_operation` to evaluate
    cache misses transiently on immutable caches without writing to disk.

- Step 2 (TreeStore-backed DISK layout):
  - In `src/blosc2/remote_store_cache.py`, migrated `StoreDiskCache` to store active generations as
    genuine TreeStore directories: `<generation>.b2d/` containing `embed.b2e` and logical leaf files
    `<generation>.b2d/<key>.b2nd`.
  - Replaced legacy `manifest.msgpack` with atomic `active_generation.json`.
  - Updated cache discard logic to remove obsolete `.b2d` generations and reject incompatible legacy
    development caches with actionable errors.

- Step 3 (Portable `.b2z` reference export):
  - Implemented `RemoteStore.save(destination, *, include_cache=True, mutable=None, overwrite=False)`
    and `RemoteStore.to_cframe()` in `src/blosc2/remote_store.py`.
  - Implemented export precedence: explicit `save(mutable=...)`, then assigned `.mutable` value,
    otherwise `False`.
  - Added subtree export support (`store["group"].save(...)`) with remapped root dataset and relative child keys.
  - Preserves empty groups, attributes, and unsupported-node diagnostics.
  - Validates destination extension (`.b2z`), rejects directory destinations, enforces `overwrite` policy,
    and rejects destinations located inside the live cache storage directory.
  - Validates retained cache payload against declared `max_cache_bytes` at save-time.

- Step 4 (Root marker dispatch and shared-owner restoration):
  - Updated `_open_special_store` in `src/blosc2/schunk.py` to detect `"b2remote_store"` in `embed.b2e`
    metadata and route `blosc2.open("snapshot.b2z")` directly to `RemoteStore._open_artifact`.
  - **Immutable snapshots (`mutable=False`)**: served directly in-place from `.b2z` member offsets
    or `.b2d` paths without disk writes. Verified on `chmod 0o444` read-only files with byte-for-byte
    SHA-256 preservation across hits and misses. Misses are fetched transiently.
  - **Mutable snapshots (`mutable=True`)**: safely staged into an independent writable runtime storage
    directory under standard policy and LRU eviction, keeping the original `.b2z` file untouched.
  - Budget validation: opening an immutable snapshot with an allowance smaller than its retained payload
    is rejected with an actionable error (`"smaller than retained immutable payload"`). Opening a warm
    snapshot with `CachePolicy.NONE` is rejected. Mutable snapshots opened with a smaller requested budget
    are trimmed via LRU eviction before returning.
  - Preserves single HDF5 Kerchunk reference map across leaves without repeated translation on reopen.
  - Refreshing an immutable artifact is rejected with an actionable error.

- Step 5 (Comprehensive acceptance tests and validation):
  - Full default test suite after review fixes passed on macOS: **9,918 passed, 34 skipped in 30.33 seconds**.
  - Focused remote store, array, fsspec, proxy, and `b2view` hierarchy tests after review fixes:
    **371 passed, 5 skipped, 8 deselected in 7.29 seconds**.
  - `b2view` hierarchy tests: **16 passed**.
  - Ruff formatting and linting: zero errors across `src/blosc2` and `tests`.
  - Added acceptance tests in `tests/test_remote_store.py` covering:
    - B2Z, HDF5, and Zarr v2/v3 export and reopen (warm, cold, immutable, mutable).
    - `chmod 0o444` read-only byte preservation across hits and misses.
    - Budget validation, eviction trimming on mutable reopen, and rejection of smaller budgets on immutable snapshots.
    - Subtree exports with remapped root datasets and relative child keys.
    - Single HDF5 Kerchunk reference map reuse without repeated translation on reopen.
    - `.mutable` property hierarchy inheritance and validation.
    - Save destination validation (extension, directory, collision, and live cache nesting).

- Review follow-up (completed 2026-09-10):
  - Validate artifact generations, manifest paths and archive members before staging; reject directory
    symlinks. Reuse generation validation wherever generation values form disk-cache paths.
  - Release the cache owner and lock on extraction failure, and publish the active generation only
    after successful restoration. Regression tests verify reopening the same cache after failure.
  - Make immutable `RemoteArray.get_chunk()` and `aget_chunk()` misses transient, preserving retained
    payload and fetched-state metadata. Reject `fetch()`, `afetch()`, and `trim_cache()` on immutable caches.
  - Attach immutable carriers read-only even when opened with mode `"a"`, and avoid eviction on cached reads.
  - Reject standalone immutable RemoteArray payloads exceeding their persisted cache allowance on both
    file and CFrame reopen. Regression tests verify unchanged carrier bytes and cache accounting.
  - Ruff lint/format checks on the five files changed during review and `git diff --check` passed.

## Objective

Represent a RemoteStore DISK cache generation as a genuine TreeStore directory
(`.b2d`) and add `RemoteStore.save()` to produce a portable `.b2z` reference
archive. Transport the source description, discovered hierarchy and optional warm
payload together, preserving shared RemoteStore ownership and cache accounting
when reopened.

An export remains a remote reference. Missing regions require access to the
original source and runtime credentials. Saving must not download every array
or claim fully offline operation.

## Prototype evidence

A local exploratory test used existing APIs to insert two RemoteArray leaves
into a TreeStore `.b2d`, pack it with `to_b2z()`, delete the original runtime cache
and staging directory, move the archive, and reopen it as a TreeStore.

| Original backend | Exported archive bytes | Warm slice source bytes |
| --- | ---: | ---: |
| B2Z | 5,999 | 0 |
| Zarr v3 | 3,686 | 0 |
| HDF5 | 4,895 | 0 |

Uncached slices returned correct values by fetching from the original sources.
An empty group and its attributes also survived. These were small synthetic
arrays on fsspec's memory filesystem, in one process; they are not network
benchmarks or proof of cross-machine portability.

The experiment exposed the remaining work:

- Reopened leaves are independent RemoteArrays, with independent cache owners.
- Each exported HDF5 leaf duplicates the reference map; the example stored two
  compressed copies of 546 bytes each.
- Direct TreeStore insertion currently gives RemoteArray leaves a `.b2f` suffix.
  Ensure store-owned NDArray carriers use the appropriate `.b2nd` representation.
- Existing carrier-copy branches need auditing for warm-payload preservation;
  packing a tree alone does not define RemoteStore serialization semantics.

## Public behavior

Public API and dispatch:

```python
with blosc2.RemoteStore(url, cache_dir="remote-cache") as store:
    with store["experiment/temperature"] as array:
        values = array[:100]
    store.mutable = False  # optional: immutable is already the export default
    store.save("snapshot.b2z", include_cache=True)

# Root marker dispatch: recognize the remote-store root marker.
with blosc2.open("snapshot.b2z") as restored:
    array = restored["experiment/temperature"]
    values = array[:100]
    array.close()
```

- `save(destination, *, include_cache=True, mutable=None, overwrite=False)` writes `.b2z`.
  Do not add directory export or extra packaging options unless needed.
- The default includes retained payload only. `include_cache=False` preserves
  reference/discovery metadata and omits fetched data, without clearing the live
  cache. Inline HDF5 values and incidental metadata-prefix bytes remain metadata.
- Support saves from NONE, MEMORY and DISK stores through one export path.
  Saving a group view should export that subtree with relative keys and the
  correct original-source root. Reject invalid destinations before copying data.
- Preserve empty groups, attributes, unsupported-node diagnostics and partial
  discovery state. Exporting a partially discovered Zarr hierarchy must not force
  a recursive listing; unknown children remain discoverable from the source.
- Standalone RemoteArray exports remain self-contained. Store-owned carriers may
  depend on the root manifest, but exporting one leaf must still produce a valid
  independent RemoteArray reference, including HDF5 references when required.
- Archives contain neither credentials nor machine-specific cache paths.
  Credentials and storage options are supplied by the receiving process.

## Persisted cache mutability

Agreed API: RemoteStore and RemoteArray expose a boolean `.mutable` property
that sets the default for future exports. Do not add `mutable` to constructors.
New live objects retain their existing cache-policy behavior; their export default
is `False`. The property is not a runtime freeze/unfreeze switch.

- Export precedence is: explicit `save(mutable=...)`, then an explicitly assigned
  `.mutable` value, otherwise `False`. Use an omitted sentinel internally (shown
  as `None` in the proposed signature). Apply the same rule to RemoteArray's
  `to_cframe()`. Re-saving an opened artifact without either explicit choice also
  defaults to `False`; its loaded runtime mode is separate from the export default.
- Neither the setter nor an export override changes the current object's cache
  behavior, contents or remote source. Validate boolean assignments. Store views
  share the owner's export default; independent arrays own theirs.
- On opening an artifact saved with `mutable=False`, serve its included regions
  without changing chunks, fetched maps, manifest or LRU metadata. Fetch misses
  transiently without retaining them across operations. Repeated misses may fetch
  again. Source arrays remain read-only under the immutable-source contract.
- On opening an artifact saved with `mutable=True`, permit cache fills, eviction
  and bookkeeping in writable runtime storage under the existing policy, budget
  and ownership rules. ZIP exports use existing extraction/staging machinery
  initially; the transported archive changes only through an explicit save.
- `include_cache` and `mutable` are independent. A cold immutable export is a
  streaming reference. To obtain a writable version of an immutable artifact,
  export a separate artifact with `mutable=True` and reopen it; setting the export
  default alone does not make the currently opened cache writable.
- Refresh must not silently replace an immutable artifact's runtime generation.
  Use a new live session or a separately exported writable artifact. Mutability
  is a behavior declaration, not a security boundary or filesystem permission.

The remote-proxy branch is unreleased. Update its descriptors and fixtures
directly; no migration or compatibility shim for prior RemoteArray/RemoteStore
artifacts is required. Store-derived leaves and group views expose the root's
setting; setters update the shared owner rather than creating per-leaf overrides.
Independent RemoteArray objects own their setting. Version the resulting schema
and reject unsupported artifacts clearly.

## TreeStore-backed DISK layout

Keep source-derived ownership directories and generation boundaries from v13.
Make the active generation a TreeStore, rather than making lock files and old
generations part of its logical hierarchy:

```text
cache_dir/
└── <source identity hash>/
    ├── owner.lock
    ├── <atomic active-generation record>
    └── <generation UUID>.b2d/
        ├── embed.b2e
        └── experiment/
            └── temperature.b2nd
```

The exact manifest placement is an implementation decision. Prefer existing
TreeStore metadata/storage primitives: one reserved root descriptor and one
shared discovery record. A compressed metadata object may suit large HDF5 maps
better than a large attribute. Do not duplicate the map in per-leaf carriers or
maintain two competing hierarchy/manifest authorities.

Retain logical dataset paths instead of hashed leaf filenames where TreeStore
can safely represent them. Check collisions with TreeStore's reserved names,
attribute files and object boundaries before committing to a mapping. Either
provide an unambiguous reversible mapping for such paths or reject them clearly;
never silently hide a valid source dataset.

Reuse existing per-array Proxy caches, fetched maps, partial-block bookkeeping,
dirty-state recovery and aggregate LRU coordinator. The storage layout changes;
the payload-fetch and eviction algorithms should not need replacement.

## Root descriptor and shared manifest

Introduce a versioned remote-store root marker distinguishable from an ordinary
TreeStore and existing object roots. It should identify:

- The portable source descriptor, selected root and immutable-source contract.
- The discovery generation and references to shared backend metadata.
- Known node kinds, attributes, diagnostics and which groups have been listed.
- Dataset-to-carrier associations and each carrier's generation/source identity.
- Requested cache policy and aggregate allowance, subject to explicit receiving
  process overrides decided below.
- Cache mutability, independent of source immutability and payload inclusion.

Preserve v13 backend metadata reuse: B2Z directory/header locators, one HDF5
Kerchunk map, and lazily acquired Zarr decoding and listing metadata. Keep
metadata separate from evictable payload. Define `metadata_bytes` as encoded
descriptor/discovery bytes, excluding ZIP and filesystem overhead.

Validate schemas, paths, identities, reference URLs and byte-range bounds before
attaching readers or trusting fetched maps. Use existing serialization helpers;
never pickle filesystem, archive, browser or source-reader objects.

## Export transaction

1. Validate destination, overwrite policy and subtree scope. Reject destinations
   inside live cache storage when they could overwrite or recursively package it.
2. Hold the existing store operation lock while capturing a consistent generation
   and cache state. A first implementation may copy under that lock; annotate the
   throughput limitation and defer background snapshot machinery.
3. Build a portable TreeStore snapshot using existing carriers and metadata.
   Include only the active generation and requested subtree. Preserve fetched
   maps with their matching compressed chunks, including partial chunks.
4. Reuse TreeStore/DictStore `to_b2z()` and its atomic temporary-file replacement.
   Do not blindly archive the v13 ownership directory: its lock, stale generations,
   temporary files and absolute-path state do not belong in an export.
5. Clean temporary state after success or failure. An interrupted save must leave
   an existing destination and the live RemoteStore usable.

Prefer the same export machinery for MEMORY and DISK snapshots. Do not promote
export to an implicit full-data materialization operation. Establish whether any
source metadata reads are necessary and test/document them separately from
payload fetches.

## Reopening and lifetime

Teach the local archive-opening path to recognize the root marker and construct
one RemoteStore owner. Ordinary `.b2z` TreeStores must retain their current
dispatch and behavior. Agreed reopening decisions:

- Use `blosc2.open("snapshot.b2z")` as the primary local entry point, recognizing
  the RemoteStore root descriptor automatically.
- Mutable archives use existing safe extraction/staging into independently owned
  writable runtime storage. The transported archive remains unchanged until an
  explicit save. Do not add an immutable-base/writable-overlay cache framework.
- Included immutable payload is part of the cache budget. There is no exempt
  snapshot tier: `cache_bytes` includes all retained cached payload, whether loaded
  from the artifact or subsequently fetched, across every leaf.
- The receiving process supplies any writable cache location and runtime storage
  options; never persist an exporting machine's absolute cache path. Retain the
  artifact's requested allowance unless the receiver explicitly overrides it.
- Audit read-only attachment so open and hits do not write Proxy bookkeeping.
  A read-only file mode alone does not establish immutable-cache behavior.

Before returning the restored store, account for all included leaf payloads,
including leaves not selected by the receiver. Writable runtime caches apply a
smaller requested budget by local eviction before returning. An immutable snapshot
cannot be trimmed in place: reject an allowance smaller than its retained payload
with an actionable error. The caller can select a sufficient allowance or produce
a separate smaller/cold export. Do not silently exceed the allowance or exclude
immutable bytes from accounting. Immutable snapshots retain no new misses; mutable
caches can replace imported chunks through the ordinary aggregate LRU.

The bound remains compressed retained payload, including charged partial-block
duplicates, rather than total `.b2z` file size or physical disk allocation.
Manifest bytes, container overhead, output arrays and transient buffers remain
outside it, as in v13. Do not introduce a second allowance for the imported data.
Validate the snapshot's retained payload against its declared finite allowance
when saving as well as opening; preserve DISK's explicit unbounded option.

NONE keeps its no-retained-payload meaning. A conflicting request to reopen a warm
artifact under NONE must not create an exempt payload tier; reject it clearly and
direct callers to a cold export. Merely reading an artifact's metadata or inline
source values does not turn it into a retained payload cache.

Aliases must share one leaf cache, one aggregate coordinator, one
discovery session and one HDF5 map. Never fall back silently to independent
RemoteArray owners while reporting a store-wide limit.

The archive remains unchanged by reads, eviction or refresh. Writable runtime
storage must have independent ownership. Preserve last-handle cleanup, process
exclusion, source-generation validation and stale-child errors after refresh.
Close all extracted-file/archive resources only after dependent handles finish.

Remote range-opening of an exported `.b2z` is a separate decision. Current B2Z
readers reject object carriers as plain NDArray members. Local transport and
reopening must work first; do not promise hosted-reference traversal without
explicit reader support and credential/range validation.

## Existing formats and unreleased artifacts

- Keep ordinary TreeStore/DictStore archives and standalone RemoteArray carriers
  working, including existing eager localization and direct-array opening.
- Treat current v13 hashed directories and remote-reference artifacts as disposable
  development state. No migration is required; reject incompatible state clearly
  and rebuild fixtures for the new schema.
- Version the root schema and provide useful errors for unknown versions, missing
  manifests, mismatched generations and malformed carriers.
- Audit DictStore insertion, embedding and externalization paths before changing
  accepted object types or suffix selection. Avoid broad serialization refactors.

## Implementation sequence

1. [x] Settle root schema, reserved-path handling, local reopen API and policy rules.
   Trace TreeStore packing/opening and RemoteArray carrier-copy paths end to end.
2. [x] Make DISK generations real TreeStores while preserving v13 locking, refresh,
   aggregate accounting and metadata reuse. Reject incompatible development caches.
3. [x] Add atomic RemoteStore `.b2z` export with cold/warm and subtree variants.
4. [x] Add root-marker dispatch and shared-owner restoration. Preserve standalone leaf
   exports and ordinary TreeStore behavior.
5. [x] Add round-trip tests, documentation and an executable transport example; run
   focused suites, default pytest and Ruff in the `blosc2` conda environment.

## Acceptance tests

- B2Z, HDF5 and Zarr v2/v3: export, move to another directory, remove the original
  runtime cache, and reopen in a fresh process. Use local HTTP fixtures so the
  source remains reachable across processes without relying on memory filesystem
  state. Verify values for warm slices and misses separately.
- Count actual source requests/bytes where applicable. Warm reads must avoid
  payload downloads; metadata/source validation traffic must be reported honestly.
- Verify one shared owner/budget after reopening, A/B/A warm revisits, restrictive
  aggregate limits, aliases, partial chunks, oversized reads and unselected caches.
- Count imported immutable payload in `cache_bytes` and the allowance. Reject
  oversized immutable snapshots or conflicting NONE requests; trim writable
  runtime copies under a smaller allowance without modifying the original archive.
  Check combined imported/new payload bounds and save-time allowance validation.
- Preserve one HDF5 reference map per store and prove no repeated translation on
  valid reopen. Independently exported HDF5 leaves must still carry their own map.
- Cold export excludes fetched payload without mutating the live cache. Warm
  export never marks absent or partially copied data as fetched.
- Immutable exports perform no hidden chunk, bitmap, manifest or recency writes
  during open, hits or misses. Test genuinely read-only files, repeated misses,
  and byte-for-byte artifact preservation. Mutable exports retain new data in
  writable runtime storage. Test persisted defaults, explicit writable copies,
  export-default changes, shared-owner propagation and budget/policy conflicts for
  both public types. Verify that omitted export arguments inherit the current
  setting and explicit overrides do not mutate the live object.
- Cover empty groups, attributes, unsupported nodes, subtree exports, reserved
  names and unlisted Zarr groups. Saving must not enumerate an entire lazy source.
- Inject export/publication failures; verify destination atomicity, cleanup and
  live-cache usability. Reject unsafe archive paths, leaked credentials, invalid
  locators and mismatched generations before use.
- Verify receiver lifetime/exclusion, smaller-budget restoration and refresh;
  ensure the transported archive remains unchanged.
- Regress ordinary TreeStore packing, RemoteArray persistence and b2view behavior.
  Record platform and network validation limits rather than claiming unrun checks.

## Non-goals

Full-source downloads, guaranteed offline archives, remote writes, automatic
mutable-source detection, concurrent writable owners, strict filesystem quotas,
new compression/container formats, and a generic cache-backend framework.

This proposal is a storage/serialization revision, not a reason to replace the
working v13 reader and cache machinery. Review the open decisions before starting.
