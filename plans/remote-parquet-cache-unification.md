# Unify Parquet with the remote table cache lifecycle

Status: cache behavior ready for manual checks. The live cache uses readable
source directories, `StoreDiskCache` generations, native group `.b2d` payloads,
and `CacheCoordinator` accounting. Generation roots reopen as Parquet tables,
and `.save("reference.b2z")` exports retained groups as a single archive.

Implementation decision: keep the Parquet owner and storage adapter for Arrow
row groups. `RemoteDiscovery` assumes existing Blosc2 array leaves for its
format dispatch, while Parquet creates converted table segments on demand.
Forcing that class to own Parquet would add format branches across discovery,
table storage, and export. The shared cache primitives now provide the intended
layout and lifecycle behavior. The deeper `RemoteDiscovery` integration below
remains an optional architectural follow-up rather than a requirement for
manual cache checks.

The source revision is still part of Parquet's cache identity, so `refresh()`
uses a new readable source directory after a file changes. Moving that revision
into a new generation under a stable source directory is a separate lifecycle
change; the current behavior keeps old revisions isolated and old references
detect source changes.

## Objective

Treat a Parquet file as a table source managed by the existing RemoteStore /
RemoteCTable machinery. Use the same source naming, generation manifests,
ownership, cache budgets, refresh, direct cache reopening, and portable exports
as other remote table formats. Keep Parquet decoding and row mapping in its
source adapter.

The unit fetched from Parquet can remain a physical field within a row group.
Consistency does not require mapping a Parquet row group to one Blosc2 chunk.

## Current differences

| Concern | Existing remote store machinery | Current Parquet implementation |
| --- | --- | --- |
| Disk identity | Readable source basename plus identity hash | `parquet/<64-character hash>` |
| Live container | `<generation>.b2d` and `active_generation.json` | Independent row-group `.b2z` files and optional `.npy` row maps |
| Ownership | `StoreDiskCache`; `SharedStoreCache` for shared operations | Separate `_disk_guard` used for disk reads |
| Cache budget | `CacheCoordinator` shared by retained payloads | Private memory LRU and disk eviction by modification time |
| Refresh | Prepare/publish generation, invalidate old handles | Reopen a separate Parquet owner and change its generation counter |
| Export | Remote-store manifest and retained payloads in a `.b2z` archive | `remote_parquet` CFrame reference containing cache files as byte strings |
| Reopening live cache | Root manifest reconstructs the remote table | Individual entries open as partial ordinary CTables |

Relevant code: `remote_parquet.py`, `remote_store.py`, `remote_store_cache.py`,
`remote_ctable.py`, `ctable_storage.py`, and `proxy.py`.

A terminology correction: `.b2d` is a directory container. It does not imply
that every array inside it uses `contiguous=False`. The common array code has
distinct ordinary contiguous carriers and shared sparse frames. Preserve that
distinction; do not turn every Parquet payload into a sparse frame merely to
obtain a `.b2d` root.

## Proposed layout and identity

```text
proves/
  chicago-taxi.parquet--<identity-hash>/
    owner.lock
    active_generation.json
    <generation>.b2d/
      embed.b2e
      _parquet/
        row-map.b2nd                 # Only when flattening requires a row map
        groups/
          0/
            <physical-field-key>.b2d/
              ...native CTable payloads...
          1/
            <physical-field-key>.b2d/
              ...native CTable payloads...
```

Use `StoreDiskCache.path_for()` and the existing `cache_directory_name()` /
path-component helpers. Do not implement another basename sanitizer or hashing
convention. The illustrated payload subtree is private; the public entry point
is the generation root, just as for other cached remote tables.

The identity must include the normalized source location, access-configuration
fingerprint, projection, row limit, effective null policy, and options that
change decoded values or their cache representation. Keep credentials out of
persisted descriptors. Record cache format/conversion compatibility versions
explicitly. Persist the source revision marker in the generation metadata so
refresh replaces a generation within the same logical cache directory.

Use a versioned Parquet section of the common manifest for the compiled schema,
physical/logical field mapping, row-group boundaries, decoding options, revision
marker, row map, and published group entries. Register the root as a `ctable`
node; auxiliary group tables are implementation details, not user-visible
tables in the logical hierarchy.

## Reuse the lifecycle, retain a small Parquet adapter

1. Add Parquet discovery and reconstruction to `RemoteDiscovery` and the common
   source-descriptor validators. Route Parquet table construction through that
   owner. Preserve the existing public `blosc2.open()` options and read-only
   table behavior. Keeping `RemoteParquetCTable` as a thin compatibility class is
   fine; it should no longer own a separate cache lifecycle.
2. Keep fsspec/PyArrow handles, footer decoding, logical row mapping, and
   `CTable.from_arrow()` conversion in `remote_parquet.py`. Preserve current
   nested/null/dictionary semantics and importer parity.
3. Make `ParquetTableStorage` obtain converted groups from the common owner.
   Retain its column adapter for gathering values across unequal groups. The
   existing common table storage assumes B2Z members in several paths, so simply
   changing the owner class is insufficient: add explicit Parquet dispatch at
   those storage boundaries.
4. Store each converted group/physical-field unit as native directory-backed
   CTable payloads under the generation, replacing its ZIP wrapper. This avoids
   requiring global string offsets or dictionary codes before all groups have
   been visited. A physical field may supply multiple logical leaves and masks;
   record that mapping rather than assuming one physical field is one scalar
   column.
5. Register these units with `CacheCoordinator` through one small adapter, using
   the existing remote batch cache as the model. The coordinator currently
   expects concrete cache methods/maps; implement that interface, not a second
   coordinator or a general plugin framework.

This is substantive lifecycle reuse, not a `.b2d` wrapper around the current
standalone `.b2z` cache directory. Directly reopening the generation must expose
the complete logical table and fetch missing groups through its source adapter.

## Reads, publication, accounting, and sharing

A miss converts the required physical field for its row group, including any
dependent masks, dictionaries, and nested structure. Build it under a temporary
sibling directory. Under the common ownership lock, publish the completed unit
and then its manifest entry. A crash before publication must not turn a partial
unit into a cache hit. Recover abandoned staging/unreferenced payloads through
the same generation/dirty-state machinery used for other formats.

Retain and evict complete converted units initially, including their dependent
payloads. This keeps dictionary and offset state coherent. Cache hits touch the
common LRU; reopened caches restore accounting from validated manifest entries.
Count retained compressed data consistently with the shared coordinator and
report metadata separately. Count directory/manifest overhead separately from
the compressed-payload budget, as with other stores.

A row group larger than the budget must still be readable. Define the retained
state using the common budget contract: serve through a temporary live unit and
release it when it cannot be retained, rather than keeping an oversized group
indefinitely. Do not claim this bounds peak Arrow conversion memory.

Ordinary DISK access uses the existing lifetime ownership lock. Shared access
uses `SharedStoreCache` and operation-scoped locking, with the normal manifest
reload, dirty recovery, and stale-handle checks. A shared cold miss must not
publish duplicate or inconsistent units. Complete group payloads can remain
immutable after publication; sparse per-chunk writes are unnecessary for them.

NONE and MEMORY policies retain their documented behavior through the same
owner, without creating disk artifacts. Refresh prepares compatible metadata
and a new generation before retiring the old one. Failed refresh leaves the
previous table usable. Preserve source-change detection and require explicitly
defined validation when reopening a persisted generation; do not accidentally
weaken the current Parquet revision checks while adopting shared machinery.

## Opening and transport

These should be equivalent ways to recover the same logical remote table:

```python
t = blosc2.open(url, cache_dir="proves")
t = blosc2.open("proves/chicago-taxi.parquet--<hash>/<generation>.b2d")
```

The generation root contains enough metadata to expose the whole schema and
reuse retained groups. Unknown sizes should be reported as unavailable or
explicitly partial; information display must not fetch all groups to compute
them. Opening and reads must honor common closed/stale-handle semantics.

Use the existing remote-store `.save()` archive writer, extending payload
enumeration for Parquet group units and the row map. The new default portable
table artifact is a genuine `.b2z` remote-reference archive:

```python
t.save("taxi-reference.b2z")  # Include currently retained data
t.save("taxi-cold.b2z", include_cache=False)
u = blosc2.open("taxi-reference.b2z")
```

Export must not fetch unvisited groups. Stream/copy stored payloads into the
archive rather than embedding every file as an in-memory byte string in a
CFrame metadata payload. The shared archive reader must understand Parquet
segments and serve warm data from the exported file. Uncached reads still need
the source and runtime credentials. Cache mutability on reopening should follow
the existing remote-reference rules, independently of source read-only status.

Local Parquet sources remain supported. Keep the distinction between reopening
a local cache and transporting a remote reference: a local source path does
not become available on another machine merely because its cache was exported.

## Compatibility and rollout

Implement and verify the internal integration before changing the disk layout.
Do not independently land a cosmetic rename that causes an extra cache rebuild.

- Treat old `parquet/<hash>` runtime caches as disposable. Create the new cache
  on the next open; leave old directories untouched and document cleanup. No
  automatic migration is needed for these runtime files.
- Preserve a read-only compatibility loader for existing version-1
  `remote_parquet` CFrame references. Seed their retained data into the common
  representation after validation. New `.save()` output uses the common archive
  format; the loader must not retain a second runtime cache implementation.
- Version new manifest fields and validate paths, source revision, group
  identity, shape, and conversion identity before accepting a cached unit.
- Keep current import options and numerical/null semantics. Do not combine this
  change with a new Parquet page reader, global dictionary scheme, or per-chunk
  group conversion strategy.

## Implementation sequence and acceptance gates

1. **Shared source ownership:** Add the Parquet table node and descriptor
   metadata, reuse ordinary/shared ownership and generations, and preserve the
   existing importer parity tests, including variable-length and nested data.
2. **Native payloads and common accounting:** Replace per-group ZIP files with
   native group directories, integrate `CacheCoordinator`, and test publication,
   eviction, reopened cache hits, and shared concurrent misses.
3. **Reopen/export integration:** Open the generation root as a full
   `RemoteCTable`, export/import standard `.b2z` references, and support old saved
   references through the compatibility loader.
4. **Remove duplicate paths and document:** Remove Parquet-specific path naming,
   disk locking, LRU/eviction, lifecycle, and new-reference CFrame serialization.
   Document the common layout and remaining Parquet fetch granularity.

Completion requires tests covering unequal row groups, a read of another column
and another group, repeated warm reads with no payload refetch, persisted reopen,
complete logical schema at the generation root, refresh/stale views, aborted
publication, oversized units, and two processes sharing a cold miss. Export to
a different directory/process must preserve warm reads and allow cold misses to
fetch the source; old-reference loading must retain its source checks. Run the
existing remote-store/CTable cache and export tests alongside Parquet parity.

Compare cold open, one narrow cold read, warm read, reopen, and export on a
multi-group mixed-type fixture. Record source traffic, retained bytes, and file
count. Changes must preserve column/group isolation and avoid repeated
conversion of retained data; no full-file import is an acceptable fallback.
