# RemoteCTable reference saving

Status: deferred proposal for later consideration. This document does not change
the current save behavior.

## Goal

Make `RemoteCTable.save()` preserve a remote reference and optional retained
cache, consistent with `RemoteStore.save()`. Keep explicit independent local
exports available through `to_b2z()`, `to_b2d()` and `copy()`.

Do not change `CTable.save()` for ordinary local tables. Reuse the existing
RemoteStore artifact format, cache ownership and reader infrastructure rather
than introduce a second table-specific reference format.

## Current behavior and dependencies

- RemoteCTable inherits CTable.save(), which materializes live rows into a local
  table and returns None.
- RemoteStore.save() writes a portable .b2z reference archive containing its
  source descriptor, discovery metadata and optionally already-retained data.
  It returns the output path and does not fetch all missing data.
- RemoteStore._collect_export_nodes() explicitly rejects CTable nodes. Its
  existing export logic also does not preserve CTable schema metadata as needed
  for reconstructing those nodes. Removing the rejection alone is insufficient.
- CTable.to_b2z() and to_b2d() dispatch to self.save() on remote root tables.
  These must retain materializing semantics when RemoteCTable overrides save().
- No direct RemoteCTable.save() callers were found in the repository during the
  current audit. test_remote_utf8_nested_lifetime indirectly relies on save()
  through table.to_b2z(), and verifies an independent local table. External
  callers are unknown; document the change rather than assuming none exist.
- RemoteCTable now exposes refresh() and is_cache_mutable. Artifact readers
  must preserve their cache-writability and stale-handle contracts.

## Proposed public contract

Proposed signature, to confirm when implementing:

```python
def save(destination, *, include_cache=True, mutable=None, overwrite=False) -> str: ...
```

Match RemoteStore.save() for accepted options, defaults, .b2z destinations,
return value, overwrite checks and destination safety. This changes both the
meaning and return value of the inherited method; note the destination keyword
change from CTable.save(urlpath=...) as well.

- include_cache=True exports only already-retained data, not missing chunks.
- include_cache=False exports enough metadata to reconstruct the table without
  its cached column payloads. Bootstrap metadata is still necessary; do not
  promise a file containing literally zero source bytes.
- mutable controls whether the reopened local artifact/cache can grow, not
  whether the remote table becomes writable. Reuse the store's default rules.
- blosc2.open(saved_path) returns RemoteCTable for a table-root reference and
  RemoteStore for a group-root reference. Ordinary materialized CTable archives
  continue to open as local CTable instances.
- A standalone table and a full table selected from RemoteStore can be saved.
  Row-filtered/projected/computed views are not new remote-reference objects in
  this phase; their existing explicit materialization paths remain available.
- Cached reads use local data; missing reads contact the original source.
  A saved reference is not a guaranteed complete offline copy or a frozen copy
  of the original remote object.

## Implementation steps

### 1. Preserve explicit materialization before overriding save

Audit every self.save() dispatch in CTable conversion helpers. Make the root
remote to_b2z()/to_b2d() paths call the materializing implementation explicitly,
or use a narrowly shared internal writer if that avoids duplication. Keep local
fast pack/unpack paths and view compaction unchanged.

Do not require copy().save() merely to bypass virtual dispatch: that can add an
unnecessary whole-table in-memory copy. Retain the current writer's resource
behavior. Test the explicit export methods before changing reference saving.

### 2. Extend the shared artifact exporter

Allow a selected CTable root and CTable descendants of exported groups. Preserve
the CTable node metadata, schema, user attributes and source dataset selection.
Retain original archive member paths consistently with source descriptors,
bootstrap seeds and fingerprints; avoid casually rebasing paths.

Account for all physical arrays, not just visible logical column names:

- Fixed-width and shaped columns.
- UTF-8 offsets and bytes companions.
- Null-mask sidecars and the live-row/deletion mask.

Export existing cached portions through the existing carrier-copy machinery.
Do not eagerly open every column, decode every string or fetch missing chunks
to produce an include_cache=True artifact. Metadata-only and never-read tables
must also export correctly. Keep cache-budget checks and atomic file publication.

Keep destination exclusions: no overwriting a live cache, the source artifact,
or another existing destination unless explicitly authorized by overwrite=True.
Use the existing source-descriptor/storage-options fingerprint handling; never
introduce serialized credentials or live filesystem/session objects.

### 3. Extend shared artifact validation and dispatch

Trace artifact loading end to end: manifest validation, immutable/mutable
opening, owner initialization, root-kind checks, blosc2.open() dispatch and
nested lookup. Support table roots without loosening group/array validation or
accepting malformed CTable schema and member mappings.

Reuse RemoteCTable._from_owner() and RemoteTableStorage. Preserve owner lifetime
rules so closing a parent handle does not prematurely close a returned table.
Ensure cache-only export/reopen does not reintroduce remote identity requests
under the immutable-source assumption when all bootstrap metadata is present.

### 4. Implement RemoteCTable.save()

Delegate to the shared exporter with the table's selected dataset; do not
duplicate ZIP construction and manifest serialization. Keep root/selection
checks explicit and preserve closed/stale-handle errors.

Runtime max_concurrency and buffer settings are not a new serialized artifact
contract in this phase: use defaults or caller overrides on reopen, consistent
with the existing readers. Do not extend blosc2.open() with table buffer options.

### 5. Integrate artifact lifecycle

Verify immutable artifacts report is_cache_mutable=False, never modify their
files during reads and reject refresh(). Missing data may still be fetched into
transient memory, as with store artifacts.

Verify mutable artifacts use the existing writable-cache machinery, preserve
cached data across reopening and support standalone table.refresh(). Refresh
must retain settings, replace the cache generation and invalidate borrowed
columns/views. Tables obtained through a group still refresh via the root store.

### 6. Documentation and migration

Document reference saving versus materialization side by side:

```python
remote.save("reference.b2z")  # source + metadata + retained cache
remote.save("cold-reference.b2z", include_cache=False)
remote.to_b2z("complete-local.b2z")  # independent table data
remote.to_b2d("complete-local.b2d")
local = remote.copy()  # independent in-memory CTable
```

Explain network access, immutability, storage credentials on another machine,
cache writability and overwrite behavior. State explicitly that local
CTable.save() remains unchanged. Update the guide's current statement that
portable table-reference export is unsupported only once implemented and tested.

## Verification

Use focused fixtures and parametrization rather than live-network-only tests:

1. Root and nested table reference round trips through blosc2.open(); mixed
   groups containing arrays and tables continue to return the appropriate types.
2. Fixed-width/shaped/nullable columns, multilingual UTF-8, empty tables and
   deleted rows preserve schema, attributes and values.
3. include_cache=True preserves partial warm reads; include_cache=False leaves
   data cold. Instrument info and byte-range transport separately. Saving does
   not fetch missing payloads, and a fully cached selection reopens locally.
4. Uncached reads after reopening still work, including UTF-8 spans requiring
   both offsets and bytes. Exporting a nested table does not include unrelated
   sibling data or confuse absolute dataset paths.
5. Immutable and mutable artifacts obey existing cache policy/budget rules;
   refresh, closing, parent lifetime and stale handles behave consistently.
6. Failure/overwrite/path-safety tests leave existing artifacts and caches
   intact. Invalid manifests and unsupported table kinds fail clearly.
7. to_b2z()/to_b2d() remain standalone local exports: remove the test remote
   source or forbid transport before reopening and reading the exported table.
8. Existing RemoteStore and RemoteArray reference artifacts remain readable.
   Version the manifest only if its actual compatibility requirements demand it.

Run focused CTable, B2Z and RemoteStore tests, then the full suite and Ruff in
the blosc2 conda environment. Finish only when reference saving no longer
changes the semantics of explicit materialization methods.

## Boundaries

No remote writes, automatic change detection, new remote formats, automatic
full-cache population, reference export of arbitrary table views, or changes
to local CTable.save(). The proposal is intentionally deferred; revisit the
public signature and compatibility notes before starting implementation.
