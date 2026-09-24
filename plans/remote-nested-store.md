# RemoteStore references inside TreeStore

Status: implemented and verified, 2026-09-21.

## Goal and agreed semantics

Allow a TreeStore to contain references to independent RemoteStore sources:
B2Z, HDF5 and Zarr, including a selected subgroup. The hosting TreeStore may
later be opened locally or hosted remotely and opened through RemoteStore.
Reuse the existing format readers; this feature connects stores together.

Names are always explicit:

```python
with blosc2.RemoteStore("https://example.org/weather.zarr") as weather:
    with blosc2.TreeStore("catalog.b2z", mode="w") as tree:
        tree["/external/weather"] = weather
```

The assignment persists a reference, not the remote contents or a borrowed live
Python object. Closing the original handle must not invalidate the reference.
There is no automatic naming or extension stripping. The mount path remains
`/external/weather` in reference exports and materialized outputs.

Opening and listing the local tree must not contact referenced sources. Access
to the reference returns a RemoteStore; discovery is deferred until an operation
needs its metadata or payload. The linked store remains read-only even when the
hosting TreeStore is writable. Deleting the local reference never deletes remote
data. Replacing an object root follows TreeStore's explicit-delete convention.

## Reference representation and integration

Use TreeStore's existing object registry and object-root boundary handling for a
versioned `remote_store` object. Persist the descriptor in local metadata, readable
without constructing a child reader. Choose one authoritative descriptor location
and avoid duplicating it across registry and carrier metadata. Registry writes
for references must propagate failures rather than use best-effort registration.

The descriptor contains the source kind, credential-free URL, selected dataset,
format version and immutable-source assumption. Persist portable cache-policy
preferences for standalone local reopening, but no credentials, filesystem
objects, callbacks or machine-specific cache directories. Cache payload, when
included, uses the existing remote artifact machinery with source-scoped keys.

Validate descriptors and logical paths before publishing or opening them. Reject
unsupported versions, invalid source kinds, traversal and credentials embedded
in URLs according to existing remote-source rules. Preserve ordinary groups,
CTables and RemoteArray leaves. Links target groups; array and table roots retain
their existing object types.

Integration points:

- `tree_store.py`: assignment, registry persistence, object-root lookup,
  containment, listing, subtree views, deletion and handle cleanup.
- `remote_store.py`: discovery, lazy linked handles, path routing, ownership,
  manifest validation, refresh and portable export. Its current node kinds and
  single-source owner require explicit extension.
- `remote_store_cache.py`: existing aggregate cache coordination, sparse-cache
  identity and process-safe operations.
- `schunk.py`: artifact dispatch only where the new manifest requires it.
- Existing B2Z, HDF5 and Zarr readers remain responsible for format access.

The hosting B2Z discovery must recognize the reference boundary from metadata
and hide its storage internals, as it does for CTable roots. Do not mistake a
reference carrier for an ordinary group or array. Unknown reference versions
must report a clear unsupported-node diagnostic.

## Opening locally and traversing remotely

Local use:

```python
with blosc2.TreeStore("catalog.b2z", mode="r") as tree:
    with tree["/external/weather"] as weather:
        with weather["temperature"] as temperature:
            values = temperature[:100]
```

Root and immediate-child enumeration expose the mount without expanding it.
RemoteStore discovery reports `kind="remote_store"` for the reference, while
ordinary subgroups continue to report `group`. Looking up the mount returns a
RemoteStore. Opening a child or asking the linked store for its keys may contact
that source. Failures at one target must leave unrelated nodes usable.

Support both chained lookup and paths crossing a mount, including mounts under
subgroup views. Resolve the remaining suffix relative to the linked dataset,
never relative to the source container root. Keep local structural walking from
implicitly following links; recursive materialization explicitly follows them.
Update consumers of node kinds, including browser presentation, so references
are browsable without eager expansion.

The current RemoteStore constructor performs discovery immediately. Introduce
only the deferred initialization needed for persisted references; do not claim
network-free reference lookup while constructing a normal eager RemoteStore.

## Cache policy, credentials and lifetime

When opened through a local TreeStore, a linked RemoteStore uses its persisted
standalone policy, defaulting to MEMORY. Provide a runtime override and runtime
credentials through an explicit reference-opening path. Proposed API:
`tree.open_remote(path, *, storage_options=None, cache_policy=None,
max_cache_bytes=None, cache_dir=None)`. Ordinary indexing uses the saved defaults.
A saved DISK preference requires a runtime cache location or the existing safe
temporary-directory convention; never reuse a persisted absolute directory.

When reached through an outer RemoteStore, the outer owner determines cache
policy and the aggregate byte budget for all linked sources and descendants.
Inner saved policies do not override it. NONE, MEMORY, DISK and shared sparse
caching must work across mounts, including CTables and their index sidecars.

Keep discovery and refresh state per source while reusing aggregate cache
coordination and traffic accounting. Namescope cache entries by source identity,
dataset and generation, not just relative leaf path. Equal leaf names in
different stores must never collide. Retained compressed data shares the outer
budget; metadata, decoded working buffers and output allocations remain separately
accounted, following existing conventions.

Credentials are runtime-only and source-specific. Add a minimal resolver keyed
by the validated source descriptor for nested access, and thread it through
reference reopening. Do not forward parent credentials or filesystem instances
to an unrelated endpoint. A resolver is not serialized. Reuse existing source
authorization hooks before contacting linked targets.

Use existing dependent-handle ownership rules: closing a parent handle does not
close resources still held by children. Release linked owners when their last
dependent closes. An outer refresh invalidates previously issued linked handles
and cached discovery consistently, without eagerly opening undiscovered targets.
Route refresh of borrowed linked handles through the outer root, matching tables.
Standalone links opened from a local tree can refresh their own remote session.

## Reference saving

`RemoteStore.save()` retains its reference-export meaning. Preserve the root
source plus every known linked-source descriptor and its mount path. A cold export
must reopen links later without losing dataset selection or source identity.
`include_cache=True` includes only already retained payload; it must not expand
unvisited links or download missing chunks. Include known linked discovery needed
to use warm caches offline. `include_cache=False` omits payload at every depth.

TreeStore assignment stores a cold reference by default; it does not implicitly
copy the assigned handle's potentially large cache. Local TreeStore packing and
format conversion preserve references. They must not silently acquire recursive
materialization semantics.

Version manifests when required by the additional source graph. Keep old artifacts
readable; reject unsupported new manifests clearly. Saved subtree exports include
only reachable mounts and their retained caches, without unrelated siblings.

## Explicit materialization

Add `materialize(destination, *, overwrite=False)` to RemoteStore and TreeStore
as an explicit operation. A `.b2z` destination produces a local archive; a `.b2d`
destination produces a local directory tree. Use existing export machinery for
ordinary leaves and CTables. This proposed API avoids changing reference saving
or TreeStore's existing format-conversion behavior.

Materialization recursively downloads the hosting tree and every reachable linked
store. Replace each reference with an ordinary structural subtree at the same
explicit path, preserving group attributes, supported arrays, tables and their
schema. External RemoteArray references must also become local data. Convert HDF5
and Zarr contents to supported Blosc2 objects; do not copy their native containers
as opaque files. The result opens as a self-contained TreeStore with networking
disabled. CTable indexes must be preserved through valid local export or rebuilt
consistently; never carry stale source-bound descriptors into the output.

Detect cycles using source kind, canonical source identity and dataset along the
active expansion chain. Repeated references in separate branches are valid and
materialize separately. Catch cycles through subgroup selection as traversal
reaches them; do not globally reject every repeated source URL. Apply a finite
depth guard as protection against aliases that cannot be canonicalized reliably.
Errors identify the mount chain. Unsupported nodes fail explicitly rather than
silently produce incomplete output.

Stream leaves with existing buffer limits instead of collecting the complete
tree in memory. Stage output and publish only after success. Errors, missing
credentials, cycles and interrupted downloads must leave existing destinations
intact and clean up owned staging resources.

## Implementation sequence and verification

1. **Persist explicit references in TreeStore.** Implement descriptor validation,
   object boundaries, assignment, deletion and local reopening for `.b2d` and
   `.b2z`. Test explicit names, subgroup targets, replacement rules, malformed
   descriptors and original-handle closure. Assert local open/list/lookup of a
   deferred handle performs no linked-source requests.
2. **Discover and traverse remote mounts.** Add the node kind, deferred discovery
   and path routing. Cover B2Z, HDF5 and Zarr targets, nested mounts, subgroup
   views, empty groups, direct/chained lookup, missing targets and unaffected
   siblings. Check browser and metadata APIs without eager target expansion.
3. **Integrate ownership and shared caching.** Add runtime option resolution,
   per-source discovery and outer aggregate cache ownership. Verify all cache
   policies, eviction across sources, identity collisions, sparse reuse, distinct
   credentials, CTable/index leaves, close ordering and refresh invalidation.
   Test local saved defaults separately from outer-owner overrides.
4. **Extend portable reference exports.** Round-trip cold and warm nested
   references, subtree exports and policy overrides. Assert saving performs no
   missing-payload fetches and unvisited links remain unopened. Test offline warm
   reads, runtime credential injection, artifact compatibility and malformed
   multi-source manifests.
5. **Implement recursive materialization.** Cover mixed-format links, multiple
   levels, subgroup mounts, tables, RemoteArray leaves, attributes, repeated
   targets, cycles and unsupported nodes. Reopen results with networking forbidden
   and compare hierarchy and values. Inject failures to verify destination safety
   and bounded streaming.
6. **Document and verify the complete feature.** Explain explicit naming, local
   versus remote cache ownership, credentials, lazy discovery, save versus
   materialize and cycle handling. Add one small example. Run focused TreeStore,
   RemoteStore, RemoteArray and CTable tests, Ruff and documentation checks, then
   the full default suite in the `blosc2` conda environment. Record actual commands,
   results and remaining limitations in this plan.

Use deterministic local fixtures and instrumented transport for request assertions;
do not depend on public remote services. Reuse existing optional HDF5 and Zarr test
fixtures. Keep format-reader regressions and ordinary same-source subgroup behavior
covered while adding cross-source ownership.

## Implementation record

The six implementation points were completed in order:

1. `f4dc723c` — Persist RemoteStore references in TreeStore.
2. `b55b70ef` — Traverse nested RemoteStore references.
3. `667c7a62` — Share cache ownership with nested stores.
4. `fcc8513d` — Preserve nested stores in reference exports.
5. `2aed755c` — Materialize nested RemoteStore trees.
6. `31ea5af7` — Document nested RemoteStore references.

The implementation stores validated, credential-free descriptors in the existing
TreeStore object registry. Local lookup creates a deferred handle, while remote
B2Z discovery exposes a `remote_store` boundary and routes path suffixes through
the mounted owner. B2Z, HDF5, Zarr v2 and Zarr v3 mounts use their existing
readers. Nested owners share the outer cache policy, coordinator, traffic counter
and aggregate allowance; source-specific namespaces prevent equal leaf paths from
colliding. Runtime credentials can be supplied per source.

Cold exports retain descriptors without opening targets. Warm exports recursively
include only metadata and payload already retained by opened targets. Recursive
materialization writes `.b2z` or `.b2d`, copies array data by first-axis chunk
slabs, rebuilds CTable indexes, accepts repeated targets in separate branches,
detects active-chain cycles, enforces a depth limit of 64 and publishes only after
the staged tree succeeds.

Verification in the `blosc2` conda environment:

- `pytest -q tests/test_remote_store.py tests/test_tree_store.py
  tests/ctable/test_remote_ctable.py tests/ctable/test_ctable_indexing.py`:
  **496 passed, 6 skipped**.
- Ruff on every changed Python file: **passed**.
- `python -m sphinx -b html doc /tmp/python-blosc2-nested-store-docs`:
  **passed**; the build retained the repository's existing documentation warnings.
- `pytest -q`: **10,392 passed, 36 skipped**.

Remaining limits are deliberate: references are read-only and assume immutable
sources; credentials remain runtime-only; cold artifacts still need their remote
sources; a saved DISK preference opened directly from a local TreeStore needs an
explicit runtime `cache_dir`; repeated targets are materialized independently;
and materialization supports only node kinds already readable as Blosc2 arrays,
CTables or groups. There is no automatic mount naming, remote transaction layer,
content deduplication or source-change monitor.

## Out of scope

Archives physically embedded inside other archives, automatic mount naming,
remote writes, automatic source-change detection, cross-store transactions and
content deduplication of materialized repeated targets. No new format readers,
general virtual-filesystem framework or remote query service is required.
