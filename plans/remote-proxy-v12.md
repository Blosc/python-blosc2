# RemoteProxy v12: remote hierarchy browsing in b2view

Status: implemented and validated on 2026-09-09.

## Goal

Browse a remote container from its root, expand groups, inspect attributes, and
preview selected arrays without downloading the complete container first:

```sh
b2view --profile blosc2 --endpoint-url https://s3.us-west-001.backblazeb2.com s3://blosc2/hierarchy.b2z
b2view --profile blosc2 --endpoint-url https://s3.us-west-001.backblazeb2.com s3://blosc2/hierarchy.zarr
b2view --profile blosc2 --endpoint-url https://s3.us-west-001.backblazeb2.com s3://blosc2/hierarchy.h5
```

Support a URL pointing at a subgroup as well as the container root. Preserve the
existing standalone presentation when the URL selects an array: no tree panel,
no redundant internal root path, and the original source in the header.

All three formats are feasible using existing dependencies and leaf readers.
B2Z needs archive discovery; Zarr needs group discovery; HDF5 needs its existing
Kerchunk references retained and reused across dataset selections. The main
shared work is connecting discovery to the browser without opening every leaf.

## Current behavior and reusable pieces

- `b2view/model.py:StoreBrowser` calls `blosc2.open()` and treats only a
  `TreeStore` as a hierarchy. Its previews already handle remote array objects.
- `StoreBrowser.list_children()` queries descendants and opens terminal nodes
  to classify them. That is unsuitable for remote discovery and also cannot
  use the absence of descendants to distinguish an empty group from an array.
- Group metadata currently reads the store's attributes, rather than necessarily
  the selected group's attributes. A remote adapter must return attributes for
  the actual selected node.
- `b2view/app.py` already displays trees, expands nodes on demand, navigates to
  an initial path, and hides the tree for standalone objects. Opening, listing,
  and panel updates currently include synchronous calls on the UI thread.
- `core.py:parse_container_url()` handles container and dataset addressing.
  Reuse it, including its distinction between dataset syntax and chained fsspec
  URLs; do not add another suffix parser in the application.
- A whole remote B2Z currently bypasses lazy opening in `StoreBrowser`. Without
  a local cache, the generic fsspec path in `schunk.py` reads the entire object
  and passes ZIP bytes to `from_cframe()`. This is the wrong opening path.
- `b2z_source.py:B2ZNDSource` already reads the ZIP directory through a bounded,
  seekable range view and opens external NDArray members lazily. It validates
  member windows and uses native Blosc2 chunk/block reads. It requires a leaf
  path and rejects embedded leaves and object carriers.
- `dict_store.py` defines canonical external member names and logical key
  mapping. TreeStore also uses `embed.b2e` for information that cannot be
  reconstructed from external member names alone.
- `zarr_source.py:ZarrNDSource` already opens an fsspec-backed Zarr array and
  exposes its attributes. It explicitly rejects groups.
- `hdf5_source.py:available_datasets()` and `HDF5NDSource` already translate HDF5
  into Kerchunk references. The source accepts an existing reference dictionary,
  recognizes `.zgroup` and `.zarray`, and opens arrays through a reference store.
  Repeating translation for each selected dataset would waste substantial work.
- The recent b2view Zarr fix initializes the Numcodecs Blosc mutex before
  Textual captures stderr. Group-root startup must preserve this behavior, and
  HDF5 leaves using that decoder need the same fresh-process check.

## Scope and decisions

1. Deliver read-only b2view browsing first. Keep `RemoteProxy` an array operand;
   do not make it represent groups or implement mutable remote TreeStore.
2. Use a small internal hierarchy adapter, shared by the three remote formats.
   Do not subclass `TreeStore`: its local storage and mutation contracts do not
   describe these sources. Do not introduce a plugin registry or public store
   API for this release.
3. Keep existing `blosc2.open(..., lazy=True, dataset=...)` array behavior.
   b2view explicitly selects the hierarchy adapter for container/group targets.
   The erroneous generic B2Z root path also needs a shared dispatch guard so
   other callers receive a useful error instead of a full download followed by
   a frame-decoding failure. Existing explicit `cache_dir` localization remains
   available; this plan does not make it b2view's default.
4. Opening and expanding a hierarchy may read metadata and list object keys.
   It must not materialize every dataset or fetch array payloads merely to
   classify children. Metadata cost can scale with container size; it is not
   necessarily a constant number of bytes or requests.
5. Preview support matches the existing leaf readers. Unsupported datasets
   remain visible with an explanatory message; one unsupported leaf must not
   prevent browsing its siblings.
6. Keep remote sources immutable for a browsing session. Explicit refresh
   rebuilds discovery state and invalidates cached node metadata and leaf
   objects. Live mutation detection and persistent hierarchy caches are deferred.
7. Use the existing optional fsspec, Zarr, and HDF5/Kerchunk dependency groups.
   B2Z browsing must work without Zarr, h5py, or Kerchunk installed.

## Browser integration

Introduce one internal module, tentatively `b2view/hierarchy.py`, for discovery
and format adapters. Keep format-specific range/translation helpers in their
existing source modules when also used by array readers.

The browser needs only these operations:

- Resolve whether the requested node is a group, supported array, or unsupported
  object, independently of whether it has children.
- List direct children with their logical path, display name, kind, and expansion
  capability. Reuse `NodeInfo` where practical; do not load data for classification.
- Read one node's metadata and user attributes.
- Open a selected supported leaf using the existing array source/proxy machinery.
- Close resources and rebuild the session on refresh.

Keep these operations internal and concrete; a base class is unnecessary unless
implementation reveals meaningful shared behavior. `StoreBrowser` remains the
UI-facing adapter and routes hierarchy operations through this small surface.
Audit all uses of `self.store` and `is_tree`, especially `_get_object()`, `kind()`,
`get_info()`, `list_children()`, and `close()`. Existing local TreeStore and CTable
query behavior must remain intact.

Separate node kind from expansion state. An empty group is still a group and
shows its own attributes. For a remote group whose children have not been
listed, allow expansion without recursively probing its descendants. Omit the
descendant count when obtaining it would require walking the whole hierarchy;
show direct-child counts after listing, rather than inventing a zero count.

Use browser-relative paths for subtree views: `/` is the requested group, and
leaf resolution joins that view to the actual container path. The header keeps
the supplied source URL; the metadata path identifies the selected node within
the view. Handle root, trailing slash, subgroup, and direct-array URLs explicitly.
Preserve query strings, credentials, and storage options when resolving leaves;
never build child URLs by appending strings after a URL query or fragment.

Retain discovery metadata for the session. Reuse the existing current leaf proxy
on repeated panel reads; avoid retaining an unbounded number of payload caches
while traversing a large tree. Start with the selected leaf and release it when
selection changes, unless existing cache ownership already provides a bound.

## B2Z discovery

### First milestone: external arrays and inferred groups

Extract only the archive-opening/index functionality needed by both discovery
and `B2ZNDSource`. Preserve bounded range reads, ZIP64 support, malformed-header
checks, member validation, traffic accounting, and existing leaf cache identity.
Use `zipfile`, not a new ZIP parser. Share archive identity and directory results
within the browser session rather than rediscovering them on every click.

Map canonical member names to logical keys using DictStore's existing rules.
Construct parent groups from the paths, hide storage implementation members,
and list direct children without opening the corresponding frames. Resolve a
selected external NDArray through the existing B2Z leaf reader. Validate the
selected frame before preview; a `.b2nd` suffix is not proof of a plain NDArray.

Reject ambiguous duplicate logical keys, unsafe paths, and leaf/group collisions
with an actionable archive error. Preserve the existing reader's rejection of
encrypted or ZIP-compressed array members. Surface unsupported ordinary object
types as unavailable leaves where they can be identified reliably.

This milestone is sufficient for the supplied hierarchy if it contains ordinary
external arrays. Verify its actual directory before claiming that coverage.

### Complete the hierarchy's metadata

Inspect the actual `embed.b2e` layout and TreeStore subtree metadata conventions
before implementation. External filenames alone do not establish empty groups,
embedded leaves, group attributes, or logical CTable object boundaries.

Reuse EmbedStore decoding to read the index and group metadata required for
discovery. Determine whether these can be read through a bounded member window
without loading the entire embedded payload. Do not assume `embed.b2e` is small
or silently download it wholesale. Record the measured access pattern in tests.

Show explicit empty groups and the correct root/subgroup attributes when the
stored format records them. If the format does not persist empty groups, document
that limitation rather than synthesizing them. Identify embedded objects and
CTable roots from their metadata and show an unsupported-preview message; do not
expose CTable column carriers as if they were an ordinary user group.

If bounded embedded-index access needs a larger storage-layer change, deliver the
external-array milestone with a clear partial-discovery indication and document
the missing metadata. Do not call that full B2Z hierarchy support. Remote embedded
payload and CTable preview support are separate follow-up work.

## Zarr discovery

Open the requested node through Zarr's existing read-only fsspec store path and
distinguish arrays from groups using node metadata. Reuse that store and its
filesystem across navigation; keep leaf reads in `ZarrNDSource` and the existing
proxy implementation.

Use Zarr's supported group/member APIs for immediate children and group attributes.
Verify the exact installed API and supported dependency versions during
implementation. Support both Zarr v2 and v3 storage layouts already covered by
the array reader. Use consolidated metadata when available and supported; fall
back to normal metadata discovery without requiring users to consolidate stores.

Avoid recursive enumeration of chunk objects. An unconsolidated remote store may
require LIST requests and metadata requests for child nodes, and some backends
cannot list at all. Report missing listing capability/permission clearly. Direct
array URLs should keep working when the user can read an array but cannot list
its parent. Preserve empty groups and per-group attributes.

List datasets with unsupported dtypes/codecs, but defer decoding to selection and
show the source reader's specific limitation. Do not create another Zarr decoder
or promise finer data-fetch granularity than the current reader provides.

## HDF5 discovery

Translate the file to Kerchunk references once per browsing session, reusing the
existing translation and filter-registration path. Factor the shared operation
out of `available_datasets()` and `_load_or_scan_refs()` only as needed; do not
add an independent HDF5 traversal implementation inside b2view.

Build the hierarchy from `.zgroup`, `.zarray`, and `.zattrs` entries, including
explicit empty groups. Reuse those references when opening the selected dataset
through `HDF5NDSource` or the existing proxy path that accepts `refs`. Passing a
fresh URL alone must not trigger another translation for each leaf. Preserve
dataset identity and original source information when using the shared refs.

Display root and group attributes from their reference metadata, filtering only
known adapter bookkeeping, consistently with existing leaf attributes. Measure
the initial scan: HDF5 translation can visit metadata across the file, enumerate
many chunk references, and inline some small values. Promise no full-file
localization, not zero payload bytes or metadata cost independent of file size.

Check how the installed translator handles unsupported datasets, hard-link
aliases, soft/external links, and cycles. Do not recursively follow arbitrary
links or contact additional external sources as part of listing. Prefer visible
unsupported nodes where the translator supplies them. If translation omits an
object or aborts the whole scan, expose/document that limitation; isolating
unsupported siblings is a completion requirement for the supported test matrix.
Do not describe the resulting view as covering every HDF5 object type.

## UI responsiveness, failures, and lifecycle

Run remote opening, expansion, metadata loading, and array previews through
Textual workers, following existing worker patterns in the app. Keep local-only
operations simple. Capture selection/slice state before dispatch and apply results
on the UI thread only if they still match the active request and browser session.

Provide loading state for startup and expansion, keep navigation and quit usable,
and make failed group listings retryable. Do not add paths to `loaded_paths` until
listing succeeds. A stale worker must not repaint an old selection or repopulate
a tree after refresh. Cancellation may not interrupt an underlying blocking read;
discard stale results and close their resources after that read completes.

Refresh must replace the discovery session, clear loaded paths, discard old leaf
caches, and attempt to restore the selected logical path, falling back to the
view root if it vanished. Ensure normal shutdown, failed startup, and refresh
close owned handles without closing a filesystem still in use by another worker.
Keep credentials in runtime storage options and out of errors or serialized state.

## Implementation order and checkpoints

1. **Reproduce and characterize.** Add a small valid in-memory B2Z root case that
   demonstrates the bad dispatch, and inspect representative fixtures for all
   three formats. Confirm embedded metadata and HDF5 translation behavior.
2. **B2Z external-array vertical slice.** Implement shared archive discovery,
   minimal browser routing, root/subgroup navigation, and selected-array preview.
   Guard the generic non-lazy B2Z path and preserve explicit local caching.
3. **Finish browser behavior.** Add remote workers, stale-result handling, refresh,
   failures, correct node attributes, and empty-group behavior. Complete the B2Z
   metadata milestone or explicitly mark its bounded-index blocker.
4. **Zarr groups.** Add discovery through native group APIs and reuse leaf readers.
   Exercise v2/v3, consolidated/unconsolidated metadata, and restricted listing.
5. **HDF5 groups.** Reuse one translation and reference dictionary across selections;
   verify attributes, empty groups, supported filters, and unsupported objects.
6. **Validate and document.** Run the targeted suite and real remote commands,
   record traffic and startup observations, and update user-facing examples.

The earlier one-to-two-day estimate applies only to the narrow B2Z external-array
browser. It does not cover all three formats, complete embedded metadata support,
or the UI lifecycle work above. Re-estimate after the first checkpoint resolves
the embedded-index and translator limitations.

## Validation

Extend existing pytest modules and use local temporary containers plus fsspec's
memory filesystem for deterministic checks. Use a counting filesystem/store to
test actual reads, not merely calls to adapter methods.

- Root and subgroup discovery returns stable direct children with correct kinds;
  empty groups remain groups. Selected group attributes differ from root attrs.
- Equivalent small B2Z, Zarr, and HDF5 arrays produce identical bounded previews
  through root navigation and direct leaf URLs, including slicing and paging.
- B2Z startup does not request the complete archive or read all array frames.
  Selecting one array does not fetch sibling payloads. Cover large ZIP directories,
  malformed/duplicate entries, object carriers, and embedded metadata boundaries.
- Zarr discovery does not decode chunks; cover v2/v3, consolidated metadata,
  unconsolidated metadata, listing failure, empty groups, and unsupported leaves.
- HDF5 translation runs once per session, not per leaf; test multiple selections,
  group attributes, empty groups, unsupported/link cases, and refresh translation.
- Root-open dispatch errors occur before a bulk download. Explicit cached B2Z
  opening and existing direct remote leaf opening retain their behavior.
- Fresh-process TUI tests cover compressed Zarr and HDF5 leaves reached from a
  group root, preserving the decoder-mutex regression check.
- Headless TUI tests verify tree visibility, keyboard focus, expansion, initial
  path navigation, selected path display, refresh, quit, and retry after failure.
  Use controlled slow reads to verify that stale results cannot repaint a newer
  selection or session and that shutdown does not race resource cleanup.
- Existing local TreeStore, standalone NDArray/CTable, and remote leaf tests pass.
  Verify optional dependency isolation so B2Z does not acquire Zarr/HDF5 imports.

Run Python and all tests in the `blosc2` conda environment. A starting focused run:

```sh
conda run --no-capture-output -n blosc2 pytest tests/b2view tests/test_b2z_source.py tests/test_zarr_source.py tests/test_hdf5_source.py -m 'not network and not heavy' -q
```

Include new discovery tests and affected generic-open/remote-proxy tests in that
run. Explicitly include TUI markers, which the default configuration excludes.
Run Ruff and whitespace checks on the changed files; broaden testing when shared
open/source code changes warrant it.

Finally run the three commands in Goal against available remote fixtures with
the supplied profile/endpoint. For each, verify root, nested group, array values,
attributes, refresh, and direct leaf behavior. Record first-open metadata bytes,
request counts, first-preview bytes, and repeated-preview behavior separately.
Keep credentials out of fixtures and logs. Do not claim support based only on a
mocked tree or one successful array read.

## Documentation and completion criteria

Update `doc/guides/b2view.rst` with remote root/subgroup examples, dependency
requirements, listing permissions, read-only behavior, and format limitations.
Clarify that b2view's hierarchy adapter does not change the array-only contract
of lazy `blosc2.open()` or introduce a persisted RemoteProxy hierarchy descriptor.

The full v12 plan is complete when all three supported hierarchy views navigate
correctly, selected arrays reuse existing lazy readers, metadata and unsupported
nodes are represented honestly, UI operations remain responsive, and the tests
above pass. A delivered B2Z external-array milestone should be labeled as such
until the remaining metadata work is complete.

Deferred: remote writes, full remote TreeStore/CTable semantics, embedded B2Z
payload previews, cross-container links, recursive search, persistent hierarchy
indexes, live watching, and support for additional codecs/dtypes beyond the leaf
readers. These are not prerequisites for the three-format read-only browser.

## Implementation record

The remote fixtures in the goal were exercised through the headless b2view CLI
from the root and a subgroup, and direct array URLs were exercised through
``StoreBrowser``. Nested navigation, attributes, array values, refresh, and quit
passed for B2Z, Zarr, and HDF5. The measured root opens were:

| Format | Initial metadata bytes | Initial S3 operations | First preview bytes | Repeated preview bytes |
| --- | ---: | --- | ---: | ---: |
| B2Z | 8,192 | 1 HEAD, 1 GET | 32,225 | 0 |
| Zarr | 132 | 7 HEAD, 5 LIST, 2 GET | 12,420 | 0 |
| HDF5 | 52,976 | 1 HEAD, 54 GET | 13,662 | 0 |

Zarr discovery grew to 6,515 bytes after expanding through ``/d0/d1``. HDF5
translation remained at 52,976 bytes through hierarchy expansion because its
references were built once at session start. The request counts reflect the
installed s3fs/Zarr/Kerchunk versions and the fixture's current metadata layout;
they are observations rather than API guarantees.

B2Z embedded arrays and remote CTable previews remain unavailable as planned.
Embedded group attributes are read with bounded native chunk access; layouts
whose metadata lives in native chunks larger than 1 MiB show an explicit partial
metadata notice. Soft and external HDF5 links and group cycles are not followed.
