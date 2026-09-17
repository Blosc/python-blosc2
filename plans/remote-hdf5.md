# Native remote HDF5 reader

## Status and objective

Implemented on the current `remote-hdf5` branch. This document records the
architecture, compatibility policy, and verification scope.

Replace the former HDF5 reference-based access with a native metadata index and range reader.
Keep h5py for metadata discovery and a compatibility fallback, fsspec for remote
I/O, and the existing Blosc2 chunk cache. Neither the legacy HDF5 reference layer nor Zarr should be
required or imported by the new HDF5 path.

Agreed decoding policy:

- Direct reads for uncompressed chunked datasets and pipelines composed entirely
  of HDF5 deflate (filter 1), HDF5 shuffle (filter 2), and Blosc2 (filter 32026).
- Use `h5py.Dataset[selection]` for other filter pipelines, with `hdf5plugin`
  supplying registered HDF5 filters where needed.
- Select the path for the complete dataset filter pipeline. An unsupported
  filter selects the fallback even if some chunks skip that filter.
- Keep fallback handles open across chunk reads so HDF5 can reuse metadata.
- Both paths produce the same logical Blosc2 cache chunks.

No new Python source modules or mandatory codec packages are needed. Implementation
belongs primarily in `src/blosc2/hdf5_source.py`, with integration in the existing
remote array/store modules. NumPy, Blosc2, and standard-library `zlib` provide
the direct decoding primitives. Protocol-specific fsspec backends remain optional.

## Scope and explicit limits

Preserve the public HDF5 URL syntax, dataset selection, immutable-source contract,
attributes, lazy expressions, references, cache policies, and hierarchy browsing.
Preserve direct local h5py access without fsspec or Zarr dependencies.

Support fixed-size dtypes already accepted by `HDF5NDSource`; reject object and
variable-length data, null dataspaces, and oversized logical chunks clearly.
Check structured dtypes and non-native byte order explicitly, rather than assuming
that `dtype.str` fully describes every accepted dtype.

Initial layout policy:

- Chunked, self-contained datasets: native index and the pipeline policy above.
- Contiguous and compact datasets: h5py slicing with bounded logical Blosc2
  chunks, including scalars and zero-length axes. This keeps arbitrary
  multidimensional contiguous slicing out of the first range-reader implementation.
- External raw storage and virtual datasets: mark unsupported initially; do not
  follow additional files silently.
- Preserve existing link traversal behavior: ordinary object traversal, with
  external and soft links omitted and hard-link aliases not promised as separate
  leaves. Test the behavior and state it in hierarchy notices.

Out of scope: writing remote HDF5, changing source mutation semantics, parsing
HDF5 structures ourselves, general legacy HDF5 reference execution, multi-file
aggregation, standalone LZF/Blosc1/bitshuffle decoders, in-memory HDF5 chunk
injection, and block-level reads within Blosc2-filter HDF5 chunks.

## Current integration points

| File | Required change |
| --- | --- |
| `src/blosc2/hdf5_source.py` | Replace translation, Zarr-backed arrays, filter monkey-patches, and sync-loop workarounds with indexing, direct decoding, and fallback ownership. |
| `src/blosc2/remote_array.py` | Persist native indexes, share container snapshots between leaves, restore carriers, and handle legacy metadata explicitly. |
| `src/blosc2/remote_store.py` | Discover nodes from native metadata, validate/restore manifests, share the index, and close HDF5 resources correctly. |
| `src/blosc2/remote_store_cache.py` | Adjust manifest compatibility checks only if the HDF5 metadata discriminator cannot be handled in the owner. |
| `src/blosc2/zarr_source.py` | Keep Zarr behavior intact; remove HDF5's reliance on its sync lock and conversion helper. |
| `pyproject.toml` | Remove the legacy HDF5 reference layer from HDF5, development, and test dependencies; retain Zarr for actual Zarr support. |
| Existing HDF5/store/fsspec/b2view tests | Replace legacy reference-layer-specific fixtures and assertions with behavioral checks. |
| HDF5 docs and examples | Document native indexing, fallback filters, dependencies, and legacy reference handling. |

In particular, `RemoteStore._close_resources()` currently assumes every HDF5
source has `source.array.store.close()`. Replace this with an explicit HDF5 source
resource lifecycle. `RemoteArray.close()` currently only releases store-derived
handles; do not assume it already closes standalone HDF5 sources.

Use non-underscored names for helpers imported between modules. Tentative shared
entry points are `scan_hdf5_index()` and `validate_hdf5_index()`; private decoding
and serialization helpers remain in `hdf5_source.py`.

## 1. Native metadata format and scanner

Introduce a JSON-compatible, versioned container snapshot with an unambiguous
format discriminator, for example `format: "blosc2-hdf5-index", version: 1`.
Store one source identity and file size at container level; chunk records contain
offsets and lengths, not arbitrary remote URLs.

Each dataset entry needs:

- Dataset path, shape, complete dtype description, physical HDF5 layout/chunks,
  fill value, and attributes.
- Ordered filter records from the dataset creation property list: filter ID,
  flags, and client-data parameters. Do not use private `Dataset._filters`.
- Allocated chunk records: element coordinates, file byte offset, stored byte
  length, and per-chunk filter mask. Omitted logical chunks mean unallocated
  storage and must produce the HDF5 fill value.
- Enough layout information to select direct reads or fallback at runtime.
  Availability of installed plugins is runtime state, not a persisted guarantee.

Store group entries and isolated unsupported-leaf descriptions as well. Keep
physical metadata independent of caller-selected Blosc2 blocks and compression
parameters so one index serves different array carriers.

Use tagged encodings for bytes, NumPy scalars/arrays, structured dtype
descriptions, and fill values where ordinary JSON would lose information. Specify
round trips for NaNs, complex values, fixed-size strings, and structured values.
Do not use pickle. Reuse suitable existing serialization helpers after inspection.

Scanner procedure:

1. Resolve/reuse the supplied filesystem and storage options. Open a seekable
   binary handle with read-ahead disabled, following the existing HTTP convention
   `block_size=1, cache_type="none"`.
2. Wrap reads for traffic accounting and open h5py read-only with scoped ownership.
3. Traverse groups/datasets; extract metadata through public h5py APIs.
4. Enumerate allocated chunks using `get_num_chunks()`/`get_chunk_info()` or a
   supported efficient iterator. Confirm enumeration is not quadratic on large
   indexes; record the minimum h5py version if relying on a newer API.
5. Never decompress dataset payloads while indexing chunked datasets. Attribute
   and compact-layout metadata can themselves contain inline values.
6. Close HDF5 before the underlying file object, including all exception paths.

Scan once per container and share the snapshot, preserving current semantics.
An all-fallback dataset does not need a complete chunk offset table; avoid
enumerating it if its metadata is sufficient for later h5py slicing.

Validate snapshots before use: version, dataset paths, dtype/shape consistency,
chunk alignment and bounds, duplicate coordinates, nonnegative integer offsets
and sizes (excluding booleans), known filter-mask bits, and ranges within the
recorded file size. Bind the snapshot to the requested source identity and retain
existing credential/persistable-URL rules. Unknown versions must fail clearly.

## 2. Direct range reader and decoding

For each requested logical chunk:

1. Validate the chunk number and translate it into the HDF5 chunk coordinates.
2. If no allocated record exists, construct values from the dataset fill value
   without a remote read. Never mistake an absent index table for missing chunks.
3. Fetch exactly `[offset, offset + size)` using the shared filesystem's range
   API. Avoid concurrent `seek`/`read` on one mutable file cursor. Honor the
   existing concurrency limit and async filesystem ownership conventions.
4. Check for short reads. Decode filters in reverse creation order, skipping a
   filter when its corresponding mask bit is set.
5. Validate the decoded byte count and interpret using the stored dtype and
   physical chunk shape. Clip to the valid edge extent, then construct the
   bounded logical cache chunk with the existing edge-padding convention.
6. Recompress into the caller's Blosc2 chunks/blocks/cparams and return the chunk
   bytes through the existing source protocol.

Decoder details:

- Deflate: HDF5's zlib stream through `zlib`, with bounded expected output and
  explicit truncated/corrupt-stream errors.
- Shuffle: reverse the HDF5 byte-plane shuffle using its element-size parameter;
  handle itemsize 1 and validate buffer divisibility. This is separate from the
  internal shuffle/bitshuffle used by a Blosc2 frame.
- Blosc2: preserve support for both ordinary compressed buffers and super-chunk
  frames. Normalize `SChunk` bytes and any `NDArray` frame results correctly;
  use logical values for shaped frames rather than assuming their internal block
  ordering is NumPy C order. Test with real `hdf5plugin.Blosc2` outputs as well as
  manually written raw chunks.

Do not fall back silently after a known direct decoder reports corrupt data or
an inconsistent index. Fallback is a planned capability decision, not recovery
from arbitrary exceptions. Include dataset and chunk coordinates in failures.

Keep conversion in `hdf5_source.py` initially. It must not enter `ZARR_SYNC_LOCK`.
Do not return the original Blosc2 frame as a cache chunk without conversion:
physical framing and requested logical cache geometry may differ.

## 3. h5py fallback and resource lifecycle

Open the fallback HDF5 file lazily on the first uncached read that needs it.
Use h5py slicing for the requested logical chunk; load `hdf5plugin` opportunistically
so installed plugin filters register with HDF5. Report unavailable filter IDs
and installation guidance without excluding readable siblings from a store.

The same conversion code should normalize both direct and fallback values.
Cache hits, including persisted warm chunks, must not open the fallback file.
Fallback opening may reread HDF5 metadata even when the native snapshot is warm;
the snapshot is not an HDF5 metadata-cache image.

Ownership requirements:

- Standalone source: own the fallback handle, provide idempotent source cleanup,
  and retain a finalizer for abandoned sources. Define how explicit standalone
  `RemoteArray.close()` reaches this cleanup without changing unrelated sources.
- Store: share one lazily initialized fallback reader per container owner where
  practical, so sibling leaves reuse metadata. Leaf closure must not close a
  reader still used by another leaf. Owner close/refresh releases it once.
- Protect lazy initialization and close-versus-read races. Close h5py first,
  then its file object, and only close filesystem sessions owned by this reader.
- Preserve generation checks and existing stale/closed-handle errors.

h5py holds its global HDF5 lock during fallback network reads and decoding.
Concurrent fallback reads are therefore serialized; `max_concurrency` does not
promise parallel h5py I/O. Direct range requests and standalone decoders must
remain outside that lock and outside any new container-wide I/O lock. Reuse the
current fetch scheduling rather than adding a second thread pool per source.

Measure traffic at one I/O boundary per path. With read-ahead disabled, h5py's
file read callbacks can be charged directly. Avoid counting an already counted
filesystem read again. Keep the established exclusion of non-payload `info`/HEAD
calls, and distinguish transport counts from h5py callback counts in benchmarks.

## 4. Persistence, hierarchy integration, and compatibility

Persist the native snapshot in carriers and in the shared container cache so
reopening a direct dataset needs no h5py rescan. Prefer a new `hdf5-index` vlmeta
key and `.hdf5-index.b2` sidecar rather than disguising the new format as a
legacy HDF5 reference dictionary. Update saving, loading, cloning, cache export,
lazy-expression reopening, and shared-index publication together.

`RemoteStore` should build its node tree directly from snapshot groups and
datasets, replacing `.zarray`/`.zgroup` parsing. Keep one snapshot in the owner
and pass it to leaves. Restore and validate that format in manifests; update
the user-facing hierarchy notice and b2view integration.

Compatibility policy for this implementation:

- Keep the existing `hdf5_index=` argument as an input slot for a native snapshot or
  its JSON path, with its new accepted format documented. Do not add another
  public keyword merely to rename it in this change.
- Explicit legacy HDF5 reference dictionaries/JSON files: detect and reject with a
  precise migration message explaining how to omit `hdf5_index` and rescan the source.
  Do not silently reinterpret arbitrary multi-file reference maps or fetch their
  targets. This is a documented compatibility break.
- Implicit legacy snapshots found in reusable array caches: rescan the original
  HDF5 source and check geometry, dtype, and conversion identity before retaining
  any warm chunks. A failed rescan or mismatch must not relabel old cached data
  as valid. Document that this migration requires source access.
- Legacy store manifests: report the established incompatible-cache error and
  advise a new cache directory; do not silently delete or rewrite generations.
- Version native metadata independently of logical chunk encoding. Retain the
  existing encoding stamp only where conversion semantics and geometry are
  demonstrably identical; otherwise bump it and reject incompatible cache reuse.
  Correct structured-dtype identity at the same boundary if needed.

Retire the old internal scanner after updating callers. The public
`scan_hdf5_index()` helper returns the native format. `available_datasets()` must
accept native indexes and retain direct local-file discovery.

## 5. Dependencies and removal of workarounds

- Remove the legacy HDF5 reference layer from the `hdf5` extra and development/test dependency groups.
- Retain `h5py` and the existing convenience installation of `hdf5plugin` in the
  HDF5 extra; runtime plugin use remains optional. Keep fsspec in its existing
  extra, so remote installation remains `blosc2[hdf5,fsspec]`.
- Ensure CI installs `hdf5plugin` for actual plugin-filter tests, rather than
  allowing all such tests to skip. Keep platform markers consistent with h5py.
- Remove HDF5's numcodecs registration, legacy HDF5 filter monkey-patch, Zarr store adapter,
  recursive Buffer conversion, scan timeout retries, and global Zarr-loop reset.
- Keep Zarr dependencies and sync machinery needed by genuine Zarr readers.
- Test HDF5 functionality with the legacy HDF5 reference layer, Zarr, and numcodecs imports blocked;
  test package import without any HDF5 optional dependencies.

Update `doc/guides/remote_arrays.md`, `doc/reference/remotearray.rst`,
`doc/getting_started/installation.rst`, `doc/guides/b2view.rst`, HDF5 source
docstrings, and relevant examples. Explain pipeline selection, warm snapshots,
fallback locking, and legacy migration. Do not describe installing `hdf5plugin`
as supplying standalone decoders.

## 6. Verification matrix

Extend existing test modules; no new source modules are required.

### Deterministic offline correctness

Use generated HDF5 fixtures on `memory://` and the existing local HTTP range
server. Compare complete results and slices against NumPy/h5py ground truth.

- Direct pipelines: no filters, shuffle only, deflate only, shuffle + deflate,
  Blosc2, and valid combined pipelines. Include big-endian and fixed-size string
  dtypes, supported compound dtypes, and itemsize 1.
- Per-chunk masks: use raw chunk writes to construct chunks that skip individual
  filters; validate order and masks independently of high-level writer defaults.
- Sparse allocation: missing chunks, custom/nonzero/NaN fill values, fully
  unallocated datasets, partial edge chunks, and empty axes.
- Local and fallback layouts: scalars, contiguous arrays, compact storage, and
  bounded conversion of a large contiguous array.
- Blosc2: ordinary compressed buffers, SChunk frames, shaped frames where
  supported, and plugin-generated multidimensional edge chunks.
- Fallback: LZF where available, Fletcher32 pipelines, scale-offset, and at least
  one `hdf5plugin` filter outside the direct set (such as Blosc1 or bitshuffle).
  Verify every requested chunk uses h5py and the handle is reused.
- Failures: unsupported/null/object datasets, missing plugins, corrupt streams,
  short ranges, invalid chunk metadata, and malformed/unknown-version snapshots.

### Cache, ownership, and concurrency

- Native snapshot round trips preserve dtype, fill, attrs, filters, and offsets.
- Cold scan once per container; opening sibling leaves does not rescan.
- Persisted direct snapshots reopen without scanning; warm cached chunks require
  no data requests. An uncached allocated direct chunk makes one range fetch.
- Missing chunks make no range fetch. Fallback metadata reads occur only when
  needed, and repeated fallback access reuses the handle.
- Assert overlapping direct requests using synchronization barriers in a fake
  range backend, rather than brittle wall-clock thresholds. Verify the fetch
  cap and concurrent correctness without involving h5py in direct decoding.
- Source/store close, partial initialization failures, refresh, stale handles,
  finalizers, sibling ownership, and externally owned filesystem lifetimes.
- Native/legacy carrier cases, explicit legacy indexes, store manifests, shared
  sidecars, saved expressions, and cache export/reopen.
- Keep HTTP range behavior and moto/S3 coverage: a memory filesystem alone does
  not exercise fsspec's real async transport/session ownership.
- Replace tests asserting HDF5 reference translation counts with scanner counts, and
  remove tests whose only purpose was resetting Zarr resources for HDF5 scans.

Relevant suites: `tests/test_hdf5_source.py`, `tests/test_remote_store.py`,
`tests/test_fsspec.py`, `tests/test_fsspec_s3.py`, `tests/test_remote_array.py`,
and `tests/b2view/test_hierarchy.py`. Also run Zarr regression tests because the
old conversion helper and resource assumptions are shared.

### Public HTTPS fixture

Use the credential-free URL supplied for this work:

`https://f001.backblazeb2.com/file/blosc2/hierarchy.h5`

Metadata inspected on 2026-09-17: file size 1,037,240 bytes. The groups `d0`,
`d0/d1`, and `d0/d1/d2` each contain:

| Leaf | Shape | dtype | HDF5 chunks | Filters |
| --- | --- | --- | --- | --- |
| `a0` | `()` | int32 | contiguous | none |
| `a1` | `(10000,)` | float32 | `(2500,)` | Blosc2, ID 32026 |
| `a2` | `(1000, 1000)` | int32 | `(500, 500)` | Blosc2, ID 32026 |
| `a3` | `(10, 1000, 1000)` | int32 | `(2, 500, 500)` | Blosc2, ID 32026 |

Verified h5py sample reads: `d0/a0` is 0, and both `d0/d1/a2[:3, :3]`
and `d0/d1/d2/a3[0, :3, :3]` return rows `[0, 1, 2]`,
`[1000, 1001, 1002]`, and `[2000, 2001, 2002]`. The first raw chunk of
`d0/d1/a2`, fetched separately by byte range, opens with `blosc2.from_cframe()`
as an `NDArray` of shape `(500, 500)`. This is a concrete shaped-frame decoder
case, not just a hypothetical extension. These checks validate the fixture and
primitives; they do not constitute a test of the proposed reader.

Existing S3 tests assume `d0/d1/a2` is 3-D. Do not copy that assumption to the
HTTPS tests, or assume the two endpoints necessarily serve identical versions.

Add separately marked network tests for hierarchy discovery, scalar reads,
1-D/2-D/3-D slices, a slice crossing chunk boundaries, persistent reopening,
and no extra traffic on a repeated cached slice. Use h5py reference reads or a
temporary local copy for value comparison; exclude reference traffic from the
adapter's measurements. Generated fixtures remain the authoritative coverage
for gzip/shuffle and plugin fallbacks, absent from this public file.

Keep normal CI independent of this endpoint. Live failures should distinguish
transport/fixture changes from incorrect values, rather than skip any exception.

### Commands and performance evidence

Run all Python/test/build commands in the `blosc2` conda environment. If the
conda launcher fails due to its installed solver plugin, use that environment's
Python executable directly; do not substitute the base interpreter.

After implementation, run targeted suites first, Ruff on changed Python files,
then the default suite (warnings are errors). Run public tests explicitly, e.g.:

```sh
conda run -n blosc2 pytest tests/test_hdf5_source.py tests/test_remote_store.py tests/test_fsspec.py tests/b2view/test_hierarchy.py
conda run -n blosc2 pytest -m network tests/test_hdf5_source.py -k https
conda run -n blosc2 pytest
```

Record cold discovery, first chunk, adjacent/distant uncached chunks, cached
rereads, and persisted reopening separately. Report elapsed time, payload bytes,
data requests, and index size. Compare direct and fallback paths with identical
values/chunk geometry and several concurrency settings. Do not turn external
network timing into a CI pass/fail threshold.

## Implementation sequence and completion criteria

1. **Index contract and fixtures:** implement serialization/validation and scanner;
   establish dtype, fill, layout, and filter-mask tests before replacing callers.
2. **Direct reader:** implement exact range reads, three filter decoders, and
   cache-chunk conversion; prove no h5py calls during direct chunk retrieval.
3. **Fallback and lifecycle:** add lazy retained h5py readers, plugin errors,
   resource ownership, and fallback layout handling.
4. **Integration and persistence:** switch HDF5NDSource, RemoteArray, RemoteStore,
   discovery, manifests, and b2view; implement the stated compatibility policy.
5. **Dependency cleanup and documentation:** remove legacy HDF5 reference/Zarr machinery,
   update packaging and examples, and verify missing-dependency behavior.
6. **Validation:** run offline suites, live HTTPS tests, and request/concurrency
   measurements; resolve regressions before declaring the replacement complete.

Completion means HDF5 reads and browsing work without the legacy HDF5 reference layer, Zarr, or numcodecs;
all agreed direct filters use indexed range reads; other supported HDF5 filters
work through the retained h5py fallback; native indexes survive cache round trips;
warm cached reads avoid remote I/O; ownership and legacy errors are tested; and
the public fixture and generated test matrix pass. The first release does not
require optimizing contiguous reads or parallelizing the fallback.
