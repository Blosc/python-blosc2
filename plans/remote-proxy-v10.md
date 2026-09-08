# Remote proxy v10: native array reads inside remote B2Z hierarchies

Status: implemented and validated (2026-09-08). Authorized B2Z sparse attachment remains explicitly
unsupported, as permitted by this plan; ordinary RemoteProxy caches and direct
Proxy persistence are supported.

The implementation below is the delivered v10 scope. The remaining items in
Deferred work are not prerequisites for this version.

## Implementation results

- Added `B2ZNDSource` using a seekable view with bounded opening buffers for ZIP
  discovery and bounded native Blosc2 range reads for the selected member.
- Integrated all three addressing forms, source descriptors, cache identity,
  carrier reopening, and durable B2Z operand references for saved expressions.
- Added validation for member bounds, ZIP64, duplicate/encrypted/compressed
  members, malformed headers, and unsupported object carriers.
- Updated the remote proxy API documentation and S3 example.
- Default suite: 9750 passed, 29 skipped. Subsequently added dependency-isolation
  coverage passed with the focused B2Z suite (36 passed, 1 network test deselected).
  Focused existing remote/Proxy regressions: 299 passed, 6 deselected.
  Final combined B2Z/remote regressions after URL-parser edge-case checks:
  300 passed, 7 deselected. Ruff and whitespace checks passed.
- The exact S3 example succeeded against the remote archive: 8.85 KiB for opening
  (2592.3 ms), 76.42 KiB for the first slice (921.0 ms), and zero bytes for the
  cached repeat (0.9 ms). Values and native geometry match the supplied example.
  These are individual observed timings, not performance guarantees.
- Tests needing localhost servers and the S3 check were run outside the network
  sandbox. All Python commands used the `blosc2` conda environment.

## Opening optimization measurements

Three fresh-process S3 runs per version against `hierarchy.b2z::/d0/a3`:

| Metric | Initial reader | Buffered reader |
|--------|---------------:|----------------:|
| Median opening time | 2268.9 ms | 1197.3 ms |
| Opening requests | 2 HEAD + 7 GET | 1 HEAD + 2 GET |
| Opening payload | 9067 bytes | 24576 bytes |
| First-slice payload | 78257 bytes | 78257 bytes |
| Repeated-slice payload | 0 bytes | 0 bytes |

Opening is approximately 47% faster in these measurements. An 8 KiB tail and
16 KiB member prefix combine ZIP and native-frame header reads, while larger
directories, comments, and extra fields retain exact-read fallbacks. Buffers
are released after source construction; payload cache ownership is unchanged.
Source stamps now use the existing archive-info response instead of a second
identity lookup. This changes stamps from the initial reader, so older warm
caches may refetch once. Values, geometry, and cached-repeat behavior were
verified on every measurement. These timings do not promise a first-slice speedup.
Validation: 39 focused B2Z tests passed; the default suite passed with 9754 tests
and 29 skips. Ruff and whitespace checks passed.

## Objective

Open an external NDArray leaf inside an immutable remote `.b2z` archive as a
`RemoteProxy`, fetching only the archive metadata and Blosc2 byte ranges needed
for reads. Reuse native Blosc2 chunk/block reading and the existing cache policies.

```python
arr = blosc2.open(
    "s3://blosc2/hierarchy.b2z::/d0/a3",
    lazy=True,
    storage_options={
        "profile": "blosc2",
        "endpoint_url": "https://s3.us-west-001.backblazeb2.com",
    },
)
values = arr[:10, 0, :5]
```

This extends the dataset addressing introduced in v9. It does not require
kerchunk, Zarr, a Caterva2 server, archive extraction, or conversion of source
chunks into a different storage format.

## Scope and fixed decisions

- Support read-only access to external NDArray leaves stored as `ZIP_STORED`
  members in `.b2z` archives. This covers the default TreeStore layout.
- Require `assume_immutable=True`; reject mutable B2Z sources explicitly.
- Require an explicit dataset path. Do not silently choose the first array.
- Preserve native shape, chunks, blocks, dtype, and compression parameters.
- Reuse `ByteRangeNDSource` and the existing Proxy/RemoteProxy cache machinery,
  including its current whole-chunk versus block-read decisions.
- Use Python's `zipfile` for archive directory parsing and ZIP64 support.
  Do not write another ZIP directory parser.
- Use the existing optional fsspec dependency and protocol backends. Add no
  mandatory dependencies and no C/Cython changes unless a demonstrated blocker
  requires them.
- Preserve existing local store opening and standalone remote frame behavior.
- Full hierarchy browsing and embedded leaves are outside this version.

## Evidence and reusable code

`src/blosc2/dict_store.py` already writes external leaves as uncompressed ZIP
members containing self-contained Blosc2 frames. Its `member_window()` method
returns `(offset, length)` for a local external leaf. `_get_zip_offsets()`
calculates member data offsets using the local ZIP header, and
`_logical_key_from_relpath()` defines the external-member-to-logical-key mapping.
Reuse these rules without constructing a local DictStore for a remote archive.

`src/blosc2/proxy_source.py` provides `ByteRangeNDSource`, which interprets a
frame using a `read_range(offset, size)` transport. `FsspecNDSource` supplies
the current fsspec transport, identity handling, and traffic accounting.

`src/blosc2/core.py` provides `parse_container_url()`. Public remote opening
and option validation live in `src/blosc2/schunk.py`. Source descriptors,
reconstruction, identity, and sparse attachment live in
`src/blosc2/remote_proxy.py`.

A read-only diagnostic against the local `hierarchy.b2z` found 12 external
`.b2nd` members and `embed.b2e`, all `ZIP_STORED`. The member `d0/a3.b2nd`
starts at byte 939145 and occupies 312995 bytes. A temporary
`ByteRangeNDSource` subclass translating reads into that member, wrapped in
the existing `Proxy`, reproduced the sample values with:

| Operation | Bytes read |
|-----------|-----------:|
| Frame metadata | 8192 |
| First `[:10, 0, :5]` slice | 78257 |
| Repeated slice | 0 |

These are local range-read measurements, excluding ZIP discovery. They are
evidence that the native reader can handle the member, not an S3 performance
result or an assertion that the remote archive is identical.

## Public API and addressing

Support the same three addressing forms as v9:

```python
blosc2.open("s3://bucket/hierarchy.b2z::/d0/a3", lazy=True)
blosc2.open("s3://bucket/hierarchy.b2z/d0/a3", lazy=True)
blosc2.open("s3://bucket/hierarchy.b2z", dataset="d0/a3", lazy=True)

blosc2.RemoteProxy(
    "s3://bucket/archive",
    source_format="b2z",
    dataset="d0/a3",
    cache_policy=blosc2.CachePolicy.MEMORY,
)
```

Add `"b2z"` as an explicit source format. Infer it from a `.b2z` URL path
component, including nested dataset syntax; inspect parsed URL paths rather
than query-string text. Explicit source-format selection retains precedence.
Preserve fsspec protocol-chain handling when interpreting `::`.

Normalize optional leading/trailing dataset slashes to the same canonical
identity. Reject conflicting URL and keyword dataset specifications. Use the
existing TreeStore key rules; reject invalid traversal or malformed paths
rather than normalizing them into another leaf. A logical key is not a raw ZIP
filename: resolve `d0/a3` to the canonical external member `d0/a3.b2nd`.

Missing dataset, missing leaf, group selection, and unsupported leaf kinds
must produce actionable errors. A known embedded-only leaf should explain the
scope limitation if cheaply identifiable; do not add remote EmbedStore reading
just to improve an error message. Otherwise report that no supported external
NDArray exists at the selected path.

Dataset selection through this remote path requires `lazy=True`. Existing
non-lazy whole-archive localization, where supported, remains unchanged.

## Archive discovery and bounded frame reads

Implement a small B2Z source adapter around the existing native frame reader.
Prefer a focused adapter in `src/blosc2/b2z_source.py`, reusing fsspec transport
code where practical. Avoid a general archive framework or unrelated reader
refactoring. The adapter must also accept a supplied filesystem internally for
deterministic tests and authorized transport attachment.

### Opening

1. Resolve the filesystem and archive path with the existing fsspec conventions.
2. Open a seekable read-only view for `zipfile.ZipFile`, with buffering explicitly
   controlled so a seek does not cause a large default read-ahead or full download.
3. Read the ZIP directory and locate the selected canonical array member.
   Central-directory work scales with archive member count; do not open every
   member or construct the complete TreeStore.
4. Reject duplicate matches, encrypted members, and compression methods other
   than `ZIP_STORED`. Check the selected local header and its consistency with
   the directory before using its data offset. Reuse the local-header offset
   calculation, with explicit short-read and signature validation.
5. Validate the member window against the archive size. Account for ZIP64 using
   `zipfile`'s decoded metadata; stored compressed and uncompressed sizes must
   agree. Reject corrupt or impossible windows before frame construction.
6. Initialize the native frame reader over this member window. Validate that
   it holds a supported NDArray and that its declared frame length fits inside
   the member. Do not fetch the selected array in full to inspect it.

Close discovery handles after resolving the member; payload reads should use
stateless range requests suitable for the existing concurrent fetch scheduler.

### Reading

Translate frame-relative reads as:

```text
archive_start = member_offset + frame_offset
```

Bound each read by the member length. Preserve normal end-of-file short-read
semantics for speculative header/tail reads, while letting existing frame
validation reject truncated required data. Reject negative or invalid ranges.
No frame operation may read an adjacent member by running past its window.

Reuse `get_chunk()`, block layout parsing, index handling, and fetch scheduling
from the native source. Do not decompress/recompress every chunk through the
HDF5/Zarr conversion path. Existing block assembly may still apply when the
cache fetches partial chunks.

Traffic accounting must include directory discovery, local-header reads, frame
metadata, indexes, and payloads exactly once. Measure bytes at the transport
boundary when buffering is present; counting bytes returned by a buffered file
can conceal read-ahead. ZIP metadata reads may overlap payload bytes in a small
archive, but opening must not intentionally materialize array members.

## Identity, cache policies, and persistence

Use a distinct source descriptor:

```json
{
  "kind": "b2z",
  "version": 1,
  "urlpath": "s3://bucket/hierarchy.b2z",
  "dataset": "d0/a3",
  "assume_immutable": true
}
```

The canonical identity includes both archive URL and dataset. Two leaves with
identical geometry must never collide in automatic cache paths or source-spec
comparisons. Keep credentials and live storage options out of persisted metadata.
Apply the existing persistable-URL validation and fail-closed descriptor checks.

Resolve the member window afresh on reopening in the first implementation.
Do not persist an unchecked byte offset as the authority for future reads.
Combine archive identity, selected member/window, and native frame interpretation
as needed for a stable source stamp using existing identity conventions. Retain
geometry validation on reopening. Do not claim to detect payload-only replacement
under the immutable contract; replacing the archive at the same identity can
serve stale data and requires replacing its cache.

Update all relevant paths together:

- Source opening, format validation, dataset normalization, and `urlpath`.
- Descriptor validation, `_source_identity()`, stamps, and payload reconstruction.
- NONE, MEMORY, bounded/unbounded DISK, and automatic cache locations.
- `save()`, `to_cframe()`, carrier reopening, and sparse-cache reconstruction.
- Direct persistent `Proxy` metadata/reconstruction if the adapter is exposed
  for direct use; never serialize it as a standalone frame at the archive URL.

Reuse existing retention and eviction behavior. Preserve storage options during
in-process reconstruction; fresh processes resolve their own credentials.
Repeated cache hits under the immutable contract must perform no remote payload
or metadata reads. Reopening may reread ZIP and frame metadata; eliminating
those reads is not a requirement for v10.

## Authorized transports and Caterva2 boundary

Actual Caterva2 federation remains a separate integration task. Within
Python-Blosc2, either support B2Z explicitly through the existing authorized
sparse-attachment interface or reject it explicitly until that path is complete.
Do not let a subclass pass a broad FsspecNDSource check while losing its dataset
identity or member bounds.

If supported, validate the supplied concrete source against the full descriptor
before cache access, including hits. Directory discovery, header reads, payload
reads, and any reconstruction must retain the supplied filesystem and must not
fall back to unrestricted URL opening. Archive members are byte windows, not
external references to follow. Preserve existing server transport restrictions.

## Implementation sequence and validation

### 1. Source adapter

Create small temporary TreeStore archives using existing test conventions.
Exercise a selected external NDArray through `Proxy`, using counted local or
memory-backed fsspec transport. Confirm native values and geometry for nested
keys, multiple chunks, edge chunks, and representative fixed-size dtypes.
Check scalar and empty arrays where the native reader supports them, preserving
clear errors for any existing native limitations.

Verify selected-member bounds, malformed/truncated archives, missing members,
duplicate selected names, compressed/encrypted members, and non-NDArray leaves.
Cover ZIP64 local-header behavior without allocating a multi-gigabyte fixture.
Ensure concurrent reads do not share an unsafe seek position.

### 2. Public dispatch and cache integration

Update `parse_container_url()`, `blosc2.open()`, and `RemoteProxy` together.
Test all addressing forms, leading slashes, explicit suffix-free format,
conflicting dataset specifications, missing datasets, and mutable-source
rejection. Preserve existing HDF5/Zarr/standalone Blosc2 dispatch tests.

Assert metadata-only opening with transport read logs on an archive large enough
to distinguish directory/header reads from downloading members. Verify correct
slices, zero reads on repeated hits, absent-only fetches on overlapping slices,
and refetch after eviction. Exercise all cache policies using existing tests
and helpers rather than duplicating their complete suites.

### 3. Persistence and attachment

Test cold/warm carrier reopening, memory exports, bounded disk caches, and
source identity isolation between same-shaped leaves. Verify credential
exclusion, geometry mismatch rejection, materialization, and a simple lazy
expression round trip. Test direct persistent Proxy use if supported.

For authorized attachment, use a supplied fake filesystem and make unrestricted
opening raise. Exercise directory discovery, misses, hits, and source-descriptor
mismatch. If attachment is deferred, test its explicit rejection instead.

### 4. Documentation and example

Extend the existing remote proxy documentation and `examples/remote/s3-access.py`
with `.b2z` dataset selection. Keep the metadata, cold/warm slice, and traffic
reporting comparable across formats. Document immutable archives, external
NDArray-only scope, directory-discovery cost, and unsupported embedded leaves.

Run the requested S3 example when network access and credentials are available.
Check sample values against the known data and compare with the HDF5 example;
report measured metadata bytes, first-read bytes/latency, and warm-cache behavior.
Do not assume the local archive's byte offsets match the remote copy. Keep public
S3 tests marked `network` and outside the default suite.

### 5. Final checks

Use the `blosc2` conda environment for all Python, tests, and build commands.
Run focused B2Z, fsspec, Proxy, RemoteProxy, and URL parsing tests, followed by the
default suite and repository lint checks. Verify unrelated local use and native
remote B2ND reads still work without Zarr, kerchunk, or h5py. Record network or
optional-dependency checks that could not run.

## Completion criteria

- [x] The requested `.b2z::/d0/a3` example opens as a RemoteProxy and returns correct
  values without downloading or extracting the archive in full.
- [x] Native frame reads remain within the selected member, preserving native
  geometry and the existing chunk/block fetch behavior.
- [x] Opening reads archive/frame metadata; warm slice hits perform zero remote reads.
- [x] Traffic counters include discovery and payload transport without double counting.
- [x] Cache policies, exports, and reopening retain the selected dataset identity.
- [x] Unsupported representations fail clearly; authorized attachment either retains
  its transport and descriptor constraints or rejects B2Z explicitly.
- [x] Existing local stores and remote Blosc2/HDF5/Zarr behavior remain intact.

## Deferred work

Hierarchy browsing/discovery APIs, group metadata, remote `.b2d` hierarchies,
embedded leaves inside `embed.b2e`, SChunk/ObjectArray/BatchArray/CTable leaves,
Caterva2 reference leaves, compressed ZIP members, mutable archives, archive
writing, persisted ZIP indexes, and Caterva2 server-side federation.
