# Remote cache design

This note describes the implemented cache model for remote native Blosc2 arrays,
B2Z containers, HDF5 files, and Zarr stores. For usage, see
{doc}`../guides/remote_objects`, {doc}`../guides/remote_arrays`, and
{doc}`../guides/remote_tables`. Cache layouts and metadata fields described here
are private implementation details, not portable format contracts.

## Shared model

Opening a remote object discovers enough metadata to describe it. Payload reads
are deferred until indexing or computation needs them, except where bounded
prefetch also retrieves payload. Three kinds of cached information must be
distinguished:

| Layer | Purpose | Current implementation |
| --- | --- | --- |
| Discovery metadata | Reconstruct readers, locate data, and navigate containers | Array carriers and store manifests; format-specific headers, indexes, or metadata objects |
| Compressed payload | Reuse fetched or converted data | Native Blosc2 chunks/blocks, or chunks converted from HDF5/Zarr |
| Complete source object | Reuse original bytes across dataset scopes | HDF5 files and B2Z archives up to 8 MiB |

A warm metadata cache does not imply a warm payload cache. Reopening may avoid
discovery requests but still fetch data for a preview. HDF5 and B2Z complete-source
caches bridge that gap for small files; the other routes do not provide a
general persistent copy of the remote source.

### Identity and ownership

Managed cache paths distinguish source URLs and non-reversible
`storage_options` fingerprints. Dataset-scoped entries also distinguish the
selected leaf or subtree. Source stamps and recorded geometry determine whether
existing payload can be reused; their inputs differ by format. A stamp is not
a universal content checksum or an automatic remote freshness check.

A standalone `RemoteArray` owns its cache carrier. A `RemoteStore` discovery
owner shares transport, metadata, traffic, and a cache coordinator across its
leaves and table columns. Store disk caches retain a manifest and payload under
an active generation. Ordinary store disk caches have exclusive ownership;
the separate sparse shared-cache path uses operation-scoped locking.
Dependent handles keep the owner alive until its last user closes.

### Policies, accounting, and persistence

`MEMORY` retains payload in RAM; `DISK` retains it across sessions; `NONE` does
not retain payload between operations. NONE does not mean that readers discard
all discovery state or temporary buffers. Array disk caching supports an
explicit `cache_path` or a managed `cache_dir`; stores use `cache_dir`.

`max_cache_bytes` bounds retained compressed payload, with LRU enforcement after
operations. Store-owned readers share the owner's allowance. The default bound
is 256 MiB; DISK also accepts `None` for unbounded retention. This is not a limit
on decompressed results, temporary conversion buffers, discovery metadata, or
total filesystem usage. The HDF5 source-copy exclusion is described below.

`Traffic` measures transport activity at the reader's instrumentation points,
not local cache reads. Discovery and payload reads share an owner's counter.
It is not a packet-level HTTP trace: backend metadata probes, retries, and
batched operations need not map one-to-one to its request count.

Portable reference exports are distinct from disposable runtime caches. They
record source locators and optionally warm payload, not a promise that all source
data is local. Immutable references do not grow an on-disk cache; mutable
references use a writable runtime cache without modifying the original archive.

### Freshness and refresh

Remote reads assume immutable sources by default. For standalone `.b2nd` and
Caterva2 sources, `assume_immutable=False` enables identity checking and cache
invalidation. The B2Z, HDF5, and Zarr adapters do not support that mutable-source
mode; publishing new data at a new URL remains the simplest safe contract.

Explicit table/store refresh prepares new discovery before replacing the active
generation and invalidating derived caches. Failure during preparation leaves
the previous generation usable. Existing child handles of a refreshed store
become stale and must be reacquired. Other independently opened scopes are not
automatically refreshed; HDF5/B2Z shared-source version checks add the specific
reopen behavior described below. Immutable reference snapshots cannot refresh.

## Native Blosc2 arrays and B2Z containers

### Standalone `.b2nd`

`FsspecNDSource` uses the native frame reader in `proxy_source.py`. Opening reads
the frame header; chunk offsets and payload ranges are obtained as needed.
Native compressed chunks and, where supported, individual blocks can be fetched
without decoding an entire array or translating it to another storage format.
The cache retains that compressed payload, not a copy of the source file.

Source identity uses transport information, including HTTP validators when
available. Reopening a disk carrier does not guarantee zero bootstrap traffic
for this route. The immutable-source policy avoids repeated identity checks
before each data operation; it is not the HDF5 whole-file persistence mechanism.

### `.b2z` archives

`B2ZArchive` discovers ZIP members through bounded byte ranges. Lazy NDArray
reads address external `ZIP_STORED` members and translate native frame offsets
to archive offsets. Table/store discovery shares the archive reader and metadata
across leaves, rather than opening each member as an unrelated remote file.

Persisted archive metadata records object identity and captured discovery
ranges. Leaf carriers can also store a `b2z-frame` bootstrap seed, allowing a
reader to be reconstructed without fetching its header again. A populated
table/store cache trusts the saved archive identity on reopen; older caches
may need an identity lookup when upgrading their metadata.

Cold discovery eagerly downloads archives up to 8 MiB and retains the bytes for
all members. HTTP discovery first requests the ZIP tail to learn the object size;
if the tail contains the whole archive it is reused, otherwise a second request
fetches the complete small archive. Larger archives keep the range-read path.
With an explicit DISK cache directory, the shared source copy survives reopening.
All archive read paths, including parallel table reads, can use these bytes.

Bounded member prefetch can also contain a complete small member. Its payload is
transferred to the normal chunk cache, while the persistent leaf bootstrap keeps
the metadata it needs. This remains useful for large archives without a source
copy. Replacing an archive at the same URL requires refresh or cache replacement.

## Zarr stores

`ZarrNDSource` delegates Zarr v2/v3 metadata and codec handling to Zarr. A
counting store wrapper retains metadata objects such as `.zarray`, `.zgroup`,
`.zattrs`, `.zmetadata`, and `zarr.json`, including remembered missing metadata
keys. Standalone carriers can restore this metadata; store discovery shares and
persists it in its manifest. Resolving a selected array directly avoids an
unnecessary parent listing, but unseen hierarchy paths can still require reads.

On a payload miss, Zarr reads and decodes the required source chunk data. The
adapter converts the resulting values into a Blosc2-compressed logical chunk,
which enters the ordinary payload cache. Repeated reads can therefore bypass
both remote transfer and Zarr decoding. The persisted payload is not a mirror
of the original Zarr objects. Sharded layouts and codec details remain Zarr's
responsibility, so a logical chunk read need not equal one whole-object GET.

The source stamp includes location, geometry, dtype, conversion layout, and the
storage-options fingerprint. It does not detect changed values under unchanged
metadata. Source immutability and explicit refresh/cache replacement are thus
essential. There is no whole-store download or shared raw-source cache, and
peak memory can exceed the retained-payload budget during decode/conversion.

## HDF5 files

The native index and source cache are Blosc2 mechanisms, not HDF5 conventions
or fsspec cache formats.

### Discovery and reads

Remote discovery uses h5py to record dataset metadata and allocated chunk byte
ranges. Selecting a dataset avoids traversing unrelated siblings; PyTables index
groups are discovered lazily. Opening a hierarchy discovers its nodes, but
allocated-chunk maps are deferred until a leaf is read.

A standalone `RemoteArray` stores its native index in the carrier's
`schunk.vlmeta["hdf5-index"]`. A `RemoteStore` shares discovery metadata across
its leaves and persists it in a manifest. Supported filter pipelines decode
fsspec range reads directly; other pipelines use a retained h5py reader.

For objects up to 8 MiB, discovery fetches the complete file once: one bounded
transfer avoids many latency-bound metadata reads. Retaining these bytes lets
later reads and PyTables index conversion reuse the same download.

## Shared complete-source cache (HDF5 and B2Z)

The complete source bytes, discovery metadata, and converted Blosc2 chunks have
different lifetimes. Source bytes are shared across dataset scopes; metadata and
converted payload remain in their existing array/store caches.

With an explicit `cache_dir` and `CachePolicy.DISK`, source bytes persist under
`cache_dir/hdf5-sources/` or `cache_dir/b2z-sources/`. Both readers reuse the
integrity, publication, and locking helpers in `remote_source_cache.py`.
The source identity reuses `fsspec_cache_path()`:
normalized base URL plus the non-reversible `storage_options` fingerprint,
without the dataset scope. Different datasets in one file share a single copy;
different cache directories, URLs, or credential fingerprints remain isolated.

Each source has a `.hdf5-source` or `.b2z-source` file, a JSON marker, and a publication lock.
The marker records its schema version, byte size, SHA-256, and version token.
These names and fields are private implementation details.

Source copies do not count against `max_cache_bytes`. The 8 MiB ceiling is per
file; aggregate source-cache disk usage is unbounded. They survive individual
dataset-generation cleanup. Close active cache users before removing
the corresponding `hdf5-sources/` or `b2z-sources/` directory to clear source copies,
or the whole cache directory to
reset all caches. There is no automatic source eviction or freshness check.

MEMORY/NONE policies can retain prefetched source bytes for the current session,
but do not persist them. `cache_path`-only opens and portable references without
an explicit shared root do not gain a persistent source copy. Saved metadata
alone does not force a whole-archive download on portable B2Z reference reopen.
Zarr does not use this source cache.

### Publication and reuse

Normal carrier/manifest initialization precedes source publication. Existing
atomic-write helpers publish the blob and then its marker. A short source-level
file lock serializes version checks and publication, not downloads. No
store/array lock is acquired while holding the source lock. A stale opener must
retry rather than overwrite a newer refreshed source.

Readers validate marker schema, file size, bounded reads, and SHA-256. Missing,
corrupt, or interrupted file/marker pairs are cache misses, so ordinary remote
reads remain available. Local source reads do not contribute to `Traffic`.
Valid bytes can also supply discovery for a new dataset scope without a remote
size lookup or download.

### Version compatibility and refresh

Generated HDF5 indexes and B2Z discovery metadata record `source_sha256` when
complete bytes are available. Small B2Z member stamps derive from that content
identity rather than transport-dependent header fields.
An index incompatible with the shared source must be rebuilt, and its derived
payload invalidated. Legacy small-source disk metadata without a digest is
rebound by rebuilding discovery and invalidating derived payload once.

Table/store refresh bypasses the retained bytes and prepares fresh discovery.
A preparation failure leaves the old live generation usable. Successful refresh
publishes replacement metadata and source bytes; other scopes check compatibility
on their next open. Existing handles may retain their old immutable snapshot.

If the refreshed file exceeds 8 MiB, publication replaces the old marker with a
new version token and null checksum, then removes the old blob. Discovery metadata
record this token as `source_cache_version`, so sibling scopes invalidate old
metadata and payload even though there is no replacement cached blob.

Explicit `hdf5_index=` never causes a complete download just to populate this
cache. It may reuse verified cached bytes; a recorded digest mismatch is an
error requiring a regenerated sidecar, not silent replacement of the supplied
index. Legacy explicit indexes without a digest retain the immutable-URL trust
contract: matching sizes alone cannot prove version compatibility.

## Boundaries and future work

The common model does not imply identical bootstrap costs or invalidation
mechanisms across formats. The 8 MiB complete-source threshold, checksum marker,
and cross-scope source-version reconciliation apply to HDF5 and B2Z, not to
standalone `.b2nd` files or Zarr stores.

Zarr would require an object-level policy rather than this single-file design.
Automatic freshness checks for containers, aggregate
source-cache eviction, and a global disk-space budget are not implemented by
this design.
