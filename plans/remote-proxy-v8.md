# Remote proxy v8: remote arrays with immutable-by-default Blosc2 caching

## Objective and status

Status: implemented in this branch.

Add `ZarrNDSource`, a `ProxyNDSource` adapter that reads remote Zarr arrays and
returns Blosc2 compressed chunks. Reuse the existing `Proxy` and `RemoteProxy`
cache implementations, including bounded memory caches, portable B2ND carriers,
and the sparse runtime caches introduced for Caterva2 in v7.

RemoteProxy sources are assumed immutable by default. Callers following a mutable
single-file source can opt into identity checks with `assume_immutable=False`.
Zarr remains immutable-only in this version because changes to one object do not
provide an identity for the complete array. Zarr remains an optional dependency,
like fsspec. The adapter belongs in Python-Blosc2 and must be usable without a
Caterva2 server.

## Fixed decisions

- Read Zarr through Zarr-Python's public array/store APIs. Do not implement a
  second metadata parser, chunk-key encoder, codec pipeline, or shard reader.
- Cache converted Blosc2 chunks in existing B2ND containers. Do not use Zarr's
  `CacheStore` or maintain a second cache of Zarr objects.
- Fetch and convert whole logical Zarr chunks. A shard is a storage object, not
  the cache chunk shape. Let Zarr resolve chunk reads within shards.
- Support Zarr format 2 and 3 through a supported Zarr-Python 3 release. This
  does not require supporting the older Zarr-Python 2 API.
- Start with fixed-size numeric and boolean arrays representable by B2ND.
  Reject unsupported dtypes explicitly before creating a cache.
- `RemoteProxy(..., assume_immutable=True)` is the default for every source and
  skips metadata polling before reads. `False` retains identity refresh and cache
  invalidation for mutable single-file and Caterva2 sources.
- A Zarr source identity names immutable contents for the lifetime of its caches.
  No per-chunk ETag checks, TTL, or mutable Zarr-store support.
- Apply the same immutable default and persisted flag to Blosc2 and Caterva2
  source descriptors.
- Do not change C/Cython code unless implementation exposes a demonstrated
  blocker that cannot be handled by existing Python APIs.

## Existing code to reuse

`src/blosc2/proxy_source.py` defines `ProxyNDSource`: shape, chunks, blocks,
dtype, compression parameters, and `get_chunk(nchunk)`. Its optional block-range
interface is unnecessary for this adapter. `Traffic` already provides
thread-safe counters for received bytes and data-bearing requests.

`src/blosc2/proxy.py` creates a B2ND cache from that geometry, fetches missing
chunks, inserts compressed chunks, and tracks retention and eviction. These
mechanisms should remain the owners of cache state.

`src/blosc2/remote_proxy.py` handles source descriptors, geometry validation,
source stamps, carriers, and sparse attachment. Its URL-string branch currently
always creates `FsspecNDSource`. Source identity, payload reconstruction, and
authorized attachment currently distinguish only fsspec and Caterva2 sources.

`src/blosc2/schunk.py` owns the public `blosc2.open()` remote dispatch and cache
options. Changes must reach this shared path, not just the S3 example.

`pyproject.toml` already defines `zarr = ["zarr"]` and a separate `fsspec` extra.
Reuse these extras. The existing example already opens Zarr stores and measures
their reads, but its Zarr branch returns an uncached Zarr array.

## Proposed public opening API

Add `source_format=None` to `RemoteProxy` and accept it through
`blosc2.open(..., lazy=True)`. Initially allow `None`, `"blosc2"`, and `"zarr"`.
Add `assume_immutable=True` to both entry points and persist it in every source
descriptor. Users following replaceable sources must set it to `False`.

```python
arr = blosc2.open(
    "s3://blosc2/cube-1k-1k-1k.zarr",
    lazy=True,
    source_format="zarr",
    cache_policy=blosc2.CachePolicy.MEMORY,
    storage_options={"anon": True},
)
values = arr[:10, 0, :5]

arr = blosc2.RemoteProxy(
    "s3://bucket/hierarchy.zarr/d0/d1/a2",
    source_format="zarr",
    cache_policy=blosc2.CachePolicy.DISK,
    cache_path="array-cache.b2nd",
)
```

An explicit format supports URLs without a `.zarr` suffix. With no explicit
format, recognize a `.zarr` path component, including a trailing slash and nested
array paths; retain the existing Blosc2 default for other remote URLs. Inspect
the parsed URL path, not query-string text. Explicit selection wins over the
heuristic. Do not catch arbitrary opening errors and retry another format.

The URL names the array itself, including its path within any hierarchy. Opening
a group raises an actionable error asking for an array path. Do not recursively
discover or select an array. ZIP-backed Zarr is outside this first version and
must not be misclassified as ordinary directory/object-store Zarr.

Keep existing cache-policy defaults: direct `RemoteProxy` and `blosc2.open`
already have their own defaults. Adding Zarr must not silently change them.
Reject incompatible format options on Caterva2 `URLPath` inputs and reject Zarr
opening without `lazy=True` in this release. Existing local Blosc2 opening stays
unchanged; a direct adapter can accept a local store for testing.

## Adapter implementation

Create `src/blosc2/zarr_source.py` and export `ZarrNDSource` from `blosc2`.
The module may import NumPy and existing Blosc2 helpers at module scope, but must
not import Zarr or fsspec until construction needs them.

### Construction and metadata

1. Import optional dependencies with actionable errors.
2. Open a read-only store and use `zarr.open_array(..., mode="r")`.
3. Read metadata only: shape, logical chunks, dtype, and the metadata required
   to validate compatibility. Construction must not fetch array payloads.
4. Normalize geometry to Python tuples and dtype to a NumPy dtype. Validate
   dimensions, positive chunk extents, dtype support, and Blosc2 size limits
   before allocating conversion buffers or cache containers.
5. Set `serves_blocks = False`. Expose the existing concurrency and traffic
   conventions so `Proxy` can fetch chunks through its current executor.

Do not treat Zarr's `arr.blocks` as a block shape: it is an indexing interface.
Choose cache blocks with the existing Blosc2 partition helper, supplying the
chosen chunk shape. Use normal Blosc2 compression defaults initially. These
parameters describe the converted cache, not the original Zarr compressor.

Cache partitioning must be reproducible across reopening. Persisted carrier
chunks and blocks are authoritative when restoring a cache; pass that geometry
through the adapter reconstruction path and validate it against the source.
Do not accidentally reject old caches because an automatic block-size heuristic
or a default compression setting changed in a later library release.

### `get_chunk(nchunk)`

1. Validate the chunk number and map it to C-order logical chunk coordinates.
   Reuse an existing coordinate helper if one fits; otherwise use NumPy's
   unraveling with the grid obtained by ceiling-dividing shape by chunks.
2. Compute the clipped array slice for that chunk and read it through Zarr.
3. Normalize the decoded result to the advertised dtype and contiguous order.
   Preserve numeric values, including endian conversion where needed; never
   reinterpret foreign-endian bytes as native values.
4. For edge chunks, place valid values in an initialized full-chunk buffer.
   Padding lies outside the logical array and must never contain uninitialized
   memory. Missing Zarr chunks inside the logical array are filled by Zarr using
   the declared fill value, not by substituting zero in the adapter.
5. Encode a single-chunk B2ND temporary with the cache's exact chunks, blocks,
   dtype, and compression parameters, then return `get_chunk(0)` from it.
   This delegates multidimensional block layout and padding to Blosc2.

Keep each conversion buffer and temporary container local to the call. Reuse
`Proxy`'s fetch scheduling; do not add another thread pool or a shared mutable
scratch array. Investigate Zarr's internal concurrency when measuring peak
memory: the outer limit bounds adapter chunk calls, not necessarily every
internal store request. Add an asynchronous adapter method only if an existing
public async path requires it, and use Zarr's supported scheduling mechanism.

Document the initial memory ceiling: concurrently decoded whole chunks plus
conversion/compression buffers. `max_cache_bytes` limits retained compressed
payload, not transient decoded memory or process RSS.

### Supported representations

Cover bool, integer, floating-point, and complex dtypes supported by B2ND, both
Zarr formats, alternative codec pipelines, nonzero fill values, edge chunks,
and Zarr v3 sharding. Let Zarr decode storage order and transpose codecs.

Reject object, variable-length, string, structured, and other unsupported dtype
representations with a clear `TypeError` in this first version. Test scalar and
zero-length arrays against existing B2ND geometry constraints; support them if
the normal path works, otherwise reject explicitly at construction and document
the limitation. Never defer such failures until after a partially written cache.

## Immutable identity and persistence

Use a distinct source descriptor, keeping the existing outer RemoteProxy payload
version if its schema remains compatible:

```json
{
  "kind": "zarr",
  "version": 1,
  "urlpath": "s3://bucket/hierarchy.zarr/d0/d1/a2",
  "assume_immutable": true
}
```

Version 1 records `assume_immutable`, which defaults to `true`. Zarr requires it
to remain true in this version. Persisting the option preserves read behavior
after reopening and makes the assumption visible to clients and Caterva2.
Unknown fields, versions, and unsupported values fail closed.

Generate a stable non-null source stamp from a canonical, credential-free
descriptor plus the adapter encoding version and normalized cache geometry.
Use a deterministic digest, not Python's randomized `hash()`. This stamp denotes
an identity under the immutable contract; it is not a remotely verified content
hash. Include interpretation metadata where practical to detect changed metadata
on reopening, without claiming to detect payload-only mutations.

Do not implement a remote refresh method that fetches metadata on every read.
Ensure `_prepare_read()` keeps the stable source and cache instead of treating
it as an unstamped source. Existing geometry validation still runs on reopening.
Document that replacing data under the same identity violates the contract and
may serve stale data or mix old cached chunks with newly fetched chunks. A new
dataset should have a new immutable URL; manual cache replacement is the escape
hatch for intentionally reused URLs.

Update every relevant branch together: `_open_source`, `_source_identity`,
`urlpath`, payload validation/reconstruction, `with_sparse_cache`, carrier
exports, and source-spec comparisons. Preserve live storage options on any
reconstruction within a process, but never persist them. A fresh process must
resolve credentials from its own environment, as with existing sources.

Check direct `Proxy(ZarrNDSource(...))` metadata too: it must not be labeled as
a local Blosc2 source merely because it exposes `urlpath`. If direct persistent
Proxy caches are supported, add their source reconstruction alongside the
RemoteProxy path; do not produce a cache that saves successfully but reopens
through the wrong backend.

## Optional dependencies

- Reuse `blosc2[zarr]`; do not add Zarr to mandatory runtime dependencies.
- Select the minimum Zarr-Python 3 version actually required by the public APIs
  used and verify it. Do not require local version 3.3 merely because it is
  installed. Adjust the existing extra's version floor only with that evidence.
- Keep fsspec independently optional. Document remote installs as
  `pip install "blosc2[zarr,fsspec]"`, adding `s3fs` for S3 as today.
- Import Zarr inside construction or a file-local helper, following the
  actionable-error pattern of `core._import_fsspec`. A missing Zarr error should
  name `blosc2[zarr]`; missing protocol backends retain their useful errors.
- Importing `blosc2`, using local B2ND arrays, and opening remote B2ND sources
  must work without Zarr installed. Local Zarr adapter use should not require
  fsspec when the chosen store does not need it.
- Move the S3 example's unconditional Zarr import into the branch that needs
  it. Preserve any existing user edits when implementing the example update.
- Optional dependency tests use `pytest.importorskip`; also test the actual
  missing-dependency error path separately so skipping cannot hide it.

## Traffic, example, and performance expectations

Count encoded bytes received beneath the Zarr decoder. Reuse the example's
instrumentation approach as a reference, but implement the adapter's tracking
through supported store/transport extension points. Cover ordinary reads and
partial/shard reads; avoid double-counting delegated batched operations.
The counter measures received payload bytes, not headers or decoded array size.
Place it below any local cache so a Blosc2 hit charges no remote payload.

Update `examples/remote/s3-access.py` so directory/object-store Zarr arrays open
through `blosc2.open(..., lazy=True)` and print RemoteProxy cache details.
Display the cache's actual block shape. Keep ZIP behavior explicitly separate.
Metadata, first slice, second slice, and transferred-byte reporting should remain
comparable to the current example.

The first miss pays Zarr decoding plus Blosc2 encoding. A hit reads the Blosc2
cache without repeating either remote reads or Zarr decoding. No timing speedup
is promised for a cold read. Warm-cache tests assert zero payload reads, not
a machine-dependent millisecond threshold. Measure cold/warm latency, encoded
network bytes, retained Blosc2 bytes, and transient memory separately.

## Caterva2 integration boundary

The Python-Blosc2 implementation comes first. Actual Caterva2 changes belong in
the Caterva2 repository as a subsequent integration phase; v7 remains the cache
lifecycle and quota design.

Prepare the adapter to accept a server-supplied, already-authorized store or
filesystem internally. Extend authorized sparse attachment to recognize the
concrete Zarr source and validate the exact descriptor against it. Do not accept
arbitrary source objects as implicitly authorized.

An authorized attachment must retain the supplied transport for metadata,
chunks, shard indexes, and all subsequent reads. It must never reopen the URL
using an unrestricted default filesystem. Keep authorization before cache access,
including hits, as required by the existing Caterva2 contract.

Caterva2 must authorize an array prefix and validate all derived object paths;
authorizing one metadata URL alone is insufficient for a multi-object source.
Preserve HTTPS allowlists, DNS pinning, redirect refusal, credential isolation,
geometry limits, and embedded-reference restrictions. Confirm the Zarr pipeline
cannot escape the authorized transport/prefix through an unsupported store or
codec; restrict server support explicitly where necessary.

Once those boundaries are validated, reuse the existing sparse generation,
locking, dirty recovery, eviction, quota, and warm-export paths. Downstream
Caterva2 clients receive ordinary Blosc2 chunks and need no Zarr dependency.
The Caterva2 server reading the external Zarr source needs the optional extra.
An immutable-source stamp does not itself grant authorization.

## Implementation sequence and checks

### 1. Adapter and dependency isolation

Implement the adapter, deferred import, metadata validation, and chunk conversion.
Add local temporary Zarr fixtures in `tests/test_zarr_source.py` and exercise the
adapter through `Proxy`, not only by calling `get_chunk()` directly.

Verify format 2/3, one and multiple dimensions, exact and edge chunks, missing
chunks/nonzero fill, endian handling, representative supported dtypes, unsupported
dtypes, and sharded format 3. Check scalar/empty geometry explicitly. Include a
codec other than Blosc to establish that conversion does not assume Blosc bytes.

### 2. RemoteProxy opening and immutable caching

Implement dispatch, immutable descriptors/stamps, and policy integration.
Use counted local or memory-backed stores for deterministic tests without public
network access. Ensure metadata opening does not fetch chunk data.

Test explicit/automatic selection, nested arrays, suffix-free explicit URLs,
trailing slashes, group errors, and useful authentication/opening failures.
Test NONE, MEMORY, bounded DISK, and unbounded DISK using existing conventions.
Repeated and overlapping reads must fetch only absent chunks; eviction must
cause a later read to fetch again. A repeated hit must not poll metadata either.
An oversized chunk must obey the existing post-operation retention bound.

### 3. Persistence and server attachment

Test cold and warm carrier reopening, MEMORY exports remaining cold, geometry
mismatch rejection, changed encoding identity, source-spec mismatch, and credential
exclusion. Exercise `save()`, `to_cframe()`, materialization, and a simple lazy
expression round trip. Check direct persistent Proxy reconstruction if exposed.

Exercise sparse attachment and warm seeding with an authorized fake transport.
Patch unrestricted source opening to raise and verify metadata reads, misses,
and hits still follow the authorized path. Preserve existing sparse recovery
and eviction behavior rather than duplicating its tests for every Zarr dtype.

### 4. Documentation and example

Document the adapter API, supported arrays, optional installation, cache
representation, immutable contract, and memory/performance limits. Add the new
source to the existing proxy API documentation and example descriptions.
Run the S3 example manually against the supplied unsharded and sharded datasets
when credentials/network access are available; keep public S3 tests marked
`network` and outside the default suite.

### 5. Validation and handoff

Use the `blosc2` conda environment for all Python, installation, and tests.
Run focused adapter/RemoteProxy/Proxy/fsspec tests first, then the default suite
and repository lint checks after integration. Use the established pytest fixtures
and parametrization rather than adding a new test framework.

Validate optional imports in a subprocess with Zarr imports blocked, and with
fsspec blocked for unrelated functionality. Verify the minimum supported Zarr
version in a suitable test environment before setting its requirement floor.
Record any dependency-version or network checks that could not run.

## Completion criteria

- A remote Zarr array opens as a RemoteProxy and produces correct slice values.
- Retained payloads are usable Blosc2 chunks with correct B2ND block layout.
- A warm hit performs no remote payload or metadata reads under the immutable
  contract; eviction and cache limits retain their existing semantics.
- DISK carriers and sparse caches reopen safely with the correct source and
  geometry; credentials are absent from persisted metadata.
- Zarr remains optional and missing dependencies produce actionable errors.
- An authorized server source cannot fall back to unrestricted transport.
- Existing Blosc2/Caterva2 source tests continue to pass.

## Deferred work

Mutable-store validation, per-object versions, refresh policies, and TTL are
future changes requiring a new explicit consistency contract. Direct reuse of
compatible compressed Zarr chunks, block-range conversion, shard batching,
custom codec optimizations, variable-length dtypes, ZIP sources, and hierarchy
browsing are deferred until a concrete workload needs them.

The ponytail choice is one source adapter using existing Zarr decoding and
Blosc2 caching. No cache engine, codec framework, or generalized plugin registry
is needed for this feature.
