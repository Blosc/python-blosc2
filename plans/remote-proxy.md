# Plan: Persistable `RemoteProxy`

## Motivation

Python-Blosc2 can already access remote B2ND arrays lazily:

```python
array = blosc2.open(
    remote_urlpath,
    lazy=True,
    cache_path="mycache.b2nd",
)
```

The resulting `mycache.b2nd` can be uploaded to Caterva2 and acts as a proxy,
but it is fundamentally a **persistent cache**.  As remote chunks are read,
the file acquires compressed data plus cache bookkeeping.  This is useful for
offline reuse, but it is not the right representation when the desired object
is only a small, immutable reference to a remote array.

The missing abstraction is a persistable proxy whose stored B2ND file contains
the remote source description and array geometry, but never becomes the data
cache.  When uploaded to Caterva2, reads should be forwarded to the referenced
Caterva2 or fsspec source under an explicit server-side security policy.

## Decision Summary

Introduce a new public `RemoteProxy` type rather than extending
`SimpleProxy` or broadening `C2Array`:

- `SimpleProxy` remains the generic, non-persistable adapter for arbitrary
  array-like Python objects.
- `C2Array` remains the direct Caterva2 client object and its existing
  persistence format remains supported.
- `Proxy` remains the implementation for reusable memory and disk caches.
- `RemoteProxy` is a persistable, backend-neutral remote reference with an
  explicit `CachePolicy`.

The public cache policy should be an enum:

```python
class CachePolicy(Enum):
    NONE = "none"
    MEMORY = "memory"
    DISK = "disk"
```

Python cannot parse `CachePolicy.None` because `None` is a keyword, so the
public spelling must be `blosc2.CachePolicy.NONE`.  Uppercase enum members also
match the enum style already used by Python-Blosc2.

## Implementation Status

The Python-Blosc2 client implementation on this branch covers the core
RemoteProxy design and the first four implementation phases:

- `CachePolicy.NONE`, `MEMORY`, and `DISK` are public and validated at runtime.
- `RemoteProxy` supports Caterva2 `URLPath`/`C2Array` sources and contiguous
  single-file fsspec B2ND URLs.
- `NONE` uses direct Caterva2 indexing or operation-scoped fsspec assembly;
  `MEMORY` and `DISK` reuse the existing `Proxy` machinery.
- Memory caches default to a 256 MiB post-operation compressed-payload bound;
  disk caches are unlimited by default and support the same optional LRU bound.
- Remote lazy open accepts explicit cache policies and limits while preserving
  the pre-existing `blosc2.open(..., lazy=True)` `Proxy` behavior when neither
  is supplied.
- Reference-only `remote_proxy` carriers, strict source/geometry validation,
  authentication omission, fsspec `Ref` values, LazyExpr operands, and
  ObjectArray/BatchArray msgpack round trips are implemented and documented.
- Client-side URL safety checks reject local filesystem URLs, chained fsspec
  URLs, user information, fragments, and credential-like query parameters.

The following remain future work or deliberate follow-ups:

- Opening a local reference carrier with an explicit runtime policy (for
  example, `blosc2.open("reference.b2nd", cache_policy=...)`) is not yet a
  convenience path. Reconstructing `RemoteProxy(reference.urlpath, ...)` is
  the current explicit opt-in.
- Cache-oriented `fetch()`/`afetch()` methods are not exposed on `RemoteProxy`
  under `NONE`; a separate materialization API can be designed later.
- Caterva2 server-side discovery, default-deny protocol/destination policy,
  credential selection, SSRF protection, resource limits, reference-cycle
  handling, and tenant isolation remain to be implemented in Caterva2.
- Pinned reference semantics, broader fsspec/server protocol allowlists, and
  any `C2Array.save(as_remote_proxy=True)` convenience are future decisions.

## Goals

The first implementation should allow this workflow:

```python
proxy = blosc2.RemoteProxy(
    "s3://example/dataset.b2nd",
    cache_policy=blosc2.CachePolicy.NONE,
)
proxy.save("dataset-proxy.b2nd")

# Upload dataset-proxy.b2nd to a suitably configured Caterva2 server.
```

It should also work for a Caterva2 source:

```python
proxy = blosc2.RemoteProxy(
    blosc2.URLPath(
        "@public/dataset.b2nd",
        urlbase="https://example.org/caterva2",
    ),
    cache_policy=blosc2.CachePolicy.NONE,
)
```

The saved file should:

- be metadata-sized rather than proportional to the remote array
- reopen as `RemoteProxy`
- preserve shape, dtype, chunks, blocks, and the safe source descriptor
- serve slices and chunks from either Caterva2 or supported fsspec URLs
- remain byte-for-byte unchanged after reads under `CachePolicy.NONE`
- never contain credentials

## Non-Goals For The First Version

- Persisting arbitrary `SimpleProxy` sources or Python callables.
- Serializing fsspec filesystem instances or arbitrary `storage_options`.
- Embedding authentication tokens, cookies, cloud keys, or signed credentials.
- Proxying every fsspec object type.  Initially support a single remote
  contiguous B2ND frame; remote directory stores such as `.b2d` can be added
  after their object and authorization semantics are designed.
- Making an uploaded descriptor safe merely through client-side URL
  validation.  Caterva2 must enforce its own resolution policy.
- Changing the existing on-disk cache proxy format.

## Why `RemoteProxy` Should Be A Separate Type

### Do not make `SimpleProxy` persistable

`SimpleProxy` accepts a broad array-like object with `shape`, `dtype`, and
`__getitem__`.  Most such objects have no stable or safe reconstruction recipe.
Making the class conditionally persistable only for Caterva2 and fsspec inputs
would give one public type two substantially different contracts and invite
accidental serialization of arbitrary Python state.

`RemoteProxy` can instead require a constrained, versioned source descriptor.
This makes persistence an invariant of the type rather than a special case.

### Do not make `C2Array` backend-neutral

`C2Array` models Caterva2 operations and authentication.  Teaching it about
fsspec URLs would mix the Caterva2 protocol with byte-range filesystem access.
It would also leave no natural home for the cache policy shared by both
backends.

### Reuse implementation, not identity

`RemoteProxy` should delegate to existing components:

- `C2Array` for Caterva2 reads
- `FsspecNDSource` for fsspec metadata, chunks, and byte ranges
- `Proxy` for reusable memory or disk caching

It should not duplicate those implementations, and it should not itself be a
subclass of `SimpleProxy` unless that inheritance remains strictly an internal
convenience with no effect on serialization.

## Public API

### Construction

Proposed primary constructor:

```python
proxy = blosc2.RemoteProxy(
    urlpath,
    cache_policy=blosc2.CachePolicy.NONE,
    cache_path=None,
    cache_dir=None,
    max_cache_bytes=None,
)
```

The omitted `max_cache_bytes` value uses the policy-dependent default; an
explicit `None` requests an unlimited MEMORY or DISK cache.

The constructor should discover the source kind and remote array metadata.  A
future explicit `source_kind=` escape hatch can be added if URL recognition is
ambiguous, but should not be needed initially.

`RemoteProxy` should expose at least:

- `shape`, `dtype`, `chunks`, `blocks`, and `cparams`
- `urlpath` and a read-only normalized source descriptor
- `cache_policy`
- synchronous `__getitem__` and `get_chunk()`
- asynchronous counterparts where the selected backend supports them
- traffic information compatible with the current remote-access diagnostics
- `to_cframe()` and `save(urlpath)`

It should participate as an expression operand in the same way as `C2Array`
and other remote array-like operands.

### Integration with `blosc2.open`

Once the explicit class is stable, extend the existing remote open path:

```python
blosc2.open(
    remote_urlpath,
    lazy=True,
    cache_policy=blosc2.CachePolicy.NONE,
)
```

Recommended compatibility mapping:

| Arguments | Effective policy | Result |
| --- | --- | --- |
| `lazy=True` only | `MEMORY` | Preserve current process-local lazy cache behavior |
| `lazy=True, cache_policy=NONE` | `NONE` | `RemoteProxy` with no retained data cache |
| `lazy=True, cache_policy=MEMORY` | `MEMORY` | `RemoteProxy` backed by the current memory `Proxy` |
| `lazy=True, cache_path=...` | `DISK` | Preserve current persistent-cache behavior |
| `lazy=True, cache_dir=...` | `DISK` | Preserve current persistent-cache behavior |
| `lazy=True, cache_policy=DISK, cache_path/cache_dir=...` | `DISK` | Explicit persistent cache |

Validation rules:

- `NONE` or `MEMORY` combined with `cache_path`/`cache_dir` is an error.
- `DISK` without a cache location is an error unless a documented automatic
  cache-location policy is deliberately introduced.
- `max_cache_bytes` is invalid with `NONE`, defaults to 256 MiB with `MEMORY`,
  and defaults to unlimited with `DISK`. A positive explicit value bounds
  either memory or disk cache payload. The implementation needs an internal
  sentinel to distinguish an omitted policy-dependent default from explicit
  `None`, which means unlimited.
- Supplying both `cache_path` and `cache_dir` continues to follow the existing
  validation rule.
- The Python API should require a `CachePolicy` instance rather than expose
  several string aliases.  The serialized payload uses stable lowercase string
  values so it is independent of Python enum internals.

The default constructor policy for an explicit `RemoteProxy` should be `NONE`,
because its defining purpose is a reference-only proxy.  The default for the
pre-existing `blosc2.open(..., lazy=True)` call should remain `MEMORY` to avoid
a silent performance regression.

## Precise Cache Semantics

The word "none" must describe retained cache state, not prohibit every
temporary buffer.  fsspec range reads need somewhere to assemble compressed
blocks for a slice.

### `CachePolicy.NONE`

- No fetched chunk or block is retained between independent operations.
- No fetched bitmap, chunk payload, or cache index is written to the carrier.
- The serialized carrier is never used as scratch space.
- Temporary buffers may exist in memory for the duration of one operation.
- If an operation requires an NDArray workspace, it is operation-scoped and
  discarded before returning.
- Repeating the same data read is expected to contact the remote source again.
- Metadata may be retained in the live Python object's immutable fields;
  reopening the carrier does not imply a data fetch.

### `CachePolicy.MEMORY`

- Fetched data may be retained for the lifetime of the Python object.
- Repeating a covered read should be served without remote data traffic.
- Retained compressed payload is limited to 256 MiB by default. The caller may
  select another positive `max_cache_bytes`, or explicitly request an
  unlimited cache with `None` through an API representation that distinguishes
  it from an omitted policy-dependent default.
- After each operation, least-recently-used chunks are evicted until the
  retained payload is within the bound.
- `save()` still writes a reference-only carrier with persisted policy `NONE`,
  not a snapshot of this process-local cache.
- Closing or dropping the object loses the cache.

### `CachePolicy.DISK`

- Reuse the existing persistent `Proxy` cache behavior and format.
- The disk cache is unlimited by default for compatibility and because its
  purpose is cross-process reuse. An explicit positive `max_cache_bytes`
  enables the same post-operation LRU bound as the memory cache.
- Eviction replaces whole cached chunks with `UNINIT`, clears their fetched
  bitmap entries, and shrinks the live `.b2nd` payload. Compact contiguous
  files may need to move later compressed data, so frequent disk eviction can
  be more expensive than memory eviction.
- The disk cache and a reference carrier are distinct concepts.
- `RemoteProxy.save()` should always save the reference-only representation and
  normalize the persisted policy to `NONE`. Memory and disk policies describe
  the live process, not portable behavior to impose on another machine.
- The configured cache lives at `cache_path` or under `cache_dir`; it is not the
  destination passed to `RemoteProxy.save()`.
- Existing `Proxy` cache files continue reopening as `Proxy`, not
  `RemoteProxy`.

This distinction prevents a supposedly portable descriptor from silently
growing or containing a partial snapshot because it happened to be read before
upload.

### Meaning and accounting of `max_cache_bytes`

The bound applies after an operation completes and covers retained compressed
cache payload, including partial chunks and duplicated hot partial-block
payloads. It does not bound:

- the compressed working set needed to complete the current operation
- in-flight concurrent responses
- decompression and assembly buffers
- the NumPy result returned to the caller
- total process RSS, because an allocator may retain freed arenas

Eviction must happen only after the requested result has been assembled. The
current proxy sequence fetches all required regions before reading the result
from its cache, so evicting during that fetch could discard an early chunk and
produce an incorrect result.

LRU granularity is one chunk even when only some blocks in that chunk are
cached. Cache hits refresh recency as well as remote fetches. A chunk larger
than the bound may be used for the current operation and then evicted, leaving
the retained payload below the limit.

For a reopened disk cache, exact recency from a previous process need not be
persisted initially. Existing fetched chunks are seeded in deterministic chunk
order as older than chunks touched by the new process. This preserves the
bound and correctness without rewriting LRU metadata after every read; it only
reduces eviction quality immediately after reopen.

## Source Model

Only explicitly supported, reconstructable source kinds should be serialized.
Extend the reference model with a versioned fsspec source kind while retaining
the existing Caterva2 reference kind.

Conceptually:

```python
RemoteSource = Caterva2SourceRef | FsspecSourceRef
```

Suggested descriptors:

```python
{
    "kind": "caterva2",
    "version": 1,
    "path": "@public/dataset.b2nd",
    "urlbase": "https://example.org/caterva2",
}
```

```python
{
    "kind": "fsspec",
    "version": 1,
    "urlpath": "s3://public-bucket/dataset.b2nd",
}
```

The source reference must contain only location and format information.  It
must not include headers, bearer tokens, passwords, signed query parameters,
filesystem objects, or arbitrary fsspec keyword arguments.

The current `Ref.from_object(Proxy)` behavior should not be changed as part of
this feature: persisted lazy-expression operands may rely on it unwrapping to
the proxy cache.  Add a dedicated remote-source encoder/resolver instead of
silently changing generic `Ref` semantics.

## Serialized B2 Object Format

Use the existing B2 object carrier mechanism with a new object kind,
`remote_proxy`.  The carrier is an empty, structurally valid NDArray containing
array geometry in its normal metadata and a versioned B2 object payload in
variable-length metadata.

Example payload:

```python
{
    "kind": "remote_proxy",
    "version": 1,
    "source": {
        "kind": "fsspec",
        "version": 1,
        "urlpath": "s3://public-bucket/dataset.b2nd",
    },
    "cache_policy": "none",
}
```

The carrier should include:

- shape and dtype
- chunk and block geometry
- compression parameters needed to interpret fetched chunks
- the `remote_proxy` payload

Version 1 always writes `"cache_policy": "none"`. This field makes the
reference-only behavior explicit and leaves room for future policy negotiation,
but a carrier must not request server memory or disk use based on the creating
process's runtime policy. A caller reopening locally can select a runtime cache
policy through an explicit future open override; without one, it remains
`NONE`.

It must not include:

- `proxy-source`
- `proxy-fetched` or `proxy-index`
- fetched compressed chunks
- access credentials or client configuration

`process_opened_object()` should dispatch `b2o.kind == "remote_proxy"` to the
new decoder.  The existing `proxy-source` check must continue to identify
legacy/current persistent cache proxies before generic B2 object dispatch.

### Source identity and mutation

Version 1 may optionally store a non-secret source stamp such as an ETag,
content length, or backend revision when available.  It should not rely on one
being available for every backend.

Recommended initial semantics are a **floating reference with structural
validation**:

- reads see the source's current contents
- the source must still match the carrier's shape, dtype, chunks, and blocks
- a mismatch raises a clear stale-reference error before returning data

A future pinned mode can require an exact source stamp for reproducibility.  It
should be a separate, explicit option rather than an accidental consequence of
metadata captured at creation time.

## Runtime Design

### Common `RemoteProxy` layer

`RemoteProxy` owns:

- the normalized source descriptor
- immutable array geometry captured in the carrier
- the selected `CachePolicy`
- a backend adapter
- optional memory/disk cache state according to policy
- the policy-dependent retained-cache limit and chunk-level LRU state

The public layer validates the source metadata against the carrier and provides
consistent indexing, persistence, traffic reporting, and error behavior.

### Caterva2 backend

For `NONE`, delegate slices and chunks directly to `C2Array`; no assembly cache
is necessary.  For `MEMORY` and `DISK`, either continue direct delegation when
it satisfies the operation or wrap it with the existing cache proxy machinery.
The policy must still determine whether results are reusable between calls.

The serialized source should reuse the existing `C2Array` reference fields,
including the rule that authentication tokens are not persisted.

### fsspec backend

Use `FsspecNDSource` for metadata, chunk, and range access.  Because this source
does not provide general `__getitem__`, slicing needs an assembly layer:

- `NONE`: construct an in-memory operation-scoped workspace for the requested
  slice/chunks, return the result, then discard the workspace
- `MEMORY`: use one in-memory `Proxy` cache retained by `RemoteProxy`
- `DISK`: use the existing persistent `Proxy` at the configured cache location,
  with optional post-operation LRU eviction

The first implementation should favor correctness and a clean policy boundary.
An optimized no-cache slice assembler can replace the operation-scoped proxy
later without changing the public or serialized formats.

### Fetch APIs

The first implementation deliberately keeps cache-oriented `fetch()` and
`afetch()` on `Proxy` rather than exposing them on `RemoteProxy`.  This avoids
an ambiguous operation under `CachePolicy.NONE`: indexing and `get_chunk()`
return results while retaining no reusable data, whereas a future named
materialization API can explicitly return a new `NDArray` without changing the
reference carrier.  Any such API must not silently convert a no-cache proxy
into a retained cache.

## Caterva2 Server Contract

Serving an uploaded `RemoteProxy` requires explicit support in Caterva2.  The
Python-Blosc2 carrier is only a descriptor; Caterva2 is responsible for deciding
whether and how it may be resolved.

Recommended request lifecycle:

1. Upload stores the carrier as an immutable reference object.
2. Catalog and shape/dtype inspection may use the carrier metadata without an
   outbound request.
3. On first data access, the server parses and validates the descriptor against
   its configured remote-source policy.
4. It resolves current remote metadata and verifies the carrier geometry.
5. It performs only the byte ranges or slices needed for the request.
6. Any temporary assembly state lives outside the uploaded dataset and is
   discarded for `NONE`.
7. The carrier's contents, size, and modification time remain unchanged.

An installation may optionally validate reachability at upload time, but that
cannot replace validation at read time: DNS, redirects, credentials, and the
remote object can all change later.

Until Caterva2 implements the security and resource controls below, it should
reject `remote_proxy` carriers by default rather than resolve arbitrary URLs.

## Security Requirements

An uploaded remote reference asks the server to make outbound requests chosen
by a client.  This is an SSRF and resource-exhaustion boundary, not merely a new
file type.

### Protocol and destination policy

Caterva2 should:

- deny all source protocols by default
- enable only administrator-configured protocols, for example `https` or `s3`
- reject local and process-oriented schemes such as `file`, `memory`, and
  arbitrary chained fsspec URLs
- optionally allow only configured hosts, ports, buckets, and key prefixes
- resolve hostnames and reject loopback, link-local, private, multicast, and
  cloud-instance-metadata destinations unless explicitly authorized
- recheck the destination after every redirect and cap redirect count
- protect against DNS rebinding by validating the actual connection target,
  not only the submitted hostname
- normalize URLs before policy checks to prevent parser or encoding bypasses

Support for fsspec protocol chaining such as archive-over-network URLs should
be out of scope initially because every layer expands the policy surface.

### Credentials

- Never serialize client credentials in the B2 object.
- Strip or reject user-info, sensitive query parameters, custom headers,
  cookies, tokens, and arbitrary `storage_options` at creation and upload.
- Configure server credentials out of band.
- Scope credentials to the smallest allowed host, bucket, and prefix.
- Select credentials from the validated destination, never from untrusted
  descriptor-provided provider names.
- Avoid reflecting secrets or sensitive internal response bodies in errors.

Public Caterva2 references should continue to work without a persisted auth
token.  Private references require credentials available to the resolving
server; client credentials cannot make an uploaded proxy portable safely.

### Resource limits

Caterva2 should configure and enforce:

- connection, read, and total request timeouts
- maximum redirects, range requests, retries, and concurrency per operation
- maximum bytes fetched for metadata and for one user request
- maximum rank, shape, logical `nbytes`, chunk count, and metadata size
- decompression and expansion limits before allocating output buffers
- per-user or per-tenant rate and bandwidth limits
- cancellation of upstream requests when the client request is cancelled

Carrier geometry is untrusted input and must be validated before multiplication
or allocation.

### Reference graphs

A remote target may itself be another proxy, possibly pointing back to the
original object.  The server must enforce:

- maximum proxy depth and total remote hops
- cycle detection using normalized source identities
- rejection of direct or indirect self-references
- one cumulative resource budget across the entire reference chain

### Tenant isolation and observability

- Never share authenticated sessions or memory caches across security
  principals unless the cache key includes the full authorization context.
- Log descriptor identity, resolved destination, bytes, request count, timing,
  and policy decision without logging secrets.
- Expose actionable but non-sensitive failures for denied destinations,
  unavailable credentials, stale geometry, and exhausted limits.

## Compatibility And Migration

- Existing persisted `C2Array` carriers keep their current kind and decoder.
- `C2Array.save()` remains unchanged initially; it may delegate internally in a
  later cleanup but should not silently start emitting `remote_proxy`.
- Existing persistent cache files using `proxy-source` keep their current
  format and reopen behavior.
- `SimpleProxy` remains non-persistable.
- Existing `blosc2.open(remote, lazy=True)` behavior remains memory-cached.
- Existing `cache_path` and `cache_dir` calls remain disk-cached.
- The new B2 object kind must fail clearly on older readers, as other unknown B2
  object kinds do, without being mistaken for an ordinary empty NDArray.
- Stored policy values are stable lowercase strings; decoder code maps these to
  enum members and rejects unknown values rather than guessing.

## Proposed Code Organization

### New module

Add `src/blosc2/remote_proxy.py` containing:

- `RemoteProxy`
- source normalization and safe descriptor validation used by the client
- backend selection
- policy-specific runtime adapters
- B2 object payload encode/decode helpers where this avoids import cycles

Client-side validation improves error messages and prevents accidentally
writing credentials, but must be documented as distinct from Caterva2's
authoritative server-side policy.

### Existing modules

- `src/blosc2/__init__.py`
  - define or re-export `CachePolicy`
  - export `RemoteProxy`
  - add both to `__all__`
- `src/blosc2/ref.py`
  - add a versioned fsspec reference representation or a dedicated
    remote-source reference helper
  - do not change generic `Proxy` unwrapping semantics
- `src/blosc2/b2objects.py`
  - encode and decode the `remote_proxy` B2 object kind
  - build the metadata-only carrier
- `src/blosc2/schunk.py`
  - recognize the new B2 object during `blosc2.open()` dispatch
  - preserve precedence of persistent cache-proxy detection
- `src/blosc2/proxy.py`
  - expose reusable internal assembly/cache pieces only as needed
  - leave `SimpleProxy`'s public contract unchanged
- `src/blosc2/c2array.py`
  - expose any small backend-neutral hooks needed by `RemoteProxy`
  - retain its existing public persistence behavior

If defining `CachePolicy` in `__init__.py` creates import cycles, place it in a
small non-private core module and re-export it from `blosc2`.  The public name
and enum values are the compatibility surface, not its physical module.

## Implementation Phases

### Phase 0: Settle contracts

- Confirm that `NONE` means no reusable data across independent operations,
  while permitting operation-scoped memory.
- Decide `fetch()`/`afetch()` behavior under `NONE`.
- Confirm floating-reference semantics and structural validation.
- Agree on the Caterva2 protocol/host policy and initially supported fsspec
  schemes.
- Version and document the `remote_proxy` payload before writing code.

### Phase 1: Enum and source descriptors

- Add and export `CachePolicy` with `NONE`, `MEMORY`, and `DISK`.
- Add normalized Caterva2 and fsspec descriptor creation.
- Reject secrets and unsupported URL constructions.
- Add round-trip tests for descriptors and enum payload values.

### Phase 2: Runtime `RemoteProxy`

- Implement construction and metadata discovery.
- Implement Caterva2 reads for all policies.
- Implement fsspec reads with operation-scoped assembly for `NONE`.
- Reuse existing memory and persistent `Proxy` caching for the other policies.
- Add traffic accounting and synchronous/asynchronous behavior.
- Verify source geometry before serving data.
- Add shared chunk-level LRU accounting and post-operation eviction for bounded
  memory and disk caches.

### Phase 3: Persistence

- Add `remote_proxy` carrier encoding, `to_cframe()`, and `save()`.
- Add open-time dispatch and decoding.
- Guarantee reference-only persistence independently of runtime cache state,
  normalizing the saved policy to `NONE`.
- Verify that reads never mutate a `NONE` carrier.

### Phase 4: `blosc2.open` integration

- Add `cache_policy` to the remote lazy-open path.
- Preserve old defaults and infer `DISK` from existing cache-location
  arguments.
- Add conflict validation and focused regression tests.
- Keep the explicit `RemoteProxy` constructor available so descriptor creation
  does not depend on overloaded `open()` behavior.

### Phase 5: Caterva2 support

- Add `remote_proxy` discovery and read dispatch to Caterva2.
- Implement default-deny protocol and destination configuration.
- Add credential selection outside the descriptor.
- Add SSRF, redirect, DNS, resource-limit, cycle, and tenant-isolation tests.
- Confirm the uploaded carrier is never mutated.

This phase may live in the Caterva2 repository, but the feature should not be
presented as safe for arbitrary uploads until both sides are complete.

### Phase 6: Documentation and examples

- Add a `RemoteProxy` API page and include it in the reference toctree.
- Document the three cache policies and their lifetime guarantees.
- Add examples for a public Caterva2 source and an allowed fsspec HTTPS/S3
  source.
- Document that private sources use server-side credentials.
- Add a Caterva2 administrator guide for the security policy and operational
  limits.

## Test Plan

### Unit tests

- `CachePolicy` exports and serialized values.
- `RemoteProxy` metadata and indexing for mocked Caterva2 and fsspec sources.
- Descriptor normalization and rejection of credentials/unsupported schemes.
- B2 object cframe and file round trips for each supported source kind.
- Unknown payload versions, source kinds, and policy values fail clearly.
- Source geometry changes produce a stale-reference error.
- `RemoteProxy` works as a lazy-expression operand.

### Cache-policy tests

- Under `NONE`, two identical reads each cause remote data traffic.
- Under `NONE`, carrier size, bytes, mtime, and metadata are unchanged after
  reads.
- Under `MEMORY`, an identical covered second read causes no remote data
  traffic within the same object lifetime.
- The default memory cache retains at most 256 MiB of compressed payload after
  each operation.
- A bounded memory cache evicts least-recently-used chunks and refetches them
  when accessed again.
- Reopening a `MEMORY` carrier starts with an empty memory cache.
- Under `DISK`, a covered read survives close/reopen through the configured
  cache path.
- A disk cache is unlimited by default; with an explicit bound its live payload
  and file size fall after LRU eviction, allowing only fixed metadata overhead.
- Reopening a bounded disk cache preserves the limit and seeds deterministic
  recency without requiring persistent timestamps.
- `RemoteProxy.save()` remains reference-only after memory or disk cache use.
- Invalid policy/cache-location combinations raise deterministic errors.

### Compatibility tests

- Existing `C2Array` cframes/files still reopen as `C2Array`.
- Existing `proxy-source` cache files still reopen as `Proxy` and preserve
  fetched state.
- Existing remote `lazy=True` calls retain their current default caching.
- Existing lazy-expression serialization involving `Proxy` operands is
  unchanged.
- The default non-network suite uses local mocks or a local range-capable HTTP
  fixture; real services remain under the `network` marker.

### Caterva2 security tests

- Default-deny behavior for all remote descriptors.
- Allowed public destination succeeds.
- Local file, loopback, private/link-local addresses, cloud metadata endpoints,
  forbidden ports, and disallowed buckets/prefixes are rejected.
- Redirect from an allowed URL to a denied destination is rejected.
- DNS rebinding or changed resolution is rejected at connection time.
- User-info, sensitive query data, custom credentials, and chained fsspec
  protocols are rejected.
- Cycles and excessive reference depth are rejected.
- Byte, range, concurrency, timeout, decompression, and allocation limits are
  enforced.
- One tenant cannot observe or reuse another tenant's authenticated cache or
  session.

## Acceptance Criteria

The feature is ready when all of the following hold:

1. A user can persist and reopen a `RemoteProxy` for both Caterva2 and one
   supported fsspec-backed single-file B2ND source.
2. A `NONE` carrier stays metadata-sized and byte-for-byte unchanged after any
   supported read.
3. Repeated reads demonstrate observably different traffic behavior for
   `NONE`, `MEMORY`, and `DISK`.
4. Bounded memory and disk caches evict whole least-recently-used chunks after
   each operation without affecting the returned result; the default bound is
   256 MiB for memory and unlimited for disk.
5. `RemoteProxy.save()` never serializes fetched data or credentials.
6. Legacy C2Array and persistent cache-proxy files keep their behavior.
7. Caterva2 rejects remote proxies by default and resolves them only through an
   administrator-controlled destination and credential policy.
8. Geometry changes, unavailable credentials, denied destinations, and
   resource-limit failures produce clear errors.
9. API and administrator documentation explain both caching semantics and the
   outbound-request security boundary.

## Future Considerations

- Which fsspec schemes should a Caterva2 server implementation support?  A
  narrow starting set such as HTTPS and S3 is preferable to arbitrary plugins.
- When should pinned references be added, and what exact mismatch exception
  should they raise?
- Should a future `C2Array.save(as_remote_proxy=True)` convenience exist, or is
  the explicit `RemoteProxy(c2array.urlpath)` conversion clearer?
- Should local URLs remain accepted by Python-Blosc2 for testing while
  Caterva2 rejects them, or should client-side construction enforce remote-only
  schemes everywhere?

## Initial Slice (Completed)

The smallest end-to-end path was implemented as follows:

1. `CachePolicy` and explicit `RemoteProxy` construction.
2. Public, single-file HTTPS/fsspec B2ND source.
3. `CachePolicy.NONE` with operation-scoped in-memory assembly.
4. `remote_proxy` cframe/file round trip.
5. Local mocked tests proving the carrier is immutable and repeated reads do
   not reuse data.

Caterva2-source support and memory/disk policy integration are also implemented
in the Python client.  Caterva2 server resolution remains a separate follow-up
behind a default-deny configuration, as described in Phase 5.
