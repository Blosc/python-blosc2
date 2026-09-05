# Plan: Self-Caching RemoteProxy

## Status

Superseded by remote-proxy-v3.md for the Python client API and cache policies.

Implemented and validated in Python-Blosc2 and Caterva2. The product and
persistence contracts are settled; the phase checklist and test matrix below
describe the delivered version-1 behavior.

## Purpose

Allow Caterva2 to host a small `.b2nd` proxy for a public remote B2ND array.
When Caterva2 reads missing regions, the proxy may retain the fetched compressed
chunks inside that same `.b2nd` file, up to a configured storage limit.

The proxy file is therefore a mutable persistent cache, not an immutable remote
reference plus a separate server cache.

## Scope

The first version supports:

- direct RemoteProxy `.b2nd` files
- one public, contiguous B2ND source available over HTTPS
- public S3 objects through their HTTPS object URLs
- operation with embedded caching disabled or enabled
- a bound on compressed chunk payload retained in each proxy file
- reuse of cached chunks after process and server restarts
- warm-carrier and cache-free export/download forms
- invalidation when the remote object identity changes
- Caterva2's existing `api/info`, `api/fetch`, `api/chunk`, and physical
  download endpoints

The first version does not support:

- credentials, signed URLs, cookies, custom headers, or private sources
- native `s3://` resolution in Caterva2
- redirects or arbitrary fsspec protocol chains
- remote references embedded in expressions or other object graphs
- cumulative network-byte or request-count budgets
- a new Caterva2 HTTP endpoint for proxy creation or cache management
- a separate RemoteProxy-specific server quota

Ordinary connection timeouts, remote-read concurrency, destination policy, and
structural validation remain in scope.

## Core Decisions

### The carrier is the cache

The uploaded `.b2nd` contains both:

- stable identity: versioned source descriptor and array geometry
- mutable state: fetched compressed chunks, source stamp, fetched bitmap,
  compressed-size accounting, and LRU bookkeeping

Reads may change the file's contents, size, modification time, and cache
metadata. The source descriptor and geometry must not change as a consequence
of reading or eviction.

### Cache data is disposable

Cached chunks are an optimization, never the authority for array identity.
Deleting every cached chunk leaves a valid proxy. If the source identity changes,
the cache is emptied logically before any old chunk can be served.

### Reuse the existing Proxy cache engine

`blosc2.Proxy` already implements the required persistent-cache mechanisms:

- UNINIT chunks for cache misses
- chunk/block fetched bitmaps
- remote source stamps
- compressed-byte accounting
- whole-chunk LRU eviction
- reopening and adopting a valid cache

RemoteProxy should reuse these mechanics with its own carrier passed as the
Proxy cache. It should not implement a second cache engine.

### RemoteProxy has no memory-cache policy

RemoteProxy supports only `CachePolicy.NONE` and `CachePolicy.DISK`. A retained
RemoteProxy cache belongs in its carrier, so a `MEMORY` policy would violate the
self-caching artifact model and could multiply memory use across Caterva2
workers and hosted proxies.

Remove `CachePolicy.MEMORY` while the API is still unreleased. Existing
`blosc2.open(remote, lazy=True)` process-local memory caching continues to
return the established generic `Proxy`; users who explicitly need an ephemeral
memory cache can also construct that type directly. Caterva2 therefore never
needs to translate, reject, or impose a quota on a persisted memory-cache
policy because no valid RemoteProxy carrier can contain one.

### The proxy specifies caching; the customer quota bounds storage

The persisted proxy specifies whether it retains misses and its maximum cached
compressed payload. This behavior travels with the proxy instead of being
selected independently by each Caterva2 server.

Each customer owns a virtual Caterva2 server whose users share one state
directory and one server-wide quota. Embedded proxy-cache growth is charged to
that existing customer quota just like other stored data. No user-level
attribution or separate RemoteProxy quota is needed.

When the customer quota cannot accommodate another cached chunk, Caterva2 still
serves the remotely fetched result but skips retaining that chunk. A proxy with
cache policy `NONE` never retains misses. Already embedded valid chunks remain
readable without an outbound fetch.

### Public HTTPS remains the network boundary

Caterva2 keeps the security work already implemented:

- remote resolution disabled by default
- exact administrator host allowlists
- HTTPS only
- public-address validation and DNS pinning
- redirects disabled
- URL credentials, queries, and fragments rejected
- descriptor inspection before generic Blosc2 object decoding
- carrier/source geometry validation

These controls remain necessary because cache misses still cause outbound
requests selected by uploaded data.

## Carrier Format

The RemoteProxy format has not been released, so define version 1 directly as
the self-caching format. No upgrade path or compatibility contract is needed
for the earlier development-only reference format:

```python
{
    "kind": "remote_proxy",
    "version": 1,
    "source": {
        "kind": "fsspec",
        "version": 1,
        "urlpath": "https://datasets.example.org/array.b2nd",
    },
    "cache_policy": "disk",
    "max_cache_bytes": 268435456,
}
```

The ordinary B2ND chunk slots form the cache. Proxy-owned variable metadata
records the source stamp, fetched bitmap, cached sizes, and any persisted index
needed by the remote reader.

Do not serialize:

- server cache limits or server policy
- local paths
- credentials or request configuration
- locks, sessions, or live filesystem objects

Do not add `proxy-source` metadata merely to trigger the legacy open path. The
`remote_proxy` B2 object kind remains the authoritative discriminator; its
decoder can construct a `Proxy` over the carrier after the source has been
resolved safely.

## Python-Blosc2 Behavior

### Creation

Creating and saving a `NONE` RemoteProxy produces a metadata-sized carrier whose
data chunks are UNINIT:

```python
proxy = blosc2.RemoteProxy("https://datasets.example.org/array.b2nd")
proxy.save("array-proxy.b2nd")
```

To create a self-caching carrier directly, select `DISK`, its path, and an
optional finite limit:

```python
proxy = blosc2.RemoteProxy(
    "https://datasets.example.org/array.b2nd",
    cache_policy=blosc2.CachePolicy.DISK,
    cache_path="array-proxy.b2nd",
    max_cache_bytes=256 * 2**20,
)
```

Opening that carrier in append mode uses the carrier itself as the persistent
cache; it does not require a second `cache_path`. Read-only mode can consume
warm chunks but does not retain misses.

### Reads

For each requested chunk or block:

1. Validate or refresh the remote source identity.
2. Serve a valid fetched entry from the carrier when available.
3. Fetch missing compressed bytes from the authorized source.
4. Return the requested logical result.
5. If writes are enabled, retain fetched chunks and enforce the configured cap.

Eviction replaces complete least-recently-used chunks with UNINIT and updates
the fetched bitmap atomically enough that an interrupted write cannot cause an
unfetched chunk to be trusted.

### Saving, CFrames, and downloads

By default, `save()` and `to_cframe()` serialize the current physical carrier,
including valid warm cache data. Downloading the proxy `.b2nd` from Caterva2
likewise returns the physical self-caching carrier with its warm chunks. A proxy
with no retained data remains metadata-sized. Credentials and runtime server
policy are never included.

This is distinct from `api/fetch`: fetching returns the requested logical array
or slice, not the physical proxy/cache file. A concurrent physical download must
take the carrier lock or copy under that lock so the downloaded B2ND is
internally consistent.

An explicit `include_cache=False` option produces a cold carrier without
mutating the warm source proxy:

```python
proxy.save("cold-proxy.b2nd", include_cache=False)
frame = proxy.to_cframe(include_cache=False)
```

Caterva2 exposes the same choice as an optional `include_cache=false` parameter
on its existing physical download operation. Cold export preserves descriptor,
geometry, compression parameters, cache policy, and cache limit, while replacing
cached chunks with UNINIT and clearing fetched bitmaps, cached-size/LRU state,
stored remote indexes, and source stamps. It does not contact the remote source.

### Source changes

ETag or another stable source token is stored with the cache. Before serving a
cached entry after reopening, compare the current source token with the stored
token. A mismatch clears the fetched bitmap and accounting before reading data.
Geometry mismatch remains a hard error rather than an automatic rewrite of the
carrier's identity.

If an HTTPS source supplies no stable validator, persistent reuse across
independent opens is unsafe. Such a source may be read without retention, or its
cache must be treated as empty on every new open.

## Caterva2 Behavior

### Existing API surface

No new HTTP endpoint is needed:

- upload stores the proxy through the existing upload path
- `api/info` reads local geometry and descriptor metadata without an outbound
  request
- `api/fetch` resolves the source under policy and serves slices or indices
- `api/chunk` can serve a compressed chunk while applying the same cache and
  quota rules
- physical download accepts `include_cache=false` to export a cold copy

The physical carrier must not be mistaken for a complete materialized array
frame when serving a whole logical-array fetch.

### Configuration and customer quota

The existing remote-source security configuration remains:

```toml
[server.remote_proxy]
enabled = true
allowed_hosts = ["datasets.example.org"]
timeout = 30
max_concurrency = 8
```

No Caterva2-specific cache limit is added. `max_cache_bytes` belongs to the
proxy payload and measures retained compressed chunk payload for that carrier;
small fixed metadata/index overhead is excluded. A `DISK` proxy must specify a
positive finite limit.

Caterva2's existing server-wide `quota` is the aggregate storage bound for one
customer's virtual server. Automatic proxy fills must join the same disk-usage
checking and accounting path currently used by uploads and explicit chunk
writes. The check must cover concurrent fills rather than merely noticing the
larger file during a later state-directory scan.

### Opening and resolution

Caterva2 continues to inspect the carrier before `blosc2.open()` can resolve an
untrusted source. After HTTPS authorization and geometry validation, it opens
the carrier in the appropriate mode and constructs the existing Proxy cache
engine over the secure `FsspecNDSource`.

- `NONE`: misses are not retained
- `DISK`: the carrier is opened append/write; misses populate it and its own LRU
  eviction enforces `max_cache_bytes`
- customer quota exhausted: the result is served, but a miss is not retained

### Concurrency

Because reads may now write, Caterva2 must synchronize access per carrier. The
safe initial rule is one active cache-mutating operation per carrier, including
across server workers. Read-only metadata inspection need not take the write
lock. Locking must cover source-stamp validation, fetched-bitmap changes, chunk
writes, eviction, and bookkeeping persistence.

The lock must follow Caterva2's existing dataset mutation/locking conventions
where possible. Process-local Python locks alone are insufficient when multiple
workers can open the same file.

### HTTP metadata and ETags

A cache fill changes the physical file mtime but not the logical remote-array
identity. API validators must not accidentally present cache churn as a user
dataset edit. The implementation must distinguish:

- physical carrier identity, relevant when downloading or backing up the proxy
- logical array identity, derived from descriptor plus current source stamp

This can initially be conservative by changing the API ETag after cache writes,
but clients must never receive stale logical data. A stable logical ETag is a
later optimization. `api/info` exposes the portable `b2o` descriptor but not
binary fetched bitmaps, source stamps, or other cache-engine bookkeeping.

## Changes To The Current Caterva2 Slice

Keep:

- policy configuration and default deny
- raw pre-resolution carrier inspection
- HTTPS validation, public DNS pinning, and disabled redirects
- geometry/rank/chunk validation
- existing `api/info` and `api/fetch` dispatch
- rejection of embedded remote references

Replace or revise:

- replace operation-scoped `ServerRemoteProxy` assembly with a Proxy backed by
  the uploaded carrier
- remove the requirement that persisted policy is `none`
- open data requests with controlled write access when caching is enabled
- replace byte-for-byte immutability assertions with bounded-cache assertions
- document physical mutation and source-stamp invalidation

## Implementation Phases

### Phase 0: Format and cache prototype

- [x] Finalize the version-1 self-caching payload and metadata invariants.
- [x] Prototype `Proxy(src, _cache=carrier, _max_cache_bytes=...)` over a RemoteProxy
  carrier.
- [x] Confirm eviction reclaims physical storage with bounded overhead.

### Phase 1: Python self-caching carrier

- [x] Make the decoded RemoteProxy retain its carrier.
- [x] Reuse the carrier as the Proxy cache for persistent mode.
- [x] Preserve and serialize fetched bitmap, stamp, sizes, and LRU state.
- [x] Make `save()` and `to_cframe()` include valid warm chunks.
- [x] Add non-destructive `include_cache=False` cold exports.

### Phase 2: Caterva2 cache integration

- [x] Replace no-retention resolution with the carrier-backed Proxy.
- [x] Integrate automatic cache fills with the existing customer-server quota
  checks and accounting; skip retention when no quota remains.
- [x] Add per-carrier cross-worker locking.
- [x] Keep metadata inspection local and source resolution default-deny.
- [x] Validate persisted cache policy and require a finite positive per-proxy limit.

### Phase 3: Tests and documentation

- [x] Update Python and Caterva2 API documentation.
- [x] Add cold-read, warm-read, restart, eviction, and source-change tests.
- [x] Add warm and cold physical download/export tests.
- [x] Add concurrent-fill and interrupted-write tests.
- [x] Verify default-deny and HTTPS security tests still pass.
- [x] Verify existing legacy Proxy caches still reopen in the full suites.

## Test Matrix

- Empty carrier is metadata-sized and reports complete local geometry.
- First read fetches and embeds the required compressed chunks.
- Repeated covered read causes no remote data traffic.
- Reopening the same file reuses its embedded cache.
- A `NONE` proxy leaves carrier bytes and mtime unchanged.
- Bounded caching evicts whole LRU chunks and stays within payload limit plus
  documented fixed overhead.
- `save()`/`to_cframe()` preserve valid warm chunks.
- `include_cache=False` exports a metadata-sized cold carrier without mutating
  the warm proxy or contacting its source.
- Physical download preserves valid warm chunks by default and supports a cold
  copy, while `api/fetch` returns logical array data.
- Changed source stamp invalidates all old cached entries before use.
- Changed source geometry raises without rewriting carrier identity.
- Missing source validator cannot silently reuse cache across opens.
- Concurrent reads cannot corrupt chunks, bitmaps, or accounting.
- Cache fills count against the customer virtual server's existing quota; when
  it is exhausted, reads succeed without retaining additional chunks.
- Caterva2 default deny, destination allowlist, DNS pinning, redirect rejection,
  and credential rejection remain effective.
- Existing info/fetch clients need no endpoint changes.

## Acceptance Criteria

1. A RemoteProxy `.b2nd` is both the portable descriptor and its persistent
   bounded cache.
2. Caterva2 can host it through existing info/fetch APIs.
3. Public HTTPS cache misses are resolved only through server policy.
4. Warm chunks survive restart and avoid remote data traffic.
5. The proxy's compressed-payload cap is enforced by whole-chunk LRU
   eviction.
6. Cache mutation never changes source descriptor or geometry.
7. Replaced sources cannot cause stale cached data to be served.
8. Concurrent requests cannot corrupt the carrier.
9. No credentials or server-specific runtime configuration are serialized.
10. Existing persistent Proxy caches retain documented compatibility.
11. RemoteProxy exposes no memory-cache policy; existing generic Proxy memory
    behavior remains unchanged.
12. Users can download or export a cold proxy without altering its warm carrier.

## Settled Product Decisions

No RemoteProxy format has shipped, so payload version 1 is the self-caching
format and has no upgrade path. RemoteProxy accepts only `NONE` and `DISK`;
`MEMORY` is removed before release. Ordinary `save()`, `to_cframe()`, and
physical proxy downloads preserve valid warm cache data by default. Explicit
`include_cache=False` creates or downloads a cold copy without modifying the
hosted proxy. Logical `api/fetch` operations continue to return array data.
