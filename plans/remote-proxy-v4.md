# Plan: Caterva2 executes MEMORY RemoteProxy carriers without retained caching and supports unbounded DISK cache (v4)

## Status and decision

Implemented in Caterva2 and Python-Blosc2. This extends v3's client API,
replaces its server rejection of MEMORY carriers, and adds support for unbounded
persistent DISK caching (`max_cache_bytes=None`). It does not supersede the
existing HTTPS security boundary.

1. Keep Python-Blosc2's lazy remote-open default at MEMORY with a 256 MiB retained
   compressed-payload limit. Accept persisted MEMORY carriers in Caterva2, but
   execute them using the same no-retention path as NONE. Preserve the original
   MEMORY policy and limit in the uploaded file and its downloads.
2. Support `max_cache_bytes=None` for `CachePolicy.DISK` in both Python-Blosc2 and
   Caterva2. Passing `None` disables LRU cache eviction (unbounded cache size).
   On Caterva2, an unbounded DISK cache operates without eviction when no server
   customer quota is set, and is clamped to remaining capacity (`retained + available`)
   when a customer storage quota is configured.

No server memory-cache registry, aggregate memory-cache quota, new configuration
knob, carrier format version, or Python-Blosc2 runtime-construction hook is needed.
This removes retained server data caching for MEMORY, not temporary memory use.

## Inspected baseline

Inspected on 2026-09-05:

- Python-Blosc2 checkout: `/Users/faltet/blosc/python-blosc2`, HEAD
  `dfb193e706f4f5801d8116cc4941aada8204cae9`, including the local v3 review changes.
- Caterva2 checkout: `/Users/faltet/ironArray/caterva2`, HEAD
  `7478babd6e66dd32413ebe20691f931a8355215c`. No tracked diff was reported there
  during inspection.

Caterva2 file paths below are relative to that repository. Function names are
the primary implementation anchors; line numbers may move.

### Current resolver and runtime construction

`caterva2/services/remote_proxy.py` already separates secure resolution from
runtime execution:

1. `inspect()` opens the physical carrier with `raw_carrier()`, inspects its B2
   marker and payload, and avoids generic B2-object decoding. Inspection uses
   carrier locks; a lock sidecar may be created even for observational access.
2. `_validated_source()` checks default-deny configuration, exact payload/source
   fields, format versions, cache policy, and URL restrictions. It currently
   accepts only NONE and DISK. DISK requires a positive integer limit, excluding
   booleans; NONE requires a null limit.
3. `resolve()` validates public DNS answers, creates a pinned HTTPS filesystem,
   and passes that exact filesystem to `blosc2.FsspecNDSource(..., _filesystem=fs)`.
   It validates source/carrier geometry and rank, logical-byte, and chunk limits.
4. It returns `ServerRemoteProxy(source, expected, carrier, payload)` directly.
   It does **not** construct a Python `blosc2.RemoteProxy` from a URL or decode
   its persisted policy through `RemoteProxy._from_payload()`.

`ServerRemoteProxy` currently copies the payload's `cache_policy` and
`max_cache_bytes`. Its `_backend()` uses a temporary
`blosc2.Proxy(self.src, _refresh_source=False)` whenever policy is not DISK or
the operation's disk-cache allowance is zero. That temporary cache is local to
the operation. `read()` assembles the result through it, while `get_chunk()`
reads directly from the authorized source on this path.

Only the DISK path opens the carrier in append mode and attaches it as `_cache`.
`current_cache_bytes()` already returns zero for non-DISK policies.

**Consequence:** accepting MEMORY in validation would already reach the desired
non-disk branch. Nevertheless, explicitly normalizing the runtime policy makes
the contract visible and avoids relying on an accidental "anything but disk"
fallback.

### API, quota, metadata, and download integration

- `services/server.py::open_b2()` calls `inspect()` and `resolve()` before generic
  `blosc2.open()`, translating resolution denial to HTTP 403.
- `remote_proxy_cache_limit()` returns zero for non-DISK runtimes. DISK remains
  subject to the carrier cap and existing customer storage quota.
- `read_remote_proxy()` applies dataset locking and disk-growth accounting to
  slice/index reads. The `api/chunk` branch similarly obtains a cache allowance
  and invokes `ServerRemoteProxy.get_chunk()` in a worker thread.
- `services/srv_utils.py::read_metadata()` inspects the raw carrier without
  resolving its source. `api/info` marks RemoteProxy data as
  `accept_ranges="none"` and exposes only its portable `b2o` variable metadata,
  removing cache-engine bookkeeping from the response.
- The physical download path inspects the carrier and calls `export_cframe()`
  under locks. Warm export serializes the raw carrier; cold export constructs a
  cold carrier with the original payload. Neither resolves the source.

These paths already support the separation between logical reads and physical
downloads required by this proposal.

## Behavior contract

| Stored policy | Stored limit | Caterva2 effective policy | Retained server data | Python reopen |
| --- | --- | --- | --- | --- |
| NONE | null | NONE | None | NONE |
| MEMORY | positive integer; default 268435456 | NONE | None between operations | MEMORY with original limit |
| DISK | positive integer | DISK | Carrier cache within cap and quota | DISK with positive limit |
| DISK | null (unbounded) | DISK | Carrier cache within customer quota (unbounded if no quota) | DISK with unbounded cache (no eviction) |

For a MEMORY carrier:

- Each independent read fetches the required upstream data again; there is no
  reusable server array-data cache across reads or requests.
- Temporary compressed assembly, metadata, transport state, decompression
  buffers, and output may exist. Do not assert that literally no memory is used
  or that every HTTP request is duplicated identically across operations.
- The stored positive limit is validated and preserved, but does not set a
  runtime memory budget or allocate that amount on the server.
- Neither reads nor quota handling populate the physical carrier or change its
  payload. Carrier bytes, size, and mtime remain unchanged by those reads;
  lock-sidecar activity is outside that invariant.
- Physical downloads preserve the requested policy and limit. A cold download
  also preserves them; `include_cache=false` is not a policy override.
- Embedded chunk data in a crafted MEMORY carrier is ignored for logical reads.
  Physical warm download can preserve those original bytes; cold download drops
  cache state. Do not trust them merely because the carrier is structurally valid.
- Actual source replacement between API requests is observed through fresh
  resolution. V4 does not promise an atomic snapshot across multiple upstream
  reads or add source refresh inside a long-lived ServerRemoteProxy instance.

For an unbounded DISK carrier (`max_cache_bytes=None`):

- In Python-Blosc2, chunks fetched into the carrier are retained without LRU
  eviction (`proxy.cache_bytes` reflects `carrier.schunk.cbytes`).
- In Caterva2, `_validated_source()` accepts `max_cache_bytes: null` for `disk`.
- When server customer storage quota is enabled, `remote_proxy_cache_limit()` clamps
  the effective limit to remaining capacity (`retained + available`), preventing
  the unbounded carrier from exceeding server quota. When quota is disabled, the
  proxy operates without a limit (`None`).
- `ServerRemoteProxy.current_cache_bytes()` falls back to `carrier.schunk.cbytes`
  when `proxy-cache-sizes` is not present, since `blosc2.Proxy` does not maintain
  an LRU size table for unbounded caches.
- Exporting a DISK proxy to DISK preserves `max_cache_bytes=None`; exporting to
  MEMORY falls back to `DEFAULT_DISK_CACHE_BYTES` (since MEMORY requires a finite
  positive integer limit); exporting to NONE sets `max_cache_bytes=None`.

## Implementation steps

### 1. Accept and validate MEMORY descriptors

In `_validated_source()`:

- Keep the exact field, version, source-kind, and default-deny checks.
- Validate both `"memory"` and `"disk"` using the same positive-integer rule.
  Reject missing/null limits, booleans, floats, strings, zero, and negatives.
- Keep `"none"` restricted to a null limit and reject unknown policy strings.
- Update denial messages so MEMORY is recognized rather than described as an
  unsupported policy.
- Preserve the input payload; do not rewrite it to NONE or remove its limit.

All URL checks still apply to MEMORY: public credential-free HTTPS only, exact
host allowlist, no query or fragment, no user information, and no custom source
fields. Client support for runtime signed URLs does not extend server support.

### 2. Make runtime policy explicit in ServerRemoteProxy

Use the existing constructor with the already authorized source. Proposed
initialization, after validation by `resolve()`:

```python
self.requested_cache_policy = payload["cache_policy"]
self.requested_max_cache_bytes = payload["max_cache_bytes"]
self.cache_policy = (
    "none" if self.requested_cache_policy == "memory"
    else self.requested_cache_policy
)
self.max_cache_bytes = (
    self.requested_max_cache_bytes if self.cache_policy == "disk" else None
)
```

Here `cache_policy` remains the execution-facing attribute used by existing
backend and quota branches. Optionally provide a read-only
`effective_cache_policy` alias for diagnostics; do not introduce two mutable
execution-policy fields. Constructor documentation must distinguish requested
and effective values. If adding a mapping helper, reject unknown policies rather
than silently converting them to NONE, and reuse it for optional diagnostics.

Do not mutate `payload`, `carrier.schunk.vlmeta`, or any process-global resolver.
Do not call `blosc2.RemoteProxy(url)` or `blosc2.open(url)` to perform conversion:
that could recreate the source outside the authorized filesystem boundary.

Keep `_backend()`, `read()`, `get_chunk()`, and `current_cache_bytes()` on their
existing non-DISK paths. A temporary generic Proxy for slice assembly is correct
NONE behavior; no persistent MEMORY backend is ever attached to the runtime.

### 3. Audit server consumers without adding cache infrastructure

Confirm all branches consuming `ServerRemoteProxy.cache_policy` use the effective
value. Preserve `remote_proxy_cache_limit()`'s zero result for effective NONE.
Keep existing dataset and carrier locking; removing locks is not part of v4.
Exercise both full/sliced/fancy-index fetches and compressed chunk requests.

No automatic cache-growth charge should occur for MEMORY reads. Ordinary upload
storage and existing lock files remain subject to current server accounting.
Keep DISK quota enforcement, invalidation, and serialization unchanged.

### 4. Preserve metadata and exports

Keep `api/info`'s stored `b2o` descriptor untouched and `accept_ranges="none"`.
Do not insert requested/effective fields inside `b2o`: Python-Blosc2 validates
that payload's exact field set.

Recommended initial scope: expose requested/effective values on the internal
runtime and document their mapping; keep the public metadata schema unchanged.
Public diagnostics can be a follow-up. If included now, add a declared optional
response-model field outside the portable descriptor, with client compatibility
tests. Describe it as the configured execution mapping, not proof that resolution
is enabled or that a particular URL is authorized. Metadata must remain local
and usable when resolution is disabled.

Physical downloads must continue using `export_cframe(carrier, original_payload)`
and must never serialize a normalized runtime object or normalized payload.

### 5. Documentation and compatibility

Update Caterva2 `doc/utilities/cat2-server.md` and comments in
`caterva2-server.sample.toml`: MEMORY carriers are accepted under the same source
policy but execute without retained caching. No new TOML option is introduced.

Update Python-Blosc2 `doc/guides/remote_arrays.md`,
`doc/reference/remoteproxy.rst`, and the status of the v3 server-policy section
when the server change ships. Explain client/server performance differences and
the unchanged 256 MiB client default. Describe older Caterva2 servers as still
rejecting MEMORY resolution, rather than claiming universal support.

No B2 object version bump is needed: MEMORY is already a Python-Blosc2 v3 policy.
Check Caterva2's declared Python-Blosc2 dependency and release floor before
shipping. Reopening downloaded MEMORY carriers requires a client version with
MEMORY support. Do not infer that version solely from `hasattr(RemoteProxy)`.

### 6. Support unbounded DISK cache (`max_cache_bytes=None`)

In Python-Blosc2:

- Update `_normalize_limit()` in `blosc2/remote_proxy.py`: allow `value is None`
  when `policy is CachePolicy.DISK`, returning `None`. Continue strictly requiring
  a positive integer for `MEMORY`, and forbidding limits for `NONE`.
- Update `_export_carrier()`: when exporting with `cache_policy=CachePolicy.DISK`,
  preserve `self.max_cache_bytes` (which can be `None`), rather than coercing it
  to `DEFAULT_DISK_CACHE_BYTES`. When exporting to `MEMORY`, fall back to
  `DEFAULT_DISK_CACHE_BYTES` (since MEMORY requires a finite positive integer limit).
- Add `_validate_payload_limit()` helper and update `_from_payload()`: allow
  `max_cache_bytes: null` for persisted DISK payloads, while continuing to reject
  booleans, strings, zero, and negative values.
- Document unbounded DISK caching in docstrings (`RemoteProxy`, `blosc2.open`) and
  Sphinx/Myst documentation (`doc/reference/remoteproxy.rst`, `doc/guides/remote_arrays.md`).
- Add tests in `tests/test_remote_proxy.py`: test initialization, rejection of
  `None` for MEMORY, rejection of non-integers/negatives for DISK, persistence round-trip
  and payload validation, and verify that chunks are retained without eviction.

In Caterva2:

- In `caterva2/services/remote_proxy.py::_validated_source()`: accept
  `max_cache_bytes: null` for `disk`, while rejecting invalid types (booleans,
  strings, non-positive numbers). Keep `memory` strictly requiring positive integers.
- In `caterva2/services/remote_proxy.py::ServerRemoteProxy`:
  - In `current_cache_bytes()`: check if `proxy-cache-sizes` is present in `vlmeta`;
    if absent (as is the case when `blosc2.Proxy` runs with `max_cache_bytes=None`),
    fall back to `carrier.schunk.cbytes`.
  - In `_backend()`: handle `self.max_cache_bytes is None` when applying `cache_limit`,
    avoiding `TypeError` in `min()`.
- In `caterva2/services/server.py::remote_proxy_cache_limit()`: when
  `proxy.max_cache_bytes is None`, return `None` if no customer quota is configured;
  if customer storage quota is enabled, clamp to available quota (`retained + available`).
- Update docs and sample config (`doc/utilities/cat2-server.md`, `caterva2-server.sample.toml`).
- Add tests in `caterva2/tests/test_remote_proxy.py` and `caterva2/tests/test_api.py`.

## Verification plan

Extend `caterva2/tests/test_remote_proxy.py` using its deterministic fixtures:

1. Replace the old blanket MEMORY-rejection expectation in
   `test_cache_specification_is_strict`. Add valid MEMORY cases with default and
   custom limits, invalid-limit cases for MEMORY and DISK, and unknown policies.
2. Parameterize default-deny and unsafe-destination tests over all three policies
   with valid corresponding limits. Preserve private-address, pinned resolver,
   redirect-disabled, source-field, and embedded-reference rejection coverage.
3. Extend `test_allowed_source_is_resolved_with_the_secure_filesystem` to MEMORY.
   Assert identity of the supplied `_filesystem`, requested MEMORY/effective NONE,
   and no call to the ordinary RemoteProxy constructor or generic decoder.
4. Extend `_server_proxy` fixtures to create MEMORY carriers. Read the same slice
   twice on one runtime and after reconstructing the runtime. Assert correct
   values and upstream chunk/block data calls for each read. Repeat for chunks.
   Instrument data operations, not just aggregate metadata/request counters.
5. Compare carrier bytes, size, and mtime before and after reads; allow sidecar
   creation. Assert no attached reusable backend, current cache bytes zero, and
   zero disk-cache allowance regardless of storage quota.
6. Exercise a MEMORY carrier containing synthetic warm cache bookkeeping/data;
   logical reads must still use the authorized source and ignore those chunks.
7. Preserve the payload through both warm and cold `export_cframe()` calls and
   reopen the exported MEMORY artifact in Python. Verify its original policy,
   limit, correct data, and client cache reuse with a controlled source fixture.
8. Keep existing DISK warm-reopen, cold-export, zero-quota, concurrent-fill, and
   secure-filesystem tests passing. Add source/geometry replacement cases across
   resolutions for MEMORY without claiming stronger within-operation consistency.

Extend `caterva2/tests/test_api.py`:

- Parameterize existing discovery/default-deny and physical-download tests for
  MEMORY; info/download must work without outbound resolution even when disabled.
- Add enabled-resolution fetch and chunk coverage using a controlled source and
  server fixture. The current API tests use a running server; monkeypatching only
  the client test process does not patch that server. Use an in-process fixture
  with injection at the authorized filesystem boundary, or a controlled HTTPS
  fixture with explicitly test-scoped DNS classification.
- Assert requested descriptor preservation, logical fetch results rather than
  carrier placeholders, repeated upstream data reads, no carrier mutation, and
  unchanged range-advertisement behavior.

### Verification results

The implementation is verified across both repositories in the `blosc2` conda environment:

1. **Python-Blosc2**:
   - `pytest tests/test_remote_proxy.py`: 62 passed in 1.27s.
   - Tested MEMORY/DISK/NONE policy configurations, parameter defaults, strict descriptor validation,
     unbounded DISK caching without LRU eviction (`test_unlimited_disk_cache_does_not_evict`),
     persisted payload decode/reopen (`test_reference_accepts_disk_policy_with_none_limit_in_payload`),
     and invalid payload limits (`test_reference_rejects_invalid_disk_limit_in_payload`).
   - `ruff check` and `ruff format --check`: all checks passed cleanly.

2. **Caterva2**:
   - `pytest caterva2/tests/test_remote_proxy.py`: 58 passed in 1.01s.
   - Covered default-deny across policies, allowed HTTPS destinations, strict cache specification,
     secure filesystem pinning, ServerRemoteProxy MEMORY execution without retained caching,
     carrier file immutability, synthetic cached chunk rejection, warm/cold cframe export and Python reopen,
     geometry replacement detection, customer quota clamping for bounded and unbounded proxies,
     unbounded DISK server caching without eviction (`test_unlimited_disk_server_proxy_caches_without_eviction`),
     and concurrent fills.
   - `pytest caterva2/tests/test_api.py`: 126 passed, 109 skipped in 5.05s.
     Included discovery of NONE/MEMORY/DISK (bounded and unbounded), resolution denial before open,
     and in-process ASGI resolution and fetch (`test_remote_proxy_memory_enabled_resolution_fetch_and_chunk`).
   - `ruff check` and `ruff format --check`: all checks passed cleanly.

## Limits and deferred work

Existing rank, logical-byte, chunk-count, timeout, and concurrency controls remain
in effect. They are not an aggregate process RAM cap or an operation-wide network
budget. MEMORY-to-NONE does not solve large result allocation, many concurrent
requests, multi-worker peak memory, or a source changing during range assembly.
Do not describe it as immunity from memory exhaustion.

Server MEMORY retention, shared cache registries, memory quotas, signed/private
server sources, nested reference resolution, and broader snapshot guarantees
remain separate future designs. The small policy translation proposed here
should not expand those boundaries.

## Acceptance criteria

- [x] Valid public HTTPS MEMORY carriers resolve under the existing opt-in policy.
- [x] Their runtime uses the already authorized source and the existing NONE path.
- [x] Repeated logical reads retain no array-data cache between operations and do not
  modify the uploaded carrier.
- [x] Stored/downloaded policy remains MEMORY with its original positive limit, and
  a compatible Python client restores MEMORY behavior.
- [x] `CachePolicy.DISK` supports `max_cache_bytes=None` (unbounded cache) in Python-Blosc2
  and Caterva2, bypassing LRU eviction while respecting Caterva2 customer storage quota if configured.
- [x] NONE and DISK behavior and all security gates remain intact.
- [x] No Python-Blosc2 construction hook, memory quota knob, or carrier format change
  is introduced merely to implement this mapping.
