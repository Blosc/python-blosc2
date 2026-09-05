# Plan: Unified RemoteProxy with Memory and Disk Caching (v3)

## Status

Planned for implementation in Python-Blosc2.

## Purpose

Unify all lazy remote dataset access under `RemoteProxy`. Previously, `blosc2.open(url, lazy=True)` returned a legacy `blosc2.Proxy` when opened without disk storage options, and a `blosc2.RemoteProxy` when `cache_dir` or `cache_path` was specified.

In v3, `blosc2.open(url, lazy=True)` **always** returns a `RemoteProxy`. To achieve this cleanly while maintaining the fast, ephemeral in-memory caching behavior users expect, `CachePolicy.MEMORY` is reinstated as a first-class policy alongside `CachePolicy.DISK` and `CachePolicy.NONE`.

## Core Decisions

### 1. Unified Return Type: Always `RemoteProxy`

`blosc2.open(url, lazy=True)` unconditionally returns an instance of `blosc2.RemoteProxy`:
- When neither `cache_dir` nor `cache_path` is specified: defaults to `CachePolicy.MEMORY`.
- When `cache_dir` or `cache_path` is specified: defaults to `CachePolicy.DISK`.
- When `cache_policy` is passed explicitly: respects the requested policy (`NONE`, `MEMORY`, or `DISK`).

Users interact with a single, consistent API (`.source`, `.info`, `.traffic`, `.cache_policy`, `.cache_bytes`, `.max_cache_bytes`, `.fetch()`, `.save()`).

### 2. First-Class Cache Policies

`blosc2.CachePolicy` provides three explicit retention policies:

1. **`NONE`**:
   - Stateless floating reference.
   - Fetches only data required for the current operation and retains no cached chunks.
   - `max_cache_bytes` must be `None`.

2. **`MEMORY`**:
   - Ephemeral in-memory cache held in client RAM during the process lifetime.
   - Bounded by `max_cache_bytes` (defaults to 256 MiB).
   - Automatically applies LRU chunk eviction when retained payload exceeds `max_cache_bytes`.
   - Requires no local files or directories (`cache_dir` and `cache_path` must be `None`).

3. **`DISK`**:
   - Persistent carrier cache on disk.
   - Bounded by `max_cache_bytes` (defaults to 256 MiB) with LRU chunk eviction.
   - Requires `cache_dir` or `cache_path` when creating from a remote URL.

### 3. Server-Side Protection in Caterva2

Caterva2 maintains its strict server-side gate in `caterva2/services/remote_proxy.py`:
- Carriers uploaded to Caterva2 with `cache_policy` other than `"none"` or `"disk"` raise `RemoteProxyDenied` (HTTP 403).
- Caterva2 servers are therefore immune to memory-hogging exploits from crafted carriers, without requiring artificial restrictions on Python-Blosc2 client code.

### 4. Direct Carrier Export & Deserialization

- For `CachePolicy.MEMORY`:
  - `save()` or `to_cframe()` exports the carrier structure and metadata with `cache_policy: "memory"`.
  - Since in-memory chunks are process-local, only the cold descriptor is persisted.
  - When reopened via `blosc2.open("saved.b2nd")`, it initializes as an in-memory `RemoteProxy` ready to cache misses in RAM.
- For `CachePolicy.DISK`:
  - `save()` and `to_cframe()` include warm cached chunks by default unless `include_cache=False`.
- For `CachePolicy.NONE`:
  - Persisted as a cold reference without data.

### 5. `fetch()` and `afetch()` Support on `RemoteProxy`

`RemoteProxy` exposes `fetch(item=None)` and `afetch(item=None)`:
- When caching is enabled (`MEMORY` or `DISK`), delegates to the internal cache engine and returns the cache container.
- When `cache_policy` is `NONE`, raises `NotImplementedError`.
- Exposes `cache` property returning the cache container or `None`.

## Implementation Tasks

1. **Enum & Policies (`src/blosc2/__init__.py`)**:
   - Re-introduce `MEMORY = "memory"` in `CachePolicy`.

2. **RemoteProxy Implementation (`src/blosc2/remote_proxy.py`)**:
   - Update `_normalize_limit` to accept `_POLICY_DEFAULT` (256 MiB) for `CachePolicy.MEMORY`.
   - Update `__init__` validation: ensure `cache_dir` and `cache_path` are only used with `DISK`.
   - In `_attach_carrier_cache`: instantiate an in-memory `Proxy` with `_max_cache_bytes` when `cache_policy is CachePolicy.MEMORY`.
   - In `_export_carrier`: return `_to_b2object_carrier()` for `MEMORY` (no disk carrier to export).
   - In `_from_payload`: decode `CachePolicy.MEMORY` with positive `max_cache_bytes`, instantiating with in-memory cache.
   - Add `fetch(item=None)`, `async afetch(item=None)`, and `@property def cache`.

3. **Open Integration (`src/blosc2/schunk.py`)**:
   - In `_remote_proxy_options`: when `lazy=True`, always return options (defaulting to `DISK` if disk options are present, else `MEMORY`).
   - In `_open_fsspec_url` and `_open_c2_urlpath`: always return `RemoteProxy` for `lazy=True`.

4. **Documentation & Tests**:
   - Update `doc/guides/remote_arrays.md` and docstrings to describe `CachePolicy.MEMORY`.
   - Update tests in `tests/test_remote_proxy.py`, `tests/test_fsspec.py`, and `tests/ndarray/test_c2array_blocks.py`.
   - Run full pytest suite and lint checks.
