# Migrate from `requests` to `httpx`

## Why

`Proxy.afetch` + `C2Array.aget_chunk` (see the Caterva2 remote-peer-mounts
plan, §3) added `httpx` as an optional `[remote]` extra purely for its async
client. But `httpx` also has a sync `Client`/module-level API that mirrors
`requests` closely enough to fully replace it. Carrying two HTTP libraries
(one hard dependency for sync, one optional for async) is dead weight once
one library covers both — this plan retires `requests` and makes `httpx` the
single, hard HTTP dependency.

Not urgent: the current split (`requests` hard, `httpx` optional) works fine
and nothing is broken. Do this opportunistically, not under time pressure.

## Call sites to migrate

- `src/blosc2/c2array.py`
  - `_requests()` lazy-import helper -> replace with `_httpx()` (already
    exists, added for `aget_chunk`).
  - `_xget` / `_xpost` (module-level GET/POST helpers) -> `httpx.get`/`httpx.post`,
    or better, a shared `httpx.Client` (see "Connection reuse" below).
  - `login()` -> `httpx.post`; note `requests`' `resp.cookies.items()` needs
    the `httpx` equivalent (`resp.cookies` is also a mapping-like in httpx,
    but verify the JWT cookie round-trip still matches what Caterva2's
    server expects).
  - `C2Array.__init__` -> `except _requests().HTTPError` becomes
    `except httpx.HTTPStatusError` (httpx splits `HTTPError` into
    `HTTPStatusError` for 4xx/5xx and `RequestError` for connection-level
    failures — pick the right one, or catch both).
  - `C2Array.get_chunk` (sync) -> once `_xget` is on httpx, this is unchanged.
- `src/blosc2/b2view/app.py`
  - `_http_download`: `requests.get(url, stream=True)` + `resp.iter_content(...)`
    -> `httpx.stream("GET", url)` + `resp.iter_bytes(...)` (httpx's streaming
    API shape differs from requests' — not a 1:1 rename).
  - `_fetch_remote_size`: trivial swap, `requests.get` -> `httpx.get`.
- `src/blosc2/core.py`
  - Version banner (`print(f"requests version: ...")`) -> report `httpx`
    version instead (or both, if worried about the deprecation window).
- `pyproject.toml`
  - Move `httpx[http2]` from `optional-dependencies.remote` into
    `dependencies`; drop `requests`; delete the `remote` extra (or keep it
    as a no-op alias for one release if external code references it).
- Tests
  - `tests/conftest.py` (`requests.exceptions.RequestException`),
    `tests/test_open_c2array.py` (`requests.post` in a fixture/setup helper),
    `tests/ndarray/test_c2array_async.py` (`_FakeRequests`/`_requests`
    monkeypatch scaffolding written for this migration's predecessor) all
    need their fake/mocked transport updated from `requests`-shaped to
    `httpx`-shaped (`httpx.MockTransport` already used in
    `test_c2array_async.py` for the async half — reuse the same pattern for
    sync via `httpx.Client(transport=...)`).

## Gotchas

- **Cookie-based auth header.** Current code sends `auth_token` as a raw
  `Cookie` header string (`headers["Cookie"] = auth_token`), not via
  `requests`' cookie jar. This is header-level, not cookie-jar-level, so it
  should port to httpx unchanged — but verify against a live Caterva2 server
  since `login()`'s `"=".join(list(resp.cookies.items())[0])` depends on
  `requests`-specific cookie-jar iteration order/format.
- **`raise_for_status()` exception hierarchy.** `requests.HTTPError` vs
  httpx's `HTTPStatusError`/`RequestError` split — anywhere catching
  `requests.HTTPError` (currently just `C2Array.__init__`) needs updating,
  and callers catching bare `Exception` around HTTP calls are unaffected.
- **Streaming API shape.** `resp.iter_content(chunk_size=...)` (requests) vs
  `resp.iter_bytes()` / `client.stream(...)` context manager (httpx) — the
  b2view download path needs an actual rewrite, not a search-and-replace.
- **Timeout semantics.** Both accept a plain float total-timeout, so `TIMEOUT`
  constants should port as-is, but double check httpx's default connect vs.
  read timeout split doesn't change observed behavior for slow Caterva2
  servers.
- **Connection reuse.** `requests` module-level `get`/`post` don't pool
  connections across calls; neither does bare `httpx.get`/`httpx.post`. If
  reducing per-call handshake overhead matters (it does for the chunk-fetch
  hot path, hence `C2Array._aclient` already caching an `AsyncClient`),
  consider adding a lazily-created sync `httpx.Client` cached the same way,
  rather than porting `_xget`/`_xpost` to bare `httpx.get`/`httpx.post`.

## Suggested order

1. Swap `c2array.py`'s sync helpers (`_xget`, `_xpost`, `login`, the
   `HTTPError` catch) to httpx first — it's the highest-traffic path and
   already has async test coverage to model sync tests on.
2. Swap `b2view/app.py`'s two call sites (small, isolated, no other module
   depends on their internals).
3. Update `core.py`'s version banner.
4. Flip `pyproject.toml` dependencies and update the three affected test
   files.
5. Grep for `requests` across `src/` and `tests/` once more to confirm
   nothing was missed, then delete the now-dead `_requests()` helper and
   the `remote` extra.
