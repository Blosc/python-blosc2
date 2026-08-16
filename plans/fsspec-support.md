# Plan For Reading And Writing Blosc2 Files Through fsspec

## Goal

Let `blosc2.open()` and the save helpers accept
[fsspec](https://filesystem-spec.readthedocs.io) URLs, so that Blosc2
containers can live wherever fsspec can reach — object stores
(`s3://bucket/key.b2nd`, `gs://`, `abfs://`), archives (`zip://`, `tar://`),
remote filesystems (`sftp://`, `smb://`), or memory (`memory://`) — without the
caller first downloading them by hand.

S3 is the motivating case throughout and the one the testing and rollout
sections concretise, but nothing in the implementation is S3-specific: the
dispatch is a single protocol-agnostic branch, so every fsspec driver comes
along at no extra cost.

It is staged so that each phase is independently shippable and each one is
useful on its own; phase 1 alone already covers the common case.

**Status: all three phases are implemented** (2026-08-16, branch
`fsspec-support-plan`), phase 3 by route 3a. The recommendation sections are
kept as written and annotated where reality diverged from them.

## Motivation

Today there is no S3 support at all. `s3fs` appears in the repo only in
[bench/ndarray/download_data.py](/Users/faltet/blosc/python-blosc2/bench/ndarray/download_data.py)
and in the `dev` dependency group of
[pyproject.toml](/Users/faltet/blosc/python-blosc2/pyproject.toml). Passing
`s3://...` to `blosc2.open()` falls through the store-probing branches in
[src/blosc2/schunk.py](/Users/faltet/blosc/python-blosc2/src/blosc2/schunk.py)
and ends in a `FileNotFoundError`.

Users who keep data in object storage therefore have to write the
download-to-tempfile dance themselves, which is both boilerplate and, for the
whole-file case, exactly what a five-line branch in `open()` would do.

The remote story that *does* exist — `blosc2.URLPath` / `C2Array`, see
[src/blosc2/c2array.py](/Users/faltet/blosc/python-blosc2/src/blosc2/c2array.py)
— is specific to a Caterva2 server speaking HTTP with a chunk-fetch endpoint.
It is not a generic object-store client and should stay untouched by this work.

## Current Situation

Relevant facts established while scoping this:

- A `.b2nd` / `.b2f` file on disk **is** a contiguous frame. Reading the file's
  bytes and passing them to `blosc2.from_cframe()` reconstructs a working
  `NDArray` / `SChunk` / `EmbedStore` / `ObjectArray` / `BatchArray`. Verified
  against a file written by `blosc2.asarray(..., urlpath=...)`.
- `fsspec` is not a declared runtime dependency, but it is present in the
  `blosc2` conda env today (2026.7.0) — while `s3fs`, `gcsfs` and `zarr` are
  not. So the dev environment already exercises the "fsspec but no backend"
  configuration that most users installing `[fsspec]` will be in.
- fsspec ships a `memory://` filesystem in the stdlib-equivalent sense: no
  extra package, no network, no credentials. It exercises the same code path a
  future `s3://` branch would take.
- fsspec's `filecache`/`simplecache` layers expose a **local path** for a
  remote object (`fs.open(key).name`), and `blosc2.open()` on that path works
  unmodified — verified. This is the cheapest route to full format coverage.
- c-blosc2 exposes a user-defined I/O plugin API
  (`blosc2_register_io_cb` / `blosc2_get_io_cb`, blosc2.h:1058) and
  python-blosc2 already routes opens through it:
  `blosc2_schunk_open_offset_udio` is called at
  [src/blosc2/blosc2_ext.pyx](/Users/faltet/blosc/python-blosc2/src/blosc2/blosc2_ext.pyx):1747,
  3406 and 3422, for the mmap backend (`BLOSC2_IO_FILESYSTEM_MMAP`) and for the
  locking `blosc2_io`. What python-blosc2 does *not* do today is register a
  callback set of its own — both existing users are backends c-blosc2 ships.
- Container layouts differ in a way that matters here:
  - `.b2nd`, `.b2f`, `.b2e` (`EmbedStore`), `.b2z` (zip-backed store) — single
    file, so a single object in S3.
  - `.b2d` (`DictStore`/`TreeStore` directory format) — a *directory* of files
    ([src/blosc2/dict_store.py](/Users/faltet/blosc/python-blosc2/src/blosc2/dict_store.py):209),
    so it needs prefix-level sync, not a single GET.
  - Sparse frames (`contiguous=False`) are likewise directories.

## Non-Goals

- Replacing or extending `C2Array` / Caterva2. `http://` and `https://` stay
  reserved for that path and are explicitly excluded from the new branch.
- A blosc2-specific S3 client. Everything goes through fsspec; credentials,
  retries, endpoint overrides, anonymous access and profile handling are
  fsspec/`s3fs` concerns and are configured by the caller.
- Concurrent writers / locking semantics against an object store. S3 has no
  rename and no file locks; `mode="a"` on a remote URL is out of scope for
  every phase below and should raise.

## Phase 1 — Whole-object read and write — DONE

The minimum that is genuinely useful.

**As implemented**, with the two places it departs from the sketch below:

- `is_fsspec_url()` and `fsspec_open()` live in
  [src/blosc2/core.py](/Users/faltet/blosc/python-blosc2/src/blosc2/core.py);
  `open()` dispatches to `_open_fsspec_url()` in
  [src/blosc2/schunk.py](/Users/faltet/blosc/python-blosc2/src/blosc2/schunk.py).
  The read branch became its own function only because inlining it pushed
  `open()` past ruff's complexity limit.
- The write branch went into `pack_tensor()` rather than into `save_array` and
  `save_tensor` separately: both delegate to it, so one branch serves all three
  entry points (plus `pack_array2`) instead of three copies.
- `.b2d` raises `NotImplementedError`; sparse frames are not detected up front
  and fail on the `from_cframe` instead. Both messages now point at phase 2's
  `cache_storage=`, which is the actual fix.
- Tests: `tests/test_fsspec.py`, 12 tests over `memory://` plus one chained
  `zip://…::file://` URL, in the default suite behind `importorskip("fsspec")`.
  No tier-2 network test, per the open question below.
- Later addition: `NDArray.save()` and `blosc2.save()` write to a URL the same
  way, since the plan's write story covered only the `save_array`/`save_tensor`
  helpers and left `save()` — the natural call for a container that already
  exists — failing in C. The rejection this section predicted for
  `copy(urlpath=...)` is now an explicit `ValueError` from `Storage` and
  `SChunk.__init__`, naming `save()`, rather than a `RuntimeError` from the C
  layer.

The rest of this section is the original design, kept as the record of why the
code looks the way it does.

**Dependency.** A new optional extra in
[pyproject.toml](/Users/faltet/blosc/python-blosc2/pyproject.toml), so nothing
changes for users who do not want it:

```toml
[project.optional-dependencies]
fsspec = ["fsspec"]
```

**On the name.** `[s3]` was the first instinct and it is wrong: nothing in the
implementation is S3-specific, and the extra cannot carry the backends anyway —
S3 needs `s3fs`, GCS needs `gcsfs`, Azure needs `adlfs`, and so on for a dozen
more. An extra named after one of them misrepresents what it delivers.
`[remote]` is wrong in the other direction, since fsspec also drives purely
local protocols (`zip://`, `tar://`, `dir://`, `memory://`). `[fsspec]` names
exactly what it installs, and "an fsspec URL" is precisely the capability the
docs will describe. It breaks the `[tui]` / `[hires]` / `[parquet]` convention
of naming the capability rather than the package, which is acceptable here
because the capability has no better English name that is not a lie.

Backends stay the caller's install. That is not a gap to paper over: fsspec
already raises an actionable error when a protocol's driver is missing, from
inside the `fsspec.open()` call in our own branch, so it propagates untouched
and we neither write the message nor maintain a table of which package serves
which scheme. Verified against fsspec 2026.7.0 with no backends installed:

```
s3://bucket/key.b2nd   -> ImportError: Install s3fs to access S3
gcs://bucket/key.b2nd  -> ImportError: Please install gcsfs to access Google Storage
gs://bucket/key.b2nd   -> ImportError: Please install gcsfs to access Google Storage
nosuchproto://b/k      -> ValueError: Protocol not known: nosuchproto
```

The wording is not consistent between backends ("Install s3fs" vs "Please
install gcsfs"), which is one more reason to let fsspec own these strings
rather than mirroring them in our docs or asserting on them in tests.

The second case matters because the branch fires on any `"://"`: a typo'd or
unsupported scheme produces a clear `ValueError` rather than falling through to
a misleading `FileNotFoundError`. Neither case needs handling from us; both
should be covered by a negative test in tier 1.

`fsspec` itself is imported lazily, inside the branch, so a missing extra costs
an `ImportError` rather than an import-time cost for everybody.

**Read.** One branch in `blosc2.open()`
([src/blosc2/schunk.py](/Users/faltet/blosc/python-blosc2/src/blosc2/schunk.py):2075,
immediately after the `pathlib.PurePath` normalisation and before the
`.b2d`/`.b2z`/`.b2e` dispatch):

```python
if "://" in urlpath and not urlpath.startswith(("file://", "http://", "https://")):
    if mode != "r":
        raise NotImplementedError("remote URLs can only be opened with mode='r'")
    import fsspec

    with fsspec.open(urlpath, "rb") as f:
        return blosc2.from_cframe(f.read())
```

Notes on the details:

- The `file://` exclusion lets fsspec-style local URLs keep working through
  the normal local path, which supports mmap and every container format.
  *(This turned out to need more than the exclusion: nothing downstream stripped
  the scheme, so a `file://` URL was taken as a literal filename and failed. It
  is normalized to a native path now, in `open()`, `NDArray.save()`, `Storage`
  and the two constructor paths that bypass `Storage`.)*
- `offset != 0` should raise for now; the embedded-object case is a phase-3
  concern.
- `copy=False` on `from_cframe` is tempting (it pins the read buffer instead of
  copying it) but the buffer is a throwaway `bytes` we just built, so `copy=True`
  and `copy=False` cost the same peak memory here and `False` merely keeps the
  buffer alive longer. Leave the default.

**Write.** The mirror, in the save helpers rather than in `open()`:
`blosc2.save_array` / `save_tensor`
([src/blosc2/core.py](/Users/faltet/blosc/python-blosc2/src/blosc2/core.py):528,
750) grow the same URL test and become
`fsspec.open(urlpath, "wb").write(arr.to_cframe())`. `NDArray.copy(urlpath=...)`
and friends keep rejecting remote URLs — the C layer writes incrementally and
cannot target an object store.

**Documented limits of phase 1**, stated in the docstring rather than
discovered by users:

- the whole object is read into memory, twice at the peak (the `bytes` plus the
  reconstructed container) unless `copy=False`;
- read-only;
- single-file formats only (`.b2nd`, `.b2f`, `.b2e`, `.b2z`); `.b2d` and sparse
  frames raise a clear `NotImplementedError` naming phase 2.

## Phase 2 — Local cache, full format coverage — DONE

The lazy way to get every container format, mmap, and repeat-run speed without
writing a byte-range reader.

**As implemented:** `blosc2.open(url, cache_storage=...)`, backed by
`localize_fsspec_url()` in
[src/blosc2/core.py](/Users/faltet/blosc/python-blosc2/src/blosc2/core.py), which
returns a local path that `open()` then re-enters with. All four open questions
below were settled as recommended, plus these decisions taken while building it:

- **One knob, not two.** `cache_storage=` alone turns caching on; there is no
  separate `cache=True`, since an explicit directory already says everything a
  boolean would. No module-level `set_remote_cache()` either — add it if someone
  asks.
- **`check_files=True` is mandatory, and was not free.** fsspec's `filecache`
  does *not* check staleness by default (`check_files=False`), contrary to what
  this plan assumed: it served a cached array whose remote bytes had changed.
  Caught by a test that mutates the object between two opens. `simplecache` was
  not added as a flag; it is what fsspec's own default already behaved like, and
  nobody has asked for it.
- **Directory containers carry their own manifest.** fsspec has no `filecache`
  equivalent for a prefix, so `.b2d` stores and sparse frames are fetched with
  one `fs.get(recursive=True)` into a URL-hashed subdirectory, alongside a JSON
  manifest of the remote `fs.find(detail=True)` listing. A changed listing
  re-fetches the whole prefix; no per-file delta sync.
- **Unset kwargs are not a request.** The no-cache path rejects `mmap_mode`,
  `offset` and friends by pointing at `cache_storage=`, but ignores kwargs whose
  value is `None` — `load_tensor()` passes `dparams=None` unconditionally.
- Tests grew to 20, still `memory://` only: cache hit (no refetch), staleness
  re-fetch for both files and directories, mmap over a cached file, a `.b2d`
  `DictStore`, and a sparse frame.

Write-back stayed out, as the section below says it should.

The rest of this section is the original design.

fsspec's `filecache` downloads an object once into a local cache directory and
hands back a real local file path. `blosc2.open()` on that path is the ordinary
local path, so *everything* works: sparse frames, `.b2d` directories (via
`fs.get()` of the prefix), `mmap_mode`, `offset`. Verified working against
`memory://` in scoping.

Shape of it:

```python
def _localize(urlpath, cache_storage=None):
    """Download a remote container into the local fsspec cache, return its path."""
```

- single-file containers: `fsspec.filesystem("filecache", target_protocol=...,
  cache_storage=...).open(key).name`;
- directory containers (`.b2d`, sparse frames): `fs.get(prefix, localdir,
  recursive=True)` into the same cache root, return the local directory.

Open questions to settle before implementing:

- **Cache location and lifetime.** Default to `platformdirs`-style user cache
  or require an explicit `cache_storage=`? An unbounded implicit cache that
  silently fills a laptop disk is the classic footgun here; an explicit
  argument is the honest default, with a module-level
  `blosc2.set_remote_cache(...)` for people who want it global.
- **Staleness.** `filecache` checks the remote mtime; S3 ETags make that
  cheap-ish but not free (one HEAD per open). `simplecache` skips the check
  entirely. Probably: `filecache` by default, `simplecache` behind a flag.
- **Interaction with phase 1.** Once phase 2 exists, phase 1's in-memory read
  is still the right default for a one-shot read of a small object. Suggested
  rule: `blosc2.open(url)` stays in-memory; `blosc2.open(url, cache=True)` (or a
  global setting) goes through the cache. Do not silently switch behaviour.

Phase 2 also unlocks write-back for single-file containers — write locally,
`fs.put()` on close — but that is a separate, opt-in `mode="w"` story and
should not be smuggled in with the read work.

## Phase 3 — Byte-range chunk access — DONE, via 3a

**As implemented:** `blosc2.open(url, lazy=True)` returns a `Proxy` over the new
`blosc2.FsspecNDSource`
([src/blosc2/proxy.py](/Users/faltet/blosc/python-blosc2/src/blosc2/proxy.py)),
which reads the frame's header and offsets at open (three small reads) and then
one range read per chunk a slice touches. Measured on a 36 KB frame over
`memory://`: 276 bytes at open, 2 KB for a 50-element slice.

Route 3a was chosen over the plan's recommendation of prototyping 3b first,
because **the offset-table blocker turned out to be much smaller than this plan
assumed**:

- The frame header *is* a msgpack array. `msgpack.unpackb(header)` yields
  `header_len`, the compressed size, the chunk size and the metalayer map with no
  byte arithmetic at all. Exactly one field has to be located by hand —
  `header_len`, at byte 0x0B — because it is needed to know how much to unpack.
- The `b2nd` metalayer then gives shape, chunks, blocks and dtype, so no sparse
  local skeleton or C accessor is needed to describe the array.
- The offsets are one Blosc2 chunk at `header_len + compressed_size`, decompressed
  with `blosc2.decompress2`. Two corrections to what this plan and the format doc
  say: the offsets are relative to the **end of the header**, not to its
  beginning; and a *negative* offset is not a position but a run-length chunk
  (zeros, NaN, uninitialized) that was never written, which the source rebuilds
  locally.

That is about 45 lines of format knowledge, against 3b's Cython callback bridge,
its permanent registry id and its per-block `open()`. The judgement stands that
3b is the architecturally cleaner one — if the frame format ever grows a variant
this parser does not know, it will be 3b that survives it — but at this size 3a
was not worth deferring for it. Everything the parser reads is validated by the
tests decompressing real chunks through it.

`aget_chunk` is implemented too, since it is the reason the plan kept 3a on the
table: `Proxy.afetch` overlaps up to 8 chunk fetches on async backends (s3fs and
friends), and falls back to the blocking path elsewhere. Only the fallback is
covered by tests — `memory://` is not async — so the concurrent path is the one
piece of this work that a real S3 endpoint would exercise first.

Later, the batching the plan wanted from `aget_chunk` was given to the *sync*
path as well, since that is the one ordinary slicing uses: `get_chunk` became a
single stateless range read (it cost two, one for the chunk header), which made
it thread-safe, and `Proxy.fetch` grew a `max_concurrency=` thread pool.
Threads rather than asyncio, because driving `afetch` from `__getitem__` would
mean `asyncio.run()` inside a sync method — a `RuntimeError` in any notebook,
and the first sync-over-async in the library. The test asserts overlap with a
barrier rather than a stopwatch, since `memory://` has no latency to hide.

It defaults to 8 rather than to serial. The *cost of being wrong* is small and
measured: over `memory://`, where the pool can only lose, a 100-chunk read goes
from 1.1 ms to 2.2 ms, about 10 µs per chunk. The gain is 7.4x on a 100-chunk
read against a 5 ms simulated round trip
([examples/ndarray/concurrent-fsspec.py](/Users/faltet/blosc/python-blosc2/examples/ndarray/concurrent-fsspec.py)),
and unmeasured against a real endpoint. That asymmetry, plus `afetch` already
defaulting to 8 for remote sources, made serial-by-default the inconsistent
choice rather than the conservative one.

That example had to invent its own latency: nothing that runs offline has any.
`memory://`, `zip://` and `tar://` are local reads, the regime where the pool
only costs, and `http://` is reserved for Caterva2 so a local server is not
reachable either. It subclasses fsspec's in-memory filesystem with a fixed delay
and says so, rather than implying a benchmark it cannot run.

Two examples cover the feature:
[rw-fsspec.py](/Users/faltet/blosc/python-blosc2/examples/ndarray/rw-fsspec.py)
for the three read modes and the write, and `concurrent-fsspec.py` for
`max_concurrency`.

`lazy=True` and `cache_storage=` compose rather than excluding each other, which
is a departure from how phase 2 framed the choice: `cache_storage` means "where
this container's local copy lives", and `lazy` decides whether that copy is the
whole thing or only the chunks touched so far. A persistent chunk cache is
stamped with `fs.ukey()` and thrown away when that changes, since chunks fetched
by offsets from a replaced frame are not merely stale but wrong. The stamp
started as a hand-rolled `[size, mtime or LastModified]` tuple and was wrong:
see the testing note below.

Not done: `lazy=True` needs a contiguous frame carrying a `b2nd` metalayer.
Plain SChunks, sparse frames and `.b2d` stores raise and point at
`cache_storage=`. `offset != 0` likewise raises.

The rest of this section is the original design, including 3b, which stays
unbuilt.

Only worth doing when someone actually has a container too large to download
and wants to slice a small part of it. Two candidate designs. They are *not*
strictly ranked: 3b is correct and complete, 3a is the one that can be fast.

### 3a — `ProxyNDSource` over byte ranges

Implement the
[src/blosc2/proxy.py](/Users/faltet/blosc/python-blosc2/src/blosc2/proxy.py):38
interface with `get_chunk(nchunk)` doing `fs.read_block(url, offset, length)`,
mirroring what `C2Array.get_chunk` does over HTTP
([src/blosc2/c2array.py](/Users/faltet/blosc/python-blosc2/src/blosc2/c2array.py):372).
The `Proxy` machinery then caches decompressed chunks locally, and the async
`aget_chunk` hook can prefetch several ranges at once — which, per the latency
discussion below, is the whole reason this design stays on the table.

The blocker: this needs the frame's **chunk offset table**, and python-blosc2
exposes no way to get chunk offsets out of a cframe without opening it first.
So 3a requires either a small pure-Python frame-header/trailer parser (fragile,
duplicates format knowledge that belongs in C) or a new C-level accessor. Cost
this honestly before choosing it.

### 3b — A user-defined I/O plugin bridging to an fsspec file object

Register a `blosc2_io_cb` whose `open`/`read`/`size` callbacks reach an fsspec
file object, and open through `blosc2_schunk_open_offset_udio`. The C library
then does its own range reads, no format knowledge leaks into Python, and
`offset != 0`, sparse frames and every container format are fixed at once.

**This needs no c-blosc2 modification to be correct.** Two facts establish that:

- `frame.c` routes every read through
  `blosc2_get_io_cb(frame->schunk->storage->io->id)` — around 148 call sites,
  covering the header, the offsets table, chunk fetch and lazy-block fetch.
  Nothing bypasses the callback table.
- The python-blosc2 side is already plumbed: `blosc2_schunk_open_offset_udio`
  is called at
  [src/blosc2/blosc2_ext.pyx](/Users/faltet/blosc/python-blosc2/src/blosc2/blosc2_ext.pyx):1747,
  3406 and 3422. A new backend only has to supply the `blosc2_io{id, name,
  params}` struct.

Three constraints to design around, in increasing order of how much they hurt:

**Registration id must be ≥ 160.** `blosc2_register_io_cb` rejects any id below
`BLOSC2_IO_REGISTERED` (blosc2.c:6813), and `id` is a `uint8_t`, so the usable
range is 160–255 — the Blosc plugin-registry range. (`BLOSC2_IO_USER_DEFINED =
256` is unreachable through a `uint8_t`.) So the id has to be coordinated with
upstream rather than picked freely. Registration is also process-global and
permanent: `g_ios` is a fixed array with no unregister call.

**`open()` is called per block, not per file.** frame.c:163-167 restricts the
cached-handle path to `BLOSC2_IO_FILESYSTEM`:

```c
if (io->id != BLOSC2_IO_FILESYSTEM) {
    // Third-party backends keep the documented one-handle-per-reader contract
    return io_cb->open(frame->urlpath, "rb", io->params);
}
```

Every other backend re-opens on each `frame_reader_acquire()`, and `blosc_d`
acquires once per *lazy block* (blosc2.c:1806-1854). The mmap backend survives
this only because its `open` is a cheap pointer return. So the fsspec backend
must be equally cheap: the `params` struct holds an already-open handle and
`open()` just hands the pointer back, doing no Python work — no refcounting, no
GIL — at open/close time. Workable, but it constrains the design from the start.

**`read` runs concurrently on the blosc worker threads.** `blosc_d` is the
per-block decompression function, so any Python touched inside `read` has to
take the GIL and block reads serialise against each other. Probably acceptable
— S3 round-trip latency dominates, and the GIL is released while fsspec waits
on the socket — but measure it under `nthreads > 1` rather than assuming.
Prototype against a local file first, where the GIL cost is visible without
network noise masking it.

### Where a c-blosc2 change would actually pay

Not required, but worth proposing upstream if 3b goes ahead:

- **Handle caching for third-party ids.** Either extend the reader cache past
  `BLOSC2_IO_FILESYSTEM`, or add a flag on `blosc2_io_cb`
  (`caches_handles` / `open_is_cheap`) that lets a backend opt in. Small and
  localised to `frame_reader_acquire()`; removes the second constraint above.

- **A prefetch/range-coalescing hook.** This is the limitation no flag fixes:
  `blosc2_io_cb` has no way to express *"I will need blocks X through Y"*, so
  c-blosc2 issues one range GET per block with no batching. Against local disk
  that is fine; against S3 it is pure latency, one round trip per block. There
  is no cheap upstream fix — it would mean a scatter/gather read callback or a
  readahead hint in the I/O API — which is exactly why 3a's `aget_chunk` batching
  keeps its appeal despite the offset-table problem.

### Recommendation

Do not start phase 3 speculatively; wait for a concrete user with a container
too big for phase 2's cache. When that arrives, prototype 3b first — it is the
architecturally correct one and needs no upstream change to work — and measure
against a real S3 endpoint before deciding whether the per-block round trips
justify the extra machinery of 3a.

*Superseded: 3a shipped instead, and without waiting for the user. See the notes
at the top of this section — the offset table cost 45 lines of msgpack reading
rather than the C accessor this plan feared, which changed the arithmetic. 3b
stays unbuilt and its analysis below stays valid.*

## Testing

The point here is that **almost none of this needs AWS, credentials, or a new
dependency**.

**Tier 1 — `memory://`, always on, no extra deps.** The dispatch and
serialisation path is protocol-generic, so fsspec's built-in in-memory
filesystem covers it. Lives in `tests/test_fsspec.py`, no marker, runs in the
default suite:

```python
def test_fsspec_roundtrip():
    a = blosc2.arange(10, dtype="i4")
    with fsspec.open("memory://x.b2nd", "wb") as f:
        f.write(a.to_cframe())
    assert np.array_equal(blosc2.open("memory://x.b2nd")[:], a[:])
```

Plus negative tests: `mode="a"` raises, `offset != 0` raises, `http://` still
routes to `C2Array`, `.b2d` raises the phase-2 `NotImplementedError`. These
catch every regression that is actually about *our* code.

Phase 2's cache path is equally testable this way — `filecache` over
`target_protocol="memory"` with `cache_storage=tmp_path` gives a local file and
a real cache-hit assertion (open twice, assert one remote fetch), again with no
network.

Skip condition: `pytest.importorskip("fsspec")`, since fsspec is optional.

**Tier 2 — real S3, opt-in.** One test against a public anonymous bucket,
marked `network`. [pytest.ini](/Users/faltet/blosc/python-blosc2/pytest.ini)
already excludes that marker from the default run
(`-m "not network and not heavy and not tui"`), so CI stays offline and the
test is run deliberately:

```
conda run -n blosc2 pytest -m network tests/test_s3.py
```

Requires `s3fs`, already in the `dev` group. It needs a stable, publicly
readable object to point at — either one published under a Blosc-controlled
bucket as part of this work, or a well-known open dataset. Decide which before
writing the test; do not let it depend on a bucket that can vanish.

**Tier 3 — a real S3 protocol, locally.** `moto[server]` gives a local
S3-compatible endpoint that `s3fs` can be pointed at with `endpoint_url`. This
is a new dev dependency and it is only worth adding once phase 3 exists, since
`memory://` cannot exercise range requests and `moto` can. Not before.

*Decided after phase 3 shipped: still no.* Three things `memory://` cannot
reach — the async `aget_chunk` path (memory is not an async backend), reads
through a buffered/block-caching file object, and the actual latency win — and
`moto` only buys the first two. That is not worth a dev dependency plus a local
HTTP server, and `blockcache::memory://` covers the second for free. Revisit if
the async path is ever to be locked down before someone runs this against real
S3.

**What `memory://` gets wrong, and it is not what you would expect.** It is not
too *unrealistic* — it is **poorer in metadata than any real backend**, exposing
no `mtime` or `LastModified`. That silently degraded the hand-rolled cache stamp
to size-only, so a frame replaced by one of identical size was served from a
stale chunk cache; and the staleness test passed anyway, because compression had
made the two frames different sizes. `moto` would have *hidden* this bug, not
caught it. The fixes were `fs.ukey()`, which asks fsspec what identifies these
bytes rather than guessing which fields a backend exposes, and a test that
writes both frames uncompressed so only a real content check can pass.

**What not to build:** no `boto3` stubbing, no fixture framework, no
per-protocol parametrisation across `s3`/`gcs`/`az`. The code path is one
branch; one filesystem exercising it is enough.

## Recommendation

**Ship phase 1 and stop.** It is roughly ten lines plus the extra plus the
tier-1 tests, it covers "my arrays are in S3 and I want to read them", and it is
the only phase whose value is certain today.

*Superseded: phase 1 shipped, and phase 2 followed immediately after rather than
waiting for someone to hit the memory ceiling — it turned out to be about forty
lines and it closes the format gap (`.b2d`, sparse frames, mmap, offset), which
is worth more than the pause was. The "and stop" now applies at phase 3, where
the reasoning below is unchanged.*

Note the asymmetry that argues for the pause: every open question below except
the last two is a *phase 2* question. The caching layer is where the design
decisions and the footguns live, not the feature itself. Wait until someone
actually hits phase 1's memory ceiling before paying for that.

Phase 3 stays parked. It is the most interesting engineering here and the least
justified: it needs either a frame-offset parser we do not have (3a) or a C
callback bridge constrained by the GIL and per-block `open()` (3b), to serve a
user who has not shown up yet.

*Superseded: 3a shipped. The frame-offset parser we "do not have" was 45 lines,
which is what changed the answer — not a user showing up.*

## Open Questions, With Recommendations

Each of these is stated in context in the phase it belongs to; this is the
summary and the current leaning.

The four phase-2 questions are now **settled, each the way it was recommended**:
explicit `cache_storage=` with no default, staleness-checked on every open
(which took `check_files=True`, see the phase 2 notes), caching opt-in so
`open(url)` is unchanged, and no tier-2 network test.

The two phase-3 questions are **moot**: 3a needs no I/O plugin, so no registry id
was burned and no upstream change is on the table.

What replaces them is one open question, the only one left in this document:
**how the concurrent fetch behaves against a real endpoint.** Both paths that
overlap requests — the thread pool the sync path uses by default, and
`aget_chunk`'s async one, which `memory://` never even enters — are correct by
test and plausible by arithmetic, and neither has met real S3. The first thing
to do with a bucket in hand is check that 8 in flight is a sensible default
rather than one that trips throttling.

- **Phase 2 cache location and lifetime.** *Recommendation: require an explicit
  `cache_storage=`, no implicit default.* An implicit `platformdirs` cache that
  silently fills a laptop disk with multi-GB arrays is the classic footgun, and
  there is no eviction story we would want to write. Explicit is one argument
  the caller types once.

- **Phase 2 `filecache` vs `simplecache`.** *Recommendation: `filecache`, i.e.
  pay the HEAD-per-open to check staleness.* Silently serving a stale array is
  the worst failure mode this feature has; one round trip against S3 latency is
  not the thing to optimise. `simplecache` behind a flag for callers who know
  their data is immutable.

- **Does phase 2 change what `open(url)` does?** *Recommendation: no.*
  In-memory stays the default and caching is opt-in (`cache=True`, or a
  module-level setting). Silently upgrading a working call from "one GET" to
  "writes a file on your disk" is the kind of surprise that generates issues.

- **Which bucket the tier-2 network test points at.** *Recommendation: do not
  write tier 2 at first.* `memory://` covers every line of code we own; a
  real-S3 test exercises fsspec and AWS, and buys a permanent dependency on an
  object staying public. Add it the first time a bug escapes tier 1 — and if it
  does, publish the fixture under a Blosc-controlled bucket rather than
  borrowing someone's open dataset.

- **The `blosc2_io_cb` id ≥ 160 (phase 3).** *Recommendation: ignore until
  phase 3 is greenlit.* The constraint is "coordinate with the Blosc plugin
  registry", which is not a real obstacle for this project — but do not burn an
  id speculatively, since registration is permanent.

- **Whether to propose the c-blosc2 handle-caching flag upstream (phase 3).**
  *Recommendation: only after measuring, and only if phase 3 happens.* The
  per-block `open()` is a real cost but a speculative one until a workload shows
  it.

One thing that is **not** an open question, recorded here so it does not become
one: `mode="a"` on a remote URL should raise permanently, not "for now". S3 has
no rename and no locks, so append semantics would be a trap rather than a
missing feature.

## Rollout Notes

- Phase 1 is additive: a URL that previously raised `FileNotFoundError` now
  works. Nothing existing changes behaviour, so it does not need a deprecation
  cycle.
- Document the extra in the install docs alongside `[tui]` / `[hires]`. Say
  "any fsspec URL" rather than enumerating protocols — the list is fsspec's to
  grow, not ours to track — and give `pip install "blosc2[fsspec]" s3fs` as the
  S3 recipe so the backend install is visible rather than implied.
- State explicitly that credentials are configured through `s3fs`/AWS
  conventions, not through blosc2.
- Keep the error message for a missing `fsspec` actionable:
  `pip install "blosc2[fsspec]"`. Missing *backends* are fsspec's error to
  raise, not ours to intercept.
- Chained URLs (`filecache::s3://bucket/key.b2nd`,
  `zip://inner.b2nd::s3://bucket/archive.zip`) go through the same
  `fsspec.open()` call and work for free, including reading a Blosc2 file
  straight out of a remote zip. Worth one line in the docs; worth no code.
