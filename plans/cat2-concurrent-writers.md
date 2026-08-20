# Concurrent Chunk Writers For Caterva2

Written 2026-08-21, after [plans/cat2-block-granularity.md](cat2-block-granularity.md)
gave `C2Array` block-granular *reads* over HTTP ranges (branch
`cat2-block-granularity`; the Caterva2 side is `range-honesty`).

Nothing here is implemented yet.  Everything under *What was verified* was
measured or read out of the code on 2026-08-20/21; everything under *Plan* is
proposal.

## The question

Several processes, on several machines, want to fill one `.b2nd` living in a
Caterva2 server, each writing its own chunks, at the same time.  What does that
take, and how much of the block-granularity work carries over?

**Verdict: almost none of the read machinery carries over, and that is fine,
because the write path turns out to be small — one endpoint — provided the
array is pre-sized and each chunk is written exactly once.**  The read side made
transports uniform behind one primitive (`read_range`); writing has no safe
mirror of that, and this stays a Caterva2 capability rather than a blosc2-remote
one (see [Non-goals](#non-goals)).

## What was verified

Measured on this machine (Apple M4 Pro, APFS, python-blosc2 4.11.1.dev0 against
c-blosc2 3.3.2), on local files.  Code references: c-blosc2 at
`/Users/faltet/blosc/c-blosc2` (`main`), Caterva2 at
`/Users/faltet/ironArray/Caterva2` (`range-honesty`).

### A contiguous frame compacts on rewrite, but appends on first write

`frame_update_chunk()` (`blosc/frame.c:4281`) writes the new chunk *in place* and
moves the whole payload tail when the compressed size changes
(`frame.c:4546-4580`).  Cost is therefore O(bytes after the chunk), not O(file),
and the physically-last chunk already costs nothing: `tail_nbytes` is 0 and the
move is skipped (`frame.c:4548-4549`).  Median of 5 `update_chunk` calls, 1 MB
chunks, clevel 1:

| file | chunk 0 | middle | last | same cbytes |
|---|---|---|---|---|
| 6.9 MB (20 chunks) | 1.482 ms | 0.852 ms | 0.244 ms | 0.285 ms |
| 27.6 MB (80) | 5.638 ms | 2.293 ms | 0.269 ms | 0.291 ms |
| 110.2 MB (320) | 21.213 ms | 10.828 ms | 0.296 ms | 0.282 ms |

This is the behaviour of `9200990b` ("Fix contiguous-frame b2nd resize growth on
chunk updates", 2026-03-23), which replaced append-at-end-leaving-a-hole with
compaction.  It is in the bundled 3.3.2.  The older hole behaviour is not
recoverable as a flag without a format change: `get_coffsets()` locates the
offsets block at `header_len + cbytes` (`frame.c:1841-1846`), so the header's
`cbytes` is simultaneously the payload extent and the user-visible compressed
size; holes make those diverge and there is no second field.

**But a chunk whose previous content was *special* does not compact at all.**
`old_chunk_is_regular = (!frame->sframe && old_offset >= 0)` (`frame.c:4377`);
zero/NaN/uninit chunks live entirely in the offsets with the high bit set and
carry no payload, so there is no tail to move and the new chunk is appended at
`new_chunk_offset = cbytes` (`frame.c:4379`).  Filling a 320-chunk array
pre-sized with `blosc2.uninit()`, in random chunk order:

```
pre-sized uninit file on disk:   221 bytes  (335.5 MB logical)
fill all 320, random order:      median 0.459 ms/chunk, max 4.136 ms → 110.2 MB
rewrite an already-written one:  9.157 – 14.029 ms   (the compaction above)
```

Flat, position-independent, and **no other chunk's offset changes**.  That one
fact is what the whole design below is built on.

(`clevel=0` gives the same flatness for repeated rewrites, since every chunk is
exactly `nbytes + overhead` and the tail never moves — 0.44-0.49 ms at any
position on a 336 MB file.  Kept here as a note; the write-once design does not
need it.)

### `uninit` is a usable sentinel; `zeros` is not

Both are special chunks, so "written or not" is legible from the offsets in
either case — the tag carries it, never the data.  The difference is what
happens when a writer legitimately stores an all-zero chunk:

```
compress2(np.zeros(...))   → cbytes=32,   special=ZERO
compress2(np.arange(...))  → cbytes=1152, special=regular
```

Blosc2 detects the run and emits a special ZERO chunk, so with a `zeros`
pre-fill a genuinely-all-zero written chunk is indistinguishable from a
never-written slot.  With `uninit` the two separate cleanly
(`schunk.iterchunks_info()`):

```
chunk 0: special=ZERO         ← written, data really was zeros
chunk 1: special=NOT_SPECIAL  ← written, real data
chunk 2: special=UNINIT       ← never written
```

Cost of `uninit`: an unwritten chunk reads as undefined bytes, so completeness
has to be part of the contract rather than a nicety.  See
[Progress is the offsets block](#progress-is-the-offsets-block).

### The frame length is not a validator

A special-chunk write sets `chunk_cbytes = 0` and leaves `new_cbytes`
unchanged, so the file length moves only if the recompressed offsets block
happens to change size:

```
after uninit create      size=  221  md5=672911ce0aca
after ZERO chunk write   size=  277  md5=063520e5ea6c
after 2nd ZERO write     size=  277  md5=63bf170ef1c8   ← same length, new content
after regular write      size= 1429  md5=9aed1d5dbddb
```

Worse than a missed invalidation: since `new_cbytes == cbytes`, that write
rewrites the offsets block **in place**, where a regular append writes it past
the new chunk.  So a zeros write both opens a torn-read window on the offsets
and is invisible to a length check.

### The generation counter is

`.b2lock` carries a `uint64` at offset 8 (`FRAME_LOCK_SEQ_OFFSET`,
`frame.c:130`), bumped by every exclusive acquisition (`frame.c:269-271`).
c-blosc2's own comment states the reason: it "detects mutations by other handles
exactly, even when the frame length on disk ends up unchanged".  It lives
outside the frame bytes, so only a server with local filesystem access can serve
it — which is exactly what Caterva2 is.

### Caterva2 has no chunk-write endpoint, and its write path is accidentally safe

Write surface today is `api/upload` (whole file, `server.py:1279`), `api/append`
(axis 0, `server.py:1413`) and `api/upload_lazyarr`; `api/chunk`
(`server.py:924`) is GET-only.  Neither write endpoint takes any lock.  They are
safe today only because they are `async def` bodies that never await across
their blocking blosc2 calls, in a single-process deployment
(`uvicorn.run(app)`, `server.py:3351`).  Moving the write to a threadpool —
which concurrency requires — removes that accident, so the locking is not
optional extra credit.

`locking=True` (`src/blosc2/storage.py:212`), `holding_lock()`
(`src/blosc2/schunk.py:476`) and the cross-process multi-writer tests already
exist; see `todo/locking-mwmr.md`, whose item 7 is this use case.

### Pre-sizing needs no new endpoint

A pre-sized uninit array is **221 bytes for a 335.5 MB logical array**, and
`.b2nd` is in `BLOSC2_NATIVE_SUFFIXES` (`caterva2/services/srv_utils.py:35`), so
`api/upload` already stores it verbatim.  Creation is
`blosc2.uninit(...)` locally plus an existing upload; the file *is* the geometry
specification.  (Quota is then accounted at 221 bytes, so the chunk-write
endpoint has to re-check it — see phase 1.)

## The design

### Pre-sized, write-once

1. The owner creates the array locally with `blosc2.uninit(shape, dtype, chunks,
   blocks, cparams)` and uploads it (~200 bytes).  Geometry is fixed here and
   never changes: **no writer ever resizes**.
2. Writers own disjoint chunk indices, agreed between themselves; the server
   does not arbitrate the partition.
3. Each chunk is written **exactly once**.  A second write is refused.

Everything good follows from 3: writes never move data (~0.5 ms), never
invalidate another reader's chunk offsets, and never need a read-modify-write of
a partially covered chunk.

### Progress is the offsets block

The UNINIT-vs-everything-else tag *is* the completion record.  No manifest, no
sidecar bitmap, no progress endpoint:

- **Write-once enforcement**: the server checks slot *n* is UNINIT before
  accepting.  Note it must test UNINIT specifically, not "is special" — a
  written all-zero chunk is special too.
- **Atomic by construction**: the tag flips in the same offsets rewrite that
  publishes the chunk, under the same lock.  No window where a chunk is on disk
  but unrecorded, or the reverse.
- **Readers get it free**: `ByteRangeNDSource` already decodes a negative offset
  and reconstructs the special chunk locally (`src/blosc2/proxy_source.py:706`,
  `853`), so an unwritten chunk costs zero bytes and zero requests.
- **Progress is one range read**: the offsets block is a single span the branch
  already knows how to locate.

It deliberately records no in-progress state, no identity, no timing and no
history.  That gives crash *recovery* (rerun the unwritten set) but not
*leases*: two writers who both believe they own chunk 7 are resolved by the
refusal, not prevented.

### The one remaining tearing window

For a regular append, the new chunk is written at `header_len + cbytes` — which
is exactly where the *old* offsets block lives.  A reader that fetched the
header and then reads the offsets can therefore land on a half-written chunk.
This is the branch's two-request open, and it is why an ETag is load-bearing
rather than a nicety.

## Plan

### Phase 1 — `POST api/chunk/{path}` (caterva2) — the main piece

Body is one compressed chunk; `nchunk` is a query parameter.  Under
`get_writable_path(path, user)`:

1. Refuse anything not a stored contiguous `.b2nd` — lazy expressions, `.b2z`
   members, HDF5 leaves.  `api/info`'s discriminator from phase 2 of the
   block-granularity plan already reasons about this.
2. Validate the chunk header against the array's geometry (`nbytes`,
   `blocksize`, `typesize`).  A mismatched chunk corrupts the array outright, so
   this is not optional.
3. Re-check quota against the *delta*, since creation only accounted ~200 bytes.
4. Open with `locking=True`, and inside `holding_lock()`: read the offsets,
   refuse with **409** unless slot *n* is UNINIT, then `update_chunk`.
5. Run the whole thing in a threadpool — it is blocking, and it must not hold
   the event loop.

Acceptance: N processes filling disjoint chunk sets of one array converge to the
exact expected contents; a second write to any slot returns 409; a torn or
mis-shaped chunk is refused before it reaches `update_chunk`.

### Phase 2 — ETag from the generation counter (caterva2) — small, load-bearing

Serve the `.b2lock` counter as a strong `ETag` on `api/info` and on ranged
`api/fetch`/`api/download`, so a client can prove its header and its offsets came
from the same frame.  A `pread` of 8 bytes.

- Not the file length: proved above that a zeros write can leave it unchanged.
- Not `If-Match` on the *write* path: the UNINIT check is already the
  compare-and-swap, and a better one — it tests the real state, not a token.
- Define the fallback for an array with no sidecar yet (never written under
  locking): either create it on first open, or serve a documented weaker
  validator.

### Phase 3 — `C2Array.update_chunk` / `written_chunks` (blosc2) — small

`update_chunk(nchunk, chunk)` and `aupdate_chunk` through the pooled client the
branch added, plus `written_chunks() -> np.ndarray[bool]`, one range read of the
offsets, decoded locally.  No general `__setitem__`: a partially covered chunk
is a networked read-modify-write and would need CAS to be safe.

### Phase 4 — `stamp`: appended-to vs replaced (blosc2) — small, needs a decision

`C2Array.stamp` is `mtime:cbytes` (`src/blosc2/c2array.py:749`) and answers "are
these the same bytes?".  Under append-only writing the answer is "no" after
every chunk write, which would discard a `Proxy` cache that is still entirely
valid, since existing chunks never move.  The stamp needs to answer the narrower
question — *replaced, or merely appended to?*  There is no UUID in the frame
header, so this is the one genuinely open design question here.  Options to
weigh: a server-side identity token (inode + creation time) carried in
`api/info`; a creation nonce written into vlmeta at pre-size time; or splitting
the stamp into an identity part and a freshness part.

### Phase 5 — Completion and publish (caterva2)

The completion condition is free: after each accepted write, inside the same
`holding_lock()` region, scan the already-decompressed offsets for remaining
UNINIT slots.  State lives in vlmeta: `filling → publishing → published(url)`.

- On zero remaining, compare-and-set `filling → publishing`.  The lock makes it
  **exactly-once**: two writers finishing together both see zero, one wins the
  flip, the winner owns the publish.
- Do the upload **outside** the lock, then flip to `published` with the URL.  A
  slow upload must not block writers.
- `POST api/publish/{path}` is the primitive; auto-trigger on completion is a
  thin layer over it, which also gives a manual retry for the stuck-in-
  `publishing` case.
- **The destination must not come from the client.**  A client-supplied `s3://`
  URL lets the server be aimed at a bucket the caller controls.  The server
  config names the destination root; the array supplies a relative key only.
  Credentials stay on the server, which also means writers never hold them.

What lands in S3 is a finished contiguous frame — exactly what this branch's
`FsspecNDSource` reads with byte ranges.  Caterva2 is the write path, the object
store is the read path, and both ends already work.  Publishing has none of the
problems of writing chunks to S3: the frame is immutable by then, so no locking,
no ETag, no partial writes.

Acceptance: an array filled by N writers publishes exactly once, is readable
from S3 by `blosc2.open(url, lazy=True)` with block granularity, and a crash
mid-publish is recoverable through the explicit endpoint.

### Phase 6 — Tests and a bench

- Cross-process hammer: N writers × disjoint chunks against a live server, plus
  a reader sampling throughout; assert no torn chunk, exact final contents, and
  `written_chunks()` monotone.
- The zeros case explicitly: an array whose writers all send ZERO chunks must
  still complete and publish (this is the case `zeros` pre-filling would break).
- ETag: a zeros write must change it (the length does not).
- Extend `bench/ndarray/cat2-block-granularity.py`'s stand-in server to accept
  chunk writes, so the write path is measurable without a deployment, the way
  the read path already is.

## Risks and open questions

- **Phase 4 is unresolved** and everything else can land without it; the cost of
  deferring is that a `Proxy` over an array still being filled re-fetches more
  than it needs.
- **Crash mid-fill** leaves an array permanently incomplete.  Correct, but a
  coordinator needs `written_chunks()` and a reassignment story; consider a
  reporting-only staleness timeout.
- **Multi-worker deployment**: `locking=True` covers the frame, but the
  process-local caches in `server.py` (the mtime-keyed opened-array cache, the
  `locks` dict at `server.py:494`/`952`) do not.  Decide whether multi-worker is
  in scope now or after.
- **Crash mid-write** hands the next lock holder a possibly torn frame; there is
  no journal.  Same accepted limitation as item 5 of `todo/locking-mwmr.md`.
- **Lock fairness**: `flock` has no FIFO ordering, so a read-heavy array could
  starve writers.

## Non-goals

- **Chunk writes over fsspec/S3.**  Three independent blockers, only the last of
  which is about validators: object stores have no partial write, so every chunk
  write rewrites the whole frame object; the offsets block is shared mutable
  state, so concurrent writers lose updates (S3 conditional writes give CAS, but
  each retry is another full-object rewrite, so it degrades exactly where it
  should scale); and there is no lock, hence no generation counter.  The read
  side generalised because reading needs one primitive that every backend has.
  Writing needs mutual exclusion plus partial in-place writes, and only a server
  with a real filesystem has both.  Do not add a `write_range` to
  `ByteRangeNDSource`.
- **Rewriting live chunks.**  Allowed in principle, costs 9-21 ms of compaction,
  and forfeits write-once enforcement, offset stability and the completion
  record all at once.  Refuse it; revisit only with a use case.
- **Leases / ownership arbitration.**  The offsets record what is written, not
  who is writing.  An external coordinator's job.
- **Server-mediated writes to a frame that lives on S3.**  Interesting, and the
  natural extension of phase 5 in the other direction; out of scope here.
- **A hole-plus-repack update mode** to make rewrites cheap.  Needs a second
  counter in the format (the trailer is msgpack and variable, so a softer home
  than the fixed header) plus a `vacuum`/`repack`, which neither repo has.
  Format project, not a plan item.

## Reproducing the measurements

Each was a short script run against local files; none needs a server.

- **Rewrite cost by position**: build a contiguous `.b2nd` of *n* 1 MB chunks,
  time `schunk.update_chunk()` on chunk 0, *n*/2 and *n*-1 with freshly
  compressed data, and again with the chunk's own bytes (the same-cbytes case).
- **Append on a special slot**: `blosc2.uninit(...)`, then fill all chunks in a
  random permutation, timing each; then rewrite three of them.
- **Sentinel**: `compress2` an all-zero buffer and read the special bits at
  `chunk[31] >> 4 & 0x7`; cross-check with `schunk.iterchunks_info()`.
- **Length is not a validator**: stat + md5 the file after a create, two ZERO
  writes and a regular write.

## What landed

Phases 1, 2, 3 and 5 (2026-08-21), on `cat2-concurrent-writers` here and
`c2cache-monorepo` in Caterva2.  Phase 4 is still open, on purpose; phase 6's
tests landed with the phases they cover, its bench did not.

| phase | where | commit |
|---|---|---|
| 1. `POST api/chunk` | caterva2 `server.py` | *Accept one chunk at a time into a slot that holds none* |
| 2. ETag | caterva2 `server.py` | the same commit |
| 3. client writes | blosc2 `c2array.py`, `proxy_source.py` | *Write a chunk of a remote array, and read which ones were written* |
| 5. completion and publish | caterva2 `server.py` | *Publish an array once every one of its chunks has landed* |
| 5a. atomic publish | caterva2 `server.py` | *Move a published array into place instead of streaming into it* |

The shape held: a pre-sized `uninit` array, one write per slot, the offsets as
the record, and a 409 as the whole of the coordination.  15 tests against a live
subscriber (`caterva2/tests/test_chunk_writes.py`) and 10 against a stand-in
(`tests/ndarray/test_c2array_writes.py`), which is where the client-side
behaviour is pinned without a service.

### What the work found

- **Reading the index is not the same question as reading blocks.**
  `C2Array.written_chunks()` was gated on `serves_blocks`, which also weighs
  whether *splitting a chunk into blocks* would pay — a frame of small chunks
  reported that it served no blocks and so could not say which chunks were
  written either.  The geometry half is now `_reports_geometry` and the index is
  read whatever the chunks cost.  `serves_blocks` is unchanged for the block
  path, which reads it at `Proxy` build time and must keep costing nothing.
- **Invalidating the index means the header too.**  A write moves the frame's
  length and its payload extent, and the offsets are found through both, so
  dropping the offsets alone left the next read looking for them at the old
  position.  `ByteRangeNDSource.invalidate_index()` marks both stale and reads
  neither until something asks.
- **The completion scan had to move off `iterchunks_info`.**  It reads a lazy
  chunk apiece — 3.5 µs each, so 17.8 ms on 5000 chunks, which would have made a
  fill cost the square of its length.  The write-once check reads the one
  chunk's header instead (~3 µs), and the count comes from the offsets in one go
  (0.37 ms at 5000 chunks, 0.39 ms at 20000 — flat where the walk is linear).
- **A publish that streams into place is readable before it is whole.**  The
  destination file exists from its first byte and a frame is not readable until
  its last, so a reader polling for the published array opened it mid-copy and
  got a NULL back.  Found by the test doing exactly that, which had passed only
  while the copy won the race.  Published under a name of its own and moved into
  place.
- **A stale blosc2 handle corrupts a write silently.**  Two handles open over one
  frame, one of them writing, leaves the frame unreadable — `Invalid arguments
  for stdio write` under `BLOSC_TRACE=1`, and nothing at all without it: the
  write itself does not raise.  This is the hazard `todo/locking-mwmr.md`
  documents, but the silence of it is worth knowing.  Both the endpoint and the
  test stand-in are written to hold exactly one handle and to drop it before
  anything reads the file again.

### Phase 4 is still open

`C2Array.stamp` is still `mtime:cbytes`, so a `Proxy` cache over an array being
filled is still discarded on every write, though every chunk it holds is still
exactly where it was.  The ETag added in phase 2 is the *freshness* half of the
question and does not answer this one: it changes on an append as readily as on a
replacement, which is precisely the distinction wanted.

What would answer it is an identity that survives appends and not replacement.
The frame header has no UUID to use.  The candidate is a nonce written into
vlmeta when the array is laid out — `api/info` already carries vlmeta, so it
costs no request — with today's stamp as the fallback for an array created
without one.  That is a convention as much as a change, so it is left for a
decision rather than settled here.

### Left undone

- The bench of phase 6: `bench/ndarray/cat2-block-granularity.py`'s stand-in
  does not accept writes, so the write path has no measured numbers of its own
  against a loopback server.
- Multi-worker deployment.  `locking=True` covers the frame across processes and
  the `.b2lock` counter is read from disk, so the ETag is right there too; the
  per-path `asyncio.Lock` is not, and neither are the mtime-keyed open-array
  caches.  Nothing here depends on it, and nothing here provides it.
