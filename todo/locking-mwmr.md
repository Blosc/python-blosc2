# MWMR (Multiple Writer, Multiple Reader) — steps to get there

## Context

Status as of 2026-07-08, after the locking/SWMR push landed in both repos
(c-blosc2 `plans/todo-locking-swmr.md` has the mirror view; design details in
`plans/file-locking.md` and c-blosc2's `plans/high-level-formats-locking.md`).

Key realization from the 2026-07-08 review: **at coarse granularity, locking
mode already is MWMR** — and the docs quietly claim it ("so several processes
can safely write", `doc/getting_started/sharing_across_processes.rst`). The
naming split is: SWMR = the non-locking contract (single writer, readers
follow); locking = the multi-writer contract. Evidence in place today:

- Every mutating frame op takes the exclusive sidecar lock; the generation
  counter forces an exact re-sync of stale handles.
- Append/insert re-sync the cached nbytes/cbytes/nchunks counters under the
  lock before applying deltas (c-blosc2 `3cd3bfe5`, 2026-07-08); update/delete
  refresh via their chunk read; `b2nd_resize` holds the exclusive lock across
  its whole metalayer+chunks sequence.
- `blosc2_schunk_lock()`/`SChunk.holding_lock()` give callers multi-op
  transactions.
- Store-level cross-process multi-writer support exists **and is tested**
  (`test_embed_store_cross_process_writers`, `test_dict_store_cross_process_writers`).

What separates "it basically works" from "we support MWMR" is the list below.
Items 1–4 are roughly a week of work combined; after them MWMR can be
advertised honestly. Items 5–6 are documented limitations / non-goals.

The driving external use case is multi-worker Caterva2 (several server
processes fetching into one shared peercache pool) — see item 7.

---

## 1. Cross-process multi-writer hammer tests — DONE (2026-07-08, both repos)

The real gap was: current frame-level multi-writer evidence was same-process
only (two handles, `test_stale_append_resync` in c-blosc2's
`tests/test_frame_lock.c`); the fork hammer was one writer vs readers; the
cross-process *writer* tests existed only at the store level.

Landed:

- c-blosc2 `tests/test_frame_lock.c`: `test_fork_two_appenders` (two child
  processes append through their own handle; on-disk header counters and
  chunk parity re-read from a fresh open must equal the union of both
  appends — pins the `3cd3bfe5` counter-resync fix cross-process).
- python-blosc2 `tests/test_locking.py`: `test_cross_process_multiwriter_append`
  (N writers append disjoint-signature chunks), `test_cross_process_multiwriter_update`
  (N writers update disjoint chunks while a reader samples for torn/mixed
  content), `test_cross_process_multiwriter_ndarray` (N writers `resize()` +
  fill disjoint row regions via `holding_lock()`, exercising the b2nd
  metalayer path).

**Found a real bug while writing the NDArray hammer test** (not present in
the append/update SChunk tests — needed the tighter growth loop of several
concurrent `resize()`s plus several concurrent fresh opens racing from time
zero): `blosc2_schunk_open_offset_udio()` could return `NULL` under
concurrent frame growth. Root cause: `frame_from_file_offset()`'s bootstrap
read (`stat()` for the file size, then the header) runs before any lock is
taken; a writer growing the frame between the `stat()` and the header read
can leave the header advertising a `frame_len` larger than the now-stale
`file_size` snapshot, which was treated as a hard "frame length exceeds file
boundary" error instead of the transient race it is. Fixed in c-blosc2
(`blosc/frame.c`, `blosc/schunk.c`): bounded retry (50 attempts, 1ms backoff)
around the bootstrap read whenever locking is requested, via a new
`frame_locking_requested()` helper. Pinned by a new fork-based regression
test, `test_fork_open_race` (4 concurrent appenders vs. 4 concurrent
openers) — reproduces the `NULL` return reliably without the fix (4/5
trials), clean across 20+ trials with it. This needs to land in c-blosc2
before the 3.2.0 tag alongside the rest of this feature set (see Release
coupling below); until then `BLOSC2_BUNDLED_VERSION` should move past this
fix's commit.

## 2. Bracket `b2nd_set_slice` in the exclusive lock — DONE (2026-07-08, c-blosc2)

A slice write spanning multiple chunks was N independently-locked chunk
updates. Two writers on overlapping slices interleaved at chunk granularity —
no corruption, but a locked reader could observe a half-applied write and the
merged result was chunk-wise last-writer-wins. `b2nd_resize` already held the
lock across its whole sequence; `b2nd_set_slice_cbuffer` (the actual exported
function behind "set_slice") now does the same, wrapping its call to
`get_set_slice()` in `frame_lock(frame, true)`/`frame_unlock(frame)` (the
bracket nests via `lock_depth`, so the inner per-chunk locks are free, and is
a no-op for unlocked handles). python-blosc2 inherits it through
`__setitem__` with no code changes — confirmed directly, see below.

Landed as `blosc/b2nd.c` (5-line change). Tests:

- c-blosc2 `tests/test_b2nd_set_slice_lock.c` (new file, GLOB-picked-up):
  two writer processes repeatedly overwrite the whole multi-chunk array with
  their own constant value; a reader wrapped in `blosc2_schunk_lock()`/
  `unlock()` samples the whole array and asserts it is never a mix.
  Reproduces the mix in 10/10 trials without the fix, clean across 15+
  trials with it.
- python-blosc2 `tests/test_locking.py::test_cross_process_overlapping_slice_atomic`:
  same shape, through `arr[:, :] = value` and `holding_lock()` — confirms the
  fix reaches Python with zero code changes on this side (also reproduced
  the mix reliably without the c-blosc2 fix during verification, then
  confirmed clean with it).

## 3. Audit remaining mutating paths for RMW-under-stale gaps — MEDIUM (c-blosc2)

Sweep every exported mutating entry point and confirm it either (a) re-syncs
under the exclusive lock before trusting cached state, or (b) is documented
as out of the MWMR contract:

- vlmeta add/update/delete: locked; `blosc2_vlmeta_get`/`_exists` poll — OK.
- Fixed metalayers: `blosc2_meta_update` from one writer is invisible to
  other handles' `blosc2_meta_exists`/`_get` (static-inline, stale-blind —
  known, blocked by the plugin no-link design decision; c-blosc2 todo item 6).
  For MWMR, document: fixed-meta RMW across handles is out of contract.
- `blosc2_schunk_fill_special`, cframe import paths, and anything else that
  writes header/index without going through the locked chunk-op wrappers.

## 4. Documentation: promote and pin the MWMR contract — MEDIUM (python-blosc2)

Extend the locking section of `sharing_across_processes.rst` from a sentence
to a stated contract:

- Multiple writers are supported with `locking=True` on **every** handle
  (advisory; one non-locking handle voids it).
- Atomicity is per operation; a slice write is atomic after item 2 lands.
- Read-modify-write (`arr[i] += 1`, append-position-dependent logic) races
  between writers unless wrapped in `holding_lock()` — show the idiom with a
  two-process example. Per-op locks give serialization, not transactions.
- Concurrent writes to the same region: last-writer-wins (chunk-wise today,
  slice-wise after item 2).
- Crash caveat from item 5, NFS/mmap caveats as already documented.
- Stores: point at the existing EmbedStore/DictStore cross-process guarantees
  and accepted races; `.b2z` stays snapshot-only.

## 5. Crash robustness under multiple writers — LOW (document now, build later)

flock auto-releases when a process dies — good for liveness, but a writer
crashing mid-mutation hands the next lock holder a possibly torn frame, with
no journal to recover from. The stores document this as an accepted race; the
frame level currently doesn't. Action now: document it (item 4). A real fix
(shadow-write / atomic-commit / per-chunk journaling) is a genuine project —
parked until a use case demands it; do not start it speculatively.

## 6. Explicit non-goals (record, do not do)

- **High-concurrency MWMR** (chunk-level locking, MVCC/snapshot isolation):
  one exclusive lock per frame serializes writers and excludes readers during
  writes. Correct, not concurrent. Fine for coordination workloads (peer
  caches, occasional multi-writer stores); parallel-write throughput is a
  different project and a different design.
- **Lock fairness**: flock has no FIFO ordering; writer starvation under
  read-heavy load is possible. Document if it ever bites.
- **NFS**: unchanged, unsupported.

## 7. Downstream consumer: multi-worker Caterva2 — (caterva2 repo)

The convergence point. Several gunicorn workers sharing one peercache pool is
exactly the MWMR use case: the blosc2 layer already makes the cache frames
safe for it (`locking=True` is on every peer-cache handle since caterva2
`2f8eacb`), but Caterva2's fetch→read→touch critical section is an
**asyncio** lock (process-local), and the atime sidecars + budget accounting
are process-local too. The cross-process critical section maps naturally onto
`holding_lock()`. Tracked in caterva2 `plans/c2cache-decoupling.md` §8.1 and
the out-of-scope note of `plans/peercache-locking.md`; listed here because
items 1–4 are its prerequisites.

## Release coupling

All of this rides on the pending release train (c-blosc2 3.2.0 tag →
python-blosc2 4.8.0, whose bundled pin is already at c-blosc2 `3cd3bfe5` →
caterva2 `blosc2>=` floor bump); see item 3 of c-blosc2's
`plans/todo-locking-swmr.md`. The minor-version bumps (3.2.0/4.8.0 rather
than 3.1.6/4.7.1) reflect the significant API additions of this feature set
(decided 2026-07-08). Items 1–3 above should land before the tag if
practical, so 3.2.0/4.8.0 ship the tested multi-writer story rather than a
half-claimed one.

## Suggested order

1 (hammer tests) first — it either pins the claim or finds the bugs (it found
one: the open-race fix above); 2 (set_slice bracket, done) next; 3 (audit)
and 4 (docs) together before the release pair; 5–6 are documentation lines
inside 4; 7 lives in the caterva2 repo.

Item 1's open-race fix is committed upstream as `fa742207`, and
python-blosc2's `BLOSC2_BUNDLED_VERSION` already pins to it. Item 2's
set_slice bracket (`blosc/b2nd.c`, `tests/test_b2nd_set_slice_lock.c`) is
still a local, uncommitted change on top of `fa742207` as of this writing —
it needs its own commit, and the bundled pin needs to move past it before
python-blosc2's overlapping-slice test is green against a non-local
c-blosc2.
