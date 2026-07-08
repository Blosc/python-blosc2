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

## 0. NDArray.holding_lock() convenience + open-race gap found while building an example — DONE (2026-07-08, both repos)

Added `NDArray.holding_lock()` (python-blosc2 `src/blosc2/ndarray.py`), a
one-line delegate to `self.schunk.holding_lock()`, matching the existing
`meta`/`vlmeta`/`urlpath` proxy pattern. Test: `test_ndarray_holding_lock`
in `tests/test_locking.py`.

While building `examples/ndarray/mwmr-mode.py` to demonstrate the feature,
hit real `RuntimeError: blosc2_schunk_open_offset(...) returned NULL`
crashes (not the intended lost-update demo) with ~20-40% frequency across
several from-scratch reproductions. Root cause in c-blosc2
`blosc2_schunk_open_offset_udio()` (`blosc/schunk.c`): the bounded retry
added by item 1's open-vs-growth fix (`fa742207`) only wrapped the
*bootstrap* read; the second read done under `force_refresh` (which fires on
nearly every open of a previously-mutated frame) called
`frame_from_file_offset()` directly with no retry, so it could still hit the
transient "frame length exceeds file boundary" race and fail hard. Fixed by
factoring both reads through a shared `frame_from_file_offset_retrying()`
helper. python-blosc2's bundled pin needs to move past this fix.

**Reproduction is a genuine Heisenbug** — it fired 3 times organically while
building the example (proving it's real), then stopped reproducing on the
same machine minutes later even with the fix reverted, across dozens of
retries with several different scripts/parameter combinations, including a
purpose-built C fork test with confirmed several-KB frame_len churn on every
single write (see below). Independent analysis (asked Fable 5 for a second
opinion, see conversation) pinned the mechanical reason: `frame_update_trailer()`
writes the trailer/truncates *before* rewriting the header's `frame_len`
(`blosc/frame.c` ~1265-1279, confirmed by reading the code), so on-disk
inconsistency windows here are single-digit-microsecond reader-side TOCTOU
gaps, not the much wider multi-syscall windows append/growth produces (which
is why the sibling append-vs-open test reproduces reliably and this one
didn't). Deterministic solution: c-blosc2 `blosc/frame.c` gained a
`BLOSC_TESTING`-only fault-injection hook (`blosc2_test_arm_open_race()`)
that directly tampers with the on-disk `frame_len` between the stat() and
header read on a chosen call, instead of relying on scheduling luck.
`tests/test_frame_lock.c::test_open_race_deterministic` uses it and is a
real, bisection-confirmed regression test (fails cleanly on reverted
`schunk.c`, passes on the fix, across 5 repeat runs) — compiled only into
the test-only `blosc_testing` static lib, verified absent from the real
`libblosc2.a` via `nm`. The probabilistic tests
(`test_fork_open_race_update` in c-blosc2, `test_cross_process_open_race_under_update`
in python-blosc2) are kept as best-effort stress coverage of the concurrent
open+update path in general, not as the regression guarantee anymore — that
job now belongs to the deterministic test. Also note for future repro
attempts: the target frame **must be contiguous** — `frame_from_file_offset()`'s
file-boundary check that this race lives in is gated by `if (!sframe)`, so
sparse/directory frames never exercise it at all (a first attempt using
`contiguous=False` silently could not reproduce for this reason, independent
of the timing issue).

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

## 3. Audit remaining mutating paths for RMW-under-stale gaps — DONE (2026-07-08, c-blosc2)

Swept every exported mutating entry point in `blosc/schunk.c` and `blosc/b2nd.c`.
Findings:

- vlmeta add/update/delete: locked; `blosc2_vlmeta_get`/`_exists` poll — OK.
- Fixed metalayers: `blosc2_meta_update`/`_add` write `frame_update_header`
  directly with no `frame_lock` bracket at all (schunk.c:2298-2340), and are
  invisible to other handles' `blosc2_meta_exists`/`_get` (static-inline,
  stale-blind — known, blocked by the plugin no-link design decision;
  c-blosc2 todo item 6). Confirmed out of contract, not a new bug — documented
  in item 4 below (use vlmeta if cross-handle visibility matters).
- `blosc2_schunk_fill_special` has a check-then-act on `nbytes`/`cbytes`
  emptiness outside the lock, but the actual mutation is inside `frame_lock`;
  two racing callers just resolve to last-writer-wins on the whole frame, the
  same accepted semantics as any other overlapping write. Not a new gap
  (fill_special is a single-shot creation-time op in practice).
- `b2nd_insert`/`b2nd_append` are each two separately-locked calls
  (`b2nd_resize` then `b2nd_set_slice_cbuffer`, or `append_buffer` then
  `resize`) — not atomic as a composite op. This is exactly the per-operation
  (not per-call) atomicity the design already commits to elsewhere; the
  NDArray hammer test (item 1) already brackets `resize()`+fill in
  `holding_lock()` for this reason. Documented in item 4, not fixed — fixing
  would mean bracketing the whole of `b2nd_insert`/`b2nd_append` in one lock,
  which changes the composability of resize+write as independent atomic
  primitives; no use case has asked for it.
- `b2nd_delete` is a single `b2nd_resize` call — already fully atomic.
- `blosc2_schunk_from_buffer`/`b2nd_from_cframe`: construct a fresh in-memory
  schunk from a buffer, not a shared on-disk frame — no other handle exists
  to race against, out of scope.

No new bugs found (unlike item 1, which found the open-race bug). Everything
above was either already correct or is an accepted per-operation-atomicity
limit that item 4 needed to document rather than a gap to close.

## 4. Documentation: promote and pin the MWMR contract — DONE (2026-07-08, python-blosc2)

Extended `sharing_across_processes.rst`:

- Multiple writers are supported with `locking=True` on **every** handle
  (advisory; one non-locking handle voids it) — already stated.
- New "Per-operation atomicity" section: `SChunk` chunk updates are atomic
  per chunk; `NDArray` slice writes (`arr[...] = value`) are atomic for the
  whole slice since item 2. Fixed metalayers (`schunk.meta`) called out as
  outside the locking contract, with `schunk.vlmeta` as the alternative when
  cross-handle visibility matters (item 3 finding).
- `holding_lock()` section extended with the two-process `arr[i] += 1`
  idiom via `arr.schunk.holding_lock()`, and a note that `NDArray.append()`
  is a resize+fill composite that needs the same bracket against other
  writers (item 3 finding on `b2nd_insert`/`b2nd_append`).
- Crash caveat, NFS/mmap caveats, and store guarantees were already present
  from earlier passes on this doc — no change needed there.

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
python-blosc2 4.8.0 → caterva2 `blosc2>=` floor bump); see item 3 of
c-blosc2's `plans/todo-locking-swmr.md`. The minor-version bumps (3.2.0/4.8.0
rather than 3.1.6/4.7.1) reflect the significant API additions of this
feature set (decided 2026-07-08). Items 1–3 above should land before the tag
if practical, so 3.2.0/4.8.0 ship the tested multi-writer story rather than a
half-claimed one.

Item 0's `frame_from_file_offset_retrying()` fix is a local, uncommitted
change in c-blosc2 as of this writing (built and tested locally via
`CMAKE_ARGS="-DFETCHCONTENT_SOURCE_DIR_BLOSC2=~/blosc/c-blosc2"`) — it needs
its own commit, and python-blosc2's `BLOSC2_BUNDLED_VERSION` needs to move
past it before the 3.2.0/4.8.0 tag.

## Suggested order

1 (hammer tests) first — it either pins the claim or finds the bugs (it found
one: the open-race fix above); 2 (set_slice bracket) next; 3 (audit) and 4
(docs) together before the release pair; 5–6 are documentation lines inside
4; 7 lives in the caterva2 repo.

Items 1–4 are all done as of 2026-07-08. c-blosc2: `fa742207` (open-race fix,
item 1) and `87a3af7d` (set_slice bracket, item 2) are committed.
python-blosc2's `BLOSC2_BUNDLED_VERSION` is bumped past both (`d58a2c07`,
`24b358ac`), so the overlapping-slice test is green against a non-local
c-blosc2. Item 3 (audit) found no new bugs. Item 4 (docs) is landed in
`doc/getting_started/sharing_across_processes.rst`.

Remaining before the release tag: nothing blocking from this list. What's
left is the release-coupling mechanics themselves (tagging c-blosc2 3.2.0,
cutting python-blosc2 4.8.0, bumping the caterva2 `blosc2>=` floor) and item
7 (caterva2's asyncio-lock → `holding_lock()` port), which lives in the
caterva2 repo, not here.
