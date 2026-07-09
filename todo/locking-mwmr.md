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
  `resize`) — not atomic as a composite op. Originally logged here as an
  "accepted per-operation-atomicity limit, wrap in `holding_lock()`". **That
  assessment was wrong** — see the correction below, found 2026-07-08 later
  the same day while building `examples/ndarray/mwmr-enlarge.py`.
- `b2nd_delete` is a single `b2nd_resize` call — already fully atomic.
- `blosc2_schunk_from_buffer`/`b2nd_from_cframe`: construct a fresh in-memory
  schunk from a buffer, not a shared on-disk frame — no other handle exists
  to race against, out of scope.

### Correction (2026-07-08, later the same day): `NDArray.append()` was a real data-loss bug, not an accepted limit

The original item 3 pass concluded `holding_lock()` alone made
`append()`/`insert()` safe as a composite op — untested. Building
`examples/ndarray/mwmr-enlarge.py` (several processes `append()`-ing tagged
batches to one 1-D array, each call wrapped in `holding_lock()` per the
existing doc advice) failed on the *first* run: final length 800 instead of
3000, i.e. most batches vanished.

Root cause (python-blosc2 `src/blosc2/ndarray.py`, `NDArray.append()`):
`old_size = int(self.shape[0])` reads the **cached, unrefreshed** shape —
`NDArray.shape` is a bare Cython field read (`blosc2_ext.pyx`), no
staleness check. `super().resize((old_size + len(appended),))` then calls
into `b2nd_resize()`, whose *own* `refresh_if_stale()` correctly updates
`array->shape` to the true (larger, because other writers already appended
under the same lock's earlier turns) on-disk value -- but only *after* the
too-small `new_shape` argument was already computed from the stale
`old_size`. `b2nd_resize()` then sees `new_shape <= (now-refreshed, larger)
array->shape` and takes the **shrink** path, which calls
`shrink_shape()` -> `blosc2_schunk_delete_chunk()` on the trailing chunks —
deleting other writers' just-appended data outright. `holding_lock()`
prevents *concurrent* interleaving but does nothing about a *stale value
read before* the lock's protection actually mattered here, since the read
itself was never gated on a refresh.

Fix: `append()` now calls `self.refresh()` before reading `old_size`. Since
the whole call already runs inside the caller's `holding_lock()`, this
makes the refresh-then-resize-then-fill sequence correctly see any writes
that landed before it and correctly build on top of them, instead of
silently discarding them. Overhead is negligible: benchmarked 561.9 vs
542.7 µs/append (~3.5%) over 2000 single-process appends — the same
`refresh_if_stale()` check `resize()` already runs internally, just done
once more, earlier. Bisected: reverting the one-line fix reproduces the
loss reliably (3/3 trials, 750-900 of 3000 items survived); with it, 5+
clean runs of 3000/3000. `sharing_across_processes.rst`'s `append()`
guidance corrected to describe the fixed (refresh-then-resize-then-fill)
contract accurately.

**Regression test added**: `test_cross_process_multiwriter_ndarray_append`
in python-blosc2 `tests/test_locking.py` (4 writer processes, 30
`append()`s of 25 uniquely-tagged items each, `holding_lock()`-wrapped;
verifies final length, untorn blocks, and exact tag multiset). Bisected
same as the example: fails reliably (3/3, assert 775/800/900 == 3000) on
reverted `ndarray.py`, passes clean (3/3 full-suite runs, 24 tests) with
the fix.

**Leftover**: `b2nd_insert()` (the general C-level resize-at-arbitrary-position
primitive) is not exposed to Python at all — no `NDArray.insert()` exists in
`ndarray.py`/`blosc2_ext.pyx`, so it's out of scope for python-blosc2 today,
but any future C-level or Python binding of it should apply the same
refresh-before-computing-position fix. Confirmed directly: `NDArray.resize()`
(`blosc2_ext.pyx`) always calls `b2nd_resize(self.array, new_shape_, NULL)`
— `start` is hardcoded `NULL`, so mid-array insert (`extend_shape`/
`shrink_shape` with a non-null `start`) is unreachable from Python at all.
Would need a C-level test if this path is ever exercised (item 3's C-side
audit didn't fork-test it either — a genuine remaining gap on the C side).

### Follow-up creative test pass (2026-07-08, later the same day, python-blosc2)

Prompted by the `append()` bug: what else in the concurrent-growth space
isn't exercised? Four new tests added to `tests/test_locking.py`, all
bisection-relevant scenarios that don't already exist, all confirmed
passing reliably (8/8 repeat runs each):

- `test_cross_process_multiwriter_ndarray_2d_grow`: N-D growth has no
  `append()` convenience (1-D only), so users must refresh+resize+fill by
  hand — confirms the fix principle generalizes to N-D and to manually-driven
  `resize()`, not just the library's own 1-D path.
- `test_cross_process_shrink_after_multiwriter_growth`: shrink was not
  exercised by *any* prior multi-writer test (they only grow). Runs growers
  to completion, then shrinks and checks the surviving prefix is untouched —
  `shrink_shape()`'s chunk deletion against the messy, interleaved-order
  chunk layout several concurrent writers produce (not the tidy single-writer
  layout it's usually exercised against).
- `test_cross_process_vlmeta_and_growth_interleaved`: vlmeta and the b2nd
  shape metalayer share the same reload point
  (`frame_refresh_if_stale()` reloads both together, per the answer above)
  but no test mutated both in the same locked block concurrently — checks
  neither desyncs or clobbers the other.
- `test_cross_process_multiwriter_ndarray_append_sparse_nonaligned`: the
  original append() regression test only covered contiguous storage
  starting from an empty, chunk-aligned array. This variant uses sparse
  storage (`contiguous=False`, each chunk its own file — a different
  rewrite path) *and* a non-chunk-aligned starting length (37 items), so
  the first append per writer has to complete a partial chunk first.

None of the four found a new bug — all pass reliably, which is itself useful
signal that the `append()` fix's underlying principle (refresh under the
lock before computing a position) is sound generally, not narrowly patched
for the one reported case.

**`test_growth_swmr_cross_process[False]` flake — investigated and fixed
(2026-07-08, later still, c-blosc2)**. Originally dismissed above as
"matches the documented SWMR-without-locking limitation, not a regression."
That assessment was wrong in the same way the item 3 "no new bugs" call
was wrong — it was a real, fixable gap, not an inherent limit. Root cause,
via `BLOSC_TRACE=1`: `frame_check_stale()`'s *header* read (`frame.c`) is
already opportunistic on a transient failure (keep the cached view, retry
later — a deliberate, commented design choice), but `frame_refresh_if_stale()`'s
*trailer* read, called right after once staleness is detected, was not — a
torn trailer (a writer without locking has zero coordination preventing a
reader from landing mid-rewrite; this can't happen under `locking=True`,
where writers fully serialize) hard-errored via `BLOSC2_ERROR_FILE_READ`,
propagating all the way to Python as `RuntimeError: Error while refreshing
the array`. Same inconsistency-not-limitation pattern as the `blosc2_meta_get`
gap fixed alongside it (see below) — the "opportunistic poll" design was
applied to the header read and to vlmeta, but missed the trailer read in
between.

Fix (`blosc/frame.c`, `frame_refresh_if_stale()`): reordered so the trailer
is read and validated *before* any frame state (`frame->len`, `frame->coffsets`,
`frame->force_refresh`) is mutated, then made the three trailer-read failure
paths (short read, bad magic byte, invalid trailer length) return `0`
(opportunistic, unchanged) instead of erroring. The reorder matters, not
just the return value: once `frame->len` was overwritten, bailing out
opportunistically would have left the frame in a self-inconsistent state
(new length, stale trailer) — the original code mutated state *before*
validating the read that state depends on. Also fixed the same class of gap
one level up in `blosc/b2nd.c`'s own `refresh_if_stale()`: its `blosc2_meta_get(sc,
"b2nd"/"caterva", ...)` re-fetch (a separate read, after `frame_check_stale()`
already succeeded) had the identical hard-error-on-transient-failure bug.

First bisection: 50/50 clean runs with the fix (was failing intermittently,
`BLOSC_TRACE=1` showed `Invalid trailer in frame.` / `Cannot read the
trailer out of the frame.` / a genuine short-read timeout). Looked done.
**It wasn't** — see the two follow-up rounds below, both triggered by
pushback that turned out to be right.

**Round 2 (user caught a real problem in the `b2nd.c` fix, escalated to
Fable 5 for advice)**: the user asked directly whether the `b2nd.c`
`blosc2_meta_get` fix might be shadowing a real issue rather than handling
a transient one. Good catch — my original comment claimed
`schunk->metalayers[]` was "already reloaded fresh" by that point, which is
wrong: `blosc2_meta_get` is a pure in-memory lookup (confirmed by reading
`include/blosc2.h`), and I'd conflated "the header read succeeded" with
"the metalayers are populated," when the actual repopulation happens in a
*separate*, later step. Reverted the `b2nd.c` fix to confirm: stress-testing
with ONLY the `frame.c` trailer fix dropped the flake from ~15-20% to ~2.5%,
not to zero — proving the `b2nd.c` fix was doing real work, just for the
wrong stated reason. Re-investigated and found the *actual* mechanism:
`frame_get_metalayers()` (called inside `frame_refresh_if_stale()`'s
metalayer-reload step) does its OWN fully independent header re-read
(fresh `get_header_info()` + fresh header bytes), a FOURTH uncoordinated
disk read separate from the trailer read and from `frame_check_stale()`'s
own header read. Restored the `b2nd.c` fix with the corrected reasoning.

Escalated the *remaining* ~2.5% to Fable 5 (full prompt/context in this
session's transcript) for advice on the third failure point:
`frame_refresh_if_stale()`'s metalayer/vlmetalayer reload block itself
still hard-erred on failure (trace: `Cannot reload the metalayers from the
refreshed frame.` / `Invalid data`). **Fable's key finding went beyond what
either of us had been chasing**: this wasn't just a flake, it was a
**permanent stuck-state bug** — by the time that reload runs,
`frame->len`/`trailer_len` are already committed and `force_refresh`
already cleared, so if the reload fails, the *next* poll's early-out check
(`frame_len_on_disk == frame->len`) skips retrying forever. If the writer's
*last* mutation happened to land in that window, the reader would be stuck
with an empty/stale metalayer set permanently, not just transiently.
Fable also verified (by reading `get_meta_from_header()`/
`get_vlmeta_from_trailer()` in full) that a bounded retry-in-place is
memory-safe: both parsers `calloc` + publish each slot immediately (no
garbage-pointer state at any early-return point), and `schunk_free_metalayers()`/
`schunk_free_vlmetalayers()` are NULL-safe and idempotent across repeated
attempts. Fix: capture `old_len`/`old_trailer_len` at function entry;
bounded retry (50 attempts, 1ms backoff, matching the existing
`frame_from_file_offset_retrying()` pattern in `schunk.c`) around the
reload; on exhaustion, roll back `frame->len`/`trailer_len`, set
`frame->force_refresh = true` (forces a full retry next poll instead of
short-circuiting), return `0` (opportunistic, not an error). 200/200 clean
runs after — up from the ~2.5% residual.

**Round 3 (user asked about worst-case latency, then asked to escalate
again)**: walking through the retry loop precisely — worst case ~50ms (49
sleeps of 1ms, no sleep after the final attempt) with silent return-0 on
exhaustion, no error, no warning to the caller. Escalated this design
tradeoff to Fable 5 for a second look at the *complete* three-fix picture.
Two more real issues found by reading the final code as a whole:

- **The retry loop is the wrong shape under `locking=True`.**
  `frame_check_stale()` (the only caller) holds the *shared* frame lock
  across the entire `frame_refresh_if_stale()` call when locking is
  enabled, fully excluding writers for its duration — so under locking, a
  reload failure literally cannot be a torn read; it's genuine corruption
  or a real I/O error. The old code would burn ~50ms of guaranteed-futile
  retries while holding that lock (blocking any waiting writer), then
  silently roll back and repeat forever on every subsequent poll — masking
  real corruption from exactly the users who opted into strong guarantees
  via `locking=True`. Verified precisely: `frame_lock()` (`frame.c:211-213`)
  is a true no-op without `frame->locking`, a real OS lock with it.
  Fix: `max_attempts = frame->locking ? 1 : 50`; under locking, a reload
  failure now propagates as a hard error immediately (as it always should
  have), matching the mode's actual guarantees. The no-locking mode keeps
  the full bounded-retry-then-silent-rollback behavior.
- **The rollback wasn't a true restoration.** The cached offsets index
  (`frame->coffsets`) was being dropped *before* the retry loop, alongside
  the `len`/`trailer_len` commit — so on rollback, `len`/`trailer_len` were
  correctly restored but `coffsets` stayed gone, meaning the "restored" old
  view was weaker than the trailer read's own opportunistic bail-outs a few
  lines up (which keep `coffsets` fully intact). Verified `frame_get_metalayers()`/
  `frame_get_vlmetalayers()` never touch `frame->coffsets` (confirmed by
  Fable reading both functions, and independently by me via grep), so
  deferring the drop is safe. Fix: moved the `coffsets` invalidation to
  after the reload loop succeeds, so rollback now restores a genuinely
  fully-consistent old view, on par with the other opportunistic paths.

Also confirmed (Fable, and independently reasoned through): the metalayer
reload exhaustion path is not actually silent in the sense that mattered —
`BLOSC_TRACE_ERROR` fires on every exhausted cycle, printing under
`BLOSC_TRACE=1` with no level filtering, so a chronically-wedged handle is
distinguishable from a genuinely idle one for anyone who goes looking. No
programmatic counter/escalating-warning API was added on top of that —
would be a deliberate API addition (e.g. for Caterva2's peer-cache
diagnostics) if ever needed, not something to bolt on here.

Final verification after all three rounds: 100/100 clean runs for
`locking=False`, 50/50 for `locking=True`, full `tests/test_locking.py` +
`tests/ndarray/` (4386 tests) and c-blosc2's full ctest suite (1661 tests)
all green. No dedicated new C-level regression test added for this whole
area (unlike the open-race and `b2nd_insert` findings) — the existing
Python-level test now serves as an adequate pin across all three rounds; a
deterministic fault-injection test analogous to `test_open_race_deterministic`
would be the natural follow-up if this area gets touched again.

**Process lesson, not just a technical one**: every fix in this
`test_growth_swmr_cross_process[False]` saga survived exactly one round of
scrutiny before the *next* round found something real underneath it —
first "documented limitation, not a bug" (wrong), then "the fix is
correct and complete" (wrong, `b2nd.c` reasoning was backwards), then
"the fix is correct and complete" again (wrong, missed the permanent
stuck-state case), then "the fix is correct and complete" a third time
(wrong, missed the locking-mode futility and the weakened rollback). Two
of those four corrections came from the user asking a specific, pointed
question rather than accepting a confident-sounding summary — asking "are
you sure" at each layer, and escalating for a genuinely independent second
read of the *whole* picture rather than just the newest diff, both pulled
their weight here.

### C-side follow-up test pass — DONE (2026-07-08, later the same day, c-blosc2)

Extended the creative pass to the C side, targeting the one confirmed gap
(mid-array insert, `start != NULL`, unreachable from Python) plus two more:

- `tests/test_b2nd_multiwriter_lock.c` (new file): `test_fork_multiwriter_insert_middle`
  (N processes `b2nd_insert()` at position 0 repeatedly, forcing
  `b2nd_resize()`'s non-NULL `start` branch every call — the path
  `NDArray.resize()`'s hardcoded `start=NULL` makes unreachable from Python),
  `test_fork_shrink_vs_grow` (growers + a shrinker truly concurrent, not
  sequenced like the Python version — shrink had no multi-writer coverage
  anywhere before this), `test_fork_insert_vs_open_race` (bonus best-effort
  attempt at the open-race bug via a wider-window operation than
  `test_fork_open_race_update`'s same-slot update).
- `tests/test_frame_lock.c`: `test_fork_multiwriter_insert_chunk` (raw
  `blosc2_schunk_insert_chunk()`, not b2nd, at index 0 from N processes).

**Found a real, if narrow, C-level version of the append() bug.** The first
draft of `test_fork_multiwriter_insert_middle` called `b2nd_insert()` bare
(no external lock), matching the — wrong — assumption from the "why does
append() work" investigation that `b2nd_insert()`/`b2nd_append()` are
internally safe because they call `refresh_if_stale()` before touching
`array->shape`. That's true, but incomplete: `b2nd_insert()`'s own
`refresh_if_stale()` takes and releases its own shared lock, then
*separately* calls `b2nd_resize()`, which takes and releases its *own*
exclusive lock — two independent lock cycles with a real gap between them.
A concurrent writer growing the array in that gap makes the just-computed
`newshape` stale relative to what `b2nd_resize()`'s own (second) refresh
sees, so it takes the shrink path and deletes the other writer's data —
the exact same bug class as `NDArray.append()`, just with a much narrower
window (a couple of lock cycles apart, not a bare unrefreshed field read),
which is why it needed real fork() concurrency to land (single-process
sequential-handle-reuse and same-process multi-handle round-robin — both
tried as isolating probes — never reproduced it; only true OS-level
concurrent execution did, reliably, 100% of trials before the fix).

Root-caused precisely via `lldb` register inspection (ruled out both
`position < 0` and `ptr == NULL` in `blosc2_stdio_write` as red herrings —
those only ever fired during harmless creation-time writes, confirmed via
an insert-free baseline) and a same-process 4-handle round-robin probe that
passed cleanly, isolating the bug to genuine multi-process interleaving
rather than any simpler "handles need refreshing" explanation.

Fix: bracket every `b2nd_insert()`/`b2nd_append()` call (in the test, not
the library — this is a caller-discipline requirement, exactly like Python
`holding_lock()`) in `blosc2_schunk_lock()`/`blosc2_schunk_unlock()`. This
is the C equivalent of what python-blosc2's own `append()`, `insert()` (if
it existed), and every multi-writer test already do — confirms the *whole
call must be externally bracketed*, not just relying on the library's
internal per-substep locking, is the general rule for any composite
grow/shrink operation in this codebase, C or Python.

By contrast, `blosc2_schunk_insert_chunk()` (raw SChunk, not b2nd) genuinely
*is* safe bare: it takes one lock, does its stale check and the insert
under that single continuous hold, then releases — confirmed by
`test_fork_multiwriter_insert_chunk` passing reliably with no external
bracket at all.

Also found (and fixed) a test-design bug of my own while chasing an
intermittent ~1/15 failure in `test_fork_shrink_vs_grow` even *after*
correct bracketing: the verification assumed only the tail-most run could
be legitimately short (partially shrunk). Wrong — since growers keep
appending after a shrink, *any* block (including the initial prefix, if
the shrinker gets ahead of the growers early) can end up truncated
mid-array with fresh, complete blocks appended after it. Fixed the
invariant to just "no run mixes two values, none exceeds its max possible
length" with no positional requirement; 70/70 clean runs after.

All four C tests pass reliably (15-70 repeat runs each, per test). Full
ctest suite (1661 tests) green.

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
