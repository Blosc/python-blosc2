# utf8 columns: closing the write/ingest gap (incremental plan)

**Status:** I1 and I2 both LANDED (2026-07-17, branch `enhancing-ctable3`).
Taxi ingest: 3622 ms → 870.5 ms after I1 → 598.6 ms after I2, now within
~20% of `string()` (490.4 ms) and clearly ahead of `vlstring()`
(1292.2 ms). See "Honest assessment" below for the full before/after
table. Successor work to `plans/utf8-reads-filter-optim.md` (U1 + U2,
landed on branch `enhancing-ctable3`, commits `3a7d9a3d`, `45ec7906`).
Read that plan first for background and for the exact working rebuild
commands referenced below; this plan assumes it.

**Audience:** an implementing model/developer who has NOT read the
discussions that produced this plan. Everything needed is in this file.

**Practical notes (carried over from the read-side plan, still true):**

- Locate code by **symbol name with grep**, never by line number.
- Run Python/pytest through the `blosc2` conda env:
  `conda run -n blosc2 python -m pytest ...` (or
  `/Users/faltet/miniforge3/envs/blosc2/bin/python`). Never use the repo
  `.venv` (stale).
- **Editing `.pyx` files does NOT trigger a rebuild in the editable
  install.** I2 needs a real rebuild — see `plans/utf8-reads-filter-optim.md`'s
  U2 section for the exact working commands (`cmake --build build_py314
  --target utf8_ext`, then copy the resulting `.so` into
  `.../site-packages/blosc2/`) and the `NPY_TARGET_VERSION` gotcha already
  solved there (not directly applicable to this kernel, which doesn't use
  NumPy's StringDType C API, but the rebuild workflow is identical). This
  is why I2 is sequenced after I1 and not merged with it.
- **Docstrings and code comments must be self-contained.** Never reference
  this plan, phase labels ("I1", "I2"), or benchmark history from source,
  tests, or bench scripts — state the semantics directly.
- utf8 tests live in `tests/ctable/test_utf8.py`; match neighboring style —
  in particular mirror the existing `force_kernel_mode` fixture (read-side
  kernel toggle) for a new write-side sibling.
- The benchmark harness is `bench/ctable/bench_string_kinds.py` (real
  Chicago-taxi `company` column at 1e7 rows + synthetic high-cardinality
  free text at 2e6 rows). The `ingest` line is the one this plan targets.
- The dev machine is an Apple-silicon Mac; benchmark targets assume it.

---

## The problem, quantified

`Utf8Array` ingest (`src/blosc2/utf8_array.py`) is far slower than the
alternatives on the same benchmark (`bench_string_kinds.py`,
`t.extend({"s": values, "val": float_vals}, validate=False)` on 1e7 rows
of the Chicago-taxi `company` column):

| kind       | ingest  | vs utf8 |
|------------|---------|---------|
| `utf8()`   | 3622 ms | —       |
| `string(max_length=44)` | 478 ms | 7.6x faster |
| `vlstring()` | 1193 ms | 3x faster |

**Root cause (verified against current code):** `Utf8Array.extend()` /
`.append()` buffer rows one at a time in a pure-Python loop:

```python
def extend(self, values: Iterable[Any]) -> None:
    for v in values:
        v = self._coerce(v)
        self._pending.append(v)
        self._pending_chars += len(v)
        self._flush_if_needed()
```

and `_rewrite_from()` (called every `_FLUSH_ROWS = 4096` rows) re-encodes
every row individually:

```python
encoded = [v.encode("utf-8") for v in values]  # one new bytes object per row
blob = b"".join(encoded)  # bulk, fine
...
lengths = np.fromiter(
    (len(e) for e in encoded), dtype=np.int64, count=len(encoded)
)  # per-row len(), NOT vectorized despite being a numpy call
```

By contrast, `string()` ingest is a **single** `np.ascontiguousarray(raw,
dtype='U44')` call with zero per-row Python — that's the ceiling utf8 is
chasing. `CTable`'s dispatch layer (`ctable.py`) is **not** the
bottleneck: it routes `utf8`/`vlstring`/`vlbytes` through one shared
generic branch (`_is_varlen_scalar_column`) with no utf8-specific
inefficiency beyond one incidental, cheap `list(raw_columns[name])` copy.

A standalone reproduction of the two hot Python loops above (taxi-like
ASCII workload, 1e7 rows, Python-level cost only, no NDArray I/O) measured
**1557 ms** — i.e. these two loops account for ~43% of the 3622 ms total;
the remaining ~2065 ms is NDArray resize/write/compression cost that the
items below do not touch (flagged as a follow-up, out of scope here).

**Dead end already checked, do not repropose:** `np.strings.encode()` on a
`StringDType` array was measured *slower* than the current per-row Python
loop (0.385s vs 0.049s / 2M rows) **and** is unsafe here — it returns a
fixed-width `'S<n>'` array padded with trailing NUL bytes, indistinguishable
from real trailing-NUL string content that utf8 columns must support
losslessly (see the NUL-bearing tests already in `tests/ctable/test_utf8.py`).

---

## Design decisions already made (do not re-litigate)

1. **I1 covers `_rewrite_from` too, not just `extend()`.** The per-row
   `.encode()` loop there is pure Python and rebuild-free — it's the
   single biggest lever in I1 (item I1.c below).

2. **`str.isascii()` is O(1) per call.** It reads a cached PEP-393 flag
   set at string-creation time, not a per-character scan — verified
   empirically (~11.5ns/call, doesn't scale with string length).

3. **`"".join(values).encode("ascii") == b"".join(v.encode("utf-8") for v
   in values)` whenever every value is ASCII.** UTF-8/ASCII encoding has
   no cross-character state, so join commutes with encode — verified by
   direct byte-equality assertion in a prototype, not just argued. For
   ASCII values, `len(v)` (character count) already equals the UTF-8 byte
   count, so lengths can be computed straight from the un-encoded strings,
   no intermediate `encoded` list needed.

4. **Critical pitfall — call out prominently in review.** `flush()` does
   `values, self._pending = self._pending, []` — a **rebind**, not a
   mutation. Any chunked-extend implementation must re-read
   `self._pending` after a flush that may have run mid-loop; never cache
   `self._pending.append`/`.extend` as a local variable across a flush
   boundary. Getting this wrong silently drops rows with **no exception**
   — a naive prototype implementation did exactly this, dropping 99.96% of
   a 1e7-row batch silently. Add a regression test asserting `len(arr) ==
   n` after a multi-flush single `extend()` call, not just content
   equality (content-equality tests alone would not have caught this).

5. **Land order: I1 first (no rebuild, always-on), I2 second (needs a real
   rebuild).** I2's fallback tier is I1's optimized pure-Python path, not
   the pre-I1 naive path. Mirrors the U1→U2 precedent in the read-side
   plan. I1.a and I1.c land together as one item ("I1") rather than being
   split — no strong reason to separate them; I1.c's benefit is best
   demonstrated with I1.a's chunking already in place.

6. **I2's kernel is a second function in the existing
   `src/blosc2/utf8_ext.pyx`**, not a new `.pyx` file. Verified: the
   custom command in `CMakeLists.txt` already depends on the whole file
   and regenerates `utf8_ext.c` from it — adding a `def` requires **zero**
   `CMakeLists.txt` changes.

7. **`PyUnicode_AsUTF8AndSize` is already declared in Cython's
   `cpython.unicode` pxd** (verified directly against
   `.../site-packages/Cython/Includes/cpython/unicode.pxd`, Cython 3.2.5)
   with an `except NULL` clause — Cython auto-propagates the Python
   exception (`TypeError`, `UnicodeEncodeError` for a lone surrogate,
   etc.) on failure, no manual NULL-check/re-raise needed. It requires the
   GIL (mutates the string object's internal UTF-8 cache); do not attempt
   a `nogil` second pass — U2 didn't bother either for the analogous
   reason, and the encode call itself (not the memcpy) is the dominant
   cost.

---

## I1 — Pure-Python/NumPy ingest speedup (no rebuild)

### I1.a — Chunked bulk-check `extend()`

**Where:** `Utf8Array.extend` (`src/blosc2/utf8_array.py`).

Pull `values` in chunks of `_FLUSH_ROWS` via `itertools.islice` (keeps
support for genuinely lazy iterables — `Utf8Array.copy()` calls
`out.extend(self)`). Per chunk, try a bulk fast path; fall back per-item
only for that chunk if needed:

```python
def extend(self, values: Iterable[Any]) -> None:
    it = iter(values)
    while True:
        chunk = list(itertools.islice(it, _FLUSH_ROWS))
        if not chunk:
            break
        if all(type(v) is str for v in chunk):
            self._pending.extend(chunk)
            self._pending_chars += sum(map(len, chunk))
        else:
            for v in chunk:
                v = self._coerce(v)
                self._pending.append(v)
                self._pending_chars += len(v)
        self._flush_if_needed()
```

Key properties:

- `type(v) is str` (not `isinstance`) deliberately excludes `numpy.str_`
  (a `str` subclass), so `np.array(["uno", "dos"])`-style input still
  falls to the slow per-item path and gets `_coerce()`'s `str(value)`
  normalization — preserves `test_ctable_utf8_extend_numpy_fixed_width_input`
  unchanged.
- A chunk containing `None` (nullable columns) or a non-`str` fails the
  bulk check and falls to the slow path, preserving `_coerce()`'s
  sentinel substitution and exact `TypeError` messages.
- Per design decision 4: never cache `self._pending.append`/`.extend`
  outside this loop body.

**Measured** (extend-loop only, `_rewrite_from` unchanged, taxi-like ASCII
workload, 1e7 rows): 1557ms → 943ms (**-39%**).

**`append()`/`_flush_if_needed()`:** leave essentially untouched — already
minimal single-row overhead, no measurable win from rewriting them.
`set_all()` (used by `sort_by(inplace=True)`/`compact()`) is unaffected by
I1.a but benefits automatically from I1.c since both share `_rewrite_from`.

### I1.b — Document the flush-cadence trade-off (no code change)

Under I1.a, `_pending_chars`'s `_FLUSH_CHARS` bound is only checked once
per `_FLUSH_ROWS`-row chunk instead of every row, so an unusual batch of
many multi-MB strings could overshoot `_FLUSH_CHARS` by up to one chunk
before flushing. Low risk (`_FLUSH_ROWS` is the binding bound for
realistic short-string workloads — flushes trigger every 4096 rows on the
taxi data, well before the char bound). Document this explicitly in the
code comment; add a sanity test (extend with ~20 multi-MB strings, assert
correct read-back and roughly-bounded peak memory) rather than adding
complexity to close the soft-bound gap.

### I1.c — Bulk `_rewrite_from` via `str.join` + `isascii()` fast path

**Where:** `Utf8Array._rewrite_from` (`src/blosc2/utf8_array.py`).

```python
def _rewrite_from(self, pos: int, values: list[str]) -> None:
    if values and all(v.isascii() for v in values):
        blob = "".join(values).encode("ascii")
        lengths = np.fromiter(map(len, values), dtype=np.int64, count=len(values))
    else:
        encoded = [v.encode("utf-8") for v in values]
        blob = b"".join(encoded)
        lengths = np.fromiter(
            (len(e) for e in encoded), dtype=np.int64, count=len(encoded)
        )
    ...  # rest (resize/slice-write/cumsum) unchanged
```

The non-ASCII branch is byte-for-byte today's existing code — zero
behavior change for multi-byte/NUL-bearing/mixed batches; the fast path
only ever engages when *every* value in the batch is ASCII (per Design
decision 3).

**Measured** (1e6-row microbenchmark, isolated): ASCII workload 82.2ms →
39.4ms (**2.09x**); mixed non-ASCII workload 84.1ms → 81.4ms (falls to
unchanged path, ~3ms `isascii()`-scan overhead, matches expectation).

**Combined I1.a + I1.c** (full extend+flush pipeline, 1e7 rows,
Python-level cost only): 1557ms → 630ms (**-60%**). Projected onto the
full 3622ms end-to-end benchmark (NDArray-side cost assumed unaffected,
flush count unchanged): **≈2695ms** (-26%).

### I1.d — Raising `_FLUSH_ROWS` (folded into I1)

**Measured** (standalone sweep script, taxi-like ASCII workload, 1e7 rows,
full `CTable` ingest via `bench_string_kinds.py`'s own code path, `min` of
3 reps, `_FLUSH_ROWS` monkeypatched per value, `tracemalloc` peak as a
memory proxy):

| `_FLUSH_ROWS` | ingest   | peak (tracemalloc) |
|--------------:|---------:|--------------------:|
| 4096 (was)    | 3691.6 ms |  85.9 MB |
| 8192          | 2327.1 ms |  86.0 MB |
| 16384         | 1541.9 ms |  86.0 MB |
| 32768         | 1134.3 ms |  86.0 MB |
| **65536**     | **926.2 ms** |  86.2 MB |
| 131072        | 825.0 ms |  86.2 MB |
| 262144        | 806.0 ms |  91.0 MB |
| 524288        | 776.9 ms | 105.3 MB |

This confirms the hypothesis: resize-call overhead (one
`NDArray.resize()` + slice-write pair per flush) was a large, previously
unattributed chunk of the "~2065ms of inherent NDArray-side cost" the
original problem statement assumed was a hard floor — it was not. Gains
are monotonic but sharply diminishing past 65536 (each doubling beyond
that buys under 100ms), while peak memory starts climbing beyond 131072
(the `_FLUSH_CHARS` bound starts binding before `_FLUSH_ROWS` does, so
larger `_FLUSH_ROWS` stops mattering and only adds idle buffer capacity).
**Picked `_FLUSH_ROWS = 65536`**: captures nearly all of the available win
(926ms vs. the 777ms floor) with peak memory indistinguishable from the
original 4096 baseline. Landed as a one-line constant change plus an
explanatory code comment (no new code paths, no test changes needed since
existing tests reference `_FLUSH_ROWS` symbolically, not as a literal).

### I1 edge cases requiring dedicated tests

- Empty `values`/no-op flush (confirm `_rewrite_from` still only ever
  called non-empty; keep defensive `if values and all(...)` guard).
- Mixed valid/invalid batch straddling a chunk boundary (a `None` at
  index 4095 vs 4097).
- `append()` calls interleaved with `extend()` before a flush.
- An all-ASCII chunk containing a NUL byte (`"nul\x00in"` is ASCII — NUL
  is code point 0) — confirm the join+encode fast path preserves it
  exactly (this codebase has a known prior NumPy StringDType NUL bug on
  the *read* side, so be explicit here on the *write* side too).
- Non-ASCII batch (existing `SAMPLE` fixture) — must produce
  byte-identical output to before; extend the existing differential-test
  style (`test_utf8_array_bulk_read_matches_python_ground_truth`) to also
  cover writes.
- Design-decision-4 regression: `extend()` with >3x `_FLUSH_ROWS` rows in
  one call, assert `len(arr) == n` and full content equality.
- `set_all()` round-trip (`sort_by(inplace=True)`, `compact()`) still
  correct, since it shares the changed `_rewrite_from`.
- Non-str rejection and null-sentinel substitution
  (`test_utf8_array_rejects_non_str`,
  `test_ctable_utf8_not_nullable_rejects_none`,
  `test_ctable_utf8_explicit_null_value`) — exact same `TypeError`
  messages, now raised from the chunk's slow-path fallback.
- `test_ctable_utf8_extend_numpy_fixed_width_input` unchanged.

**Benchmark gate (record actuals in this file when landing):**
`bench_string_kinds.py` taxi ingest **≤ 2800 ms** (from 3622 ms;
prototype projects ≈2695 ms, possibly better after I1.d's sweep). No
regression elsewhere in that script, or in `sort_by`/groupby/`to_arrow`
(which reuse `_rewrite_from` via `set_all`/`copy`).

**ACTUAL (2026-07-17, `enhancing-ctable3`, Apple-silicon Mac, I1.a + I1.c +
I1.d combined):** taxi ingest **870.5 ms** (from 3622 ms, **-76%**) —
gate passed with a huge margin, and utf8 ingest is now *faster* than both
`string()` (1011.1 ms) and `vlstring()` (1106.1 ms) on this workload. No
regression in the rest of `bench_string_kinds.py`'s output (full read,
filter, groupby, sort, `to_arrow`, both workloads, all three column
kinds) beyond ordinary run-to-run noise. Full `tests/` suite: 7513
passed, 22 skipped (unchanged from pre-change baseline).

---

## I2 — C-level bulk UTF-8 encode kernel (needs rebuild)

**Where:** a second function in the existing `src/blosc2/utf8_ext.pyx`,
alongside `pack_utf8_span`. No `CMakeLists.txt` changes needed (Design
decision 6).

**Signature and algorithm:**

```cython
def encode_utf8_span(list values not None):
    """Return (data, lengths) for *values* (a list of str).

    data: uint8 NDArray -- concatenated UTF-8 encoding of every value.
    lengths: int64 NDArray, length len(values) -- each value's UTF-8 byte length.
    """
```

1. `n = len(values)`; `n == 0` → return two empty arrays (mirror
   `pack_utf8_span`'s early return).
2. Allocate `lengths` (`int64`, size `n`) up front — write directly into
   it during pass 1.
3. Allocate a temporary C buffer of `n` `const char*` pointers
   (`malloc`/`free` in a `try/finally`, mirroring `pack_utf8_span`'s
   existing `NpyString_acquire_allocator`/`try/finally` pattern) to
   remember each value's cached UTF-8 pointer between passes.
4. **Pass 1** (GIL held): for each value, call
   `PyUnicode_AsUTF8AndSize(value, &size)` — this both encodes (or reuses
   the string's cached UTF-8 representation) and hands back a *borrowed*
   pointer with no new `bytes` allocation. Store pointer + `size`; a
   `TypeError`/`UnicodeEncodeError` propagates automatically via the
   `except NULL` clause (Design decision 7) — ensure the temp buffer is
   still freed via `try/finally` on that path. Accumulate `total`.
5. Allocate `data` (`uint8`, size `max(total, 1)`, matching
   `_rewrite_from`'s existing zero-length convention).
6. **Pass 2** (GIL held — see Design decision 7 on why not `nogil`):
   `memcpy` each value's bytes from its stored pointer into `data` at the
   running cumulative offset.
7. Return `(data, lengths)`.

**Caller integration (`_rewrite_from`):**

```python
kernel = _encode_utf8_kernel()
if kernel is not None and values:
    data_arr, lengths = kernel(values)
elif values and all(v.isascii() for v in values):
    data_arr = np.frombuffer("".join(values).encode("ascii"), dtype=np.uint8)
    lengths = np.fromiter(map(len, values), dtype=np.int64, count=len(values))
else:
    encoded = [v.encode("utf-8") for v in values]
    data_arr = np.frombuffer(b"".join(encoded), dtype=np.uint8)
    lengths = np.fromiter((len(e) for e in encoded), dtype=np.int64, count=len(encoded))
```

**Graceful degradation:** new lazy helper `_encode_utf8_kernel()` in
`utf8_array.py`, mirroring `_pack_utf8_kernel()` exactly (`try: from
blosc2 import utf8_ext / except ImportError: return None`). Falls back to
I1.c's already-optimized path, not the pre-I1 naive path. Test via a
`force_write_kernel_mode` fixture, sibling to the existing
`force_kernel_mode` (they toggle independent lazy helpers — read-side vs
write-side kernel).

**I2 test coverage (in addition to I1's edge cases, all still apply):**

- Kernel-on/kernel-off parametrized versions of I1.c's edge cases (empty,
  NUL-bearing, multi-KB, multi-byte, mixed batches).
- A lone-surrogate value (e.g. `"\udc80"`) — assert `UnicodeEncodeError`
  raised, matching `str.encode("utf-8")`'s own behavior; assert no
  leak/corruption by checking a subsequent `extend()`/read still works
  (regression test for the `try/finally` temp-buffer cleanup on the error
  path).
- A very large single string (multi-MB) through the kernel — sanity-check
  `total`/offset accumulation.

**Benchmark gate (provisional — no prototype backing this number, unlike
I1's; record the actual on landing):** `bench_string_kinds.py` taxi
ingest **≤ 1500 ms** (from ≈2695 ms after I1). If missed, profile before
adding cleverness — ~2065 ms of the original 3622 ms is inherent NDArray
write/compression cost outside I1/I2's scope, so there is a hard floor.

**ACTUAL (2026-07-17, `enhancing-ctable3`, Apple-silicon Mac):** taxi
ingest **598.6 ms** (from 870.5 ms after I1 alone, **-31%** further;
**-83.5%** from the original 3622 ms) — gate passed with a large margin,
now within ~22% of `string()` (490.4 ms) on this workload. No regression
elsewhere in `bench_string_kinds.py` (full read, filter, groupby, sort,
`to_arrow`, both workloads, all three column kinds) beyond ordinary
run-to-run noise.

**Acceptance:** full `tests/` green with the kernel active (7525 passed,
22 skipped) AND with the compiled extension entirely absent from the
environment (`utf8_ext.cpython-*.so` moved aside — simulates a
NumPy-only/no-toolchain install; same 113/113 in `test_utf8.py` passed,
falling all the way back to I1's pure-Python path with no import error at
`import blosc2` time). The per-test `force_write_kernel_mode` fixture
additionally parametrizes every new I2 test over kernel-on/kernel-off
without needing the extension physically absent.

**Build note:** built via `cmake --build build_py314 --target utf8_ext`
(clean build, no new warnings — `CMakeLists.txt` needed zero changes, as
predicted by design decision 6, since its custom command already
regenerates `utf8_ext.c` from the whole `.pyx` file) and the resulting
`.so` copied into the active env's `site-packages/blosc2/`, exactly
mirroring U2's rebuild workflow.

---

## Honest assessment / what this plan does NOT close

**Superseded by measurement.** The projections below (written before I1
landed) assumed I1.d was a minor, uncertain lever and that ~2065ms of
NDArray-side cost was an inherent floor. Both assumptions were wrong: I1
(including I1.d) landed at **870.5ms**, already faster than `string()`
(1011.1ms) and `vlstring()` (1106.1ms) on the taxi benchmark — beating
I2's own provisional gate (≤1500ms) before I2 was even started. Original
text, kept for the record:

- ~~I1 alone lands at ≈2695ms — still 5.6x slower than `string()` (478ms)
  and 2.3x slower than `vlstring()` (1193ms). Real, verified, low-risk
  win; not a full fix.~~
- ~~I2 is provisionally estimated at ≈1500ms — still ~3x slower than
  `string()` and roughly on par with `vlstring()`. The remaining
  ~2065ms of NDArray resize/write/compression cost is untouched by either
  item as scoped.~~ — this cost was in fact mostly per-flush resize
  overhead, not inherent, and I1.d's larger `_FLUSH_ROWS` eliminated most
  of it directly.
- I1.d (the `_FLUSH_ROWS` sweep) is the cheapest way to find out whether
  that follow-up is even necessary before investing in it. (It was the
  right call: the sweep alone found nearly all of the remaining win.)

**Update after I2 landed:** the recommendation above (re-evaluate before
building I2, since the realistic ceiling looked like only ~150ms) turned
out to be pessimistic — I2 delivered another **272ms** (870.5ms →
598.6ms, -31%), more than the ~150ms estimated from the I1.d
`_FLUSH_ROWS` sweep's asymptote. The compiled kernel avoids not just the
per-row `.encode()` calls but also the intermediate `list` of `bytes`
objects and the `b"".join()` allocation that I1.c's Python fast path
still pays; those turned out to matter more than the sweep alone
suggested. Final state: utf8 ingest 3622ms → 598.6ms (**-83.5%**),
within ~22% of `string()` (490.4ms) and clearly ahead of `vlstring()`
(1292.2ms). Remaining gap to `string()` (~108ms) is presumably the last
of the inherent NDArray resize/write/compression cost referenced above —
not investigated further; not gated by this plan.

---

## Sequencing and cross-cutting rules

- Land order: **I1 (I1.a + I1.c + I1.d sweep) → I2**. I1 lands as one
  item (unlike U1.a/U1.b's split — no strong reason to separate I1.a/I1.c
  here).
- Each item lands with its gate recorded in this file (numbers, machine,
  command), following the read-side plan's convention: if a gate fails,
  do not merge the fast path — record the number and the root cause here
  and stop.
- Never regress `string()`/`vlstring()` ingest performance — guard with
  the full `bench_string_kinds.py` script, not just utf8's rows.
- No new public API — everything here is internal (`Utf8Array` methods,
  one new lazy-import helper, one new compiled function in the existing
  `utf8_ext` module).

---

## Critical files

- `src/blosc2/utf8_array.py` — `Utf8Array.extend`, `_rewrite_from`, new
  `_encode_utf8_kernel()` helper (I1.a, I1.c, I2 caller-side wiring).
- `src/blosc2/utf8_ext.pyx` — new `encode_utf8_span` function alongside
  the existing `pack_utf8_span` (I2 kernel).
- `tests/ctable/test_utf8.py` — new/extended tests per the edge-case
  lists above; mirror the existing `force_kernel_mode` fixture for a
  `force_write_kernel_mode` sibling.
- `bench/ctable/bench_string_kinds.py` — gate measurement (ingest step)
  for both I1 and I2; no code changes needed, just run it.
- `plans/utf8-reads-filter-optim.md` — style/convention reference and
  U2's build-workflow section (rebuild commands for I2).

## Verification (when this plan is implemented)

1. `conda run -n blosc2 python -m pytest tests/ctable/test_utf8.py -q`
   green after I1, then again after I2 (both with kernel active and with
   `force_kernel_mode`/`force_write_kernel_mode` forcing fallback).
2. `conda run -n blosc2 python -m pytest tests/ -q --timeout=600` full
   suite green, kernel-on and kernel-off.
3. `conda run -n blosc2 python bench/ctable/bench_string_kinds.py` —
   record the taxi `ingest` line after I1 (gate ≤2800ms) and after I2
   (gate ≤1500ms, provisional), plus confirm no regression in any other
   row of that script's output (full read, filter, groupby, sort,
   to_arrow — for all three column kinds).
4. Ruff check on changed `.py` files
   (`conda run -n blosc2 ruff check src/blosc2/utf8_array.py
   tests/ctable/test_utf8.py`).
5. For I2 specifically: build via `cmake --build build_py314 --target
   utf8_ext`, copy the `.so` into the active conda env's
   `site-packages/blosc2/`, and confirm `from blosc2 import utf8_ext;
   utf8_ext.encode_utf8_span(...)` round-trips correctly against a
   Python-level reference implementation before wiring it into
   `_rewrite_from`.
