# utf8 columns: closing the read/filter gap (incremental plan)

**Status:** U1 (U1.a + U1.b) and U2 LANDED 2026-07-17; U3 PARKED. Successor work to
`plans/enhancing-ctable-phase3.md` (P3 + its P3.d follow-up + post-review
fixes, all landed on branch `enhancing-ctable3`, commits `0cd1ca78`,
`77bf321d`, `564180b1`). Read that plan's P3 sections first if you need
background on what a utf8 column *is*; this plan assumes it.

**Audience:** an implementing model/developer who has NOT read the
discussions that produced this plan. Everything needed is in this file.
When in doubt, prefer the laziest change that satisfies the acceptance
criteria — do not add abstractions, options, or protocols this plan does
not ask for.

**Practical notes (carried over from phase 3, still true, verified
2026-07-17):**

- Locate code by **symbol name with grep**, never by line number.
- Run Python/pytest through the `blosc2` conda env:
  `conda run -n blosc2 python -m pytest ...` (or
  `/Users/faltet/miniforge3/envs/blosc2/bin/python`). Never use the repo
  `.venv` (stale).
- Editing `.pyx` files does NOT trigger a rebuild in the editable install.
  This is why item U1 below must be pure NumPy and why item U2 (the one
  C-level item) carries an explicit build-workflow warning.
- **Docstrings and code comments must be self-contained.** Never reference
  this plan, phase labels ("U1", "P3"), or benchmark history from source,
  tests, or bench scripts — state the semantics directly. Explicit
  maintainer rule.
- utf8 tests live in `tests/ctable/test_utf8.py`; match neighboring style.
- The dev machine is an Apple-silicon Mac; benchmark targets assume it.
- The benchmark harness for this plan is
  `bench/ctable/bench_string_kinds.py` (real Chicago-taxi `company`
  column at 1e7 rows + synthetic high-cardinality free text at 2e6 rows).
  All "measured" numbers below come from it or from prototypes run against
  the same 1e7-row taxi column on 2026-07-16/17.

---

## The problem, quantified

Every utf8 bulk read funnels through `Utf8Array._read_persisted_span`
(`src/blosc2/utf8_array.py`), which ends in a per-row Python loop:

```python
out = np.empty(n, dtype=StringDType())
for i in range(n):
    out[i] = blob[rel[i] : rel[i + 1]].decode("utf-8")
```

Each iteration pays two int64→int conversions, a `bytes` slice
allocation, a `str.decode` allocation, and a StringDType `__setitem__`
(arena copy) — ~200–230 ns/row. Fetching the offsets and byte-blob slices
for the whole 1e7-row taxi column costs only ~64 ms; the loop costs
~2,300 ms. **~97% of a utf8 full read is this loop, not I/O.**

Measured on the taxi `company` column (1e7 rows, ~60 distinct values,
≤44 chars; `bench_string_kinds.py`):

| operation              | utf8() today | string(44) | why utf8 is slower        |
|------------------------|--------------|------------|---------------------------|
| full column read       | 2368 ms      | 293 ms     | the decode loop           |
| filter `s == value`    | 1819–2035 ms | 75 ms      | decode loop + numexpr gap |
| sort_by (copy)         | 7712 ms      | 2395 ms    | decode loop feeds sort keys |

groupby keys and dense-table Arrow export were already fixed (phase 3's
P3.d follow-up: `Utf8Factorizer`; post-review fixes: `arrow_slice`) —
both bypass the decode loop. Filters, full reads, sort-key
materialization, and column-vs-column comparisons still pay it.

**Root constraint:** NumPy provides no bulk constructor for a
`StringDType` array from an offsets+bytes buffer pair. StringDType
manages a private arena with per-element small-string optimization; the
only *Python-level* ways in are element assignment or conversion from
another array, both element-at-a-time. So the strategy is: (U1) stop
materializing StringDType where it isn't needed at all — filters; (U2)
build the missing bulk constructor at C level for the cases where a
StringDType array genuinely must exist; (U3, parked) push string
predicates into miniexpr only if a real workload later proves it pays.

---

## Design decisions already made (do not re-litigate)

1. **Filters do NOT go through `Utf8Factorizer`.** This was prototyped
   and measured on 2026-07-17 (1e7-row taxi column, most-frequent-value
   probe, 2.34e6 hits), all results verified equal to ground truth:

   | path                                   | time     | speedup |
   |----------------------------------------|----------|---------|
   | current (`_utf8_chunked_bool` + decode)| 2035 ms  | —       |
   | factorizer-based (fresh vocabulary)    | 764 ms   | 2.7x    |
   | **byte-level compare (no decode)**     | **156 ms** | **13x** |
   | fixed-width `string()` reference       | 75 ms    | —       |

   The factorizer pays for building a vocabulary (hash + verify + sort of
   new values) that a filter uses exactly once, and it *regresses below
   the current path* on high-cardinality columns where nearly every row
   is a new value. Filters need one boolean per row, not value
   identities. **A note in `plans/enhancing-ctable-phase3.md` ("route
   comparisons through the `Utf8Factorizer`…") predates this measurement
   and is superseded by this plan — leave that file as is, it is a
   historical record.**

2. **U1 is pure NumPy.** No Cython/C for filters: the byte-level path is
   already fully vectorized (whole-array ops only), a C version would buy
   maybe 2x more while adding build friction, and the repo's editable
   install does not rebuild `.pyx`. Precedent:
   `_factorize_fixed_width_str` and `Utf8Factorizer` solve the same class
   of problem the same way.

3. **Byte-lexicographic order on UTF-8 bytes equals Unicode code-point
   order.** This is a designed property of UTF-8 encoding and it is what
   makes ordering comparisons (`<`, `<=`, `>`, `>=`) implementable on raw
   bytes without decoding. Python `str` comparison is code-point order,
   so byte-lex results match Python/StringDType semantics exactly. (The
   same property already justifies `Utf8Factorizer`'s rank codes.)

4. **Null semantics are frozen.** A null (sentinel) value on either side
   never satisfies any comparison — SQL `WHERE` semantics, pinned by
   existing tests in `tests/ctable/test_utf8.py` ("Comparisons and
   filtering" section). Every U1 path must reproduce this bit-for-bit,
   including the corner where the probe *equals* the sentinel (the result
   must then be all-False for `==`, because every matching row is by
   definition null).

5. **Existing observable behavior is the contract.** All current
   comparison tests must pass unchanged. The result stays a
   physical-length boolean `blosc2.NDArray` already intersected with the
   live-row mask, exactly as `Column._utf8_compare` returns today.

---

## U1 — Pure-NumPy byte-level scalar comparisons

### U1.a Equality (`==`, `!=`) against a `str` scalar

**Where:** a new method on `Utf8Array` (`src/blosc2/utf8_array.py`), plus
wiring in `Column._utf8_compare` (`src/blosc2/ctable.py`, grep for
`def _utf8_compare`).

**Algorithm** (this is the measured 156 ms prototype, reproduce it
faithfully):

```python
def equal_mask_span(self, value: str, a: int, b: int) -> np.ndarray:
    """Boolean mask for rows [a, b): row bytes == value's UTF-8 bytes."""
    enc = value.encode("utf-8")
    length = len(enc)
    target = np.frombuffer(enc, dtype=np.uint8)
    offs = np.asarray(self._offsets[a : b + 1], dtype=np.int64)
    rel = offs - int(offs[0])
    data = np.asarray(self._data[int(offs[0]) : int(offs[-1])])
    lengths = np.diff(rel)
    mask = lengths == length  # length must match first
    if length and mask.any():
        idx = rel[np.flatnonzero(mask)]  # start offset of each candidate
        hit = np.ones(len(idx), dtype=np.bool_)
        for i in range(length):  # `length` whole-array compares
            hit &= data[idx] == target[i]
            idx = idx + 1
        cand = np.flatnonzero(mask)
        mask[cand[~hit]] = False
    return mask
```

Key properties to preserve:

- The inner loop runs `len(probe_bytes)` times (e.g. 25 for a 25-char
  ASCII probe), each iteration a C-speed whole-array gather+compare over
  the *candidates only* — never a per-row Python loop.
- `length == 0` (empty-string probe) is handled by the `lengths == 0`
  mask alone.
- Comparison is on raw bytes, so NUL-bearing and multi-byte values are
  exact by construction (no NumPy `S`-dtype trailing-NUL truncation, no
  decode).
- The candidate index vector is *reused and incremented in place*
  (`idx = idx + 1` creates one new array per byte position; that is
  fine — the point is never materializing a `(k, L)` int64 index matrix).
- **Pending rows:** call `self.flush()` at the start of the public entry
  point (precedent: `Utf8Factorizer.__init__` and `factorize_span` flush;
  it is a no-op unless there are buffered rows, and read-only tables
  cannot have any).

**Wiring in `Column._utf8_compare`:** the scalar branch currently builds
the predicate via `self._utf8_chunked_bool(fn)` where `fn` compares a
materialized StringDType chunk. Replace only the *scalar-operand* case:

- `numpy_op is np.equal` → chunked `equal_mask_span`.
- `numpy_op is np.not_equal` → complement of the equality mask **within
  the logical length** (rows past the logical length must stay False in
  the physical-length result — mirror how `_utf8_chunked_bool` zero-fills
  beyond `n_logical`), then null exclusion as below.
- Null fusing: the sentinel-null mask is just `equal_mask_span(nv, ...)`
  over the same span — compute both masks per chunk and combine
  (`eq & ~null` / `ne & ~null`), keeping the current single-pass shape.
  When the probe equals the sentinel the two masks are identical and the
  fused result is all-False — that is the required behavior, add a test
  for it (one probably exists; extend it if it only covers `==`).
- Column-vs-Column operands keep the existing StringDType path untouched.
- The final assembly (`blosc2.asarray(raw) & self._lazy_valid_rows()`)
  stays exactly as is.

Also refactor `arrow_slice`'s inline sentinel-null matcher (grep
`nv_enc` in `utf8_array.py`) to call the shared helper — same algorithm,
currently duplicated. Behavior must not change (its tests pin it).

### U1.b Ordering (`<`, `<=`, `>`, `>=`) against a `str` scalar

**Algorithm:** per-byte vectorized lexicographic compare against the
probe's bytes, grouped by row byte-length (the same grouping loop
`Utf8Array.factorize_span` uses — bincount on `np.diff(rel)`, then one
iteration per distinct length; distinct lengths are few in practice and
each row is touched once regardless).

For a length group with row length `L` and probe byte length `P`,
`m = min(L, P)`:

```
undecided = all rows in group          # bytes equal so far
lt = gt = all-False
for i in 0 .. m-1:
    b = data[start_of_row + i]         # vectorized gather, index vector reused
    lt |= undecided & (b < target[i])
    gt |= undecided & (b > target[i])
    undecided &= (b == target[i])
# rows still undecided are byte-prefix-equal over m bytes:
#   L < P  → row is a strict prefix of probe → row < probe   (lt)
#   L > P  → probe is a strict prefix of row → row > probe   (gt)
#   L == P → row == probe                                    (eq)
```

Then per operator: `<` = lt; `<=` = lt | eq; `>` = gt; `>=` = gt | eq.
Null exclusion fuses exactly as in U1.a (`pred & ~null_mask`).

Edge cases that MUST have dedicated tests because they are where this
algorithm goes wrong if implemented sloppily:

- probe is a strict prefix of a value and vice versa ("Taxi" vs
  "Taxi Affiliation"), including at length-group boundaries;
- empty-string probe (everything except "" is `>` it; "" is `==`);
- empty-string rows vs non-empty probe;
- multi-byte values ordered across byte-length boundaries (e.g. "z" <
  "é" < "日" must hold: code points 0x7A < 0xE9 < 0x65E5, and their
  UTF-8 encodings byte-compare in the same order — assert against
  Python's own `<` on the str values, which is the ground truth);
- NUL-bearing values;
- probe equal to the sentinel on a nullable column (all four ordering
  ops must exclude null rows, and rows *equal* to the sentinel are the
  null rows).

**Testing strategy for U1 overall (do all of these):**

1. Keep every existing test in `tests/ctable/test_utf8.py` green,
   untouched.
2. Add a randomized differential test: build ~5,000 rows mixing ASCII /
   multi-byte / empty / NUL-bearing / multi-KB values plus the sentinel,
   then for each of the six operators and a set of probes (present value,
   absent value, prefix of a value, empty, sentinel) assert the CTable
   filter result row-for-row against the pure-Python ground truth
   computed with list comprehensions on the original values (NOT against
   `np.unique`/StringDType helpers — NumPy has a known bug collapsing
   StringDType values that differ only after an embedded NUL; ground
   truth is Python semantics).
3. Test through a view (`t.head(n)[pred]`) and on a table with deleted
   rows, since the mask is physical-length and intersected with the
   live-row mask.
4. Run the full suite: `tests/ctable` and then all of `tests/`.

**Benchmark gate (record actuals in this file when landing):** with
`bench_string_kinds.py` on the 1e7-row taxi column, filter
`s == <most frequent value>` must land **≤ 250 ms** (prototype: 156 ms;
current: ~1,900 ms). Ordering ops have no prototype; expect roughly 2–3x
the equality cost from the per-byte loop bound by the probe length —
gate at **≤ 700 ms**. If an implementation misses its gate, profile
before adding cleverness; the prototype numbers prove the budget exists.

**LANDED 2026-07-17** (Apple-silicon Mac, `bench_string_kinds.py` on the
1e7-row taxi `company` column, probe = most frequent value = "Taxi
Affiliation Services"):

| op                | time     | gate    |
|-------------------|----------|---------|
| `s == value`      | 162.3 ms | ≤250 ms |
| `s < value`       | 368.9 ms | ≤700 ms |
| `s <= value`      | 366.5 ms | ≤700 ms |
| `s > value`       | 366.9 ms | ≤700 ms |
| `s >= value`      | 367.8 ms | ≤700 ms |

Both gates pass. `!=` shares the equality code path (one extra `~`, no
measurable difference). Ordering ops were timed with a standalone script
(same taxi data, same probe) since `bench_string_kinds.py` only exercises
`==`; full `tests/` suite green (7496 passed) with the change, plus a new
randomized differential test and dedicated edge-case tests in
`tests/ctable/test_utf8.py`.

**Explicitly out of scope for U1:** Column-vs-Column comparisons (keep
the current StringDType path; U2 speeds them up for free),
`startswith`/`endswith`/`np.strings` LazyExpr ops (they materialize reads
— U2's territory), and string-syntax expressions
(`t.where("name == 'x'")` — still guarded, U3's territory).

---

## U2 — C-level bulk StringDType constructor (fixes full reads)

**What:** a small compiled kernel that builds a StringDType array
directly from the (offsets, bytes) pair, replacing
`_read_persisted_span`'s per-row loop. This is the one place where
per-element work is irreducible at the Python level, and it fixes in one
stroke everything U1 does not: full column reads, sort-key
materialization, Column-vs-Column comparisons, `np.strings` ops, repr
previews — every consumer of `_read_persisted_span`.

**How (sketch, verify against current NumPy docs before writing code):**
NumPy ≥ 2.0 exposes a C API for StringDType packing — grep the NumPy
headers/docs for `NpyString_pack`, `NpyString_acquire_allocator`,
`npy_string_allocator`. The kernel is essentially:

```
acquire allocator for the destination StringDType descriptor
for i in 0 .. n-1:                      # C loop, ~tens of ns/row
    NpyString_pack(allocator, &out[i], data + rel[i], rel[i+1] - rel[i])
release allocator
```

- Input buffers: the *relative* offsets (`int64`, length n+1, rel[0]==0)
  and the byte blob slice — exactly what `_read_persisted_span` already
  computes before its loop. The bytes were produced by `str.encode` on
  write, so they are valid UTF-8 by construction; the kernel packs bytes,
  it does not validate.
- Home: a new small `.pyx` (e.g. alongside the existing `indexing_ext` /
  `groupby_ext` sources — follow how those are registered in the CMake
  build; grep `indexing_ext` in `CMakeLists.txt`).
- **Build-workflow warning:** the editable dev install does NOT rebuild
  `.pyx` files. Developing this requires a real rebuild (see how the
  existing extensions are built; budget for that friction). This is the
  reason U2 is sequenced after U1 and not merged with it.
- **Graceful degradation is mandatory:** `_read_persisted_span` tries the
  kernel and falls back to the current Python loop when the extension is
  unavailable (WASM builds, source installs without the toolchain, NumPy
  API drift). The fallback path must remain tested — parametrize the
  existing roundtrip tests over kernel-on/kernel-off if feasible (e.g. a
  monkeypatch fixture forcing the fallback).

**Expected benefit:** decode cost drops from ~230 ns/row (Python loop) to
~20–40 ns/row (C loop + arena copy): taxi full read 2368 ms → roughly
200–400 ms. Sort and col-vs-col compares inherit proportionally.

**Benchmark gate:** `bench_string_kinds.py` taxi full-column read
**≤ 500 ms** (from 2368 ms), and no regression anywhere else in that
script. Record actuals here.

**Acceptance:** full `tests/` green with the kernel active AND with the
fallback forced; benchmark gate met; `blosc2.utf8()` still importable and
fully functional on a NumPy-only environment without the compiled
extension.

**LANDED 2026-07-17** (Apple-silicon Mac, `bench_string_kinds.py`):

- New `src/blosc2/utf8_ext.pyx` (`pack_utf8_span`): acquires a
  `StringDType` allocator via `NpyString_acquire_allocator` and calls
  `NpyString_pack` per row in a plain (GIL-held) C loop — Cython treats
  `NpyString_pack` as GIL-requiring, so `with nogil` isn't available here.
  Needs `NPY_TARGET_VERSION=NPY_2_0_API_VERSION` as a compile definition
  (CMakeLists.txt `target_compile_definitions`); without it NumPy's
  `numpy/*.h` headers gate the 2.0 string-C-API macros behind
  `NPY_FEATURE_VERSION`, which otherwise defaults to an older, source
  -compatible value and leaves `NpyString_pack` undeclared at compile time.
  Registered in `CMakeLists.txt` exactly like `indexing_ext`/`groupby_ext`
  (own `add_custom_command`, `Python_add_library`, link/install rules).
  Measured at ~9 ns/row standalone (2e6-row synthetic column), matching
  the plan's 20-40 ns/row estimate.
- `Utf8Array._read_persisted_span` (`utf8_array.py`) tries the kernel via
  a new lazy `_pack_utf8_kernel()` helper (mirrors the
  `try: from blosc2 import groupby_ext / except ImportError: return None`
  pattern already used in `groupby.py`) and falls back to the old per-row
  decode loop when it returns `None`. Tests force the fallback by
  monkeypatching `blosc2.utf8_array._pack_utf8_kernel`
  (`force_kernel_mode` fixture in `test_utf8.py`, parametrized
  kernel/fallback).
- **Two extra fixes were needed to actually hit the gate** — the kernel
  alone cut `_read_persisted_span` itself to ~170 ms for the 1e7-row
  span, but the *observed* `t["s"][:]` benchmark stayed at ~730-750 ms
  until both landed:
  1. `Column._values_from_key`'s slice fast-path (`ctable.py`) excluded
     every `is_varlen_scalar` column, including utf8, from the
     identity-position direct-slice shortcut, even though `Utf8Array`
     slices itself efficiently. Changed the exclusion to
     `is_varlen_scalar and not is_utf8`.
  2. The real bottleneck: `Utf8Array._get_many` (used whenever
     `_has_identity_positions()` is false — the common case, since a
     table's physical capacity is normally chunk-padded past its row
     count) always sorted the index array and did a fancy-indexed
     `out[order[...]] = span[...]` StringDType scatter-copy, even for a
     plain ascending contiguous range. That scatter-copy is itself a
     full per-element StringDType copy — exactly the cost U2 exists to
     eliminate — so it silently ate the kernel's gain. Added a
     contiguous-ascending-run check at the top of `_get_many` that
     shortcuts straight to `_read_span` when `indices` is
     `arange(indices[0], indices[-1] + 1)`, skipping the sort/cluster/
     scatter-copy machinery entirely.

| operation (taxi, 1e7 rows)       | before U2 | after U2 | gate    |
|-----------------------------------|-----------|----------|---------|
| full column read                  | 2472.6 ms | 165.3 ms | ≤500 ms |
| filter `s == value` (U1, unchanged)| —        | 168.6 ms | —       |
| groupby key: sum(val)             | 969.5 ms  | 971.3 ms | no regression |
| sort_by(s) (copy)                 | 7874.4 ms | 3930.4 ms| no regression (improved — sort keys now read via the fast path) |
| to_arrow()                        | 40301.2 ms| 39992.2 ms| no regression |

Synthetic free-text workload (2e6 rows): full column read 735.4 ms → 50.3 ms.

Full `tests/` suite green with the kernel active (7506 passed) and with
the fallback forced (7505 passed, 1 unrelated pre-existing flaky failure —
a subprocess segfault in `test_dsl_kernel_scalar_constant_subexpr_runtime_no_segfault`,
a DSL/JIT kernel test with no connection to utf8 code; passes standalone).
Both gates pass.

---

## U3 — miniexpr varlen-string support (PARKED — do not start)

**What it would be:** teach miniexpr (the C expression engine at
`~/blosc/miniexpr`, used by LazyExpr for fused chunk-at-a-time
evaluation) to evaluate predicates on utf8 columns natively, enabling
single-pass fused expressions like `(t.name == "Paris") & (t.fare > 10)`
with block pruning, string-syntax `t.where("name == 'Paris'")`, and
C-speed `startswith`/`contains` inside larger expressions.

**Why it is parked (facts verified 2026-07-17 against the miniexpr
sources):** miniexpr already has `ME_STRING`, but it is *fixed-width* — a
string variable carries a per-element `itemsize`, and the evaluator,
blocking/threading logic, and all three JIT backends (libtcc, cc,
wasm32) address element `i` as `base + i * itemsize`. Variable-length
support means:

1. a new dual-buffer variable kind (int64 offsets + byte blob) threaded
   through the evaluator and all three JIT code generators;
2. bridge plumbing on the blosc2 side that feeds *synchronized pairs* of
   chunks — and the utf8 data blob's chunk boundaries are byte-aligned,
   not row-aligned, so per-block operand preparation needs offset reads
   to even locate the byte range (this misalignment is the genuinely hard
   part, it has no counterpart in the current bridge);
3. varlen lex-compare kernels (the easy part).

Cross-repo, weeks-scale. Meanwhile the composability half of the fused
story already works without it: `Column._utf8_compare` returns a boolean
NDArray that combines with other predicates via `&`/`|`/`~`; what is
lost is only single-pass evaluation of the string leg.

**Unpark criteria (any one):**

- after U1 lands, profiling a real fused-query workload shows the string
  predicate leg dominating (say >50% of wall time of a representative
  mixed query), OR
- a product requirement lands for string-syntax `where()` / DSL kernels
  over utf8 columns.

**If unparked:** U1's byte-compare semantics (including the UTF-8
byte-order property and the null fusing rules) are the reference
semantics for the C kernels — nothing built in U1 is throwaway.

---

## Sequencing and cross-cutting rules

- Land order: **U1.a → U1.b → U2**. U3 stays parked. U1.a alone is
  already the highest value-for-effort item (13x measured, ~half a day).
- Each item lands separately with its gate recorded in this file
  (numbers, machine, command), following the phase-3 convention:
  if a gate fails, do not merge the fast path — record the number and
  the root cause here and stop.
- Never regress `string()`/`vlstring()` behavior or performance; the
  guard is `bench_string_kinds.py` plus the full test suite.
- No new public API: everything here is internal (`Utf8Array` methods,
  `Column._utf8_compare` internals, an optional compiled helper).
