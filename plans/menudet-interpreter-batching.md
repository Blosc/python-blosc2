# Menudet interpreter batching for `where` expressions

## Problem

With JIT disabled, expanded loop expressions in `bench/ndarray/dsl-kernel-bench.py`
were 12–202 ms while the equivalent authored Menudet loops were faster and the
transcendental cases stayed near 1 ms. Profiling the compiled/native interpreter
showed why: the vector interpreter admitted only straight-line *total* numeric
operations. A `where(...)` node was not admitted, so any expression containing a
selection fell back to per-element scalar dispatch for the whole tree.

Expanded loop expressions are built almost entirely from `where`, so the fallback
discarded batching for all of their arithmetic. Transcendental expressions have no
selection, so they kept the batched path — hence the uneven slowdown.

## Change

`p_batch_build`/`p_batch_eval` in `src/dsl_menudet_expr.c` now admit a lazy
floating `where`:

- A selection requires floating output and a `bool` condition, with both branches
  matching the node dtype.
- The condition becomes an explicit mask instruction (`P_BATCH_MASK`), and each
  branch is built under a guard derived from that mask (`condition` for the true
  branch, its complement for the false branch). Nested selections compose guards.
- Guarded operations execute only on contiguous runs of participating lanes.
  Branches are never blended eagerly, so an inactive branch performs no work and
  raises no IEEE flags — the scalar interpreter also evaluates only the selected
  branch.
- Masks participate in common-subexpression identity, so a value computed under
  one participation domain is never reused in another.
- Guarded SIMD math with lane padding computes into a scratch tile before copying
  the participating prefix, instead of writing padded lanes over neighbours.
- Uniform conversions no longer persist across vector tiles when guarded, because
  a guarded slot may be uninitialized in earlier tiles.
- Power, math, comparison, integer and complex paths keep their existing
  admission rules; nothing new is allowed to be non-total or error-raising.

A latent interpreter bug was fixed alongside it: unary nodes (negation and
one-argument math) with a uniform operand previously entered the scalar/vector
left-operand branch, which has no unary case. They now take the unary path, which
matters for guarded `where` branches such as `sqrt(-1.0)`.

Tile lanes are additionally bounded so that the per-operation workspace stays
within about 128 KiB when a program grows towards its new 128-node limit.

## Measurements

Apple arm64, 12 threads, 1000 × 1000 float32 inputs, BLOSCLZ level 1, matched
chunk/block storage, warm medians of three samples including allocation and
compression. Both runs use the same Python sources; only the native extension
differs, so the delta isolates this change.

| Case | LazyExpr before | LazyExpr after | Menudet before | Menudet after |
|---|---:|---:|---:|---:|
| loop1 | 12.29 | 1.04 | 19.79 | 8.96 |
| loop2 | 27.47 | 2.45 | 38.98 | 18.11 |
| loop4 | 61.21 | 3.92 | 76.75 | 36.04 |
| loop4_heavy | 201.61 | 7.68 | 74.82 | 36.84 |
| nested2 | 61.61 | 4.50 | 106.38 | 50.12 |
| transcend1 | 0.97 | 1.06 | 1.07 | 1.42 |
| transcend2 | 1.11 | 1.88 | 1.19 | 2.24 |
| transcend3 | 0.83 | 1.35 | 0.96 | 1.19 |
| transcend4 | 2.57 | 2.26 | 2.05 | 2.85 |
| sincos_id | 1.84 | 1.81 | 2.16 | 1.82 |

LazyExpr loop cases improve by roughly 12–26×; authored Menudet loop cases improve
2.0–2.2× because only their arithmetic subtree was affected. Transcendental cases
are unchanged within run-to-run noise. JIT results are also unchanged
(`interpreter-batching-jit-cc.json`), since compilation never used this path.

Raw: `interpreter-batching-before.json`, `interpreter-batching-after.json`,
`interpreter-batching-jit-cc.json`.

## Remaining interpreter cost

After this change the residual no-JIT gap is interpreter control flow, not
batching. On 1M elements with resident storage:

| Program shape | Time |
|---|---:|
| `tmp = where(...); return tmp * 1` | 1.36 ms |
| same statements inside `for i in range(1)` | 2.68 ms |
| same statements inside a constant `if`/`else` | 7.97 ms |

A constant `if` costs about 6.6 ms beyond the straight-line program even though
its mask is fully active. That branch/mask bookkeeping is a separate optimization
and is not addressed here; JIT remains the recommended execution mode for loop
kernels.

## Validation

- Native `menudet_jit_parity_tcc` and `menudet_jit_parity_cc_gate` pass.
- The batch parity matrix gained `where` recipes covering nested selections,
  float32 narrowing, division, transcendental branches, and a lazy-branch case
  whose inactive branch would raise `FE_INVALID` if evaluated.
- A new `batch_guard_participation` test compares batched interpolation against
  per-lane reference evaluation for all-true, all-false, single-boundary and
  alternating masks, with signaling-NaN and infinite operands, and checks that
  the union of IEEE flags matches exactly and that the batch path is actually used.
  The all-true case asserts that no flag is raised at all.
- Full native suite: 384 tests pass (`ctest -R 'menudet|batch|numpy_compat|artifact'`).
- Python Menudet/native-graph/resident/storage/LazyExpr/benchmark suite: 1,258
  passed, 34 skipped, 584 deselected.
- The resident evaluation path also improved rather than regressed: a single
  `where` over 1M resident elements went from 67.7 ms to 11.6 ms.

The shared editable installation was not replaced; rebuilt extensions were tested
from an isolated package. Native `plans/windows-complex-support.md` remains an
unrelated untracked file.
