# Menudet loop fast-path qualification

## Scope and safety

The five loop bodies in `bench/ndarray/dsl-kernel-bench.py` are unchanged:
float32 inputs, float64 intermediates, and a final float32 output conversion.
Native artifact metadata now proves independent-lane execution for bounded,
context-free floating programs, including safe final floating narrowing.
Unproved programs retain the previous compatibility checks. Graph metadata is
separate and authoritative: a rejected graph never gains eligibility through
the artifact proof or Python source fallback.

The native proof does not evaluate user data. It rejects reductions, context,
dynamic/zero-step ranges, counter mutation, while loops, early exits, checked
casts, and unproved weak arithmetic. Analysis has depth/work limits. Nonfinite
captures are classified by their bits without disturbing caller FP flags.
Small literal loops can be unrolled (at most 16 trips and a 256-trip budget).
Only exact bounded integer bookkeeping is folded; floating sequencing and
rounding/status barriers are retained. Unsupported loops use existing lowering.

## Measurements

Apple arm64, 1,000 × 1,000 elements, 12 threads, BLOSCLZ level 1, matched
chunk/block geometry. Times include warm output allocation and compression;
best of five samples. Preparation/first execution and all samples are retained
in `menudet-loops-before-tcc.json`, `menudet-loops-after-tcc.json`, and
`menudet-loops-after-cc.json`. These are host measurements, not portable promises.

| Case | Before TCC (ms) | After TCC (ms) | Legacy, TCC run (ms) | After CC (ms) | Legacy, CC run (ms) |
|---|---:|---:|---:|---:|---:|
| loop1 | 17.81 | 1.18 | 1.93 | 0.72 | 3.55 |
| loop2 | 29.49 | 1.82 | 3.10 | 1.40 | 4.57 |
| loop4 | 53.7 | 3.19 | 5.49 | 1.54 | 7.01 |
| loop4_heavy | 56.4 | 5.21 | 6.76 | 2.15 | 7.31 |
| nested2 | 75.5 | 3.83 | 6.49 | 1.55 | 7.53 |

All five Menudet cases report JIT and `menudet-prefilter`, with the native
partitionability proof true. The internal legacy controls report compiled=true,
jit=false, ME_COMPILE_SUCCESS: these compare against the actual legacy native
interpreter, **not a verified legacy JIT**. CC preparation was roughly 0.35–0.60 s
in this run; warm results must not be confused with compilation latency.

The cProfile snapshots in `menudet-loops-profiles.txt` show the initial loop1
execution spending about 17 ms in whole-group `evaluate_block`. Afterward it
uses the native compression prefilter (~1 ms total), without `_values` staging.
The proof enables worker scheduling; loop unrolling then removes repeated
integer-counter dispatch. No historical notebook measurements were replaced.

## Validation and reproduction

- Native `menudet_jit_parity_tcc` and `menudet_jit_parity_cc_gate` pass, including
  masked/nonfinite values, FP-status parity, negative ranges, counter mutation,
  nested loops, narrowing, proof rejection, and caller FP-state preservation.
- Broad Python Menudet/native-graph/resident/storage tests: 1,186 passed,
  34 skipped (584 deselected by the normal marker configuration).
- Benchmark/decorator/storage subset: 101 passed before adding the explicit
  authoritative-graph-rejection regression.
- Benchmark outputs are checked against the equivalent NumPy expressions.
- Independent wheel built with local native-source overrides and tested from
  its extracted package (both Python and extension paths verified): 1,190 passed,
  34 skipped, 584 deselected, including the graph-rejection regression and
  benchmark tests. No shared-install files were replaced.

Run `bench/ndarray/profile-menudet-loops.py --compiler tcc --output results.json`
(or `--compiler cc`) in the `blosc2` conda environment with rebuilt extensions.
The profiling harness is an internal control, not public legacy authoring.
The ordinary benchmark console output remains unchanged.

The new native API is optional: older headers return unavailable metadata and
retain the prior restrictions. CMake now pins the native loop implementation
at `3eae28dbddbf7866cd9315130109508c46ae3790`. Qualification used the local
`FETCHCONTENT_SOURCE_DIR_MINIEXPR` override and rebuilt extensions. The shared
editable installation was left untouched.

## Follow-up

`menudet-interpreter-batching.md` documents a later no-JIT improvement: the
vector interpreter now admits lazy `where` selections, so expressions containing
selections keep batching instead of falling back to per-element scalar dispatch.
Masks and guards preserve branch laziness and IEEE-flag parity.
