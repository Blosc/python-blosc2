# LazyExpr configured-backend dispatch qualification

## Finding

The 100–1,500 ms expanded-loop timings were not an inherent cost of LazyExpr
authoring. Two admission restrictions kept these cases off the existing fast path:

1. `compute_native_prefilter_if_eligible()` rejected every explicit `jit_backend`.
   The benchmark always selects either TCC or CC, even without `--compiler`.
2. The adapter admitted only 1-D inputs, although the shared native prefilter
   already supports matching N-D full-width row slabs.

Execution then reached `slices_eval()` → `evaluate_chunk()`. That chunk evaluator
deliberately disables graph JIT when a backend is specified, since graph
preparation cannot honor task-local compiler options. Thus the expanded numeric
expression ran through the **serial native scalar interpreter**, not NumPy or
Python arithmetic, over a decompressed million-element chunk. The before profile
spends about 11.6 seconds in `evaluate_chunk` across the full validation/timing run.

## Change

For matching compressed N-D inputs, retain the existing metadata-only graph
syntax, dtype, cardinality, partitionability, and storage-layout validation.
Allow the shared prefilter to decide which N-D layouts it can handle safely.
With explicit compiler policy, prepare the graph without JIT, then import its
unchanged map artifact through the option-aware Menudet compiler. Cache this
kernel by graph, JIT policy, full execution options, and compilation environment.

No expression rewriting, changed weak-scalar rules, or accumulator widening is
introduced: the graph's NumPy semantic profile and capture payloads are preserved.
Authoring remains `blosc2.lazyexpr(...)`. Native graph rejection remains
authoritative. Nondefault FP-accuracy requests, reductions, broadcasting/mixed
layouts and unsupported storage retain prior fallbacks. Configured requests do
not enter the unconfigured stored-output fallback. Execution failures propagate
without replay. Reports use the selected kernel's actual JIT status.

## Measurements

Same arm64 host, 12 threads, 1,000 × 1,000 float32 inputs, BLOSCLZ level 1,
matched chunk/block storage, best of three warm allocation/compression samples.
CC before/after runs were profiled equally. The before control restores only the
former explicit-backend prefilter rejection. It reproduces the original timings
without modifying the arithmetic or selecting NumPy as a substitute engine.

| LazyExpr case | Before CC (ms) | After CC (ms) | After TCC (ms) | No JIT (ms) |
|---|---:|---:|---:|---:|
| loop1 | 103.30 | 0.89 | 0.97 | 12.33 |
| loop2 | 220.31 | 1.05 | 1.70 | 27.10 |
| loop4 | 496.18 | 1.15 | 2.65 | 62.07 |
| loop4_heavy | 1531.96 | 2.28 | 8.84 | 200.58 |
| nested2 | 489.11 | 1.38 | 2.80 | 63.65 |

CC LazyExpr and Menudet now have comparable warm times; the enormous previous
ratios were primarily dispatch artifacts. TCC retains extra cost in the expanded
heavy expression, where the authored Menudet loop expresses shared temporaries.
No-JIT is still native interpretation, but now uses compression-worker scheduling.
These are host samples, not portable speed guarantees or cold compilation costs.

Raw results: `lazyexpr-prefilter-before-cc.json`, `lazyexpr-prefilter-after-cc.json`,
`lazyexpr-prefilter-after-tcc.json`, `lazyexpr-prefilter-after-nojit.json`.
Profiles: `lazyexpr-prefilter-before-profile.txt`, `lazyexpr-prefilter-after-profile.txt`.
Historical benchmark and notebook measurements are untouched.

## Reproduction and tests

Use rebuilt native dependencies from miniexpr `3eae28d` and the `blosc2` conda
environment. These dispatch changes are pure Python and need no additional native
rebuild over the loop qualification build.

```sh
python bench/ndarray/profile-lazyexpr-prefilter.py --fallback-control --output before.json --profile before.txt
python bench/ndarray/profile-lazyexpr-prefilter.py --output after.json --profile after.txt
python bench/ndarray/dsl-kernel-bench.py --output tcc.json
python bench/ndarray/dsl-kernel-bench.py --no-jit --output nojit.json
```

Broad isolated-package tests: 1,243 passed, 34 skipped, 584 deselected. Added
binding/dispatch tests cover TCC/CC selection, explicit no-JIT, N-D edge rows,
compiler-option cache separation and reuse, fresh reads after operand mutation,
and FP-status parity with graph interpretation. The final storage subset including
the FP-status regression passes all 57 tests. Numerical matrices remain native.
The shared editable installation was not replaced.

Final independent-wheel qualification (local native-source overrides, extracted
wheel package and extension paths verified): **1,244 passed, 34 skipped,
584 deselected**. Ruff and `git diff --check` pass. CC, TCC, and explicit no-JIT
benchmark runs all pass their NumPy-reference correctness checks.
