# Same-process LazyExpr versus Menudet/Clang

Both methods read the exact same live compressed operands and return newly
allocated NumPy arrays. Expression: `x + (y - 1)`, 8M float64 values, eight 1M
chunks per input, 15,625-value blocks, ZSTD level 5 plus shuffle, 12 threads.
Menudet confirmed prepared JIT and portable-prefilter dispatch. Input creation
and compilation are excluded. Five correctness-checked warmups per method,
then 100 timed reads each, interleaved in deterministically shuffled order.

| Same-process run | LazyExpr median ms | Menudet/Clang median ms |
|---|---:|---:|
| First | 20.749 | 20.855 |
| Repeat in a fresh process | 19.718 | 20.322 |

In the first run, the interquartile ranges were 18.522–24.194 ms (LazyExpr) and
18.794–24.919 ms (Menudet); in the repeat, 17.448–23.044 and 17.894–23.600 ms.
Menudet was 0.5% and 3.1% slower by median: effectively the same performance
range given the observed variability. The earlier cross-run estimate of a 15%
Menudet advantage is not supported by this same-process comparison.

Reproduce with `bench/menudet-lazyexpr-same-run.py --samples 100` using conda
environment `blosc2`, `ME_DSL_JIT_COMPILER=cc CC=clang`, and explicit selection of
the uninstrumented rebuilt isolated package. The shared editable extension and
production dispatch are unchanged by this benchmark.

## Resident NumPy inputs

The same interleaved protocol with `--inputs numpy` produced:

| Same-process run | LazyExpr median ms | Menudet/Clang median ms | Menudet speedup |
|---|---:|---:|---:|
| First | 21.333 | 2.840 | 7.51x |
| Repeat in a fresh process | 21.267 | 2.769 | 7.68x |

Interquartile ranges: 21.120–21.572 / 2.689–3.000 ms in the first run,
21.072–21.530 / 2.599–2.916 ms in the repeat (LazyExpr / Menudet).
Menudet reports the parallel-resident route with twelve workers and no input
staging copies. A separate backend probe confirms LazyExpr uses its current
portable-JIT graph path (one JIT stage, no interpreted stages); that path does not
use the new resident-array worker scheduler. These are LazyExpr reads over NumPy
buffers, **not** direct NumPy arithmetic timings.

## After LazyExpr scheduler integration

Eligible single-stage LazyExpr JIT graphs now borrow their prepared native map
artifact and share the resident scheduler. No artifact recompile is performed.
Same-process NumPy-input measurements using the same protocol:

| Same-process run | LazyExpr median ms | Menudet/Clang median ms |
|---|---:|---:|
| First | 5.462 | 2.761 |
| Repeat in a fresh process | 5.428 | 2.702 |

Both reports confirm twelve-worker `parallel-resident` execution, with no input
materialization or normalization. LazyExpr improves roughly 3.9x relative to the
earlier 21.3 ms serial-graph results, although the original graph artifact still
takes about twice as long as the independently authored Menudet artifact in this
test. This measurement alone does not identify the source of that remaining gap.
Compressed-input automatic eligibility and dispatch are unchanged.

The configured suite passed 13,232 tests (76 skipped, 12,605 deselected).
Final focused qualification passed 504 tests covering graph stages, weak
captures, native-first policy, scheduler fallback, mask participation, FP status,
no-replay errors and borrowed-owner lifetime. The instrumented Cython binding
ASan gate passed 434 tests; native static archives were not instrumented and
leak detection was disabled.
