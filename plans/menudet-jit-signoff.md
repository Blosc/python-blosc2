# Menudet portable host JIT — first slice

## Implemented boundary

- Explicit 1.1 `jit=True` / `ME_JIT_ON`; checked 1.0 and default routes unchanged.
- Typed-tree float32/float64 `+`, `-`, `*`, `/`, comparisons, Boolean short-circuit
  selection and lazy `where`; final float/Boolean conversions preserve inferred
  intermediate precision. Unsupported trees/signatures remain native interpreted.
- TCC and GCC use existing runtime/compiler/cache infrastructure. Private mask and
  comparison bridge ABI; semantic profile and lowering revision enter cache identity.
- Per-call fenv/status handling and participating masks; constant operations cannot
  silently fold away observable exceptions. WASM host-pointer lowering is disabled.
- Logical arrays retain bounded gathers, broadcasting and serial M5 reductions.
  Unmasked all-direct tiles skip unnecessary per-lane coordinate/gather work.
- Native-required LazyExpr execution can explicitly request JIT. Its bounded
  compiled-plan cache is artifact/configuration keyed and never caches values.

Tests uncovered speculative float32 widening of an inactive scalar-union member
in the interpreter: finite float64 low bits could spuriously raise invalid.
A non-inlined widening helper prevents this, with a targeted bit-pattern regression.
Comparisons retain a host bridge (ordinary finite/infinite operands fast-path;
NaN operands use the exact typed evaluator path). This is not a fully inlined
comparison implementation and remains an optimization opportunity.

## Local qualification

Host: macOS arm64 / Apple M4 Pro, NumPy 2.5.3, NumExpr 2.14.2, GCC 16 and bundled
TCC. The editable integration explicitly uses the sibling miniexpr checkout.
Dependency pins and packaging requirements are not changed.

`tests/test_menudet_jit.py` requires actual compiled kernels for supported cases:
56 TCC/GCC tests cover random/edge values, finite result bits, status, lazy branches,
masked reductions, varied tiles, reversed layouts, mixed float signatures,
constant-operation flags, concurrency, recovery and graph cache/mutation behavior.
Native `tests/numpy-compat/portable-jit.c` also checks standalone lazy/masked execution.
The authoritative corpus runner now reports actual compilation instead of rejecting
every compiled artifact as an impossible state.

Local validation: full Python suite **12,759 passed / 55 skipped**; full native
suite **427 passed**; **11** ASan/UBSan corpus/array checks passed, plus forced
GCC-JIT standalone sanitizer execution. The latter found a NaN bridge overread
from copying `sizeof(me_expr)` out of header-only leaf allocations; copying only
through `offsetof(me_expr, parameters)` fixes it. The post-fix TCC/GCC focused
suite again passes all 56 tests. Authoritative request-on corpus runs execute
181 arithmetic and 15 function cases through compiled kernels, while other cases
retain their explicit diagnostic/native-contract/interpreter classifications.
GCC request-on corpus checks pass separately. The arithmetic corpus timeout was
increased from 60 to 300 seconds to include cold external compilation of hundreds
of kernels; numerical and diagnostic assertions are unchanged.

Cross-platform CI, full native SIMD qualification, libm/integer/statement lowering
and release-wheel qualification remain pending. Existing CI failures from the prior
M1–M6 work are not represented as resolved by local host qualification.

## Benchmark method

`tools/menudet_jit_bench.py` produces `plans/menudet-jit-benchmarks.json`: 84 isolated
rows, sizes 8 and 262,144, float32/64, affine/polynomial/lazy-where workloads;
interpreter/TCC/GCC/NumPy/one-thread NumExpr. Every result is value-checked.
Compiled routes assert `has_jit`; each worker has a fresh cache directory.
Artifact import/compile, first execution and warm execution are recorded separately.
Warm medians use nine calls (three interpreter calls). Flat descriptor and logical
array timings are separate; these are finite C-contiguous workloads, not claims
about compressed traversal, production concurrency or universal NumPy performance.

Reproduce in the `blosc2` conda environment with `MENUDET_WORK_DIR` pointing to the
approved OpenCode temporary directory and `python -m tools.menudet_jit_bench
--report plans/menudet-jit-benchmarks.json`.

### Warm results (milliseconds, 262,144 float64 elements, flat API)

**Follow-up:** `plans/menudet-jit-assembly-inspection.md` records actual kernel
disassembly and interleaved repeated-process timings. Those controlled where
measurements reverse the isolated table's TCC/GCC ranking (GCC/Clang about
0.72–0.74 ms versus TCC about 1.05 ms). Treat this table as recorded isolated-run
evidence, not a stable compiler ranking.
The subsequent full-table rerun in `plans/menudet-jit-interleaved.json` also finds
GCC/Clang outperform TCC on affine/polynomial; Clang polynomial measures 0.127 ms
versus NumPy 0.285 ms and one-thread NumExpr 0.308 ms. See the inspection report's
full-table section for the preferred warm compiler comparison.

| Workload | Interpreter | TCC | GCC 16 | NumPy | NumExpr (1 thread) |
|---|---:|---:|---:|---:|---:|
| `x * 2 + y / 3 - 1` | 1059.037 | 0.645 | 0.878 | 0.305 | 0.353 |
| `(x + y) * (x - y)` | 794.278 | 0.643 | 0.182 | 0.256 | 0.318 |
| `where(x > 0, y / x, x * 2 + y)` | 936.570 | 1.035 | 2.226 | 0.507 | 0.469 |

Across the six large flat workloads, actual JIT is approximately **420–4400×**
faster than the portable interpreter. The huge factor reflects interpreter overhead,
not superiority over mature array engines. Most rows still lose to NumPy/NumExpr;
the float64 polynomial GCC row is about 1.4× faster than NumPy and 1.75× than
one-thread NumExpr. Isolated medians show compiler/backend and tile-size variability,
so this is a measured example rather than a broad speedup claim.

Fresh artifact import/compile costs **2.7–2.9 ms TCC** versus **372–404 ms GCC**
on this host. TCC is the better interactive default for this first slice; GCC's
warm benefit is workload-dependent. Logical-array timings and tiny-input overhead
are preserved in the JSON, not merged with flat execution results. Next obvious
targets are eliminating comparison bridge calls and improving GCC loop generation
without weakening lazy participation, per-node precision or diagnostic contracts.
