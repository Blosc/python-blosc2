# Menudet expanded typed JIT lowering

## Scope

Portable 1.1 remains opt-in (`jit=True` / `ME_JIT_ON`). All backends receive the
same generated C source, now identified as `lowering-r8 mask-compare-math-abi-r4`.
Checked 1.0, dependency pins, default backend selection and artifact versions do
not change.

New supported families:

- Straight-line local assignments and reassignment. Typed volatile lane-local
  stores preserve assignment precision and floating exceptions, including unused
  assignments. This intentionally leaves optimization headroom.
- Simple `if`/`elif`/`else`, nested branch lowering and lane-local returns. Returns
  advance the participating lane rather than ending the whole array kernel.
  Definite assignment is rechecked conservatively across branch alternatives;
  unsupported or uncertain cases stay interpreted.
- Same-dtype signed/unsigned 8/16/32/64-bit `+`, `-`, `*` and negation. All arithmetic
  uses uint64 modular operations, then narrows unsigned and bit-copies into the
  target representation; no signed C overflow or out-of-range signed cast is used.
- Unary float32/float64 `sin`, `cos`, `tan`, `exp`, `log`, `sqrt`, `floor`, `ceil`.
  A private invocation-time bridge reuses the authoritative native `p_math`
  implementation. It retains host libm selection/precision and diagnostics, but
  still incurs a host call and function-name dispatch per participating node.

Existing lazy `where`, Boolean short circuiting, ordinary inline floating
comparisons, exceptional NaN comparison bridge, participating masks, scoped fenv
and final floating conversions are retained. Native math bindings are borrowed
typed-tree pointers supplied at invocation, not serialized in artifacts/source.

The shared arithmetic corpus caught `uint64` versus a weak signed `-1` comparison:
C's usual unsigned conversion would incorrectly equate it with UINT64_MAX. Mixed
signed/unsigned comparison nodes without explicit common promotion now fail closed
to the authoritative interpreter. Mixed-width integer conversions, integer
division/shifts, floating `//`, other math calls, loops, block-scalar reductions
and ND context remain outside this slice. WASM host-pointer lowering remains
disabled; cross-platform numerical qualification is not implied by these host tests.

## Validation

- Before the test-only bulk refactor, full Python suite: **12,878 passed, 78 skipped**;
  the focused actual-JIT suite
  separately passed **198** TCC/GCC/Clang tests. Optional corpus selectors were
  not supplied to the full Python run; authoritative corpora run natively below.
- Expanded focused Python tests require actual TCC/GCC/Clang compilation, covering
  integer overflow/min/max boundaries, finite value bits, quiet/signaling NaNs,
  function diagnostics, lazy and masked participation, local assignment rounding,
  unused assignment diagnostics, reassignment and branches.
- Standalone native JIT tests now cover locals, early returns, branch-assigned
  locals, unary math and modular integer composition, with and without masks.
- Full native suite: 427 passed after excluding the signed/unsigned comparison
  corner case, including authoritative arithmetic and function corpora.
- Explicit request-on TCC corpus execution matches all **3,193 arithmetic** and
  **1,154 function** cases. Actual compiled coverage increases from the original
  **181 to 1,249 arithmetic cases** and **15 to 135 function cases**. Remaining
  routes retain their interpreter/expected-diagnostic classification. Counts:
  `plans/menudet-jit-expanded-conformance.json`.
- Six ASan/UBSan checks passed, including forced actual GCC-JIT execution of the
  expanded standalone fixtures. Sanitizer build emitted a duplicate-library linker
  warning (`-lm`, `libminiexpr.a`); no new compiler warnings or sanitizer findings.
- Serial GCC compilation reached the former 300-second limit without a reported
  value mismatch; a subsequent partially cached serial run passed in 289.69 seconds.
  Serial Clang passed in 500.05 seconds. The test-only bulk refactor below restores
  a **60-second timeout** rather than accepting these long compilation times.

## Benchmark method

The user's `../miniexpr/bench/benchmark_dsl_interpreter_vs_jit.c` is left unchanged.
`tools/menudet_lowering_bench.py` repeats it in five fresh processes with fresh JIT
cache directories. Each process uses 65,536 elements and seven warm repeats, with
the native harness's adaptive batching and minimum timing. Compilation is excluded
and result bytes are checked against the interpreter.

This deliberately reproduces that harness, **not** the interleaved compiler-ranking
method. Use it for coverage and large interpreter speedups; fixed backend order,
minimum timing and untraced CPU placement/frequency do not support fine-grained
compiler rankings. The report records per-process rows, ranges, compiler versions,
executable/shared-library hashes and raw output.

Reproduce with `MENUDET_WORK_DIR` set to the approved temporary directory:

```sh
conda run -n blosc2 python -m tools.menudet_lowering_bench \
  "$BUILD/bench/benchmark_dsl_interpreter_vs_jit" \
  --report plans/menudet-jit-expanded-lowering.json
```

## Expanded coverage results

Median of five native process results, warm milliseconds, 65,536 elements:

| Family | Interpreter | TCC | GCC 16 | Clang 21 | Request-on route |
|---|---:|---:|---:|---:|---|
| A: `sin(x)+cos(x)` | 77.063 | 6.860 | 6.829 | 6.776 | JIT |
| B: int32 `x+y` | 68.578 | 0.230 | 0.051 | 0.020 | JIT |
| C: `x // 3.0` | 68.513 | 68.612 | 68.606 | 68.646 | interpreter |
| D: `y=x*2; return y+1` | 136.696 | 0.101 | 0.133 | 0.122 | JIT |
| E: `if x>0: return x; return -x` | 82.466 | 0.190 | 0.118 | 0.102 | JIT |
| F: `sum(x)` | 0.492 | 0.473 | 0.471 | 0.472 | interpreter |
| G: `x + _i0` | 68.797 | 69.094 | 68.965 | 69.028 | interpreter |
| Control: `where(x!=0,y/x,y)` | 192.634 | 0.216 | 0.195 | 0.144 | JIT |

All five processes report bitwise agreement with the interpreter. Four formerly
unsupported families now compile on all three backends. Math is about **11×**
faster; the cheap integer/local/branch examples improve by hundreds to thousands
of times. The interpreter dispatch overhead dominates those large ratios, rather
than demonstrating broad superiority over NumPy/NumExpr. Math still pays host
bridge/name-dispatch overhead, and volatile local stores leave optimization room.

Per-process compiler rankings fluctuate substantially under this harness (e.g.
GCC locals 0.052–0.230 ms); do not treat fine timing differences as compiler
rankings. Complete evidence: `plans/menudet-jit-expanded-lowering.json`.

### Existing interleaved workloads

The original shared-buffer/rotating-order benchmark was also rerun across five
fresh processes, without concurrent validation work. Warm milliseconds for
262,144 float64 elements; median of process medians, NumExpr single-threaded:

| Workload | Interpreter | TCC | GCC 16 | Clang 21 | NumPy | NumExpr |
|---|---:|---:|---:|---:|---:|---:|
| Affine | 1068.613 | 0.674 | 0.314 | 0.274 | 0.344 | 0.400 |
| Polynomial | 803.097 | 0.683 | 0.169 | 0.145 | 0.341 | 0.333 |
| Lazy where | 940.055 | 1.064 | 0.383 | 0.341 | 0.596 | 0.480 |

These remain consistent with the preceding inline-comparison rerun (GCC/Clang
where 0.385/0.344 ms). Values and actual compilation are checked before/after
sampling. Evidence: `plans/menudet-jit-expanded-interleaved.json`.

## Exhaustive compiler tests without per-kernel process startup

The cc corpus runner now preloads the same artifacts and invokes the same typed
portable source emitter. Each eligible function is copied unchanged into one C
translation unit, except for its unique exported symbol name. The shared helper
header must match byte-for-byte; compiler/FP configuration must agree. One normal
strict cc compilation produces the module. Every artifact retains its own typed
tree, runtime bindings and separately resolved compiled entry point.

All numerical cases and assertions remain: inferred dtype, output bytes/tolerances,
floating status, raising/recovery, repeated calls and caller-fenv restoration. No
matrix sampling or replacing compiled execution with interpreter results is used.
The library is owned by the harness and stays loaded until all artifacts are freed.
Windows/WASM keep their existing runner route. `MENUDET_JIT_SERIAL=1` opts back
into the original per-artifact runtime compilation for comparison/debugging.

Normal production compilation, cache identity and compiler options are unchanged.
Dedicated runtime/cache tests and the standalone JIT fixtures continue to exercise
per-kernel loading. Default CTest additionally includes full arithmetic/functions
cc corpus tests on hosts with an available compiler, with actual compilation
required. All exhaustive corpus timeouts are 60 seconds.

Fresh-cache timings from `tools/menudet_batch_qualification.py`:

| Corpus | Serial TCC | Bulk GCC 16 | Bulk Clang 21 | Cases / compiled kernels |
|---|---:|---:|---:|---:|
| Arithmetic | 26.11 s | 26.55 s | 25.99 s | 3,193 / 1,249 |
| Functions | 1.05 s | 1.63 s | 1.11 s | 1,154 / 135 |

Arithmetic is **over 11× faster** than GCC's timed-out >300-second serial run
(10.9× versus its partially cached 289.69-second rerun), and **19.2× faster** than
Clang's 500.05-second serial run. Functions improve by **22.6× / 32.9×** versus
the 36.75 / 36.46-second serial GCC/Clang runs. Each cc corpus invokes the external
compiler **once**, but still executes every eligible compiled kernel.

Every complete JSON result row is identical between serial TCC, bulk GCC and bulk
Clang, not just the aggregate pass count. Output SHA-256 digests and route counts
are recorded in `plans/menudet-jit-batch-qualification.json`. Remaining ~26-second
arithmetic cost is largely the unchanged exhaustive runner/evaluator work (serial
TCC takes the same time), not external compiler startup.

Post-refactor validation: **429 native tests passed** in **44.25 seconds**, including
the two new exhaustive cc tests and existing per-kernel runtime/cache tests. A
rebuilt editable Python integration passed nine TCC/GCC/Clang runtime tests for
fallback/casts, lazy masks/status/concurrency and graph reuse. ASan/UBSan passed
the normal standalone actual-JIT fixture and the full 1,154-case bulk function
corpus; no sanitizer failures were reported. Ruff and diff checks pass.
