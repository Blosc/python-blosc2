# Native execution and NumExpr-removal evaluation

Implementation branches: `improved-miniexpr` in Python-Blosc2 and miniexpr.

Implementation boundaries: Python-Blosc2 `69930b98` (native integration),
`b64cff00` (runtime removal) and `06863bfa` (memoryview owners); native miniexpr
head `92caba958102f1ae182a6a344cee6227e8c4175e`. Python's build pin and paired
qualification pin both identify that native head. These are local commits.

## Delivered

- Versioned native output descriptors, negative strides, bounds/alignment and
  conservative self-overlap validation; old contiguous APIs remain compatible.
- Bounded output scatter for maps and addressed stores for serial reductions,
  including graph sources, conversions and staged final destinations.
- Native map-partition safety, synchronous host executors, task preflight/joins,
  deterministic failure selection and aggregate iterator-workspace budgets.
- Destination-friendly traversal/partition axes for certified resident maps.
- Native identity copies without arithmetic-kernel interpretation or compilation.
- Python binding of actual allocation bounds, direct strided outputs and native
  graph scheduling through a host executor. Fused compression remains intact.
- No NumExpr runtime import/calls/dependency/thread initialization. NumPy handles
  unsupported functionality. `prepare_evaluate`/`re_evaluate` remain available.

## Local performance

M4 Pro, eight workers, 8M float64 elements, explicit shared warmed NumPy output,
seven timed samples after warmup, randomized engine order. Public API overhead is
included; preparation/compilation warmup and fresh-result allocation are excluded.
NumExpr is an external comparison package, not a Blosc2 backend.

| Recipe | NumExpr ms | Native JIT ms | Native interpreter-only ms |
| --- | ---: | ---: | ---: |
| Linear | 1.177 | 1.196 | 79.264 |
| Polynomial | 1.561 | 1.139 | 96.385 |
| Division | 1.121 | 1.164 | 50.947 |
| Power | 4.876 | 5.034 | 29.366 |
| Square-root norm | 1.490 | 1.072 | 127.497 |
| Sine + cosine | 3.827 | 4.213 | 138.661 |
| log1p + expm1 | 4.694 | 4.507 | 151.463 |
| Hypot | 1.296 | 1.366 | 89.771 |
| Fmod | 3.494 | 3.691 | 124.768 |
| Rounding | 2.209 | 1.555 | 200.759 |
| Predicate | 1.482 | 1.060 | 62.576 |
| Selection | 1.375 | 1.175 | 70.955 |
| Copy | 0.803 | 0.729 | 0.697 |
| Sum of products | 6.764 | 1.312 | 30.634 |

JIT and interpreter measurements are separate runs, not paired cross-backend
samples. Their NumExpr controls were similar but not identical. Copy uses a
native memory-copy shortcut, not the interpreter or JIT arithmetic kernel.
Automatic parallel sum policy still uses the existing Python host adapter.

At 100M float64 elements, interpreter-only configuration copied into a reused
destination in **7.021 ms**, versus NumExpr **10.159 ms** (15 samples). This is
genuine JIT-disabled native execution, unlike the earlier NumExpr fallback run.

Complex construction is an explicit NumPy fallback: a separate 8M-element run
measured **4.696 ms** versus NumExpr **1.466 ms**. It fills the real and imaginary
destination components directly, avoiding a full complex result followed by a
copy, but remains slower and is not native complex support.

An initial native scheduling experiment retained 1K-element tiles and regressed
large JIT maps to roughly 2.6–3.0 ms. Native automatic executor tiling (up to 64K
elements within workspace limits) restored the competitive JIT results above.

### Integer math and memory layouts

The native proof was subsequently extended to safe real-math promotion of integral
arrays (without admitting checked integer division, casts or powers). At 8M
int64 elements with JIT/eight workers, formerly serial cases became competitive:

| Integer-input recipe | NumExpr ms | Native JIT ms |
| --- | ---: | ---: |
| Division | 1.294 | 1.138 |
| Square-root norm | 1.786 | 1.085 |
| Sine + cosine | 5.478 | 4.799 |
| log1p + expm1 | 5.109 | 4.454 |
| Hypot | 1.801 | 1.385 |
| Fmod | 2.937 | 2.293 |
| Selection | 1.569 | 1.280 |

For matching F-order inputs/output of shape `(2000, 4000)`, float64 JIT timings
were linear **1.193 ms**, polynomial **1.157 ms**, norm **1.069 ms**, copy
**0.716 ms**, versus NumExpr **1.168/1.560/1.482/0.806 ms** respectively.
Conflicting C-input/F-output layouts remain a performance gap: polynomial/norm
were about **15 ms** versus NumExpr **8.0/7.2 ms**, and copy **7.2 ms** versus
**2.4 ms**. These need blocked gather/scatter/cache/TLB optimization, not another
Python execution path.

### Reproduction commands

Use the `blosc2` conda environment with extensions built from these branches:

```sh
conda run -n blosc2 python bench/evaluate-numpy-parity.py --items 8000000 --backend jit --threads 8 --samples 7 --json jit.json
conda run -n blosc2 python bench/evaluate-numpy-parity.py --items 8000000 --backend interpreter --threads 8 --samples 7 --json interpreter.json
conda run -n blosc2 python bench/evaluate-numpy-parity.py --items 100000000 --cases copy --backend interpreter --threads 8 --samples 15
conda run -n blosc2 python bench/evaluate-numpy-parity.py --shape 2000 4000 --input-order F --order F --backend jit --threads 8
```

The script verifies JIT availability/request and numerical/output qualification.
Non-equivalent or unsupported reference cases remain `N/A`. Compare both fresh
and reused allocations separately when extending these measurements.

## Qualification and interpretation

Native validation includes standalone release tests, ASan/UBSan graph/array
tests, and actual Emscripten/Node wasm32 graph/array tests. Python validation
includes descriptor bindings, real host-worker execution, join-on-submission
failure, ordinary expression regressions and NumExpr import-blocking processes.
Final counts/build outcomes are recorded below.

Confirmed qualification so far:

- Standalone native release suite: **354 passed**.
- Native graph/array tests: **11 passed** under ASan/UBSan and **11 passed**
  under actual Emscripten/Node wasm32 (repeated after the integer-policy update).
- Independently built/extracted wheel with NumExpr imports blocked: **11,721
  passed** in the earlier run. The final wheel rerun passed **11,726 tests**,
  with 136 skipped and 12,907 deselected, including the configured doctests and
  the compatibility-hint/native-policy changes.
- A dependency-only search path excluding both the NumExpr module and its
  distribution metadata passed **176 targeted tests** (24 external-reference
  cases skipped). No import blocker was necessary in that genuinely absent
  dependency configuration.
- A final memoryview-owner fix was checked with **182 targeted tests** and the
  genuinely absent dependency configuration above; it snapshots disjoint views
  sharing a buffer owner when native allocation-level alias rules require it.
- Wheel metadata contains no `Requires-Dist: numexpr`; shared editable extensions
  and the user's Python 3.15 build directory were not overwritten.

Earlier qualification harness issues were fixed without changing product behavior:
the isolated runner needed a multiprocessing main guard, and doctests needed a
clean working directory to avoid existing fixed-name user data files. No user
data file was removed to make the doctests pass.

Builds used the local native source override
`-Ccmake.define.FETCHCONTENT_SOURCE_DIR_MINIEXPR=/absolute/path/to/miniexpr`.
Native commits are local to the development branch until published; the aligned
Python/native pin is not a claim of remote CI qualification or fetch availability.

The main unresolved issue is **no-JIT arithmetic throughput**, not correctness or
availability of a native interpreter. Its scalar recursive evaluator is visible
in `src/dsl_portable_expr.c`; optimization should be driven by profiles and native
numerical qualification rather than replacing checked semantics with unchecked
vector loops. A vector/bytecode interpreter is a likely follow-up, not implemented
or claimed here.

Other remaining architecture work: concurrent reader capabilities, a public
C-Blosc2 executor adapter, migration of existing automatic parallel-reduction
policy/non-graph portable-DSL proofs, and fuller mixed-layout optimization.
Windows/runtime CI and remote qualification evidence remain separate checks.

**Decision:** the runtime dependency can be removed functionally, and native JIT
maps/copies remain competitive. Shipping the removal for JIT-unavailable users
requires an explicit decision about the substantial interpreter regression.
