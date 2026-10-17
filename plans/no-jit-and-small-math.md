# No-JIT arithmetic and small transcendental workloads

Native revision: `27cddcda43a33390542fdf7ca74aacd191c26b22`. Both Python build
and paired-qualification pins identify this local native commit. Local builds
use the source override; this is not a claim of remote fetch/CI availability.

## Delivered

- A bounded straight-line **vector interpreter**, not runtime code generation.
  Eligible float32/float64 arithmetic, strong same-width modular integer
  arithmetic, total numerical conversions and qualified unary math use static C
  primitives instead of recursively evaluating every tree node for every lane.
- Each intermediate retains its dtype/rounding. No fast-math, FMA contraction,
  reordered reductions or unchecked float-to-integer casts were introduced.
- Private workspace is bounded by 64 slots × 256 doubles = **128 KiB per active
  invocation**, plus existing iterator/DSL state. This is not full-result staging
  and is additional to the executor's iterator-scratch budget.
- Partial masks, checked operations, potentially uninitialized locals, excessive
  plan size/depth and unsafe aliases decline before execution. Exact same-lane
  in-place buffers are supported; other overlaps retain scalar traversal.
- Native scheduling distinguishes costly math from bandwidth-oriented maps:
  automatic thresholds are 16K versus 128K elements, with at most four tasks for
  small math jobs. Explicit host thresholds still override that policy.
- Eligible serial executor fallbacks use bounded automatic tiles rather than
  repeatedly invoking a 1K-element kernel. Python's synchronous caller also runs
  one task, saving a pool submission; joining futures directly avoids `wait()`'s
  extra notification registration. An invocation gate prevents late jobs from
  using freed native pointers even if `submit()` enqueues work and then raises.
  Join state is allocated before submission. Failure/interruption paths join
  every already-entered native call before invocation state is freed.

The mechanisms live primarily in miniexpr (`dsl_portable_expr.c`, the new
`dsl_portable_batch.c/.h`, and `dsl_graph.c`). Python only adapts the host executor.
NumExpr remains an optional external benchmark reference, never a runtime backend.

## Measured improvements

M4 Pro, eight configured workers, contiguous float64 resident inputs and a shared
warmed NumPy destination. Public API overhead is included. Compilation and fresh
result allocation are excluded. Before measurements use the previous independent
wheel; after measurements use isolated rebuilt extensions. These are separate
runs, with shuffled engine order within each run, not interleaved old/new builds.

### Genuine no-JIT execution: 8M elements

Both `ME_DSL_JIT=0` and Python's no-JIT policy are selected. The native probe
reports `has_jit=False`; no compiler is invoked by the vector interpreter.
Fifteen samples after five warmups:

| Recipe | Before ms | After ms | Speedup | NumExpr ms (after-run control) |
| --- | ---: | ---: | ---: | ---: |
| Linear | 74.807 | 4.351 | 17.2× | 1.206 |
| Polynomial | 95.120 | 3.601 | 26.4× | 1.568 |
| Division | 49.373 | 4.000 | 12.3× | 1.107 |
| Square-root norm | 123.236 | 4.233 | 29.1× | 1.519 |
| Sine + cosine | 140.120 | 6.440 | 21.8× | 3.849 |
| log1p + expm1 | 153.143 | 7.577 | 20.2× | 4.683 |

No-JIT integer-input arithmetic also improves. At 8M int64 elements, seven
samples after three warmups: linear **83.478 → 3.438 ms**, polynomial
**133.319 → 4.016 ms**, true division **65.987 → 3.976 ms**, norm
**153.985 → 4.790 ms**. Intentional dtype differences remain excluded from parity.

### Small JIT workloads: 100K elements

System-CC JIT, 31 samples after ten warmups:

| Recipe | Before ms | After ms | Speedup | NumExpr ms (after-run control) |
| --- | ---: | ---: | ---: | ---: |
| Sine + cosine | 0.437 | 0.219 | 2.0× | 0.125 |
| log1p + expm1 | 0.535 | 0.254 | 2.1× | 0.134 |
| Fmod | 0.394 | 0.238 | 1.7× | 0.138 |
| Hypot | 0.209 | 0.192 | 1.1× | 0.115 |
| Square-root norm | 0.180 | 0.184 | 1.0× | 0.116 |
| Linear | 0.125 | 0.114 | 1.1× | 0.110 |

Cheap maps do not gain parallelism just because the math threshold changed.
Large JIT maps retain the existing compiled route; the vector interpreter is not
silently substituted for it.

### What did not work

An initial vector path declined the interpreter's non-null all-ones masks and
therefore did not execute on ordinary graphs; explicit batch-admission counters
in the native tests now guard this. Typed bit fills and uniform conversions were
important to avoid replacing tree dispatch with per-lane conversion dispatch.
A larger 1024-lane private vector experiment regressed the sampled large maps;
the implementation retains 256-lane vectors and the 128 KiB bound.

## Qualification

Native tests compare vectors against one-lane scalar execution, not against the
same optimized path: float32/float64 values and FP status, signed zeros, signaling
and quiet NaNs, infinities, subnormals, overflow/underflow, all integer widths,
rounding-sensitive casts, full/partial masks and odd tile tails. Private counters
verify admission and conservative declines. Additional tests cover shifted and
exact aliases, initialized-local failures, and checked-cast prefix/error behavior.
Python tests exercise real small-math tasks, caller participation, output identity
and complete joining after submission failures or interrupted waits, including
late jobs whose `submit()` call raised after enqueueing them.

Qualification: **11,677 Python tests**, **354 native release tests**,
**12 ASan/UBSan tests**, and **12 actual wasm32 tests** passed. The native counts
include the scalar-versus-vector parity and fallback tests. The real TCC parity
matrix passed too. The independently built wheel passed **168 targeted tests**
with both NumExpr's module and distribution metadata absent (24 external-reference
cases skipped), plus **120 doctests** (two skipped) in a fresh working directory.
Wheel metadata has no NumExpr runtime dependency. Shared editable extensions and
the user's separate Python 3.15 build were not overwritten.
The wheel also executed float32/float64/int64/uint64 arithmetic with JIT disabled,
NumExpr absent and `CC` pointing to a nonexistent compiler, reporting zero JIT
stages and exact expected results.

Native release builds introduce no compiler warnings. The Cython rebuild still
emits warnings in unchanged
declarations/constness/`abs` code and the existing duplicate-library link setup;
these are not new warnings from the batch primitives.

## Remaining performance gap

This is **not universal NumExpr parity**. The six covered no-JIT float64 recipes
above are now about **1.6–3.7×** slower, rather than tens of times slower; small transcendental
JIT maps remain roughly **2×** slower. Remaining costs include intermediate-tile
traffic, instruction/operation dispatch, scalar libm callbacks, and host launch/API
overhead. Unsupported masked/checked recipes, binary math and predicate/select
operations outside the initial subset still use the scalar interpreter.
Actual Windows runtime qualification remains separate from local macOS/wasm32.

## Reproduce

Use conda environment `blosc2` with extensions from the matching native checkout:

```sh
conda run -n blosc2 python bench/evaluate-numpy-parity.py --items 100000 8000000 --backend interpreter --cases linear polynomial division sqrt trig log-exp --samples 15 --warmups 5
conda run -n blosc2 python bench/evaluate-numpy-parity.py --items 100000 --backend jit --cases linear sqrt trig log-exp hypot fmod --samples 31 --warmups 10
```

The benchmark verifies dtype, values, output identity and JIT availability before
timing; unsupported or non-equivalent reference signatures remain `N/A`.
