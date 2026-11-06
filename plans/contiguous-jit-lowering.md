# Contiguous JIT lowering: first implementation pass

Follow-up: [no-jit-and-small-math.md](no-jit-and-small-math.md) records the new
bounded vector interpreter and small-math scheduling results. The no-JIT and
small-workload limitations below describe this earlier pass.

This pass targets regular contiguous numerical maps, not strided-store tuning or
unchecked numerical transformations. Explicit `jit=False` still means no JIT;
the large no-JIT arithmetic gap is not fixed by secretly compiling that mode.

Native implementation: `f19aed813d0153342a9f2e4238c8f0fc0003b9f4`; both Python
build/qualification pins identify this revision. It is a local commit until
published; local builds use the native source override.

## Findings and changes

1. `evaluate_chunk()` forced `jit=False`, even when the surrounding evaluation
   allowed native compilation. It now requests the existing qualified lowering
   when JIT is enabled/default. Explicit no-JIT and unsupported graph backend/FP
   policies retain the interpreter. Cached plans do not retain arrays. Runtime
   failures propagate without a NumPy retry.
2. A kernel's `has_jit=True` does not imply every operation is compiled. Integral
   `minimum`, `maximum`, `fmin`, `fmax` and `fmod` could enter the checked host
   bridge, reconstruct an operation and evaluate it in the scalar interpreter
   for each lane. These matching-dtype operations now lower directly to typed C
   comparisons/remainders. Signed minimum / -1 and zero-divisor guards remain;
   floating and mixed-domain operations keep their established semantics.
3. Native graph safety metadata now admits integral extrema and identity rounding
   operations for host task partitioning. Numerical checks in child expressions
   are not weakened; checked casts, integer power and checked scalar subtrees
   still do not gain parallel admission through an enclosing math function.
4. Chunk execution uses bounded 64K tiles only for certified partitionable maps.
   It stays serial within the caller's chunk worker, avoiding nested worker pools.

The generated source revision advances to r29. Native tests verify that integral
extrema/remainder kernels use no checked-operation callback, as well as exact
interpreter/JIT values, masking, signed/unsigned boundaries and FP status. Python
tests cover dispatch/policy/output identity, error propagation and real native
host tasks rather than duplicating the native dtype matrix.

## Benchmark qualification

The resident benchmark now checks result dtype before timing, not only values.
For example, NumExpr promotes `minimum(int64, int64)` to float64, whereas NumPy
and Blosc2 preserve int64. Comparing the two with a float64 destination measures
an extra output conversion on the Blosc2 side, not matching integer kernels.
Such intentional promotion differences are now `N/A`.

For same-dtype integer comparisons, an equivalent NumExpr reference can use
`where(x < y, x, y)` / `where(x > y, x, y)`. Integer `fmod` can be compared with
`x % y` only for strictly positive inputs; the operations differ on negative
inputs. Any reported comparison using these spellings must identify them.

## Initial measurements

M4 Pro, eight workers, 8M contiguous elements, system-CC JIT, warmed explicit
destinations, seven alternating samples. Old measurements use the preceding
independently built wheel; new measurements use rebuilt isolated extensions.

| int64 recipe | Before ms | After ms | NumExpr equivalent ms |
| --- | ---: | ---: | ---: |
| `minimum(x,y)` | 792.221 | 1.223 | 1.194 (`where(x < y,x,y)`) |
| `maximum(x,y)` | 803.057 | 1.185 | 1.192 (`where(x > y,x,y)`) |
| `fmod(x,y)` | 123.096 | 1.377 | 1.215 (`x % y`, positive inputs only) |

These gains include removal of scalar-operation interpretation **and** native
parallel admission, not code generation alone. NumExpr controls in the separate
before run were 1.341/1.229/1.367 ms respectively. These remain preliminary local
measurements, not cross-platform qualification or universal speedup claims.

To reproduce the integer comparison without NumExpr's dtype promotion, use the
`blosc2` conda Python and `ME_DSL_JIT_COMPILER=cc` with this core benchmark:

```python
from statistics import median
from time import perf_counter
import blosc2
import numexpr as ne  # external reference only
import numpy as np

blosc2.set_nthreads(8)
ne.set_num_threads(8)
x = np.arange(8_000_000, dtype="int64") % 97 + 1
y = x[::-1].copy()
out = np.empty_like(x)
for expr, ref in [
    ("minimum(x,y)", "where(x < y,x,y)"),
    ("maximum(x,y)", "where(x > y,x,y)"),
    ("fmod(x,y)", "x % y"),
]:
    calls = [
        lambda: blosc2.evaluate(expr, local_dict={"x": x, "y": y}, out=out),
        lambda: ne.evaluate(ref, local_dict={"x": x, "y": y}, out=out),
    ]
    for call in calls:
        for _ in range(2):
            assert call() is out
    times = [[], []]
    for sample in range(7):
        for i in [0, 1] if sample % 2 else [1, 0]:
            start = perf_counter()
            calls[i]()
            times[i].append(1000 * (perf_counter() - start))
    print(expr, ref, [median(values) for values in times])
```

The measured run additionally checked output values against NumPy before timing.

For a serial float64 chunk, `x*2+y` took **402.326 ms before** despite `jit=True`,
and **1.769 ms after**, with no nested pool. Explicit `jit=False` remained around
**391–397 ms**. This improves actual JIT dispatch, not interpreter-only execution.
Cold compilation is excluded; these are steady-state throughput measurements.

The existing regular-map benchmark remains competitive at 8M float64 elements:

| Recipe | NumExpr ms | Blosc2 JIT ms |
| --- | ---: | ---: |
| Linear | 1.164 | 1.183 |
| Polynomial | 1.549 | 1.158 |
| Square-root norm | 1.481 | 1.089 |
| Sine + cosine | 3.845 | 4.200 |
| Selection | 1.373 | 1.144 |

These used 15 shuffled samples after warmup. Small workloads remain a gap:
100K-element sine/cosine was **0.426 ms** versus NumExpr **0.129 ms**, and norm
**0.189 ms** versus **0.102 ms**. This pass does not claim universal parity.

The generic resident benchmark includes extrema recipes and now marks the
NumExpr integral-promotion mismatch `N/A` rather than timing extra output casting.
It still accepts the original regular-map suite for wider follow-up comparisons.

## Qualification

- Real TCC native interpreter/JIT parity matrix passed, including the integral
  dtype/mask/boundary matrix and assertions that these kernels have no checked
  operation bridge.
- System-CC full parity passed. The compact CC gate hit its 180-second timeout
  during concurrent cold-cache compilation, then passed in 38.9 seconds on rerun;
  no timeout or numerical evidence requirement was relaxed.
- Native graph/array checks passed: **11 ASan/UBSan**, **11 actual wasm32**.
  This does not add a portable WASM JIT ABI; wasm32 retains its existing backend.
- Standalone native full suite: **354 passed** on the final rerun.
- Python full `tests/` suite: **11,674 passed**, 76 skipped, 12,907 deselected.
  Targeted integration/compatibility checks: **309 passed** with system compiler
  JIT enabled.
- Independently built wheel with the NumExpr module **and** distribution metadata
  absent: **165 passed**, 24 external-reference cases skipped. Wheel metadata
  still contains no NumExpr runtime requirement. Configured doctests: **120
  passed**, two skipped, in an isolated working directory.

No fast-math flags, unchecked float-to-int casts, changed reduction order or
NumExpr runtime dependency were introduced. Generic no-JIT interpreter
throughput, floating-operation callback inlining, and remaining reduction/DSL
scheduling migration remain further work; this pass does not finish every item
in the broader native-layer plan.
