# DSL → JS transpiler: coverage gaps & future work

Status of the `blosc2.dsl_js` transpiler (the `jit_backend="js"` path) versus the
miniexpr + WASM-JIT backend. Everything listed below as *unsupported* currently rides on
**miniexpr + jit-wasm** instead of the JS bridge.

## Implemented

- **P1 — Index / shape symbols** (`_i0`/`_n0`/`_ndim`/`_flat_idx`, ...). The transpiler emits them
  as trailing kernel params and the runtime driver reconstructs per-block global coordinates
  from `(off, gshape, cshape)`; see `_module_with_index` in `src/blosc2/dsl_js.py`. The
  whole-array shape is threaded `chunked_eval → _maybe_js_backend → _as_js_udf → js_kernel`.
  Requires ≥1 array operand (zero-input DSL kernels stay on miniexpr) and a known output
  shape; without a shape such kernels fall back. Covered by `tests/ndarray/test_dsl_js.py`
  (`test_index_*`) and `tests/ndarray/test_wasm_dsl_jit.py::test_wasm_dsl_index_symbols_via_js`.
- **P2 (input side) — Integer inputs with a floating output.** The JS bridge already
  float64-converts every operand, which is exactly miniexpr's promotion of integer inputs for
  a float result (so values above 2**53 lose precision identically). `_js_dtypes_ok`
  (`src/blosc2/lazyexpr.py`) now admits integer inputs when the output dtype is floating.
  Integer/complex *output* still goes to miniexpr — see the remaining P2 work below.

## Performance characteristics (and where the residual cost is)

Measured with `bench/js-transpiler/dsl-js-node.mjs` (Pyodide, ms/frame). JS beats miniexpr's
TinyCC JIT (`tcc`) on **compute-heavy** kernels and lands at parity / slightly behind on
**compute-light, vectorizable** ones:

```
kernel    js/tcc
newton     2.80x   (heavy: loop + complex arithmetic)
deepar     2.78x
idxgrad    2.00x   (P1 index symbols)
deep       1.30x
trans      0.99x
intmix     0.87x   (P2 int inputs; light, vectorizable)
poly       0.86x   (light, vectorizable)
```

Two cost components matter, and only the second remains:

- **Per-evaluation transpile + `js.eval` (amortized away).** Each `lazyudf` evaluation used to
  re-parse the kernel AST and re-`eval` the JS module, while miniexpr caches its compiled
  program by source. Now memoized: `_TRANSPILE_CACHE` (by kernel source) and `_RUN_CACHE` (the
  V8-compiled `__run`, by module string) in `src/blosc2/dsl_js.py`. This lifted every ratio
  (e.g. newton 2.20→2.80x, poly 0.77→0.86x) and is a real win for repeated / animation-loop use.

- **Per-block marshaling (the residual).** The bridge copies each block across the Python↔JS
  boundary: in via `ascontiguousarray(float64) → tobytes → Float64Array`, out via
  `to_bytes → np.frombuffer`. miniexpr's prefilter computes **in place** with zero copies. For
  light kernels (~2 ms compute) these two copies are a meaningful fraction with no compute to
  hide them behind, so JS sits at parity or just behind `tcc` there. For compute-bound kernels
  (the reason the JS backend exists) it is negligible.

**Future lever — zero-copy block I/O.** Replace the `tobytes`/`frombuffer` copies with a
`HEAPF64` view onto WASM linear memory so operands/output alias the block buffers (the
"ponytail" note in `js_kernel`). This would mostly close the gap on marshaling-bound (light)
kernels but needs care around WASM-heap lifetime/alignment, and does nothing for compute-bound
kernels — so build it only if a real marshaling-bound workload appears.

## How routing works today

Under WebAssembly with `jit_backend` unset (and `jit != False`, no `strict_miniexpr`),
blosc2 *prefers* JS for float DSL kernels and **silently falls back to miniexpr+jit-wasm**
for anything it can't transpile — see `_maybe_js_backend` (`src/blosc2/lazyexpr.py:1475`):

```python
try:
    bridge = _as_js_udf(expression)  # transpiles; raises on any unsupported construct
except Exception:
    return expression, jit, jit_backend  # fall back to miniexpr, no regression
```

With an **explicit** `jit_backend="js"`, the same gaps instead **raise** rather than fall
back (the user asked for JS specifically, so we don't second-guess them). This includes a
non-floating *output* dtype: `_maybe_js_backend` raises a clear `ValueError` up front rather
than letting the float64 bridge silently compute integer/complex output (see below).

The JS backend today covers *float64/float32 element-wise scalar kernels* using arithmetic,
`where`, comparisons, `if/elif/else`, `range` loops, and whitelisted math functions.

## Remaining P2 — Integer *output*

`_js_dtypes_ok` still sends any non-floating *output* dtype to miniexpr, because the JS
bridge computes in **float64** and can't reproduce integer semantics for the result:

- **Integer division / modulo / truncation**: `//`, `%`, `int(...)` must match C/miniexpr
  integer rules, not float `Math.floor`/`pymod`.
- **Overflow / wraparound**: miniexpr wraps at the integer width; float64 doesn't.
- **int64 range**: float64 can't represent int64 above 2**53 exactly.

Options, in rough order of effort:
- **int32 and smaller output**: representable exactly in float64; could be allowed for kernels
  that provably stay within ±2^53 with integer-valued ops and an explicit safe-range / no-
  overflow contract. Still needs integer-correct `//`/`%`/`int()` codegen.
- **int64 output**: requires BigInt or a typed-array split-word scheme — significantly more
  work and likely slower; probably not worth it until a real workload needs it.

## Other unsupported constructs (lower priority)

All of these raise `_DSLToJSError` in the transpiler → fall back (or raise under explicit
`jit_backend="js"`).

**Reductions** — any `reduce_args` (`sum`, `prod`, …) → miniexpr. Explicit
`jit_backend="js"` raises `'jit_backend="js" does not support reductions'`. A JS reduction
path would need a fundamentally different driver (accumulate, not map).

**Statements** — only `Assign, AugAssign, Return, Expr, If, For(range), While, Break,
Continue` are emitted (`_stmt`, `src/blosc2/dsl_js.py:151`). Not supported:
- Tuple / multiple / subscript assignment targets — only a single `Name` target is handled
  (`node.targets[0].id`). `a, b = ...`, `a = b = ...`, `arr[i] = ...` all fail.
- `with`, nested `def`, `try`, etc.

**Expressions** — only `Name, Constant, UnaryOp, BinOp, BoolOp, Compare, Call` (`_expr`).
Not supported:
- Python ternary `a if cond else b` (`ast.IfExp`) — must be written as `where(cond, a, b)`.
- Chained comparisons `a < b < c` — only `ops[0]`/`comparators[0]` are read.
- Subscript / indexing, attribute access (except `np.`/`numpy.`/`math.` call targets),
  tuples, lists, dicts, comprehensions, slices.

**Calls** — only `where`, `int`, `float`, `bool`, and the `_MATH` whitelist (`sin, cos, exp,
log, sqrt, pow, floor, abs, min/max, …`, see `src/blosc2/dsl_js.py:27`). Any other call name,
or a call through a non-`np`/`numpy`/`math` target → fall back.

**For-loops** — only `for v in range(...)`. Iterating over arrays/other iterables is
unsupported.

## Environment gate (by design)

Browser/Pyodide only. `_as_js_udf` raises `RuntimeError` off-WASM (`js_kernel` imports
Pyodide's `js` at run time). On native/CI, DSL kernels always go to miniexpr+jit.

## Known semantic ceilings (supported, but lossy)

These transpile but with caveats worth tracking, since miniexpr may differ:
- 64-bit integer bitwise ops degrade to int32 (JS number semantics).
- `%` uses a Python-sign helper (`pymod`); large-magnitude float edge cases may differ.
- `range()` with a non-literal step assumes a positive step (loop-direction guess).
- float64/float32 are the target; exotic dtypes untested.

## See also

- `plans/dsl-js.md` — original design, perf numbers, and the "Deferred" / "Known ceilings"
  notes this document expands on.
- `src/blosc2/dsl_js.py` — the transpiler.
- `src/blosc2/lazyexpr.py` — `_maybe_js_backend`, `_js_dtypes_ok`, `_as_js_udf` (routing).
