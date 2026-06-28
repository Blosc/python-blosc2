# Plan: Transpile blosc2 DSL kernels to JavaScript (browser/Pyodide accel)

## Context

In `newton-js-vs-numpy-vs-nojit-vs-jit.html`, the same Newton-fractal kernel runs four
ways in the browser. Measured: JS 272 ms, numpy 1302 ms, blosc2 no-JIT 3023 ms, blosc2
JIT 887 ms. Hand-written JavaScript is **~3.3× faster than the blosc2 WASM JIT** and ~11×
faster than the no-JIT interpreter, because V8 JIT-compiles a fused scalar loop to
optimized native code while blosc2's WASM JIT (tcc/miniexpr) does not.

The same blosc2 DSL kernels (`@blosc2.dsl_kernel`) are written in a strict, bounded subset
of Python that is already parsed via the stdlib `ast` module. So we can **transpile a DSL
kernel to JavaScript** and run that JS in the browser, capturing the V8 speed win — without
the user rewriting the kernel. This is browser/Pyodide-only by nature.

### Why this shape (decisions settled)

- **Single-threaded, per-block, via the existing `lazyudf` callable seam.** `lazyudf(func,
  inputs)` already accepts a plain Python callable `func(inputs_tuple, output, offset)` and
  drives it per-block through `chunked_eval`/`slices_eval`. Plugging a JS bridge in there
  needs **zero changes to compiled code** (no `.pyx` edits, no rebuild) and handles
  multi-input kernels (Newton takes `a`, `b`) correctly.
- **Not a postfilter.** A postfilter is single-input / same-itemsize / 1:1 — wrong shape for
  an N-input compute kernel. (Postfilter is the hook only for a future *transparent fused
  read*; different feature.)
- **No Web Worker pool / SharedArrayBuffer in the MVP.** That's a parallel-runtime project
  (tiling driver, COOP/COEP headers, Atomics join) mostly *outside* blosc2. Single-threaded
  JS already beats the JIT 3.3×; parallelism is deferred until measured need. Generalizing
  the per-block bridge to **per-chunk** is the natural next rung.
- **Shipped as `src/blosc2/dsl_js.py`** behind `jit_backend="js"` (a new backend alongside
  no-JIT and miniexpr-JIT). Wiring is one swap in `chunked_eval`; no compiled-code changes.
  Started as a repo-root prototype, graduated once benches confirmed the ~2× win.

## Feasibility summary

- **Grammar is bounded and known.** `DSLValidator` in `src/blosc2/dsl_kernel.py:265-492`
  enumerates exactly the supported nodes: assign/augassign, if/elif/else, `for ... in
  range()`, while, break, continue, return; binops `+ - * / // % ** & | ^ << >>`,
  single comparisons, bool ops, unary `+ - not`, calls to `range/where/int/float/bool` and
  `np.* / numpy.* / math.*`, name/constant. Ternary, chained compares, tuple-unpack, input
  reassignment are rejected. A transpiler maps this set ~1:1 to JS.
- **Kernel source is available:** `DSLKernel.dsl_source` (`src/blosc2/dsl_kernel.py:495+`),
  dedented and ready to `ast.parse`.
- **Scalar semantics:** the kernel reads like per-element scalar code (per-pixel `for`/`break`),
  exactly like the hand-written `newtonJS` in the demo. So transliterate the DSL function to a
  JS function with the **same signature**, then drive it element-by-element over each block.

## Architecture

```
lazyudf(js_kernel(newton_dsl), (A_B2, B_B2, MAXITER, relax))[:]
   -> chunked_eval / slices_eval  (existing, unchanged)
       -> per block: bridge(inputs_tuple, output, offset)
            marshal block numpy arrays -> JS typed arrays
            run transpiled JS element loop
            copy result back into `output`
```

`js_kernel(dsl_kernel)` returns the plain Python callable lazyudf expects. The transpiler runs
in Python (pure stdlib `ast`), so it works inside Pyodide too.

## Component 1 — transpiler (`dsl_to_js`)

Walk the Python `ast` of `kernel.dsl_source` and emit a JS function with the same signature
and body. Node mapping (mirror `DSLValidator`'s allowed set so we stay in lockstep):

- **Assign**: first time a local name is seen → `let x = expr;`, later → `x = expr;` (seed the
  "declared" set with the parameter names).
- **AugAssign**: `+= -= *= /=` direct; expand `**= //= %=` to the binop form below.
- **BinOp**: `+ - * /` direct; `**` → `Math.pow(a,b)`; `//` → `Math.floor(a/b)`;
  `%` → `pymod(a,b)` helper `(((a%b)+b)%b)` (Python sign convention); `& | ^ << >>` → JS
  bitwise (int32 coercion — fine for boolean masks, **ceiling:** real 64-bit int bitwise not
  supported).
- **BoolOp** `and`/`or` → `&&`/`||`. **UnaryOp** `+ - not` → `+ - !`.
- **Compare** (single): `== != < <= > >=` → `=== !== < <= > >=`.
- **Call**: `where(c,a,b)` → `(c ? a : b)`; `int(x)` → `Math.trunc(x)`; `float(x)` → `(x)`;
  `bool(x)` → `((x)!=0)`; `np.*/numpy.*/math.*` → name table to `Math.*`
  (`sin cos tan sqrt exp log abs floor ceil pow atan2 ...`); unknown name → raise.
- **For** `for k in range(a[,b[,c]])` → `for (let k=START; k<STOP; k+=STEP)`. Assume positive
  step (matches DSL/demo); **ceiling:** emit step-sign-correct condition only when step is a
  literal, else assume positive and note the limitation.
- **While / If / Break / Continue / Return / Expr-stmt** → direct, with `{ }` blocks.
- **Name / Constant** → identifier / JS literal (`True/False`→`true/false`).

Output: `(js_source_str, input_names)`. ~200-300 lines, single file, stdlib only.

**Index/shape symbols** (`_i0`, `_n0`, `_flat_idx`, …) are **out of scope for the MVP**
(Newton uses none). Detect and raise a clear "not yet supported" error if present.

## Component 2 — runtime bridge (`js_kernel`)

`js_kernel(dsl_kernel)` →:
1. `dsl_to_js(dsl_kernel)` to get JS source + param order.
2. Build a small driver module string:
   ```js
   const __k = function NAME(<params>) { <body>; };
   function __run(arrays, scalars, out, n) {
     for (let i = 0; i < n; i++) out[i] = __k(/* per param: arrays[k][i] or scalars[k] */);
   }
   ```
3. In Pyodide, materialize it once: `import js; run = js.eval("(...)")` → JS function proxy.
4. Return a callable `bridge(inputs_tuple, output, offset)` that, per block:
   - splits inputs into array operands (→ `arr.to_js()` typed arrays) and scalars,
   - calls `run(arrays, scalars, out_js, n)`,
   - copies `out_js` back into `output`.

`# ponytail: per-block to_js() copy; swap to a zero-copy HEAPF64 view onto WASM linear memory
only if marshaling shows up as the bottleneck.`

Outside Pyodide (no `js`), `js_kernel` still exposes `.js_source` for inspection/testing and
raises if you try to *run* it.

## Files (as shipped)

- **`src/blosc2/dsl_js.py`** — `dsl_to_js()`, `build_js_module()`, `js_kernel()` bridge.
- **`src/blosc2/lazyexpr.py`** — `_as_js_udf()` + the `jit_backend="js"` swap in `chunked_eval`.
- **`tests/ndarray/test_dsl_js.py`** — transpiler + node-backed numeric-equivalence tests.
- **`bench/js-transpiler/dsl-js-node.mjs`** — headless Pyodide-in-Node integration test + bench.
- **`bench/js-transpiler/newton-dsl-js.html`** — browser demo (transpiled vs hand-written JS).
- **`bench/js-transpiler/README.md`** — how to run both.

## Verification

1. **Transpiler tests** — `pytest tests/ndarray/test_dsl_js.py`: structure + index-symbol
   rejection, and (when `node` is on PATH) run the emitted JS over a grid and assert it matches
   the Python kernel to ~1e-9.
2. **Headless wired path** — `node bench/js-transpiler/dsl-js-node.mjs`: overlays the local
   pure-Python onto the PyPI wheel and drives the real `lazyudf(jit_backend="js")` path;
   asserts `js`/JIT both match numpy exactly, then benches. Exits non-zero on mismatch.
3. **Browser** — serve repo root, open `bench/js-transpiler/newton-dsl-js.html`, click Run.

## Bench findings (verified)

Measured on Apple M-series, blosc2 4.6.0 under Pyodide 314, Newton 320×213 / max_iter=48,
24-frame `relax` sweep (`dsl-js-node.mjs`):

| backend | ms/frame | vs `js` |
|---|---|---|
| `jit_backend="js"` | **~16** | — |
| miniexpr JIT | ~31 | js **~2× faster** |
| no-JIT interpreter | ~130 | js ~8× faster |

Correctness exact: `js` and JIT both `maxdiff=0.00` vs numpy.

**The PyProxy gotcha (the one real bug the headless harness caught).** The bridge must pass
the per-call operands to the JS driver as real **JS `Array`s**, not Python lists. A Python
list arrives in JS as a `PyProxy`, so every `ops[k][i]` in the hot inner loop crosses the
Python↔JS boundary — still correct, but ~**10× slower** (140 vs 8 ms/frame for the direct
bridge call). The browser demo never hit this because it built its arrays in JS. Fix:
`Array.new()` + `.push(...)` in `js_kernel`'s bridge. With that, per-chunk marshaling is cheap
(multi-chunk ≈ single-chunk), so the **per-chunk driver below is *not* needed** for speed.

## Deferred (explicitly not built now)

- **Whole-array / fewer-crossing driver** — per-chunk is already cheap after the PyProxy fix,
  so this is only worth it if a future kernel shows marshaling-bound; the transpiler is unchanged.
- **Web Worker pool + SharedArrayBuffer** — real multithreading; needs COOP/COEP and a tiling
  driver. Build only if single-thread proves too slow for a real workload.
- **Index/shape symbols** (`_i0`/`_n0`/`_flat_idx`) in the transpiler.
- **Postfilter-based transparent fused read** — different feature (single-input), single-threaded.

## Known ceilings / limitations

- Browser/Pyodide-only.
- 64-bit integer bitwise ops degrade to int32 (JS).
- `%` follows Python sign convention via helper; large-magnitude float edge cases may differ.
- `range()` assumes positive step unless the step is a literal.
- float64/float32 numeric kernels are the target; exotic dtypes untested.
