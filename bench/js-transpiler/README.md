# DSL → JavaScript transpiler benches

Benches/demos for `blosc2.dsl_js`, which transpiles a `@blosc2.dsl_kernel` to JavaScript so
kernels run at V8-optimized native speed in the browser/Pyodide (the `jit_backend="js"`
path). Design and findings: [`plans/dsl-js.md`](../../plans/dsl-js.md).

Both run a Newton-fractal kernel (high arithmetic intensity + per-pixel early exit) and
compare backends. Run everything from the **repo root**.

## Headless (Node + Pyodide) — `dsl-js-node.mjs`

Integration test **and** perf bench, no browser. Installs the blosc2 wasm wheel from PyPI,
overlays this working tree's pure-Python (`src/blosc2/dsl_js.py` + `lazyexpr.py`) on top of
it before importing blosc2 — so the wired `jit_backend="js"` path is exercised **without
rebuilding a wheel**. Asserts `js` and miniexpr-JIT both match a numpy reference exactly,
then benches a 24-frame `relax` sweep.

```sh
npm i                                         # pulls pyodide@314 (see package.json)
node bench/js-transpiler/dsl-js-node.mjs       # correctness + 24-frame bench
node bench/js-transpiler/dsl-js-node.mjs 48    # N frames
```

Needs network on first run (PyPI wheel via micropip). Exits non-zero on a correctness
mismatch, so it works as a smoke test. Typical output (Apple M-series, blosc2 4.6.0):

```
correctness vs numpy:  js maxdiff=0.00e+0  jit maxdiff=0.00e+0  OK
  jit_backend="js" : ~16 ms/frame
  jit (miniexpr)   : ~31 ms/frame      -> js ~2x faster
  no-JIT           : ~130 ms/frame     -> js ~8x faster
```

> The overlay pins `blosc2==4.6.0` to keep the compiled `blosc2_ext` ABI in step with the
> pure-Python we drop on top. Once these changes ship in a Pyodide-installable wheel, the
> overlay can go away. If the overlay import ever breaks on version skew, overlay all of
> `src/blosc2/*.py` (or bump the pin).

## Browser — `newton-dsl-js.html`

Visual proof in a real browser: transpiles a real `@blosc2.dsl_kernel` under Pyodide, checks
the emitted JS against a numpy reference on the **same** inputs, and times it against a
hand-written JS kernel over the 24-frame sweep (ratio should sit near 1.00 — the transpiler
reaches hand-written-JS speed), then renders the fractal.

```sh
python3 -m http.server          # from the repo root
# open http://localhost:8000/bench/js-transpiler/newton-dsl-js.html  and click Run
```

Serve from the repo root (not `file://`): the page fetches `/src/blosc2/dsl_js.py` (the
local transpiler, newer than the PyPI wheel) at a server-root-absolute path.
