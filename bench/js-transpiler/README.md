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
node bench/js-transpiler/dsl-js-node.mjs       # correctness + kernel sweep, 12 reps
node bench/js-transpiler/dsl-js-node.mjs 24    # N reps
```

Needs network on first run (PyPI wheel via micropip). Exits non-zero on a correctness
mismatch, so it works as a smoke test. It benches four kernel shapes so the js-vs-JIT ratio
can be read against the kernel, not generalized from one. Representative (Apple M2, 4.6.0):

```
  kernel    js     jit    nojit   js/jit  js/nojit
  newton    12.1   26.1   114.6   2.15x    9.44x     arithmetic + branches + early-exit
  deepar    22.5   50.2    53.1   2.23x    2.36x     deep pure-arithmetic loop
  deep     120.1  143.1   104.4   1.19x    0.87x     deep loop, libm sin every iter
  trans      5.0    5.0     6.8   1.00x    1.34x     transcendental-heavy
  poly       3.4    3.0     3.6   0.87x    1.05x     light, branch-free
```

**The takeaway: there is no single "js is N× the JIT" number — it depends on what the kernel
is bottlenecked on.**

- **Arithmetic / control-flow bound** (newton, deepar) → V8's optimizing JIT beats blosc2's
  miniexpr WASM codegen by **~2×**. This is the sweet spot.
- **Transcendental bound** (trans, deep) → **~1×**: time is spent in `sin`/`exp`/`log` (libm),
  which costs about the same whoever runs the loop — `nojit` even edges `js` on `deep`.
- **Light / trivial** (poly) → **<1×**: the kernel does almost no compute, so the blosc2
  pipeline + per-call JS marshaling dominate, and `js` can be *slightly slower* than the JIT.

So the honest generalization is qualitative: transpiling to JS wins (~2×, single-threaded)
for **compute-bound float kernels dominated by arithmetic and control flow**, and is roughly
a wash for transcendental-bound or trivial kernels.

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

## Multithreading ceiling — `worker-pool-bench.mjs`

Throwaway exploration of *how fast the transpiled kernel could go* with real JS
multithreading: pure Node (`worker_threads` + `SharedArrayBuffer`), **no Pyodide, no
blosc2**. Same Newton kernel, partitioned across a persistent worker pool with an Atomics
barrier; reports speedup vs single-thread for 1/2/4/N workers.

```sh
node bench/js-transpiler/worker-pool-bench.mjs
```

Findings (Apple M2, 4 performance + 4 efficiency cores; laptop numbers vary ±10–15% with
thermal/P-vs-E scheduling, so treat these as representative, not exact):

| workers | ms/frame | speedup |
|---|---|---|
| single-thread | ~11.3 | 1.0× |
| ×2 | ~5.9 | ~1.9× (~94% eff) |
| ×4 | ~3.1 | ~3.5× (~88% eff) |
| ×8 | ~2.3 | ~4.8× (~60% eff — E-cores) |

- The worker mechanism is ~free (×1 ≈ 1.0×); scaling is near-linear up to the performance
  core count. The ×8 drop-off is the M2's efficiency cores, not overhead.
- **Load balancing is essential.** Contiguous row-bands regress badly (×4 fell to 1.48×)
  because the per-pixel early-`break` makes some bands all-max-iter and others trivial.
  **Striped** rows (worker `i` → rows `i, i+nw, …`) fix it — that's what the bench uses.

Why this stays a *headroom* result, not a shipped feature: it measures **pure compute**.
The real `jit_backend="js"` path also pays ~8 ms/frame of blosc2 decompress/compress that
does not parallelize this way, plus Pyodide orchestration is single-threaded — so realistic
end-to-end gain is a fraction of 5×. And a browser integration needs pure-JS workers (not the
Pyodide bridge), a SharedArrayBuffer, COOP/COEP cross-origin isolation, and a path to get
decompressed chunks into shared memory. See "Deferred" in [`plans/dsl-js.md`](../../plans/dsl-js.md).
