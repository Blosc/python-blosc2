// Headless integration test + perf bench for the DSL->JS backend (jit_backend="js"),
// using Pyodide-in-Node. Installs the blosc2 wasm wheel from PyPI, then OVERLAYS this
// working tree's pure-Python (src/blosc2/dsl_js.py + lazyexpr.py) on top of it before
// importing blosc2 -- so the wired path runs without waiting for a new wheel.
//
//   npm i                                         # pulls pyodide@314 (see package.json)
//   node bench/js-transpiler/dsl-js-node.mjs      # correctness + 24-frame bench
//   node bench/js-transpiler/dsl-js-node.mjs 48   # bench with N frames
import { loadPyodide } from "pyodide";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

// Resolve paths from this file, so the harness runs from any CWD (repo root is ../../).
const ROOT = fileURLToPath(new URL("../../", import.meta.url));
const NFRAMES = Number(process.argv[2]) || 24;

// Kernel + bench live in a real module file: @blosc2.dsl_kernel runs inspect.getsource(),
// which needs the function to be file-backed (not exec'd from a string).
const PYSRC = String.raw`
import json, time
import numpy as np
import blosc2

WIDTH, HEIGHT, MAXITER = 320, 213, 48
SPANX = 3.4
ASPECT = HEIGHT / WIDTH
DTYPE = np.float64

@blosc2.dsl_kernel
def newton_dsl(a, b, max_iter, relax):
    za = a
    zb = b
    mif = float(max_iter)
    it = mif
    for k in range(max_iter):
        a2 = za * za
        b2 = zb * zb
        fr = za * a2 - 3.0 * za * b2 - 1.0
        fi = 3.0 * a2 * zb - zb * b2
        dr = 3.0 * (a2 - b2)
        di = 6.0 * za * zb
        den = dr * dr + di * di + 0.000000000001
        qr = relax * (fr * dr + fi * di) / den
        qi = relax * (fi * dr - fr * di) / den
        za = za - qr
        zb = zb - qi
        if qr * qr + qi * qi < 0.000001:
            it = float(k)
            break
    d0 = (za - 1.0) * (za - 1.0) + zb * zb
    d1 = (za + 0.5) * (za + 0.5) + (zb - 0.8660254) * (zb - 0.8660254)
    d2 = (za + 0.5) * (za + 0.5) + (zb + 0.8660254) * (zb + 0.8660254)
    root = 0.0
    md = d0
    if d1 < md:
        md = d1
        root = 1.0
    if d2 < md:
        root = 2.0
    return root + 0.9 * (it / mif)

def newton_numpy(a, b, max_iter, relax):
    za = a.copy(); zb = b.copy()
    mif = float(max_iter)
    it = np.full(a.shape, mif)
    alive = np.ones(a.shape, dtype=bool)
    for k in range(max_iter):
        a2 = za * za; b2 = zb * zb
        fr = za * a2 - 3.0 * za * b2 - 1.0; fi = 3.0 * a2 * zb - zb * b2
        dr = 3.0 * (a2 - b2); di = 6.0 * za * zb; den = dr * dr + di * di + 1e-12
        qr = relax * (fr * dr + fi * di) / den; qi = relax * (fi * dr - fr * di) / den
        za = za - qr; zb = zb - qi
        c = alive & ((qr * qr + qi * qi) < 1e-6); it[c] = k; alive &= ~c
        if not alive.any():
            break
    d0 = (za - 1.0) ** 2 + zb ** 2
    d1 = (za + 0.5) ** 2 + (zb - 0.8660254) ** 2
    d2 = (za + 0.5) ** 2 + (zb + 0.8660254) ** 2
    root = np.zeros(a.shape); md = d0.copy()
    m = d1 < md; md = np.where(m, d1, md); root = np.where(m, 1.0, root)
    m = d2 < md; root = np.where(m, 2.0, root)
    return root + 0.9 * (it / mif)

_x = np.linspace(-SPANX / 2, SPANX / 2, WIDTH, dtype=DTYPE)
_y = np.linspace(-SPANX * ASPECT / 2, SPANX * ASPECT / 2, HEIGHT, dtype=DTYPE)
A_NP, B_NP = np.meshgrid(_x, _y)
_chunks = (min(100, HEIGHT), min(150, WIDTH))
_blocks = (max(1, _chunks[0] // 4), max(1, _chunks[1] // 3))
_cp = blosc2.CParams(codec=blosc2.Codec.LZ4, clevel=1)
A_B2 = blosc2.asarray(A_NP, chunks=_chunks, blocks=_blocks, cparams=_cp)
B_B2 = blosc2.asarray(B_NP, chunks=_chunks, blocks=_blocks, cparams=_cp)

def run(relax, backend):
    kw = {"dtype": DTYPE, "cparams": _cp}
    if backend == "js":
        kw["jit_backend"] = "js"
    elif backend == "jit":
        kw["jit"] = True
    else:
        kw["jit"] = False
    return blosc2.lazyudf(newton_dsl, (A_B2, B_B2, MAXITER, relax), **kw)[:]

def correctness():
    rx = 1.37
    ref = newton_numpy(A_NP, B_NP, MAXITER, rx)
    return {
        "diff_js": float(np.max(np.abs(run(rx, "js") - ref))),
        "diff_jit": float(np.max(np.abs(run(rx, "jit") - ref))),
    }

def bench(nframes):
    relaxes = [float(v) for v in np.linspace(1.0, 1.85, nframes)]
    def sweep_ms(backend):
        for r in relaxes[:2]:
            run(r, backend)            # warm
        best = float("inf")
        for _ in range(3):
            t = time.perf_counter()
            for r in relaxes:
                run(r, backend)
            best = min(best, (time.perf_counter() - t) * 1000)
        return best
    return {b: sweep_ms(b) for b in ("js", "jit", "nojit")}

def debug_bridge():
    # Call the JS bridge directly (no lazyudf/blosc2 machinery) to isolate marshal+compile+compute.
    import blosc2.dsl_js as dj
    bridge = dj.js_kernel(newton_dsl)
    out = np.empty((HEIGHT, WIDTH), dtype=DTYPE)
    inp = (A_NP, B_NP, MAXITER, 1.37)
    t = time.perf_counter(); bridge(inp, out, 0); first = (time.perf_counter() - t) * 1000
    best = float("inf")
    for _ in range(8):
        t = time.perf_counter(); bridge(inp, out, 0); best = min(best, (time.perf_counter() - t) * 1000)
    return {"first_ms": first, "warm_ms": best}

def result(nframes):
    out = correctness()
    out["ms"] = bench(nframes)
    out["bridge"] = debug_bridge()
    out["nframes"] = nframes
    return json.dumps(out)
`;

const py = await loadPyodide();
await py.loadPackage("micropip");
// Pin to the release this tree is based on, to keep the C-extension ABI in step with the
// pure-Python we overlay. The compiled blosc2_ext comes from the wheel; only .py is ours.
await py.runPythonAsync(`import micropip; await micropip.install("blosc2==4.6.0")`);

// find_spec does NOT import blosc2 -- so we can patch files before first import.
const pkgdir = await py.runPythonAsync(
  `import importlib.util, os; os.path.dirname(importlib.util.find_spec("blosc2").origin)`,
);
for (const f of ["dsl_js.py", "lazyexpr.py"]) {
  py.FS.writeFile(`${pkgdir}/${f}`, readFileSync(`${ROOT}src/blosc2/${f}`));
}
await py.runPythonAsync(`
import sys, blosc2
assert hasattr(sys.modules["blosc2.lazyexpr"], "_as_js_udf"), "overlay did not take"
`);
console.log("blosc2", await py.runPythonAsync("blosc2.__version__"),
            "| Pyodide", py.version, "| frames", NFRAMES);

py.FS.writeFile("/newton_bench.py", new TextEncoder().encode(PYSRC));
const out = await py.runPythonAsync(`
import sys
if "/" not in sys.path: sys.path.insert(0, "/")
import newton_bench
newton_bench.result(${NFRAMES})
`);

const r = JSON.parse(out);
const ok = r.diff_js < 1e-9 && r.diff_jit < 1e-9;
console.log(`\ncorrectness vs numpy:  js maxdiff=${r.diff_js.toExponential(2)}  ` +
            `jit maxdiff=${r.diff_jit.toExponential(2)}  ${ok ? "OK" : "MISMATCH"}`);
const per = ms => `${ms.toFixed(0)} ms total (${(ms / r.nframes).toFixed(2)} ms/frame)`;
console.log(`\nperf (${r.nframes}-frame relax sweep, best of 3):`);
console.log(`  jit_backend="js" : ${per(r.ms.js)}`);
console.log(`  jit (miniexpr)   : ${per(r.ms.jit)}`);
console.log(`  no-JIT           : ${per(r.ms.nojit)}`);
console.log(`  js vs jit  : ${(r.ms.jit / r.ms.js).toFixed(2)}x   ` +
            `js vs no-jit : ${(r.ms.nojit / r.ms.js).toFixed(2)}x`);
console.log(`\nbridge probe (direct call, no blosc2 machinery):  ` +
            `first=${r.bridge.first_ms.toFixed(1)} ms  warm=${r.bridge.warm_ms.toFixed(1)} ms`);
if (!ok) process.exit(1);
