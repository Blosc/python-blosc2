// Headless integration test + perf bench for the DSL->JS backend (jit_backend="js"),
// using Pyodide-in-Node. Installs the blosc2 wasm wheel from PyPI, then OVERLAYS this
// working tree's pure-Python (src/blosc2/dsl_js.py + lazyexpr.py) on top of it before
// importing blosc2 -- so the wired path runs without waiting for a new wheel.
//
// Benches a spread of kernel shapes to show how the js-vs-JIT ratio depends on the kernel:
// branchy + early-exit (newton), branch-free light (poly), transcendental-heavy (trans),
// deep no-exit loop (deep). Reports js / jit / no-jit per kernel.
//
//   npm i                                         # pulls pyodide@314 (see package.json)
//   node bench/js-transpiler/dsl-js-node.mjs       # correctness + bench, 12 reps
//   node bench/js-transpiler/dsl-js-node.mjs 24    # N reps
import { loadPyodide } from "pyodide";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

// Resolve paths from this file, so the harness runs from any CWD (repo root is ../../).
const ROOT = fileURLToPath(new URL("../../", import.meta.url));
const NFRAMES = Number(process.argv[2]) || 12;

// Kernels + bench live in a real module file: @blosc2.dsl_kernel runs inspect.getsource(),
// which needs the function to be file-backed (not exec'd from a string).
const PYSRC = String.raw`
import json, time
import numpy as np
import blosc2

WIDTH, HEIGHT, MAXITER = 320, 213, 48
SPANX = 3.4
ASPECT = HEIGHT / WIDTH
DTYPE = np.float64

# --- kernels spanning the cost/control-flow spectrum -------------------------------------
@blosc2.dsl_kernel  # branchy, deep, per-pixel early exit
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

@blosc2.dsl_kernel  # light, branch-free, vectorizable arithmetic
def poly_dsl(a, b):
    a2 = a * a
    b2 = b * b
    return a2 * a - 3.0 * a * b2 + 2.0 * b2 * b - a + 0.5 * b

@blosc2.dsl_kernel  # transcendental-heavy (exercises each engine's libm). miniexpr wants
def trans_dsl(a, b):  # bare sin/cos/... (not np.sin); the transpiler maps both to Math.*
    msq = a * a + b * b
    sc = sin(a * 3.0) * cos(b * 2.0)
    ex = exp(msq * -0.5)
    return sc + ex + sqrt(msq + 1.0) + log(msq + 2.0)

@blosc2.dsl_kernel  # deep fixed loop, transcendental-bound (libm sin every iter)
def deep_dsl(a, b):
    acc = a
    for k in range(64):
        acc = acc * 0.99 + sin(acc + b)
    return acc

@blosc2.dsl_kernel  # deep fixed loop, pure arithmetic (no libm, no branches); contractive
def deepar_dsl(a, b):
    acc = a * 0.1
    t = b * 0.1
    for k in range(64):
        t = t * 0.5 - acc * 0.25 + 0.1
        acc = acc * 0.5 + t * 0.25 + 0.1
    return acc + t

_x = np.linspace(-SPANX / 2, SPANX / 2, WIDTH, dtype=DTYPE)
_y = np.linspace(-SPANX * ASPECT / 2, SPANX * ASPECT / 2, HEIGHT, dtype=DTYPE)
A_NP, B_NP = np.meshgrid(_x, _y)
_chunks = (min(100, HEIGHT), min(150, WIDTH))
_blocks = (max(1, _chunks[0] // 4), max(1, _chunks[1] // 3))
_cp = blosc2.CParams(codec=blosc2.Codec.LZ4, clevel=1)
A_B2 = blosc2.asarray(A_NP, chunks=_chunks, blocks=_blocks, cparams=_cp)
B_B2 = blosc2.asarray(B_NP, chunks=_chunks, blocks=_blocks, cparams=_cp)

# (name, kernel, operand tuple). Fixed inputs -> each rep does identical work.
KERNELS = [
    ("newton", newton_dsl, (A_B2, B_B2, MAXITER, 1.37)),
    ("poly",   poly_dsl,   (A_B2, B_B2)),
    ("trans",  trans_dsl,  (A_B2, B_B2)),
    ("deep",   deep_dsl,   (A_B2, B_B2)),
    ("deepar", deepar_dsl, (A_B2, B_B2)),
]

def run(func, ops, backend):
    kw = {"dtype": DTYPE, "cparams": _cp}
    if backend == "js":
        kw["jit_backend"] = "js"
    elif backend == "jit":
        kw["jit"] = True
    else:
        kw["jit"] = False
    return blosc2.lazyudf(func, ops, **kw)[:]

def bench(func, ops, backend, reps):
    run(func, ops, backend)            # warm
    best = float("inf")
    for _ in range(3):
        t = time.perf_counter()
        for _ in range(reps):
            run(func, ops, backend)
        best = min(best, (time.perf_counter() - t) * 1000 / reps)
    return best

def debug_bridge():
    import blosc2.dsl_js as dj
    bridge = dj.js_kernel(newton_dsl)
    out = np.empty((HEIGHT, WIDTH), dtype=DTYPE)
    inp = (A_NP, B_NP, MAXITER, 1.37)
    t = time.perf_counter(); bridge(inp, out, 0); first = (time.perf_counter() - t) * 1000
    best = float("inf")
    for _ in range(8):
        t = time.perf_counter(); bridge(inp, out, 0); best = min(best, (time.perf_counter() - t) * 1000)
    return {"first_ms": first, "warm_ms": best}

def result(reps):
    import math
    kernels = []
    for name, func, ops in KERNELS:
        rj = run(func, ops, "js")
        rjit = run(func, ops, "jit")
        diff = float(np.max(np.abs(rj - rjit)))
        diff = diff if math.isfinite(diff) else 1e30   # keep JSON valid; flags as mismatch
        ms = {b: bench(func, ops, b, reps) for b in ("js", "jit", "nojit")}
        kernels.append({"name": name, "ms": ms, "diff": diff})
    return json.dumps({"kernels": kernels, "bridge": debug_bridge(), "reps": reps})
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
            "| Pyodide", py.version, "| reps", NFRAMES);

py.FS.writeFile("/kernel_bench.py", new TextEncoder().encode(PYSRC));
const out = await py.runPythonAsync(`
import sys
if "/" not in sys.path: sys.path.insert(0, "/")
import kernel_bench
kernel_bench.result(${NFRAMES})
`);

const r = JSON.parse(out);
const fmt = (x, w) => String(x).padStart(w);
const bad = r.kernels.filter((k) => k.diff > 1e-5);
console.log(`\ncorrectness (js vs JIT maxdiff): ${bad.length ? "MISMATCH " + bad.map((k) => k.name) : "OK"}`);
console.log("\nper-kernel bench (ms/frame, lower is better):");
console.log("  kernel    js     jit    nojit   js/jit  js/nojit  diff");
for (const k of r.kernels) {
  const { js, jit, nojit } = k.ms;
  console.log(
    `  ${k.name.padEnd(7)} ${fmt(js.toFixed(1), 6)} ${fmt(jit.toFixed(1), 6)} ${fmt(nojit.toFixed(1), 7)}` +
    `  ${fmt((jit / js).toFixed(2) + "x", 6)} ${fmt((nojit / js).toFixed(2) + "x", 8)}  ${k.diff.toExponential(1)}`,
  );
}
console.log(`\nnewton bridge probe (no blosc2 machinery): first=${r.bridge.first_ms.toFixed(1)} ms  warm=${r.bridge.warm_ms.toFixed(1)} ms`);
if (bad.length) process.exit(1);
