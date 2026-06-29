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

@blosc2.dsl_kernel  # P1: index/shape symbols -> per-element global coords (radial gradient)
def idxgrad_dsl(a):
    dx = float(_i0) - _n0 * 0.5  # noqa: F821
    dy = float(_i1) - _n1 * 0.5  # noqa: F821
    return a + sqrt(dx * dx + dy * dy)  # noqa: F821

@blosc2.dsl_kernel  # P2: integer inputs, float output (the bridge float64-converts the operands)
def intmix_dsl(a, b):
    return sqrt(a * a + b * b) * 0.25 + (a - b)  # noqa: F821

# Path-coverage kernels (used only by path_check, not the float sweep):
@blosc2.dsl_kernel  # int *output* -> default must fall back to miniexpr (float64 bridge unsafe)
def int_dsl(a, b):
    return a * 2 + b * 3

@blosc2.dsl_kernel  # int *inputs*, float output -> JS is safe (bridge float64-converts operands)
def intin_dsl(a, b):
    return (a + b) * 0.5

@blosc2.dsl_kernel  # index/shape symbols -> JS reconstructs global coords per block
def idx_dsl(a):
    return a + float(_i0)  # noqa: F821

@blosc2.dsl_kernel  # expm1() is valid DSL/miniexpr but outside the JS Math.* set -> falls back
def unsup_dsl(a, b):
    return expm1(a + b) * 0.5  # noqa: F821

_x = np.linspace(-SPANX / 2, SPANX / 2, WIDTH, dtype=DTYPE)
_y = np.linspace(-SPANX * ASPECT / 2, SPANX * ASPECT / 2, HEIGHT, dtype=DTYPE)
A_NP, B_NP = np.meshgrid(_x, _y)
_chunks = (min(100, HEIGHT), min(150, WIDTH))
_blocks = (max(1, _chunks[0] // 4), max(1, _chunks[1] // 3))
_cp = blosc2.CParams(codec=blosc2.Codec.LZ4, clevel=1)
A_B2 = blosc2.asarray(A_NP, chunks=_chunks, blocks=_blocks, cparams=_cp)
B_B2 = blosc2.asarray(B_NP, chunks=_chunks, blocks=_blocks, cparams=_cp)
# Small-magnitude integer operands for the P2 (int in / float out) kernels.
AI_NP = (A_NP * 10).astype(np.int64)
BI_NP = (B_NP * 10).astype(np.int64)
AI_B2 = blosc2.asarray(AI_NP, chunks=_chunks, blocks=_blocks, cparams=_cp)
BI_B2 = blosc2.asarray(BI_NP, chunks=_chunks, blocks=_blocks, cparams=_cp)

# (name, kernel, operand tuple). Fixed inputs -> each rep does identical work.
KERNELS = [
    ("newton", newton_dsl, (A_B2, B_B2, MAXITER, 1.37)),
    ("poly",   poly_dsl,   (A_B2, B_B2)),
    ("trans",  trans_dsl,  (A_B2, B_B2)),
    ("deep",   deep_dsl,   (A_B2, B_B2)),
    ("deepar", deepar_dsl, (A_B2, B_B2)),
    ("idxgrad", idxgrad_dsl, (A_B2,)),         # P1: index/shape symbols (float out)
    ("intmix",  intmix_dsl,  (AI_B2, BI_B2)),  # P2: integer inputs, float out
]

def run(func, ops, backend, dtype=DTYPE):
    kw = {"dtype": dtype, "cparams": _cp}
    if backend == "js":
        kw["jit_backend"] = "js"
    elif backend == "tcc":
        kw["jit"] = True
        kw["jit_backend"] = "tcc"   # miniexpr JIT, TinyCC backend (explicit)
    elif backend == "nojit":
        kw["jit"] = False
    # "default": pass nothing -> under WASM this prefers js, falling back to miniexpr.
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

def path_check():
    # The default backend must agree with miniexpr (tcc) on every kernel, whether it runs
    # the kernel through JS (index symbols, int inputs+float out) or transparently falls back
    # to miniexpr where JS can't go (int output, unsupported constructs) -- with no error.
    int_def = run(int_dsl, (AI_B2, BI_B2), "default", dtype=np.int64)  # -> falls back to miniexpr
    int_tcc = run(int_dsl, (AI_B2, BI_B2), "tcc", dtype=np.int64)
    intin_def = run(intin_dsl, (AI_B2, BI_B2), "default")              # -> JS (int in, float out)
    intin_tcc = run(intin_dsl, (AI_B2, BI_B2), "tcc")
    idx_def = run(idx_dsl, (A_B2,), "default")                        # -> JS (index symbols)
    idx_tcc = run(idx_dsl, (A_B2,), "tcc")
    unsup_def = run(unsup_dsl, (A_B2, B_B2), "default")               # -> falls back to miniexpr
    unsup_tcc = run(unsup_dsl, (A_B2, B_B2), "tcc")
    return {
        "int_ok": bool(np.array_equal(int_def, int_tcc)),
        "intin_ok": bool(np.allclose(intin_def, intin_tcc)),
        "idx_ok": bool(np.allclose(idx_def, idx_tcc)),
        "unsup_ok": bool(np.allclose(unsup_def, unsup_tcc)),
    }

def kernel_names():
    return json.dumps([name for name, _f, _o in KERNELS])

def bench_kernel(i, reps):
    # One kernel at a time so the driver can print each row as soon as it is computed.
    import math
    name, func, ops = KERNELS[i]
    rj = run(func, ops, "js")
    rtcc = run(func, ops, "tcc")
    diff = float(np.max(np.abs(rj - rtcc)))
    diff = diff if math.isfinite(diff) else 1e30   # keep JSON valid; flags as mismatch
    ms = {b: bench(func, ops, b, reps) for b in ("default", "js", "tcc", "nojit")}
    return json.dumps({"name": name, "ms": ms, "diff": diff})

def summary():
    return json.dumps({"bridge": debug_bridge(), "paths": path_check()})
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
await py.runPythonAsync(`
import sys
if "/" not in sys.path: sys.path.insert(0, "/")
import kernel_bench
`);

const fmt = (x, w) => String(x).padStart(w);
const names = JSON.parse(await py.runPythonAsync("kernel_bench.kernel_names()"));

// Stream the table: print each row as soon as its kernel finishes benchmarking.
console.log("\nper-kernel bench (ms/frame, lower is better; 'default' = prefer-js w/ fallback,");
console.log("'tcc' = miniexpr JIT, 'nojit' = miniexpr interpreter):");
const cols = ["default", "js", "tcc", "nojit", "js/tcc"];
console.log("  " + "kernel".padEnd(8) + cols.map((c) => fmt(c, 8)).join(""));
const bad = [];
for (let i = 0; i < names.length; i++) {
  const k = JSON.parse(await py.runPythonAsync(`kernel_bench.bench_kernel(${i}, ${NFRAMES})`));
  const { default: def, js, tcc, nojit } = k.ms;
  const cells = [def, js, tcc, nojit].map((v) => v.toFixed(1)).concat((tcc / js).toFixed(2) + "x");
  console.log("  " + k.name.padEnd(8) + cells.map((c) => fmt(c, 8)).join(""));
  if (k.diff > 1e-5) bad.push(k.name);
}

const s = JSON.parse(await py.runPythonAsync("kernel_bench.summary()"));
console.log(`\ncorrectness (js vs tcc maxdiff): ${bad.length ? "MISMATCH " + bad : "OK"}`);
const fb = s.paths;
const fbOk = fb.int_ok && fb.intin_ok && fb.idx_ok && fb.unsup_ok;
console.log(`default backend (no jit_backend) vs miniexpr: ` +
            `int-out=${fb.int_ok ? "ok" : "FAIL"} int-in=${fb.intin_ok ? "ok" : "FAIL"} ` +
            `index-symbol=${fb.idx_ok ? "ok" : "FAIL"} unsupported=${fb.unsup_ok ? "ok" : "FAIL"}` +
            `  -> ${fbOk ? "all paths agree" : "BROKEN"}`);
console.log(`\nnewton bridge probe (no blosc2 machinery): first=${s.bridge.first_ms.toFixed(1)} ms  warm=${s.bridge.warm_ms.toFixed(1)} ms`);
if (bad.length || !fbOk) process.exit(1);
