#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Tip 15: @blosc2.jit with control flow. The same Mandelbrot escape-time
# computation run four ways -- a plain Python per-element loop, a vectorized
# NumPy mask iteration, @blosc2.jit (auto-detects the loop/break and compiles
# the whole kernel as DSL), and @blosc2.jit(strict=True) (forces that route) --
# plus the elementwise contrast that explains why functions *without* control
# flow still trace: tracing is faster than the forced DSL route for pure
# elementwise expressions (measured against plain NumPy too, as the scale
# anchor for all three jit routes).  A final pair measures the same DSL kernels
# compiled with jit_backend="cc" (system C compiler) instead of the bundled tcc.
#
# Peak memory is the same story for every variant here (the result array plus
# a few temporaries), so the plot is time-only. The script prints a
# correctness check of every mode against the pure-Python reference, and
# shows that strict=False cannot even trace this function: the loop condition
# depends on the traced arrays, so tracing raises instead of recording one
# path for every element.

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from common import COLOR_NAIVE, COLOR_TIP, GRID, INK, MUTED, OUT_DIR, measure

import blosc2

W = H = 512
MAX_ITER = 64

# The grid. float32, matching what a real image pipeline would use.
_ys = np.linspace(-1.2, 1.2, H, dtype=np.float32)
_xs = np.linspace(-2.0, 0.6, W, dtype=np.float32)
CR, CI = np.meshgrid(_xs, _ys)


def mandel_py(cr, ci, max_iter):
    """Reference: one Python loop per pixel. Correct but slow."""
    zr = zi = 0.0
    n = 0
    while zr * zr + zi * zi <= 4.0 and n < max_iter:
        zr, zi = zr * zr - zi * zi + cr, 2 * zr * zi + ci
        n += 1
    return n


def mandel_numpy(cr, ci, max_iter):
    """Vectorized alternative without jit: one masked iteration per step.

    The fiddly bits are exactly what the jit kernel below gets for free: the
    alive/escaped bookkeeping, and the overflow trap (zr/zi keep growing after
    escape and go inf/nan in float32, so the mask loop must not rely on their
    values -- and nan would keep `alive` true forever if the bookkeeping ever
    missed an element).
    """
    zr = np.zeros_like(cr)
    zi = np.zeros_like(cr)
    out = np.zeros(cr.shape, dtype=np.int32)
    alive = np.ones(cr.shape, dtype=bool)
    with np.errstate(over="ignore", invalid="ignore"):
        for k in range(max_iter):
            zr2 = zr * zr - zi * zi + cr
            zi2 = 2 * zr * zi + ci
            zr, zi = zr2, zi2
            escaped = (zr * zr + zi * zi) > 4.0
            newly = escaped & alive
            out[newly] = k + 1
            alive &= ~escaped
            if not alive.any():
                break
    out[alive] = max_iter
    return out


@blosc2.jit
def mandel_jit(cr, ci, max_iter):
    # Same computation, written the natural way. jit detects the control flow
    # at decoration time and compiles the whole function as a DSL kernel.
    # DSL-form rules: simple assignments only (a tuple assignment like
    # `zr, zi = ...` silently falls back to tracing, which then raises when
    # the branch is reached), and no docstring inside the kernel body.
    zr = 0.0
    zi = 0.0
    n = 0
    for _ in range(max_iter):
        if zr * zr + zi * zi > 4.0:
            break
        zr2 = zr * zr - zi * zi + cr
        zi = 2 * zr * zi + ci
        zr = zr2
        n += 1
    return n


@blosc2.jit(strict=True)
def mandel_strict(cr, ci, max_iter):
    zr = 0.0
    zi = 0.0
    n = 0
    for _ in range(max_iter):
        if zr * zr + zi * zi > 4.0:
            break
        zr2 = zr * zr - zi * zi + cr
        zi = 2 * zr * zi + ci
        zr = zr2
        n += 1
    return n


@blosc2.jit(jit_backend="cc")
def mandel_cc(cr, ci, max_iter):
    # Same kernel, but compiled with the system C compiler (clang/gcc) instead
    # of the bundled tcc: slower one-time compile, faster generated code.
    zr = 0.0
    zi = 0.0
    n = 0
    for _ in range(max_iter):
        if zr * zr + zi * zi > 4.0:
            break
        zr2 = zr * zr - zi * zi + cr
        zi = 2 * zr * zi + ci
        zr = zr2
        n += 1
    return n


def py_loop():
    return np.array(
        [[mandel_py(CR[y, x], CI[y, x], MAX_ITER) for x in range(W)] for y in range(H)],
        dtype=np.int32,
    )


def numpy_masked():
    return mandel_numpy(CR, CI, MAX_ITER)


def jit_default():
    return mandel_jit(CR, CI, MAX_ITER)


def jit_strict():
    return mandel_strict(CR, CI, MAX_ITER)


def jit_cc():
    return mandel_cc(CR, CI, MAX_ITER)


# --- Elementwise contrast: why functions without control flow still trace ---

X = np.random.default_rng(0).random(8_000_000, dtype=np.float32)


@blosc2.jit
def elementwise(x):
    return (
        np.sin(x)
        + np.cos(x * 2)
        + np.exp(x * 0.5) * np.sin(x * 3)
        + np.sqrt(np.abs(x))
        + np.log1p(np.abs(x))
    )


@blosc2.jit(strict=True)
def elementwise_dsl(x):
    return (
        np.sin(x)
        + np.cos(x * 2)
        + np.exp(x * 0.5) * np.sin(x * 3)
        + np.sqrt(np.abs(x))
        + np.log1p(np.abs(x))
    )


@blosc2.jit(strict=True, jit_backend="cc")
def elementwise_dsl_cc(x):
    return (
        np.sin(x)
        + np.cos(x * 2)
        + np.exp(x * 0.5) * np.sin(x * 3)
        + np.sqrt(np.abs(x))
        + np.log1p(np.abs(x))
    )


def numpy_route():
    # The same expression in plain NumPy: the scale anchor for the three jit
    # routes below, and the first bar of the elementwise plot.
    x = X
    return (
        np.sin(x)
        + np.cos(x * 2)
        + np.exp(x * 0.5) * np.sin(x * 3)
        + np.sqrt(np.abs(x))
        + np.log1p(np.abs(x))
    )


def trace_route():
    return elementwise(X)


def dsl_route():
    return elementwise_dsl(X)


def dsl_route_cc():
    return elementwise_dsl_cc(X)


# --- One-time compile cost -------------------------------------------------
#
# The timings above are steady state: measure() reports the best of three
# calls, so the compile that happens on the *first* call is excluded. This
# section measures that first call instead -- the penalty a user actually pays
# -- as (first call) - (best of the five that follow).
#
# The two backends amortize it very differently, which is why "cold" and "warm"
# are measured separately. tcc compiles in memory, every process, every time.
# cc writes a shared object into $TMPDIR/miniexpr-jit keyed by a fingerprint of
# the kernel IR + dtypes + toolchain, so only the first process on a machine
# pays the compiler; later ones dlopen the cached artifact. Each measurement
# therefore runs in a fresh subprocess with TMPDIR pointed at a scratch dir
# that is either wiped first (cold) or kept from the previous run (warm) -- and
# never at the user's real cache, which the steady-state timings above rely on.
_COMPILE_DRIVER = """\
import importlib.util, json, os, sys, time
sys.path.insert(0, os.path.dirname(sys.argv[1]))
spec = importlib.util.spec_from_file_location("_bench_mod", sys.argv[1])
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)  # module-level setup, and the @jit decorators
fn = getattr(mod, sys.argv[2])
t0 = time.perf_counter()
fn()
first = time.perf_counter() - t0
steady = float("inf")
for _ in range(5):
    t0 = time.perf_counter()
    fn()
    steady = min(steady, time.perf_counter() - t0)
print(json.dumps({"first": first, "steady": steady}))
"""


def measure_compile(func_name, cache_dir, cold, reps=3):
    """First-call penalty of func_name(), in a fresh process with its own JIT cache.

    Best of `reps` processes, for the same reason common.measure() takes the best
    of three calls: the very first cc compile of a session runs ~1.5x slower than
    the settled one (the compiler binary itself is cold), and that is an artifact
    of the harness, not of the backend.
    """
    best = float("inf")
    for _ in range(reps):
        if cold:
            shutil.rmtree(cache_dir, ignore_errors=True)
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(
            [sys.executable, "-c", _COMPILE_DRIVER, str(Path(__file__).resolve()), func_name],
            capture_output=True,
            text=True,
            env={**os.environ, "TMPDIR": str(cache_dir)},
        )
        if proc.returncode != 0:
            raise RuntimeError(f"{func_name} failed:\n{proc.stderr}")
        data = json.loads(proc.stdout.strip().splitlines()[-1])
        best = min(best, data["first"] - data["steady"])
    return best


def bars(ax, title, labels, values, fmt, log=False, colors=None):
    """One cluster of direct-labeled bars, tip-14 style.

    `colors`: per-bar colors; default is the 2-bar naive/tip convention
    (first two bars naive-blue, the rest tip-aqua).
    """
    x = np.arange(len(labels), dtype=float)
    top = max(values)
    bottom = min(values)
    for i, h in enumerate(values):
        color = colors[i] if colors is not None else (COLOR_NAIVE if i < 2 else COLOR_TIP)
        ax.bar(i, h, width=0.55, color=color)
        y = h * 1.12 if log else h + top * 0.03
        ax.text(i, y, fmt(h), ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_title(title, fontsize=9.5, color=INK)
    ax.set_ylabel("Time (s, log scale)" if log else "Time (s)", color=INK, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, labelleft=False)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_ylim(*((bottom / 3, top * 8) if log else (0, top * 1.5)))
    if log:
        ax.set_yscale("log")


def save(fig, name):
    fig.tight_layout()
    out_path = OUT_DIR / name
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"plot saved to {out_path}")


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # --- Correctness: every mode against the pure-Python reference ---
    ref = py_loop()
    for name, fn in (
        ("numpy_masked", numpy_masked),
        ("jit_default", jit_default),
        ("jit_strict", jit_strict),
        ("jit_cc", jit_cc),
    ):
        ok = np.array_equal(fn(), ref)
        print(f"{name:<13} correct: {ok}")
        assert ok, name

    @blosc2.jit(strict=False)
    def mandel_trace(cr, ci, max_iter):
        zr = 0.0
        zi = 0.0
        n = 0
        for _ in range(max_iter):
            if zr * zr + zi * zi > 4.0:
                break
            zr2 = zr * zr - zi * zi + cr
            zi = 2 * zr * zi + ci
            zr = zr2
            n += 1
        return n

    try:
        mandel_trace(CR, CI, MAX_ITER)
        raise AssertionError("strict=False should not be able to trace this function")
    except ValueError as e:
        print("strict=False raises at call time as expected:", str(e)[:70], "...")

    # Same expression, two engines: results agree to float32 precision (1 ulp)
    # even though trace and DSL associate the arithmetic differently.
    eq = np.allclose(elementwise(X), elementwise_dsl(X), rtol=1e-6, atol=1e-7)
    print(f"elementwise trace/dsl agree: {eq}")
    assert eq
    eq_cc = np.allclose(elementwise_dsl_cc(X), elementwise_dsl(X), rtol=1e-6, atol=1e-7)
    print(f"elementwise tcc/cc agree: {eq_cc}")
    assert eq_cc

    # --- Timings ---
    mandel_names = ("py_loop", "numpy_masked", "jit_default", "jit_strict", "jit_cc")
    t = {}
    for name in mandel_names + ("numpy_route", "trace_route", "dsl_route", "dsl_route_cc"):
        t[name], rss = measure(__file__, name)
        print(f"{name:<13} {t[name]:8.4f}s   peak {rss / 1e6:6.1f} MB")

    print("\nmandelbrot speedups vs py_loop:")
    for name in mandel_names[1:]:
        print(f"  {name:<13} {t['py_loop'] / t[name]:6.1f}x faster than py_loop")
    print(f"  elementwise: trace {t['dsl_route'] / t['trace_route']:.2f}x faster than forced DSL")

    print("\nelementwise vs plain NumPy:")
    for name in ("trace_route", "dsl_route", "dsl_route_cc"):
        print(f"  {name:<13} {t['numpy_route'] / t[name]:.2f}x faster than numpy_route")
    print("\njit_backend='cc' vs default tcc (steady state):")
    print(f"  mandel:      {t['jit_default'] / t['jit_cc']:.2f}x faster with cc")
    print(f"  elementwise: {t['dsl_route'] / t['dsl_route_cc']:.2f}x faster with cc")

    # --- One-time compile cost (first call minus steady state) ---
    print("\none-time compile cost (first call - steady state), fresh process:")
    with tempfile.TemporaryDirectory(prefix="tip15-jitcache-") as cache_dir:
        c = {}
        for label, func_name in (
            ("mandel  tcc", "jit_default"),
            ("mandel  cc ", "jit_cc"),
            ("elemwise tcc", "dsl_route"),
            ("elemwise cc ", "dsl_route_cc"),
        ):
            cold = measure_compile(func_name, cache_dir, cold=True)
            warm = measure_compile(func_name, cache_dir, cold=False)
            c[func_name] = (cold, warm)
            print(f"  {label}   cold {cold * 1000:7.1f} ms   warm cache {warm * 1000:6.1f} ms")
    # How many calls it takes for cc's steady-state win to repay its cold compile.
    for label, tcc_name, cc_name in (
        ("mandel", "jit_default", "jit_cc"),
        ("elementwise", "dsl_route", "dsl_route_cc"),
    ):
        saved = t[tcc_name] - t[cc_name]
        print(f"  {label}: cc repays its cold compile after {c[cc_name][0] / saved:.0f} calls")

    msecs = lambda v: f"{v * 1000:.0f}ms"  # noqa: E731

    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    fig.suptitle(
        f"Mandelbrot escape times, {W}×{H} grid, max_iter={MAX_ITER}",
        fontsize=10.5, color=INK,
    )  # fmt: skip
    bars(
        ax, "Time",
        ("Python\nper-element", "NumPy\nmask loop", "@jit\n(default)", '@jit\n(jit_backend="cc")'),
        [t[n] for n in ("py_loop", "numpy_masked", "jit_default", "jit_cc")], msecs, log=True,
    )  # fmt: skip
    save(fig, "tip_15a_jit_control_flow.png")

    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    fig.suptitle(
        "Elementwise expression, 8M float32 — no control flow, so jit traces",
        fontsize=10.5, color=INK,
    )  # fmt: skip
    bars(
        ax, "Time",
        ("NumPy", "@jit\n(default)", '@jit(strict=True)\n(jit_backend="tcc")', '@jit(strict=True)\n(jit_backend="cc")'),
        [t["numpy_route"], t["trace_route"], t["dsl_route"], t["dsl_route_cc"]], msecs,
        colors=(COLOR_NAIVE, COLOR_TIP, COLOR_TIP, COLOR_TIP),
    )  # fmt: skip
    save(fig, "tip_15b_jit_elementwise.png")
