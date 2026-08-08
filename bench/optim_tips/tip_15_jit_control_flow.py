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
# elementwise expressions.  A final pair measures the same DSL kernels compiled
# with jit_backend="cc" (system C compiler) instead of the bundled tcc.
#
# Peak memory is the same story for every variant here (the result array plus
# a few temporaries), so the plot is time-only. The script prints a
# correctness check of every mode against the pure-Python reference, and
# shows that strict=False cannot even trace this function: the loop condition
# depends on the traced arrays, so tracing raises instead of recording one
# path for every element.

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


def trace_route():
    return elementwise(X)


def dsl_route():
    return elementwise_dsl(X)


def dsl_route_cc():
    return elementwise_dsl_cc(X)


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
    for name in mandel_names + ("trace_route", "dsl_route", "dsl_route_cc"):
        t[name], rss = measure(__file__, name)
        print(f"{name:<13} {t[name]:8.4f}s   peak {rss / 1e6:6.1f} MB")

    print("\nmandelbrot speedups vs py_loop:")
    for name in mandel_names[1:]:
        print(f"  {name:<13} {t['py_loop'] / t[name]:6.1f}x faster than py_loop")
    print(f"  elementwise: trace {t['dsl_route'] / t['trace_route']:.2f}x faster than forced DSL")
    print("\njit_backend='cc' vs default tcc (steady state):")
    print(f"  mandel:      {t['jit_default'] / t['jit_cc']:.2f}x faster with cc")
    print(f"  elementwise: {t['dsl_route'] / t['dsl_route_cc']:.2f}x faster with cc")

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
        ("@jit\n(default)", '@jit(strict=True)\n(jit_backend="tcc")', '@jit(strict=True)\n(jit_backend="cc")'),
        [t["trace_route"], t["dsl_route"], t["dsl_route_cc"]], msecs,
        colors=(COLOR_NAIVE, COLOR_TIP, COLOR_TIP),
    )  # fmt: skip
    save(fig, "tip_15b_jit_elementwise.png")
