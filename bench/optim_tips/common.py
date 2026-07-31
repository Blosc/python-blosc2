#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Shared helpers for the doc/guides/optimization_tips.md benchmark scripts:
# measuring elapsed time + peak memory for a "naive" vs "tip" variant, and
# plotting the two side by side.
#
# Each variant is measured in its own fresh subprocess rather than in-process:
# a same-process before/after comparison is unreliable here because (a)
# ru_maxrss is a whole-process *high-water mark* that never drops, so
# whichever variant runs second inherits the first one's peak, and (b)
# tracemalloc only sees Python-level allocations, missing the C-Blosc2
# extension's native buffers entirely. A subprocess per variant sidesteps
# both: each gets an identical, independent baseline (same module-level setup
# code re-run fresh), and resource.getrusage() in that subprocess reports the
# real OS-level peak RSS for everything it did, C allocations included.

import json
import platform
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "doc" / "guides" / "optim_tips"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ru_maxrss is in KiB on Linux but bytes on macOS.
_RSS_UNIT = 1 if platform.system() == "Darwin" else 1024

# dataviz reference palette: slot 1 (blue) = naive/before, slot 2 (aqua) = tip/after
COLOR_NAIVE = "#2a78d6"
COLOR_TIP = "#1baf7a"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"

_DRIVER = """\
import gc, importlib.util, json, os, platform, resource, sys, time, tracemalloc
sys.path.insert(0, os.path.dirname(sys.argv[1]))  # so the script's own `from common import ...` resolves
spec = importlib.util.spec_from_file_location("_bench_mod", sys.argv[1])
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)  # runs the script's module-level setup once
fn = getattr(mod, sys.argv[2])
unit = 1 if platform.system() == "Darwin" else 1024
# Pass 1 -- timing, with nothing attached.  tracemalloc hooks every Python
# allocation, so leaving it on here inflates read-heavy tips several-fold and,
# worse, adds run-to-run variance far larger than the difference being measured.
#
# Best of REPS rather than a single shot.  The first call is reliably the
# slowest (cold allocator, lazy imports, first touch of a mapping), and it hits
# the two variants unequally, so a single sample both adds noise and biases the
# comparison.  Minimum is the right statistic: interference only ever makes a
# run slower.  Measured convergence on tip_06 -- 1 rep 1.27x, 2 reps 1.13x,
# 3 reps 1.12x, and no further movement out to 8.
#
# A tip whose point is a *one-shot* operation sets BENCH_REPS = 1 instead: when
# one variant does real filesystem work every time and the other goes fully
# page-cached on the second call, repeating flatters the second one and the
# number stops describing what a user would see.  (Measured: tip_09 reads 2.8x
# at one rep and 7.0x at three; tip_03b, 180x and 404x.)
REPS = getattr(mod, "BENCH_REPS", 3)
gc.collect()
rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * unit
elapsed = float("inf")
for _ in range(REPS):
    t0 = time.perf_counter()
    fn()
    elapsed = min(elapsed, time.perf_counter() - t0)
rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * unit
rss_delta = max(0, rss_after - rss_before)
# Pass 2 -- memory only.  Costs a second call of fn(); that is the price of an
# uncontaminated timing, and every measured function here is re-runnable.
gc.collect()
tracemalloc.start()
fn()
_, py_peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
print(json.dumps({"elapsed": elapsed, "rss": max(py_peak, rss_delta)}))
"""


def measure(script_path, func_name):
    """Run func_name() from script_path in a fresh subprocess.

    Returns (elapsed_seconds, peak_bytes): elapsed covers only the call to
    func_name() (module-level setup runs first and is excluded); peak_bytes is
    the larger of (a) the tracemalloc peak during func_name() -- accurate for
    NumPy/Python-level array materialization -- and (b) the growth in the
    subprocess's peak RSS over a post-setup, post-gc.collect() baseline, which
    catches native/C-Blosc2-level allocations tracemalloc can't see. Running
    each variant in its own fresh process (rather than two in-process
    before/after snapshots) matters for (b): ru_maxrss is a high-water mark
    that never drops, so a same-process comparison would silently inherit
    whichever variant ran first's peak.

    elapsed is the *best* of three calls with no profiler attached, then a
    fourth call runs under tracemalloc for (a). Two separate reasons for that
    shape. Measuring time and memory in one pass costs the timing --
    tracemalloc's per-allocation hook dominates any tip whose work is
    Python-side, and its overhead varies enough between runs to hide the effect
    the tip is demonstrating. And timing a single call costs it again: the first
    call is reliably the slowest, unequally so between the two variants, which
    biases the comparison on top of the noise it adds.
    """
    proc = subprocess.run(
        [sys.executable, "-c", _DRIVER, str(Path(script_path).resolve()), func_name],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"{script_path}:{func_name} failed:\n{proc.stderr}")
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    return data["elapsed"], data["rss"]


def fmt_bytes(n):
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(n) < 1024 or unit == "GiB":
            return f"{n:.1f} {unit}"
        n /= 1024


def save_plot(png_name, title, naive_label, tip_label, naive_time, tip_time, naive_mem, tip_mem):
    """Two-panel bar chart (time, peak memory), naive vs tip, with value labels."""
    fig, (ax_t, ax_m) = plt.subplots(1, 2, figsize=(8, 3.2))
    fig.suptitle(title, fontsize=11, color=INK)

    for ax, values, ylabel, fmt in (
        (ax_t, (naive_time, tip_time), "Time (s)", lambda v: f"{v:.3g}s"),
        (ax_m, (naive_mem, tip_mem), "Peak memory", fmt_bytes),
    ):
        bars = ax.bar(
            [naive_label, tip_label],
            values,
            color=[COLOR_NAIVE, COLOR_TIP],
            width=0.55,
        )
        ax.set_ylabel(ylabel, color=INK, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.set_yticklabels([])  # values are direct-labeled on the bars instead
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        top = max(values) if max(values) > 0 else 1
        ax.set_ylim(0, top * 1.2)
        for bar, v in zip(bars, values, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + top * 0.03,
                fmt(v),
                ha="center",
                va="bottom",
                fontsize=9,
                color=INK,
            )

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out_path = OUT_DIR / png_name
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def make_table(n, urlpath, seed=42):
    """Build (and close, to trigger SUMMARY index creation) an on-disk CTable
    with n rows, shared by the tips that need a persistent table."""
    import shutil
    from dataclasses import dataclass

    import blosc2

    @dataclass
    class Row:
        sensor_id: int = blosc2.field(blosc2.int64())
        temperature: float = blosc2.field(blosc2.float64())
        region: int = blosc2.field(blosc2.int32())

    np_dtype = np.dtype([("sensor_id", np.int64), ("temperature", np.float64), ("region", np.int32)])
    rng = np.random.default_rng(seed)
    data = np.empty(n, dtype=np_dtype)
    data["sensor_id"] = np.arange(n, dtype=np.int64)
    data["temperature"] = 15.0 + rng.random(n) * 25
    data["region"] = rng.integers(0, 8, size=n, dtype=np.int32)

    p = Path(urlpath)
    if p.is_dir():
        shutil.rmtree(p)
    else:
        p.unlink(missing_ok=True)
    with blosc2.CTable(Row, urlpath=urlpath, mode="w", expected_size=n) as t:
        t.extend(data)
    return blosc2.CTable.open(urlpath)
