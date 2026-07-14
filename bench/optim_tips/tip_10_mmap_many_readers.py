#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Tip: when several processes read the same blosc2 file, open it with
# mmap_mode="r" in every reader. Each access then goes straight to the shared
# mapped pages instead of paying a syscall plus a page-cache-to-private-buffer
# copy, and the advantage *grows* with the number of concurrent readers.
#
# This one doesn't fit the common naive()/tip() harness (it measures waves of
# concurrent reader processes), so it drives its own subprocesses and draws a
# grouped-bar chart. See the "sharing containers across processes" guide for
# the full discussion, including why RSS is misleading for mmap readers.

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common import COLOR_NAIVE, COLOR_TIP, GRID, INK, MUTED, OUT_DIR

ROWS, COLS = 400_000, 100  # 320 MB float64 uncompressed
K_READS = 300  # random slices per reader
SLICE_ROWS = 2_000  # ~1.6 MB decompressed per read
NPROCS = (1, 4, 8)
URL = str(Path(__file__).parent / "tip_10.b2nd")

_READER = """
import json, os, resource, sys, time
import numpy as np
import blosc2

url, use_mmap, k, slice_rows = sys.argv[1], sys.argv[2] == "1", int(sys.argv[3]), int(sys.argv[4])
kw = {"mmap_mode": "r"} if use_mmap else {}
a = blosc2.open(url, **kw)
nrows = a.shape[0]
rng = np.random.default_rng(os.getpid())
t0 = time.perf_counter()
acc = 0.0
for _ in range(k):
    start = int(rng.integers(0, nrows - slice_rows))
    acc += float(a[start : start + slice_rows].sum())
elapsed = time.perf_counter() - t0
ru = resource.getrusage(resource.RUSAGE_SELF)
print(json.dumps({"elapsed": elapsed, "cpu": ru.ru_utime + ru.ru_stime}))
"""


def _build():
    if os.path.exists(URL):
        return
    import blosc2

    # Random mantissas compress poorly (cratio ~1.2), so the I/O path stays
    # visible instead of being masked by decompression time.
    data = np.random.default_rng(0).random((ROWS, COLS))
    blosc2.asarray(data, urlpath=URL, mode="w")


def _run_wave(nproc, use_mmap):
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", _READER, URL, "1" if use_mmap else "0", str(K_READS), str(SLICE_ROWS)],
            stdout=subprocess.PIPE,
            text=True,
        )
        for _ in range(nproc)
    ]
    t0 = time.perf_counter()
    results = [json.loads(p.communicate()[0].strip().splitlines()[-1]) for p in procs]
    wall = time.perf_counter() - t0
    return wall, sum(r["cpu"] for r in results)


if __name__ == "__main__":
    _build()
    # Warm the page cache so both modes start from the same state.
    with open(URL, "rb") as f:
        while f.read(1 << 24):
            pass

    io_wall, mmap_wall = [], []
    for nproc in NPROCS:
        w_io, cpu_io = _run_wave(nproc, use_mmap=False)
        w_mm, cpu_mm = _run_wave(nproc, use_mmap=True)
        io_wall.append(w_io)
        mmap_wall.append(w_mm)
        print(
            f"P={nproc:2d}  regular I/O: {w_io:5.2f}s (cpu {cpu_io:5.2f}s)   "
            f'mmap_mode="r": {w_mm:5.2f}s (cpu {cpu_mm:5.2f}s)   '
            f"speedup: {w_io / w_mm:.1f}x"
        )

    fig, ax = plt.subplots(figsize=(8, 3.2))
    fig.suptitle(
        f"{K_READS} random slice reads per reader — {ROWS:,}x{COLS} float64, warm cache",
        fontsize=11,
        color=INK,
    )
    x = np.arange(len(NPROCS))
    width = 0.38
    for off, vals, color, label in (
        (-width / 2, io_wall, COLOR_NAIVE, "regular I/O"),
        (width / 2, mmap_wall, COLOR_TIP, 'mmap_mode="r"'),
    ):
        bars = ax.bar(x + off, vals, width, color=color, label=label)
        for bar, v in zip(bars, vals, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(io_wall) * 0.02,
                f"{v:.2f}s",
                ha="center",
                va="bottom",
                fontsize=9,
                color=INK,
            )
    ax.set_xticks(x, [f"{p} reader{'s' if p > 1 else ''}" for p in NPROCS])
    ax.set_ylabel("Wall time (s)", color=INK, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_yticklabels([])
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(io_wall) * 1.2)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out_path = OUT_DIR / "tip_10_mmap_many_readers.png"
    fig.savefig(out_path, dpi=150)
    print(f"plot saved to {out_path}")
