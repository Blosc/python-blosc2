#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""Mmap reads across block sizes and block-boundary alignment.

Similar to tip_02_chunk_aligned_slicing.py, but benchmarks the effect
of block size variation and block-boundary alignment on mmap read times.

Read times can be 1.5-2x worse when crossing block boundaries instead
of aligning, increasing alongside the chosen block size.

Mmap accesses compressed data through OS-managed file pages. However,
these timings show the combined cost of that access and decompression;
they do not count pages or isolate page-cache activity.
"""

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from optim_tips.common import measure

import blosc2

# Test differences in block size, keep all other variables stable
ROWS, COLS = 16_000, 2_000
K_READS = 400
CHUNK_ROWS = 4_000
READ_ROWS = 50
BLOCK_ROWS = (50, 100, 200)

# Pick K_READS=400 slice locations that are on the boundaries for every
# block size.
_positions_rng = np.random.default_rng(1)
_chunk_ids = _positions_rng.integers(0, ROWS // CHUNK_ROWS, size=K_READS)
_block_ids = _positions_rng.integers(1, CHUNK_ROWS // max(BLOCK_ROWS), size=K_READS)

# Pick block-aligned reads inside chunks, then shift to cross block
# boundaries.
_aligned = _chunk_ids * CHUNK_ROWS + _block_ids * max(BLOCK_ROWS)
_crossing = _aligned - READ_ROWS // 2

# Same data construction as tip 02; aligns with the chosen block sizes.
_base = np.arange(COLS, dtype=np.float64)
_data = np.tile(_base, (ROWS, 1)) + np.arange(ROWS, dtype=np.float64)[:, None] * 0.001

# Write three temporary files, then open each through mmap for the read
# tests.
_temp_dir = tempfile.TemporaryDirectory(prefix="blosc2-mmap-blocks-")
_paths = {rows: Path(_temp_dir.name) / f"blocks-{rows}.b2nd" for rows in BLOCK_ROWS}
_arrays = {}
for _rows, _path in _paths.items():
    blosc2.asarray(_data, chunks=(CHUNK_ROWS, COLS), blocks=(_rows, COLS), urlpath=_path, mode="w")
    _arrays[_rows] = blosc2.open(_path, mmap_mode="r")
del _data


def read_slices(arr, starts):
    for start in starts:
        arr[start : start + READ_ROWS, :]


def read_50_aligned():
    return read_slices(_arrays[50], _aligned)


def read_50_crossing():
    return read_slices(_arrays[50], _crossing)


def read_100_aligned():
    return read_slices(_arrays[100], _aligned)


def read_100_crossing():
    return read_slices(_arrays[100], _crossing)


def read_200_aligned():
    return read_slices(_arrays[200], _aligned)


def read_200_crossing():
    return read_slices(_arrays[200], _crossing)


def save_plot(results):
    x = np.arange(len(results))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(x - width / 2, [row[1] for row in results], width, label="Aligned")
    crossing_bars = ax.bar(x + width / 2, [row[2] for row in results], width, label="Crossing")
    ax.bar_label(
        crossing_bars,
        labels=[f"{crossing / aligned:.2f}x" for _, aligned, crossing in results],
        padding=3,
    )
    ax.set_ylim(0, max(row[2] for row in results) * 1.2)

    ax.set_xticks(x, [str(row[0]) for row in results])
    ax.set_xlabel("Block rows")
    ax.set_ylabel(f"Time for {K_READS} reads (s)")
    ax.set_title("Mmap read time by block size and alignment")
    ax.legend()
    fig.tight_layout()

    path = Path(__file__).resolve().parent.parent / "doc/guides/mmap_block_locality.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Plot saved to {path}")


if __name__ == "__main__":
    print(f"Mmap reads: {K_READS} slices of {READ_ROWS} X {COLS} float64 values")
    print(f"Fixed chunk shape: {CHUNK_ROWS} X {COLS}")
    print(f"{'block rows':>10} {'file MiB':>10} {'aligned s':>11} {'crossing s':>12} {'penalty':>9}")

    results = []
    for rows, path in _paths.items():
        # Perform and measure aligned and crossing reads
        aligned_time, _ = measure(__file__, f"read_{rows}_aligned")
        crossing_time, _ = measure(__file__, f"read_{rows}_crossing")

        results.append((rows, aligned_time, crossing_time))
        print(
            f"{rows:10d} {path.stat().st_size / 1024**2:10.1f} "
            f"{aligned_time:11.4f} {crossing_time:12.4f} "
            f"{crossing_time / aligned_time:8.2f}x"
        )

    save_plot(results)
