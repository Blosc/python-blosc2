#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Tip 6: blosc2.open(path, mmap_mode="r") memory-maps a read-only container
# instead of going through regular file I/O for every chunk access. For a
# workload that touches many scattered chunks, mapping the file once avoids
# repeated open/seek/read syscalls per chunk.

from pathlib import Path

import numpy as np

import blosc2
from common import fmt_bytes, measure, save_plot

N, COLS, CHUNK = 200_000, 500, 500
URLPATH = str(Path(__file__).parent / "tip_06.b2nd")
N_READS = 8000


def _build():
    Path(URLPATH).unlink(missing_ok=True)
    base = np.arange(COLS, dtype=np.float64)
    data = np.tile(base, (N, 1)) + np.arange(N, dtype=np.float64)[:, None] * 0.001
    blosc2.asarray(data, chunks=(CHUNK, COLS), urlpath=URLPATH, mode="w")


_build()

_idxs = np.random.default_rng(1).integers(0, N - CHUNK, size=N_READS)


def _scattered_reads(arr):
    total = 0.0
    for i in _idxs:
        total += arr[i : i + 5, :50].sum()
    return total


def naive():
    arr = blosc2.open(URLPATH)
    return _scattered_reads(arr)


def tip():
    arr = blosc2.open(URLPATH, mmap_mode="r")
    return _scattered_reads(arr)


if __name__ == "__main__":
    naive_t, naive_m = measure(__file__, "naive")
    tip_t, tip_m = measure(__file__, "tip")

    print(f"naive  plain open()         : {naive_t:.3f}s  peak {fmt_bytes(naive_m)}")
    print(f"tip    open(mmap_mode='r')  : {tip_t:.3f}s  peak {fmt_bytes(tip_m)}")
    print(f"speedup: {naive_t / tip_t:.1f}x")

    save_plot(
        "tip_06_mmap_read.png",
        f"open(mmap_mode='r') vs plain open() — {N_READS:,} scattered slice reads",
        "open()",
        "open(mmap_mode='r')",
        naive_t,
        tip_t,
        naive_m,
        tip_m,
    )
