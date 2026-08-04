#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Tip 12: when a large array is a combination of smaller ones, broadcast the
# small blosc2 operands into a lazy expression and .compute() it straight to
# disk. Blosc2 broadcasts the operands chunk by chunk, so the full uncompressed
# result never exists in memory -- unlike NumPy broadcasting, which materializes
# it before asarray() ever sees it.

from pathlib import Path

import numpy as np

import blosc2
from common import fmt_bytes, measure, save_plot

N, COLS, CHUNK = 200_000, 500, 500  # 100M float64 = ~800 MiB uncompressed
URLPATH = str(Path(__file__).parent / "tip_12.b2nd")


def naive():
    cols = np.arange(COLS, dtype=np.float64)
    rows = np.arange(0, N * 0.001, 0.001, dtype=np.float64).reshape(N, 1)
    return blosc2.asarray(rows + cols, chunks=(CHUNK, COLS), urlpath=URLPATH, mode="w")


def tip():
    cols = blosc2.arange(COLS, dtype=np.float64)
    rows = blosc2.arange(0, N * 0.001, 0.001, dtype=np.float64, shape=(N, 1))
    return (rows + cols).compute(chunks=(CHUNK, COLS), urlpath=URLPATH, mode="w")


if __name__ == "__main__":
    # Same bytes both ways, or the comparison means nothing.
    np.testing.assert_array_equal(naive()[:], tip()[:])

    naive_t, naive_m = measure(__file__, "naive")
    tip_t, tip_m = measure(__file__, "tip")

    print(f"naive  asarray(rows + cols)    : {naive_t:.3f}s  peak {fmt_bytes(naive_m)}")
    print(f"tip    (rows + cols).compute() : {tip_t:.3f}s  peak {fmt_bytes(tip_m)}")
    print(f"speedup: {naive_t / tip_t:.1f}x   memory: {naive_m / tip_m:.1f}x less")

    save_plot(
        "tip_12_broadcast_build.png",
        f"Broadcast lazy expression vs NumPy staging — {N:,}x{COLS} float64 on disk",
        "asarray(rows + cols)",
        "(rows + cols).compute",
        naive_t,
        tip_t,
        naive_m,
        tip_m,
    )
