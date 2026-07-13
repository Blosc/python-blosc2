#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Tip 2: NDArray.slice() has a fast path when the slice boundaries land
# exactly on chunk boundaries — whole chunks are copied without
# decompressing/recompressing. An off-by-a-few-rows slice falls back to the
# general (decompress + recompress) path.

import numpy as np

import blosc2
from common import fmt_bytes, measure, save_plot

ROWS, COLS = 16_000, 2_000
CHUNK_ROWS = 4_000  # 4 chunks along axis 0

# Representative (compressible) data, not incompressible random noise -- blosc2
# arrays are typically built from data with real structure.
_base = np.arange(COLS, dtype=np.float64)
_data = np.tile(_base, (ROWS, 1)) + np.arange(ROWS, dtype=np.float64)[:, None] * 0.001
arr = blosc2.asarray(_data, chunks=(CHUNK_ROWS, COLS))


def unaligned():
    # starts 500 rows past a chunk boundary -> general path
    return arr.slice((slice(500, 500 + 3 * CHUNK_ROWS), slice(None)))


def aligned():
    # starts and ends exactly on chunk boundaries -> fast path
    return arr.slice((slice(CHUNK_ROWS, 4 * CHUNK_ROWS), slice(None)))


if __name__ == "__main__":
    naive_t, naive_m = measure(__file__, "unaligned")
    tip_t, tip_m = measure(__file__, "aligned")

    print(f"naive  unaligned slice: {naive_t:.3f}s  peak {fmt_bytes(naive_m)}")
    print(f"tip    aligned slice  : {tip_t:.3f}s  peak {fmt_bytes(tip_m)}")
    print(f"speedup: {naive_t / tip_t:.1f}x   (this tip is about speed; memory use is comparable)")

    save_plot(
        "tip_02_chunk_aligned_slicing.png",
        f"NDArray.slice(): chunk-aligned vs unaligned — {ROWS}x{COLS} float64, chunks=({CHUNK_ROWS},{COLS})",
        "unaligned slice",
        "chunk-aligned slice",
        naive_t,
        tip_t,
        naive_m,
        tip_m,
    )
