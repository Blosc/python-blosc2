#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Tip 1: build large arrays with blosc2's own constructors (arange/linspace/
# fromiter), which fill chunk-by-chunk, instead of building a full NumPy
# array first and compressing it via asarray().

import numpy as np

import blosc2
from common import fmt_bytes, measure, save_plot

N = 200_000_000  # 200M float64 = ~1.5 GiB as a plain NumPy array


def naive():
    return blosc2.asarray(np.linspace(0, 1, N))


def tip():
    return blosc2.linspace(0, 1, N)


if __name__ == "__main__":
    naive_t, naive_m = measure(__file__, "naive")
    tip_t, tip_m = measure(__file__, "tip")

    print(f"naive  asarray(np.linspace(N={N:,})): {naive_t:.3f}s  peak {fmt_bytes(naive_m)}")
    print(f"tip    blosc2.linspace(N={N:,})     : {tip_t:.3f}s  peak {fmt_bytes(tip_m)}")
    print(f"speedup: {naive_t / tip_t:.1f}x   memory: {naive_m / tip_m:.1f}x less")

    save_plot(
        "tip_01_constructors.png",
        "blosc2.linspace() vs asarray(np.linspace()) — 200M float64 elements",
        "asarray(np.linspace)",
        "blosc2.linspace",
        naive_t,
        tip_t,
        naive_m,
        tip_m,
    )
