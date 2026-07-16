#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Tip 3b: NDArray.iter_sorted() streams top-k values straight from a FULL
# index instead of materializing the full permutation.  argsort() reads the
# entire sorted order (all N positions); iter_sorted(start=-k) reads just
# the tail of the index sidecar.

from pathlib import Path

import blosc2
import numpy as np
from common import fmt_bytes, measure, save_plot

N = 20_000_000
URLPATH = str(Path(__file__).parent / "tip_03b.b2nd")
TOPK = 10

# Module-level: create the persistent array with a FULL index once.
p = Path(URLPATH)
if not p.exists():
    p.unlink(missing_ok=True)
    rng = np.random.default_rng(42)
    data = 15.0 + rng.random(N, dtype=np.float64) * 25
    arr = blosc2.asarray(data, urlpath=URLPATH, mode="w")
    arr.create_index(kind=blosc2.IndexKind.FULL)
    del arr, data


def naive():
    # argsort materialises the full permutation even
    # though we only keep the top 10 positions.
    arr = blosc2.open(URLPATH)
    return arr[arr.argsort()[-TOPK:]]


def tip():
    # iter_sorted reads just the last 10 entries from the index sidecar.
    arr = blosc2.open(URLPATH)
    return list(arr.iter_sorted(start=-TOPK))


if __name__ == "__main__":
    naive_t, naive_m = measure(__file__, "naive")
    tip_t, tip_m = measure(__file__, "tip")

    print(f"naive  arr.argsort()[-10:]           : {naive_t:.3f}s  peak {fmt_bytes(naive_m)}")
    print(f"tip    arr.iter_sorted(start=-10)    : {tip_t:.3f}s  peak {fmt_bytes(tip_m)}")
    print(f"speedup: {naive_t / tip_t:.1f}x   memory: {naive_m / tip_m:.1f}x less")

    save_plot(
        "tip_03b_ndarray_iter_sorted.png",
        f"NDArray.iter_sorted(start=-10) top-10 — {N:,}-element 1-D array, FULL index",
        "arr.argsort()[-10:]",
        "arr.iter_sorted(start=-10)",
        naive_t,
        tip_t,
        naive_m,
        tip_m,
    )
