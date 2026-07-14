#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Tip: Column reductions accept a where= predicate that is pushed down into
# the chunk-wise scan. The NumPy-style alternative materializes the value
# column *and* the predicate column in full just to keep a fraction of the
# rows.

from pathlib import Path

import blosc2
from common import fmt_bytes, make_table, measure, save_plot

N = 20_000_000
URLPATH = str(Path(__file__).parent / "tip_08.b2d")

make_table(N, URLPATH)


def naive():
    t = blosc2.CTable.open(URLPATH)
    temp = t["temperature"][:]  # whole column decompressed
    reg = t["region"][:]  # and the predicate column too
    return temp[reg == 3].sum()


def tip():
    t = blosc2.CTable.open(URLPATH)
    return t["temperature"].sum(where=t.region == 3)  # pushed-down filter


if __name__ == "__main__":
    naive_t, naive_m = measure(__file__, "naive")
    tip_t, tip_m = measure(__file__, "tip")

    print(f"naive  mask two full columns : {naive_t:.4f}s  peak {fmt_bytes(naive_m)}")
    print(f"tip    sum(where=...)        : {tip_t:.4f}s  peak {fmt_bytes(tip_m)}")
    print(f"speedup: {naive_t / tip_t:.1f}x   memory: {naive_m / tip_m:.1f}x less")

    save_plot(
        "tip_08_where_pushdown.png",
        f"col.sum(where=...) vs NumPy-style masking — {N:,}-row table",
        "mask full columns",
        "sum(where=...)",
        naive_t,
        tip_t,
        naive_m,
        tip_m,
    )
