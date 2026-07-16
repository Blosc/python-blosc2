#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Tip 3: CTable.sort_by(view=True) returns a lightweight sorted *view* that
# shares the parent's column data and gathers rows on demand, instead of
# materializing a whole sorted copy of the table. On a FULLy-indexed column
# it streams straight from the index, so the table is never sorted at all.

from pathlib import Path

import blosc2
from common import fmt_bytes, make_table, measure, save_plot

N = 20_000_000
URLPATH = str(Path(__file__).parent / "tip_03.b2d")
TOPK = 10

make_table(N, URLPATH)  # closing already built a SUMMARY index on "temperature"
with blosc2.CTable.open(URLPATH, mode="a") as t:
    t.drop_index("temperature")
    t.create_index("temperature", kind=blosc2.IndexKind.FULL)


def naive():
    # Sorts (and materializes) the whole table, then takes the top 10.
    t = blosc2.CTable.open(URLPATH)
    return t.sort_by("temperature")[:TOPK]


def tip():
    # Zero-copy sorted view, streamed from the FULL index.
    t = blosc2.CTable.open(URLPATH)
    return t.sort_by("temperature", view=True)[:TOPK]


if __name__ == "__main__":
    naive_t, naive_m = measure(__file__, "naive")
    tip_t, tip_m = measure(__file__, "tip")

    print(f"naive  sort_by()[:10]            : {naive_t:.3f}s  peak {fmt_bytes(naive_m)}")
    print(f"tip    sort_by(view=True)[:10]   : {tip_t:.3f}s  peak {fmt_bytes(tip_m)}")
    print(f"speedup: {naive_t / tip_t:.1f}x   memory: {naive_m / tip_m:.1f}x less")

    save_plot(
        "tip_03_sort_by_view.png",
        f"CTable.sort_by(view=True) top-10 — {N:,}-row table, FULL index",
        "sort_by()[:10]",
        "sort_by(view=True)[:10]",
        naive_t,
        tip_t,
        naive_m,
        tip_m,
    )
