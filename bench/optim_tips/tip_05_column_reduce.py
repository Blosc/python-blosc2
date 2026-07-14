#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Tip 5: Column.__getitem__ always materializes a full NumPy array (with
# null-sentinel processing). Column's own reduction methods (sum/mean/...)
# work chunk-by-chunk without ever holding the whole column decompressed
# in one block -- so reduce the Column directly instead of slicing first.

from dataclasses import dataclass
from pathlib import Path

import numpy as np

import blosc2
from common import fmt_bytes, measure, save_plot

N = 50_000_000
URLPATH = str(Path(__file__).parent / "tip_05.b2d")


@dataclass
class Row:
    val: float = blosc2.field(blosc2.float64())


def _build():
    import shutil

    p = Path(URLPATH)
    if p.is_dir():
        shutil.rmtree(p)
    data = np.random.default_rng(0).random(N)
    with blosc2.CTable(Row, urlpath=URLPATH, mode="w", expected_size=N) as t:
        t.extend({"val": data})


_build()


def naive():
    t = blosc2.CTable.open(URLPATH)
    return t["val"][:].sum()  # materializes the whole column as NumPy first


def tip():
    t = blosc2.CTable.open(URLPATH)
    return t["val"].sum()  # chunk-wise reduction, no full materialization


if __name__ == "__main__":
    naive_t, naive_m = measure(__file__, "naive")
    tip_t, tip_m = measure(__file__, "tip")

    print(f"naive  col[:].sum() : {naive_t:.4f}s  peak {fmt_bytes(naive_m)}")
    print(f"tip    col.sum()    : {tip_t:.4f}s  peak {fmt_bytes(tip_m)}")
    print(f"speedup: {naive_t / tip_t:.1f}x   memory: {naive_m / tip_m:.1f}x less")

    save_plot(
        "tip_05_column_reduce.png",
        f"col.sum() vs col[:].sum() — {N:,}-row column",
        "col[:].sum()",
        "col.sum()",
        naive_t,
        tip_t,
        naive_m,
        tip_m,
    )
