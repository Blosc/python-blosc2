#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Tip 4: closing a CTable auto-builds SUMMARY indexes (per-block min/max) for
# its eligible scalar columns. Column.min()/max() then answer straight from
# the precomputed per-block summaries instead of decompressing the column.
#
# (SUMMARY indexes can *also* skip whole blocks in a selective where() query,
# but only when the column's values are ordered/clustered enough that a
# predicate's range excludes entire blocks -- with IID data every 16k-row
# block spans almost the full value range, so there's nothing to skip. The
# min/max reduction win below is dramatic and doesn't depend on that.)

from dataclasses import dataclass
from pathlib import Path

import numpy as np

import blosc2
from common import fmt_bytes, measure, save_plot

N = 10_000_000
URL_INDEXED = str(Path(__file__).parent / "tip_04_indexed.b2d")
URL_NOINDEX = str(Path(__file__).parent / "tip_04_noindex.b2d")


@dataclass
class Row:
    sensor_id: int = blosc2.field(blosc2.int64())
    temperature: float = blosc2.field(blosc2.float64())
    region: int = blosc2.field(blosc2.int32())


def _build(urlpath, create_summary_index):
    import shutil

    p = Path(urlpath)
    if p.is_dir():
        shutil.rmtree(p)
    np_dtype = np.dtype([("sensor_id", np.int64), ("temperature", np.float64), ("region", np.int32)])
    rng = np.random.default_rng(42)
    data = np.empty(N, dtype=np_dtype)
    data["sensor_id"] = np.arange(N, dtype=np.int64)
    data["temperature"] = 15.0 + rng.random(N) * 25
    data["region"] = rng.integers(0, 8, size=N, dtype=np.int32)
    with blosc2.CTable(
        Row, urlpath=urlpath, mode="w", expected_size=N, create_summary_index=create_summary_index
    ) as t:
        t.extend(data)


_build(URL_NOINDEX, create_summary_index=False)
_build(URL_INDEXED, create_summary_index=True)


def naive():
    t = blosc2.CTable.open(URL_NOINDEX)
    return t["temperature"].max()


def tip():
    t = blosc2.CTable.open(URL_INDEXED)
    return t["temperature"].max()


if __name__ == "__main__":
    naive_t, naive_m = measure(__file__, "naive")
    tip_t, tip_m = measure(__file__, "tip")

    print(f"naive  col.max() no SUMMARY index : {naive_t:.4f}s  peak {fmt_bytes(naive_m)}")
    print(f"tip    col.max() with SUMMARY index: {tip_t:.4f}s  peak {fmt_bytes(tip_m)}")
    print(f"speedup: {naive_t / tip_t:.1f}x")

    save_plot(
        "tip_04_summary_index_where.png",
        f"Column.max() with vs without a SUMMARY index — {N:,} rows",
        "no index",
        "SUMMARY index",
        naive_t,
        tip_t,
        naive_m,
        tip_m,
    )
