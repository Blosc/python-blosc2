#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""Group-by aggregation speed by key type (1e7 rows, low-cardinality keys).
String keys are the case to watch: they miss every
Cython fast path and go through hash-based factorization in
``_factorize_fixed_width_str``."""

import time
from dataclasses import dataclass

import numpy as np

import blosc2
from blosc2 import CTable

N = 10_000_000
rng = np.random.default_rng(42)

int_keys = rng.integers(0, 20, N)
float_vals = rng.random(N) * 100
cities = np.array(["Paris", "Rome", "Berlin", "Madrid", "Lisbon"])
str_keys = cities[rng.integers(0, 5, N)]


@dataclass
class Row:
    ikey: int = blosc2.field(blosc2.int64())
    skey: str = blosc2.field(blosc2.string(max_length=8))
    dkey: str = blosc2.field(blosc2.dictionary())
    ukey: str = blosc2.field(blosc2.utf8())
    val: float = blosc2.field(blosc2.float64())


print(f"building table ({N:.0e} rows)...", flush=True)
t = CTable(Row)
t.extend(
    {
        "ikey": int_keys,
        "skey": str_keys,
        "dkey": [str(s) for s in str_keys],
        "ukey": [str(s) for s in str_keys],
        "val": float_vals,
    },
    validate=False,
)


def bench(label, fn, reps=3):
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    print(f"{label:45s} {min(times) * 1000:8.1f} ms", flush=True)


bench("int key, sum", lambda: t.group_by("ikey").sum("val"))
bench("int key, mean", lambda: t.group_by("ikey").agg({"val": "mean"}))
bench("string key, sum", lambda: t.group_by("skey").sum("val"))
bench("dict key, sum", lambda: t.group_by("dkey").sum("val"))
bench("utf8 key, sum", lambda: t.group_by("ukey").sum("val"))
bench("two keys (int+dict), sum", lambda: t.group_by(["ikey", "dkey"]).sum("val"))
bench("two keys (int+utf8), sum", lambda: t.group_by(["ikey", "ukey"]).sum("val"))

try:
    import pandas as pd
except ImportError:
    pass
else:
    df = pd.DataFrame({"ikey": int_keys, "skey": str_keys, "val": float_vals})
    bench("pandas int key, sum", lambda: df.groupby("ikey")["val"].sum())
    bench("pandas string key, sum", lambda: df.groupby("skey")["val"].sum())

# rough speed-of-light reference for the int-key case
bench("numpy bincount int key, sum", lambda: np.bincount(int_keys, weights=float_vals))
