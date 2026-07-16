#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""Compare the three plain-string column representations head to head:
``utf8()`` (offsets + bytes, StringDType reads), ``string(max_length=L)``
(fixed-width UTF-32), and ``vlstring()`` (msgpack cells).

Two workloads:
- "taxi company": the real ``company`` column from the Chicago taxi dataset
  (medium-length, low-cardinality strings), when the parquet file is present.
- "synthetic free text": high-cardinality random words of 0-60 chars with
  some multi-byte values — the workload utf8() is designed for.

For each representation: ingest, storage footprint, full read, equality
filter, groupby-key aggregation, sort, and Arrow export.  Operations a
representation does not support are reported as such rather than skipped
silently.
"""

import pathlib
import time
from dataclasses import make_dataclass

import numpy as np

import blosc2
from blosc2 import CTable

N_TAXI = 10_000_000
N_SYNTH = 2_000_000  # fixed-width U~130 at 1e7 rows would need a ~5 GB ingest buffer
REPS = 3
TAXI_PARQUET = pathlib.Path(__file__).parent.parent / "chicago-taxi" / "chicago-taxi-flat.parquet"

rng = np.random.default_rng(42)


def bench(label, fn, reps=REPS):
    times = []
    result = None
    for _ in range(reps):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    print(f"  {label:34s} {min(times) * 1e3:9.1f} ms")
    return result


def load_taxi_company(n):
    import pyarrow.parquet as pq

    tbl = pq.read_table(TAXI_PARQUET, columns=["company"])
    col = tbl.column("company").combine_chunks()
    values = col.to_pylist()[:n]
    return [v if v is not None else "" for v in values]


def synth_free_text(n):
    words = np.array(
        ["taxi", "río", "航空", "boulevard", "x" * 40, "café", "", "downtown", "zürich", "o'hare"]
    )
    # 2-6 words per row, high cardinality via a row counter suffix on ~half.
    parts = words[rng.integers(0, len(words), (n, 3))]
    joined = [" ".join(p) for p in parts]
    salt = rng.integers(0, 100_000, n)  # ~100k distinct values: high cardinality, sane group count
    return [f"{s} #{salt[i]}" if i % 2 else s for i, s in enumerate(joined)]


def run_workload(title, values, filter_value):
    n = len(values)
    max_len = max(len(v) for v in values)
    float_vals = rng.random(n)
    print(f"\n=== {title} ({n:.0e} rows, max length {max_len} chars) ===")

    specs = {
        "utf8": blosc2.utf8(),
        "string": blosc2.string(max_length=max_len),
        "vlstring": blosc2.vlstring(),
    }
    for kind, spec in specs.items():
        row_cls = make_dataclass(
            "Row", [("s", str, blosc2.field(spec)), ("val", float, blosc2.field(blosc2.float64()))]
        )
        print(f"[{kind}]")
        t0 = time.perf_counter()
        t = CTable(row_cls)
        t.extend({"s": values, "val": float_vals}, validate=False)
        t._flush_varlen_columns()
        print(f"  {'ingest':34s} {(time.perf_counter() - t0) * 1e3:9.1f} ms")

        col = t._cols["s"]
        nbytes = getattr(col, "nbytes", None)
        cbytes = getattr(col, "cbytes", None)
        if nbytes is None:  # NDArray-backed fixed-width column
            nbytes, cbytes = col.schunk.nbytes, col.schunk.cbytes
        print(
            f"  {'storage nbytes -> cbytes':34s} {nbytes / 2**20:7.1f} MB -> {cbytes / 2**20:7.1f} MB (cratio {nbytes / cbytes:.1f}x)"
        )

        bench("full column read", lambda t=t: t["s"][:])
        try:
            bench("filter: count(s == value)", lambda t=t: int((t.s == filter_value)[:].sum()))
        except (NotImplementedError, TypeError) as exc:
            print(f"  {'filter: count(s == value)':34s}   unsupported: {str(exc)[:60]}")
        try:
            bench("groupby key: sum(val)", lambda t=t: t.group_by("s").sum("val"), reps=1)
        except (NotImplementedError, TypeError) as exc:
            print(f"  {'groupby key: sum(val)':34s}   unsupported: {str(exc)[:60]}")
        try:
            bench("sort_by(s) (copy)", lambda t=t: t.sort_by("s"), reps=1)
        except (NotImplementedError, TypeError) as exc:
            print(f"  {'sort_by(s) (copy)':34s}   unsupported: {str(exc)[:60]}")
        try:
            bench("to_arrow()", lambda t=t: t.to_arrow(), reps=1)
        except Exception as exc:
            print(f"  {'to_arrow()':34s}   failed: {str(exc)[:60]}")
        del t


if TAXI_PARQUET.exists():
    print("loading taxi company column...", flush=True)
    taxi = load_taxi_company(N_TAXI)
    # the most frequent company value as the filter probe
    vals, counts = np.unique(np.array(taxi, dtype=np.dtypes.StringDType()), return_counts=True)
    run_workload("chicago-taxi company", taxi, str(vals[np.argmax(counts)]))
    del taxi
else:
    print(f"({TAXI_PARQUET} not found; skipping the real-data workload)")

print("building synthetic free text...", flush=True)
synth = synth_free_text(N_SYNTH)
run_workload("synthetic free text", synth, synth[123])
