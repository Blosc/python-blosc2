#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Benchmark: what does a SUMMARY index buy you on a where() query?
#
# The same selective range query runs three ways on identical data:
#   1. no index          -- full scan, every row is visited
#   2. SUMMARY (chunk)   -- per-chunk min/max can rule out whole chunks
#   3. SUMMARY (block)   -- per-block min/max rules out smaller segments,
#                           so it can skip more, at the cost of more metadata
#
# A SUMMARY index stores only the min/max of each segment.  A segment is
# scanned only when its [min, max] range intersects the query predicate;
# "rows scanned" below is computed with exactly that rule from the data
# itself (for the no-index case every row is scanned).  The data is
# clustered (value grows with the row number), which is the favourable
# case for segment skipping; on shuffled data the gains would shrink.
#
# Self-contained: run right after `pip install blosc2`, no dataset needed.

from dataclasses import dataclass
from time import perf_counter as time

import numpy as np

import blosc2


@dataclass
class Row:
    id: int = blosc2.field(blosc2.int64(ge=0))
    value: float = blosc2.field(blosc2.float64(ge=0), default=0.0)


N = 3_000_000
THRESHOLD = 2_700_000  # top ~10% of the value range
QUERY = f"value > {THRESHOLD}"
REPEATS = 3

print(f"SUMMARY index benchmark  |  N = {N:,} rows, query: {QUERY}\n")

data = np.empty(N, dtype=np.dtype([("id", np.int64), ("value", np.float64)]))
data["id"] = np.arange(N, dtype=np.int64)
data["value"] = np.arange(N, dtype=np.float64)
values = data["value"]


def rows_scanned(segment: int) -> int:
    """Rows in segments whose [min, max] intersects the query range."""
    scanned = 0
    for start in range(0, N, segment):
        seg = values[start : start + segment]
        if seg.max() > THRESHOLD:
            scanned += len(seg)
    return scanned


def run_query(ct) -> tuple[int, float]:
    """Best-of-REPEATS timing of the query; returns (matches, seconds)."""
    best = float("inf")
    matches = 0
    for _ in range(REPEATS):
        t0 = time()
        result = ct.where(QUERY)
        best = min(best, time() - t0)
        matches = result.nrows
    return matches, best


variants = []
for label, granularity in [("no index", None), ("SUMMARY (chunk)", "chunk"), ("SUMMARY (block)", "block")]:
    # Start every variant with no automatic SUMMARY index so that the only
    # difference is the explicit index built below.
    ct = blosc2.CTable(Row, expected_size=N, create_summary_index=False)
    ct.extend(data)
    build_s = 0.0
    segment = N  # a full scan visits every row
    if granularity is not None:
        t0 = time()
        ct.create_index("value", kind=blosc2.IndexKind.SUMMARY, granularity=granularity)
        build_s = time() - t0
        col = ct["value"].raw
        segment = col.chunks[0] if granularity == "chunk" else col.blocks[0]
    matches, query_s = run_query(ct)
    variants.append((label, build_s, rows_scanned(segment), matches, query_s))
    del ct

expected = variants[0][3]
assert all(v[3] == expected for v in variants), "variants disagree on the result!"

baseline = variants[0][4]
print(f"{'VARIANT':<18} {'BUILD (s)':>10} {'ROWS SCANNED':>14} {'MATCHES':>10} {'TIME (s)':>10} {'SPEEDUP':>8}")
print("-" * 74)
for label, build_s, scanned, matches, query_s in variants:
    print(
        f"{label:<18} {build_s:>10.3f} {scanned:>14,} {matches:>10,} "
        f"{query_s:>10.3f} {baseline / query_s:>7.2f}x"
    )
print("-" * 74)
print(
    "\nTakeaway: the block-granularity SUMMARY index skips the most segments\n"
    "on clustered data, so it is fastest here; chunk granularity stores less\n"
    "metadata but can only rule out coarser ranges.  Both cost a one-time\n"
    "build (and a little storage) that pays off on repeated selective queries."
)
