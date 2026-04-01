#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

from __future__ import annotations

import statistics
import time

import numpy as np

import blosc2


SIZES = (1_000_000, 2_000_000, 5_000_000, 10_000_000)
CHUNK_LEN = 100_000
BLOCK_LEN = 20_000
REPEATS = 5


def build_array(size: int) -> blosc2.NDArray:
    dtype = np.dtype([("id", np.int64), ("payload", np.float32)])
    data = np.empty(size, dtype=dtype)
    data["id"] = np.arange(size, dtype=np.int64)
    data["payload"] = (np.arange(size, dtype=np.float32) % 1024) / 1024
    return blosc2.asarray(data, chunks=(CHUNK_LEN,), blocks=(BLOCK_LEN,))


def benchmark_once(expr, *, use_index: bool) -> tuple[float, int]:
    start = time.perf_counter()
    result = expr.compute(_use_index=use_index)[:]
    elapsed = time.perf_counter() - start
    return elapsed, len(result)


def benchmark_size(size: int) -> dict:
    arr = build_array(size)
    lo = size // 2
    width = max(10_000, size // 1_000)
    hi = min(size, lo + width)

    build_start = time.perf_counter()
    arr.create_index(field="id")
    build_time = time.perf_counter() - build_start

    expr = blosc2.lazyexpr(f"(id >= {lo}) & (id < {hi})", arr.fields).where(arr)
    explanation = expr.explain()

    warm_scan, scan_len = benchmark_once(expr, use_index=False)
    warm_index, index_len = benchmark_once(expr, use_index=True)
    assert scan_len == index_len
    del warm_scan, warm_index

    scan_runs = [benchmark_once(expr, use_index=False)[0] for _ in range(REPEATS)]
    index_runs = [benchmark_once(expr, use_index=True)[0] for _ in range(REPEATS)]

    return {
        "size": size,
        "query_rows": index_len,
        "build_s": build_time,
        "scan_ms": statistics.median(scan_runs) * 1_000,
        "index_ms": statistics.median(index_runs) * 1_000,
        "speedup": statistics.median(scan_runs) / statistics.median(index_runs),
        "candidate_chunks": explanation["candidate_chunks"],
        "total_chunks": explanation["total_chunks"],
    }


def main() -> None:
    print("Structured range-query benchmark with chunk zone-map indexes")
    print(f"chunks={CHUNK_LEN:,}, blocks={BLOCK_LEN:,}, repeats={REPEATS}")
    print(
        "size,query_rows,build_s,scan_ms,index_ms,speedup,candidate_chunks,total_chunks"
    )
    for size in SIZES:
        result = benchmark_size(size)
        print(
            f"{result['size']},"
            f"{result['query_rows']},"
            f"{result['build_s']:.4f},"
            f"{result['scan_ms']:.3f},"
            f"{result['index_ms']:.3f},"
            f"{result['speedup']:.2f},"
            f"{result['candidate_chunks']},"
            f"{result['total_chunks']}"
        )


if __name__ == "__main__":
    main()
