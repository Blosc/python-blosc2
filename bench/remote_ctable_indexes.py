"""Compare indexed and forced-scan RemoteCTable query traffic.

Run with the repository's development environment::

    conda run -n blosc2 python bench/remote_ctable_indexes.py --rows 200000
"""

from __future__ import annotations

import argparse
import dataclasses
import tempfile
import time
from pathlib import Path

import numpy as np

import blosc2


@dataclasses.dataclass
class Row:
    value: int = blosc2.field(blosc2.int64(), chunks=(65536,), blocks=(4096,))


def measure(url: str, threshold: int, *, use_index: bool) -> tuple[int, float, int, int]:
    with blosc2.RemoteCTable(url, cache_policy=blosc2.CachePolicy.NONE) as table:
        table.traffic.reset()
        start = time.perf_counter()
        expr = table.value >= threshold
        result = table[expr].value[:] if use_index else expr.compute()[:]
        elapsed = time.perf_counter() - start
        count = len(result) if use_index else int(np.count_nonzero(result))
        return count, elapsed, table.traffic.requests, table.traffic.nbytes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=200_000)
    parser.add_argument("--threshold", type=int)
    args = parser.parse_args()
    threshold = args.threshold if args.threshold is not None else args.rows - max(1, args.rows // 100)

    import fsspec

    with tempfile.TemporaryDirectory(prefix="remote-ctable-index-") as tmp:
        path = Path(tmp) / "indexed.b2z"
        with blosc2.CTable(
            Row,
            [(i,) for i in range(args.rows)],
            urlpath=path,
            mode="w",
            create_summary_index=False,
        ) as table:
            table.create_index("value", kind="summary")
        url = "memory://remote-ctable-index-benchmark.b2z"
        fsspec.filesystem("memory").pipe(url, path.read_bytes())

        for label, use_index in (("indexed", True), ("scan", False)):
            count, elapsed, requests, nbytes = measure(url, threshold, use_index=use_index)
            print(f"{label:7} rows={count:,} time={elapsed:.4f}s requests={requests:,} bytes={nbytes:,}")


if __name__ == "__main__":
    main()
