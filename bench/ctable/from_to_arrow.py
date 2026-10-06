"""Time in-memory CTable <-> PyArrow conversions, including PyCapsule streams.

Run from the repository root in the blosc2 environment::

    conda run -n blosc2 python bench/ctable/from_to_arrow.py

Inputs have bool, int64 and string columns. Data generation, correctness checks,
GC and CTable cleanup are outside timings. Each operation gets one warmup, then
the median of repeated runs is reported. Import includes compression and table
creation; export includes decompression. No disk I/O or pandas conversion.
Streaming exports drain the reader, rather than just timing capsule creation.
"""

import argparse
import gc
import platform
import statistics
from time import perf_counter

import numpy as np
import pyarrow as pa

import blosc2


def make_arrow(rows, batch_size):
    rng = np.random.default_rng(42)
    labels = np.array([f"label-{i:05d}" for i in range(10_000)])
    table = pa.table(
        {
            "flag": rng.integers(0, 2, rows, dtype=np.int8).astype(bool),
            "count": np.arange(rows, dtype=np.int64),
            "name": labels[rng.integers(0, len(labels), rows)],
        }
    )
    # Both ingestion APIs see the same input batch boundaries.
    return pa.Table.from_batches(table.to_batches(max_chunksize=batch_size), schema=table.schema)


def drain_batches(batches):
    return sum(batch.num_rows for batch in batches)


def check_result(result, source):
    if isinstance(result, int):
        assert result == source.num_rows
    else:
        actual = result.to_arrow() if isinstance(result, blosc2.CTable) else result
        assert actual.cast(source.schema).equals(source, check_metadata=False)


def measure(operation, source, repeats):
    samples = []
    for repetition in range(repeats + 1):
        gc.collect()
        start = perf_counter()
        result = operation()
        elapsed = perf_counter() - start
        try:
            check_result(result, source)
        finally:
            if isinstance(result, blosc2.CTable):
                result.close()
            del result
        if repetition:  # Discard the warmup.
            samples.append(elapsed)
    return statistics.median(samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1_000, 10_000, 100_000])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=65_536)
    parser.add_argument("--nthreads", type=int, default=blosc2.nthreads)
    args = parser.parse_args()
    if min(*args.sizes, args.repeats, args.batch_size, args.nthreads) < 1:
        parser.error("sizes, repeats, batch-size and nthreads must be positive")
    blosc2.set_nthreads(args.nthreads)
    print(f"Python {platform.python_version()} | {platform.system()} {platform.machine()}")
    print(f"Blosc2 {blosc2.__version__} | PyArrow {pa.__version__} | NumPy {np.__version__}")
    print(f"Threads: {blosc2.nthreads} | repeats: {args.repeats} + warmup | batch size: {args.batch_size:,}")
    print("Schema: flag: bool, count: int64, name: string (10,000 distinct labels; no nulls)")
    print("to_arrow()/pa.table()/capsule export use CTable's default batch size.")
    print(
        f"\n{'Rows':>10}  {'Arrow MiB':>9}  {'Operation':<37}  "
        f"{'Median ms':>10}  {'Mrows/s':>9}  {'MiB/s':>9}"
    )
    for rows in args.sizes:
        source = make_arrow(rows, args.batch_size)
        batches = source.to_batches()
        with blosc2.CTable.from_arrow(source) as table:
            jobs = [
                (
                    "Arrow -> CTable (schema + batches)",
                    lambda source=source, batches=batches: blosc2.CTable.from_arrow(source.schema, batches),
                ),
                ("Arrow -> CTable (PyCapsule)", lambda source=source: blosc2.CTable.from_arrow(source)),
                ("CTable -> Arrow (to_arrow)", table.to_arrow),
                ("CTable -> Arrow (pa.table/PyCapsule)", lambda table=table: pa.table(table)),
                (
                    "CTable -> batches (explicit drain)",
                    lambda table=table: drain_batches(table.iter_arrow_batches(batch_size=args.batch_size)),
                ),
                (
                    "CTable -> batches (PyCapsule drain)",
                    lambda table=table: drain_batches(pa.RecordBatchReader.from_stream(table)),
                ),
            ]
            for name, operation in jobs:
                elapsed = measure(operation, source, args.repeats)
                mib = source.nbytes / 2**20
                print(
                    f"{rows:>10,}  {mib:>9.2f}  {name:<37}  {elapsed * 1000:>10.2f}  "
                    f"{rows / elapsed / 1e6:>9.2f}  {mib / elapsed:>9.2f}"
                )
        del source, batches, jobs, operation


if __name__ == "__main__":
    main()
