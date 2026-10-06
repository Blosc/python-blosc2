"""Benchmark DuckDB <-> CTable conversions with bool/int64/UTF-8 columns.

Run in the blosc2 environment::

    conda run -n blosc2 python bench/ctable/from_to_duckdb.py

DuckDB input is an already materialized in-memory table. Imports include query
execution and CTable compression. Exports create a materialized DuckDB table,
not merely a lazy registration; the Arrow-table route includes to_arrow().
Generation, validation, GC and cleanup are outside timings. Every result is
checked against the original Arrow input. Report medians after one warmup.
"""

import argparse
import gc
import platform
import statistics
from time import perf_counter

import duckdb
import numpy as np
import pyarrow as pa
from from_to_arrow import check_result, make_arrow

import blosc2


# Observations from a Darwin arm64 run (Python 3.14.4, DuckDB 1.5.2,
# PyArrow 24.0.0, 8 threads, median of 5 runs after warmup):
# At 100,000 rows, DuckDB -> CTable took 11.27 ms via direct PyCapsule,
# 10.89 ms via an Arrow table, and 16.04 ms via a 65,536-row Arrow reader.
# CTable -> materialized DuckDB took 7.88 ms directly versus 7.82 ms via
# to_arrow(). Direct streams use provider defaults, so the reader comparison
# also reflects different batch boundaries, not just interface overhead.
# DuckDB already uses Arrow interchange and our direct UTF-8 buffer writer.
# These results reveal no obvious additional DuckDB-specific fast path.
# Batch sizing and compressed-storage writes are candidates for further work,
# but profiling would be needed to establish the remaining bottlenecks.


def measure(operation, source, connection, repeats):
    samples = []
    for repetition in range(repeats + 1):
        gc.collect()
        start = perf_counter()
        result = operation()
        elapsed = perf_counter() - start
        try:
            actual = result.to_arrow_table() if isinstance(result, duckdb.DuckDBPyRelation) else result
            check_result(actual, source)
        finally:
            if isinstance(result, blosc2.CTable):
                result.close()
            else:
                connection.execute("DROP TABLE IF EXISTS output_table")
                connection.unregister("input_data")
            del result
        if repetition:
            samples.append(elapsed)
    return statistics.median(samples)


def materialize(connection, table, via_arrow):
    source = table.to_arrow() if via_arrow else table
    connection.register("input_data", source)
    connection.execute("CREATE TEMP TABLE output_table AS SELECT * FROM input_data")
    return connection.table("output_table")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1_000, 10_000, 100_000])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=65_536)
    parser.add_argument("--nthreads", type=int, default=blosc2.nthreads)
    args = parser.parse_args()
    if min(*args.sizes, args.repeats, args.batch_size, args.nthreads) < 1:
        parser.error("sizes, repeats, batch-size and nthreads must be positive")
    blosc2.set_nthreads(args.nthreads)
    print(f"Python {platform.python_version()} | {platform.system()} {platform.machine()}")
    print(f"Blosc2 {blosc2.__version__} | DuckDB {duckdb.__version__} | PyArrow {pa.__version__}")
    print(f"NumPy {np.__version__} | threads: {blosc2.nthreads} | repeats: {args.repeats} + warmup")
    print(f"Explicit reader batch size: {args.batch_size:,}; direct streams use provider defaults")
    print("Schema: flag: bool, count: int64, name: utf8 (10,000 labels; no nulls)")
    print("CTable -> DuckDB timings include CREATE TABLE AS SELECT (full materialization).")
    print(f"\n{'Rows':>10}  {'Operation':<44}  {'Median ms':>10}  {'Mrows/s':>9}")
    with duckdb.connect(config={"threads": args.nthreads}) as connection:
        for rows in args.sizes:
            source = make_arrow(rows, args.batch_size)
            connection.register("arrow_source", source)
            connection.execute("CREATE OR REPLACE TEMP TABLE source_table AS SELECT * FROM arrow_source")
            connection.unregister("arrow_source")
            with blosc2.CTable.from_arrow(source) as table:
                jobs = [
                    (
                        "DuckDB -> CTable (direct PyCapsule)",
                        lambda: blosc2.CTable.from_arrow(connection.table("source_table")),
                    ),
                    (
                        "DuckDB -> CTable (Arrow reader)",
                        lambda: blosc2.CTable.from_arrow(
                            connection.table("source_table").to_arrow_reader(args.batch_size)
                        ),
                    ),
                    (
                        "DuckDB -> CTable (Arrow table)",
                        lambda: blosc2.CTable.from_arrow(connection.table("source_table").to_arrow_table()),
                    ),
                    ("CTable -> DuckDB (direct PyCapsule)", lambda: materialize(connection, table, False)),
                    ("CTable -> DuckDB (via to_arrow)", lambda: materialize(connection, table, True)),
                ]
                for name, operation in jobs:
                    elapsed = measure(operation, source, connection, args.repeats)
                    print(f"{rows:>10,}  {name:<44}  {elapsed * 1000:>10.2f}  {rows / elapsed / 1e6:>9.2f}")
            connection.execute("DROP TABLE source_table")


if __name__ == "__main__":
    main()
