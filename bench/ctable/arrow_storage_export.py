"""Compare existing export paths; run from the repo root in the blosc2 env."""

import argparse
import copy
import dataclasses
import io
import json
import random
import runpy
import statistics
import tempfile
import time
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

import blosc2


def main():  # noqa: C901
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=2048)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    output = Path(tempfile.mkdtemp(prefix="blosc2-arrow-storage-"))
    reading = runpy.run_path("examples/ctable/remote_handling.py")["Reading"]
    source = blosc2.open("readings-idx.b2z", mode="r")
    reference = source.to_arrow()
    tables = {"original": source}
    for label in ("rebuilt-msgpack", "utf8-arrow"):
        fields = []
        for field in dataclasses.fields(reading):
            spec = copy.deepcopy(source._schema.columns_by_name[field.name].spec)
            if label == "utf8-arrow":
                if field.name == "message":
                    spec = blosc2.utf8(null_storage="mask")
                elif field.name == "tags":
                    spec.serializer = "arrow"
            fields.append((field.name, field.type, blosc2.field(spec)))
        row_type = dataclasses.make_dataclass("ReadingCopy", fields)
        path = output / f"{label}.b2z"
        with blosc2.CTable(
            row_type,
            urlpath=str(path),
            mode="w",
            expected_size=len(source),
            validate=False,
            create_summary_index=False,
        ) as table:
            for batch in reference.to_batches(max_chunksize=65536):
                table.extend(batch.to_pydict(), validate=False)
        tables[label] = blosc2.open(str(path), mode="r")
        assert tables[label].to_arrow().cast(reference.schema).equals(reference), label

    con = duckdb.connect(config={"threads": 4})
    sql = (
        "select status, count(*), avg(temperature) from data where humidity > 50 "
        "group by status order by status"
    )
    con.register("data", reference)
    expected = con.sql(sql).fetchall()
    con.unregister("data")
    samples = {}
    jobs = [
        (label, op) for label in tables for op in ("all", "note", "message", "tags", "duckdb", "parquet")
    ]
    rng = random.Random(42)
    for repetition in range(6):
        rng.shuffle(jobs)
        for label, op in jobs:
            table = tables[label]
            start = time.perf_counter()
            if op == "duckdb":
                reader = pa.RecordBatchReader.from_batches(
                    table._arrow_schema_for_columns(),
                    table.iter_arrow_batches(batch_size=args.batch_size),
                )
                con.register("data", reader)
                actual = con.sql(sql).fetchall()
                con.unregister("data")
            elif op == "parquet":
                buffer = io.BytesIO()
                table.to_parquet(buffer, batch_size=args.batch_size)
            else:
                rows = sum(
                    len(batch)
                    for batch in table.iter_arrow_batches(
                        columns=None if op == "all" else [op], batch_size=args.batch_size
                    )
                )
            elapsed = (time.perf_counter() - start) * 1000
            if op == "duckdb":
                for got, want in zip(actual, expected, strict=True):
                    assert got[:2] == want[:2]
                    assert abs(got[2] - want[2]) < 1e-10
            elif op == "parquet":
                assert (
                    pq.read_table(pa.BufferReader(buffer.getvalue()))
                    .cast(reference.schema)
                    .equals(reference)
                )
            else:
                assert rows == len(source)
            if repetition:
                samples.setdefault((label, op), []).append(elapsed)
    results = {
        "rows": len(source),
        "batch_size": args.batch_size,
        "repetitions": 5,
        "blosc2": blosc2.__version__,
        "pyarrow": pa.__version__,
        "duckdb": duckdb.__version__,
        "blosc2_threads": blosc2.nthreads,
        "duckdb_threads": 4,
        "output": str(output),
        "timings": [
            {"table": label, "operation": op, "median_ms": statistics.median(values), "samples_ms": values}
            for (label, op), values in sorted(samples.items())
        ],
    }
    (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2), flush=True)
    con.close()
    for table in tables.values():
        table.close()


if __name__ == "__main__":
    main()
