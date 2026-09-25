"""Sample incremental RSS during streaming export, with a fresh process per run."""

import argparse
import json
import select
import statistics
import subprocess
import sys
from pathlib import Path

import psutil


def worker(path, size):
    import pyarrow as pa

    import blosc2

    # Initialize Arrow's Python conversion machinery before measuring export.
    pa.array([1, None])
    with blosc2.open(path, mode="r") as table:
        schema = table._arrow_schema_for_columns()
        print(json.dumps({"rows": len(table), "columns": len(schema)}), flush=True)
        input()
        rows = maximum = 0
        for batch in table.iter_arrow_batches(batch_size=size):
            rows += len(batch)
            maximum = max(maximum, batch.nbytes)
        assert rows == len(table)
        print(json.dumps({"rows": rows, "max_batch_bytes": maximum}), flush=True)
        input()  # Keep the process alive for the final RSS sample.


def measure(path, size):
    with subprocess.Popen(
        [sys.executable, __file__, "--worker", str(path), "--batch-size", str(size)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    ) as child:
        ready = child.stdout.readline()
        if not ready:
            raise RuntimeError("Memory worker failed before export")
        metadata = json.loads(ready)
        process = psutil.Process(child.pid)
        baseline = peak = process.memory_info().rss
        child.stdin.write("go\n")
        child.stdin.flush()
        while True:
            peak = max(peak, process.memory_info().rss)
            readable, _, _ = select.select([child.stdout], [], [], 0.001)
            if readable:
                result = json.loads(child.stdout.readline())
                peak = max(peak, process.memory_info().rss)
                break
        child.stdin.write("done\n")
        child.stdin.flush()
        assert child.wait() == 0
    return {
        **metadata,
        **result,
        "baseline_rss": baseline,
        "peak_rss": peak,
        "incremental_peak_rss": peak - baseline,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker")
    parser.add_argument("--batch-size", type=int, default=2048)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.batch_size)
        return
    prior = json.loads(Path("plans/direct-arrow-list-export-results.json").read_text())
    run = next(r for r in prior if r["stage"] == "after" and r["batch_size"] == 8192)
    tables = {
        "original-msgpack": Path("readings-idx.b2z"),
        "utf8-arrow": Path(run["output"]) / "utf8-arrow.b2z",
    }
    results = []
    for name, path in tables.items():
        for size in (2048, 8192, 65536):
            samples = [measure(path, size) for _ in range(3)]
            result = {
                "table": name,
                "path": str(path),
                "batch_size": size,
                "samples": samples,
                "median_incremental_peak_rss": statistics.median(
                    sample["incremental_peak_rss"] for sample in samples
                ),
            }
            results.append(result)
            print(json.dumps(result), flush=True)
    Path("plans/arrow-export-memory-results.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
