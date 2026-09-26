"""Count in-memory Parquet read operations and bytes for representative tables."""

import io
import time
import tracemalloc

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq

import blosc2


def measure(name, table, row_group_size, column):
    output = io.BytesIO()
    pq.write_table(table, output, row_group_size=row_group_size)
    blob = output.getvalue()
    url = f"memory:///remote-parquet-{name}.parquet"
    fsspec.filesystem("memory").pipe(f"/remote-parquet-{name}.parquet", blob)
    tracemalloc.start()
    started = time.perf_counter()
    with blosc2.open(url) as remote:
        open_time = time.perf_counter() - started
        _, open_peak = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        opened = (remote.traffic.requests, remote.traffic.nbytes)
        started = time.perf_counter()
        remote[column][-1]
        cold_time = time.perf_counter() - started
        _, cold_peak = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        cold = (remote.traffic.requests - opened[0], remote.traffic.nbytes - opened[1])
        cache_bytes = remote.cache_bytes
        started = time.perf_counter()
        remote[column][-1]
        warm_time = time.perf_counter() - started
        warm = (remote.traffic.requests - opened[0] - cold[0], remote.traffic.nbytes - opened[1] - cold[1])
    tracemalloc.stop()
    print(
        f"{name}: file={len(blob)} B open={opened[0]} reads/{opened[1]} B/{open_time:.3f} s "
        f"cold={cold[0]} reads/{cold[1]} B/{cold_time:.3f} s "
        f"warm={warm[0]} reads/{warm[1]} B/{warm_time:.3f} s "
        f"cache={cache_bytes} B python_peak={max(open_peak, cold_peak)} B"
    )


if __name__ == "__main__":
    n = 10_000
    measure("scalar", pa.table({"x": list(range(n)), "y": list(range(n))}), 1_000, "x")
    measure("wide", pa.table({f"c{i}": list(range(n)) for i in range(20)}), 1_000, "c0")
    measure("strings", pa.table({"s": [f"value-{i:05d}" for i in range(n)]}), 1_000, "s")
    measure(
        "dictionary",
        pa.table({"d": pa.array([f"v{i % 100}" for i in range(n)]).dictionary_encode()}),
        1_000,
        "d",
    )
    root = pa.Table.from_arrays(
        [pa.array([[{"x": i}] for i in range(n)], type=pa.list_(pa.struct([("x", pa.int64())])))],
        names=[""],
    )
    measure("unnamed-root", root, 1_000, "x")
