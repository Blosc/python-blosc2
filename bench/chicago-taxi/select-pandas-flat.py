import argparse
import time

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)

parser = argparse.ArgumentParser(description="Filter the flat chicago-taxi parquet with pandas.")
parser.add_argument("path", nargs="?", default="chicago-taxi-flat.parquet")
parser.add_argument(
    "--nthreads", type=int, default=1,
    help="threads for the PyArrow Parquet read only. pandas' filter/sort is "
         "single-threaded numpy regardless, so this affects load time, not compute.",
)
args = parser.parse_args()

# pandas itself is single-threaded here; the only parallelism is the columnar
# read via PyArrow's CPU thread pool.
pa.set_cpu_count(max(1, args.nthreads))

# Flat file: columns are already flat (no list<struct>), so there is nothing to
# flatten -- just read the columns we need and hand them to pandas.  We read the
# filter column (trip.begin.lon) too, then drop it from the output.
out = ["payment.tips", "payment.total", "trip.sec", "trip.km", "company"]
need = out + ["trip.begin.lon"]

t0 = time.perf_counter()
df = pq.read_table(args.path, columns=need, use_threads=args.nthreads > 1).to_pandas()
t1 = time.perf_counter()

mask = (df["payment.tips"] > 100) & (df["trip.km"] > 0) & (df["trip.begin.lon"] < 0)
result = (
    df.loc[mask, out]
    .sort_values("trip.sec", kind="stable", ignore_index=True)
)
t2 = time.perf_counter()

print(result)
t3 = time.perf_counter()

print(f"load:    {t1 - t0:.6f} s  (read threads={args.nthreads})")
print(f"compute: {t2 - t1:.6f} s")
print(f"print:   {t3 - t2:.6f} s")
print(f"total:   {t3 - t0:.6f} s")
