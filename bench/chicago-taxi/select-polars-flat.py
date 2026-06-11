import argparse
import os
import time

parser = argparse.ArgumentParser(description="Filter the flat chicago-taxi parquet with polars.")
parser.add_argument("path", nargs="?", default="chicago-taxi-flat.parquet")
parser.add_argument(
    "--engine", choices=["streaming", "in-memory"], default="streaming",
    help="polars collect engine. 'streaming' processes the file in chunks with "
         "a bounded working set; 'in-memory' loads everything first.",
)
parser.add_argument(
    "--nthreads", type=int, default=0,
    help="size of polars' thread pool (0 = polars default = all cores). Must be "
         "set before polars is imported, so this is applied via POLARS_MAX_THREADS.",
)
args = parser.parse_args()

# Polars fixes its thread pool at import time, so set the env var first.
if args.nthreads > 0:
    os.environ["POLARS_MAX_THREADS"] = str(args.nthreads)

import polars as pl  # noqa: E402  (must come after POLARS_MAX_THREADS is set)

out = ["payment.tips", "payment.total", "trip.sec", "trip.km", "company"]

# Lazy scan: filter + projection push down into the Parquet reader, so only the
# needed columns are read and the predicate is applied during the scan -- the
# same lazy model as blosc2.where().  collect(engine="streaming") then runs it
# with a bounded-memory streaming executor instead of materializing everything.
t0 = time.perf_counter()
lf = (
    pl.scan_parquet(args.path)
    .filter(
        (pl.col("payment.tips") > 100)
        & (pl.col("trip.km") > 0)
        & (pl.col("trip.begin.lon") < 0)
    )
    .select(out)
    .sort("trip.sec")
)
t1 = time.perf_counter()

result = lf.collect(engine=args.engine)
t2 = time.perf_counter()

pdf = result.to_pandas()
print(pdf.to_string())
# Footer matching pandas' "[N rows x M columns]" so benchmark harnesses can do a
# cross-method row-count check (to_string() omits it).
print(f"[{len(pdf)} rows x {pdf.shape[1]} columns]")
t3 = time.perf_counter()

print(f"plan:    {t1 - t0:.6f} s")
print(f"compute: {t2 - t1:.6f} s  (engine={args.engine}, threads={pl.thread_pool_size()})")
print(f"print:   {t3 - t2:.6f} s")
print(f"total:   {t3 - t0:.6f} s")
