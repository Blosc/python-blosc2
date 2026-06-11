import argparse
import time

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds

parser = argparse.ArgumentParser(description="Filter the flat chicago-taxi parquet.")
parser.add_argument("path", nargs="?", default="chicago-taxi-flat.parquet")
parser.add_argument(
    "--nthreads", type=int, default=0,
    help="size of Arrow's CPU thread pool used by the scanner. 0 = Arrow "
         "default = all cores in the box (mirrors polars' --nthreads 0); "
         "1 = single threaded (lowest memory); higher values cap the pool. "
         "Unlike the nested script this is not a Python worker count: the "
         "scanner streams batches and Arrow parallelizes them across this pool.",
)
args = parser.parse_args()
path = args.path

# Arrow's global CPU pool already defaults to pa.cpu_count() (all cores), so
# 0 just means "leave it at the default".  Only override when given an
# explicit cap, and only run serially when the user asks for a single thread.
if args.nthreads > 0:
    pa.set_cpu_count(args.nthreads)
nthreads = 1 if args.nthreads == 1 else pa.cpu_count()
use_threads = nthreads > 1

out = ["payment.tips", "payment.total", "trip.sec", "trip.km", "company"]

# Filter pushdown: the predicate is handed to the scanner so reading,
# decompression and filtering happen lazily inside to_table() -- mirroring
# blosc2's where(), where "open" is just metadata and the work lands in
# "compute".  Note: pc.field("trip.km") is a single dotted name, NOT nested.
predicate = (
    (pc.field("payment.tips") > 100)
    & (pc.field("trip.km") > 0)
    & (pc.field("trip.begin.lon") < 0)
)

t0 = time.perf_counter()
# pre_buffer=False stops Parquet from coalescing whole column chunks into RAM
# before decoding; bounded readahead keeps only a couple of row groups in
# flight.  Together they cut peak memory ~485 -> ~275 MB at the same speed.
# (The ~115 MB floor is just importing Arrow's C++ libraries.)
scan_opts = ds.ParquetFragmentScanOptions(pre_buffer=False)
fmt = ds.ParquetFileFormat(default_fragment_scan_options=scan_opts)
dataset = ds.dataset(path, format=fmt)
t1 = time.perf_counter()

scanner = dataset.scanner(
    columns=out, filter=predicate, use_threads=use_threads,
    batch_size=65536, batch_readahead=2, fragment_readahead=2,
)
result = scanner.to_table().sort_by("trip.sec")
t2 = time.perf_counter()

print(result.to_pandas())
t3 = time.perf_counter()
print(f"open:    {t1 - t0:.6f} s")
print(f"compute: {t2 - t1:.6f} s  (nthreads={nthreads})")
print(f"print:   {t3 - t2:.6f} s")
print(f"total:   {t3 - t0:.6f} s")
