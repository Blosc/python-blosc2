"""Micro-benchmark comparing a normal build against an abi3 (Limited API) build.

Deliberately weighted towards *per-call* overhead, because that is where the
Limited API is expected to cost something: macros that used to be direct struct
accesses become real function calls, and `cdef class` instances become heap
types whose attribute lookups no longer go through a static type slot.

Bulk operations (large slices, big compress) are included as controls: they
spend nearly all their time inside C-Blosc2, so they should show no difference.
Any regression there would point at something other than the ABI.

Run inside a venv with blosc2 installed; writes JSON to stdout.
"""

import gc
import json
import statistics
import sys
import time

import numpy as np

import blosc2

REPEAT = 7  # timed repetitions; we report the minimum


def bench(fn, *, repeat=REPEAT):
    """Return the minimum wall time of `repeat` runs, in seconds."""
    gc.collect()
    gc.disable()
    try:
        times = []
        for _ in range(repeat):
            t0 = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t0)
    finally:
        gc.enable()
    return min(times), statistics.median(times)


RESULTS = {}


def record(name, fn, **kw):
    try:
        lo, med = bench(fn, **kw)
        RESULTS[name] = {"min": lo, "median": med}
        print(f"  {name:34s} {lo * 1e3:9.3f} ms", file=sys.stderr)
    except Exception as e:  # keep going; a missing API shouldn't kill the run
        RESULTS[name] = {"error": f"{type(e).__name__}: {e}"}
        print(f"  {name:34s} SKIP ({type(e).__name__}: {e})", file=sys.stderr)


# --------------------------------------------------------------------------
# per-call overhead: small payloads, many crossings of the Python/C boundary
# --------------------------------------------------------------------------

small = np.arange(1024, dtype=np.int64)  # 8 KB
small_bytes = small.tobytes()
small_c = blosc2.compress2(small_bytes)


def compress_small():
    for _ in range(5000):
        blosc2.compress2(small_bytes)


def decompress_small():
    for _ in range(5000):
        blosc2.decompress2(small_c)


record("compress2 8KB x5000", compress_small)
record("decompress2 8KB x5000", decompress_small)

# --------------------------------------------------------------------------
# cdef class attribute access -- SChunk is a `cdef class`, so under the Limited
# API it is built with PyType_FromSpec and its attributes are looked up through
# the generic heap-type path rather than a static slot.  This is the single
# most direct probe of the abi3 cost.
# --------------------------------------------------------------------------

schunk = blosc2.SChunk(chunksize=8 * 1024)
for _ in range(64):
    schunk.append_data(small)


def schunk_attrs():
    # Collected into a tuple rather than left as bare expressions so ruff's B018
    # stays quiet.  The extra tuple build is identical in both builds, so it
    # cancels in the ratio, which is all this script reports.
    s = schunk
    last = None
    for _ in range(200_000):
        last = (s.nchunks, s.cbytes, s.nbytes)
    return last


def schunk_decompress():
    s = schunk
    for i in range(64):
        s.decompress_chunk(i)


record("SChunk attr access x600k", schunk_attrs)
record("SChunk decompress_chunk x64", schunk_decompress)

# --------------------------------------------------------------------------
# NDArray: scalar getitem is call-overhead bound, big slice is C bound
# --------------------------------------------------------------------------

arr = blosc2.arange(0, 1000 * 1000, dtype=np.int64, shape=(1000, 1000))


def nd_scalar_getitem():
    a = arr
    for i in range(5000):
        a[i % 1000, 0]


def nd_big_slice():
    arr[:, :]


def nd_row_slices():
    a = arr
    for i in range(1000):
        a[i]


record("NDArray scalar getitem x5000", nd_scalar_getitem)
record("NDArray full slice (control)", nd_big_slice)
record("NDArray row slice x1000", nd_row_slices)

# --------------------------------------------------------------------------
# compute engine / lazy expressions
# --------------------------------------------------------------------------

a = blosc2.linspace(0, 1, 4_000_000, dtype=np.float64, shape=(2000, 2000))
b = blosc2.linspace(1, 2, 4_000_000, dtype=np.float64, shape=(2000, 2000))


def lazyexpr_eval():
    (a**2 + b * 2).compute()


def lazyexpr_where():
    blosc2.where(a > 0.5, a, b).compute()


def reduction_sum():
    (a + b).sum()


record("lazyexpr a**2+b*2 (4M f64)", lazyexpr_eval, repeat=5)
record("where(a>0.5,a,b) (4M f64)", lazyexpr_where, repeat=5)
record("sum(a+b) (4M f64)", reduction_sum, repeat=5)

# --------------------------------------------------------------------------
# bulk compress control -- almost entirely inside C-Blosc2
# --------------------------------------------------------------------------

big = np.arange(8 * 1024 * 1024, dtype=np.int64)  # 64 MB
big_bytes = big.tobytes()


def compress_big():
    blosc2.compress2(big_bytes)


record("compress2 64MB (control)", compress_big, repeat=5)

# --------------------------------------------------------------------------
# CTable: utf8 ingest, groupby and where().  These exercise utf8_ext,
# groupby_ext and indexing_ext, which the earlier set only touched indirectly.
# --------------------------------------------------------------------------

try:
    from dataclasses import dataclass

    @dataclass
    class SalesRow:
        city: str = blosc2.field(blosc2.utf8())
        category: int = blosc2.field(blosc2.int32())
        sales: float = blosc2.field(blosc2.float64(), default=0.0)
        qty: int = blosc2.field(blosc2.int32(), default=0)

    CITIES = ["Paris", "Rome", "Berlin", "Madrid", "Lisbon", "Vienna", "Oslo", "Prague"]
    NROWS = 200_000
    rng = np.random.default_rng(42)
    ROWS = [
        (
            CITIES[i % len(CITIES)],
            int(rng.integers(0, 8)),
            float(i % 1000),
            int(i % 97),
        )
        for i in range(NROWS)
    ]

    def utf8_ingest():
        blosc2.CTable(SalesRow, new_data=ROWS)

    record("CTable utf8 ingest 200k rows", utf8_ingest, repeat=3)

    table = blosc2.CTable(SalesRow, new_data=ROWS)

    def ctable_groupby():
        table.group_by("city", sort=True).agg({"sales": ["sum", "mean", "count"]})

    def ctable_groupby_multi():
        table.group_by(["city", "category"], sort=True).size()

    def ctable_where():
        table.where(table["sales"] > 500.0)

    record("CTable group_by+agg 200k", ctable_groupby, repeat=5)
    record("CTable group_by 2 keys 200k", ctable_groupby_multi, repeat=5)
    record("CTable where() 200k", ctable_where, repeat=5)
except Exception as e:
    print(f"  CTable benchmarks skipped: {type(e).__name__}: {e}", file=sys.stderr)

# --------------------------------------------------------------------------
# provenance: prove which build actually got measured
# --------------------------------------------------------------------------

meta = {
    "python": sys.version.split()[0],
    "blosc2": blosc2.__version__,
    "numpy": np.__version__,
}
try:
    from blosc2 import blosc2_ext

    meta["ext_file"] = blosc2_ext.__file__
    meta["abi3"] = ".abi3." in blosc2_ext.__file__
except Exception as e:
    meta["ext_file"] = f"unknown: {e}"

print(json.dumps({"meta": meta, "results": RESULTS}, indent=2))
