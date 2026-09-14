"""Evaluate a lazy expression over remote B2Z, Zarr, and HDF5 arrays.

Usage: python examples/remote/lazy-expr.py [--cache-dir CACHE_DIR]
"""

import argparse
import time
from contextlib import ExitStack

import blosc2


def open_array(stack, label, url, cache_dir):
    print(f"\n[{label}] opening {url}", flush=True)
    print("  Reading remote metadata (the first HDF5 open may take a while)...", flush=True)
    started = time.perf_counter()
    array = stack.enter_context(blosc2.open(url, lazy=True, cache_dir=cache_dir))
    elapsed = time.perf_counter() - started
    print(f"  Ready in {elapsed:.2f}s: shape={array.shape}, dtype={array.dtype}", flush=True)
    print(
        f"  Opening traffic: {array.traffic.requests} requests, {array.traffic.nbytes / 1024:.1f} KiB",
        flush=True,
    )
    return array


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        default="remote-cache",
        help="Persistent cache directory (default: remote-cache)",
    )
    args = parser.parse_args()

    base = "https://s3.us-west-001.backblazeb2.com/blosc2/hierarchy"
    print(f"Using persistent cache directory: {args.cache_dir}", flush=True)
    print("  It is kept after this run so later runs can reuse and inspect it.", flush=True)
    with ExitStack() as stack:
        # Open HDF5 before Zarr so its translation does not reset Zarr's shared runtime.
        h5 = open_array(stack, "HDF5", f"{base}.h5::/d0/d1/d2/a1", args.cache_dir)
        zarr = open_array(stack, "Zarr", f"{base}.zarr::/d0/d1/a1", args.cache_dir)
        b2z = open_array(stack, "B2Z", f"{base}.b2z::/d0/a1", args.cache_dir)

        print("\n[expression] building b2z + zarr * h5", flush=True)
        started = time.perf_counter()
        expression = b2z + zarr * h5
        print(f"  Built in {time.perf_counter() - started:.3f}s; no array data fetched (yet)", flush=True)

        print("\n[evaluation] requesting the first 10 values", flush=True)
        before = sum(array.traffic.nbytes for array in (h5, zarr, b2z))
        started = time.perf_counter()
        print(expression[:10], flush=True)
        elapsed = time.perf_counter() - started
        after = sum(array.traffic.nbytes for array in (h5, zarr, b2z))
        print(f"  Evaluated in {elapsed:.2f}s; fetched {(after - before) / 1024:.1f} KiB", flush=True)

        print("\n[evaluation] requesting the same slice again", flush=True)
        before = sum(array.traffic.nbytes for array in (h5, zarr, b2z))
        print(expression[:10], flush=True)
        after = sum(array.traffic.nbytes for array in (h5, zarr, b2z))
        print(f"  Fetched {(after - before) / 1024:.1f} KiB (zero means a cache hit)", flush=True)
        print("  Use expression.compute() to materialize the complete result.", flush=True)


if __name__ == "__main__":
    main()
