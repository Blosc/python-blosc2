#######################################################################
# Copyright (c) 2019-present, Blosc Development Team
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""Compare lazy access to the same array through fsspec and Caterva2.

The HTTPS path needs the fsspec extra.  Install it with:

    pip install "blosc2[fsspec]"

By default, caches are kept under ``./fsspec-cat2-cache``. Run the example again to
see the first data access served by the cache left by the previous process.
"""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np

import blosc2

# Using the Caterva2 API
CATERVA2_URL = blosc2.URLPath(
    "@public/examples/cube-1k-1k-1k.b2nd",
    urlbase="https://cat2.cloud/demo",
)
# ...and also using the fsspec path via fetch URL in Caterva2
FSSPEC_URL = "https://cat2.cloud/demo/api/fetch/@public/examples/cube-1k-1k-1k.b2nd"
# The same contents are published in this Backblaze B2 bucket with a ``-2`` suffix.
# FSSPEC_URL = "https://f001.backblazeb2.com/file/blosc2/cube-1k-1k-1k-2.b2nd"

SLICE = np.s_[100:110, 200:300, 400:500]


def traffic_text(traffic: blosc2.Traffic | None) -> str:
    if traffic is None:
        return "traffic unavailable"
    request_word = "request" if traffic.requests == 1 else "requests"
    return f"{traffic.requests} {request_word}, {traffic.nbytes / 2**20:.3f} MiB"


def size_text(size: int) -> str:
    return f"{size / 2**20:.3f} MiB"


def benchmark(label: str, urlpath, cache_dir: Path) -> np.ndarray:
    cache_existed = cache_dir.is_dir() and any(cache_dir.glob("*.b2nd"))

    start = perf_counter()
    array = blosc2.open(urlpath, lazy=True, cache_dir=cache_dir)
    open_time = perf_counter() - start
    open_traffic = traffic_text(array.traffic)

    metadata = (array.shape, array.dtype, array.chunks, array.blocks)
    cache_path = Path(array.urlpath).resolve()

    array.traffic.reset()
    start = perf_counter()
    data = array[SLICE]
    first_read_time = perf_counter() - start
    first_traffic = traffic_text(array.traffic)
    cache_size = cache_path.stat().st_size

    # Open a fresh remote handle over the same on-disk cache. This demonstrates
    # that cached data survives the Proxy object, not merely one array access.
    del array
    start = perf_counter()
    reopened = blosc2.open(urlpath, lazy=True, cache_dir=cache_dir)
    reopen_time = perf_counter() - start
    reopen_traffic = traffic_text(reopened.traffic)

    reopened.traffic.reset()
    start = perf_counter()
    cached = reopened[SLICE]
    cached_read_time = perf_counter() - start
    cached_traffic = traffic_text(reopened.traffic)
    np.testing.assert_array_equal(cached, data)

    print(f"\n{label}")
    print(f"  metadata: shape={metadata[0]}, dtype={metadata[1]}")
    print(f"            chunks={metadata[2]}, blocks={metadata[3]}")
    print(f"  persistent cache: {cache_path} ({'existing' if cache_existed else 'new'})")
    print(f"  {'open + remote metadata:':<27}{open_time * 1000:.0f} ms ({open_traffic})")
    print(f"  {'first data slice:':<27}{first_read_time * 1000:.0f} ms ({first_traffic})")
    print(f"  {'cache after slice:':<27}{size_text(cache_size)}")
    print(f"  {'reopen + remote metadata:':<27}{reopen_time * 1000:.0f} ms ({reopen_traffic})")
    print(f"  {'same slice after reopen:':<27}{cached_read_time * 1000:.0f} ms ({cached_traffic})")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("fsspec-cat2-cache"),
        help="persistent cache root (default: ./fsspec-cat2-cache)",
    )
    args = parser.parse_args()
    root = args.cache_dir

    print(f"Persistent cache root: {root.resolve()}")
    print("Run this command again to reuse these cache files.")
    cat2_data = benchmark("Caterva2", CATERVA2_URL, root / "caterva2")
    fsspec_data = benchmark("fsspec over HTTPS", FSSPEC_URL, root / "fsspec")
    np.testing.assert_array_equal(cat2_data, fsspec_data)
    print("\nBoth services returned identical data.")


if __name__ == "__main__":
    main()
