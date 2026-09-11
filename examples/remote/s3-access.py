#!/usr/bin/env python3
#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""Open a remote S3 array (Blosc2 .b2nd/.b2z, Zarr .zarr, or HDF5 .h5) and print sample data.

Usage:
    python s3-access.py <url> [--profile PROFILE] [--endpoint-url ENDPOINT_URL]

Examples:
    python s3-access.py s3://blosc2/cube-1k-1k-1k.b2nd
    python s3-access.py s3://blosc2/cube-1k-1k-1k.zarr
    python s3-access.py s3://blosc2/cube-1k-1k-1k-1shard.zarr
    python s3-access.py s3://blosc2/hierarchy.zarr/d0/d1/a2
    python s3-access.py s3://blosc2/hierarchy.zarr::d0/d1/a2
    python s3-access.py s3://blosc2/hierarchy.h5/d0/d1/a2
    python s3-access.py s3://blosc2/hierarchy.h5::d0/d1/a2
    python s3-access.py s3://blosc2/hierarchy.b2z::/d0/a3
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any

import blosc2

DEFAULT_PROFILE = "blosc2"
DEFAULT_ENDPOINT_URL = "https://s3.us-west-001.backblazeb2.com"


def get_sample_slice(arr: Any) -> Any:
    """Extract a representative small sample slice regardless of array dimensionality."""
    ndim = getattr(arr, "ndim", None)
    if ndim is None:
        ndim = len(arr.shape) if hasattr(arr, "shape") else 0
    if ndim == 0:
        return arr[()]
    if ndim == 1:
        return arr[: min(10, arr.shape[0])]
    if ndim == 2:
        return arr[: min(10, arr.shape[0]), : min(5, arr.shape[1])]

    idx: list[Any] = [slice(0, min(10, arr.shape[0]))]
    for _ in range(ndim - 2):
        idx.append(0)
    idx.append(slice(0, min(5, arr.shape[-1])))
    return arr[tuple(idx)]


class Traffic:
    """Track data transferred over the network."""

    def __init__(self, nbytes: int = 0) -> None:
        self.nbytes = nbytes

    def charge(self, n: int) -> None:
        self.nbytes += n


class TrackingFile:
    """Wrapper around a file-like object to track bytes read."""

    def __init__(self, f: Any, traffic: Traffic) -> None:
        self._f = f
        self._traffic = traffic

    def read(self, *args: Any, **kwargs: Any) -> Any:
        data = self._f.read(*args, **kwargs)
        if data:
            self._traffic.charge(len(data))
        return data

    def readinto(self, b: Any) -> Any:
        n = self._f.readinto(b)
        if n:
            self._traffic.charge(n)
        return n

    def __getattr__(self, name: str) -> Any:
        return getattr(self._f, name)


def open_remote_array(
    url: str,
    profile: str = DEFAULT_PROFILE,
    endpoint_url: str = DEFAULT_ENDPOINT_URL,
) -> tuple[str, Any]:
    """Open remote array depending on extension (.b2nd vs .zarr/.zip vs .h5).

    Returns (format_name, array_object).
    """
    storage_options = {
        "profile": profile,
        "endpoint_url": endpoint_url,
    }

    clean_url = url.split("::", 1)[0].rstrip("/")
    if clean_url.endswith((".zarr.zip", ".zip")):
        import fsspec
        import zarr
        from zarr.storage import ZipStore

        traffic = Traffic()
        raw_f = fsspec.open(url, "rb", **storage_options).open()
        tf = TrackingFile(raw_f, traffic)
        store = ZipStore(tf, mode="r")
        arr = zarr.open(store=store)
        arr.traffic = traffic
        return "Zarr (Zip)", arr

    # If the URL targets an HDF5 container without a dataset path, list available datasets
    base_url, detected_dataset, hint = blosc2.remote_array.parse_container_url(url)
    if hint == "hdf5" and detected_dataset is None:
        available = blosc2.available_datasets(base_url, storage_options=storage_options)
        raise ValueError(
            f"HDF5 files require specifying the dataset path using '/dataset_name' or '::dataset_name' "
            f"(e.g. {base_url}/d0/d1/a2 or {base_url}::d0/d1/a2). Available datasets: {available}"
        )

    arr = blosc2.open(url, lazy=True, storage_options=storage_options)
    kind = arr.source.get("kind", "")
    if kind == "hdf5":
        label = "HDF5"
    elif kind == "zarr":
        label = "Zarr"
    elif kind == "b2z":
        label = "Blosc2 B2Z"
    else:
        label = "Blosc2"
    return f"{label} (Lazy RemoteArray)", arr


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Open a remote array in S3 (.b2nd or .zarr) and print metadata and data.",
    )
    parser.add_argument("url", help="Remote S3 URL (e.g. s3://blosc2/cube.b2nd or s3://blosc2/cube.zarr)")
    parser.add_argument(
        "--profile",
        default=DEFAULT_PROFILE,
        help=f"AWS CLI credential profile (default: '{DEFAULT_PROFILE}')",
    )
    parser.add_argument(
        "--endpoint-url",
        default=DEFAULT_ENDPOINT_URL,
        help=f"S3 endpoint URL (default: '{DEFAULT_ENDPOINT_URL}')",
    )

    args = parser.parse_args()

    print(f"Accessing: {args.url}")
    t0 = time.perf_counter()
    try:
        fmt, arr = open_remote_array(
            url=args.url,
            profile=args.profile,
            endpoint_url=args.endpoint_url,
        )
    except Exception as exc:
        print(f"Error opening remote array: {exc}", file=sys.stderr)
        return 1
    t_open = time.perf_counter() - t0

    def get_traffic_bytes() -> int | None:
        traffic = getattr(arr, "traffic", None)
        if traffic is None:
            src = getattr(arr, "src", None)
            traffic = getattr(src, "traffic", None)
        if traffic is not None and hasattr(traffic, "nbytes"):
            return int(traffic.nbytes)
        return None

    b_open = get_traffic_bytes()

    print(f"\n[Format: {fmt}]")
    if hasattr(arr, "info"):
        print(arr.info, end="")
    else:
        print(f"{'shape':<12} : {arr.shape}")
        print(f"{'dtype':<12} : {arr.dtype}")
        chunks = getattr(arr, "chunks", None)
        if chunks is not None:
            print(f"{'chunks':<12} : {chunks}")
        blocks = getattr(arr, "blocks", None)
        if blocks is not None:
            print(f"{'blocks':<12} : {blocks}")
    meta = getattr(arr, "meta", None)
    if meta:
        print(f"{'meta':<12} : {dict(meta)}")
    vlmeta = getattr(arr, "vlmeta", None)
    if vlmeta is not None:
        print(f"{'vlmeta':<12} : {dict(vlmeta) if vlmeta else {}}")

    print("\nSample slice data (1st fetch):")
    t0 = time.perf_counter()
    sample = get_sample_slice(arr)
    t_fetch1 = time.perf_counter() - t0
    b_after_fetch1 = get_traffic_bytes()
    b_fetch1 = (b_after_fetch1 - b_open) if (b_after_fetch1 is not None and b_open is not None) else None
    print(sample)

    # Re-fetch the same slice to test caching behavior
    t0 = time.perf_counter()
    _ = get_sample_slice(arr)
    t_fetch2 = time.perf_counter() - t0
    b_after_fetch2 = get_traffic_bytes()
    b_fetch2 = (
        (b_after_fetch2 - b_after_fetch1)
        if (b_after_fetch2 is not None and b_after_fetch1 is not None)
        else None
    )

    print("\nTiming & Network Traffic:")
    if b_open is not None:
        print(f"  - Metadata open  : {t_open * 1000:7.1f} ms  ({b_open / 1024:8.2f} KB transferred)")
    else:
        print(f"  - Metadata open  : {t_open * 1000:7.1f} ms")

    if b_fetch1 is not None:
        print(f"  - 1st slice fetch: {t_fetch1 * 1000:7.1f} ms  ({b_fetch1 / 1024:8.2f} KB transferred)")
    else:
        print(f"  - 1st slice fetch: {t_fetch1 * 1000:7.1f} ms")

    if b_fetch2 is not None:
        tag = " (cache hit!)" if b_fetch2 == 0 else ""
        print(
            f"  - 2nd slice fetch: {t_fetch2 * 1000:7.1f} ms  ({b_fetch2 / 1024:8.2f} KB transferred){tag}"
        )
    else:
        print(f"  - 2nd slice fetch: {t_fetch2 * 1000:7.1f} ms")

    if b_after_fetch2 is not None:
        print(f"  - Total network  : {b_after_fetch2 / 1024:8.2f} KB transferred from S3")

    return 0


if __name__ == "__main__":
    sys.exit(main())
