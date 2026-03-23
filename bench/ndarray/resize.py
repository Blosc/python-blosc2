#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

from __future__ import annotations

import argparse
import os
import time

import numpy as np

import blosc2


def parse_nitems(text: str) -> int:
    suffixes = {"k": 1_000, "m": 1_000_000, "g": 1_000_000_000}
    text = text.strip().lower()
    if text[-1:] in suffixes:
        return int(float(text[:-1]) * suffixes[text[-1]])
    return int(text)


def sizeof_path(path: str) -> int:
    if os.path.isdir(path):
        total = 0
        for root, _, files in os.walk(path):
            for name in files:
                total += os.path.getsize(os.path.join(root, name))
        return total
    return os.path.getsize(path)


def format_bytes(nbytes: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(nbytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{nbytes} B"


def pick_layout(nitems: int) -> tuple[tuple[int], tuple[int]]:
    chunks = (max(1, min(nitems, 16_384)),)
    blocks = (max(1, min(chunks[0], 256)),)
    return chunks, blocks


def create_extended_array(
    path: str, nitems: int, dtype: np.dtype, chunks: tuple[int], blocks: tuple[int], bsize: int
) -> blosc2.NDArray:
    array = blosc2.empty((0,), dtype=dtype, chunks=chunks, blocks=blocks, urlpath=path, mode="w")
    for start in range(0, nitems, bsize):
        stop = min(start + bsize, nitems)
        array.resize((stop,))
        array[start:stop] = np.arange(start, stop, dtype=dtype)
    return array


def create_full_array(path: str, data: np.ndarray, chunks: tuple[int], blocks: tuple[int]) -> blosc2.NDArray:
    return blosc2.asarray(data, chunks=chunks, blocks=blocks, urlpath=path, mode="w")


def time_random_access(array: blosc2.NDArray, indices: np.ndarray) -> tuple[float, int]:
    total = 0
    t0 = time.perf_counter_ns()
    for index in indices:
        total += int(array[int(index)])
    elapsed_ns = time.perf_counter_ns() - t0
    return elapsed_ns / len(indices) / 1_000_000, total


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare resizing an on-disk NDArray in batches vs creating it in one go."
    )
    parser.add_argument("--nitems", type=parse_nitems, default=parse_nitems("1M"))
    parser.add_argument("--bsize", type=parse_nitems, default=parse_nitems("1K"))
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", default="int64")
    parser.add_argument("--extended-path", default="resize-batched.b2nd")
    parser.add_argument("--full-path", default="resize-one-go.b2nd")
    args = parser.parse_args()

    dtype = np.dtype(args.dtype)
    chunks, blocks = pick_layout(args.nitems)
    data = np.arange(args.nitems, dtype=dtype)
    rng = np.random.default_rng(args.seed)
    indices = rng.integers(0, args.nitems, size=args.samples)

    for path in (args.extended_path, args.full_path):
        blosc2.remove_urlpath(path)

    t0 = time.perf_counter()
    extended = create_extended_array(args.extended_path, args.nitems, dtype, chunks, blocks, args.bsize)
    extend_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    full = create_full_array(args.full_path, data, chunks, blocks)
    full_time = time.perf_counter() - t0

    extended_size = sizeof_path(args.extended_path)
    full_size = sizeof_path(args.full_path)

    extended_access_ns, extended_checksum = time_random_access(extended, indices)
    full_access_ns, full_checksum = time_random_access(full, indices)

    print(f"nitems: {args.nitems:_}")
    print(f"dtype: {dtype}")
    print(f"chunks: {chunks}")
    print(f"blocks: {blocks}")
    print(f"batch size: {args.bsize:_}")
    print(f"resize build time: {extend_time:.3f} s")
    print(f"one-go build time: {full_time:.3f} s")
    print(f"resized array file size: {extended_size} bytes ({format_bytes(extended_size)})")
    print(f"one-go array file size: {full_size} bytes ({format_bytes(full_size)})")
    print(f"random access samples: {args.samples:_}")
    print(f"resized array random access: {extended_access_ns:.6f} ms/item")
    print(f"one-go array random access: {full_access_ns:.6f} ms/item")

    if extended_checksum != full_checksum:
        raise RuntimeError("Random-access checksums differ between arrays")


if __name__ == "__main__":
    main()
