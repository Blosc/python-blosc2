#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# This benchmarks BatchArray random single-item reads. It supports
# msgpack or arrow, configurable codec/compression level, optional
# dictionary compression, and in-memory vs persistent mode.

from __future__ import annotations

import argparse
import random
import statistics
import time

import blosc2


URLPATH = "bench_batch_array.b2b"
NBATCHES = 10_000
OBJECTS_PER_BATCH = 100
TOTAL_OBJECTS = NBATCHES * OBJECTS_PER_BATCH
ITEMS_PER_BLOCK = 32
N_RANDOM_READS = 1_000


def make_rgb(batch_index: int, item_index: int) -> dict[str, int]:
    global_index = batch_index * OBJECTS_PER_BATCH + item_index
    return {
        "red": batch_index,
        "green": item_index,
        "blue": global_index,
    }


def make_batch(batch_index: int) -> list[dict[str, int]]:
    return [make_rgb(batch_index, item_index) for item_index in range(OBJECTS_PER_BATCH)]


def expected_entry(batch_index: int, item_index: int) -> dict[str, int]:
    return {
        "red": batch_index,
        "green": item_index,
        "blue": batch_index * OBJECTS_PER_BATCH + item_index,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark BatchArray single-entry reads.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--codec", type=str, default="ZSTD", choices=[codec.name for codec in blosc2.Codec])
    parser.add_argument("--clevel", type=int, default=5)
    parser.add_argument("--serializer", type=str, default="msgpack", choices=["msgpack", "arrow"])
    parser.add_argument("--use-dict", action="store_true", help="Enable dictionaries for ZSTD/LZ4/LZ4HC codecs.")
    parser.add_argument("--in-mem", action="store_true", help="Keep the BatchArray purely in memory.")
    return parser


def build_array(
    codec: blosc2.Codec, clevel: int, use_dict: bool, serializer: str, in_mem: bool
) -> blosc2.BatchArray | None:
    if in_mem:
        storage = blosc2.Storage(mode="w")
        barr = blosc2.BatchArray(
            storage=storage,
            items_per_block=ITEMS_PER_BLOCK,
            serializer=serializer,
            cparams={
                "codec": codec,
                "clevel": clevel,
                "use_dict": use_dict and codec in (blosc2.Codec.ZSTD, blosc2.Codec.LZ4, blosc2.Codec.LZ4HC),
            },
        )
        for batch_index in range(NBATCHES):
            barr.append(make_batch(batch_index))
        return barr

    blosc2.remove_urlpath(URLPATH)
    storage = blosc2.Storage(urlpath=URLPATH, mode="w", contiguous=True)
    cparams = {
        "codec": codec,
        "clevel": clevel,
        "use_dict": use_dict and codec in (blosc2.Codec.ZSTD, blosc2.Codec.LZ4, blosc2.Codec.LZ4HC),
    }
    with blosc2.BatchArray(
        storage=storage, items_per_block=ITEMS_PER_BLOCK, serializer=serializer, cparams=cparams
    ) as barr:
        for batch_index in range(NBATCHES):
            barr.append(make_batch(batch_index))
    return None


def measure_random_reads(barr: blosc2.BatchArray) -> tuple[list[tuple[int, int, int, dict[str, int]]], list[int]]:
    rng = random.Random(2024)
    samples: list[tuple[int, int, int, dict[str, int]]] = []
    timings_ns: list[int] = []

    for _ in range(N_RANDOM_READS):
        batch_index = rng.randrange(len(barr))
        item_index = rng.randrange(OBJECTS_PER_BATCH)
        t0 = time.perf_counter_ns()
        value = barr[batch_index][item_index]
        timings_ns.append(time.perf_counter_ns() - t0)
        if value != expected_entry(batch_index, item_index):
            raise RuntimeError(f"Value mismatch at batch={batch_index}, item={item_index}")
        samples.append((timings_ns[-1], batch_index, item_index, value))

    return samples, timings_ns


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    codec = blosc2.Codec[args.codec]
    use_dict = args.use_dict and codec in (blosc2.Codec.ZSTD, blosc2.Codec.LZ4, blosc2.Codec.LZ4HC)

    mode_label = "in-memory" if args.in_mem else "persistent"
    article = "an" if args.in_mem else "a"
    print(f"Building {article} {mode_label} BatchArray with 1,000,000 RGB dicts and timing 1,000 random scalar reads...")
    print(f"  codec: {codec.name}")
    print(f"  clevel: {args.clevel}")
    print(f"  serializer: {args.serializer}")
    print(f"  use_dict: {use_dict}")
    print(f"  in_mem: {args.in_mem}")
    t0 = time.perf_counter()
    barr = build_array(
        codec=codec, clevel=args.clevel, use_dict=use_dict, serializer=args.serializer, in_mem=args.in_mem
    )
    build_time_s = time.perf_counter() - t0
    if args.in_mem:
        assert barr is not None
        read_array = barr
    else:
        read_array = blosc2.BatchArray(urlpath=URLPATH, mode="r", contiguous=True, items_per_block=ITEMS_PER_BLOCK)
    samples, timings_ns = measure_random_reads(read_array)
    t0 = time.perf_counter()
    checksum = 0
    nitems = 0
    for item in read_array.iter_items():
        checksum += item["blue"]
        nitems += 1
    iter_time_s = time.perf_counter() - t0

    print()
    print("BatchArray benchmark")
    print(f"  build time: {build_time_s:.3f} s")
    print(f"  batches: {len(read_array)}")
    print(f"  items: {TOTAL_OBJECTS}")
    print(f"  items_per_block: {read_array.items_per_block}")
    print()
    print(read_array.info)
    print(f"Random scalar reads: {N_RANDOM_READS}")
    print(f"  mean: {statistics.fmean(timings_ns) / 1_000:.2f} us")
    print(f"  max:  {max(timings_ns) / 1_000:.2f} us")
    print(f"  min:  {min(timings_ns) / 1_000:.2f} us")
    print(f"Item iteration via iter_items(): {iter_time_s:.3f} s")
    print(f"  per item: {iter_time_s * 1_000_000 / nitems:.2f} us")
    print(f"  checksum: {checksum}")
    print("Sample reads:")
    for timing_ns, batch_index, item_index, value in samples[:5]:
        print(f"  {timing_ns / 1_000:.2f} us -> read_array[{batch_index}][{item_index}] = {value}")
    if args.in_mem:
        print("BatchArray kept in memory")
    else:
        print(f"BatchArray file at: {read_array.urlpath}")


if __name__ == "__main__":
    main()
