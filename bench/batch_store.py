#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

from __future__ import annotations

import argparse
import random
import statistics
import time

import blosc2


URLPATH = "bench_batch_store.b2b"
NBATCHES = 10_000
OBJECTS_PER_BATCH = 100
TOTAL_OBJECTS = NBATCHES * OBJECTS_PER_BATCH
BLOCKSIZE_MAX = 32
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
        description="Benchmark BatchStore single-entry reads.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--codec", type=str, default="ZSTD", choices=[codec.name for codec in blosc2.Codec])
    parser.add_argument("--clevel", type=int, default=5)
    parser.add_argument("--use-dict", action="store_true", help="Enable dictionaries for ZSTD/LZ4 codecs.")
    parser.add_argument("--in-mem", action="store_true", help="Keep the BatchStore purely in memory.")
    return parser


def build_store(codec: blosc2.Codec, clevel: int, use_dict: bool, in_mem: bool) -> blosc2.BatchStore | None:
    if in_mem:
        storage = blosc2.Storage(mode="w")
        store = blosc2.BatchStore(
            storage=storage,
            blocksize_max=BLOCKSIZE_MAX,
            cparams={
                "codec": codec,
                "clevel": clevel,
                "use_dict": use_dict and codec in (blosc2.Codec.ZSTD, blosc2.Codec.LZ4),
            },
        )
        for batch_index in range(NBATCHES):
            store.append(make_batch(batch_index))
        return store

    blosc2.remove_urlpath(URLPATH)
    storage = blosc2.Storage(urlpath=URLPATH, mode="w", contiguous=True)
    cparams = {
        "codec": codec,
        "clevel": clevel,
        "use_dict": use_dict and codec in (blosc2.Codec.ZSTD, blosc2.Codec.LZ4),
    }
    with blosc2.BatchStore(storage=storage, blocksize_max=BLOCKSIZE_MAX, cparams=cparams) as store:
        for batch_index in range(NBATCHES):
            store.append(make_batch(batch_index))
    return None


def measure_random_reads(store: blosc2.BatchStore) -> tuple[list[tuple[int, int, int, dict[str, int]]], list[int]]:
    rng = random.Random(2024)
    samples: list[tuple[int, int, int, dict[str, int]]] = []
    timings_ns: list[int] = []

    for _ in range(N_RANDOM_READS):
        batch_index = rng.randrange(len(store))
        item_index = rng.randrange(OBJECTS_PER_BATCH)
        t0 = time.perf_counter_ns()
        value = store[batch_index][item_index]
        timings_ns.append(time.perf_counter_ns() - t0)
        if value != expected_entry(batch_index, item_index):
            raise RuntimeError(f"Value mismatch at batch={batch_index}, item={item_index}")
        samples.append((timings_ns[-1], batch_index, item_index, value))

    return samples, timings_ns


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    codec = blosc2.Codec[args.codec]
    use_dict = args.use_dict and codec in (blosc2.Codec.ZSTD, blosc2.Codec.LZ4)

    mode_label = "in-memory" if args.in_mem else "persistent"
    article = "an" if args.in_mem else "a"
    print(f"Building {article} {mode_label} BatchStore with 1,000,000 RGB dicts and timing 1,000 random scalar reads...")
    print(f"  codec: {codec.name}")
    print(f"  clevel: {args.clevel}")
    print(f"  use_dict: {use_dict}")
    print(f"  in_mem: {args.in_mem}")
    t0 = time.perf_counter()
    store = build_store(codec=codec, clevel=args.clevel, use_dict=use_dict, in_mem=args.in_mem)
    build_time_s = time.perf_counter() - t0
    if args.in_mem:
        assert store is not None
        read_store = store
    else:
        read_store = blosc2.BatchStore(urlpath=URLPATH, mode="r", contiguous=True, blocksize_max=BLOCKSIZE_MAX)
    samples, timings_ns = measure_random_reads(read_store)
    t0 = time.perf_counter()
    checksum = 0
    nobjects = 0
    for obj in read_store.iter_objects():
        checksum += obj["blue"]
        nobjects += 1
    iter_time_s = time.perf_counter() - t0

    print()
    print("BatchStore benchmark")
    print(f"  build time: {build_time_s:.3f} s")
    print(f"  batches: {len(read_store)}")
    print(f"  objects: {TOTAL_OBJECTS}")
    print(f"  blocksize_max: {read_store.blocksize_max}")
    print()
    print(read_store.info)
    print(f"Random scalar reads: {N_RANDOM_READS}")
    print(f"  mean: {statistics.fmean(timings_ns) / 1_000:.2f} us")
    print(f"  max:  {max(timings_ns) / 1_000:.2f} us")
    print(f"  min:  {min(timings_ns) / 1_000:.2f} us")
    print(f"Object iteration via iter_objects(): {iter_time_s:.3f} s")
    print(f"  per object: {iter_time_s * 1_000_000 / nobjects:.2f} us")
    print(f"  checksum: {checksum}")
    print("Sample reads:")
    for timing_ns, batch_index, item_index, value in samples[:5]:
        print(f"  {timing_ns / 1_000:.2f} us -> read_store[{batch_index}][{item_index}] = {value}")
    if args.in_mem:
        print("BatchStore kept in memory")
    else:
        print(f"BatchStore file at: {read_store.urlpath}")


if __name__ == "__main__":
    main()
