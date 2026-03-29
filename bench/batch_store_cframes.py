#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

from __future__ import annotations

import argparse
import math
import pathlib
import random
import sys
import time
from bisect import bisect_right

import blosc2
from blosc2._msgpack_utils import msgpack_packb


URLPATH = "bench_batch_store_cframes.b2b"
DEFAULT_NFRAMES = 1_000
DEFAULT_NELEMENTS = 1_000
DEFAULT_NBATCHES = 1_000
_DICT_CODECS = {blosc2.Codec.ZSTD, blosc2.Codec.LZ4, blosc2.Codec.LZ4HC}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build or read an on-disk BatchStore containing batches of Blosc2 CFrames.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--urlpath", type=str, default=None, help="Path to the BatchStore file.")
    parser.add_argument(
        "--nframes-per-batch", type=int, default=DEFAULT_NFRAMES, help="Number of CFrames stored in each batch."
    )
    parser.add_argument(
        "--nelements-per-frame",
        type=int,
        default=DEFAULT_NELEMENTS,
        help="Number of array elements stored in each frame.",
    )
    parser.add_argument("--nbatches", type=int, default=DEFAULT_NBATCHES, help="Number of batches to append.")
    parser.add_argument(
        "--nframes-per-block",
        type=int,
        default=None,
        help="Maximum number of frames per internal block. Default is automatic inference.",
    )
    parser.add_argument("--codec", type=str, default="ZSTD", choices=[codec.name for codec in blosc2.Codec])
    parser.add_argument("--clevel", type=int, default=5)
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed for reproducible random reads.")
    parser.add_argument(
        "--random-read",
        type=int,
        default=1,
        help="Read N random serialized CFrames and report timing. When passed explicitly, reads an existing store.",
    )
    parser.add_argument(
        "--random-read-cframe",
        type=int,
        default=0,
        help=(
            "Read N random single frames by fetching a stored CFrame and deserializing it "
            "with blosc2.ndarray_from_cframe(). Requires --urlpath."
        ),
    )
    parser.add_argument(
        "--random-read-element",
        type=int,
        default=0,
        help=(
            "Read N random single elements by fetching a random frame, unpacking its CFrame, "
            "and indexing a random element. Requires --urlpath."
        ),
    )
    parser.add_argument(
        "--use-dict",
        action="store_true",
        help="Enable dictionaries for codecs that support them (ZSTD, LZ4, LZ4HC).",
    )
    return parser


def make_batch(nframes: int, frame: bytes) -> list[bytes]:
    return [frame] * nframes


def format_size(nbytes: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    size = float(nbytes)
    unit = units[0]
    for candidate in units:
        unit = candidate
        if size < 1024 or candidate == units[-1]:
            break
        size /= 1024
    if unit == "B":
        return f"{nbytes} bytes ({nbytes} {unit})"
    return f"{nbytes} bytes ({size:.2f} {unit})"


def format_count(value: int) -> str:
    return f"{value:_} ({value:.2e}, 2**{math.log2(value):.3f})"


def print_store_counts(store: blosc2.BatchStore) -> None:
    total_frames = sum(len(batch) for batch in store)
    print(f"  total frames: {format_count(total_frames)}")
    if total_frames == 0:
        print("  total elements: 0")
        return

    first_frame = store[0][0]
    array = blosc2.ndarray_from_cframe(first_frame)
    nelements_per_frame = math.prod(array.shape)
    total_elements = total_frames * nelements_per_frame
    print(f"  nelements per frame: {nelements_per_frame}")
    print(f"  total elements: {format_count(total_elements)}")


def sample_random_reads(store: blosc2.BatchStore, nreads: int, rng: random.Random) -> list[tuple[int, int, int, int]]:
    batch_lengths = [len(batch) for batch in store]
    total_frames = sum(batch_lengths)
    if total_frames == 0:
        return []

    prefix = [0]
    for length in batch_lengths:
        prefix.append(prefix[-1] + length)

    sample_size = min(nreads, total_frames)
    flat_indices = rng.sample(range(total_frames), sample_size)
    results: list[tuple[int, int, int, int]] = []

    for flat_index in flat_indices:
        batch_index = bisect_right(prefix, flat_index) - 1
        frame_index = flat_index - prefix[batch_index]
        t0 = time.perf_counter_ns()
        frame = store[batch_index][frame_index]
        elapsed_ns = time.perf_counter_ns() - t0
        results.append((batch_index, frame_index, len(frame), elapsed_ns))

    return results


def print_random_read_stats(store: blosc2.BatchStore, nreads: int, rng: random.Random) -> None:
    samples = sample_random_reads(store, nreads, rng)
    if not samples:
        print("random scalar reads: store is empty")
        return

    timings_ns = [elapsed_ns for _, _, _, elapsed_ns in samples]
    print(f"random scalar reads: {len(samples)}")
    print(f"  mean: {sum(timings_ns) / len(timings_ns) / 1_000:.2f} us")
    print(f"  min: {min(timings_ns) / 1_000:.2f} us")
    print(f"  max: {max(timings_ns) / 1_000:.2f} us")
    batch_index, frame_index, frame_len, elapsed_ns = samples[0]
    print(
        f"  first sample: store[{batch_index}][{frame_index}] -> {frame_len} bytes "
        f"in {elapsed_ns / 1_000:.2f} us"
    )


def sample_random_cframe_reads(
    store: blosc2.BatchStore, nreads: int, rng: random.Random
) -> list[tuple[int, int, tuple[int, ...], int]]:
    batch_lengths = [len(batch) for batch in store]
    total_frames = sum(batch_lengths)
    if total_frames == 0:
        return []

    prefix = [0]
    for length in batch_lengths:
        prefix.append(prefix[-1] + length)

    sample_size = min(nreads, total_frames)
    flat_indices = rng.sample(range(total_frames), sample_size)
    results: list[tuple[int, int, tuple[int, ...], int]] = []

    for flat_index in flat_indices:
        batch_index = bisect_right(prefix, flat_index) - 1
        frame_index = flat_index - prefix[batch_index]
        t0 = time.perf_counter_ns()
        frame = store[batch_index][frame_index]
        array = blosc2.ndarray_from_cframe(frame)
        elapsed_ns = time.perf_counter_ns() - t0
        results.append((batch_index, frame_index, array.shape, elapsed_ns))

    return results


def print_random_cframe_read_stats(store: blosc2.BatchStore, nreads: int, rng: random.Random) -> None:
    samples = sample_random_cframe_reads(store, nreads, rng)
    if not samples:
        print("random cframe reads: store is empty")
        return

    timings_ns = [elapsed_ns for _, _, _, elapsed_ns in samples]
    print(f"random cframe reads: {len(samples)}")
    print(f"  mean: {sum(timings_ns) / len(timings_ns) / 1_000:.2f} us")
    print(f"  min: {min(timings_ns) / 1_000:.2f} us")
    print(f"  max: {max(timings_ns) / 1_000:.2f} us")
    batch_index, frame_index, shape, elapsed_ns = samples[0]
    print(f"  first sample: store[{batch_index}][{frame_index}] -> shape={shape} in {elapsed_ns / 1_000:.2f} us")


def sample_random_element_reads(
    store: blosc2.BatchStore, nreads: int, rng: random.Random
) -> list[tuple[int, int, int, int | float | bool, int]]:
    batch_lengths = [len(batch) for batch in store]
    total_frames = sum(batch_lengths)
    if total_frames == 0:
        return []

    prefix = [0]
    for length in batch_lengths:
        prefix.append(prefix[-1] + length)

    samples: list[tuple[int, int, int, int | float | bool, int]] = []
    for _ in range(nreads):
        flat_index = rng.randrange(total_frames)
        batch_index = bisect_right(prefix, flat_index) - 1
        frame_index = flat_index - prefix[batch_index]
        t0 = time.perf_counter_ns()
        frame = store[batch_index][frame_index]
        array = blosc2.ndarray_from_cframe(frame)
        element_index = rng.randrange(array.shape[0])
        value = array[element_index].item()
        elapsed_ns = time.perf_counter_ns() - t0
        samples.append((batch_index, frame_index, element_index, value, elapsed_ns))
    return samples


def print_random_element_read_stats(store: blosc2.BatchStore, nreads: int, rng: random.Random) -> None:
    samples = sample_random_element_reads(store, nreads, rng)
    if not samples:
        print("random element reads: store is empty")
        return

    timings_ns = [elapsed_ns for *_, elapsed_ns in samples]
    print(f"random element reads: {len(samples)}")
    print(f"  mean: {sum(timings_ns) / len(timings_ns) / 1_000:.2f} us")
    print(f"  min: {min(timings_ns) / 1_000:.2f} us")
    print(f"  max: {max(timings_ns) / 1_000:.2f} us")
    batch_index, frame_index, element_index, value, elapsed_ns = samples[0]
    print(
        f"  first sample: store[{batch_index}][{frame_index}][{element_index}] -> {value!r} "
        f"in {elapsed_ns / 1_000:.2f} us"
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    random_read_requested = any(arg == "--random-read" or arg.startswith("--random-read=") for arg in sys.argv[1:])

    if args.nframes_per_batch <= 0:
        parser.error("--nframes-per-batch must be > 0")
    if args.nelements_per_frame <= 0:
        parser.error("--nelements-per-frame must be > 0")
    if args.nbatches <= 0:
        parser.error("--nbatches must be > 0")
    if args.nframes_per_block is not None and args.nframes_per_block <= 0:
        parser.error("--nframes-per-block must be > 0")
    if args.random_read <= 0:
        parser.error("--random-read must be > 0")
    if args.random_read_cframe < 0:
        parser.error("--random-read-cframe must be >= 0")
    if args.random_read_element < 0:
        parser.error("--random-read-element must be >= 0")
    if not 0 <= args.clevel <= 9:
        parser.error("--clevel must be between 0 and 9")
    if (random_read_requested or args.random_read_cframe > 0 or args.random_read_element > 0) and args.urlpath is None:
        parser.error("--random-read, --random-read-cframe and --random-read-element require --urlpath")

    codec = blosc2.Codec[args.codec]
    use_dict = args.use_dict and codec in _DICT_CODECS
    total_frames = args.nframes_per_batch * args.nbatches
    total_elements = total_frames * args.nelements_per_frame
    rng = random.Random(args.seed)

    if args.use_dict and not use_dict:
        print(f"Codec {codec.name} does not support use_dict; disabling it.")

    if random_read_requested or args.random_read_cframe > 0 or args.random_read_element > 0:
        store = blosc2.open(args.urlpath, mode="r")
        if not isinstance(store, blosc2.BatchStore):
            raise TypeError(f"{args.urlpath!r} is not a BatchStore")
        print("Reading on-disk BatchStore with CFrame payloads")
        print(f"  urlpath: {args.urlpath}")
        print(f"  seed: {args.seed}")
        print_store_counts(store)
        print()
        print(store.info)
        print()
        if random_read_requested:
            print_random_read_stats(store, args.random_read, rng)
        if args.random_read_cframe > 0:
            if random_read_requested:
                print()
            print_random_cframe_read_stats(store, args.random_read_cframe, rng)
        if args.random_read_element > 0:
            if random_read_requested or args.random_read_cframe > 0:
                print()
            print_random_element_read_stats(store, args.random_read_element, rng)
        return

    cparams = blosc2.CParams(codec=codec, clevel=args.clevel, use_dict=use_dict)

    urlpath = args.urlpath or URLPATH
    blosc2.remove_urlpath(urlpath)
    source = blosc2.full(args.nelements_per_frame, 3)
    frame = source.to_cframe()
    msgpack_frame = msgpack_packb(frame)

    print("Building on-disk BatchStore with CFrame payloads")
    print(f"  urlpath: {urlpath}")
    print(f"  nbatches: {args.nbatches}")
    print(f"  nframes per batch: {args.nframes_per_batch}")
    print(f"  nelements per frame: {args.nelements_per_frame}")
    print(f"  nframes per block: {args.nframes_per_block}")
    print(f"  total frames: {format_count(total_frames)}")
    print(f"  total elements: {format_count(total_elements)}")
    print(f"  cframe bytes per frame: {len(frame)}")
    print(f"  msgpack bytes per frame: {len(msgpack_frame)}")
    print(f"  codec: {codec.name}")
    print(f"  clevel: {args.clevel}")
    print(f"  use_dict: {use_dict}")
    print(f"  seed: {args.seed}")

    with blosc2.BatchStore(
        storage=blosc2.Storage(urlpath=urlpath, mode="w", contiguous=True),
        cparams=cparams,
        items_per_block=args.nframes_per_block,
    ) as store:
        batch = make_batch(args.nframes_per_batch, frame)
        for _ in range(args.nbatches):
            store.append(batch)
        print()
        print(store.info)
        uncompressed_nbytes = store.nbytes

    size_nbytes = pathlib.Path(urlpath).stat().st_size
    print(f"store file size: {format_size(size_nbytes)}")
    print(
        f"average compressed bytes per frame: {size_nbytes / total_frames:.2f} "
        f"({uncompressed_nbytes / total_frames:.2f} uncompressed)"
    )
    print()
    print_random_read_stats(blosc2.open(urlpath, mode="r"), args.random_read, rng)


if __name__ == "__main__":
    main()
