"""Profile bounded compressed Menudet reads without changing production dispatch.

Run with the rebuilt storage-reader extension and ME_DSL_JIT_COMPILER=tcc.
Instrumented timings are diagnostic; uninstrumented medians drive comparisons.
"""

import argparse
import cProfile
import ctypes
import io
import json
import os
import platform
import pstats
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from statistics import median
from threading import Thread, local
from time import perf_counter
from unittest.mock import patch

import numpy as np

import blosc2
from blosc2.storage_readers import StorageReader


@blosc2.menudet
def adjusted(x, y):
    return x + (y - 1)


def measure(operation, expected, samples):
    np.testing.assert_array_equal(operation(), expected)
    times = []
    for _ in range(samples):
        start = perf_counter()
        operation()
        times.append((perf_counter() - start) * 1000)
    return round(median(times), 4)


def instrument(operation, samples):
    original_read = StorageReader.read
    original_span = blosc2.NDArray.get_1d_span_numpy
    readings = []
    current = {}

    def timed_read(self, indices):
        start = perf_counter()
        try:
            return original_read(self, indices)
        finally:
            current["reader_ms"] += (perf_counter() - start) * 1000
            current["reader_calls"] += 1

    def timed_span(self, *args, **kwargs):
        start = perf_counter()
        try:
            return original_span(self, *args, **kwargs)
        finally:
            current["span_ms"] += (perf_counter() - start) * 1000
            current["span_calls"] += 1

    with (
        patch.object(StorageReader, "read", timed_read),
        patch.object(blosc2.NDArray, "get_1d_span_numpy", timed_span),
    ):
        for _ in range(samples):
            current = {"reader_ms": 0, "span_ms": 0, "reader_calls": 0, "span_calls": 0}
            start = perf_counter()
            operation()
            current["total_ms"] = (perf_counter() - start) * 1000
            current["reader_python_ms"] = current["reader_ms"] - current["span_ms"]
            current["outside_reader_ms"] = current["total_ms"] - current["reader_ms"]
            readings.append(current)
    return {key: round(median(row[key] for row in readings), 4) for key in readings[0]}


def touched_blocks(size, tile, block):
    return sum((min(start + tile, size) - 1) // block - start // block + 1 for start in range(0, size, tile))


def snapshot_decoder_experiments(helper, kernel, inputs, expected, samples):  # noqa: C901 -- profiling matrix
    """Fixed ZSTD test data only; compressed snapshots are fetched before timing."""
    if any(value.schunk.nchunks != 1 for value in inputs.values()):
        raise ValueError("Snapshot experiment requires one compressed chunk per input")
    library = ctypes.CDLL(helper)
    library.menudet_profile_context.restype = ctypes.c_void_p
    library.menudet_profile_free.argtypes = [ctypes.c_void_p]
    library.menudet_profile_decode.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    library.menudet_profile_decode.restype = ctypes.c_int
    snapshots = {id(value): value.schunk.get_chunk(0) for value in inputs.values()}
    rows = []
    for workers in (1, 2, 4):
        contexts = [library.menudet_profile_context() for _ in range(workers)]
        if not all(contexts):
            raise MemoryError("Cannot create profiling contexts")
        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:

                def reusable_read(self, indices, workers=workers, contexts=contexts, pool=pool):
                    source = snapshots[id(self.value)]
                    first = int(indices[0])
                    values = np.empty(indices.size, dtype="float64")

                    def decode(worker):
                        begin = worker * indices.size // workers
                        end = (worker + 1) * indices.size // workers
                        rc = library.menudet_profile_decode(
                            contexts[worker],
                            source,
                            len(source),
                            (first + begin) * 8,
                            (end - begin) * 8,
                            values.ctypes.data + begin * 8,
                        )
                        if rc != (end - begin) * 8:
                            raise RuntimeError(f"Profiling span decode failed: {rc}")

                    if workers == 1:
                        decode(0)
                    else:
                        list(pool.map(decode, range(workers)))
                    return values

                with patch.object(StorageReader, "read", reusable_read):
                    rows.append(
                        {
                            "workers": workers,
                            "ms": measure(lambda: kernel.evaluate_sources(inputs), expected, samples),
                        }
                    )
        finally:
            for context in contexts:
                library.menudet_profile_free(context)
    pipeline_rows = []
    tile = inputs["x"].blocks[0] * 4
    for workers in (1, 2, 4, 8):
        owned_contexts = []
        worker_state = local()
        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:

                def decode_and_evaluate(start, worker_state=worker_state, owned_contexts=owned_contexts):
                    if not hasattr(worker_state, "context"):
                        worker_state.context = library.menudet_profile_context()
                        if not worker_state.context:
                            raise MemoryError("Cannot create worker profiling context")
                        owned_contexts.append(worker_state.context)
                    count = min(tile, expected.size - start)
                    resident = {}
                    for name, value in inputs.items():
                        source = snapshots[id(value)]
                        buffer = np.empty(count, dtype="float64")
                        rc = library.menudet_profile_decode(
                            worker_state.context,
                            source,
                            len(source),
                            start * 8,
                            count * 8,
                            buffer.ctypes.data,
                        )
                        if rc != count * 8:
                            raise RuntimeError(f"Profiling worker decode failed: {rc}")
                        resident[name] = buffer
                    return start, kernel.evaluate_block(resident)

                def pipeline(workers=workers, pool=pool, decode_and_evaluate=decode_and_evaluate):
                    result = np.empty_like(expected)
                    # Bounded waves: at most one tile task per worker is live.
                    for begin in range(0, expected.size, workers * tile):
                        starts = range(begin, min(begin + workers * tile, expected.size), tile)
                        for start, values in pool.map(decode_and_evaluate, starts):
                            result[start : start + values.size] = values
                    return result

                pipeline_rows.append(
                    {"workers": workers, "tile_items": tile, "ms": measure(pipeline, expected, samples)}
                )
        finally:
            for context in owned_contexts:
                library.menudet_profile_free(context)
    return {"reader_only": rows, "decode_and_jit_pipeline": pipeline_rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=int, default=1_000_000)
    parser.add_argument("--samples", type=int, default=25)
    parser.add_argument("--profile-calls", type=int, default=100)
    parser.add_argument("--native-sample", help="macOS sample output path (optional)")
    parser.add_argument("--reader-helper", help="Profiling-only compiled decoder helper (optional)")
    args = parser.parse_args()
    if min(args.items, args.samples, args.profile_calls) <= 0:
        parser.error("items, samples and profile-calls must be positive")
    x, y = np.linspace(0, 1, args.items), np.linspace(1, 2, args.items)
    expected = x + (y - 1)
    inputs = {"x": blosc2.asarray(x), "y": blosc2.asarray(y)}
    resident = {"x": x, "y": y}
    lazy = adjusted(**inputs)
    kernel = lazy.kernel
    assert kernel.has_jit, "This comparison requires actual JIT preparation"
    legacy = blosc2.lazyexpr("x + (y - 1)", inputs)
    block = inputs["x"].blocks[0]
    baseline = {
        "lazyexpr_ms": measure(lambda: legacy[:], expected, args.samples),
        "menudet_lazy_ms": measure(lambda: lazy[:], expected, args.samples),
        "resident_full_block_ms": measure(lambda: kernel.evaluate_block(resident), expected, args.samples),
    }
    rows = []
    for tile in sorted({block, 2 * block, 4 * block, 65536, 8 * block, 16 * block, args.items}):

        def operation(tile=tile):
            return kernel.evaluate_sources(inputs, tile_items=tile)

        values, report = kernel.evaluate_sources(inputs, tile_items=tile, return_report=True)
        np.testing.assert_array_equal(values, expected)
        rows.append(
            {
                "tile_items": tile,
                "compressed_ms": measure(operation, expected, args.samples),
                "resident_ms": measure(
                    lambda tile=tile: kernel.evaluate_sources(resident, tile_items=tile),
                    expected,
                    args.samples,
                ),
                "block_visits_per_input": touched_blocks(args.items, tile, block),
                "unique_blocks_per_input": (args.items + block - 1) // block,
                "instrumented": instrument(operation, args.samples),
                "report": report,
            }
        )

    # Keep index construction/callback overhead but replace decompression with
    # resident gathers. This is an experiment, not a supported execution route.
    def resident_read(self, indices):
        source = x if self.value is inputs["x"] else y
        return source[indices]

    with patch.object(StorageReader, "read", resident_read):
        no_codec_ms = measure(lambda: kernel.evaluate_sources(inputs), expected, args.samples)

    def contiguous_resident_read(self, indices):
        source = x if self.value is inputs["x"] else y
        first = int(indices[0])
        return source[first : first + indices.size]

    def contiguous_span_read(self, indices):
        # Whole-domain experiment only: no masks/broadcast/strides, one chunk.
        first = int(indices[0])
        values = np.empty(indices.size, dtype=self.value.dtype)
        self.value.get_1d_span_numpy(values, 0, first, indices.size, nthreads=1)
        return values

    def bounded_slice_read(self, indices):
        first = int(indices[0])
        return self.value[first : first + indices.size]

    reader_experiments = []
    for label, reader in (
        ("resident_contiguous", contiguous_resident_read),
        ("compressed_contiguous_span", contiguous_span_read),
        ("compressed_bounded_slice", bounded_slice_read),
    ):
        with patch.object(StorageReader, "read", reader):
            reader_experiments.append(
                {
                    "reader": label,
                    "ms": measure(lambda: kernel.evaluate_sources(inputs), expected, args.samples),
                }
            )
    # Isolate thread policy on the existing span implementation.
    span = blosc2.NDArray.get_1d_span_numpy
    thread_rows = []
    for threads in (1, 2, 4, 12):

        def threaded_span(self, *values, threads=threads, **kwargs):
            kwargs["nthreads"] = threads
            return span(self, *values, **kwargs)

        with patch.object(blosc2.NDArray, "get_1d_span_numpy", threaded_span):
            thread_rows.append(
                {
                    "threads": threads,
                    "ms": measure(lambda: kernel.evaluate_sources(inputs), expected, args.samples),
                }
            )
    profile = cProfile.Profile()
    profile.enable()
    for _ in range(args.profile_calls):
        lazy[:]
    profile.disable()
    stream = io.StringIO()
    pstats.Stats(profile, stream=stream).strip_dirs().sort_stats("cumulative").print_stats(30)
    snapshot_rows = (
        snapshot_decoder_experiments(args.reader_helper, kernel, inputs, expected, args.samples)
        if args.reader_helper
        else []
    )
    if args.native_sample:
        sampler = Thread(
            target=subprocess.run,
            args=(["/usr/bin/sample", str(os.getpid()), "5", "1", "-file", args.native_sample],),
            kwargs={"check": True, "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL},
        )
        sampler.start()
        while sampler.is_alive():
            lazy[:]
        sampler.join()
    print(
        json.dumps(
            {
                "platform": platform.platform(),
                "package": blosc2.__file__,
                "extension": blosc2.blosc2_ext.__file__,
                "items": args.items,
                "samples": args.samples,
                "chunks": inputs["x"].chunks,
                "blocks": inputs["x"].blocks,
                "cparams": {key: str(value) for key, value in asdict(inputs["x"].schunk.cparams).items()},
                "jit": kernel.has_jit,
                "jit_compiler_request": os.environ.get("ME_DSL_JIT_COMPILER", "default"),
                "cc_request": os.environ.get("CC", "default"),
                "baseline": baseline,
                "tiles": rows,
                "resident_reader_no_codec_ms": no_codec_ms,
                "span_threads": thread_rows,
                "reader_experiments": reader_experiments,
                "snapshot_decoder_experiments": snapshot_rows,
                "cprofile": stream.getvalue(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
