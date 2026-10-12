"""Profile a live-storage, chunk-dense / reader-sparse Menudet prototype.

This benchmark only supports matching 1-D float64 compressed operands and a
lane-local kernel. It does not change production dispatch or use chunk snapshots.
"""

import argparse
import cProfile
import io
import json
import os
import platform
import pstats
from dataclasses import replace
from statistics import median
from time import perf_counter

import numpy as np

import blosc2


@blosc2.menudet
def adjusted(x, y):
    return x + (y - 1)


@blosc2.dsl_kernel
def existing_adjusted(x, y):
    return x + (y - 1)


def measure(operation, expected, samples):
    np.testing.assert_array_equal(operation(), expected)
    times = []
    for _ in range(samples):
        start = perf_counter()
        result = operation()
        times.append((perf_counter() - start) * 1000)
        del result
    return round(median(times), 4)


class HybridPrototype:
    """Single-chunk-at-a-time buffers; no arbitrary recipes or output aliasing."""

    def __init__(self, kernel, inputs, selection, force=None):
        self.kernel, self.inputs = kernel, inputs
        source = next(iter(inputs.values()))
        assert all(
            value.shape == source.shape and value.chunks == source.chunks for value in inputs.values()
        )
        assert all(value.ndim == 1 and value.dtype == np.dtype("float64") for value in inputs.values())
        self.selection = selection
        self.start, self.stop, self.step = selection.indices(source.shape[0])
        if self.step <= 0:
            raise ValueError("Profiling prototype requires positive steps")
        self.count = len(range(self.start, self.stop, self.step))
        self.chunk_items = source.chunks[0]
        self.parts = []
        if self.count:
            last = self.start + (self.count - 1) * self.step
            for chunk in range(self.start // self.chunk_items, last // self.chunk_items + 1):
                origin = chunk * self.chunk_items
                begin = max(self.start, origin)
                begin += (self.start - begin) % self.step
                end = min(self.stop, origin + self.chunk_items)
                count = len(range(begin, end, self.step))
                if count:
                    self.parts.append((chunk, begin - origin, end - origin, count))
        self.decoded_items = sum(source.schunk.chunksize // source.dtype.itemsize for _ in self.parts)
        block_items = source.blocks[0]
        self.touched_blocks = sum(
            count
            if self.step >= block_items
            else (begin + (count - 1) * self.step) // block_items - begin // block_items + 1
            for _, begin, _, count in self.parts
        )
        self.chunk_block_capacity = self.decoded_items // block_items
        density = self.count / self.decoded_items if self.decoded_items else 0
        self.route = force or ("chunks" if density >= 0.5 else "readers")
        self.stats = {}

    def __call__(self, instrument=False):
        if self.route == "readers":
            return self.kernel.evaluate_sources(self.inputs, selection=(self.selection,))
        buffers = {
            name: np.empty(value.schunk.chunksize // value.dtype.itemsize, dtype=value.dtype)
            for name, value in self.inputs.items()
        }
        result = np.empty(self.count, dtype=self.kernel.output_dtype) if len(self.parts) != 1 else None
        offset = 0
        decode_ms = evaluate_ms = publish_ms = 0
        for chunk, begin, end, count in self.parts:
            start = perf_counter() if instrument else 0
            for name, value in self.inputs.items():
                value.schunk.decompress_chunk(chunk, dst=buffers[name])
            if instrument:
                decode_ms += (perf_counter() - start) * 1000
            operands = {name: buffer[begin : end : self.step] for name, buffer in buffers.items()}
            start = perf_counter() if instrument else 0
            values = self.kernel.evaluate_block(operands)
            if instrument:
                evaluate_ms += (perf_counter() - start) * 1000
            if result is None:
                result = values
            else:
                start = perf_counter() if instrument else 0
                result[offset : offset + count] = values
                if instrument:
                    publish_ms += (perf_counter() - start) * 1000
            offset += count
            del operands, values
        self.stats = {
            "decode_ms": decode_ms,
            "jit_binding_and_eval_ms": evaluate_ms,
            "publish_ms": publish_ms,
            "chunk_reads": len(self.parts) * len(self.inputs),
            "input_buffer_bytes": sum(buffer.nbytes for buffer in buffers.values()),
            "selected_items": self.count,
            "decoded_items_per_input": self.decoded_items,
            "maximum_chunk_result_items": max((part[3] for part in self.parts), default=0),
        }
        return result


def profile(operation, calls=20):
    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(calls):
        operation()
    profiler.disable()
    output = io.StringIO()
    pstats.Stats(profiler, stream=output).strip_dirs().sort_stats("cumulative").print_stats(15)
    return output.getvalue()


def run_case(size, chunk_items, samples, dispatch_only=False):
    x, y = np.linspace(0, 1, size), np.linspace(1, 2, size)
    expected = x + (y - 1)
    layout = {} if chunk_items is None else {"chunks": (chunk_items,), "blocks": (15625,)}
    inputs = {"x": blosc2.asarray(x, **layout), "y": blosc2.asarray(y, **layout)}
    lazy = adjusted(**inputs)
    kernel = lazy.kernel
    assert kernel.has_jit, "Prepared JIT is required"
    legacy = blosc2.lazyexpr("x + (y - 1)", inputs)
    existing_kernel = blosc2.lazyudf(existing_adjusted, tuple(inputs.values()), dtype=np.float64, jit=True)
    row = {
        "items": size,
        "chunks": inputs["x"].chunks,
        "blocks": inputs["x"].blocks,
        "nchunks": inputs["x"].schunk.nchunks,
        "lazyexpr_ms": measure(lambda: legacy[:], expected, samples),
        "existing_kernel_ms": measure(lambda: existing_kernel[:], expected, samples),
        "menudet_auto_ms": measure(lambda: lazy[:], expected, samples),
        "menudet_route": dict(lazy._source_execution_report),
        "bounded_menudet_ms": measure(lambda: kernel.evaluate_sources(inputs), expected, samples),
        "whole_partition_ms": measure(
            lambda: kernel.evaluate_block({name: value[:] for name, value in inputs.items()}),
            expected,
            samples,
        ),
    }
    if dispatch_only:
        return row
    dense_rows = []
    prototype = HybridPrototype(kernel, inputs, slice(None))
    original_dparams = {name: value.schunk.dparams for name, value in inputs.items()}
    for threads in (1, 2, 4, 12):
        for name, value in inputs.items():
            value.schunk.dparams = replace(original_dparams[name], nthreads=threads)
        ms = measure(prototype, expected, samples)
        components = []
        for _ in range(samples):
            prototype(instrument=True)
            components.append(dict(prototype.stats))
        dense_rows.append(
            {
                "decode_threads": threads,
                "ms": ms,
                "instrumented": {
                    key: round(median(component[key] for component in components), 4)
                    for key in components[0]
                },
            }
        )
    for name, value in inputs.items():
        value.schunk.dparams = original_dparams[name]
    selections = {
        "tiny_contiguous": slice(1000, min(size, 5096)),
        "dense_prefix": slice(0, size * 9 // 10),
        "sparse_stride": slice(None, None, 1000),
    }
    partial = []
    for label, selection in selections.items():
        hybrid = HybridPrototype(kernel, inputs, selection)
        forced_dense = HybridPrototype(kernel, inputs, selection, force="chunks")
        reference = expected[selection]
        partial.append(
            {
                "selection": label,
                "hybrid_route": hybrid.route,
                "selected_items": hybrid.count,
                "touched_blocks_per_input": hybrid.touched_blocks,
                "touched_chunk_block_capacity": hybrid.chunk_block_capacity,
                "lazyexpr_ms": measure(lambda selection=selection: legacy[selection], reference, samples),
                "readers_ms": measure(
                    lambda selection=selection: kernel.evaluate_sources(inputs, selection=(selection,)),
                    reference,
                    samples,
                ),
                "chunk_dense_ms": measure(forced_dense, reference, samples),
                "hybrid_ms": measure(hybrid, reference, samples),
            }
        )
    row.update(
        {
            "dense_chunk_sweep": dense_rows,
            "partial": partial,
            "dense_profile": profile(prototype),
            "bounded_profile": profile(lambda: lazy[:]),
        }
    )
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=25)
    parser.add_argument("--large-items", type=int, default=8_000_000)
    parser.add_argument(
        "--dispatch-only", action="store_true", help="Only benchmark production route comparisons"
    )
    args = parser.parse_args()
    if args.samples <= 0 or args.large_items <= 0:
        parser.error("samples and large-items must be positive")
    rows = [
        run_case(100_000, None, args.samples, args.dispatch_only),
        run_case(1_000_000, None, args.samples, args.dispatch_only),
        run_case(args.large_items, 1_000_000, args.samples, args.dispatch_only),
    ]
    print(
        json.dumps(
            {
                "platform": platform.platform(),
                "package": blosc2.__file__,
                "extension": blosc2.blosc2_ext.__file__,
                "samples": args.samples,
                "compiler": os.environ.get("ME_DSL_JIT_COMPILER", "default"),
                "cc": os.environ.get("CC", "default"),
                "prototype_policy": "chunks when selected/touched-chunk capacity >= 0.5; readers otherwise",
                "automatic_policy": "budgeted single chunk; matching resident multi-chunk prefilter; readers otherwise",
                "rows": rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
