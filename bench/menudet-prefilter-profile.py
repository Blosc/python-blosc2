"""Profile portable prefilter wall phases, callback work and isolated ablations."""

import argparse
import json
import os
import platform
import random
import subprocess
from statistics import median
from threading import Thread
from time import perf_counter

import numpy as np

import blosc2


@blosc2.menudet
def adjusted(x, y):
    return x + (y - 1)


def measure(operation, expected, samples, report=None):
    np.testing.assert_array_equal(operation(), expected)
    rows, times = [], []
    for _ in range(samples):
        start = perf_counter()
        values = operation()
        times.append((perf_counter() - start) * 1000)
        if report is not None:
            rows.append(dict(report()))
        del values
    return {
        "wall_ms": round(median(times), 4),
        "phases": {
            key: round(median(row[key] for row in rows), 4) for key in rows[0] if key.startswith("profile_")
        }
        if rows
        else {},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--items", type=int, default=8_000_000)
    parser.add_argument("--modes", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--native-sample")
    args = parser.parse_args()
    if args.samples <= 0 or args.items < 2_000_000 or args.items % 1_000_000:
        parser.error("samples must be positive; items must be a multiple of 1M and at least 2M")
    if any(mode not in range(4) for mode in args.modes):
        parser.error("modes must be 0 (normal), 1 (no evaluation), 2 (no decode), or 3 (neither)")
    x, y = np.linspace(0, 1, args.items), np.linspace(1, 2, args.items)
    expected = x + (y - 1)
    compressed = {
        "x": blosc2.asarray(x, chunks=(1_000_000,), blocks=(15625,)),
        "y": blosc2.asarray(y, chunks=(1_000_000,), blocks=(15625,)),
    }
    lazy = adjusted(**compressed)
    resident = adjusted(x, y)
    assert lazy.kernel.has_jit
    result = {
        "platform": platform.platform(),
        "extension": blosc2.blosc2_ext.__file__,
        "samples": args.samples,
        "items": args.items,
        "threads": blosc2.nthreads,
        "inputs": {
            name: {"cparams": repr(value.schunk.cparams), "cbytes": value.schunk.cbytes}
            for name, value in compressed.items()
        },
        "resident": measure(lambda: resident[:], expected, args.samples),
        "modes": {},
    }
    references = {
        mode: expected if mode == 0 else np.full(args.items, -1.0 if mode == 2 else 0.0)
        for mode in args.modes
    }
    times, reports = {mode: [] for mode in args.modes}, {mode: [] for mode in args.modes}
    for mode in args.modes:
        os.environ["B2_PREFILTER_PROFILE_MODE"] = str(mode)
        np.testing.assert_array_equal(lazy[:], references[mode])
    generator = random.Random(42)
    for _ in range(args.samples):
        order = list(args.modes)
        generator.shuffle(order)
        for mode in order:
            os.environ["B2_PREFILTER_PROFILE_MODE"] = str(mode)
            start = perf_counter()
            values = lazy[:]
            times[mode].append((perf_counter() - start) * 1000)
            reports[mode].append(dict(lazy._source_execution_report))
            del values
    for mode in args.modes:
        rows = reports[mode]
        result["modes"][str(mode)] = {
            "wall_ms": round(median(times[mode]), 4),
            "phases": {
                key: round(median(row[key] for row in rows), 4)
                for key in rows[0]
                if key.startswith("profile_")
            },
        }
    os.environ["B2_PREFILTER_PROFILE_MODE"] = "0"
    result["output_report"] = reports[args.modes[0]][-1]
    # Full-chunk parallel decoding cost on live input storage, without evaluation.
    buffers = {name: np.empty(1_000_000, dtype=np.float64) for name in compressed}

    def decode():
        for chunk in range(compressed["x"].schunk.nchunks):
            for name, value in compressed.items():
                value.schunk.decompress_chunk(chunk, dst=buffers[name])
        return buffers["x"]

    final_chunk = x[-1_000_000:]
    result["input_decode_only"] = measure(decode, final_chunk, args.samples)
    if args.native_sample:
        sampler = Thread(
            target=subprocess.run,
            args=(["/usr/bin/sample", str(os.getpid()), "5", "1", "-file", args.native_sample],),
            kwargs={"check": True, "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL},
        )
        sampler.start()
        end = perf_counter() + 7
        while perf_counter() < end:
            lazy[:]
        sampler.join()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
