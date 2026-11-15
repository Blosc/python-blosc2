"""Benchmark synthetic constructors in an explicitly selected isolated package.

Run with ``conda run -n blosc2 python -S ... --package /path/to/package``.
The package directory must contain blosc2 and its matching compiled extensions.
"""

import argparse
import gc
import hashlib
import json
import statistics
import sys
import sysconfig
import time
from pathlib import Path


def _benchmark_reads(arr, args):
    import numpy as np

    nchunks = (args.size + arr.chunks[0] - 1) // arr.chunks[0]
    chunks = np.linspace(0, nchunks - 1, min(args.read_chunks, nchunks), dtype=np.int64)
    if args.read_mode == "chunk":
        values = np.empty(arr.chunks[0], dtype=arr.dtype)
    start = time.perf_counter()
    for chunk in chunks:
        if args.read_mode == "chunk":
            arr.schunk.decompress_chunk(int(chunk), values)
        else:
            first = int(chunk) * arr.chunks[0]
            values = arr[first : first + arr.chunks[0]]
    elapsed = time.perf_counter() - start
    return {
        "read_seconds": elapsed,
        "read_nbytes": sum(
            min(arr.chunks[0], args.size - int(chunk) * arr.chunks[0]) * arr.dtype.itemsize
            for chunk in chunks
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=10**9)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--levels", type=int, nargs="+", choices=range(1, 10), default=[1])
    parser.add_argument(
        "--read-chunks", type=int, default=0, help="Read this many spread-out chunks per cell"
    )
    parser.add_argument("--read-mode", choices=("slice", "chunk"), default="slice")
    parser.add_argument(
        "--validate-all", action="store_true", help="Verify every element on the first trial"
    )
    args = parser.parse_args()
    if args.size < 2 or args.repeats < 1 or args.warmups < 0 or args.read_chunks < 0:
        parser.error("size must be >= 2, repeats >= 1, and warmups/read-chunks >= 0")
    package = args.package.resolve()
    sys.path[:0] = [str(package), sysconfig.get_paths()["purelib"]]
    import numpy as np

    import blosc2

    assert Path(blosc2.__file__).is_relative_to(package)
    extension = Path(blosc2.blosc2_ext.__file__)
    assert extension.is_relative_to(package)
    report = {
        "package": str(package),
        "extension_sha256": hashlib.sha256(extension.read_bytes()).hexdigest(),
        "size": args.size,
        "threads": args.threads,
        "levels": args.levels,
        "read_chunks": args.read_chunks,
        "read_mode": args.read_mode,
        "fully_validated": args.validate_all,
        "rows": [],
    }
    cells = [
        (task, codec, level)
        for task in ("arange", "linspace")
        for codec in ("BLOSCLZ", "LZ4")
        for level in args.levels
    ]
    for trial in range(args.warmups + args.repeats):
        order = cells[trial % len(cells) :] + cells[: trial % len(cells)]
        if trial % 2:
            order.reverse()
        for task, codec, level in order:
            cparams = {
                "codec": getattr(blosc2.Codec, codec),
                "clevel": level,
                "filters": [blosc2.Filter.SHUFFLE, blosc2.Filter.BYTEDELTA],
                "nthreads": args.threads,
            }
            gc.collect()
            start = time.perf_counter()
            if task == "arange":
                arr = blosc2.arange(args.size, cparams=cparams)
            else:
                arr = blosc2.linspace(0, 1, args.size, cparams=cparams)
            elapsed = time.perf_counter() - start
            indices = sorted({0, 1, arr.blocks[0] - 1, arr.blocks[0], args.size // 2, args.size - 1})
            indices = [i for i in indices if i < args.size]
            expected = np.asarray(indices, dtype=arr.dtype)
            if task == "linspace":
                expected = expected / (args.size - 1)
            np.testing.assert_allclose([arr[i] for i in indices], expected, rtol=1e-14, atol=0)
            if args.validate_all and trial == 0:
                for first in range(0, args.size, arr.chunks[0]):
                    stop = min(first + arr.chunks[0], args.size)
                    expected = np.arange(first, stop, dtype=arr.dtype)
                    if task == "linspace":
                        expected *= 1 / (args.size - 1)
                    np.testing.assert_allclose(arr[first:stop], expected, rtol=1e-14, atol=0)
            row = {
                "task": task,
                "codec": codec,
                "level": level,
                "trial": trial - args.warmups,
                "seconds": elapsed,
                "cbytes": arr.schunk.cbytes,
                "cratio": arr.schunk.cratio,
                "chunks": arr.chunks,
                "blocks": arr.blocks,
                "dtype": str(arr.dtype),
            }
            if args.read_chunks:
                row.update(_benchmark_reads(arr, args))
            if trial >= args.warmups:
                report["rows"].append(row)
            print(json.dumps(row), flush=True)
            del arr
    report["summary"] = [
        {
            "task": task,
            "codec": codec,
            "level": level,
            "median_seconds": statistics.median(
                r["seconds"]
                for r in report["rows"]
                if (r["task"], r["codec"], r["level"]) == (task, codec, level)
            ),
        }
        for task, codec, level in cells
    ]
    if args.read_chunks:
        for summary in report["summary"]:
            summary["median_read_seconds"] = statistics.median(
                r["read_seconds"]
                for r in report["rows"]
                if (r["task"], r["codec"], r["level"])
                == (summary["task"], summary["codec"], summary["level"])
            )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
