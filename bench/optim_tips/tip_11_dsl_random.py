########################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
########################################################################
"""Qualify an integer hash, NOT an independent-sample random generator.

Public portable authoring: @blosc2.menudet.
The legacy-engine arm is an internal performance control, not a supported API.
Every route materializes an int32 NDArray with identical storage settings.
Example: python bench/optim_tips/tip_11_dsl_random.py --items 20000000 --compiler cc
"""

import argparse
import gc
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np

import blosc2
from blosc2.dsl_kernel import SourceKernel

# Typed captures avoid weak-scalar intermediates. Each multiply stays below
# 2**59; masking performs uint32 truncation, and the final cast wraps to int32.
SEED = np.int64(42)
SHIFT = np.int64(16)
MULTIPLIER = np.int64(0x45D9F3B)
MASK = np.int64(0xFFFFFFFF)


@blosc2.menudet(ndim=1, require_jit=True)
def random_int32() -> np.int32:
    x = _flat_idx ^ SEED  # noqa: F821
    x = (((x >> SHIFT) ^ x) * MULTIPLIER) & MASK
    x = (((x >> SHIFT) ^ x) * MULTIPLIER) & MASK
    return int32((x >> SHIFT) ^ x)  # noqa: F821


@SourceKernel
def legacy_random_int32(seed):
    x = _flat_idx ^ seed  # noqa: F821
    x = (((x >> 16) ^ x) * 0x45D9F3B) & 0xFFFFFFFF
    x = (((x >> 16) ^ x) * 0x45D9F3B) & 0xFFFFFFFF
    return (x >> 16) ^ x


def reference(indices):
    """Independent unsigned NumPy implementation; bit-exact, including signed output."""
    x = np.asarray(indices, dtype=np.uint64) ^ np.uint64(42)
    for _ in range(2):
        x = ((x ^ (x >> np.uint64(16))) * np.uint64(0x45D9F3B)) & np.uint64(0xFFFFFFFF)
    return (x ^ (x >> np.uint64(16))).astype(np.uint32).view(np.int32)


def bind_hash(items, partitions):
    return random_int32.specialize().lazy({}, shape=(items,), partitions=(partitions,))


def prepare(variant, args):
    storage = {"chunks": (args.chunks,), "blocks": (args.blocks,), "cparams": {"clevel": 0}}
    if variant == "numpy":

        def compute():
            rng = np.random.default_rng(42)
            return blosc2.asarray(rng.integers(-(2**31), 2**31, size=args.items, dtype=np.int32), **storage)

        return compute, None, None
    if variant in ("menudet", "menudet-serial"):
        lazy = bind_hash(args.items, args.blocks)
        if variant == "menudet-serial":
            # Internal control retaining the pre-prefilter materialization loop.
            def compute():
                output = blosc2.empty(lazy.shape, dtype=lazy.dtype, **storage)
                for start in range(0, args.items, output.chunks[0]):
                    item = slice(start, min(start + output.chunks[0], args.items))
                    output[item] = lazy[item]
                return output
        else:

            def compute():
                return lazy.compute(**storage)

        return compute, random_int32.specialize().has_jit, lazy
    lazy = blosc2.lazyudf(
        legacy_random_int32,
        (42,),
        shape=(args.items,),
        dtype=np.int32,
        chunks=(args.chunks,),
        blocks=(args.blocks,),
    )
    if variant == "legacy-engine":
        # Explicit internal control: bypass the Menudet compatibility adapter.
        # This does NOT restore public legacy authoring or load a saved recipe.
        lazy._legacy_source_recipe = True
    return lambda: lazy.compute(**storage), None, lazy


def peak_rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)


def worker(args):
    blosc2.set_nthreads(args.threads)
    with blosc2.jit_options(jit=True, jit_backend=args.compiler):
        start = perf_counter()
        compute, has_jit, lazy = prepare(args.worker, args)
        preparation = perf_counter() - start
        baseline = peak_rss()
        start = perf_counter()
        result = compute()
        first = perf_counter() - start
        peak = peak_rss()
        legacy_probe = None
        if args.worker == "legacy-engine":
            source = legacy_random_int32.dsl_source.replace("(seed):", "():").replace("^ seed", "^ 42")
            legacy_probe = result._dsl_jit_status(source, {})
            if not legacy_probe["jit"]:
                raise RuntimeError(f"Legacy control did not produce JIT code: {legacy_probe}")
        assert result.dtype == np.dtype("int32")
        assert result.shape == (args.items,)
        output_grid = {"chunks": result.chunks, "blocks": result.blocks}
        if args.worker != "numpy":
            indices = np.unique(np.linspace(0, args.items - 1, min(args.items, 257), dtype=np.int64))
            np.testing.assert_array_equal(np.array([result[int(i)] for i in indices]), reference(indices))
        del result
        samples = []
        if not args.memory_only:
            for _ in range(args.repeat):
                gc.collect()
                start = perf_counter()
                result = compute()
                samples.append(perf_counter() - start)
                del result
        print(
            json.dumps(
                {
                    "variant": args.worker,
                    "preparation_s": preparation,
                    "first_compute_s": first,
                    "samples_s": samples,
                    "median_s": median(samples) if samples else None,
                    "peak_rss_bytes": peak,
                    "peak_growth_bytes": max(0, peak - baseline),
                    "has_jit": has_jit,
                    "legacy_jit_probe": legacy_probe,
                    "execution_report": getattr(lazy, "_source_execution_report", None)
                    or getattr(lazy, "_native_execution_report", None),
                    "output_grid": output_grid,
                    "source_blocks": getattr(lazy, "blocks", None),
                    "logical_partitions": getattr(lazy, "partitions", None),
                    "package": blosc2.__file__,
                    "extension": blosc2.blosc2_ext.__file__,
                }
            )
        )


def run_worker(args, variant, *, memory=False):
    command = [sys.executable]
    if sys.flags.no_site:
        command.append("-S")
    command += [
        str(Path(__file__).resolve()),
        "--worker",
        variant,
        "--items",
        str(args.items),
        "--repeat",
        str(args.repeat),
        "--threads",
        str(args.threads),
        "--compiler",
        args.compiler,
        "--chunks",
        str(args.chunks),
        "--blocks",
        str(args.blocks),
    ]
    if memory:
        command.append("--memory-only")
    # Preserve explicitly isolated imports, including when running Python -S.
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    completed = subprocess.run(command, env=env, capture_output=True, text=True, check=True)
    return json.loads(completed.stdout.strip().splitlines()[-1])


def quality_stats(v):
    """Descriptive checks, not RNG certification (this hash is a permutation)."""
    f = v.astype(np.float64)
    h = np.histogram(v, bins=256, range=(-(2**31), 2**31))[0]
    expected = len(v) / 256
    n_dups = min(len(v), 1_000_000)
    return {
        "mean": float(f.mean()),
        "std_scaled": float(f.std() / 2**31),
        "chi2": float(((h - expected) ** 2 / expected).sum()),
        "lag1": float(np.corrcoef(f[:-1], f[1:])[0, 1]),
        "duplicates": int(n_dups - len(np.unique(v[:n_dups]))),
        "duplicate_sample": n_dups,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=int, default=200_000_000)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--threads", type=int, default=blosc2.nthreads)
    parser.add_argument("--compiler", choices=("cc", "tcc"), default="cc")
    parser.add_argument("--chunks", type=int, default=4_194_304)
    parser.add_argument("--blocks", type=int, default=262_144)
    parser.add_argument("--quality-items", type=int, default=1_000_000)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plot", action="store_true", help="regenerate the guide's hash comparison figure")
    parser.add_argument(
        "--worker",
        choices=("numpy", "legacy-adapter", "legacy-engine", "menudet-serial", "menudet"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--memory-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 1 <= args.items <= 2**32 or min(args.repeat, args.threads, args.chunks, args.blocks) < 1:
        parser.error("require 1 <= items <= 2**32 and positive repeats/threads/chunks/blocks")
    if args.chunks % args.blocks or args.quality_items < 0:
        parser.error("chunks must be a multiple of blocks; quality-items must be nonnegative")
    # Select the same explicit native backend for every internal control arm,
    # including the direct legacy compilation probe. Menudet itself also
    # honors jit_options(jit_backend=...) without environment mutation.
    os.environ["ME_DSL_JIT_COMPILER"] = args.compiler
    if args.worker:
        worker(args)
        return
    metadata = {
        "authoring": "blosc2.menudet",
        "package": blosc2.__file__,
        "extension": blosc2.blosc2_ext.__file__,
        "python": sys.version,
        "numpy": np.__version__,
        "blosc2": blosc2.__version__,
        "c_blosc2": blosc2.blosclib_version,
        "platform": platform.platform(),
        "cpu": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        if sys.platform == "darwin"
        else platform.processor(),
        "arguments": vars(args) | {"output": str(args.output) if args.output else None},
        "environment": {k: v for k, v in os.environ.items() if k.startswith(("ME_", "BLOSC_ME"))},
        "extension_sha256": hashlib.sha256(Path(blosc2.blosc2_ext.__file__).read_bytes()).hexdigest(),
        "execution_options": blosc2.get_jit_options(),
        "protocol": "Separate subprocess per route and memory run; preparation, first compute, then warm "
        "samples. Timings include allocation and clevel=0 storage, exclude imports. "
        "Memory is OS process peak RSS during first compute; also report growth over preparation.",
    }
    metadata["execution_options"]["fp_accuracy"] = str(metadata["execution_options"]["fp_accuracy"])
    root = Path(__file__).resolve().parents[2]
    metadata["revisions"] = {}
    for label, directory in (("python_blosc2", root), ("miniexpr", root.parent / "miniexpr")):
        if (directory / ".git").exists():
            metadata["revisions"][label] = {
                "head": subprocess.check_output(
                    ["git", "-C", str(directory), "rev-parse", "HEAD"], text=True
                ).strip(),
                "tracked_changes": subprocess.check_output(
                    ["git", "-C", str(directory), "diff", "--stat"], text=True
                ),
            }
    print(f"Package: {metadata['package']}\nExtension: {metadata['extension']}")
    print(
        f"{args.items:,} int32; {args.threads} threads; compiler={args.compiler}; "
        f"chunks={args.chunks:,}, blocks/partitions={args.blocks:,}; clevel=0"
    )
    rows = []
    for variant in ("numpy", "legacy-engine", "legacy-adapter", "menudet-serial", "menudet"):
        row = run_worker(args, variant)
        memory = run_worker(args, variant, memory=True)
        row.update({key: memory[key] for key in ("peak_rss_bytes", "peak_growth_bytes")})
        rows.append(row)
        print(
            f"{variant:16s} prepare {row['preparation_s']:.4f}s  first {row['first_compute_s']:.4f}s  "
            f"warm {row['median_s']:.4f}s  peak RSS {row['peak_rss_bytes'] / 2**20:.1f} MiB  "
            f"growth {row['peak_growth_bytes'] / 2**20:.1f} MiB  JIT={row['has_jit']}",
            flush=True,
        )
    by_name = {row["variant"]: row for row in rows}
    modern_time = by_name["menudet"]["median_s"]
    print(
        f"Menudet: {by_name['menudet-serial']['median_s'] / modern_time:.2f}x faster than serial; "
        f"{modern_time / by_name['legacy-engine']['median_s']:.2f}x legacy execution time"
    )
    quality = {}
    count = min(args.items, args.quality_items)
    if count >= 2:
        with blosc2.jit_options(jit=True, jit_backend=args.compiler):
            quality["hash"] = quality_stats(bind_hash(count, args.blocks)[:])
        quality["pcg64"] = quality_stats(
            np.random.default_rng(42).integers(-(2**31), 2**31, size=count, dtype=np.int32)
        )
        print("Quality (descriptive only; hash has zero duplicates by construction):", quality)
    report = {"metadata": metadata, "results": rows, "quality": quality}
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    if args.plot:
        from common import save_plot

        numpy_row, modern = rows[0], rows[-1]
        save_plot(
            "tip_11_dsl_random.png",
            f"Integer hash vs PCG64 staging — {args.items:,} int32 items",
            "NumPy PCG64\n+ asarray",
            "Menudet hash\n(not an RNG)",
            numpy_row["median_s"],
            modern["median_s"],
            numpy_row["peak_rss_bytes"],
            modern["peak_rss_bytes"],
        )


if __name__ == "__main__":
    main()
