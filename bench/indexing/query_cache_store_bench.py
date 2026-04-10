#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import statistics
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import blosc2
from blosc2 import indexing

STRATEGIES = ("baseline", "cache_catalog", "skip_cbytes", "defer_vlmeta", "all")


@dataclass
class InsertState:
    catalog: dict | None = None
    store: object | None = None


def _make_array(path: Path, *, size: int, chunks: int, blocks: int) -> blosc2.NDArray:
    return blosc2.asarray(
        np.arange(size, dtype=np.int64),
        urlpath=path,
        mode="w",
        chunks=(chunks,),
        blocks=(blocks,),
    )


def _clear_process_caches() -> None:
    indexing._hot_cache_clear()
    indexing._QUERY_CACHE_STORE_HANDLES.clear()
    indexing._PERSISTENT_INDEXES.clear()


def _coords_for_count(count: int, spacing: int, modulo: int) -> np.ndarray:
    coords = (np.arange(count, dtype=np.int64) * spacing) % modulo
    return np.sort(coords, kind="stable")


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def _build_query_bits(arr: blosc2.NDArray, expr: str, coords: np.ndarray) -> tuple[str, dict, dict]:
    descriptor = indexing._normalize_query_descriptor(expr, [indexing.SELF_TARGET_NAME], None)
    digest = indexing._query_cache_digest(descriptor)
    scope = indexing._query_cache_scope(arr)
    indexing._hot_cache_put(digest, coords, scope=scope)
    payload_mapping = indexing._encode_coords_payload(coords)
    return digest, descriptor, payload_mapping


def _load_or_create_catalog(arr: blosc2.NDArray, state: InsertState | None, strategy: str) -> dict:
    if strategy in {"cache_catalog", "defer_vlmeta", "all"} and state is not None and state.catalog is not None:
        return state.catalog

    catalog = indexing._load_query_cache_catalog(arr)
    if catalog is None:
        catalog = indexing._default_query_cache_catalog(indexing._query_cache_payload_path(arr))

    if strategy in {"cache_catalog", "defer_vlmeta", "all"} and state is not None:
        state.catalog = catalog
    return catalog


def _load_or_create_store(arr: blosc2.NDArray, state: InsertState | None, strategy: str):
    if strategy in {"cache_catalog", "defer_vlmeta", "all"} and state is not None and state.store is not None:
        return state.store

    store = indexing._open_query_cache_store(arr, create=True)
    if strategy in {"cache_catalog", "defer_vlmeta", "all"} and state is not None:
        state.store = store
    return store


def _entry_nbytes(coords: np.ndarray, payload_mapping: dict, strategy: str) -> int:
    if strategy in {"skip_cbytes", "all"}:
        return len(payload_mapping["data"])
    return indexing._query_cache_entry_nbytes(coords)


def _insert_with_strategy(
    arr: blosc2.NDArray,
    expr: str,
    coords: np.ndarray,
    strategy: str,
    state: InsertState | None = None,
) -> float:
    start = time.perf_counter_ns()
    digest, descriptor, payload_mapping = _build_query_bits(arr, expr, coords)
    nbytes = _entry_nbytes(coords, payload_mapping, strategy)
    catalog = _load_or_create_catalog(arr, state, strategy)
    if digest in catalog.get("entries", {}):
        end = time.perf_counter_ns()
        return (end - start) / 1_000_000

    store = _load_or_create_store(arr, state, strategy)
    slot = len(store)
    store.append(payload_mapping)

    catalog["entries"][digest] = {
        "slot": slot,
        "nbytes": nbytes,
        "nrows": len(coords),
        "dtype": payload_mapping["dtype"],
        "query": descriptor,
    }
    catalog["persistent_nbytes"] = int(catalog.get("persistent_nbytes", 0)) + nbytes
    catalog["next_slot"] = slot + 1

    if strategy not in {"defer_vlmeta", "all"}:
        indexing._save_query_cache_catalog(arr, catalog)
    elif state is not None:
        state.catalog = catalog

    end = time.perf_counter_ns()
    return (end - start) / 1_000_000


def _flush_state(arr: blosc2.NDArray, state: InsertState | None, strategy: str) -> None:
    if strategy not in {"defer_vlmeta", "all"} or state is None or state.catalog is None:
        return
    indexing._save_query_cache_catalog(arr, state.catalog)


def _benchmark_fresh(
    root: Path,
    *,
    strategy: str,
    coords: np.ndarray,
    size: int,
    chunks: int,
    blocks: int,
    repeats: int,
) -> float:
    runs = []
    for idx in range(repeats):
        arr = _make_array(root / f"fresh-{strategy}-{idx}.b2nd", size=size, chunks=chunks, blocks=blocks)
        _clear_process_caches()
        state = InsertState() if strategy in {"cache_catalog", "defer_vlmeta", "all"} else None
        expr = f"(id >= {idx}) & (id <= {idx})"
        start = time.perf_counter_ns()
        _insert_with_strategy(arr, expr, coords, strategy, state)
        _flush_state(arr, state, strategy)
        end = time.perf_counter_ns()
        runs.append((end - start) / 1_000_000)
    return _median(runs)


def _benchmark_steady(
    root: Path,
    *,
    strategy: str,
    coords: np.ndarray,
    size: int,
    chunks: int,
    blocks: int,
    inserts: int,
) -> float:
    arr = _make_array(root / f"steady-{strategy}.b2nd", size=size, chunks=chunks, blocks=blocks)
    _clear_process_caches()
    state = InsertState() if strategy in {"cache_catalog", "defer_vlmeta", "all"} else None
    start = time.perf_counter_ns()
    for idx in range(inserts):
        expr = f"(id >= {idx}) & (id <= {idx})"
        _insert_with_strategy(arr, expr, coords, strategy, state)
    _flush_state(arr, state, strategy)
    end = time.perf_counter_ns()
    return ((end - start) / 1_000_000) / max(1, inserts)


def _baseline_step_breakdown(
    arr: blosc2.NDArray, expr: str, coords: np.ndarray
) -> dict[str, float | int]:
    t0 = time.perf_counter_ns()
    descriptor = indexing._normalize_query_descriptor(expr, [indexing.SELF_TARGET_NAME], None)
    digest = indexing._query_cache_digest(descriptor)
    t1 = time.perf_counter_ns()

    scope = indexing._query_cache_scope(arr)
    indexing._hot_cache_put(digest, coords, scope=scope)
    t2 = time.perf_counter_ns()

    payload_mapping = indexing._encode_coords_payload(coords)
    nbytes = indexing._query_cache_entry_nbytes(coords)
    t3 = time.perf_counter_ns()

    catalog = indexing._load_query_cache_catalog(arr)
    payload_path = indexing._query_cache_payload_path(arr)
    if catalog is None:
        catalog = indexing._default_query_cache_catalog(payload_path)
    store = indexing._open_query_cache_store(arr, create=True)
    t4 = time.perf_counter_ns()

    slot = len(store)
    store.append(payload_mapping)
    t5 = time.perf_counter_ns()

    catalog["entries"][digest] = {
        "slot": slot,
        "nbytes": nbytes,
        "nrows": len(coords),
        "dtype": payload_mapping["dtype"],
        "query": descriptor,
    }
    catalog["persistent_nbytes"] = int(catalog.get("persistent_nbytes", 0)) + nbytes
    catalog["next_slot"] = slot + 1
    indexing._save_query_cache_catalog(arr, catalog)
    t6 = time.perf_counter_ns()

    return {
        "digest_ms": (t1 - t0) / 1_000_000,
        "hot_ms": (t2 - t1) / 1_000_000,
        "encode_nbytes_ms": (t3 - t2) / 1_000_000,
        "open_store_ms": (t4 - t3) / 1_000_000,
        "append_ms": (t5 - t4) / 1_000_000,
        "catalog_ms": (t6 - t5) / 1_000_000,
        "step_total_ms": (t6 - t0) / 1_000_000,
        "entry_nbytes": nbytes,
    }


def _profile_store(arr: blosc2.NDArray, coords: np.ndarray, repeats: int, top: int) -> str:
    profiler = cProfile.Profile()

    def run():
        for idx in range(repeats):
            expr = f"(id >= {idx}) & (id <= {idx})"
            indexing.store_cached_coords(arr, expr, [indexing.SELF_TARGET_NAME], None, coords)

    profiler.enable()
    run()
    profiler.disable()

    out = io.StringIO()
    stats = pstats.Stats(profiler, stream=out).sort_stats("cumulative")
    stats.print_stats(top)
    return out.getvalue()


def _active_cache_store_cparams(arr: blosc2.NDArray) -> blosc2.CParams:
    coords = np.asarray([0], dtype=np.int64)
    indexing.store_cached_coords(arr, "(id >= 0) & (id <= 0)", [indexing.SELF_TARGET_NAME], None, coords)
    payload_path = indexing._query_cache_payload_path(arr)
    store = blosc2.VLArray(storage=blosc2.Storage(urlpath=payload_path, mode="r"))
    return store.cparams


def _print_strategy_table(title: str, rows: list[dict[str, object]]) -> None:
    columns = [
        ("coords", lambda row: f"{row['coords_count']:,}"),
        ("strategy", lambda row: str(row["strategy"])),
        ("time_ms", lambda row: f"{row['time_ms']:.3f}"),
        ("speedup", lambda row: f"{row['speedup']:.2f}x"),
    ]
    widths = []
    for name, render in columns:
        width = len(name)
        for row in rows:
            width = max(width, len(render(row)))
        widths.append(width)

    print(title)
    header = "  ".join(name.ljust(width) for (name, _), width in zip(columns, widths, strict=True))
    rule = "  ".join("-" * width for width in widths)
    print(header)
    print(rule)
    for row in rows:
        print(
            "  ".join(
                render(row).ljust(width) for (_, render), width in zip(columns, widths, strict=True)
            )
        )
    print()


def _print_breakdown(rows: list[dict[str, object]]) -> None:
    columns = [
        ("coords", lambda row: f"{row['coords_count']:,}"),
        ("entry_nbytes", lambda row: f"{row['entry_nbytes']:,}"),
        ("digest_ms", lambda row: f"{row['digest_ms']:.3f}"),
        ("hot_ms", lambda row: f"{row['hot_ms']:.3f}"),
        ("encode_nbytes_ms", lambda row: f"{row['encode_nbytes_ms']:.3f}"),
        ("open_store_ms", lambda row: f"{row['open_store_ms']:.3f}"),
        ("append_ms", lambda row: f"{row['append_ms']:.3f}"),
        ("catalog_ms", lambda row: f"{row['catalog_ms']:.3f}"),
        ("step_total_ms", lambda row: f"{row['step_total_ms']:.3f}"),
    ]
    widths = []
    for name, render in columns:
        width = len(name)
        for row in rows:
            width = max(width, len(render(row)))
        widths.append(width)

    print("Baseline Step Breakdown")
    header = "  ".join(name.ljust(width) for (name, _), width in zip(columns, widths, strict=True))
    rule = "  ".join("-" * width for width in widths)
    print(header)
    print(rule)
    for row in rows:
        print(
            "  ".join(
                render(row).ljust(width) for (_, render), width in zip(columns, widths, strict=True)
            )
        )
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Microbenchmark persistent query-cache insert strategies.")
    parser.add_argument("--size", type=int, default=1_000_000, help="Array size for the backing persistent array.")
    parser.add_argument("--chunks", type=int, default=100_000, help="Chunk length for the backing array.")
    parser.add_argument("--blocks", type=int, default=10_000, help="Block length for the backing array.")
    parser.add_argument(
        "--coords-counts",
        type=int,
        nargs="+",
        default=[1, 10, 100, 1_000],
        help="Coordinate counts to benchmark.",
    )
    parser.add_argument("--fresh-repeats", type=int, default=20, help="Repeated fresh first-insert runs.")
    parser.add_argument("--steady-inserts", type=int, default=100, help="Repeated inserts into one array.")
    parser.add_argument(
        "--breakdown-repeats", type=int, default=20, help="Repeated baseline step breakdown runs."
    )
    parser.add_argument(
        "--spacing",
        type=int,
        default=9973,
        help="Stride used to synthesize sparse sorted coordinates.",
    )
    parser.add_argument(
        "--profile-repeats",
        type=int,
        default=200,
        help="Number of repeated baseline inserts to include in the cProfile run.",
    )
    parser.add_argument(
        "--profile-top",
        type=int,
        default=25,
        help="Number of cProfile entries to print.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fresh_rows = []
    steady_rows = []
    breakdown_rows = []

    with tempfile.TemporaryDirectory(prefix="blosc2-query-cache-bench-") as tmpdir:
        root = Path(tmpdir)
        probe = _make_array(root / "cparams-probe.b2nd", size=args.size, chunks=args.chunks, blocks=args.blocks)
        _clear_process_caches()
        active_cparams = _active_cache_store_cparams(probe)
        _clear_process_caches()

        for coords_count in args.coords_counts:
            coords = _coords_for_count(coords_count, args.spacing, args.size)

            fresh_times = {}
            steady_times = {}
            for strategy in STRATEGIES:
                fresh_times[strategy] = _benchmark_fresh(
                    root,
                    strategy=strategy,
                    coords=coords,
                    size=args.size,
                    chunks=args.chunks,
                    blocks=args.blocks,
                    repeats=args.fresh_repeats,
                )
                steady_times[strategy] = _benchmark_steady(
                    root,
                    strategy=strategy,
                    coords=coords,
                    size=args.size,
                    chunks=args.chunks,
                    blocks=args.blocks,
                    inserts=args.steady_inserts,
                )

            fresh_baseline = fresh_times["baseline"]
            steady_baseline = steady_times["baseline"]
            for strategy in STRATEGIES:
                fresh_rows.append(
                    {
                        "coords_count": coords_count,
                        "strategy": strategy,
                        "time_ms": fresh_times[strategy],
                        "speedup": fresh_baseline / fresh_times[strategy] if fresh_times[strategy] else 0.0,
                    }
                )
                steady_rows.append(
                    {
                        "coords_count": coords_count,
                        "strategy": strategy,
                        "time_ms": steady_times[strategy],
                        "speedup": steady_baseline / steady_times[strategy] if steady_times[strategy] else 0.0,
                    }
                )

            baseline_steps = []
            for idx in range(args.breakdown_repeats):
                arr = _make_array(root / f"breakdown-{coords_count}-{idx}.b2nd", size=args.size, chunks=args.chunks, blocks=args.blocks)
                _clear_process_caches()
                expr = f"(id >= {idx}) & (id <= {idx})"
                baseline_steps.append(_baseline_step_breakdown(arr, expr, coords))
            breakdown_rows.append(
                {
                    "coords_count": coords_count,
                    "entry_nbytes": int(_median([float(row["entry_nbytes"]) for row in baseline_steps])),
                    "digest_ms": _median([float(row["digest_ms"]) for row in baseline_steps]),
                    "hot_ms": _median([float(row["hot_ms"]) for row in baseline_steps]),
                    "encode_nbytes_ms": _median([float(row["encode_nbytes_ms"]) for row in baseline_steps]),
                    "open_store_ms": _median([float(row["open_store_ms"]) for row in baseline_steps]),
                    "append_ms": _median([float(row["append_ms"]) for row in baseline_steps]),
                    "catalog_ms": _median([float(row["catalog_ms"]) for row in baseline_steps]),
                    "step_total_ms": _median([float(row["step_total_ms"]) for row in baseline_steps]),
                }
            )

        print(
            "Persistent query-cache insert microbenchmark "
            f"(codec={active_cparams.codec.name}, clevel={active_cparams.clevel}, use_dict={active_cparams.use_dict})"
        )
        print()
        _print_strategy_table("Fresh Insert Comparison", fresh_rows)
        _print_strategy_table("Steady Insert Comparison", steady_rows)
        _print_breakdown(breakdown_rows)

        profile_coords = _coords_for_count(args.coords_counts[0], args.spacing, args.size)
        profile_arr = _make_array(root / "profile.b2nd", size=args.size, chunks=args.chunks, blocks=args.blocks)
        _clear_process_caches()
        print(f"Baseline cProfile for coords_count={args.coords_counts[0]:,} over {args.profile_repeats} inserts")
        print(_profile_store(profile_arr, profile_coords, args.profile_repeats, args.profile_top))


if __name__ == "__main__":
    main()
