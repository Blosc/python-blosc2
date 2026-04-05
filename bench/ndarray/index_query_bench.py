#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

from __future__ import annotations

import argparse
import os
import re
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np

import blosc2
from blosc2 import indexing as blosc2_indexing

SIZES = (1_000_000, 2_000_000, 5_000_000, 10_000_000)
CHUNK_LEN = 100_000
BLOCK_LEN = 20_000
DEFAULT_REPEATS = 3
KINDS = ("ultralight", "light", "medium", "full")
DISTS = ("sorted", "block-shuffled", "random")
RNG_SEED = 0
DEFAULT_OPLEVEL = 5
FULL_QUERY_MODES = ("auto", "selective-ooc", "whole-load")


def dtype_token(dtype: np.dtype) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", np.dtype(dtype).name).strip("_")


def make_ordered_ids(size: int, dtype: np.dtype) -> np.ndarray:
    dtype = np.dtype(dtype)
    if dtype == np.dtype(np.bool_):
        values = np.zeros(size, dtype=dtype)
        values[size // 2 :] = True
        return values

    if dtype.kind in {"i", "u"}:
        info = np.iinfo(dtype)
        unique_count = min(size, int(info.max) - int(info.min) + 1)
        start = int(info.min) if unique_count < size and dtype.kind == "i" else 0
        if dtype.kind == "i" and unique_count < size:
            start = max(int(info.min), -(unique_count // 2))
        positions = np.arange(size, dtype=np.int64)
        values = start + (positions * unique_count) // size
        return values.astype(dtype, copy=False)

    if dtype.kind == "f":
        span = max(1, size)
        return np.linspace(-span / 2, span / 2, num=size, endpoint=False, dtype=dtype)

    raise ValueError(f"unsupported dtype for benchmark: {dtype}")


def fill_ids(ids: np.ndarray, ordered_ids: np.ndarray, dist: str, rng: np.random.Generator) -> None:
    size = ids.shape[0]
    if dist == "sorted":
        ids[:] = ordered_ids
        return

    if dist == "block-shuffled":
        nblocks = (size + BLOCK_LEN - 1) // BLOCK_LEN
        order = rng.permutation(nblocks)
        dest = 0
        for src_block in order:
            src_start = int(src_block) * BLOCK_LEN
            src_stop = min(src_start + BLOCK_LEN, size)
            block_size = src_stop - src_start
            ids[dest : dest + block_size] = ordered_ids[src_start:src_stop]
            dest += block_size
        return

    if dist == "random":
        ids[:] = ordered_ids
        rng.shuffle(ids)
        return

    raise ValueError(f"unsupported distribution {dist!r}")


def make_source_data(size: int, dist: str, id_dtype: np.dtype) -> np.ndarray:
    dtype = np.dtype([("id", id_dtype), ("payload", np.float32)])
    data = np.zeros(size, dtype=dtype)
    fill_ids(data["id"], make_ordered_ids(size, id_dtype), dist, np.random.default_rng(RNG_SEED))
    return data


def build_array(data: np.ndarray) -> blosc2.NDArray:
    return blosc2.asarray(data, chunks=(CHUNK_LEN,), blocks=(BLOCK_LEN,))


def build_persistent_array(data: np.ndarray, path: Path) -> blosc2.NDArray:
    return blosc2.asarray(data, urlpath=path, mode="w", chunks=(CHUNK_LEN,), blocks=(BLOCK_LEN,))


def base_array_path(size_dir: Path, size: int, dist: str, id_dtype: np.dtype) -> Path:
    return size_dir / f"size_{size}_{dist}_{dtype_token(id_dtype)}.b2nd"


def indexed_array_path(
    size_dir: Path, size: int, dist: str, kind: str, optlevel: int, id_dtype: np.dtype, in_mem: bool
) -> Path:
    mode = "mem" if in_mem else "ooc"
    return size_dir / f"size_{size}_{dist}_{dtype_token(id_dtype)}.{kind}.opt{optlevel}.{mode}.b2nd"


def benchmark_scan_once(expr) -> tuple[float, int]:
    start = time.perf_counter()
    result = expr.compute(_use_index=False)[:]
    elapsed = time.perf_counter() - start
    return elapsed, len(result)


def benchmark_index_once(arr: blosc2.NDArray, cond) -> tuple[float, int]:
    start = time.perf_counter()
    result = arr[cond][:]
    elapsed = time.perf_counter() - start
    return elapsed, len(result)


def _with_full_query_mode(full_query_mode: str):
    class _FullQueryModeScope:
        def __enter__(self):
            self.previous = os.environ.get("BLOSC2_FULL_EXACT_QUERY_MODE")
            os.environ["BLOSC2_FULL_EXACT_QUERY_MODE"] = full_query_mode

        def __exit__(self, exc_type, exc, tb):
            if self.previous is None:
                os.environ.pop("BLOSC2_FULL_EXACT_QUERY_MODE", None)
            else:
                os.environ["BLOSC2_FULL_EXACT_QUERY_MODE"] = self.previous

    return _FullQueryModeScope()


def index_sizes(descriptor: dict) -> tuple[int, int]:
    logical = 0
    disk = 0
    for level_info in descriptor["levels"].values():
        dtype = np.dtype(level_info["dtype"])
        logical += dtype.itemsize * level_info["nsegments"]
        if level_info["path"]:
            disk += os.path.getsize(level_info["path"])

    light = descriptor.get("light")
    if light is not None:
        for key in ("values_path", "bucket_positions_path", "offsets_path"):
            array = blosc2.open(light[key])
            logical += int(np.prod(array.shape)) * array.dtype.itemsize
            disk += os.path.getsize(light[key])

    reduced = descriptor.get("reduced")
    if reduced is not None:
        values = blosc2.open(reduced["values_path"])
        positions = blosc2.open(reduced["positions_path"])
        offsets = blosc2.open(reduced["offsets_path"])
        logical += values.shape[0] * values.dtype.itemsize
        logical += positions.shape[0] * positions.dtype.itemsize
        logical += offsets.shape[0] * offsets.dtype.itemsize
        disk += os.path.getsize(reduced["values_path"])
        disk += os.path.getsize(reduced["positions_path"])
        disk += os.path.getsize(reduced["offsets_path"])

    full = descriptor.get("full")
    if full is not None:
        values = blosc2.open(full["values_path"])
        positions = blosc2.open(full["positions_path"])
        logical += values.shape[0] * values.dtype.itemsize
        logical += positions.shape[0] * positions.dtype.itemsize
        disk += os.path.getsize(full["values_path"])
        disk += os.path.getsize(full["positions_path"])
    return logical, disk


def _source_data_factory(size: int, dist: str, id_dtype: np.dtype):
    data = None

    def get_data() -> np.ndarray:
        nonlocal data
        if data is None:
            data = make_source_data(size, dist, id_dtype)
        return data

    return get_data


def _ordered_ids_factory(size: int, id_dtype: np.dtype):
    ordered_ids = None

    def get_ordered_ids() -> np.ndarray:
        nonlocal ordered_ids
        if ordered_ids is None:
            ordered_ids = make_ordered_ids(size, id_dtype)
        return ordered_ids

    return get_ordered_ids


def _query_bounds(ordered_ids: np.ndarray, query_width: int) -> tuple[object, object]:
    if ordered_ids.size == 0:
        raise ValueError("benchmark arrays must not be empty")

    lo_idx = ordered_ids.size // 2
    hi_idx = min(ordered_ids.size - 1, lo_idx + max(query_width - 1, 0))
    return ordered_ids[lo_idx].item(), ordered_ids[hi_idx].item()


def _literal(value: object, dtype: np.dtype) -> str:
    dtype = np.dtype(dtype)
    if dtype == np.dtype(np.bool_):
        return "True" if bool(value) else "False"
    if dtype.kind == "f":
        return repr(float(value))
    if dtype.kind in {"i", "u"}:
        return str(int(value))
    raise ValueError(f"unsupported dtype for literal formatting: {dtype}")


def _condition_expr(lo: object, hi: object, dtype: np.dtype) -> str:
    lo_literal = _literal(lo, dtype)
    hi_literal = _literal(hi, dtype)
    return f"(id >= {lo_literal}) & (id <= {hi_literal})"


def _valid_index_descriptor(arr: blosc2.NDArray, kind: str, optlevel: int, in_mem: bool) -> dict | None:
    for descriptor in arr.indexes:
        if descriptor.get("version") != blosc2_indexing.INDEX_FORMAT_VERSION:
            continue
        if (
            descriptor.get("field") == "id"
            and descriptor.get("kind") == kind
            and int(descriptor.get("optlevel", -1)) == int(optlevel)
            and bool(descriptor.get("ooc", False)) is (not bool(in_mem))
            and not descriptor.get("stale", False)
        ):
            return descriptor
    return None


def _open_or_build_persistent_array(path: Path, get_data) -> blosc2.NDArray:
    if path.exists():
        return blosc2.open(path, mode="a")
    blosc2.remove_urlpath(path)
    return build_persistent_array(get_data(), path)


def _open_or_build_indexed_array(
    path: Path, get_data, kind: str, optlevel: int, in_mem: bool
) -> tuple[blosc2.NDArray, float]:
    if path.exists():
        arr = blosc2.open(path, mode="a")
        if _valid_index_descriptor(arr, kind, optlevel, in_mem) is not None:
            return arr, 0.0
        if arr.indexes:
            arr.drop_index(field="id")
        blosc2.remove_urlpath(path)

    arr = build_persistent_array(get_data(), path)
    build_start = time.perf_counter()
    arr.create_index(field="id", kind=kind, optlevel=optlevel, in_mem=in_mem)
    return arr, time.perf_counter() - build_start


def benchmark_size(
    size: int,
    size_dir: Path,
    dist: str,
    query_width: int,
    optlevel: int,
    id_dtype: np.dtype,
    in_mem: bool,
    full_query_mode: str,
) -> list[dict]:
    get_data = _source_data_factory(size, dist, id_dtype)
    get_ordered_ids = _ordered_ids_factory(size, id_dtype)
    arr = _open_or_build_persistent_array(base_array_path(size_dir, size, dist, id_dtype), get_data)
    lo, hi = _query_bounds(get_ordered_ids(), query_width)
    condition_str = _condition_expr(lo, hi, id_dtype)
    condition = blosc2.lazyexpr(condition_str, arr.fields)
    expr = condition.where(arr)
    base_bytes = size * arr.dtype.itemsize
    compressed_base_bytes = os.path.getsize(arr.urlpath)

    scan_ms = benchmark_scan_once(expr)[0] * 1_000

    rows = []
    for kind in KINDS:
        idx_arr, build_time = _open_or_build_indexed_array(
            indexed_array_path(size_dir, size, dist, kind, optlevel, id_dtype, in_mem),
            get_data,
            kind,
            optlevel,
            in_mem,
        )
        idx_cond = blosc2.lazyexpr(condition_str, idx_arr.fields)
        idx_expr = idx_cond.where(idx_arr)
        with _with_full_query_mode(full_query_mode):
            explanation = idx_expr.explain()
            cold_time, index_len = benchmark_index_once(idx_arr, idx_cond)
        descriptor = idx_arr.indexes[0]
        logical_index_bytes, disk_index_bytes = index_sizes(descriptor)

        rows.append(
            {
                "size": size,
                "dist": dist,
                "kind": kind,
                "optlevel": optlevel,
                "in_mem": in_mem,
                "query_rows": index_len,
                "build_s": build_time,
                "create_idx_ms": build_time * 1_000,
                "scan_ms": scan_ms,
                "cold_ms": cold_time * 1_000,
                "cold_speedup": scan_ms / (cold_time * 1_000),
                "warm_ms": None,
                "warm_speedup": None,
                "candidate_units": explanation["candidate_units"],
                "total_units": explanation["total_units"],
                "lookup_path": explanation.get("lookup_path"),
                "full_query_mode": full_query_mode,
                "logical_index_bytes": logical_index_bytes,
                "disk_index_bytes": disk_index_bytes,
                "index_pct": logical_index_bytes / base_bytes * 100,
                "index_pct_disk": disk_index_bytes / compressed_base_bytes * 100,
                "_arr": idx_arr,
                "_cond": idx_cond,
            }
        )
    return rows


def measure_warm_queries(rows: list[dict], repeats: int) -> None:
    if repeats <= 0:
        return
    for result in rows:
        arr = result["_arr"]
        cond = result["_cond"]
        with _with_full_query_mode(result["full_query_mode"]):
            index_runs = [benchmark_index_once(arr, cond)[0] for _ in range(repeats)]
        warm_ms = statistics.median(index_runs) * 1_000 if index_runs else None
        result["warm_ms"] = warm_ms
        result["warm_speedup"] = None if warm_ms is None else result["scan_ms"] / warm_ms


def parse_human_size(value: str) -> int:
    value = value.strip()
    if not value:
        raise argparse.ArgumentTypeError("size must not be empty")

    suffixes = {"k": 1_000, "m": 1_000_000, "g": 1_000_000_000}
    suffix = value[-1].lower()
    if suffix in suffixes:
        number = value[:-1]
        if not number:
            raise argparse.ArgumentTypeError(f"invalid size {value!r}")
        try:
            parsed = int(number)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid size {value!r}") from exc
        size = parsed * suffixes[suffix]
    else:
        try:
            size = int(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid size {value!r}") from exc

    if size <= 0:
        raise argparse.ArgumentTypeError("size must be a positive integer")
    return size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark python-blosc2 index kinds.")
    parser.add_argument(
        "--size",
        type=parse_human_size,
        help="Benchmark a single array size. Supports suffixes like 1k, 1K, 1M, 1G.",
    )
    parser.add_argument(
        "--query-width",
        type=parse_human_size,
        default=1_000,
        help="Width of the range predicate. Supports suffixes like 1k, 1K, 1M, 1G. Default: 1000.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help="Number of repeated warm-query measurements after the first cold query. Default: 3.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        help="Directory where benchmark arrays and index sidecars should be written and kept.",
    )
    parser.add_argument(
        "--optlevel",
        type=int,
        default=DEFAULT_OPLEVEL,
        help="Index optlevel to use when creating indexes. Default: 5.",
    )
    parser.add_argument(
        "--dtype",
        default="float64",
        help="NumPy dtype for the indexed field. Examples: float64, float32, int16, bool. Default: float64.",
    )
    parser.add_argument(
        "--dist",
        choices=(*DISTS, "all"),
        default="sorted",
        help="Distribution for the indexed field. Use 'all' to benchmark every distribution.",
    )
    parser.add_argument(
        "--in-mem",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use the in-memory index builders. Disabled by default; pass --in-mem to force them.",
    )
    parser.add_argument(
        "--full-query-mode",
        choices=FULL_QUERY_MODES,
        default="auto",
        help="How full exact queries should run during the benchmark: auto, selective-ooc, or whole-load.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repeats < 0:
        raise SystemExit("--repeats must be >= 0")
    try:
        id_dtype = np.dtype(args.dtype)
    except TypeError as exc:
        raise SystemExit(f"unsupported dtype {args.dtype!r}") from exc
    if id_dtype.kind not in {"b", "i", "u", "f"}:
        raise SystemExit(f"--dtype only supports bool, integer, and floating-point dtypes; got {id_dtype}")
    sizes = (args.size,) if args.size is not None else SIZES
    dists = DISTS if args.dist == "all" else (args.dist,)

    if args.outdir is None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_benchmarks(
                sizes,
                dists,
                Path(tmpdir),
                args.dist,
                args.query_width,
                args.repeats,
                args.optlevel,
                id_dtype,
                args.in_mem,
                args.full_query_mode,
            )
    else:
        args.outdir.mkdir(parents=True, exist_ok=True)
        run_benchmarks(
            sizes,
            dists,
            args.outdir,
            args.dist,
            args.query_width,
            args.repeats,
            args.optlevel,
            id_dtype,
            args.in_mem,
            args.full_query_mode,
        )


def run_benchmarks(
    sizes: tuple[int, ...],
    dists: tuple[str, ...],
    size_dir: Path,
    dist_label: str,
    query_width: int,
    repeats: int,
    optlevel: int,
    id_dtype: np.dtype,
    in_mem: bool,
    full_query_mode: str,
) -> None:
    all_results = []
    print("Structured range-query benchmark across index kinds")
    print(
        f"chunks={CHUNK_LEN:,}, blocks={BLOCK_LEN:,}, repeats={repeats}, dist={dist_label}, "
        f"query_width={query_width:,}, optlevel={optlevel}, dtype={id_dtype.name}, in_mem={in_mem}, "
        f"full_query_mode={full_query_mode}"
    )
    for dist in dists:
        for size in sizes:
            size_results = benchmark_size(
                size, size_dir, dist, query_width, optlevel, id_dtype, in_mem, full_query_mode
            )
            all_results.extend(size_results)

    print()
    print("Cold Query Table")
    print_table(
        all_results,
        [
            ("rows", lambda result: f"{result['size']:,}"),
            ("dist", lambda result: result["dist"]),
            ("builder", lambda result: "mem" if result["in_mem"] else "ooc"),
            ("kind", lambda result: result["kind"]),
            ("create_idx_ms", lambda result: f"{result['create_idx_ms']:.3f}"),
            ("scan_ms", lambda result: f"{result['scan_ms']:.3f}"),
            ("cold_ms", lambda result: f"{result['cold_ms']:.3f}"),
            ("speedup", lambda result: f"{result['cold_speedup']:.2f}x"),
            ("logical_bytes", lambda result: f"{result['logical_index_bytes']:,}"),
            ("disk_bytes", lambda result: f"{result['disk_index_bytes']:,}"),
            ("index_pct", lambda result: f"{result['index_pct']:.4f}%"),
            ("index_pct_disk", lambda result: f"{result['index_pct_disk']:.4f}%"),
        ],
    )
    if repeats > 0:
        measure_warm_queries(all_results, repeats)
        print()
        print("Warm Query Table")
        print_table(
            all_results,
            [
                ("rows", lambda result: f"{result['size']:,}"),
                ("dist", lambda result: result["dist"]),
                ("builder", lambda result: "mem" if result["in_mem"] else "ooc"),
                ("kind", lambda result: result["kind"]),
                ("create_idx_ms", lambda result: f"{result['create_idx_ms']:.3f}"),
                ("scan_ms", lambda result: f"{result['scan_ms']:.3f}"),
                ("warm_ms", lambda result: f"{result['warm_ms']:.3f}" if result["warm_ms"] is not None else "-"),
                (
                    "speedup",
                    lambda result: f"{result['warm_speedup']:.2f}x"
                    if result["warm_speedup"] is not None
                    else "-",
                ),
                ("logical_bytes", lambda result: f"{result['logical_index_bytes']:,}"),
                ("disk_bytes", lambda result: f"{result['disk_index_bytes']:,}"),
                ("index_pct", lambda result: f"{result['index_pct']:.4f}%"),
                ("index_pct_disk", lambda result: f"{result['index_pct_disk']:.4f}%"),
            ],
        )


def _format_row(cells: list[str], widths: list[int]) -> str:
    return "  ".join(cell.ljust(width) for cell, width in zip(cells, widths, strict=True))


def _table_rows(results: list[dict], columns: list[tuple[str, callable]]) -> tuple[list[str], list[list[str]], list[int]]:
    headers = [header for header, _ in columns]
    widths = [len(header) for header in headers]
    rows = [[formatter(result) for _, formatter in columns] for result in results]
    for row in rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row, strict=True)]
    return headers, rows, widths


def print_table(results: list[dict], columns: list[tuple[str, callable]]) -> None:
    headers, rows, widths = _table_rows(results, columns)
    print(_format_row(headers, widths))
    print(_format_row(["-" * width for width in widths], widths))
    for row in rows:
        print(_format_row(row, widths))


if __name__ == "__main__":
    main()
