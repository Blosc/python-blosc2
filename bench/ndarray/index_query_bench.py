#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

from __future__ import annotations

import argparse
import math
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
DEFAULT_REPEATS = 3
KINDS = ("ultralight", "light", "medium", "full")
DEFAULT_KIND = "light"
DISTS = ("sorted", "block-shuffled", "permuted", "random")
RNG_SEED = 0
DEFAULT_OPLEVEL = 5
FULL_QUERY_MODES = ("auto", "selective-ooc", "whole-load")
DATASET_LAYOUT_VERSION = "payload-ramp-v1"

COLD_COLUMNS = [
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
]

WARM_COLUMNS = [
    ("rows", lambda result: f"{result['size']:,}"),
    ("dist", lambda result: result["dist"]),
    ("builder", lambda result: "mem" if result["in_mem"] else "ooc"),
    ("kind", lambda result: result["kind"]),
    ("create_idx_ms", lambda result: f"{result['create_idx_ms']:.3f}"),
    ("scan_ms", lambda result: f"{result['scan_ms']:.3f}"),
    ("warm_ms", lambda result: f"{result['warm_ms']:.3f}" if result["warm_ms"] is not None else "-"),
    (
        "speedup",
        lambda result: f"{result['warm_speedup']:.2f}x" if result["warm_speedup"] is not None else "-",
    ),
    ("logical_bytes", lambda result: f"{result['logical_index_bytes']:,}"),
    ("disk_bytes", lambda result: f"{result['disk_index_bytes']:,}"),
    ("index_pct", lambda result: f"{result['index_pct']:.4f}%"),
    ("index_pct_disk", lambda result: f"{result['index_pct_disk']:.4f}%"),
]


def dtype_token(dtype: np.dtype) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", np.dtype(dtype).name).strip("_")


def source_dtype(id_dtype: np.dtype) -> np.dtype:
    return np.dtype([("id", np.dtype(id_dtype)), ("payload", np.float32)])


def payload_slice(start: int, stop: int) -> np.ndarray:
    """Deterministic nontrivial payload values for structured benchmark rows."""
    return np.arange(start, stop, dtype=np.float32)


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


def ordered_id_slice(size: int, start: int, stop: int, dtype: np.dtype) -> np.ndarray:
    dtype = np.dtype(dtype)
    if stop <= start:
        return np.empty(0, dtype=dtype)

    if dtype == np.dtype(np.bool_):
        values = np.zeros(stop - start, dtype=dtype)
        true_start = max(start, size // 2)
        if true_start < stop:
            values[true_start - start :] = True
        return values

    positions = np.arange(start, stop, dtype=np.int64)
    if dtype.kind in {"i", "u"}:
        info = np.iinfo(dtype)
        unique_count = min(size, int(info.max) - int(info.min) + 1)
        base = int(info.min) if unique_count < size and dtype.kind == "i" else 0
        if dtype.kind == "i" and unique_count < size:
            base = max(int(info.min), -(unique_count // 2))
        values = base + (positions * unique_count) // size
        return values.astype(dtype, copy=False)

    if dtype.kind == "f":
        span = max(1, size)
        values = positions.astype(np.float64, copy=False) - (span / 2)
        return values.astype(dtype, copy=False)

    raise ValueError(f"unsupported dtype for benchmark: {dtype}")


def ordered_id_at(size: int, index: int, dtype: np.dtype) -> object:
    return ordered_id_slice(size, index, index + 1, dtype)[0].item()


def ordered_ids_from_positions(positions: np.ndarray, size: int, dtype: np.dtype) -> np.ndarray:
    dtype = np.dtype(dtype)
    if positions.size == 0:
        return np.empty(0, dtype=dtype)

    if dtype == np.dtype(np.bool_):
        return (positions >= (size // 2)).astype(dtype, copy=False)

    if dtype.kind in {"i", "u"}:
        info = np.iinfo(dtype)
        unique_count = min(size, int(info.max) - int(info.min) + 1)
        base = int(info.min) if unique_count < size and dtype.kind == "i" else 0
        if dtype.kind == "i" and unique_count < size:
            base = max(int(info.min), -(unique_count // 2))
        values = base + (positions * unique_count) // size
        return values.astype(dtype, copy=False)

    if dtype.kind == "f":
        span = max(1, size)
        values = positions.astype(np.float64, copy=False) - (span / 2)
        return values.astype(dtype, copy=False)

    raise ValueError(f"unsupported dtype for benchmark: {dtype}")


def fill_ids(ids: np.ndarray, ordered_ids: np.ndarray, dist: str, rng: np.random.Generator, block_len: int) -> None:
    size = ids.shape[0]
    if dist == "sorted":
        ids[:] = ordered_ids
        return

    if dist == "block-shuffled":
        nblocks = (size + block_len - 1) // block_len
        order = rng.permutation(nblocks)
        dest = 0
        for src_block in order:
            src_start = int(src_block) * block_len
            src_stop = min(src_start + block_len, size)
            block_size = src_stop - src_start
            ids[dest : dest + block_size] = ordered_ids[src_start:src_stop]
            dest += block_size
        return

    if dist == "random":
        ids[:] = ordered_ids
        rng.shuffle(ids)
        return

    raise ValueError(f"unsupported distribution {dist!r}")


def _geometry_value_token(value: int | None) -> str:
    return "auto" if value is None else f"{value}"


def geometry_token(chunks: int | None, blocks: int | None) -> str:
    return f"chunks-{_geometry_value_token(chunks)}.blocks-{_geometry_value_token(blocks)}"


def format_geometry_value(value: int | None) -> str:
    return "auto" if value is None else f"{value:,}"


def resolve_geometry(shape: tuple[int, ...], dtype: np.dtype, chunks: int | None, blocks: int | None) -> tuple[int, int]:
    chunk_spec = None if chunks is None else (chunks,)
    block_spec = None if blocks is None else (blocks,)
    resolved_chunks, resolved_blocks = blosc2.compute_chunks_blocks(shape, chunk_spec, block_spec, dtype=dtype)
    return int(resolved_chunks[0]), int(resolved_blocks[0])


def _block_order(size: int, block_len: int) -> np.ndarray:
    nblocks = (size + block_len - 1) // block_len
    return np.random.default_rng(RNG_SEED).permutation(nblocks)


def _fill_block_shuffled_ids(
    ids: np.ndarray, size: int, start: int, stop: int, block_len: int, order: np.ndarray
) -> None:
    cursor = start
    out_cursor = 0
    while cursor < stop:
        dest_block = cursor // block_len
        block_offset = cursor % block_len
        src_block = int(order[dest_block])
        src_start = src_block * block_len + block_offset
        take = min(stop - cursor, block_len - block_offset, size - src_start)
        ids[out_cursor : out_cursor + take] = ordered_id_slice(size, src_start, src_start + take, ids.dtype)
        cursor += take
        out_cursor += take


def _permuted_position_params(size: int) -> tuple[int, int]:
    if size <= 1:
        return 1, 0
    rng = np.random.default_rng(RNG_SEED)
    step = int(rng.integers(1, size))
    while math.gcd(step, size) != 1:
        step += 1
        if step >= size:
            step = 1
    offset = int(rng.integers(0, size))
    return step, offset


def _fill_permuted_ids(ids: np.ndarray, size: int, start: int, stop: int, step: int, offset: int) -> None:
    positions = np.arange(start, stop, dtype=np.int64)
    shuffled_positions = (positions * step + offset) % size
    ids[:] = ordered_ids_from_positions(shuffled_positions, size, ids.dtype)


def _randomized_ids(size: int, dtype: np.dtype) -> np.ndarray:
    ids = make_ordered_ids(size, dtype)
    np.random.default_rng(RNG_SEED).shuffle(ids)
    return ids


def build_persistent_array(
    size: int, dist: str, id_dtype: np.dtype, path: Path, chunks: int | None, blocks: int | None
) -> blosc2.NDArray:
    dtype = source_dtype(id_dtype)
    kwargs = {"urlpath": path, "mode": "w"}
    if chunks is not None:
        kwargs["chunks"] = (chunks,)
    if blocks is not None:
        kwargs["blocks"] = (blocks,)
    arr = blosc2.zeros((size,), dtype=dtype, **kwargs)
    chunk_len = int(arr.chunks[0])
    block_len = int(arr.blocks[0])
    block_order = _block_order(size, block_len) if dist == "block-shuffled" else None
    permuted_step, permuted_offset = _permuted_position_params(size) if dist == "permuted" else (1, 0)
    random_ids = _randomized_ids(size, id_dtype) if dist == "random" else None
    for start in range(0, size, chunk_len):
        stop = min(start + chunk_len, size)
        chunk = np.zeros(stop - start, dtype=dtype)
        if dist == "sorted":
            chunk["id"] = ordered_id_slice(size, start, stop, id_dtype)
        elif dist == "block-shuffled":
            _fill_block_shuffled_ids(chunk["id"], size, start, stop, block_len, block_order)
        elif dist == "permuted":
            _fill_permuted_ids(chunk["id"], size, start, stop, permuted_step, permuted_offset)
        elif dist == "random":
            chunk["id"] = random_ids[start:stop]
        else:
            raise ValueError(f"unsupported distribution {dist!r}")
        chunk["payload"] = payload_slice(start, stop)
        arr[start:stop] = chunk
    return arr


def base_array_path(size_dir: Path, size: int, dist: str, id_dtype: np.dtype, chunks: int | None, blocks: int | None) -> Path:
    return (
        size_dir
        / f"size_{size}_{dist}_{dtype_token(id_dtype)}.{DATASET_LAYOUT_VERSION}.{geometry_token(chunks, blocks)}.b2nd"
    )


def indexed_array_path(
    size_dir: Path,
    size: int,
    dist: str,
    kind: str,
    optlevel: int,
    id_dtype: np.dtype,
    in_mem: bool,
    chunks: int | None,
    blocks: int | None,
    codec: blosc2.Codec | None,
    clevel: int | None,
    nthreads: int | None,
) -> Path:
    mode = "mem" if in_mem else "ooc"
    codec_token = "codec-auto" if codec is None else f"codec-{codec.name}"
    clevel_token = "clevel-auto" if clevel is None else f"clevel-{clevel}"
    thread_token = "threads-auto" if nthreads is None else f"threads-{nthreads}"
    return (
        size_dir
        / f"size_{size}_{dist}_{dtype_token(id_dtype)}.{DATASET_LAYOUT_VERSION}.{geometry_token(chunks, blocks)}.{codec_token}.{clevel_token}.{thread_token}"
        f".{kind}.opt{optlevel}.{mode}.b2nd"
    )


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


def _query_bounds(size: int, query_width: int, dtype: np.dtype) -> tuple[object, object]:
    if size <= 0:
        raise ValueError("benchmark arrays must not be empty")

    lo_idx = size // 2
    hi_idx = min(size - 1, lo_idx + max(query_width - 1, 0))
    return ordered_id_at(size, lo_idx, dtype), ordered_id_at(size, hi_idx, dtype)


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
        expected_ooc = descriptor.get("ooc", False) if kind == "ultralight" else (not bool(in_mem))
        if (
            descriptor.get("field") == "id"
            and descriptor.get("kind") == kind
            and int(descriptor.get("optlevel", -1)) == int(optlevel)
            and bool(descriptor.get("ooc", False)) is bool(expected_ooc)
            and not descriptor.get("stale", False)
        ):
            return descriptor
    return None


def _open_or_build_persistent_array(
    path: Path, size: int, dist: str, id_dtype: np.dtype, chunks: int | None, blocks: int | None
) -> blosc2.NDArray:
    if path.exists():
        return blosc2.open(path, mode="a")
    blosc2.remove_urlpath(path)
    return build_persistent_array(size, dist, id_dtype, path, chunks, blocks)


def _open_or_build_indexed_array(
    path: Path,
    size: int,
    dist: str,
    id_dtype: np.dtype,
    kind: str,
    optlevel: int,
    in_mem: bool,
    chunks: int | None,
    blocks: int | None,
    codec: blosc2.Codec | None,
    clevel: int | None,
    nthreads: int | None,
) -> tuple[blosc2.NDArray, float]:
    if path.exists():
        arr = blosc2.open(path, mode="a")
        if _valid_index_descriptor(arr, kind, optlevel, in_mem) is not None:
            return arr, 0.0
        if arr.indexes:
            arr.drop_index(field="id")
        blosc2.remove_urlpath(path)

    arr = build_persistent_array(size, dist, id_dtype, path, chunks, blocks)
    build_start = time.perf_counter()
    kwargs = {"field": "id", "kind": kind, "optlevel": optlevel, "in_mem": in_mem}
    cparams = {}
    if codec is not None:
        cparams["codec"] = codec
    if clevel is not None:
        cparams["clevel"] = clevel
    if nthreads is not None:
        cparams["nthreads"] = nthreads
    if cparams:
        kwargs["cparams"] = cparams
    arr.create_index(**kwargs)
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
    chunks: int | None,
    blocks: int | None,
    codec: blosc2.Codec | None,
    clevel: int | None,
    nthreads: int | None,
    kinds: tuple[str, ...],
    cold_row_callback=None,
) -> list[dict]:
    arr = _open_or_build_persistent_array(
        base_array_path(size_dir, size, dist, id_dtype, chunks, blocks), size, dist, id_dtype, chunks, blocks
    )
    lo, hi = _query_bounds(size, query_width, id_dtype)
    condition_str = _condition_expr(lo, hi, id_dtype)
    condition = blosc2.lazyexpr(condition_str, arr.fields)
    expr = condition.where(arr)
    base_bytes = size * arr.dtype.itemsize
    compressed_base_bytes = os.path.getsize(arr.urlpath)

    scan_ms = benchmark_scan_once(expr)[0] * 1_000

    rows = []
    for kind in kinds:
        idx_arr, build_time = _open_or_build_indexed_array(
            indexed_array_path(
                size_dir, size, dist, kind, optlevel, id_dtype, in_mem, chunks, blocks, codec, clevel, nthreads
            ),
            size,
            dist,
            id_dtype,
            kind,
            optlevel,
            in_mem,
            chunks,
            blocks,
            codec,
            clevel,
            nthreads,
        )
        idx_cond = blosc2.lazyexpr(condition_str, idx_arr.fields)
        idx_expr = idx_cond.where(idx_arr)
        with _with_full_query_mode(full_query_mode):
            explanation = idx_expr.explain()
            cold_time, index_len = benchmark_index_once(idx_arr, idx_cond)
        descriptor = idx_arr.indexes[0]
        logical_index_bytes, disk_index_bytes = index_sizes(descriptor)

        row = {
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
        rows.append(row)
        if cold_row_callback is not None:
            cold_row_callback(row)
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


def parse_human_size_or_auto(value: str) -> int | None:
    value = value.strip()
    if value.lower() == "auto":
        return None
    return parse_human_size(value)


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
        default=1,
        help="Width of the range predicate. Supports suffixes like 1k, 1K, 1M, 1G. Default: 1.",
    )
    parser.add_argument(
        "--chunks",
        type=parse_human_size_or_auto,
        default=None,
        help="Chunk size for the base array. Supports suffixes like 10k, 1M, and 'auto'. Default: auto.",
    )
    parser.add_argument(
        "--blocks",
        type=parse_human_size_or_auto,
        default=None,
        help="Block size for the base array. Supports suffixes like 10k, 1M, and 'auto'. Default: auto.",
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
        default="permuted",
        help="Distribution for the indexed field. Use 'all' to benchmark every distribution.",
    )
    parser.add_argument(
        "--kind",
        choices=(*KINDS, "all"),
        default=DEFAULT_KIND,
        help=f"Index kind to benchmark. Use 'all' to benchmark every kind. Default: {DEFAULT_KIND}.",
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
    parser.add_argument(
        "--codec",
        type=str,
        default=None,
        choices=[codec.name for codec in blosc2.Codec],
        help="Codec to use for index sidecars. Default: library default.",
    )
    parser.add_argument(
        "--clevel",
        type=int,
        default=None,
        help="Compression level to use for index sidecars. Default: library default.",
    )
    parser.add_argument(
        "--nthreads",
        type=int,
        default=None,
        help="Number of threads to use for index creation. Default: use blosc2.nthreads.",
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
    codec = None if args.codec is None else blosc2.Codec[args.codec]
    if args.clevel is not None and args.clevel < 0:
        raise SystemExit("--clevel must be >= 0")
    if args.nthreads is not None and args.nthreads <= 0:
        raise SystemExit("--nthreads must be a positive integer")
    sizes = (args.size,) if args.size is not None else SIZES
    dists = DISTS if args.dist == "all" else (args.dist,)
    kinds = KINDS if args.kind == "all" else (args.kind,)

    if args.outdir is None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_benchmarks(
                sizes,
                dists,
                kinds,
                Path(tmpdir),
                args.dist,
                args.query_width,
                args.repeats,
                args.optlevel,
                id_dtype,
                args.in_mem,
                args.full_query_mode,
                args.chunks,
                args.blocks,
                codec,
                args.clevel,
                args.nthreads,
            )
    else:
        args.outdir.mkdir(parents=True, exist_ok=True)
        run_benchmarks(
            sizes,
            dists,
            kinds,
            args.outdir,
            args.dist,
            args.query_width,
            args.repeats,
            args.optlevel,
            id_dtype,
            args.in_mem,
            args.full_query_mode,
            args.chunks,
            args.blocks,
            codec,
            args.clevel,
            args.nthreads,
        )


def run_benchmarks(
    sizes: tuple[int, ...],
    dists: tuple[str, ...],
    kinds: tuple[str, ...],
    size_dir: Path,
    dist_label: str,
    query_width: int,
    repeats: int,
    optlevel: int,
    id_dtype: np.dtype,
    in_mem: bool,
    full_query_mode: str,
    chunks: int | None,
    blocks: int | None,
    codec: blosc2.Codec | None,
    clevel: int | None,
    nthreads: int | None,
) -> None:
    all_results = []

    array_dtype = source_dtype(id_dtype)
    resolved_geometries = {resolve_geometry((size,), array_dtype, chunks, blocks) for size in sizes}
    if len(resolved_geometries) == 1:
        resolved_chunk_len, resolved_block_len = next(iter(resolved_geometries))
        geometry_label = f"chunks={resolved_chunk_len:,}, blocks={resolved_block_len:,}"
    else:
        geometry_label = "chunks=varies, blocks=varies"
    print("Structured range-query benchmark across index kinds")
    print(
        f"{geometry_label}, repeats={repeats}, dist={dist_label}, "
        f"query_width={query_width:,}, optlevel={optlevel}, dtype={id_dtype.name}, in_mem={in_mem}, "
        f"full_query_mode={full_query_mode}, index_codec={'auto' if codec is None else codec.name}, "
        f"index_clevel={'auto' if clevel is None else clevel}, "
        f"index_nthreads={'auto' if nthreads is None else nthreads}"
    )
    for dist in dists:
        for size in sizes:
            size_results = benchmark_size(
                size,
                size_dir,
                dist,
                query_width,
                optlevel,
                id_dtype,
                in_mem,
                full_query_mode,
                chunks,
                blocks,
                codec,
                clevel,
                nthreads,
                kinds,
            )
            all_results.extend(size_results)
    cold_widths = table_widths(all_results, COLD_COLUMNS)
    print()
    print("Cold Query Table")
    print_table(all_results, COLD_COLUMNS, cold_widths)
    if repeats > 0:
        measure_warm_queries(all_results, repeats)
        warm_widths = table_widths(all_results, WARM_COLUMNS)
        shared_width_by_header = {}
        for (header, _), width in zip(COLD_COLUMNS, cold_widths, strict=True):
            shared_width_by_header[header] = width
        for (header, _), width in zip(WARM_COLUMNS, warm_widths, strict=True):
            shared_width_by_header[header] = max(shared_width_by_header.get(header, 0), width)
        warm_widths = [shared_width_by_header[header] for header, _ in WARM_COLUMNS]
        print()
        print("Warm Query Table")
        print_table(all_results, WARM_COLUMNS, warm_widths)


def _format_row(cells: list[str], widths: list[int]) -> str:
    return "  ".join(cell.ljust(width) for cell, width in zip(cells, widths, strict=True))


def _table_rows(results: list[dict], columns: list[tuple[str, callable]]) -> tuple[list[str], list[list[str]], list[int]]:
    headers = [header for header, _ in columns]
    widths = [len(header) for header in headers]
    rows = [[formatter(result) for _, formatter in columns] for result in results]
    for row in rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row, strict=True)]
    return headers, rows, widths


def print_table(results: list[dict], columns: list[tuple[str, callable]], widths: list[int] | None = None) -> None:
    headers, rows, computed_widths = _table_rows(results, columns)
    widths = computed_widths if widths is None else widths
    print(_format_row(headers, widths))
    print(_format_row(["-" * width for width in widths], widths))
    for row in rows:
        print(_format_row(row, widths))


def print_table_header(columns: list[tuple[str, callable]], widths: list[int] | None = None) -> None:
    headers = [header for header, _ in columns]
    if widths is None:
        widths = [len(header) for header in headers]
    print(_format_row(headers, widths))
    print(_format_row(["-" * width for width in widths], widths))


def print_table_row(result: dict, columns: list[tuple[str, callable]], widths: list[int] | None = None) -> None:
    cells = [formatter(result) for _, formatter in columns]
    if widths is None:
        widths = [max(len(header), len(cell)) for (header, _), cell in zip(columns, cells, strict=True)]
    print(_format_row(cells, widths))


def progress_widths(
    columns: list[tuple[str, callable]],
    sizes: tuple[int, ...],
    dists: tuple[str, ...],
    kinds: tuple[str, ...],
    id_dtype: np.dtype,
) -> list[int]:
    max_size = max(sizes)
    max_index_bytes = max_size * max(np.dtype(id_dtype).itemsize + 8, 16)
    max_cells = {
        "rows": f"{max_size:,}",
        "dist": max(dists, key=len),
        "builder": "ooc",
        "kind": max(kinds, key=len),
        "create_idx_ms": "999999.999",
        "scan_ms": "9999.999",
        "cold_ms": "9999.999",
        "warm_ms": "9999.999",
        "speedup": "9999.99x",
        "logical_bytes": f"{max_index_bytes:,}",
        "disk_bytes": f"{max_index_bytes:,}",
        "index_pct": "100.0000%",
        "index_pct_disk": "100.0000%",
    }
    widths = []
    for header, _ in columns:
        widths.append(max(len(header), len(max_cells.get(header, ""))))
    return widths


def table_widths(results: list[dict], columns: list[tuple[str, callable]]) -> list[int]:
    _, _, widths = _table_rows(results, columns)
    return widths


if __name__ == "__main__":
    main()
