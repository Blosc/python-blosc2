#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""
Benchmark mmap read-mode vs regular read-mode for EmbedStore, DictStore and TreeStore.

This script creates deterministic datasets, then compares:
  - mode="r", mmap_mode=None
  - mode="r", mmap_mode="r"

It supports multiple read scenarios:
  * warm_full_scan: full reads with warm OS cache.
  * warm_random_slices: random small slices with warm OS cache.
  * cold_full_scan_drop_caches: full reads after dropping Linux page cache.
  * cold_random_slices_drop_caches: random small slices after dropping Linux page cache.

For cold scenarios, the cache drop mechanism relies on Linux
(/proc/sys/vm/drop_caches) and root privileges.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

import blosc2

SCENARIOS = {
    "warm_full_scan": {
        "pattern": "full",
        "drop_caches": False,
        "description": "Read all nodes completely in-process with warm cache.",
    },
    "warm_random_slices": {
        "pattern": "random",
        "drop_caches": False,
        "description": "Read random slices from each node in-process with warm cache.",
    },
    "cold_full_scan_drop_caches": {
        "pattern": "full",
        "drop_caches": True,
        "description": "Read all nodes after dropping Linux page cache before each run.",
    },
    "cold_random_slices_drop_caches": {
        "pattern": "random",
        "drop_caches": True,
        "description": "Read random slices after dropping Linux page cache before each run.",
    },
}


@dataclass
class RunMetrics:
    open_s: float
    read_s: float
    total_s: float
    bytes_read: int
    sink: float


@dataclass
class SummaryMetrics:
    open_median_s: float
    read_median_s: float
    total_median_s: float
    total_p10_s: float
    total_p90_s: float
    throughput_mib_s: float


@dataclass
class ResultRow:
    container: str
    storage: str
    layout: str
    scenario: str
    mode: str
    open_median_s: float
    read_median_s: float
    total_median_s: float
    total_p10_s: float
    total_p90_s: float
    throughput_mib_s: float


def q(values: list[float], quantile: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), quantile))


def summarize(metrics: list[RunMetrics]) -> SummaryMetrics:
    open_vals = [m.open_s for m in metrics]
    read_vals = [m.read_s for m in metrics]
    total_vals = [m.total_s for m in metrics]
    bytes_read = metrics[0].bytes_read if metrics else 0
    total_median = float(np.median(total_vals))
    mib = bytes_read / 2**20
    throughput = mib / total_median if total_median > 0 else math.inf
    return SummaryMetrics(
        open_median_s=float(np.median(open_vals)),
        read_median_s=float(np.median(read_vals)),
        total_median_s=total_median,
        total_p10_s=q(total_vals, 0.1),
        total_p90_s=q(total_vals, 0.9),
        throughput_mib_s=throughput,
    )


def drop_linux_page_cache(drop_caches_value: int = 3) -> None:
    if platform.system() != "Linux":
        raise RuntimeError("drop_caches is supported only on Linux")
    if os.geteuid() != 0:
        raise PermissionError("drop_caches requires root privileges")
    if drop_caches_value not in (1, 2, 3):
        raise ValueError("drop_caches_value must be 1, 2, or 3")

    os.sync()
    with open("/proc/sys/vm/drop_caches", "w", encoding="ascii") as f:
        f.write(str(drop_caches_value))


def container_cls(container: str):
    if container == "embed":
        return blosc2.EmbedStore
    if container == "dict":
        return blosc2.DictStore
    if container == "tree":
        return blosc2.TreeStore
    raise ValueError(f"Unknown container: {container}")


def valid_storage_values(container: str) -> tuple[str, ...]:
    if container == "embed":
        return ("b2e",)
    return ("b2d", "b2z")


def store_path(dataset_root: Path, container: str, storage: str, layout: str) -> Path:
    base = dataset_root / f"{container}_{layout}"
    if storage == "b2e":
        return base.with_suffix(".b2e")
    if storage == "b2d":
        return base.with_suffix(".b2d")
    if storage == "b2z":
        return base.with_suffix(".b2z")
    raise ValueError(f"Unknown storage format: {storage}")


def external_data_dir(dataset_root: Path, container: str, storage: str, layout: str) -> Path:
    return dataset_root / f"external_{container}_{storage}_{layout}"


def node_key(i: int) -> str:
    return f"/group_{i % 8:02d}/node_{i:05d}"


def node_array(i: int, node_len: int, dtype: np.dtype) -> np.ndarray:
    start = i * node_len
    stop = start + node_len
    return np.arange(start, stop, dtype=dtype)


def is_external_node(i: int, layout: str) -> bool:
    if layout == "embedded":
        return False
    if layout == "external":
        return True
    if layout == "mixed":
        return (i % 2) == 1
    raise ValueError(f"Unknown layout: {layout}")


def cleanup_path(path: Path) -> None:
    blosc2.remove_urlpath(str(path))


def create_dataset(
    *,
    dataset_root: Path,
    container: str,
    storage: str,
    layout: str,
    n_nodes: int,
    node_len: int,
    dtype: np.dtype,
    clevel: int,
    codec: blosc2.Codec,
) -> Path:
    if container == "embed" and layout != "embedded":
        raise ValueError("EmbedStore supports only layout=embedded in this benchmark")

    s_path = store_path(dataset_root, container, storage, layout)
    ext_dir = external_data_dir(dataset_root, container, storage, layout)

    cleanup_path(s_path)
    cleanup_path(ext_dir)
    dataset_root.mkdir(parents=True, exist_ok=True)
    ext_dir.mkdir(parents=True, exist_ok=True)

    cparams = blosc2.CParams(clevel=clevel, codec=codec)
    dparams = blosc2.DParams(nthreads=blosc2.nthreads)

    if container == "embed":
        with blosc2.EmbedStore(urlpath=str(s_path), mode="w", cparams=cparams, dparams=dparams) as store:
            for i in range(n_nodes):
                store[node_key(i)] = node_array(i, node_len, dtype)
        return s_path

    cls = container_cls(container)
    with cls(str(s_path), mode="w", threshold=None, cparams=cparams, dparams=dparams) as store:
        for i in range(n_nodes):
            key = node_key(i)
            if is_external_node(i, layout):
                epath = ext_dir / f"node_{i:05d}.b2nd"
                arr = blosc2.asarray(
                    node_array(i, node_len, dtype),
                    urlpath=str(epath),
                    mode="w",
                    cparams=cparams,
                    dparams=dparams,
                )
                store[key] = arr
            else:
                store[key] = node_array(i, node_len, dtype)

    return s_path


def open_store(container: str, path: Path, mmap_enabled: bool):
    mmap_mode = "r" if mmap_enabled else None
    dparams = blosc2.DParams(nthreads=blosc2.nthreads)

    if container == "embed":
        return blosc2.EmbedStore(urlpath=str(path), mode="r", mmap_mode=mmap_mode, dparams=dparams)
    if container == "dict":
        return blosc2.DictStore(str(path), mode="r", mmap_mode=mmap_mode, dparams=dparams)
    if container == "tree":
        return blosc2.TreeStore(str(path), mode="r", mmap_mode=mmap_mode, dparams=dparams)
    raise ValueError(f"Unknown container: {container}")


def workload_full_scan(store: Any, keys: list[str]) -> tuple[int, float]:
    bytes_read = 0
    sink = 0.0
    for key in keys:
        arr = store[key]
        data = arr[:]
        bytes_read += int(np.asarray(data).nbytes)
        if len(data) > 0:
            sink += float(np.asarray(data).reshape(-1)[0])
    return bytes_read, sink


def workload_random_slices(
    store: Any,
    keys: list[str],
    *,
    slice_len: int,
    reads_per_node: int,
    rng: np.random.Generator,
) -> tuple[int, float]:
    bytes_read = 0
    sink = 0.0
    for key in keys:
        arr = store[key]
        n = len(arr)
        if n <= 0:
            continue
        width = min(slice_len, n)
        hi = n - width
        for _ in range(reads_per_node):
            start = int(rng.integers(0, hi + 1)) if hi > 0 else 0
            data = arr[start : start + width]
            data_np = np.asarray(data)
            bytes_read += int(data_np.nbytes)
            sink += float(data_np.reshape(-1)[0])
    return bytes_read, sink


def run_once(
    *,
    container: str,
    path: Path,
    data_keys: list[str],
    mmap_enabled: bool,
    pattern: str,
    slice_len: int,
    reads_per_node: int,
    seed: int,
) -> RunMetrics:
    t0 = time.perf_counter()
    store = open_store(container, path, mmap_enabled=mmap_enabled)
    t1 = time.perf_counter()

    try:
        # Use the known data-node keys generated during dataset creation.
        # This avoids structural subtree keys in TreeStore (e.g. "/group_xx"),
        # which are valid keys but do not represent leaf array payloads.
        keys = data_keys
        if pattern == "full":
            bytes_read, sink = workload_full_scan(store, keys)
        elif pattern == "random":
            rng = np.random.default_rng(seed)
            bytes_read, sink = workload_random_slices(
                store,
                keys,
                slice_len=slice_len,
                reads_per_node=reads_per_node,
                rng=rng,
            )
        else:
            raise ValueError(f"Unknown pattern: {pattern}")
    finally:
        # DictStore/TreeStore expose close(); EmbedStore currently does not.
        # Use close() when available and otherwise just release the reference.
        close = getattr(store, "close", None)
        if callable(close):
            close()
        del store

    t2 = time.perf_counter()
    return RunMetrics(open_s=t1 - t0, read_s=t2 - t1, total_s=t2 - t0, bytes_read=bytes_read, sink=sink)


def run_mode(
    *,
    container: str,
    path: Path,
    data_keys: list[str],
    mmap_enabled: bool,
    pattern: str,
    drop_caches: bool,
    runs: int,
    slice_len: int,
    reads_per_node: int,
    seed: int,
    drop_caches_value: int,
) -> list[RunMetrics]:
    metrics: list[RunMetrics] = []
    for i in range(runs):
        if drop_caches:
            drop_linux_page_cache(drop_caches_value=drop_caches_value)
        metrics.append(
            run_once(
                container=container,
                path=path,
                data_keys=data_keys,
                mmap_enabled=mmap_enabled,
                pattern=pattern,
                slice_len=slice_len,
                reads_per_node=reads_per_node,
                seed=seed + i,
            )
        )
    return metrics


def print_scenario_header(name: str) -> None:
    meta = SCENARIOS[name]
    print(f"  Scenario: {name}")
    print(f"    Description: {meta['description']}")


def print_compare(a: SummaryMetrics, b: SummaryMetrics) -> None:
    speedup = a.total_median_s / b.total_median_s if b.total_median_s > 0 else math.inf
    print(
        "    regular median={:.4f}s ({:.1f} MiB/s) | mmap median={:.4f}s ({:.1f} MiB/s) | speedup={:.3f}x".format(
            a.total_median_s,
            a.throughput_mib_s,
            b.total_median_s,
            b.throughput_mib_s,
            speedup,
        )
    )


def parse_combinations(
    containers: list[str],
    storages: list[str],
    layouts: list[str],
) -> list[tuple[str, str, str]]:
    combos: list[tuple[str, str, str]] = []
    for container in containers:
        valid_storages = valid_storage_values(container)
        for storage in storages:
            if storage not in valid_storages:
                continue
            for layout in layouts:
                if container == "embed" and layout != "embedded":
                    continue
                combos.append((container, storage, layout))
    return combos


def validate_args(args: argparse.Namespace) -> None:
    if args.drop_caches_value not in (1, 2, 3):
        raise ValueError("--drop-caches-value must be 1, 2, or 3")

    cold_selected = any(SCENARIOS[s]["drop_caches"] for s in args.scenarios)
    if cold_selected:
        if platform.system() != "Linux":
            raise RuntimeError("cold/drop-cache scenarios are supported only on Linux")
        if os.geteuid() != 0:
            raise PermissionError("cold/drop-cache scenarios require root")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark mmap read mode for EmbedStore/DictStore/TreeStore",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("bench_mmap_store_data"))
    parser.add_argument("--container", nargs="+", default=["embed", "dict", "tree"], choices=["embed", "dict", "tree"])
    parser.add_argument("--storage", nargs="+", default=["b2e", "b2d", "b2z"], choices=["b2e", "b2d", "b2z"])
    parser.add_argument("--layout", nargs="+", default=["embedded", "external", "mixed"], choices=["embedded", "external", "mixed"])
    parser.add_argument("--scenario", nargs="+", dest="scenarios", default=list(SCENARIOS), choices=list(SCENARIOS))

    parser.add_argument("--n-nodes", type=int, default=128)
    parser.add_argument("--node-len", type=int, default=100_000)
    parser.add_argument("--dtype", type=str, default="float64")
    parser.add_argument("--clevel", type=int, default=5)
    parser.add_argument("--codec", type=str, default="ZSTD", choices=[c.name for c in blosc2.Codec])

    parser.add_argument("--runs", type=int, default=7)
    parser.add_argument("--slice-len", type=int, default=4096)
    parser.add_argument("--reads-per-node", type=int, default=8)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--drop-caches-value", type=int, default=3)

    parser.add_argument("--json-out", type=Path, default=None, help="Optional JSON output path")
    parser.add_argument("--keep-dataset", action="store_true", help="Keep generated benchmark files")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)

    dtype = np.dtype(args.dtype)
    codec = blosc2.Codec[args.codec]

    combos = parse_combinations(args.container, args.storage, args.layout)
    if not combos:
        raise ValueError("No valid (container, storage, layout) combinations selected")

    rows: list[ResultRow] = []

    print("mmap read benchmark")
    print(f"  dataset_root={args.dataset_root}")
    print(f"  runs={args.runs}, n_nodes={args.n_nodes}, node_len={args.node_len}, dtype={dtype}")

    for container, storage, layout in combos:
        print(f"\nDataset: container={container}, storage={storage}, layout={layout}")
        data_keys = [node_key(i) for i in range(args.n_nodes)]
        s_path = create_dataset(
            dataset_root=args.dataset_root,
            container=container,
            storage=storage,
            layout=layout,
            n_nodes=args.n_nodes,
            node_len=args.node_len,
            dtype=dtype,
            clevel=args.clevel,
            codec=codec,
        )

        for scenario in args.scenarios:
            print_scenario_header(scenario)
            meta = SCENARIOS[scenario]
            regular_runs = run_mode(
                container=container,
                path=s_path,
                data_keys=data_keys,
                mmap_enabled=False,
                pattern=meta["pattern"],
                drop_caches=meta["drop_caches"],
                runs=args.runs,
                slice_len=args.slice_len,
                reads_per_node=args.reads_per_node,
                seed=args.seed,
                drop_caches_value=args.drop_caches_value,
            )
            mmap_runs = run_mode(
                container=container,
                path=s_path,
                data_keys=data_keys,
                mmap_enabled=True,
                pattern=meta["pattern"],
                drop_caches=meta["drop_caches"],
                runs=args.runs,
                slice_len=args.slice_len,
                reads_per_node=args.reads_per_node,
                seed=args.seed,
                drop_caches_value=args.drop_caches_value,
            )

            regular_summary = summarize(regular_runs)
            mmap_summary = summarize(mmap_runs)
            print_compare(regular_summary, mmap_summary)

            rows.append(
                ResultRow(
                    container=container,
                    storage=storage,
                    layout=layout,
                    scenario=scenario,
                    mode="regular",
                    open_median_s=regular_summary.open_median_s,
                    read_median_s=regular_summary.read_median_s,
                    total_median_s=regular_summary.total_median_s,
                    total_p10_s=regular_summary.total_p10_s,
                    total_p90_s=regular_summary.total_p90_s,
                    throughput_mib_s=regular_summary.throughput_mib_s,
                )
            )
            rows.append(
                ResultRow(
                    container=container,
                    storage=storage,
                    layout=layout,
                    scenario=scenario,
                    mode="mmap",
                    open_median_s=mmap_summary.open_median_s,
                    read_median_s=mmap_summary.read_median_s,
                    total_median_s=mmap_summary.total_median_s,
                    total_p10_s=mmap_summary.total_p10_s,
                    total_p90_s=mmap_summary.total_p90_s,
                    throughput_mib_s=mmap_summary.throughput_mib_s,
                )
            )

        if not args.keep_dataset:
            cleanup_path(s_path)
            cleanup_path(external_data_dir(args.dataset_root, container, storage, layout))

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": {
                "dataset_root": str(args.dataset_root),
                "container": args.container,
                "storage": args.storage,
                "layout": args.layout,
                "scenarios": args.scenarios,
                "n_nodes": args.n_nodes,
                "node_len": args.node_len,
                "dtype": str(dtype),
                "clevel": args.clevel,
                "codec": args.codec,
                "runs": args.runs,
                "slice_len": args.slice_len,
                "reads_per_node": args.reads_per_node,
                "seed": args.seed,
                "drop_caches_value": args.drop_caches_value,
            },
            "results": [asdict(r) for r in rows],
        }
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved JSON results to: {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
