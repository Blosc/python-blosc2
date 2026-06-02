#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""
Benchmark for TreeStore hierarchical creation, opening, and listing.

Creates a hierarchy of N1 levels, each with N2 NDArray leaves and one
CTable (4 cols: bool, int, float, string) with N5 rows.  Leaf ``N``
receives an *N*-dimensional array (leaf0 is 0‑d, leaf1 is 1‑d, …) with
each side ``int(MAX_ELEMS ** (1/N))`` so that no array exceeds MAX_ELEMS
elements.  Everything is written to ``tree-store.b2z`` and the script
measures:

- Creation time (including compression)
- Opening time
- Listing time (walking all nodes and grabbing meta info)
"""

import argparse
import dataclasses
import os
import time

import numpy as np

import blosc2

OUTPUT_FILE = "tree-store.b2z"

# ── Row schema for the CTable ────────────────────────────────────────────


@dataclasses.dataclass
class _Row:
    a: bool = blosc2.field(blosc2.bool(), default=False)
    b: int = blosc2.field(blosc2.int64(), default=0)
    c: float = blosc2.field(blosc2.float64(), default=0.0)
    d: str = ""


# ── Helpers ──────────────────────────────────────────────────────────────


def _clean(path: str) -> None:
    """Remove *path* if it exists (file or directory)."""
    if os.path.exists(path):
        if os.path.isdir(path):
            import shutil

            shutil.rmtree(path)
        else:
            os.remove(path)


def _fmt_bytes(nbytes: int) -> str:
    """Human-friendly byte size."""
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} TB"


# ── Benchmark steps ──────────────────────────────────────────────────────


def _leaf_shape(ndim: int, max_elems: int) -> tuple[int, ...]:
    """Return a shape tuple for an *ndim*-dimensional array.

    For ndim == 0 the shape is ``()`` (scalar).  Otherwise each side is
    ``int(max_elems ** (1 / ndim))``, capped so the total never exceeds
    *max_elems*.
    """
    if ndim == 0:
        return ()
    side = int(max_elems ** (1.0 / ndim))
    return side, * ndim


def create_store(
    nlevels: int,
    nleaves: int,
    max_elems: int,
    nrows: int,
    no_vlmeta: bool = False,
) -> tuple[float, int]:
    """Create the TreeStore; return wall_clock, total_elements_written."""
    _clean(OUTPUT_FILE)

    # Pre-build one array per unique dimensionality (leaf ``i`` → *i*‑d).
    leaf_arrays_np: dict[int, np.ndarray] = {}
    for ndim in range(nleaves):
        shape = _leaf_shape(ndim, max_elems)
        nelem = int(np.prod(shape)) if shape else 1
        if ndim == 0:
            # linspace does not support 0‑d outputs; use a 0‑d array
            if not no_vlmeta:
                # blosc2 scalar so we can set vlmeta before storing
                leaf_arrays_np[ndim] = blosc2.asarray(np.array(0.5, dtype=np.float64))
            else:
                leaf_arrays_np[ndim] = np.array(0.5, dtype=np.float64)
        else:
            leaf_arrays_np[ndim] = blosc2.linspace(0, 1, num=nelem, shape=shape, dtype=np.float64)

    total_elements = sum(leaf_arrays_np[ndim].size for ndim in range(nleaves)) * nlevels

    # Pre-populate a single CTable that we will copy for every level.
    tmpl_table = blosc2.CTable(_Row, expected_size=nrows, validate=False)
    rows = [(i % 2 == 0, i, float(i) * 1.5, f"str_{i:06d}") for i in range(nrows)]
    tmpl_table.extend(rows, validate=False)

    print(
        f"\nCreating TreeStore with {nlevels} level(s), "
        f"{nleaves} leave(s) each, {nrows} CTable row(s) per level..."
    )
    print(f"  Max elements per leaf:  {max_elems:,}")
    for ndim in range(min(nleaves, 10)):
        shape = _leaf_shape(ndim, max_elems)
        nelem = int(np.prod(shape)) if shape else 1
        print(f"    leaf{ndim}: shape={shape}, elements={nelem:,}, uncompressed={_fmt_bytes(nelem * 8)}")
    if nleaves > 10:
        print(f"    ... ({nleaves - 10} more)")
    print(f"  CTable rows: {nrows}  |  uncompressed table size: {_fmt_bytes(tmpl_table.nbytes)}")

    t0 = time.perf_counter()
    tstore = blosc2.TreeStore(OUTPUT_FILE, mode="w")

    try:
        if not no_vlmeta:
            tstore.vlmeta["author"] = "benchmark"
            tstore.vlmeta["purpose"] = "testing"
            tstore.vlmeta["commit"] = "abc123"
        for level in range(nlevels):
            parent = f"/level{level}"
            # Store NDArray leaves – each leaf gets the array for its dimension
            for leaf in range(nleaves):
                key = f"{parent}/leaf{leaf}"
                arr = leaf_arrays_np[leaf]
                if not no_vlmeta:
                    # Add diverse vlmeta types
                    arr.vlmeta["is_even"] = leaf % 2 == 0  # bool
                    arr.vlmeta["index"] = leaf  # int
                    arr.vlmeta["value"] = float(leaf) * 0.5  # float
                    arr.vlmeta["complex"] = f"{leaf}+{leaf * 2}j"  # complex as string
                    arr.vlmeta["label"] = f"leaf_{leaf}"  # string
                    arr.vlmeta["tags"] = [f"tag_{leaf}", f"tag_{leaf + 1}"]  # list
                    arr.vlmeta["coords"] = [leaf, leaf * 2]  # list (vlmeta compatible)
                    arr.vlmeta["meta"] = {"key": f"val_{leaf}", "n": leaf}  # dict
                tstore[key] = arr

            # Store one CTable per level
            table_key = f"{parent}/ctable"
            tstore[table_key] = tmpl_table
            if not no_vlmeta:
                # Set vlmeta on the stored CTable while still in write mode
                ct = tstore[table_key]
                ct.vlmeta["description"] = f"Level {level} CTable"
                ct.vlmeta["author"] = "blosc2"
                ct.vlmeta["ncols"] = 4
                ct.vlmeta["has_index"] = True
                ct.vlmeta["tags_list"] = ["benchmark", "testing", f"level_{level}"]

            if (level + 1) % max(1, nlevels // 10) == 0 or level == nlevels - 1:
                print(f"  Level {level + 1}/{nlevels} done ({time.perf_counter() - t0:.2f}s so far)")
    finally:
        tstore.close()

    elapsed = time.perf_counter() - t0
    return elapsed, total_elements


def open_store() -> float:
    """Open the store read-only and return wall-clock time."""
    print("\nOpening store (mode='r') ...")
    t0 = time.perf_counter()
    tstore = blosc2.open(OUTPUT_FILE, mode="r")
    elapsed = time.perf_counter() - t0
    print(f"  Opened in {elapsed:.3f}s")
    tstore.close()
    return elapsed


def list_store() -> float:
    """Walk the store and grab meta info for all leaves; return elapsed time."""
    print("\nListing store (walk + meta info) ...")
    t0 = time.perf_counter()
    tstore = blosc2.open(OUTPUT_FILE, mode="r")
    try:
        n_arrays = 0
        n_tables = 0
        total_ndim_bytes = 0
        for path, children, nodes in tstore.walk("/"):
            for node_name in nodes:
                full_path = f"{path}/{node_name}".replace("//", "/")
                node = tstore[full_path]
                if hasattr(node, "shape"):
                    n_arrays += 1
                    total_ndim_bytes += node.nbytes
                elif hasattr(node, "nrows"):
                    n_tables += 1
    finally:
        tstore.close()

    elapsed = time.perf_counter() - t0
    print(
        f"  Walked {n_arrays} NDArray leaves ({_fmt_bytes(total_ndim_bytes)}) and {n_tables} CTable leaves"
    )
    print(f"  Listed in {elapsed:.3f}s")
    return elapsed


def open_and_list() -> tuple[float, float]:
    """Open and list in one go, returning (open_time, list_time)."""
    print("\nOpening + listing store ...")
    t0 = time.perf_counter()
    tstore = blosc2.open(OUTPUT_FILE, mode="r")
    t_open = time.perf_counter() - t0

    t1 = time.perf_counter()
    n_arrays = 0
    n_tables = 0
    for path, children, nodes in tstore.walk("/"):
        for node_name in nodes:
            full_path = f"{path}/{node_name}".replace("//", "/")
            node = tstore[full_path]
            if hasattr(node, "shape"):
                n_arrays += 1
            elif hasattr(node, "nrows"):
                n_tables += 1
    t_list = time.perf_counter() - t1

    tstore.close()

    print(f"  Open: {t_open:.3f}s  |  Listing: {t_list:.3f}s  ({n_arrays} array(s), {n_tables} CTable(s))")
    return t_open, t_list


# ── Main ─────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark TreeStore hierarchy creation / opening / listing",
    )
    parser.add_argument(
        "--nlevels",
        type=int,
        default=10,
        help="Number of hierarchy levels (default: %(default)s)",
    )
    parser.add_argument(
        "--nleaves",
        type=int,
        default=10,
        help="Number of NDArray leaves per level (default: %(default)s)",
    )
    parser.add_argument(
        "--max-elems",
        type=int,
        default=1_000_000,
        help="Max elements per leaf; leafN gets N-d shape with "
        "side = int(max_elems^(1/N)) (default: %(default)s)",
    )
    parser.add_argument(
        "--nrows",
        type=int,
        default=1000,
        help="Number of rows in the per-level CTable (default: %(default)s)",
    )
    parser.add_argument(
        "--no-create",
        action="store_true",
        help="Skip creation; only open/list an existing file",
    )
    parser.add_argument(
        "--no-vlmeta",
        action="store_true",
        help="Skip adding vlmeta attributes to leaves and groups",
    )
    args = parser.parse_args()

    total_elements = 0
    if not args.no_create:
        t_create, total_elements = create_store(
            args.nlevels,
            args.nleaves,
            args.max_elems,
            args.nrows,
            no_vlmeta=args.no_vlmeta,
        )
    else:
        if not os.path.exists(OUTPUT_FILE):
            parser.error(f"--no-create was passed but {OUTPUT_FILE} does not exist.")
        t_create = None

    t_open, t_list = open_and_list()

    # Summary
    total_objects = args.nlevels * (args.nleaves + 1)  # leaves + one CTable
    # If we didn't create, estimate total elements from the store itself
    if total_elements == 0:
        total_elements = args.nlevels * sum(
            int(np.prod(_leaf_shape(d, args.max_elems))) if _leaf_shape(d, args.max_elems) else 1
            for d in range(args.nleaves)
        )
    total_data_bytes = (
        total_elements * 8 + args.nlevels * args.nrows * (1 + 8 + 8 + 16)  # rough for table
    )
    file_size = os.path.getsize(OUTPUT_FILE)

    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"  Levels:              {args.nlevels}")
    print(f"  Leaves per level:    {args.nleaves}")
    print(f"  Max elems per leaf:  {args.max_elems:,}")
    print(f"  CTable rows/level:   {args.nrows}")
    print(f"  Total objects:       {total_objects}")
    print(f"  Est. uncompressed:   {_fmt_bytes(total_data_bytes)}")
    print(f"  File size on disk:   {_fmt_bytes(file_size)}")
    print(f"  Compression ratio:   {total_data_bytes / file_size:0.2f}x")
    if t_create is not None:
        print(f"\n  Creation time:       {t_create:0.3f}s")
        print(f"  Write throughput:    {total_data_bytes / t_create / 1e9:0.2f} GB/s")
    print(f"\n  Open time:           {t_open:0.3f}s")
    print(f"  List (walk) time:    {t_list:0.3f}s")
    print(f"\n  Output file:         {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
