#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""Conversion utilities for converting between a Blosc2 .b2nd file (NDArray) and a Zarr array."""

from __future__ import annotations

import contextlib
import itertools
import math
import os
import time
from typing import TYPE_CHECKING, Any

import numpy as np

import blosc2

if TYPE_CHECKING:
    from pathlib import Path

try:
    from rich.console import Console
    from rich.progress import (
        BarColumn,
        Progress,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
        TransferSpeedColumn,
    )
    from rich.table import Table

    HAVE_RICH = True
except ImportError:
    HAVE_RICH = False

DEFAULT_BUFFER_SIZE_MB = 128

BLOSC2_CODECS = {
    "blosclz": blosc2.Codec.BLOSCLZ,
    "lz4": blosc2.Codec.LZ4,
    "lz4hc": blosc2.Codec.LZ4HC,
    "zlib": blosc2.Codec.ZLIB,
    "zstd": blosc2.Codec.ZSTD,
}

BLOSC2_SHUFFLE_FILTERS = {
    "noshuffle": blosc2.Filter.NOFILTER,
    "shuffle": blosc2.Filter.SHUFFLE,
    "bitshuffle": blosc2.Filter.BITSHUFFLE,
}


def format_bytes(size: float) -> str:
    """Format byte count into human-readable string."""
    units = ["B", "KB", "MB", "GB", "TB"]
    unit_idx = 0
    while size >= 1024.0 and unit_idx < len(units) - 1:
        size /= 1024.0
        unit_idx += 1
    return f"{size:,.2f} {units[unit_idx]}" if unit_idx > 0 else f"{int(size)} B"


def get_path_size(path: Path) -> int:
    """Get total byte size of a file or directory tree."""
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            with contextlib.suppress(OSError):
                total += os.path.getsize(fp)
    return total


def parse_shape(shape_str: str) -> tuple[int, ...]:
    """Parse comma- or whitespace-separated integer dimensions."""
    cleaned = shape_str.replace("x", ",").replace(" ", ",").replace("(", "").replace(")", "")
    dims = [int(p.strip()) for p in cleaned.split(",") if p.strip()]
    if not dims or any(d <= 0 for d in dims):
        raise ValueError(f"Invalid shape string '{shape_str}', dimensions must be positive integers.")
    return tuple(dims)


def compute_copy_shape(
    shape: tuple[int, ...],
    target_chunks: tuple[int, ...],
    itemsize: int,
    max_buffer_bytes: int = DEFAULT_BUFFER_SIZE_MB * 1024 * 1024,
) -> tuple[int, ...]:
    """Compute batch copy shape aligned with chunk boundaries to optimize memory and throughput."""
    copy_shape = list(target_chunks)
    ndim = len(shape)

    # First attempt: expand inner dimensions to full shape (from right to left)
    # to maximize contiguous memory slicing in C-order arrays.
    for d in reversed(range(ndim)):
        test_shape = list(copy_shape)
        test_shape[d] = shape[d]
        test_bytes = math.prod(test_shape) * itemsize
        if test_bytes <= max_buffer_bytes:
            copy_shape[d] = shape[d]
        else:
            current_bytes = math.prod(copy_shape) * itemsize
            if current_bytes < max_buffer_bytes:
                bytes_per_chunk = current_bytes // math.ceil(copy_shape[d] / target_chunks[d])
                max_chunks = max_buffer_bytes // bytes_per_chunk
                num_chunks = math.ceil(shape[d] / target_chunks[d])
                mult = min(max(1, max_chunks), num_chunks)
                copy_shape[d] = min(shape[d], mult * target_chunks[d])
            break

    # Second attempt: if memory headroom remains, expand leading dimensions
    for d in range(ndim):
        current_bytes = math.prod(copy_shape) * itemsize
        if current_bytes >= max_buffer_bytes:
            break
        num_chunks = math.ceil(shape[d] / target_chunks[d])
        curr_chunks = math.ceil(copy_shape[d] / target_chunks[d])
        if curr_chunks < num_chunks:
            bytes_per_chunk = current_bytes // curr_chunks
            max_chunks = max_buffer_bytes // bytes_per_chunk
            mult = min(max(1, max_chunks), num_chunks)
            if mult > curr_chunks:
                copy_shape[d] = min(shape[d], mult * target_chunks[d])

    return tuple(copy_shape)


def compute_batches(
    shape: tuple[int, ...],
    target_chunks: tuple[int, ...],
    target_shards: tuple[int, ...] | None,
    itemsize: int,
    buffer_size_mb: int,
) -> tuple[tuple[int, ...], list[tuple[slice, ...]]]:
    if math.prod(shape) == 0:
        return shape, []
    unit_shape = target_shards if target_shards is not None else target_chunks
    copy_shape = compute_copy_shape(
        shape, unit_shape, itemsize, max_buffer_bytes=buffer_size_mb * 1024 * 1024
    )
    slices_per_dim = [
        [slice(i, min(i + step, s)) for i in range(0, s, step)]
        for s, step in zip(shape, copy_shape, strict=True)
    ]
    return copy_shape, list(itertools.product(*slices_per_dim))


def require_zarr(tool_name: str) -> Any:
    """Ensure the zarr package is installed."""
    try:
        import zarr

        return zarr
    except ImportError as exc:
        raise ImportError(f"{tool_name} requires zarr; install it with: pip install 'blosc2[zarr]'") from exc


def copy_batches(
    src: Any,
    dst: Any,
    batch_slices: list[tuple[slice, ...]],
    total_bytes: int,
    copy_direction: str,
    quiet: bool = False,
    verbose: bool = False,
) -> float:
    """Copy data slices from ``src`` to ``dst`` (Blosc2 or Zarr arrays) with progress tracking."""
    total_batches = len(batch_slices)
    if total_batches == 0:
        return 0.0
    t_start = time.perf_counter()

    if HAVE_RICH and not quiet:
        console = Console()
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TransferSpeedColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(f"Converting {copy_direction}...", total=total_bytes)
            for s in batch_slices:
                data = src[s]
                dst[s] = data
                batch_size = data.nbytes
                progress.update(task, advance=batch_size)
                del data
    else:
        for idx, s in enumerate(batch_slices):
            data = src[s]
            dst[s] = data
            del data
            if verbose and not quiet:
                pct = (idx + 1) / total_batches * 100
                print(f"Batch {idx + 1}/{total_batches} ({pct:.1f}%) written")

    return time.perf_counter() - t_start


def _fast_sample_verify(src: Any, dst: Any, slices_list: list[tuple[slice, ...]], quiet: bool) -> None:
    corner_indices = [
        tuple(0 for _ in src.shape),
        tuple(-1 for _ in src.shape),
        tuple(s // 2 for s in src.shape),
    ]
    for idx in corner_indices:
        v_src = src[idx]
        v_dst = dst[idx]
        if not np.array_equal(v_src, v_dst, equal_nan=True):
            raise ValueError(f"Verification failed at index {idx}: source={v_src} != destination={v_dst}")

    if slices_list:
        for s in (slices_list[0], slices_list[-1]):
            src_chunk = src[s]
            dst_chunk = dst[s]
            if not np.array_equal(src_chunk, dst_chunk, equal_nan=True):
                raise ValueError(f"Verification failed for slice {s}")
    if not quiet:
        print("✓ Fast sample verification passed!")


def _full_verify(src: Any, dst: Any, slices_list: list[tuple[slice, ...]], quiet: bool) -> None:
    if not quiet:
        print("Performing full verification across all slices...")
    for idx, s in enumerate(slices_list):
        src_data = src[s]
        dst_data = dst[s]
        if not np.array_equal(src_data, dst_data, equal_nan=True):
            raise ValueError(f"Full verification failed at slice {s} (batch {idx})")
    if not quiet:
        print("✓ Full element verification passed!")


def verify_arrays(
    src: Any,
    dst: Any,
    slices_list: list[tuple[slice, ...]],
    full: bool = False,
    quiet: bool = False,
) -> bool:
    """Verify that data in the destination array matches the source array."""
    if src.shape != dst.shape:
        raise ValueError(f"Shape mismatch: source {src.shape} vs destination {dst.shape}")
    if src.dtype != dst.dtype:
        raise ValueError(f"Dtype mismatch: source {src.dtype} vs destination {dst.dtype}")

    if math.prod(src.shape) == 0:
        if not quiet:
            print("✓ Empty array verification passed!")
        return True

    if not full:
        _fast_sample_verify(src, dst, slices_list, quiet)
    else:
        _full_verify(src, dst, slices_list, quiet)
    return True


def print_summary(summary: dict[str, Any], dst_label: str) -> None:
    """Print the conversion summary, using a rich table when available."""
    if HAVE_RICH:
        console = Console()
        table = Table(title="[bold green]Conversion Summary[/bold green]", box=None)
        table.add_column("Property", style="bold cyan")
        table.add_column("Value")
        table.add_row("Source", f"{summary['src_path']} ({format_bytes(summary['src_size_bytes'])})")
        table.add_row("Destination", f"{summary['dst_path']} ({format_bytes(summary['dst_size_bytes'])})")
        table.add_row("Shape / Dtype", f"{summary['shape']} / {summary['dtype']}")
        table.add_row("Chunks", str(summary["chunks"]))
        if summary.get("shards") is not None:
            table.add_row("Shards", str(summary["shards"]))
        table.add_row("Uncompressed Size", format_bytes(summary["uncompressed_bytes"]))
        table.add_row("Source Compression", f"{summary['src_cratio']:.2f}x")
        table.add_row(f"{dst_label} Compression", f"{summary['dst_cratio']:.2f}x")
        table.add_row(
            "Conversion Speed",
            f"{summary['throughput_mb_s']:.1f} MB/s ({summary['copy_duration_sec']:.2f} s)",
        )
        table.add_row("Total Time", f"{summary['total_duration_sec']:.2f} s")
        console.print()
        console.print(table)
    else:
        print("\n--- Conversion Summary ---")
        print(f"Source:           {summary['src_path']} ({format_bytes(summary['src_size_bytes'])})")
        print(f"Destination:      {summary['dst_path']} ({format_bytes(summary['dst_size_bytes'])})")
        print(f"Shape / Dtype:    {summary['shape']} / {summary['dtype']}")
        print(f"Chunks:           {summary['chunks']}")
        if summary.get("shards") is not None:
            print(f"Shards:           {summary['shards']}")
        print(f"Uncompressed:     {format_bytes(summary['uncompressed_bytes'])}")
        print(f"Source CRatio:    {summary['src_cratio']:.2f}x")
        print(f"{dst_label + ' CRatio:':<18}{summary['dst_cratio']:.2f}x")
        print(
            f"Conversion Speed: {summary['throughput_mb_s']:.1f} MB/s ({summary['copy_duration_sec']:.2f} s)"
        )
        print(f"Total Time:       {summary['total_duration_sec']:.2f} s")
