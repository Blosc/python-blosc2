#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""Convert a Blosc2 .b2nd file (NDArray) into a Zarr array."""

from __future__ import annotations

import argparse
import contextlib
import itertools
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

import blosc2

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


def require_zarr() -> Any:
    """Ensure the zarr package is installed."""
    try:
        import zarr

        return zarr
    except ImportError as exc:
        raise ImportError("b2nd-to-zarr requires zarr; install it with: pip install 'blosc2[zarr]'") from exc


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


def resolve_shuffle(arr: blosc2.NDArray, shuffle_mode: str) -> str:
    """Resolve shuffle filter mode from blosc2 cparams or explicit argument."""
    if shuffle_mode != "auto":
        return shuffle_mode
    cparams = getattr(arr, "cparams", None)
    if cparams and hasattr(cparams, "filters"):
        if blosc2.Filter.BITSHUFFLE in cparams.filters:
            return "bitshuffle"
        if blosc2.Filter.SHUFFLE in cparams.filters:
            return "shuffle"
    return "noshuffle"


def resolve_cname(arr: blosc2.NDArray, codec_name: str) -> str:
    """Resolve compression algorithm name from blosc2 cparams or explicit argument."""
    valid_cnames = ("lz4", "lz4hc", "blosclz", "snappy", "zlib", "zstd")
    if codec_name in valid_cnames:
        return codec_name
    if codec_name == "blosc":
        return "zstd"
    if codec_name == "auto":
        cparams = getattr(arr, "cparams", None)
        if cparams and hasattr(cparams, "codec"):
            candidate = cparams.codec.name.lower()
            if candidate in valid_cnames:
                return candidate
        return "zstd"
    raise ValueError(
        f"Unsupported codec '{codec_name}'. Supported: auto, blosc, zstd, lz4, lz4hc, blosclz, zlib, gzip, none"
    )


def determine_zarr_codec(
    arr: blosc2.NDArray,
    codec_name: str = "auto",
    clevel: int | None = None,
    shuffle_mode: str = "auto",
    blocksize: int | None = None,
    zarr_format: int = 3,
) -> Any:
    """Determine the Zarr compressor codec to use based on Blosc2 cparams and user options."""
    if codec_name in ("none", "uncompressed"):
        return None

    cparams = getattr(arr, "cparams", None)
    actual_clevel = clevel if clevel is not None else getattr(cparams, "clevel", 5)

    if codec_name == "gzip":
        if zarr_format == 2:
            from numcodecs import GZip

            return GZip(level=actual_clevel)
        from zarr.codecs import GzipCodec

        return GzipCodec(level=actual_clevel)

    actual_blocksize = 0
    if blocksize is not None:
        actual_blocksize = blocksize
    elif cparams and getattr(cparams, "blocksize", 0) > 0:
        actual_blocksize = cparams.blocksize

    cname = resolve_cname(arr, codec_name)
    if zarr_format == 2:
        # Zarr v2 stores a numcodecs compressor; zarr v3 codecs are rejected there.
        from numcodecs import Blosc as NumcodecsBlosc

        shuffle = {
            "noshuffle": NumcodecsBlosc.NOSHUFFLE,
            "shuffle": NumcodecsBlosc.SHUFFLE,
            "bitshuffle": NumcodecsBlosc.BITSHUFFLE,
        }[resolve_shuffle(arr, shuffle_mode)]
        return NumcodecsBlosc(
            cname=cname,
            clevel=actual_clevel,
            shuffle=shuffle,
            blocksize=actual_blocksize,
            typesize=arr.dtype.itemsize,
        )

    from zarr.codecs import BloscCodec

    return BloscCodec(
        cname=cname,
        clevel=actual_clevel,
        shuffle=resolve_shuffle(arr, shuffle_mode),
        typesize=arr.dtype.itemsize,
        blocksize=actual_blocksize,
    )


def copy_metadata(arr: blosc2.NDArray, z_arr: Any) -> None:
    """Copy non-internal metadata and vlmeta from Blosc2 NDArray to Zarr attrs."""
    schunk = getattr(arr, "schunk", None)
    if schunk is None:
        return

    user_meta: dict[str, Any] = {}
    meta = getattr(schunk, "meta", None)
    if meta is not None:
        with contextlib.suppress(Exception):
            for k in meta:
                if k != "b2nd":
                    with contextlib.suppress(Exception):
                        user_meta[k] = meta[k]

    vlmeta = getattr(schunk, "vlmeta", None)
    if vlmeta is not None:
        with contextlib.suppress(Exception):
            for k in vlmeta:
                with contextlib.suppress(Exception):
                    user_meta[k] = vlmeta[k]

    for k, v in user_meta.items():
        with contextlib.suppress(Exception):
            z_arr.attrs[k] = v


def copy_batches(
    arr: blosc2.NDArray,
    z_arr: Any,
    batch_slices: list[tuple[slice, ...]],
    total_bytes: int,
    quiet: bool = False,
    verbose: bool = False,
) -> float:
    """Copy data slices from Blosc2 NDArray to Zarr array with progress tracking."""
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
            task = progress.add_task("Converting b2nd -> zarr...", total=total_bytes)
            for s in batch_slices:
                data = arr[s]
                z_arr[s] = data
                batch_size = data.nbytes
                progress.update(task, advance=batch_size)
                del data
    else:
        for idx, s in enumerate(batch_slices):
            data = arr[s]
            z_arr[s] = data
            del data
            if verbose and not quiet:
                pct = (idx + 1) / total_batches * 100
                print(f"Batch {idx + 1}/{total_batches} ({pct:.1f}%) written")

    return time.perf_counter() - t_start


def _fast_sample_verify(
    b2_arr: blosc2.NDArray, z_arr: Any, slices_list: list[tuple[slice, ...]], quiet: bool
) -> None:
    corner_indices = [
        tuple(0 for _ in b2_arr.shape),
        tuple(-1 for _ in b2_arr.shape),
        tuple(s // 2 for s in b2_arr.shape),
    ]
    for idx in corner_indices:
        v_b2 = b2_arr[idx]
        v_z = z_arr[idx]
        if not np.array_equal(v_b2, v_z, equal_nan=True):
            raise ValueError(f"Verification failed at index {idx}: Blosc2={v_b2} != Zarr={v_z}")

    if slices_list:
        for s in (slices_list[0], slices_list[-1]):
            b2_chunk = b2_arr[s]
            z_chunk = z_arr[s]
            if not np.array_equal(b2_chunk, z_chunk, equal_nan=True):
                raise ValueError(f"Verification failed for slice {s}")
    if not quiet:
        print("✓ Fast sample verification passed!")


def _full_verify(
    b2_arr: blosc2.NDArray, z_arr: Any, slices_list: list[tuple[slice, ...]], quiet: bool
) -> None:
    if not quiet:
        print("Performing full verification across all slices...")
    for idx, s in enumerate(slices_list):
        b2_data = b2_arr[s]
        z_data = z_arr[s]
        if not np.array_equal(b2_data, z_data, equal_nan=True):
            raise ValueError(f"Full verification failed at slice {s} (batch {idx})")
    if not quiet:
        print("✓ Full element verification passed!")


def verify_arrays(
    b2_arr: blosc2.NDArray,
    z_arr: Any,
    slices_list: list[tuple[slice, ...]],
    full: bool = False,
    quiet: bool = False,
) -> bool:
    """Verify that data in Zarr array matches source Blosc2 NDArray."""
    if b2_arr.shape != z_arr.shape:
        raise ValueError(f"Shape mismatch: Blosc2 {b2_arr.shape} vs Zarr {z_arr.shape}")
    if b2_arr.dtype != z_arr.dtype:
        raise ValueError(f"Dtype mismatch: Blosc2 {b2_arr.dtype} vs Zarr {z_arr.dtype}")

    if math.prod(b2_arr.shape) == 0:
        if not quiet:
            print("✓ Empty array verification passed!")
        return True

    if not full:
        _fast_sample_verify(b2_arr, z_arr, slices_list, quiet)
    else:
        _full_verify(b2_arr, z_arr, slices_list, quiet)
    return True


def _determine_target_chunks_and_shards(
    arr: blosc2.NDArray,
    chunks: tuple[int, ...] | None,
    shards: tuple[int, ...] | None,
    sharded: bool,
) -> tuple[tuple[int, ...], tuple[int, ...] | None]:
    target_chunks = chunks if chunks is not None else arr.chunks
    target_shards = shards

    if sharded and target_shards is None:
        blocks = getattr(arr, "blocks", None)
        if blocks is not None and all(c % b == 0 for c, b in zip(arr.chunks, blocks, strict=True)):
            target_shards = arr.chunks
            target_chunks = blocks
        else:
            target_shards = arr.chunks

    target_chunks = tuple(max(1, c) for c in target_chunks)
    if target_shards is not None:
        target_shards = tuple(max(c, max(1, s)) for c, s in zip(target_chunks, target_shards, strict=True))
    return target_chunks, target_shards


def _compute_batches(
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


def b2nd_to_zarr(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    chunks: tuple[int, ...] | None = None,
    shards: tuple[int, ...] | None = None,
    sharded: bool = False,
    codec: str = "auto",
    clevel: int | None = None,
    shuffle: str = "auto",
    blocksize: int | None = None,
    zarr_format: int = 3,
    buffer_size_mb: int = DEFAULT_BUFFER_SIZE_MB,
    overwrite: bool = False,
    verify: bool = False,
    full_verify: bool = False,
    quiet: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    """Convert a Blosc2 .b2nd file containing an NDArray to a Zarr array."""
    zarr = require_zarr()

    src_path = Path(input_path).expanduser().resolve()
    if not src_path.exists():
        raise FileNotFoundError(f"Input file not found: {src_path}")

    dst_path = (
        src_path.with_suffix(".zarr") if output_path is None else Path(output_path).expanduser().resolve()
    )
    if src_path == dst_path or src_path.is_relative_to(dst_path) or dst_path.is_relative_to(src_path):
        raise ValueError("Destination must not overlap the source path")

    if dst_path.exists() and not overwrite:
        raise FileExistsError(
            f"Destination path already exists: {dst_path}. Use -f/--force/--overwrite to replace it."
        )

    t_start = time.perf_counter()
    arr = blosc2.open(str(src_path))
    if not isinstance(arr, blosc2.NDArray):
        raise TypeError(
            f"Expected a blosc2.NDArray object in '{src_path}', but found {type(arr).__name__}. "
            "Only NDArray objects are supported for now."
        )

    shape = arr.shape
    dtype = arr.dtype
    itemsize = dtype.itemsize
    uncompressed_bytes = math.prod(shape) * itemsize
    src_file_size = get_path_size(src_path)

    target_chunks, target_shards = _determine_target_chunks_and_shards(arr, chunks, shards, sharded)

    if zarr_format == 2 and target_shards is not None:
        raise ValueError("Sharding is only supported in Zarr format 3.")

    compressor = determine_zarr_codec(
        arr,
        codec_name=codec,
        clevel=clevel,
        shuffle_mode=shuffle,
        blocksize=blocksize,
        zarr_format=zarr_format,
    )
    copy_shape, batch_slices = _compute_batches(
        shape, target_chunks, target_shards, itemsize, buffer_size_mb
    )

    if verbose and not quiet:
        print(f"Source: {src_path} ({format_bytes(src_file_size)})")
        print(f"Destination: {dst_path}")
        print(f"Array shape: {shape}, dtype: {dtype}")
        print(f"Target chunks: {target_chunks}, shards: {target_shards}")
        print(f"Batch copy shape: {copy_shape} ({len(batch_slices)} batches)")
        print(f"Compressor: {compressor}")

    if dst_path.is_dir():
        shutil.rmtree(dst_path)
    else:
        dst_path.unlink(missing_ok=True)

    is_zip = dst_path.name.endswith(".zip")
    zip_store = None
    if is_zip:
        from zarr.storage import ZipStore

        zip_store = ZipStore(str(dst_path), mode="w")
        target_store = zip_store
    else:
        target_store = str(dst_path)

    z_arr = zarr.create_array(
        store=target_store,
        shape=shape,
        chunks=target_chunks,
        shards=target_shards,
        dtype=dtype,
        compressors=compressor,
        zarr_format=zarr_format,
        overwrite=overwrite,
    )

    copy_metadata(arr, z_arr)
    copy_duration = copy_batches(arr, z_arr, batch_slices, uncompressed_bytes, quiet=quiet, verbose=verbose)
    if zip_store is not None:
        zip_store.close()

    throughput_mb_s = (uncompressed_bytes / (1024 * 1024)) / copy_duration if copy_duration > 0 else 0.0

    if verify or full_verify:
        if is_zip:
            from zarr.storage import ZipStore

            read_store = ZipStore(str(dst_path), mode="r")
            z_arr_read = zarr.open(store=read_store)
            verify_arrays(arr, z_arr_read, batch_slices, full=full_verify, quiet=quiet)
            read_store.close()
        else:
            verify_arrays(arr, z_arr, batch_slices, full=full_verify, quiet=quiet)

    dst_size = get_path_size(dst_path)
    total_duration = time.perf_counter() - t_start

    return {
        "src_path": src_path,
        "dst_path": dst_path,
        "shape": shape,
        "dtype": dtype,
        "chunks": target_chunks,
        "shards": target_shards,
        "codec": compressor,
        "src_size_bytes": src_file_size,
        "dst_size_bytes": dst_size,
        "uncompressed_bytes": uncompressed_bytes,
        "copy_duration_sec": copy_duration,
        "total_duration_sec": total_duration,
        "throughput_mb_s": throughput_mb_s,
        "src_cratio": uncompressed_bytes / src_file_size if src_file_size > 0 else 1.0,
        "dst_cratio": uncompressed_bytes / dst_size if dst_size > 0 else 1.0,
    }


# Alias for symmetry with blosc2-to-zarr
blosc2_to_zarr = b2nd_to_zarr


def build_parser(prog: str = "b2nd-to-zarr") -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    return argparse.ArgumentParser(
        prog=prog,
        description="Convert a Blosc2 .b2nd file (NDArray) into a Zarr array.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""\
Examples:
  {prog} cube.b2nd
  {prog} cube.b2nd cube.zarr --chunks 20,500,500
  {prog} cube.b2nd -f --verify --full-verify
  {prog} cube.b2nd --codec zstd --clevel 3
  {prog} cube.b2nd --sharded
""",
    )


def main(argv: list[str] | None = None) -> int:
    """Main CLI entry point."""
    prog = Path(sys.argv[0]).name if sys.argv and sys.argv[0] else "b2nd-to-zarr"
    parser = build_parser(prog=prog)

    parser.add_argument("input", help="Path to input .b2nd file")
    parser.add_argument(
        "output",
        nargs="?",
        default=None,
        help="Path to output .zarr array (default: <input>.zarr)",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        dest="output_opt",
        default=None,
        help="Optional explicit output path (overrides positional output)",
    )
    parser.add_argument(
        "-c",
        "--chunks",
        type=str,
        default=None,
        help="Zarr chunk shape, e.g. '10,1000,1000' (default: inherit from .b2nd)",
    )
    parser.add_argument(
        "--shards",
        type=str,
        default=None,
        help="Zarr shard shape (for sharded arrays in Zarr v3)",
    )
    parser.add_argument(
        "--sharded",
        action="store_true",
        help="Enable sharding automatically: map .b2nd chunks to shards, and blocks to chunks",
    )
    parser.add_argument(
        "--codec",
        type=str,
        default="auto",
        choices=["auto", "blosc", "zstd", "lz4", "lz4hc", "blosclz", "zlib", "gzip", "none"],
        help="Compression codec for Zarr (default: 'auto', matching .b2nd cparams)",
    )
    parser.add_argument(
        "--clevel",
        type=int,
        default=None,
        help="Compression level (0-9, default: inherit from .b2nd)",
    )
    parser.add_argument(
        "--shuffle",
        type=str,
        default="auto",
        choices=["auto", "noshuffle", "shuffle", "bitshuffle"],
        help="Shuffle filter (default: 'auto', matching .b2nd filters)",
    )
    parser.add_argument(
        "--blocksize",
        type=int,
        default=None,
        help="Block size in bytes for Blosc codec (default: inherit from .b2nd cparams)",
    )
    parser.add_argument(
        "--zarr-format",
        type=int,
        default=3,
        choices=[2, 3],
        help="Zarr specification format (default: 3)",
    )
    parser.add_argument(
        "-b",
        "--buffer-size",
        type=int,
        default=DEFAULT_BUFFER_SIZE_MB,
        help=f"Max memory buffer size in MB for batch copying (default: {DEFAULT_BUFFER_SIZE_MB})",
    )
    parser.add_argument(
        "-f",
        "--force",
        "--overwrite",
        dest="overwrite",
        action="store_true",
        help="Overwrite destination store if it already exists",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify array integrity (corner and sample equality check)",
    )
    parser.add_argument(
        "--full-verify",
        action="store_true",
        help="Perform thorough element-by-element verification of all chunks",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show detailed conversion information",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress all progress and informational output",
    )

    args = parser.parse_args(argv)

    out_path = args.output_opt if args.output_opt is not None else args.output
    parsed_chunks = parse_shape(args.chunks) if args.chunks else None
    parsed_shards = parse_shape(args.shards) if args.shards else None

    try:
        summary = b2nd_to_zarr(
            input_path=args.input,
            output_path=out_path,
            chunks=parsed_chunks,
            shards=parsed_shards,
            sharded=args.sharded,
            codec=args.codec,
            clevel=args.clevel,
            shuffle=args.shuffle,
            blocksize=args.blocksize,
            zarr_format=args.zarr_format,
            buffer_size_mb=args.buffer_size,
            overwrite=args.overwrite,
            verify=args.verify,
            full_verify=args.full_verify,
            quiet=args.quiet,
            verbose=args.verbose,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1

    if not args.quiet:
        if HAVE_RICH:
            console = Console()
            table = Table(title="[bold green]Conversion Summary[/bold green]", box=None)
            table.add_column("Property", style="bold cyan")
            table.add_column("Value")
            table.add_row("Source", f"{summary['src_path']} ({format_bytes(summary['src_size_bytes'])})")
            table.add_row(
                "Destination", f"{summary['dst_path']} ({format_bytes(summary['dst_size_bytes'])})"
            )
            table.add_row("Shape / Dtype", f"{summary['shape']} / {summary['dtype']}")
            table.add_row("Chunks", str(summary["chunks"]))
            if summary["shards"] is not None:
                table.add_row("Shards", str(summary["shards"]))
            table.add_row("Uncompressed Size", format_bytes(summary["uncompressed_bytes"]))
            table.add_row("Source Compression", f"{summary['src_cratio']:.2f}x")
            table.add_row("Zarr Compression", f"{summary['dst_cratio']:.2f}x")
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
            if summary["shards"] is not None:
                print(f"Shards:           {summary['shards']}")
            print(f"Uncompressed:     {format_bytes(summary['uncompressed_bytes'])}")
            print(f"Source CRatio:    {summary['src_cratio']:.2f}x")
            print(f"Zarr CRatio:      {summary['dst_cratio']:.2f}x")
            print(
                f"Conversion Speed: {summary['throughput_mb_s']:.1f} MB/s ({summary['copy_duration_sec']:.2f} s)"
            )
            print(f"Total Time:       {summary['total_duration_sec']:.2f} s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
