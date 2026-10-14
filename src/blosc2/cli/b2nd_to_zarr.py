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
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import blosc2
from blosc2.cli.conversion_utils import (
    DEFAULT_BUFFER_SIZE_MB,
    compute_batches,
    copy_batches,
    format_bytes,
    get_path_size,
    parse_shape,
    print_summary,
    require_zarr,
    verify_arrays,
)


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
    zarr = require_zarr("b2nd-to-zarr")

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
    copy_shape, batch_slices = compute_batches(shape, target_chunks, target_shards, itemsize, buffer_size_mb)

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
    copy_duration = copy_batches(
        arr, z_arr, batch_slices, uncompressed_bytes, "b2nd -> zarr", quiet=quiet, verbose=verbose
    )
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
        print_summary(summary, dst_label="Zarr")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
