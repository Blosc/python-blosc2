#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""Convert a Zarr array into a Blosc2 .b2nd file (NDArray)."""

from __future__ import annotations

import argparse
import contextlib
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

import blosc2
from blosc2.cli.conversion_utils import (
    BLOSC2_CODECS,
    BLOSC2_SHUFFLE_FILTERS,
    DEFAULT_BUFFER_SIZE_MB,
    compute_batches,
    copy_batches,
    get_path_size,
    parse_shape,
    print_summary,
    require_zarr,
    verify_arrays,
)


def zarr_to_b2nd(
    input_path: str | Path,
    output_path=None,
    *,
    chunks: tuple[int, ...] | None = None,
    blocks: tuple[int, ...] | None = None,
    codec: str = "auto",
    clevel: int | None = None,
    shuffle: str = "auto",
    buffer_size_mb: int = DEFAULT_BUFFER_SIZE_MB,
    overwrite: bool = False,
    verify: bool = False,
    full_verify: bool = False,
    quiet: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    zarr = require_zarr("zarr-to-b2nd")
    src_path = Path(input_path).expanduser().resolve()
    if not src_path.exists():
        raise FileNotFoundError(f"Input file not found: {src_path}")

    if output_path is None:
        stem = src_path.name.removesuffix(".zip").removesuffix(".zarr")
        dst_path = src_path.with_name(f"{stem}.b2nd")
    else:
        dst_path = Path(output_path).expanduser().resolve()
    if src_path == dst_path or src_path.is_relative_to(dst_path) or dst_path.is_relative_to(src_path):
        raise ValueError("Destination must not overlap the source path")

    if dst_path.exists() and not overwrite:
        raise FileExistsError(
            f"Destination path already exists: {dst_path}. Use -f/--force/--overwrite to replace it."
        )

    t_start = time.perf_counter()
    z_arr, zip_store = open_zarr_input(zarr, src_path)
    try:
        shape = z_arr.shape
        dtype = np.dtype(z_arr.dtype)
        itemsize = dtype.itemsize
        uncompressed_bytes = math.prod(shape) * itemsize
        src_size = get_path_size(src_path)

        chunks, blocks = _determine_target_chunks_and_blocks(z_arr, chunks, blocks)
        cparams = determine_blosc2_cparams(z_arr, codec, clevel, shuffle)
        _, batch_slices = compute_batches(shape, chunks, None, itemsize, buffer_size_mb)

        # All validation is done, so only now remove the old output.
        if dst_path.is_dir():
            shutil.rmtree(dst_path)
        else:
            dst_path.unlink(missing_ok=True)

        b2_arr = blosc2.empty(
            shape,
            dtype=dtype,
            chunks=chunks,
            blocks=blocks,
            cparams=cparams,
            urlpath=str(dst_path),
            mode="w",
        )
        copy_duration = copy_batches(
            z_arr, b2_arr, batch_slices, uncompressed_bytes, "zarr -> b2nd", quiet=quiet, verbose=verbose
        )
        copy_attrs(z_arr, b2_arr)

        if verify or full_verify:
            verify_arrays(z_arr, b2_arr, batch_slices, full=full_verify, quiet=quiet)
    finally:
        if zip_store is not None:
            zip_store.close()

    dst_size = get_path_size(dst_path)
    total_duration = time.perf_counter() - t_start
    throughput_mb_s = (uncompressed_bytes / (1024 * 1024)) / copy_duration if copy_duration > 0 else 0.0

    return {
        "src_path": src_path,
        "dst_path": dst_path,
        "shape": shape,
        "dtype": dtype,
        "chunks": chunks,
        "blocks": blocks,
        "cparams": cparams,
        "src_size_bytes": src_size,
        "dst_size_bytes": dst_size,
        "uncompressed_bytes": uncompressed_bytes,
        "copy_duration_sec": copy_duration,
        "total_duration_sec": total_duration,
        "throughput_mb_s": throughput_mb_s,
        "src_cratio": uncompressed_bytes / src_size if src_size > 0 else 1.0,
        "dst_cratio": uncompressed_bytes / dst_size if dst_size > 0 else 1.0,
    }


def open_zarr_input(zarr: Any, path: Path) -> tuple[Any, Any]:
    """Open a Zarr array for reading; return it and the ZipStore to close (or None)."""
    zip_store = None
    if path.name.endswith(".zip"):
        zip_store = zarr.storage.ZipStore(str(path), mode="r")
    store = zip_store if zip_store is not None else str(path)

    try:
        z_arr = zarr.open_array(store=store, mode="r")
    except Exception as exc:
        is_group = type(exc).__name__ in {"ContainsGroupError", "NodeTypeValidationError"}
        if type(exc).__name__ == "ArrayNotFoundError":
            # Zarr v2 reports a missing array even when a group exists.
            with contextlib.suppress(FileNotFoundError):
                zarr.open_group(store=store, mode="r")
                is_group = True
        if zip_store is not None:
            zip_store.close()
        if is_group:
            raise ValueError(f"{path} is a Zarr group; pass the path of an array") from exc
        raise
    return z_arr, zip_store


def _determine_target_chunks_and_blocks(
    z,
    chunks: tuple[int, ...] | None,
    blocks: tuple[int, ...] | None,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Choose b2nd chunks and blocks for a Zarr array.

    Zarr shards (or chunks, if unsharded) become b2nd chunks; Zarr inner chunks become blocks.
    """
    shape = tuple(z.shape)
    dtype = np.dtype(z.dtype)

    if z.shards is None:
        target_chunks = chunks if chunks is not None else z.chunks
        target_blocks = blocks
    else:
        target_chunks = chunks if chunks is not None else z.shards
        target_blocks = blocks
        if chunks is None and blocks is None:
            target_blocks = z.chunks

    target_chunks = tuple(max(1, c) for c in target_chunks)

    # Fills in missing blocks, or checks that the given ones fit inside the chunks.
    target_chunks, target_blocks = blosc2.compute_chunks_blocks(
        shape, chunks=target_chunks, blocks=target_blocks, dtype=dtype
    )

    chunk_nbytes = math.prod(target_chunks) * dtype.itemsize
    if chunk_nbytes > blosc2.MAX_BUFFERSIZE:
        raise ValueError(
            f"Chunk shape {tuple(target_chunks)} is {chunk_nbytes} bytes, larger than the "
            f"Blosc2 limit of {blosc2.MAX_BUFFERSIZE} bytes; pass smaller --chunks"
        )
    return tuple(target_chunks), tuple(target_blocks)


def determine_blosc2_cparams(
    z_arr: Any,
    codec: str = "auto",
    clevel: int | None = None,
    shuffle: str = "auto",
) -> blosc2.CParams:
    """Translate the Zarr array's compressor into Blosc2 compression parameters."""
    compressors = z_arr.compressors
    compressor = compressors[0] if compressors else None
    kind = type(compressor).__name__
    itemsize = np.dtype(z_arr.dtype).itemsize

    if kind == "BloscCodec":
        src_cname, src_clevel, src_shuffle = compressor.cname, compressor.clevel, compressor.shuffle
    elif kind == "Blosc":
        # numcodecs shuffle is an int; -1 (auto) means bitshuffle for 1-byte items.
        src_cname, src_clevel = compressor.cname, compressor.clevel
        src_shuffle = {0: "noshuffle", 1: "shuffle", 2: "bitshuffle"}.get(
            compressor.shuffle, "bitshuffle" if itemsize == 1 else "shuffle"
        )
    elif kind in {"ZstdCodec", "Zstd"}:
        # Zstd level 0 means its default level (not "no compression"); levels go up to 22.
        src_cname = "zstd"
        src_clevel = min(compressor.level, 9) if compressor.level > 0 else 5
        src_shuffle = "shuffle"
    elif kind in {"GzipCodec", "GZip"}:
        # zlib is Blosc2's closest match to gzip.
        src_cname, src_clevel, src_shuffle = "zlib", compressor.level, "shuffle"
    else:
        src_cname, src_clevel, src_shuffle = "zstd", 5, "shuffle"

    final_cname = src_cname if codec == "auto" else codec
    final_clevel = src_clevel if clevel is None else clevel
    final_shuffle = src_shuffle if shuffle == "auto" else shuffle
    if not 0 <= final_clevel <= 9:
        raise ValueError(f"Compression level must be between 0 and 9, got {final_clevel}")

    blosc2_codec = BLOSC2_CODECS.get(final_cname.lower())
    if blosc2_codec is None:
        if codec != "auto":
            raise ValueError(f"Unsupported codec '{codec}'. Supported: auto, {', '.join(BLOSC2_CODECS)}")
        blosc2_codec = blosc2.Codec.ZSTD
    blosc2_filter = BLOSC2_SHUFFLE_FILTERS.get(final_shuffle.lower())
    if blosc2_filter is None:
        raise ValueError(
            f"Unsupported shuffle '{final_shuffle}'. Supported: auto, {', '.join(BLOSC2_SHUFFLE_FILTERS)}"
        )

    return blosc2.CParams(
        codec=blosc2_codec, clevel=final_clevel, filters=[blosc2_filter], typesize=itemsize
    )


def copy_attrs(z_arr: Any, b2_arr: blosc2.NDArray) -> None:
    """Copy the Zarr array's user attributes into the Blosc2 array's vlmeta."""
    for key, value in z_arr.attrs.items():
        with contextlib.suppress(Exception):
            b2_arr.schunk.vlmeta[key] = value


def build_parser(prog: str = "zarr-to-b2nd") -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    return argparse.ArgumentParser(
        prog=prog,
        description="Convert a Zarr array (v2 or v3) into a Blosc2 .b2nd file (NDArray).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""\
Examples:
  {prog} data.zarr
  {prog} data.zarr out.b2nd --chunks 100,1000
  {prog} data.zarr.zip -f --verify --full-verify
  {prog} data.zarr --codec zstd --clevel 3
""",
    )


def main(argv: list[str] | None = None) -> int:
    """Main CLI entry point."""
    prog = Path(sys.argv[0]).name if sys.argv and sys.argv[0] else "zarr-to-b2nd"
    parser = build_parser(prog=prog)

    parser.add_argument("input", help="Path to input Zarr array (directory or .zip)")

    parser.add_argument(
        "output",
        nargs="?",
        default=None,
        help="Path to output .b2nd file (default: <input>.b2nd)",
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
        help="Blosc2 chunk shape, e.g. '10,1000,1000' (default: the Zarr shards, or chunks if unsharded)",
    )
    parser.add_argument(
        "--blocks",
        type=str,
        default=None,
        help="Blosc2 block shape (default: the Zarr inner chunks if sharded, otherwise computed)",
    )
    parser.add_argument(
        "--codec",
        type=str,
        default="auto",
        choices=["auto", *BLOSC2_CODECS],
        help="Compression codec for Blosc2 (default: 'auto', translated from the Zarr compressor)",
    )
    parser.add_argument(
        "--clevel",
        type=int,
        default=None,
        help="Compression level (0-9, default: inherit from the Zarr compressor)",
    )
    parser.add_argument(
        "--shuffle",
        type=str,
        default="auto",
        choices=["auto", "noshuffle", "shuffle", "bitshuffle"],
        help="Shuffle filter (default: 'auto', inherit from the Zarr compressor)",
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
    parsed_blocks = parse_shape(args.blocks) if args.blocks else None

    try:
        summary = zarr_to_b2nd(
            input_path=args.input,
            output_path=out_path,
            chunks=parsed_chunks,
            blocks=parsed_blocks,
            codec=args.codec,
            clevel=args.clevel,
            shuffle=args.shuffle,
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
        print_summary(summary, dst_label="Blosc2")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
