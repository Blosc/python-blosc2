#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Tip 2: blosc2's double partition (chunks, subdivided into blocks) rewards
# aligned reads at both levels:
#   * A read aligned with the chunk grid decompresses exactly one chunk,
#     while the same-sized read shifted off-grid straddles two chunks.
#     An NDArray.slice() aligned with chunk boundaries goes further: it
#     copies whole chunks as-is, with no decompression at all.
#   * At the block level, a read aligned with the block grid decompresses
#     exactly one block.  With auto (cache-sized, small) blocks this is
#     drowned by per-call overhead, so it is demonstrated on an array with
#     explicitly larger blocks.  An NDArray.slice() at block granularity
#     cannot be chunk-aligned on both ends, so it still pays the
#     decompress/recompress cost.
#
# Each panel shows three bars: unaligned read, aligned read, and
# NDArray.slice() — all the same slice size.

import matplotlib.pyplot as plt
import numpy as np

import blosc2
from common import COLOR_NAIVE, COLOR_TIP, GRID, INK, MUTED, OUT_DIR, measure

ROWS, COLS = 16_000, 2_000
K_READS = 400
BIG_BLOCK_ROWS = 100  # explicit larger blocks for the block panel (1.6 MB each)

# Representative (compressible) data, not incompressible random noise.
_base = np.arange(COLS, dtype=np.float64)
_data = np.tile(_base, (ROWS, 1)) + np.arange(ROWS, dtype=np.float64)[:, None] * 0.001

# Panel 1 array: let blosc2 choose the partitions.
arr = blosc2.asarray(_data)
CHUNK_ROWS = arr.chunks[0]

# Panel 2 array: explicitly larger blocks so alignment pays off.
arr_big = blosc2.asarray(_data, chunks=(4 * CHUNK_ROWS, COLS), blocks=(BIG_BLOCK_ROWS, COLS))
BLOCK_ROWS = arr_big.blocks[0]
BIG_CHUNK_ROWS = arr_big.chunks[0]  # 4000

# Random start positions — aligned to the respective grid.
_rng = np.random.default_rng(1)
_chunk_starts = _rng.integers(0, ROWS // CHUNK_ROWS - 2, size=K_READS) * CHUNK_ROWS
_block_starts = _rng.integers(0, ROWS // BLOCK_ROWS - 2, size=K_READS) * BLOCK_ROWS


def read_chunk_unaligned():
    """Chunk-sized reads shifted half a chunk off the grid."""
    for s in _chunk_starts:
        arr[s + CHUNK_ROWS // 2 : s + CHUNK_ROWS // 2 + CHUNK_ROWS]


def read_chunk_aligned():
    """Chunk-sized reads starting on chunk boundaries."""
    for s in _chunk_starts:
        arr[s : s + CHUNK_ROWS]


def slice_chunk_aligned():
    """Chunk-sized slice() on chunk boundaries → fast path (no decompress)."""
    for s in _chunk_starts:
        arr.slice((slice(s, s + CHUNK_ROWS), slice(None)))


def read_block_unaligned():
    """Block-sized reads shifted half a block off the grid."""
    for s in _block_starts:
        arr_big[s + BLOCK_ROWS // 2 : s + BLOCK_ROWS // 2 + BLOCK_ROWS]


def read_block_aligned():
    """Block-sized reads starting on block boundaries."""
    for s in _block_starts:
        arr_big[s : s + BLOCK_ROWS]


def slice_block_aligned():
    """Block-sized slice() on block boundaries → general path (decompress+recompress).

    The fast path requires *chunk* alignment; at block granularity (100 rows)
    with chunk size 4000, both boundaries cannot be chunk-aligned, so every
    slice() call decompresses and recompresses the containing chunk(s).
    """
    for s in _block_starts:
        arr_big.slice((slice(s, s + BLOCK_ROWS), slice(None)))


if __name__ == "__main__":
    print(f"auto partitions: chunks={arr.chunks}, blocks={arr.blocks}")
    print(f"read-panel array: chunks={arr_big.chunks}, blocks={arr_big.blocks} (explicit)")

    chunk_unaligned_t, _ = measure(__file__, "read_chunk_unaligned")
    chunk_aligned_t, _ = measure(__file__, "read_chunk_aligned")
    chunk_slice_t, _ = measure(__file__, "slice_chunk_aligned")
    block_unaligned_t, _ = measure(__file__, "read_block_unaligned")
    block_aligned_t, _ = measure(__file__, "read_block_aligned")
    block_slice_t, _ = measure(__file__, "slice_block_aligned")

    print(f"{K_READS} chunk-sized reads, unaligned: {chunk_unaligned_t:.4f}s")
    print(
        f"{K_READS} chunk-sized reads, aligned  : {chunk_aligned_t:.4f}s"
        f"   ({chunk_unaligned_t / chunk_aligned_t:.1f}x faster)"
    )
    print(
        f"{K_READS} chunk-sized slice, aligned : {chunk_slice_t:.4f}s"
        f"   ({chunk_unaligned_t / chunk_slice_t:.1f}x faster than unaligned read)"
    )
    print(f"{K_READS} block-sized reads, unaligned: {block_unaligned_t:.4f}s")
    print(
        f"{K_READS} block-sized reads, aligned  : {block_aligned_t:.4f}s"
        f"   ({block_unaligned_t / block_aligned_t:.1f}x faster)"
    )
    print(
        f"{K_READS} block-sized slice, aligned : {block_slice_t:.4f}s"
        f"   ({block_unaligned_t / block_slice_t:.1f}x vs unaligned read)"
    )

    COLOR_SLICE = "#e8731a"

    fig, (ax_c, ax_b) = plt.subplots(1, 2, figsize=(8, 3.2))
    fig.suptitle(
        f"Aligned vs unaligned reads — {ROWS}×{COLS} float64",
        fontsize=11,
        color=INK,
    )

    chunk_sz = f"{CHUNK_ROWS}×{COLS}"
    block_sz = f"{BLOCK_ROWS}×{COLS}"

    panels = (
        (
            ax_c,
            f"{K_READS} random reads, size={chunk_sz} (chunk-sized)",
            ("unaligned\nread", "chunk-aligned\nread", "chunk-aligned\nslice()"),
            (COLOR_NAIVE, COLOR_TIP, COLOR_SLICE),
            (chunk_unaligned_t, chunk_aligned_t, chunk_slice_t),
        ),
        (
            ax_b,
            f"{K_READS} random reads, size={block_sz} (block-sized)",
            ("unaligned\nread", "block-aligned\nread", "block-aligned\nslice()"),
            (COLOR_NAIVE, COLOR_TIP, COLOR_SLICE),
            (block_unaligned_t, block_aligned_t, block_slice_t),
        ),
    )

    for ax, title, labels, colors, values in panels:
        bars = ax.bar(labels, values, color=colors, width=0.55)
        ax.set_title(title, fontsize=9, color=INK)
        ax.set_ylabel("Time (s)", color=INK, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.set_yticklabels([])
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        top = max(values)
        ax.set_ylim(0, top * 1.25)
        for bar, v in zip(bars, values, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + top * 0.03,
                f"{v:.3g}s",
                ha="center",
                va="bottom",
                fontsize=8,
                color=INK,
            )

    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out_path = OUT_DIR / "tip_02_chunk_aligned_slicing.png"
    fig.savefig(out_path, dpi=150)
    print(f"plot saved to {out_path}")
