#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Tip 16: blosc2.set_prefetch() lets a few background threads read the chunks a
# slice needs, in large parallel requests and ahead of decompression.  Without
# it, a slice of an on-disk array reads its blocks one after the other, and
# every read that misses the page cache waits for the storage.
#
# The benefit depends on the storage latency: on a local NVMe disk cold patches
# get ~1.5-3x faster, on NFS ~3-5x.  Every read here is cold: the file is
# dropped from the page cache before each one (Linux; elsewhere the reads are
# warm, and both variants measure the same).

import os
from pathlib import Path

import numpy as np
from common import fmt_bytes, measure, save_plot

import blosc2

SHAPE, CHUNKS, BLOCKS, PATCH = (4, 256, 256, 256), (1, 64, 128, 128), (1, 16, 32, 128), (128, 128, 128)
URLPATH = str(Path(__file__).parent / "tip_16.b2nd")
N_READS = 40


def _build():
    rng = np.random.default_rng(0)
    data = blosc2.zeros(SHAPE, dtype=np.float32, chunks=CHUNKS, blocks=BLOCKS, urlpath=URLPATH, mode="w")
    for c in range(SHAPE[0]):
        data[c] = rng.normal(size=SHAPE[1:]).astype(np.float32).round(2)


_build()

_starts = np.random.default_rng(1).integers(
    0, [s - p + 1 for s, p in zip(SHAPE[1:], PATCH, strict=True)], size=(N_READS, 3)
)


def _drop_from_page_cache():
    if hasattr(os, "posix_fadvise"):
        fd = os.open(URLPATH, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)


def _cold_patch_reads():
    total = 0.0
    for start in _starts:
        _drop_from_page_cache()
        arr = blosc2.open(URLPATH, mmap_mode="r")
        total += arr[
            (slice(None),) + tuple(slice(s, s + p) for s, p in zip(start, PATCH, strict=True))
        ].sum()
        del arr  # unmapped, so that the next drop reaches every page
    return total


def naive():
    return _cold_patch_reads()


def tip():
    blosc2.set_prefetch(4)  # per process; BLOSC_PREFETCH=4 in the environment does the same
    return _cold_patch_reads()


if __name__ == "__main__":
    naive_t, naive_m = measure(__file__, "naive")
    tip_t, tip_m = measure(__file__, "tip")

    print(f"naive  without read-ahead   : {naive_t:.3f}s  peak {fmt_bytes(naive_m)}")
    print(f"tip    set_prefetch(4)      : {tip_t:.3f}s  peak {fmt_bytes(tip_m)}")
    print(f"speedup: {naive_t / tip_t:.1f}x")

    save_plot(
        "tip_16_prefetch.png",
        f"set_prefetch(4) vs no read-ahead — {N_READS} cold patch reads",
        "no read-ahead",
        "set_prefetch(4)",
        naive_t,
        tip_t,
        naive_m,
        tip_m,
    )
