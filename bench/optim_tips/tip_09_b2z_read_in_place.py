#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Tip: .b2z is a single-file container for grouping related data (a CTable's
# columns and indexes, hierarchies of arrays, or both). Opened read-only with
# mmap_mode="r" it is never extracted: every member is memory-mapped in place
# at its offset inside the container. Unpacking it first just costs time and
# a second copy on disk.

import shutil
import zipfile
from pathlib import Path

import blosc2
from common import fmt_bytes, make_table, measure, save_plot

N = 10_000_000
B2D = str(Path(__file__).parent / "tip_09.b2d")
B2Z = str(Path(__file__).parent / "tip_09.b2z")

_t = make_table(N, B2D)
_t.to_b2z(B2Z, overwrite=True)
_t.close()


def naive():
    # the zip-file reflex: unpack, then open the extracted tree
    dest = Path(__file__).parent / "tip_09_extracted.b2d"
    shutil.rmtree(dest, ignore_errors=True)
    with zipfile.ZipFile(B2Z) as z:
        z.extractall(dest)
    t = blosc2.open(str(dest))
    return t["temperature"].sum()


def tip():
    # No extraction, no second copy: the container is mapped once and every
    # member is read at its offset inside the single mapped file.
    t = blosc2.open(B2Z, mmap_mode="r")
    return t["temperature"].sum()


if __name__ == "__main__":
    naive_t, naive_m = measure(__file__, "naive")
    tip_t, tip_m = measure(__file__, "tip")

    print(f"naive  extract + open : {naive_t:.4f}s  peak {fmt_bytes(naive_m)}")
    print(f"tip    open in place  : {tip_t:.4f}s  peak {fmt_bytes(tip_m)}")
    print(f"speedup: {naive_t / tip_t:.1f}x   memory: {naive_m / max(tip_m, 1):.1f}x less")

    save_plot(
        "tip_09_b2z_read_in_place.png",
        f"Reading a .b2z in place vs extracting it first — {N:,}-row table",
        "extract + open",
        "open .b2z in place",
        naive_t,
        tip_t,
        naive_m,
        tip_m,
    )
