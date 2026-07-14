#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Multiple concurrent writers: several processes can write to the same
# on-disk NDArray, not just read it, as long as every handle opens it with
# `locking=True`. See
# doc/guides/sharing_across_processes.md for the full contract.
# This example shows the two things that matter in practice:
#
# - Writers touching *disjoint* regions never need to coordinate beyond
#   `locking=True`: each write is atomic to the other handles, so the
#   result is deterministic no matter how the writers interleave.
# - A read-modify-write on the *same* cell (`arr[i] = arr[i] + 1`) is two
#   separate locked operations -- the read and the write -- so concurrent
#   writers can race between them and lose updates. `arr.holding_lock()`
#   brackets the whole read-modify-write into one atomic block, which is
#   what actually fixes it.

import multiprocessing
import time

import numpy as np

import blosc2

DISJOINT_URLPATH = "mwmr-disjoint.b2nd"
COUNTER_URLPATH = "mwmr-counter.b2nd"
NWRITERS = 4
ROWS_PER_WRITER = 200
INCREMENTS_PER_WRITER = 200


def disjoint_writer(rank):
    """Fill this writer's own row range. No two writers touch the same rows,
    so there is nothing to race on beyond the per-write atomicity locking
    already gives us."""
    arr = blosc2.open(DISJOINT_URLPATH, mode="a", locking=True)
    start = rank * ROWS_PER_WRITER
    arr[start : start + ROWS_PER_WRITER, :] = rank + 1


def counter_writer_unsafe(_rank):
    """Bump a shared counter cell without holding_lock(). Each `+= 1` is a
    locked read followed by a separate locked write, so another writer can
    slip in between them and clobber the result -- updates get lost."""
    arr = blosc2.open(COUNTER_URLPATH, mode="a", locking=True)
    for _ in range(INCREMENTS_PER_WRITER):
        arr[0] = arr[0] + 1


def counter_writer_safe(_rank):
    """Same increment, but the read and the write happen inside one
    holding_lock() block, so no other writer can interleave between them."""
    arr = blosc2.open(COUNTER_URLPATH, mode="a", locking=True)
    for _ in range(INCREMENTS_PER_WRITER):
        with arr.holding_lock():
            arr[0] = arr[0] + 1


def run(target):
    ctx = multiprocessing.get_context("spawn")
    procs = [ctx.Process(target=target, args=(rank,)) for rank in range(NWRITERS)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    for p in procs:
        assert p.exitcode == 0


if __name__ == "__main__":
    blosc2.remove_urlpath(DISJOINT_URLPATH)
    blosc2.remove_urlpath(COUNTER_URLPATH)

    # --- Disjoint regions: safe without any extra coordination -------------
    blosc2.zeros(
        (NWRITERS * ROWS_PER_WRITER, 10), dtype=np.int64, urlpath=DISJOINT_URLPATH, locking=True, mode="w"
    )
    t0 = time.time()
    run(disjoint_writer)
    arr = blosc2.open(DISJOINT_URLPATH, locking=True)
    for rank in range(NWRITERS):
        start = rank * ROWS_PER_WRITER
        expected = rank + 1
        assert np.all(arr[start : start + ROWS_PER_WRITER, :] == expected), f"writer {rank} region is wrong"
    print(f"{NWRITERS} writers filled disjoint regions correctly in {time.time() - t0:.2f}s")

    # --- Same cell, no holding_lock(): updates get lost ---------------------
    blosc2.zeros((1,), dtype=np.int64, urlpath=COUNTER_URLPATH, locking=True, mode="w")
    run(counter_writer_unsafe)
    arr = blosc2.open(COUNTER_URLPATH, locking=True)
    unsafe_count = int(arr[0])
    expected = NWRITERS * INCREMENTS_PER_WRITER
    print(
        f"unsafe counter: {unsafe_count} (expected {expected}) -- {'OK' if unsafe_count == expected else 'LOST UPDATES'}"
    )

    # --- Same cell, with holding_lock(): every update lands -----------------
    blosc2.zeros((1,), dtype=np.int64, urlpath=COUNTER_URLPATH, locking=True, mode="w")
    run(counter_writer_safe)
    arr = blosc2.open(COUNTER_URLPATH, locking=True)
    safe_count = int(arr[0])
    print(
        f"safe counter:   {safe_count} (expected {expected}) -- {'OK' if safe_count == expected else 'LOST UPDATES'}"
    )
    assert safe_count == expected, "holding_lock() should make every increment land"

    blosc2.remove_urlpath(DISJOINT_URLPATH)
    blosc2.remove_urlpath(COUNTER_URLPATH)
