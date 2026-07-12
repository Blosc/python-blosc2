#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Multiple concurrent writers, this time growing an array, not just
# writing into its existing extent. `mwmr-mode.py` only exercises writes
# into a *fixed*-shape array; this one has several processes `append()` to
# the *same* 1-D array concurrently, which is a fundamentally different
# path -- each append is a resize (grows the shape metadata) followed by a
# slice write, i.e. two separate locked operations rather than one. See
# doc/guides/sharing_across_processes.rst.
#
# Each writer's batches carry a unique tag (writer_id, batch_index) baked
# into their values, so at the end we can verify -- purely from the data,
# with no positional assumptions -- that every batch from every writer
# landed exactly once, complete and untorn, regardless of how the appends
# from different processes interleaved on disk.

import multiprocessing

import numpy as np

import blosc2

URLPATH = "mwmr-enlarge.b2nd"
NWRITERS = 4
APPENDS_PER_WRITER = 30
ITEMS_PER_APPEND = 25
DTYPE = np.int64


def writer(rank):
    arr = blosc2.open(URLPATH, mode="a", locking=True)
    for i in range(APPENDS_PER_WRITER):
        tag = rank * 1_000_000 + i
        batch = np.full(ITEMS_PER_APPEND, tag, dtype=DTYPE)
        # append() is resize() + a slice write -- two locked operations, not
        # one. Without holding_lock() two writers can both compute their
        # target position from the same pre-append length and collide.
        with arr.holding_lock():
            arr.append(batch)


if __name__ == "__main__":
    blosc2.remove_urlpath(URLPATH)
    blosc2.zeros(
        (0,),
        dtype=DTYPE,
        urlpath=URLPATH,
        locking=True,
        mode="w",
        chunks=(ITEMS_PER_APPEND * 4,),
        blocks=(ITEMS_PER_APPEND,),
    )

    ctx = multiprocessing.get_context("spawn")
    procs = [ctx.Process(target=writer, args=(rank,)) for rank in range(NWRITERS)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    for p in procs:
        assert p.exitcode == 0

    arr = blosc2.open(URLPATH, locking=True)
    expected_len = NWRITERS * APPENDS_PER_WRITER * ITEMS_PER_APPEND
    assert arr.shape[0] == expected_len, (
        f"final length {arr.shape[0]} != expected {expected_len} -- lost or duplicated an append"
    )

    # Each atomic append inserted ITEMS_PER_APPEND identical values as one
    # indivisible block, so the final array -- whatever order the appends
    # from different writers landed in -- must be an exact concatenation of
    # such blocks. Reshape and check every block is uniform (untorn) and
    # that the multiset of tags matches exactly what was appended (nothing
    # lost, nothing duplicated).
    blocks = np.asarray(arr[:]).reshape(-1, ITEMS_PER_APPEND)
    tags = []
    for block in blocks:
        assert np.all(block == block[0]), f"torn append: block mixes values {np.unique(block)}"
        tags.append(int(block[0]))

    expected_tags = sorted(
        rank * 1_000_000 + i for rank in range(NWRITERS) for i in range(APPENDS_PER_WRITER)
    )
    assert sorted(tags) == expected_tags, (
        "append tags don't match: some batch was lost, duplicated, or corrupted"
    )

    print(
        f"{NWRITERS} writers x {APPENDS_PER_WRITER} appends of {ITEMS_PER_APPEND} items each: "
        f"all {len(tags)} batches present, untorn, exactly once. Final length {arr.shape[0]}."
    )

    blosc2.remove_urlpath(URLPATH)
