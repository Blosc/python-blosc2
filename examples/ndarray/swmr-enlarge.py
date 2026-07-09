#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Pure SWMR (single writer, multiple readers), no locking at all -- SWMR
# is always on, `locking=True` plays no role here. One process grows a
# disk-based array batch by batch; several reader processes, opened
# *before* the writer starts, follow the new shape and data live, with
# no reopen. See doc/getting_started/sharing_across_processes.rst,
# "SWMR without locking", for the contract this relies on:
#
# - A reader's cached shape re-syncs the next time it *touches* the
#   container -- a data read, a vlmeta lookup, or an explicit
#   `NDArray.refresh()` poll that doesn't read any data at all.
# - Growth changes the on-disk length, which is what staleness
#   detection keys off, so it is reliably picked up.
# - A resize is not a write: the newly grown region is uninitialized
#   until the writer explicitly fills it, so a reader that only trusts
#   `shape` can observe rows that exist on disk but aren't filled yet --
#   consistency here is per-operation, not a whole-container snapshot.
#   This example sidesteps that by only treating a batch as safe to
#   verify once the *next* batch's resize has started (the same
#   technique used in tests/test_swmr.py::test_cross_process_growth_hammer),
#   and by using a vlmeta flag -- itself re-synced on every access -- as
#   an explicit "the writer is done" signal for the final tail check.
# - A reader racing the writer without locking can still occasionally hit
#   a transient read error mid-mutation, even in a region considered
#   "settled" above -- retrying is the documented workaround, so every
#   read below is wrapped in one.
# - Exactly one writer. A second writer without `locking=True` is not
#   supported -- see mwmr-mode.py / mwmr-enlarge.py for the locked
#   multi-writer story.

import multiprocessing
import time

import numpy as np

import blosc2

URLPATH = "swmr-enlarge.b2nd"
NREADERS = 3
NBATCHES = 60
ROWS_PER_BATCH = 5
NCOLS = 8
DTYPE = np.int64


def writer():
    """The single writer. No `locking=True` anywhere -- SWMR readers
    follow along through plain shape growth."""
    arr = blosc2.open(URLPATH, mode="a")
    for i in range(NBATCHES):
        start = arr.shape[0]
        arr.resize((start + ROWS_PER_BATCH, NCOLS))
        # Between the resize above and this assignment, a reader that
        # reads row `start` sees uninitialized data, not the value `i`
        # below -- that's the gap the readers are careful about.
        arr[start : start + ROWS_PER_BATCH, :] = i
        time.sleep(0.005)  # let readers observe growth in more than one step
    # Set once everything is written: readers use this as the signal that
    # it is safe to read and verify the whole array, tail included.
    arr.schunk.vlmeta["done"] = True


def reader(rank):
    """Opened once, before the writer starts; never reopened -- every
    poll below just re-syncs the same handle."""
    arr = blosc2.open(URLPATH, mode="r")
    expected_rows = NBATCHES * ROWS_PER_BATCH
    seen_shapes = []
    verified = 0
    done = False
    while not done:
        try:
            # refresh() re-syncs the cached shape without reading any data.
            arr.refresh()
            rows = arr.shape[0]
            if not seen_shapes or rows != seen_shapes[-1]:
                seen_shapes.append(rows)

            # Only the region *before* the batch currently being (possibly)
            # filled is guaranteed written -- verify it live, incrementally,
            # as it becomes safe.
            settled = max(rows - ROWS_PER_BATCH, 0)
            if settled > verified:
                block = np.asarray(arr[verified:settled, :])
                for offset in range(0, settled - verified, ROWS_PER_BATCH):
                    batch_idx = (verified + offset) // ROWS_PER_BATCH
                    rows_block = block[offset : offset + ROWS_PER_BATCH]
                    assert np.all(rows_block == batch_idx), (
                        f"reader {rank} batch {batch_idx} wrong or torn while still growing: "
                        f"{np.unique(rows_block)}"
                    )
                verified = settled

            # "done" in vlmeta re-syncs the container's metadata cache too,
            # so this also picks up the writer's final vlmeta write.
            done = "done" in arr.schunk.vlmeta
        except RuntimeError:
            # A reader can race the writer mid-mutation and hit a transient
            # read error even here -- retry on the next poll.
            pass
        if not done:
            time.sleep(0.002)

    # Saw more than one shape -- proof this reader followed growth live,
    # not just caught the final state on its first poll.
    assert len(seen_shapes) > 1, f"reader {rank} only observed one shape {seen_shapes}"
    assert seen_shapes == sorted(seen_shapes), f"reader {rank} saw shape shrink: {seen_shapes}"

    # The writer is done, so the tail we couldn't trust mid-loop is now
    # safe -- but the read itself can still race an in-flight mutation
    # from another handle, so retry on the transient error.
    for _attempt in range(50):
        try:
            arr.refresh()
            assert arr.shape == (expected_rows, NCOLS), f"reader {rank} final shape {arr.shape} is wrong"
            tail = np.asarray(arr[verified:expected_rows, :])
            break
        except RuntimeError:
            time.sleep(0.005)
    else:
        raise RuntimeError(f"reader {rank}: could not get a clean final read after retries")

    for offset in range(0, expected_rows - verified, ROWS_PER_BATCH):
        batch_idx = (verified + offset) // ROWS_PER_BATCH
        rows_block = tail[offset : offset + ROWS_PER_BATCH]
        assert np.all(rows_block == batch_idx), (
            f"reader {rank} final batch {batch_idx} wrong: {np.unique(rows_block)}"
        )

    print(
        f"reader {rank}: followed growth through {len(seen_shapes)} live shape changes, "
        f"all {NBATCHES} batches verified correct and untorn"
    )


if __name__ == "__main__":
    blosc2.remove_urlpath(URLPATH)
    blosc2.zeros(
        (0, NCOLS),
        dtype=DTYPE,
        urlpath=URLPATH,
        mode="w",
        chunks=(ROWS_PER_BATCH * 4, NCOLS),
        blocks=(ROWS_PER_BATCH, NCOLS),
    )

    ctx = multiprocessing.get_context("spawn")
    readers = [ctx.Process(target=reader, args=(rank,)) for rank in range(NREADERS)]
    for p in readers:
        p.start()
    time.sleep(0.1)  # let readers open the array before the writer starts growing it

    w = ctx.Process(target=writer)
    w.start()
    w.join()
    assert w.exitcode == 0

    for p in readers:
        p.join()
    for p in readers:
        assert p.exitcode == 0

    print(
        f"{NREADERS} readers followed {NBATCHES} batches of {ROWS_PER_BATCH} rows each, "
        "grown with pure SWMR (no locking)"
    )

    blosc2.remove_urlpath(URLPATH)
