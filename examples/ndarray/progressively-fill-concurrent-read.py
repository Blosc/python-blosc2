#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Single-writer / multiple-reader (SWMR) on a fixed-shape NDArray:
# "preallocate big, fill progressively, read concurrently".
#
# A large on-disk array is preallocated as UNINIT special chunks -- this is
# nearly free, no matter the logical size, because special chunks only exist
# as entries in the frame index (no data on disk).  A writer process then
# fills it chunk by chunk while this process reads it concurrently, following
# the writer's progress without ever reopening its handle.
#
# The two ingredients that make this safe are:
#
# - `locking=True` on *every* handle: accesses are serialized through a small
#   sidecar lock file next to the array (readers share the lock, mutations
#   take it exclusively), and a handle whose cached view became stale re-syncs
#   automatically.  It is advisory, so all handles must opt in.  Note that
#   this is not supported on network filesystems (NFS).
# - `iterchunks_info()` to tell filled chunks from not-yet-written ones: it
#   reads through the frame, so it is always current -- an UNINIT chunk reads
#   back as uninitialized data, so readers should check before trusting a
#   region (or preallocate with blosc2.zeros() for deterministic zeros).
#
# The shape is fixed for the whole session: growing the array while readers
# follow (a la HDF5 SWMR) is not supported yet -- readers would need to
# reopen to see new extents.

import multiprocessing
import time

import numpy as np

import blosc2

URLPATH = "progressive-fill.b2nd"
SHAPE = (10_000, 1_000)
CHUNKS = (500, 1_000)  # 20 chunks, filled one per step
DTYPE = np.float64


def chunk_value(nchunk):
    """The value the writer fills chunk `nchunk` with (so readers can verify)."""
    return float(nchunk + 1)


def writer():
    """Fill the array chunk by chunk, as a separate OS process would."""
    arr = blosc2.open(URLPATH, mode="a", locking=True)
    rows_per_chunk = CHUNKS[0]
    for nchunk in range(arr.schunk.nchunks):
        start = nchunk * rows_per_chunk
        # A chunk-aligned assignment is a single exclusive-locked update
        arr[start : start + rows_per_chunk] = np.full(CHUNKS, chunk_value(nchunk), dtype=DTYPE)
        time.sleep(0.05)  # simulate data arriving progressively


def filled_chunks(arr):
    """The set of chunks already written.  iterchunks_info() reads through the
    frame, so a long-lived reader handle always sees the current state."""
    return {
        info.nchunk
        for info in arr.schunk.iterchunks_info()
        if info.special == blosc2.SpecialValue.NOT_SPECIAL
    }


if __name__ == "__main__":
    blosc2.remove_urlpath(URLPATH)

    # Preallocate: all chunks are UNINIT specials, so this is instantaneous
    # and takes no disk space, no matter how large the logical array is
    t0 = time.time()
    arr = blosc2.uninit(
        SHAPE, DTYPE, urlpath=URLPATH, mode="w", chunks=CHUNKS, contiguous=False, locking=True
    )
    nchunks = arr.schunk.nchunks
    print(
        f"Preallocated {arr.schunk.nbytes / 2**30:.1f} GB logical "
        f"({nchunks} chunks) in {(time.time() - t0) * 1000:.1f} ms"
    )
    del arr

    # Start the writer in another process...
    proc = multiprocessing.get_context("spawn").Process(target=writer)
    proc.start()

    # ... and follow it from here with an independent reader handle, opened
    # once and never reopened
    reader = blosc2.open(URLPATH, locking=True)
    rows_per_chunk = CHUNKS[0]
    seen = set()
    while len(seen) < nchunks:
        for nchunk in sorted(filled_chunks(reader) - seen):
            # This region is ready: read and verify it through the stale-then-
            # resynced reader handle
            row = reader[nchunk * rows_per_chunk]
            assert np.all(row == chunk_value(nchunk)), f"bad data in chunk {nchunk}"
            seen.add(nchunk)
            print(f"read chunk {nchunk:2d} while the writer keeps going ({len(seen)}/{nchunks} filled)")
        time.sleep(0.01)

    proc.join()
    assert proc.exitcode == 0

    # Everything the writer produced is visible, still on the original handle
    assert len(filled_chunks(reader)) == nchunks
    print("All chunks written and verified concurrently.  Success!")

    blosc2.remove_urlpath(URLPATH)
