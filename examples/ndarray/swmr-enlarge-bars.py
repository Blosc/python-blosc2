#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Same pure SWMR (single writer, multiple readers) scenario as
# `swmr-enlarge.py` -- no locking at all -- but rendered live with
# `rich` progress bars so the effect is visible instead of just
# asserted: the writer's bar advances batch by batch, and each reader's
# bar chases it a beat behind, because a reader only learns about new
# data the next time it *polls* (there is no push notification). Each
# bar also reports its own I/O bandwidth (MB/s), tracked in bytes rather
# than batch counts. Run it in a real terminal to see the readers trail
# the writer in real time.
#
# The mechanics are identical to `swmr-enlarge.py` -- read that one
# first for the annotated contract (staleness detection, the
# "settled vs. just-resized" gap, the vlmeta "done" signal, and why
# reads are retried). This file adds a `multiprocessing.Queue` purely
# to ship progress numbers from the writer/reader processes back to the
# one process that owns the terminal and draws the bars; it plays no
# role in the SWMR mechanism itself.
#
# Don't read the reported MB/s or GB/s as real disk bandwidth. Every
# batch is filled with a single constant value, so it is trivially
# compressible -- the bottleneck here is compression/decompression CPU
# work, not disk I/O, and the on-disk file stays tiny regardless of how
# large NBATCHES / ROWS_PER_BATCH make the logical array. The number is
# genuine throughput, just not the kind a real (incompressible) workload
# would see; it's also the *combined* rate of one writer plus several
# independent readers running concurrently on separate cores, each
# processing the whole array on its own, not one shared data stream --
# summing the bars' rates can legitimately exceed what a single disk
# could sustain.

import multiprocessing
import time
from queue import Empty

import numpy as np
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TransferSpeedColumn,
)

import blosc2

URLPATH = "swmr-enlarge-bars.b2nd"
NREADERS = 3
# One more 2x pass, same idea as before: batch count stays put, batch
# size doubles again, so the extra data adds real write time (slowing the
# demo back down) rather than more fixed per-iteration Python overhead.
# Chunks/blocks scale with it (~82 MB chunk, ~20 MB block -- big, but
# still just a demo).
NBATCHES = 400
ROWS_PER_BATCH = 10000
NCOLS = 256
DTYPE = np.int64
ITEMSIZE = np.dtype(DTYPE).itemsize
BYTES_PER_BATCH = ROWS_PER_BATCH * NCOLS * ITEMSIZE
TOTAL_BYTES = NBATCHES * BYTES_PER_BATCH
WRITER_DELAY = 0.0012  # seconds between batches


def writer(queue):
    """The single writer. No `locking=True` anywhere -- SWMR readers
    follow along through plain shape growth."""
    arr = blosc2.open(URLPATH, mode="a")
    for i in range(NBATCHES):
        start = arr.shape[0]
        arr.resize((start + ROWS_PER_BATCH, NCOLS))
        arr[start : start + ROWS_PER_BATCH, :] = i
        queue.put(("writer", (i + 1) * BYTES_PER_BATCH))
        time.sleep(WRITER_DELAY)
    # Signal readers it is safe to trust and verify the whole array, tail
    # included -- see swmr-enlarge.py for why this is needed.
    arr.schunk.vlmeta["done"] = True


def reader(rank, queue):
    """Opened once, before the writer starts; never reopened. Each
    reader polls at its own pace, so their bars naturally spread out."""
    arr = blosc2.open(URLPATH, mode="r")
    expected_rows = NBATCHES * ROWS_PER_BATCH
    poll_delay = 0.001 + rank * 0.001  # readers 0, 1, 2 poll at 1/2/3ms
    verified = 0
    done = False
    while not done:
        try:
            arr.refresh()
            rows = arr.shape[0]
            # Only the region *before* the batch currently being (possibly)
            # filled is guaranteed written -- see swmr-enlarge.py.
            settled = max(rows - ROWS_PER_BATCH, 0)
            if settled > verified:
                block = np.asarray(arr[verified:settled, :])
                for offset in range(0, settled - verified, ROWS_PER_BATCH):
                    batch_idx = (verified + offset) // ROWS_PER_BATCH
                    rows_block = block[offset : offset + ROWS_PER_BATCH]
                    assert np.all(rows_block == batch_idx), (
                        f"reader {rank} batch {batch_idx} wrong or torn while still growing"
                    )
                verified = settled
                queue.put((f"reader{rank}", verified * NCOLS * ITEMSIZE))
            done = "done" in arr.schunk.vlmeta
        except RuntimeError:
            # A reader can race the writer mid-mutation and hit a transient
            # read error -- retry on the next poll.
            pass
        if not done:
            time.sleep(poll_delay)

    # The writer is done, so the tail is now safe -- retry on the rare
    # transient read error, same as the polling loop above.
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
        assert np.all(rows_block == batch_idx), f"reader {rank} final batch {batch_idx} wrong"

    queue.put((f"reader{rank}", TOTAL_BYTES))


if __name__ == "__main__":
    console = Console()
    console.rule("[bold]Pure SWMR (single writer, multiple readers) -- no locking[/bold]")

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
    queue = ctx.Queue()
    writer_proc = ctx.Process(target=writer, args=(queue,))
    reader_procs = [ctx.Process(target=reader, args=(rank, queue)) for rank in range(NREADERS)]

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeElapsedColumn(),
        console=console,
    )
    writer_task = progress.add_task("[bold cyan]writer   (appending) ", total=TOTAL_BYTES)
    reader_tasks = {
        rank: progress.add_task(f"[green]reader {rank} (verifying)", total=TOTAL_BYTES)
        for rank in range(NREADERS)
    }

    def drain():
        try:
            while True:
                role, completed = queue.get_nowait()
                task_id = writer_task if role == "writer" else reader_tasks[int(role[len("reader") :])]
                progress.update(task_id, completed=completed)
        except Empty:
            pass

    procs = [writer_proc, *reader_procs]
    with progress:
        for p in reader_procs:
            p.start()
        time.sleep(0.1)  # let readers open the array before the writer starts growing it
        writer_proc.start()

        while any(p.is_alive() for p in procs):
            drain()
            time.sleep(0.002)
        drain()  # final drain: a process can exit right after its last put()

    for p in procs:
        p.join()
    for p in procs:
        assert p.exitcode == 0

    console.print(
        f"\n[bold green]OK[/bold green]: {NREADERS} readers followed {NBATCHES} batches of "
        f"{ROWS_PER_BATCH} rows each, grown with pure SWMR (no locking)."
    )

    blosc2.remove_urlpath(URLPATH)
