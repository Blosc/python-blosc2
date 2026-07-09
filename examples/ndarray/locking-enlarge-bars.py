#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Same scenario and same `rich` progress-bar rendering as
# `swmr-enlarge-bars.py` -- one writer growing a disk-based array while
# several readers follow along -- but with `locking=True` everywhere and
# each batch bracketed in `holding_lock()`, instead of plain unlocked
# SWMR. Run the two side by side: this file never raises the transient
# read error `swmr-enlarge-bars.py` has to retry around, and its reader
# code is visibly simpler because of it -- no "settled vs. just-resized"
# trailing trick, no retry loop. See
# doc/getting_started/sharing_across_processes.rst, "Locking" and
# "Atomic multi-operation blocks with holding_lock()", for the contract
# this relies on.
#
# What `holding_lock()` buys here specifically: a plain resize() + a
# slice write are two separate locked operations, so without
# holding_lock() a locked reader could still observe the array right
# after the resize but before the fill -- grown, but uninitialized,
# exactly the gap `swmr-enlarge-bars.py`'s "settled" trick works around.
# Bracketing both in one holding_lock() block makes the whole batch
# atomic to other locked handles: a reader only ever sees a batch fully
# there or not there at all, so it can safely trust and verify anything
# reader.refresh() shows it, immediately, with no lag and no retries.
#
# The trade-off is contention, not correctness: the writer's exclusive
# lock and the readers' shared locks now actually serialize against each
# other, so -- unlike the lock-free SWMR version -- readers can measurably
# slow the writer down (and vice versa). Watch the writer's bar here vs.
# in `swmr-enlarge-bars.py` for the same array size to see it.
#
# As in `swmr-enlarge-bars.py`, the reported MB/s or GB/s is not real
# disk bandwidth: every batch is a constant fill value (trivially
# compressible), so the numbers reflect compression/decompression CPU
# work plus lock-wait time, not disk I/O.

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

URLPATH = "locking-enlarge-bars.b2nd"
NREADERS = 3
NBATCHES = 400
ROWS_PER_BATCH = 10000
NCOLS = 256
DTYPE = np.int64
ITEMSIZE = np.dtype(DTYPE).itemsize
BYTES_PER_BATCH = ROWS_PER_BATCH * NCOLS * ITEMSIZE
TOTAL_BYTES = NBATCHES * BYTES_PER_BATCH
WRITER_DELAY = 0.0012  # seconds between batches


def writer(queue):
    """The single writer. `holding_lock()` makes each batch's resize +
    fill atomic to other locked handles -- see module docstring."""
    arr = blosc2.open(URLPATH, mode="a", locking=True)
    for i in range(NBATCHES):
        with arr.holding_lock():
            start = arr.shape[0]
            arr.resize((start + ROWS_PER_BATCH, NCOLS))
            arr[start : start + ROWS_PER_BATCH, :] = i
        queue.put(("writer", (i + 1) * BYTES_PER_BATCH))
        time.sleep(WRITER_DELAY)
    # No "done" flag needed here, unlike swmr-enlarge-bars.py: readers can
    # tell completion from shape alone -- see reader() below.


def reader(rank, queue):
    """Opened once, before the writer starts; never reopened. Unlike
    the SWMR version, whatever shape/data this reader observes is
    always fully committed -- no lag, no retries needed."""
    arr = blosc2.open(URLPATH, mode="r", locking=True)
    expected_rows = NBATCHES * ROWS_PER_BATCH
    poll_delay = 0.005 + rank * 0.005  # readers 0, 1, 2 poll at 5/10/15ms

    def verify_up_to(rows):
        nonlocal verified
        if rows <= verified:
            return
        block = np.asarray(arr[verified:rows, :])
        for offset in range(0, rows - verified, ROWS_PER_BATCH):
            batch_idx = (verified + offset) // ROWS_PER_BATCH
            rows_block = block[offset : offset + ROWS_PER_BATCH]
            assert np.all(rows_block == batch_idx), (
                f"reader {rank} batch {batch_idx} wrong or torn -- locking should make this impossible"
            )
        verified = rows
        queue.put((f"reader{rank}", verified * NCOLS * ITEMSIZE))

    # Unlike swmr-enlarge-bars.py, shape alone is a trustworthy completion
    # signal here: holding_lock() made every batch's resize-plus-fill
    # atomic, so `shape == expected_rows` can only be observed *after* the
    # last batch's fill has landed too -- there is no "grown but not yet
    # filled" gap to guard against, so no separate "done" flag is needed.
    verified = 0
    while verified < expected_rows:
        arr.refresh()
        verify_up_to(arr.shape[0])
        if verified < expected_rows:
            time.sleep(poll_delay)

    assert arr.shape == (expected_rows, NCOLS), f"reader {rank} final shape {arr.shape} is wrong"


if __name__ == "__main__":
    console = Console()
    console.rule(
        "[bold]Locked multi-writer-safe growth (single writer here) -- holding_lock() per batch[/bold]"
    )

    blosc2.remove_urlpath(URLPATH)
    blosc2.zeros(
        (0, NCOLS),
        dtype=DTYPE,
        urlpath=URLPATH,
        mode="w",
        locking=True,
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
    writer_task = progress.add_task("[bold cyan]writer   (appending, locked)", total=TOTAL_BYTES)
    reader_tasks = {
        rank: progress.add_task(f"[green]reader {rank} (verifying, locked)", total=TOTAL_BYTES)
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
        f"{ROWS_PER_BATCH} rows each, grown with locking -- zero torn reads, zero retries."
    )

    blosc2.remove_urlpath(URLPATH)
