#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Tests for the opt-in cross-process file locking for disk-based containers
# (the `locking` storage parameter, backed by a `.b2lock` sidecar file).

import os
import subprocess
import sys
import time

import numpy as np
import pytest

import blosc2

NCHUNKS = 10
CHUNK_NITEMS = 1_000
DTYPE = np.dtype(np.int64)

pytestmark = pytest.mark.skipif(blosc2.IS_WASM, reason="no file locking on wasm32")


def create_schunk(urlpath, contiguous, locking):
    storage = {"contiguous": contiguous, "urlpath": str(urlpath), "locking": locking}
    schunk = blosc2.SChunk(
        chunksize=CHUNK_NITEMS * DTYPE.itemsize,
        cparams={"typesize": DTYPE.itemsize},
        **storage,
    )
    for nchunk in range(NCHUNKS):
        data = np.arange(nchunk * CHUNK_NITEMS, (nchunk + 1) * CHUNK_NITEMS, dtype=DTYPE)
        schunk.append_data(data)
    return schunk


def sidecar(urlpath, contiguous):
    return urlpath / ".b2lock" if not contiguous else urlpath.with_suffix(urlpath.suffix + ".b2lock")


@pytest.mark.parametrize("contiguous", [False, True])
def test_sidecar_lifecycle(tmp_path, contiguous):
    urlpath = tmp_path / "schunk-locked.b2frame"
    schunk = create_schunk(urlpath, contiguous, locking=True)
    assert schunk.locking
    # The sidecar appears with the first locked operation
    assert sidecar(urlpath, contiguous).exists()
    # ... and ordinary operations keep working
    for nchunk in range(NCHUNKS):
        data = np.frombuffer(schunk.decompress_chunk(nchunk), dtype=DTYPE)
        assert data[0] == nchunk * CHUNK_NITEMS
    del schunk
    blosc2.remove_urlpath(str(urlpath))
    assert not urlpath.exists()
    assert not sidecar(urlpath, contiguous).exists()


@pytest.mark.parametrize("contiguous", [False, True])
def test_no_sidecar_by_default(tmp_path, contiguous):
    urlpath = tmp_path / "schunk-unlocked.b2frame"
    schunk = create_schunk(urlpath, contiguous, locking=False)
    assert not schunk.locking
    schunk.update_special(0, blosc2.SpecialValue.UNINIT)
    assert not sidecar(urlpath, contiguous).exists()
    del schunk
    blosc2.remove_urlpath(str(urlpath))


def test_two_handles_coherent(tmp_path):
    # The Python twin of c-blosc2's examples/file-locking.c: a mutation through
    # one locked handle is picked up coherently by another one
    urlpath = tmp_path / "schunk-shared.b2frame"
    create_schunk(urlpath, contiguous=False, locking=True)

    h1 = blosc2.open(str(urlpath), mode="a", locking=True)
    h2 = blosc2.open(str(urlpath), locking=True)
    assert h1.locking
    assert h2.locking

    # Warm h2's view of the frame
    old_chunk = h2.get_chunk(0)
    assert len(old_chunk) > 0

    # h1 evicts chunk 0 behind h2's back
    h1.update_special(0, blosc2.SpecialValue.UNINIT)

    # h2 re-syncs: it sees the 32-byte special chunk, and it stays readable
    assert len(h2.get_chunk(0)) == 32
    assert len(h2.decompress_chunk(0)) == CHUNK_NITEMS * DTYPE.itemsize
    # An untouched chunk still reads back fine
    data = np.frombuffer(h2.decompress_chunk(1), dtype=DTYPE)
    assert data[0] == CHUNK_NITEMS

    blosc2.remove_urlpath(str(urlpath))


def test_vlmeta_two_handles_coherent(tmp_path):
    # Regression test for c-blosc2's blosc2_vlmeta_exists() staleness poll:
    # a vlmetalayer added or deleted through one locked handle must be
    # reflected in another handle without any data access in between
    urlpath = tmp_path / "schunk-vlmeta.b2frame"
    create_schunk(urlpath, contiguous=True, locking=True)

    h1 = blosc2.open(str(urlpath), mode="a", locking=True)
    h2 = blosc2.open(str(urlpath), mode="a", locking=True)
    assert "foo" not in h2.vlmeta

    h1.vlmeta["foo"] = "bar"
    assert "foo" in h2.vlmeta
    assert h2.vlmeta["foo"] == "bar"

    h1.vlmeta["foo"] = "baz"  # update flows too
    assert h2.vlmeta["foo"] == "baz"

    del h1.vlmeta["foo"]
    assert "foo" not in h2.vlmeta

    blosc2.remove_urlpath(str(urlpath))


def test_ndarray_locking(tmp_path):
    urlpath = tmp_path / "array-locked.b2nd"
    a = blosc2.full((100, 100), 3, urlpath=str(urlpath), locking=True, mode="w")
    assert np.all(a[10:20, 10:20] == 3)
    del a

    b = blosc2.open(str(urlpath), mode="a", locking=True)
    assert isinstance(b, blosc2.NDArray)
    b[0:10, 0:10] = np.ones((10, 10), dtype=b.dtype)
    assert np.all(b[0, 0:10] == 1)
    assert np.all(b[50] == 3)

    blosc2.remove_urlpath(str(urlpath))


def test_ndarray_holding_lock(tmp_path):
    urlpath = tmp_path / "array-holding-lock.b2nd"
    a = blosc2.zeros((10,), urlpath=str(urlpath), locking=True, mode="w")
    with a.holding_lock():
        a[0] = a[0] + 1
        # Nests on the same lock as the underlying schunk's holding_lock();
        # would deadlock if arr.holding_lock() acquired a separate lock.
        with a.schunk.holding_lock():
            a[1] = a[1] + 1
    assert a[0] == 1
    assert a[1] == 1
    del a

    # No-op on a handle without locking enabled, same as SChunk.holding_lock().
    unlocked_path = tmp_path / "array-no-lock.b2nd"
    b = blosc2.zeros((10,), urlpath=str(unlocked_path), mode="w")
    with b.holding_lock():
        b[0] = 5
    assert b[0] == 5

    blosc2.remove_urlpath(str(urlpath))
    blosc2.remove_urlpath(str(unlocked_path))


def test_locking_validation(tmp_path):
    urlpath = tmp_path / "schunk-valid.b2frame"
    # locking together with mmap_mode is rejected
    with pytest.raises(ValueError, match="mmap_mode"):
        blosc2.SChunk(chunksize=1000, urlpath=str(urlpath), mmap_mode="w+", locking=True)
    create_schunk(urlpath, contiguous=True, locking=False)
    with pytest.raises(ValueError, match="mmap_mode"):
        blosc2.open(str(urlpath), mmap_mode="r", locking=True)
    # locking without a urlpath (in-memory) is rejected
    with pytest.raises(ValueError, match="urlpath"):
        blosc2.SChunk(chunksize=1000, locking=True)
    blosc2.remove_urlpath(str(urlpath))


WRITER_SCRIPT = """
import sys
import numpy as np
import blosc2

urlpath, nchunks, chunk_nitems, iters = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
schunk = blosc2.open(urlpath, mode="a", locking=True)
for i in range(iters):
    nchunk = i % nchunks
    if i % 2 == 0:
        # Evict (truncates the chunk file of a sparse frame)
        schunk.update_special(nchunk, blosc2.SpecialValue.UNINIT)
    else:
        # Refill with a regular chunk
        data = np.full(chunk_nitems, i, dtype=np.int64)
        chunk = blosc2.compress2(data, typesize=data.dtype.itemsize)
        schunk.update_chunk(nchunk, chunk)
sys.exit(0)
"""


def test_cross_process_hammer(tmp_path):
    # A writer process keeps evicting/refilling chunks while this process reads
    # all of them; with locking on every handle, no read may ever fail
    urlpath = tmp_path / "schunk-hammer.b2frame"
    create_schunk(urlpath, contiguous=False, locking=True)

    iters = 500
    writer = subprocess.Popen(
        [sys.executable, "-c", WRITER_SCRIPT, str(urlpath), str(NCHUNKS), str(CHUNK_NITEMS), str(iters)]
    )
    try:
        reader = blosc2.open(str(urlpath), locking=True)
        nreads = 0
        deadline = time.monotonic() + 120  # safety net against a hung writer
        while writer.poll() is None:
            assert time.monotonic() < deadline, "writer process did not finish in time"
            for nchunk in range(NCHUNKS):
                assert len(reader.get_chunk(nchunk)) > 0
                assert len(reader.decompress_chunk(nchunk)) == CHUNK_NITEMS * DTYPE.itemsize
                nreads += 2
        # One final sweep after the writer is done
        for nchunk in range(NCHUNKS):
            assert len(reader.get_chunk(nchunk)) > 0
    finally:
        if writer.poll() is None:
            writer.kill()
        writer.wait()

    assert writer.returncode == 0, f"writer process failed with exit code {writer.returncode}"
    assert nreads > 0
    blosc2.remove_urlpath(str(urlpath))


# ---------------------------------------------------------------------------
# Multi-writer hammer tests: several writer *processes*, not just one,
# mutating the same on-disk container concurrently under locking=True.
# ---------------------------------------------------------------------------

APPEND_WRITER_SCRIPT = """
import sys
import numpy as np
import blosc2

urlpath, writer_id, chunk_nitems, iters = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
schunk = blosc2.open(urlpath, mode="a", locking=True)
for i in range(iters):
    # Tag every appended chunk with (writer_id, i) so ownership is
    # unambiguous no matter how the appends from different writers interleave
    data = np.full(chunk_nitems, writer_id * 1_000_000 + i, dtype=np.int64)
    schunk.append_data(data)
sys.exit(0)
"""


def test_cross_process_multiwriter_append(tmp_path):
    # Several writer processes append through their own locking=True handle
    # concurrently, each with a distinguishable signature; the final schunk
    # must contain the exact union of everyone's chunks, with each chunk's
    # content intact (no interleaved/torn chunk, no lost append).
    urlpath = tmp_path / "schunk-multiwriter-append.b2frame"
    nwriters = 4
    iters = 40
    schunk = create_schunk(urlpath, contiguous=False, locking=True)
    base_nchunks = schunk.nchunks
    del schunk

    writers = [
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                APPEND_WRITER_SCRIPT,
                str(urlpath),
                str(wid),
                str(CHUNK_NITEMS),
                str(iters),
            ]
        )
        for wid in range(nwriters)
    ]
    deadline = time.monotonic() + 180
    for w in writers:
        remaining = deadline - time.monotonic()
        assert remaining > 0, "writer processes did not finish in time"
        w.wait(timeout=remaining)
    assert all(w.returncode == 0 for w in writers), "a writer process failed"

    reader = blosc2.open(str(urlpath), locking=True)
    assert reader.nchunks == base_nchunks + nwriters * iters

    seen = {wid: set() for wid in range(nwriters)}
    for nchunk in range(base_nchunks, reader.nchunks):
        data = np.frombuffer(reader.decompress_chunk(nchunk), dtype=DTYPE)
        value = int(data[0])
        assert np.all(data == value), f"chunk {nchunk} mixes values from more than one writer"
        wid, i = divmod(value, 1_000_000)
        seen[wid].add(i)

    for wid in range(nwriters):
        assert seen[wid] == set(range(iters)), f"writer {wid} lost or duplicated appends"

    blosc2.remove_urlpath(str(urlpath))


UPDATE_WRITER_SCRIPT = """
import sys
import numpy as np
import blosc2

urlpath, writer_id, nwriters, chunk_nitems, iters = (
    sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
)
schunk = blosc2.open(urlpath, mode="a", locking=True)
# Each writer owns a disjoint set of chunks, so a mixed/torn chunk can only
# come from the update path racing itself across processes, not from two
# writers targeting the same chunk
owned = [nc for nc in range(schunk.nchunks) if nc % nwriters == writer_id]
for i in range(iters):
    for nchunk in owned:
        data = np.full(chunk_nitems, writer_id * 1_000_000 + i, dtype=np.int64)
        chunk = blosc2.compress2(data, typesize=data.dtype.itemsize)
        schunk.update_chunk(nchunk, chunk)
sys.exit(0)
"""


def test_cross_process_multiwriter_update(tmp_path):
    # Several writer processes repeatedly update *disjoint* chunks of the
    # same schunk through their own locking=True handle, while this process
    # samples every chunk concurrently: no chunk may ever be observed
    # torn/mixed, and after all writers exit each chunk must hold its
    # owner's last-written value.
    urlpath = tmp_path / "schunk-multiwriter-update.b2frame"
    nwriters = 4
    iters = 60
    schunk = create_schunk(urlpath, contiguous=False, locking=True)
    nchunks = schunk.nchunks
    del schunk
    assert nchunks >= nwriters  # every writer owns at least one chunk

    writers = [
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                UPDATE_WRITER_SCRIPT,
                str(urlpath),
                str(wid),
                str(nwriters),
                str(CHUNK_NITEMS),
                str(iters),
            ]
        )
        for wid in range(nwriters)
    ]
    try:
        reader = blosc2.open(str(urlpath), locking=True)
        deadline = time.monotonic() + 180
        while any(w.poll() is None for w in writers):
            assert time.monotonic() < deadline, "writer processes did not finish in time"
            for nchunk in range(nchunks):
                data = np.frombuffer(reader.decompress_chunk(nchunk), dtype=DTYPE)
                # Before its owner's first update, a chunk still holds its
                # original ramp from create_schunk() -- not a torn write
                initial_ramp = np.arange(nchunk * CHUNK_NITEMS, (nchunk + 1) * CHUNK_NITEMS, dtype=DTYPE)
                if np.array_equal(data, initial_ramp):
                    continue
                assert np.all(data == data[0]), (
                    f"chunk {nchunk} observed torn/mixed under concurrent writers"
                )
    finally:
        for w in writers:
            if w.poll() is None:
                w.kill()
            w.wait()
    assert all(w.returncode == 0 for w in writers), "a writer process failed"

    for nchunk in range(nchunks):
        wid = nchunk % nwriters
        data = np.frombuffer(reader.decompress_chunk(nchunk), dtype=DTYPE)
        expected = wid * 1_000_000 + (iters - 1)
        assert np.all(data == expected), f"chunk {nchunk} final value mismatch"

    blosc2.remove_urlpath(str(urlpath))


OPEN_RACE_UPDATE_WRITER_SCRIPT = """
import sys
import numpy as np
import blosc2

urlpath, writer_id, chunk_nitems, iters = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
schunk = blosc2.open(urlpath, mode="a", locking=True)
nchunk = writer_id % schunk.nchunks
rng = np.random.default_rng(writer_id)
for i in range(iters):
    if i % 2 == 0:
        # Shrinks the frame (truncates the chunk to a special-value marker)
        schunk.update_special(nchunk, blosc2.SpecialValue.UNINIT)
    else:
        # Random (near-incompressible) content, so the compressed size --
        # and the resulting frame layout -- actually varies update to
        # update, instead of settling on one compressed size that never
        # moves anything (constant-fill data compresses to the same size
        # every time and stops exercising the layout-shift race below)
        data = rng.integers(0, np.iinfo(np.int64).max, size=chunk_nitems, dtype=np.int64)
        chunk = blosc2.compress2(data, typesize=data.dtype.itemsize)
        schunk.update_chunk(nchunk, chunk)
sys.exit(0)
"""

OPEN_RACE_OPENER_SCRIPT = """
import sys
import blosc2

urlpath, iters = sys.argv[1], int(sys.argv[2])
for _ in range(iters):
    schunk = blosc2.open(urlpath, mode="a", locking=True)
    del schunk
sys.exit(0)
"""


def test_cross_process_open_race_under_update(tmp_path):
    # A fresh open() must never fail just because a concurrent writer is
    # mid-update, even on a fixed-size container where nothing is growing:
    # an in-place chunk update whose compressed size differs from before can
    # still rewrite the on-disk frame layout, racing an opener's unlocked
    # bootstrap read the same way a growing append does.
    #
    # This is a narrow, timing-dependent race: under c-blosc2 commit
    # 87a3af7d, `blosc2_schunk_open_offset_udio()`'s retry-on-race logic
    # (from the open-vs-growth fix, `fa742207`) only covered the *first*
    # bootstrap read -- not the second, force_refresh re-read done under the
    # freshly acquired lock (which fires on nearly every open of a
    # previously-mutated frame). That gap was observed directly (a
    # `RuntimeError` from `blosc2_schunk_open_offset` returning NULL) via
    # ad hoc reproduction while building this test, but the exact
    # interleaving needed is sensitive to system load/caching and does not
    # reproduce on every run even on unfixed code -- treat this as a stress
    # test for the concurrent open+update path in general, not a guaranteed
    # trip wire for this one bug.
    urlpath = tmp_path / "schunk-open-race-update.b2frame"
    nwriters = 4
    nopeners = 4
    iters = 150
    # Contiguous (single-file) frames only: frame_from_file_offset()'s
    # file-boundary check that this test targets is gated by `if (!sframe)`
    # in c-blosc2, so sparse (directory-based) frames never exercise it.
    schunk = create_schunk(urlpath, contiguous=True, locking=True)
    del schunk

    writers = [
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                OPEN_RACE_UPDATE_WRITER_SCRIPT,
                str(urlpath),
                str(wid),
                str(CHUNK_NITEMS),
                str(iters),
            ]
        )
        for wid in range(nwriters)
    ]
    openers = [
        subprocess.Popen([sys.executable, "-c", OPEN_RACE_OPENER_SCRIPT, str(urlpath), str(iters)])
        for _ in range(nopeners)
    ]
    procs = writers + openers
    deadline = time.monotonic() + 180
    for p in procs:
        remaining = deadline - time.monotonic()
        assert remaining > 0, "writer/opener processes did not finish in time"
        p.wait(timeout=remaining)
    assert all(p.returncode == 0 for p in procs), (
        f"a writer or opener process failed: {[p.returncode for p in procs]}"
    )

    blosc2.remove_urlpath(str(urlpath))


NDARRAY_WRITER_SCRIPT = """
import sys
import numpy as np
import blosc2

urlpath, writer_id, rows_per_writer, ncols, iters = (
    sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
)
w = blosc2.open(urlpath, mode="a", locking=True)
row0 = writer_id * rows_per_writer
for i in range(iters):
    # Bracket the resize + fill so another handle never observes a grown
    # shape with not-yet-written (still-zero) rows in this writer's region
    with w.schunk.holding_lock():
        needed = row0 + rows_per_writer
        if w.shape[0] < needed:
            w.resize((needed, ncols))
        value = writer_id * 1_000_000 + i
        w[row0 : row0 + rows_per_writer, :] = np.full((rows_per_writer, ncols), value, dtype=np.int64)
sys.exit(0)
"""


def test_cross_process_multiwriter_ndarray(tmp_path):
    # N writer processes each resize() + fill a disjoint row region of the
    # same on-disk NDArray under locking=True, exercising the b2nd metalayer
    # path (not just raw schunk chunks). The final array must equal the
    # union of every writer's last-written region.
    urlpath = str(tmp_path / "array-multiwriter.b2nd")
    nwriters = 4
    rows_per_writer = 3
    ncols = 10
    iters = 20

    a = blosc2.zeros(
        (0, ncols),
        dtype=np.int64,
        chunks=(rows_per_writer, ncols),
        blocks=(rows_per_writer, ncols),
        urlpath=urlpath,
        mode="w",
        locking=True,
    )
    del a

    writers = [
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                NDARRAY_WRITER_SCRIPT,
                urlpath,
                str(wid),
                str(rows_per_writer),
                str(ncols),
                str(iters),
            ]
        )
        for wid in range(nwriters)
    ]
    deadline = time.monotonic() + 180
    for w in writers:
        remaining = deadline - time.monotonic()
        assert remaining > 0, "writer processes did not finish in time"
        w.wait(timeout=remaining)
    assert all(w.returncode == 0 for w in writers), "a writer process failed"

    reader = blosc2.open(urlpath, mode="r", locking=True)
    assert reader.shape == (nwriters * rows_per_writer, ncols)
    result = reader[:, :]
    for wid in range(nwriters):
        row0 = wid * rows_per_writer
        expected = wid * 1_000_000 + (iters - 1)
        region = result[row0 : row0 + rows_per_writer, :]
        assert np.all(region == expected), f"writer {wid}'s region does not hold its last-written value"

    blosc2.remove_urlpath(urlpath)


NDARRAY_APPEND_WRITER_SCRIPT = """
import sys
import numpy as np
import blosc2

urlpath, writer_id, appends_per_writer, items_per_append = (
    sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
)
w = blosc2.open(urlpath, mode="a", locking=True)
for i in range(appends_per_writer):
    tag = writer_id * 1_000_000 + i
    batch = np.full(items_per_append, tag, dtype=np.int64)
    # append() is refresh + resize + a slice write -- not one atomic
    # operation on its own. Bracketing the whole call is required so no
    # other writer's growth is invisible when this one computes where to
    # place its own batch.
    with w.holding_lock():
        w.append(batch)
sys.exit(0)
"""


def test_cross_process_multiwriter_ndarray_append(tmp_path):
    # Regression test for a real data-loss bug (found 2026-07-08 while
    # building examples/ndarray/mwmr-enlarge.py): NDArray.append() read its
    # cached (unrefreshed) shape to compute the resize target. Under
    # concurrent growth, resize()'s own internal refresh would then see a
    # *larger* true shape than the stale target just computed, so
    # b2nd_resize() took the shrink path and deleted the chunks other
    # writers had just appended -- even with holding_lock() wrapping the
    # call, since holding_lock() only serializes the operations inside the
    # block, it does not retroactively fix a value read from a stale cache.
    # Fixed by refreshing before reading the old size (ndarray.py, append()).
    #
    # N writer processes each append() several uniquely-tagged batches to
    # the same 1-D array. Every atomic append() inserts a block of identical
    # values, so regardless of how the writers' appends interleaved on
    # disk, the final array must be an exact concatenation of such blocks:
    # final length matches exactly, every block is untorn, and the multiset
    # of tags is exactly what was appended -- nothing lost, nothing
    # duplicated.
    urlpath = str(tmp_path / "array-multiwriter-append.b2nd")
    nwriters = 4
    appends_per_writer = 30
    items_per_append = 25

    a = blosc2.zeros(
        (0,),
        dtype=np.int64,
        chunks=(items_per_append * 4,),
        blocks=(items_per_append,),
        urlpath=urlpath,
        mode="w",
        locking=True,
    )
    del a

    writers = [
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                NDARRAY_APPEND_WRITER_SCRIPT,
                urlpath,
                str(wid),
                str(appends_per_writer),
                str(items_per_append),
            ]
        )
        for wid in range(nwriters)
    ]
    deadline = time.monotonic() + 180
    for w in writers:
        remaining = deadline - time.monotonic()
        assert remaining > 0, "writer processes did not finish in time"
        w.wait(timeout=remaining)
    assert all(w.returncode == 0 for w in writers), "a writer process failed"

    reader = blosc2.open(urlpath, mode="r", locking=True)
    expected_len = nwriters * appends_per_writer * items_per_append
    assert reader.shape[0] == expected_len, (
        f"final length {reader.shape[0]} != expected {expected_len} -- lost or duplicated an append"
    )

    blocks = reader[:].reshape(-1, items_per_append)
    tags = []
    for block in blocks:
        assert np.all(block == block[0]), f"torn append: block mixes values {np.unique(block)}"
        tags.append(int(block[0]))

    expected_tags = sorted(wid * 1_000_000 + i for wid in range(nwriters) for i in range(appends_per_writer))
    assert sorted(tags) == expected_tags, (
        "append tags don't match: a batch was lost, duplicated, or corrupted"
    )

    blosc2.remove_urlpath(urlpath)


def test_cross_process_multiwriter_ndarray_append_sparse_nonaligned(tmp_path):
    # Same bug class as test_cross_process_multiwriter_ndarray_append, but
    # on the two physical layouts that test didn't touch: sparse storage
    # (contiguous=False, each chunk its own file, a different rewrite path
    # than a single-file frame) and a starting length that is *not* a
    # multiple of the chunk size (so the first append from every writer has
    # to fill a partial chunk before any full one -- different chunk-boundary
    # arithmetic than always starting from a clean 0).
    urlpath = str(tmp_path / "array-multiwriter-append-sparse.b2nd")
    nwriters = 4
    appends_per_writer = 20
    items_per_append = 25
    prefix_len = 37  # not a multiple of items_per_append

    a = blosc2.zeros(
        (prefix_len,),
        dtype=np.int64,
        chunks=(items_per_append * 4,),
        blocks=(items_per_append,),
        urlpath=urlpath,
        mode="w",
        locking=True,
        contiguous=False,
    )
    prefix = np.arange(prefix_len, dtype=np.int64)
    a[:] = prefix
    del a

    writers = [
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                NDARRAY_APPEND_WRITER_SCRIPT,
                urlpath,
                str(wid),
                str(appends_per_writer),
                str(items_per_append),
            ]
        )
        for wid in range(nwriters)
    ]
    deadline = time.monotonic() + 180
    for w in writers:
        remaining = deadline - time.monotonic()
        assert remaining > 0, "writer processes did not finish in time"
        w.wait(timeout=remaining)
    assert all(w.returncode == 0 for w in writers), "a writer process failed"

    reader = blosc2.open(urlpath, mode="r", locking=True)
    expected_len = prefix_len + nwriters * appends_per_writer * items_per_append
    assert reader.shape[0] == expected_len, (
        f"final length {reader.shape[0]} != expected {expected_len} -- lost or duplicated an append"
    )
    assert np.array_equal(reader[:prefix_len], prefix), (
        "pre-existing prefix was corrupted by concurrent appends"
    )

    blocks = reader[prefix_len:].reshape(-1, items_per_append)
    tags = []
    for block in blocks:
        assert np.all(block == block[0]), f"torn append: block mixes values {np.unique(block)}"
        tags.append(int(block[0]))

    expected_tags = sorted(wid * 1_000_000 + i for wid in range(nwriters) for i in range(appends_per_writer))
    assert sorted(tags) == expected_tags, (
        "append tags don't match: a batch was lost, duplicated, or corrupted"
    )

    blosc2.remove_urlpath(urlpath)


NDARRAY_2D_GROW_WRITER_SCRIPT = """
import sys
import numpy as np
import blosc2

urlpath, writer_id, appends_per_writer, rows_per_append, ncols = (
    sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
)
w = blosc2.open(urlpath, mode="a", locking=True)
for i in range(appends_per_writer):
    tag = writer_id * 1_000_000 + i
    block = np.full((rows_per_append, ncols), tag, dtype=np.int64)
    # append() only supports 1-D arrays (ndarray.py: "append() is only
    # supported for 1-D arrays"); growing an N-D array has no library
    # convenience, so callers must refresh, resize, and fill by hand. This
    # mirrors -- by hand, at the Python level -- exactly the sequence
    # append() now does internally for 1-D since the 2026-07-08 fix.
    with w.holding_lock():
        w.refresh()
        old_rows = w.shape[0]
        new_rows = old_rows + rows_per_append
        w.resize((new_rows, ncols))
        w[old_rows:new_rows, :] = block
sys.exit(0)
"""


def test_cross_process_multiwriter_ndarray_2d_grow(tmp_path):
    # NDArray.append() is 1-D only, so this exercises the *general* pattern
    # users must follow to grow an N-D array safely under concurrent
    # writers: refresh() then resize() then fill, all inside holding_lock().
    # Confirms the fix principle behind the append() bug (a stale read
    # before a lock-protected composite op silently discards concurrent
    # growth) generalizes correctly to N-D arrays and to manually-driven
    # resize() calls, not just the library's own 1-D append() path.
    urlpath = str(tmp_path / "array-multiwriter-2d-grow.b2nd")
    nwriters = 4
    appends_per_writer = 20
    rows_per_append = 5
    ncols = 8

    a = blosc2.zeros(
        (0, ncols),
        dtype=np.int64,
        chunks=(rows_per_append * 4, ncols),
        blocks=(rows_per_append, ncols),
        urlpath=urlpath,
        mode="w",
        locking=True,
    )
    del a

    writers = [
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                NDARRAY_2D_GROW_WRITER_SCRIPT,
                urlpath,
                str(wid),
                str(appends_per_writer),
                str(rows_per_append),
                str(ncols),
            ]
        )
        for wid in range(nwriters)
    ]
    deadline = time.monotonic() + 180
    for w in writers:
        remaining = deadline - time.monotonic()
        assert remaining > 0, "writer processes did not finish in time"
        w.wait(timeout=remaining)
    assert all(w.returncode == 0 for w in writers), "a writer process failed"

    reader = blosc2.open(urlpath, mode="r", locking=True)
    expected_rows = nwriters * appends_per_writer * rows_per_append
    assert reader.shape == (expected_rows, ncols), (
        f"final shape {reader.shape} != expected {(expected_rows, ncols)} -- lost or duplicated a block"
    )

    blocks = reader[:, :].reshape(-1, rows_per_append, ncols)
    tags = []
    for block in blocks:
        assert np.all(block == block[0, 0]), f"torn block: mixes values {np.unique(block)}"
        tags.append(int(block[0, 0]))

    expected_tags = sorted(wid * 1_000_000 + i for wid in range(nwriters) for i in range(appends_per_writer))
    assert sorted(tags) == expected_tags, (
        "growth tags don't match: a block was lost, duplicated, or corrupted"
    )

    blosc2.remove_urlpath(urlpath)


NDARRAY_SHRINK_WRITER_SCRIPT = """
import sys
import numpy as np
import blosc2

urlpath, writer_id, appends_per_writer, items_per_append = (
    sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
)
w = blosc2.open(urlpath, mode="a", locking=True)
for i in range(appends_per_writer):
    tag = writer_id * 1_000_000 + i
    batch = np.full(items_per_append, tag, dtype=np.int64)
    with w.holding_lock():
        w.append(batch)
sys.exit(0)
"""


def test_cross_process_shrink_after_multiwriter_growth(tmp_path):
    # Shrink is never exercised by any of the other multi-writer tests (they
    # only grow). This checks the shrink path -- b2nd_resize()'s
    # shrink_shape() -> blosc2_schunk_delete_chunk() -- correctly drops
    # exactly the tail chunks after growth from several concurrent writers
    # has produced many small, interleaved-order chunks, not the tidy
    # single-writer layout shrink is usually exercised against.
    #
    # Growers run to completion first (sequenced, not racing the shrink) so
    # the outcome is deterministic: whichever tags ended up in the final
    # array's *last* items are exactly the ones a subsequent shrink must
    # drop, and everything before that boundary must survive untouched.
    urlpath = str(tmp_path / "array-shrink-after-growth.b2nd")
    nwriters = 4
    appends_per_writer = 20
    items_per_append = 25

    a = blosc2.zeros(
        (0,),
        dtype=np.int64,
        chunks=(items_per_append * 4,),
        blocks=(items_per_append,),
        urlpath=urlpath,
        mode="w",
        locking=True,
    )
    del a

    writers = [
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                NDARRAY_SHRINK_WRITER_SCRIPT,
                urlpath,
                str(wid),
                str(appends_per_writer),
                str(items_per_append),
            ]
        )
        for wid in range(nwriters)
    ]
    deadline = time.monotonic() + 180
    for w in writers:
        remaining = deadline - time.monotonic()
        assert remaining > 0, "writer processes did not finish in time"
        w.wait(timeout=remaining)
    assert all(w.returncode == 0 for w in writers), "a writer process failed"

    full = blosc2.open(urlpath, mode="a", locking=True)
    full_len = full.shape[0]
    expected_len = nwriters * appends_per_writer * items_per_append
    assert full_len == expected_len, "growth phase itself lost or duplicated a batch"
    before = full[:].copy()

    keep = full_len - (full_len // 3)  # drop the last third
    full.resize((keep,))
    del full

    reader = blosc2.open(urlpath, mode="r", locking=True)
    assert reader.shape == (keep,), f"shrink left shape {reader.shape}, expected ({keep},)"
    assert np.array_equal(reader[:], before[:keep]), (
        "shrink corrupted the surviving prefix instead of only dropping the tail"
    )

    blosc2.remove_urlpath(urlpath)


VLMETA_AND_GROWTH_WRITER_SCRIPT = """
import sys
import numpy as np
import blosc2

urlpath, writer_id, appends_per_writer, items_per_append = (
    sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
)
w = blosc2.open(urlpath, mode="a", locking=True)
key = f"writer_{writer_id}"
for i in range(appends_per_writer):
    tag = writer_id * 1_000_000 + i
    batch = np.full(items_per_append, tag, dtype=np.int64)
    with w.holding_lock():
        w.append(batch)
        # vlmeta and the b2nd shape metalayer both go through the same
        # frame-level metalayer reload on staleness (frame_refresh_if_stale
        # reloads header metalayers *and* trailer vlmetalayers together) --
        # exercise them interleaved, in the same locked block, to check
        # neither write clobbers or desyncs the other.
        w.schunk.vlmeta[key] = str(i).encode()
sys.exit(0)
"""


def test_cross_process_vlmeta_and_growth_interleaved(tmp_path):
    # vlmeta and the b2nd shape metalayer are reloaded together by the same
    # staleness check (see frame_refresh_if_stale() in frame.c), but no
    # existing test exercises a writer mutating both in the same locked
    # block concurrently with other writers doing the same. Verifies growth
    # data integrity (same checks as test_cross_process_multiwriter_ndarray_append)
    # *and* that every writer's final vlmeta value survives correctly.
    urlpath = str(tmp_path / "array-vlmeta-and-growth.b2nd")
    nwriters = 4
    appends_per_writer = 20
    items_per_append = 25

    a = blosc2.zeros(
        (0,),
        dtype=np.int64,
        chunks=(items_per_append * 4,),
        blocks=(items_per_append,),
        urlpath=urlpath,
        mode="w",
        locking=True,
    )
    del a

    writers = [
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                VLMETA_AND_GROWTH_WRITER_SCRIPT,
                urlpath,
                str(wid),
                str(appends_per_writer),
                str(items_per_append),
            ]
        )
        for wid in range(nwriters)
    ]
    deadline = time.monotonic() + 180
    for w in writers:
        remaining = deadline - time.monotonic()
        assert remaining > 0, "writer processes did not finish in time"
        w.wait(timeout=remaining)
    assert all(w.returncode == 0 for w in writers), "a writer process failed"

    reader = blosc2.open(urlpath, mode="r", locking=True)
    expected_len = nwriters * appends_per_writer * items_per_append
    assert reader.shape[0] == expected_len, (
        f"final length {reader.shape[0]} != expected {expected_len} -- lost or duplicated an append"
    )

    blocks = reader[:].reshape(-1, items_per_append)
    tags = sorted(int(block[0]) for block in blocks)
    for block in blocks:
        assert np.all(block == block[0]), f"torn append: block mixes values {np.unique(block)}"
    expected_tags = sorted(wid * 1_000_000 + i for wid in range(nwriters) for i in range(appends_per_writer))
    assert tags == expected_tags, "growth tags don't match: a batch was lost, duplicated, or corrupted"

    for wid in range(nwriters):
        key = f"writer_{wid}"
        assert key in reader.schunk.vlmeta, f"vlmeta key {key} missing"
        assert reader.schunk.vlmeta[key] == str(appends_per_writer - 1).encode(), (
            f"vlmeta {key} does not hold its writer's last-written value"
        )

    blosc2.remove_urlpath(urlpath)


OVERLAPPING_SLICE_WRITER_SCRIPT = """
import sys
import numpy as np
import blosc2

urlpath, value, iters = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
w = blosc2.open(urlpath, mode="a", locking=True)
buf = np.full(w.shape, value, dtype=np.int64)
for i in range(iters):
    w[:, :] = buf
sys.exit(0)
"""


def test_cross_process_overlapping_slice_atomic(tmp_path):
    # Two writer processes repeatedly overwrite the *same* (multi-chunk)
    # NDArray region with their own distinguishable constant value via
    # __setitem__ (b2nd_set_slice_cbuffer at the C level); a locked reader
    # sampling the whole region under holding_lock() must only ever see one
    # writer's complete pass, never a chunk-wise mix of both (the atomicity
    # b2nd_set_slice_cbuffer's exclusive-lock bracket provides -- python
    # inherits it through __setitem__ with no code changes of its own).
    urlpath = str(tmp_path / "array-overlap.b2nd")
    nrows, ncols = 200, 50
    iters = 150

    a = blosc2.zeros(
        (nrows, ncols),
        dtype=np.int64,
        chunks=(nrows // 4, ncols),
        blocks=(nrows // 4, ncols),
        urlpath=urlpath,
        mode="w",
        locking=True,
    )
    del a

    writers = [
        subprocess.Popen(
            [sys.executable, "-c", OVERLAPPING_SLICE_WRITER_SCRIPT, urlpath, str(v), str(iters)]
        )
        for v in (1, 2)
    ]
    try:
        reader = blosc2.open(urlpath, mode="r", locking=True)
        nreads = 0
        deadline = time.monotonic() + 180
        while any(w.poll() is None for w in writers):
            assert time.monotonic() < deadline, "writer processes did not finish in time"
            with reader.schunk.holding_lock():
                data = reader[:, :]
            first = data.flat[0]
            assert first in (0, 1, 2), f"unexpected value {first} in the array"
            assert np.all(data == first), f"observed a mixed (half-applied) slice write: {np.unique(data)}"
            nreads += 1
    finally:
        for w in writers:
            if w.poll() is None:
                w.kill()
            w.wait()

    assert all(w.returncode == 0 for w in writers), "a writer process failed"
    assert nreads > 0
    blosc2.remove_urlpath(urlpath)


# ---------------------------------------------------------------------------
# EmbedStore under locking: transactional writes + key-map re-sync
# ---------------------------------------------------------------------------


def open_estore(path, mode):
    storage = blosc2.Storage(contiguous=True, urlpath=str(path), mode=mode, locking=True)
    return blosc2.EmbedStore(urlpath=str(path), mode=mode, storage=storage)


def test_embed_store_two_handles(tmp_path):
    # A second locked handle follows mutations made through the first one
    path = tmp_path / "shared.b2e"
    h1 = open_estore(path, "w")
    h1["/one"] = np.arange(10)

    h2 = open_estore(path, "a")
    assert set(h2.keys()) == {"/one"}

    h1["/two"] = np.arange(20)
    assert set(h2.keys()) == {"/one", "/two"}
    assert np.array_equal(h2["/two"][:], np.arange(20))

    del h1["/one"]
    assert "/one" not in h2
    # ... and the other direction
    h2["/three"] = np.arange(30)
    assert np.array_equal(h1["/three"][:], np.arange(30))


def test_embed_store_env_locking(tmp_path):
    # BLOSC_LOCKING makes a plain EmbedStore shared, with no locking parameter
    path = str(tmp_path / "env.b2e")
    script = """
import sys
import blosc2
import numpy as np

estore = blosc2.EmbedStore(urlpath=sys.argv[1], mode="w")
estore["/a"] = np.arange(5)
assert estore._shared, "BLOSC_LOCKING did not mark the store as shared"
"""
    env = {**os.environ, "BLOSC_LOCKING": "1"}
    subprocess.run([sys.executable, "-c", script, path], check=True, env=env)
    # Sidecar next to the container proves the C level locked too
    assert os.path.exists(path + ".b2lock")


ESTORE_WRITER = """
import sys
import numpy as np
import blosc2

path, tag, nkeys = sys.argv[1], sys.argv[2], int(sys.argv[3])
storage = blosc2.Storage(contiguous=True, urlpath=path, mode="a", locking=True)
estore = blosc2.EmbedStore(urlpath=path, mode="a", storage=storage)
for i in range(nkeys):
    estore[f"/{tag}/{i}"] = np.arange(i, i + 10)
"""


def test_embed_store_cross_process_writers(tmp_path):
    # Two writer processes adding disjoint keys concurrently, while this
    # process keeps reading: no lost updates, every value must round-trip
    path = tmp_path / "hammer.b2e"
    estore = open_estore(path, "w")
    estore["/seed"] = np.arange(10)

    nkeys = 20
    tags = ("w1", "w2")
    writers = [
        subprocess.Popen([sys.executable, "-c", ESTORE_WRITER, str(path), tag, str(nkeys)]) for tag in tags
    ]
    try:
        nreads = 0
        deadline = time.monotonic() + 120
        while any(w.poll() is None for w in writers):
            assert time.monotonic() < deadline, "writer processes did not finish in time"
            keys = list(estore)
            assert "/seed" in keys
            for key in keys[-3:]:
                node = estore.get(key)
                if node is not None:  # a concurrent delete cannot happen here
                    assert len(node[:]) == 10
                    nreads += 1
    finally:
        for w in writers:
            if w.poll() is None:
                w.kill()
            w.wait()

    assert all(w.returncode == 0 for w in writers), "a writer process failed"
    expected = {"/seed"} | {f"/{tag}/{i}" for tag in tags for i in range(nkeys)}
    assert set(estore.keys()) == expected
    for tag in tags:
        for i in range(nkeys):
            assert np.array_equal(estore[f"/{tag}/{i}"][:], np.arange(i, i + 10))
    assert nreads > 0


# ---------------------------------------------------------------------------
# DictStore (.b2d) under locking: store-wide mutations + map re-sync
# ---------------------------------------------------------------------------


def test_dict_store_two_handles(tmp_path):
    # A second locked handle follows keys added/removed through the first one,
    # both for external leaves (threshold=0 default) and embedded values
    path = str(tmp_path / "shared.b2d")
    h1 = blosc2.DictStore(path, mode="w", locking=True)
    h1["/ext"] = np.arange(100)  # external leaf (above default threshold)

    h2 = blosc2.DictStore(path, mode="a", locking=True)
    assert set(h2.keys()) == {"/ext"}

    h1["/dir/other"] = np.arange(50)
    assert "/dir/other" in h2
    assert np.array_equal(h2["/dir/other"][:], np.arange(50))

    del h1["/ext"]
    assert set(h2.keys()) == {"/dir/other"}
    # ... and the other direction
    h2["/back"] = np.arange(7)
    assert np.array_equal(h1["/back"][:], np.arange(7))
    h1._closed = h2._closed = True  # skip pack-on-close for these handles


def test_dict_store_locking_validation(tmp_path):
    with pytest.raises(ValueError, match="zip"):
        blosc2.DictStore(str(tmp_path / "s.b2z"), mode="w", locking=True)
    with pytest.raises(ValueError, match="mmap_mode"):
        blosc2.DictStore(str(tmp_path / "s.b2d"), mode="r", mmap_mode="r", locking=True)


DSTORE_WRITER = """
import sys
import numpy as np
import blosc2

path, tag, nkeys = sys.argv[1], sys.argv[2], int(sys.argv[3])
# threshold=500: the 800-byte arrays become external leaves, the 40-byte ones embedded
dstore = blosc2.DictStore(path, mode="a", threshold=500, locking=True)
for i in range(nkeys):
    dstore[f"/{tag}/ext{i}"] = np.arange(i, i + 100)
    dstore[f"/{tag}/emb{i}"] = np.arange(5)
dstore._closed = True
"""


def test_dict_store_cross_process_writers(tmp_path):
    # Two writer processes adding disjoint keys concurrently, while this
    # process keeps reading: no lost updates, and directory + maps agree
    path = str(tmp_path / "hammer.b2d")
    dstore = blosc2.DictStore(path, mode="w", threshold=500, locking=True)
    dstore["/seed"] = np.arange(100)

    nkeys = 8
    tags = ("w1", "w2")
    writers = [
        subprocess.Popen([sys.executable, "-c", DSTORE_WRITER, path, tag, str(nkeys)]) for tag in tags
    ]
    try:
        deadline = time.monotonic() + 120
        nreads = 0
        while any(w.poll() is None for w in writers):
            assert time.monotonic() < deadline, "writer processes did not finish in time"
            keys = list(dstore.keys())
            assert "/seed" in keys
            for key in keys:
                if key.endswith("ext3"):
                    node = dstore.get(key)
                    if node is not None:
                        assert len(node[:]) == 100
                        nreads += 1
    finally:
        for w in writers:
            if w.poll() is None:
                w.kill()
            w.wait()

    assert all(w.returncode == 0 for w in writers), "a writer process failed"
    expected = {"/seed"}
    for tag in tags:
        expected |= {f"/{tag}/ext{i}" for i in range(nkeys)}
        expected |= {f"/{tag}/emb{i}" for i in range(nkeys)}
    assert set(dstore.keys()) == expected
    for tag in tags:
        for i in range(nkeys):
            assert np.array_equal(dstore[f"/{tag}/ext{i}"][:], np.arange(i, i + 100))
            assert np.array_equal(dstore[f"/{tag}/emb{i}"][:], np.arange(5))
    dstore._closed = True


# ---------------------------------------------------------------------------
# Growth-SWMR: a reader handle follows a writer process resizing an on-disk
# NDArray, both via implicit re-sync on data access and via explicit
# refresh().  The Python twin of c-blosc2's examples/b2nd/example_growth_swmr.c,
# but cross-process and pinning both the locking and plain (FRAME_LEN-poll)
# SWMR contracts.
# ---------------------------------------------------------------------------

GROWTH_WRITER = """
import sys
import time
import numpy as np
import blosc2

path, locking, steps, rows_per_step, ncols = (
    sys.argv[1], sys.argv[2] == "1", int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
)
w = blosc2.open(path, mode="a", locking=locking)
for _ in range(steps):
    old_rows = w.shape[0]
    new_rows = old_rows + rows_per_step
    w.resize((new_rows, ncols))
    data = np.arange(old_rows * ncols, new_rows * ncols, dtype=np.int64).reshape(rows_per_step, ncols)
    w[old_rows:new_rows, :] = data
    time.sleep(0.05)
"""


@pytest.mark.parametrize("locking", [False, True])
def test_growth_swmr_cross_process(tmp_path, locking):
    urlpath = str(tmp_path / "growth.b2nd")
    rows0, ncols = 5, 10
    steps, rows_per_step = 6, 5
    final_rows = rows0 + steps * rows_per_step

    a = blosc2.zeros(
        (rows0, ncols), dtype=np.int64, chunks=(5, ncols), blocks=(5, ncols), urlpath=urlpath, mode="w"
    )
    del a

    # Two readers opened before the writer starts growing the array: `reader`
    # polls via explicit refresh(), `reader2` is left untouched to prove
    # implicit re-sync on data access (a freshly-opened handle would just
    # read the current on-disk shape, defeating that check)
    reader = blosc2.open(urlpath, mode="r", locking=locking)
    reader2 = blosc2.open(urlpath, mode="r", locking=locking)
    assert reader.shape == (rows0, ncols)
    assert reader2.shape == (rows0, ncols)

    writer = subprocess.Popen(
        [
            sys.executable,
            "-c",
            GROWTH_WRITER,
            urlpath,
            "1" if locking else "0",
            str(steps),
            str(rows_per_step),
            str(ncols),
        ]
    )
    try:
        deadline = time.monotonic() + 60
        seen_rows = [reader.shape[0]]
        while writer.poll() is None:
            assert time.monotonic() < deadline, "writer process did not finish in time"
            reader.refresh()
            seen_rows.append(reader.shape[0])
            time.sleep(0.01)
        writer.wait()
    finally:
        if writer.poll() is None:
            writer.kill()
            writer.wait()
    assert writer.returncode == 0, f"writer process failed with exit code {writer.returncode}"

    # Growth as observed via explicit refresh() must never go backwards
    assert seen_rows == sorted(seen_rows)

    # A final explicit refresh must catch up to the writer's final shape
    reader.refresh()
    assert reader.shape == (final_rows, ncols)
    # The original rows0 rows stay zero (never overwritten); the grown rows
    # hold the flat row-major index the writer filled them with
    expected = np.zeros((final_rows, ncols), dtype=np.int64)
    expected[rows0:, :] = np.arange(rows0 * ncols, final_rows * ncols).reshape(final_rows - rows0, ncols)
    np.testing.assert_array_equal(reader[:, :], expected)

    # reader2 never called refresh(): its cached shape is still stale, and
    # only a data access (not the .shape read above) resyncs it
    assert reader2.shape == (rows0, ncols)
    np.testing.assert_array_equal(reader2[-1, :], expected[-1, :])
    assert reader2.shape == (final_rows, ncols)

    blosc2.remove_urlpath(urlpath)
