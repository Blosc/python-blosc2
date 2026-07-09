#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Tests for growth-SWMR (single writer, multiple readers): a reader handle on
# a disk-based NDArray follows shape changes made through another handle --
# typically another process -- without reopening.

import subprocess
import sys
import time

import numpy as np
import pytest

import blosc2

DTYPE = np.dtype(np.int32)

pytestmark = pytest.mark.skipif(blosc2.IS_WASM, reason="no shared file handles on wasm32")


def create_array(urlpath, contiguous=True, locking=False):
    return blosc2.zeros(
        (10, 10),
        dtype=DTYPE,
        chunks=(5, 10),
        urlpath=str(urlpath),
        contiguous=contiguous,
        locking=locking,
        mode="w",
    )


@pytest.mark.parametrize("contiguous", [True, False])
def test_reader_follows_growth(tmp_path, contiguous):
    urlpath = tmp_path / "growth.b2nd"
    w = create_array(urlpath, contiguous)
    r = blosc2.open(str(urlpath))
    _ = r[0]  # warm the reader's cached state

    w.resize((20, 10))
    w[10:20] = np.arange(100, dtype=DTYPE).reshape(10, 10)

    # Reading the grown region just works; the shape follows too
    np.testing.assert_array_equal(r[15], np.arange(50, 60, dtype=DTYPE))
    assert r.shape == (20, 10)


def test_explicit_refresh(tmp_path):
    urlpath = tmp_path / "refresh.b2nd"
    w = create_array(urlpath)
    r = blosc2.open(str(urlpath))
    # A fresh handle is already current
    assert r.refresh() is False

    # refresh() observes the new shape with no data access involved
    w.resize((20, 10))
    assert r.refresh() is True
    assert r.shape == (20, 10)
    assert r.refresh() is False

    # Shrinking that drops chunks is followed too
    w.resize((5, 10))
    assert r.refresh() is True
    assert r.shape == (5, 10)


def test_setitem_follows_growth(tmp_path):
    urlpath = tmp_path / "setitem.b2nd"
    w = create_array(urlpath)
    r = blosc2.open(str(urlpath), mode="a")
    _ = r[0]

    w.resize((20, 10))
    # Writing beyond the reader's stale shape must re-sync, not raise
    r[15] = np.ones(10, dtype=DTYPE)
    np.testing.assert_array_equal(w[15], np.ones(10, dtype=DTYPE))


def test_refresh_in_memory_noop():
    a = blosc2.zeros((4, 4), dtype=DTYPE)
    assert a.refresh() is False


def test_schunk_explicit_refresh(tmp_path):
    urlpath = tmp_path / "schunk-refresh.b2frame"
    w = blosc2.SChunk(
        chunksize=DTYPE.itemsize * 10,
        cparams={"typesize": DTYPE.itemsize},
        urlpath=str(urlpath),
        mode="w",
    )
    w.append_data(np.arange(10, dtype=DTYPE))
    r = blosc2.open(str(urlpath))
    # A fresh handle is already current
    assert r.refresh() is False

    # refresh() observes the new chunk with no data access involved
    w.append_data(np.arange(10, 20, dtype=DTYPE))
    assert r.refresh() is True
    assert r.nchunks == 2
    assert r.refresh() is False


def test_schunk_refresh_in_memory_noop():
    s = blosc2.SChunk(chunksize=DTYPE.itemsize * 10, cparams={"typesize": DTYPE.itemsize})
    assert s.refresh() is False


def test_locking_detects_same_length_rewrite(tmp_path):
    # A shrink within the last chunk leaves the frame length unchanged (the
    # documented blind spot for unlocked handles); the locking generation
    # counter must still detect it exactly.
    urlpath = tmp_path / "blindspot.b2nd"
    w = blosc2.zeros((10, 10), dtype=DTYPE, chunks=(10, 10), urlpath=str(urlpath), locking=True, mode="w")
    r = blosc2.open(str(urlpath), locking=True)
    assert r.refresh() is False

    w.resize((8, 10))  # no chunk is deleted: metalayer rewrite only
    assert r.refresh() is True
    assert r.shape == (8, 10)


def test_vlmeta_follows(tmp_path):
    urlpath = tmp_path / "vlmeta.b2nd"
    w = create_array(urlpath)
    w.schunk.vlmeta["heartbeat"] = "old"
    r = blosc2.open(str(urlpath))
    assert r.schunk.vlmeta["heartbeat"] == "old"

    w.schunk.vlmeta["heartbeat"] = "brand-new and longer"
    assert r.schunk.vlmeta["heartbeat"] == "brand-new and longer"


WRITER_SCRIPT = """
import sys
import numpy as np
import blosc2

urlpath, iters, batch = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
a = blosc2.open(urlpath, mode="a", locking=True)
for i in range(iters):
    old = a.shape[0]
    new = old + batch
    a.resize((new,))
    a[old:new] = np.arange(old, new, dtype=np.int64)
sys.exit(0)
"""


def test_cross_process_growth_hammer(tmp_path):
    # A writer process keeps growing a 1-d array while this process follows
    # it; every value the reader can see must already be correct
    urlpath = tmp_path / "hammer.b2nd"
    batch = 100
    iters = 200
    a = blosc2.zeros((0,), dtype=np.int64, chunks=(1000,), urlpath=str(urlpath), locking=True, mode="w")
    del a

    writer = subprocess.Popen([sys.executable, "-c", WRITER_SCRIPT, str(urlpath), str(iters), str(batch)])
    try:
        reader = blosc2.open(str(urlpath), locking=True)
        seen = 0
        deadline = time.monotonic() + 120  # safety net against a hung writer
        while writer.poll() is None:
            assert time.monotonic() < deadline, "writer process did not finish in time"
            reader.refresh()
            n = reader.shape[0]
            # Seeing shape n means the resize to n completed, hence every
            # batch written *before* that resize started is final.  The very
            # last batch may still be being filled (readers can see the fill
            # value there, as in HDF5 SWMR), so verify only below n - batch.
            settled = n - batch
            if settled > seen:
                np.testing.assert_array_equal(reader[seen:settled], np.arange(seen, settled, dtype=np.int64))
                seen = settled
        # Final sweep over the whole array
        reader.refresh()
        assert reader.shape == (iters * batch,)
        np.testing.assert_array_equal(reader[:], np.arange(iters * batch, dtype=np.int64))
    finally:
        if writer.poll() is None:
            writer.kill()
        writer.wait()

    assert writer.returncode == 0, f"writer process failed with exit code {writer.returncode}"
    assert seen > 0
