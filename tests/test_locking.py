#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Tests for the opt-in cross-process file locking for disk-based containers
# (the `locking` storage parameter, backed by a `.b2lock` sidecar file).

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
