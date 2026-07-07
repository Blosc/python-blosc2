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
