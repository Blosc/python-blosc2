#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# append_data(), insert_data(), update_data(), SChunk(data=...) and set_slice() compress through
# c-blosc2's blosc2_schunk_{append,insert,update}_buffer(), which write frames from a compression
# buffer reused by the calling thread.

import threading

import numpy as np
import pytest

import blosc2

CHUNK_LEN = 20_000
STORAGES = [
    {"contiguous": False},
    {"contiguous": True},
    {"contiguous": True, "urlpath": "data_writes.b2frame"},
    {"contiguous": False, "urlpath": "data_writes_s.b2frame"},
]


@pytest.fixture(params=STORAGES, ids=["memory", "cframe", "disk-cframe", "disk-sframe"])
def storage(request):
    blosc2.remove_urlpath(request.param.get("urlpath"))
    yield request.param
    blosc2.remove_urlpath(request.param.get("urlpath"))


@pytest.fixture(params=[False, True], ids=["gil", "nogil"])
def releasegil(request):
    old = blosc2.set_releasegil(request.param)
    yield request.param
    blosc2.set_releasegil(old)


def new_schunk(storage, nthreads=2, **kwargs):
    return blosc2.SChunk(
        chunksize=CHUNK_LEN * 4,
        cparams={"typesize": 4, "nthreads": nthreads},
        dparams={"nthreads": nthreads},
        **storage,
        **kwargs,
    )


def chunks_of(schunk):
    return [np.frombuffer(schunk.decompress_chunk(i), dtype=np.int32) for i in range(schunk.nchunks)]


def test_data_writes(storage, releasegil):
    schunk = new_schunk(storage)
    expected = []
    for i in range(4):
        expected.append(np.arange(CHUNK_LEN, dtype=np.int32) * (i + 1))
        assert schunk.append_data(expected[-1]) == i + 1
    expected.insert(2, np.full(CHUNK_LEN, -3, dtype=np.int32))
    assert schunk.insert_data(2, expected[2], copy=False) == 5
    expected.insert(5, np.full(CHUNK_LEN, -5, dtype=np.int32))
    assert schunk.insert_data(5, expected[5], copy=True) == 6
    # Several rounds of updates, so that frames get the reused buffer with other contents each time
    for rnd in range(3):
        for n in reversed(range(len(expected))):
            expected[n] = np.arange(CHUNK_LEN, dtype=np.int32) + 1000 * n + 100_000 * rnd
            assert schunk.update_data(n, expected[n], copy=bool(rnd % 2)) == len(expected)
    for got, exp in zip(chunks_of(schunk), expected, strict=True):
        np.testing.assert_array_equal(got, exp)
    assert schunk.nbytes == len(expected) * CHUNK_LEN * 4
    if "urlpath" in storage:
        reopened = blosc2.open(storage["urlpath"])
        for got, exp in zip(chunks_of(reopened), expected, strict=True):
            np.testing.assert_array_equal(got, exp)


def test_data_writes_reach_prefilter(storage, releasegil):
    # The prefilter computes each item from its position, so it must see the right chunk number.
    # It runs Python, so the writes keep the GIL even with set_releasegil(True).
    schunk = new_schunk(storage, nthreads=1)

    @schunk.prefilter(np.int32, np.int32)
    def from_position(input, output, offset):
        output[:] = offset + np.arange(output.size, dtype=np.int32)

    zeros = np.zeros(CHUNK_LEN, dtype=np.int32)
    for _ in range(3):
        schunk.append_data(zeros)
    schunk.update_data(2, zeros, copy=False)
    schunk.insert_data(1, zeros, copy=False)
    schunk.remove_prefilter("from_position")
    # The chunks written at positions 0, 1 (inserted), 1 (moved up) and 2 (updated)
    for got, pos in zip(chunks_of(schunk), [0, 1, 1, 2], strict=True):
        np.testing.assert_array_equal(got, np.arange(pos * CHUNK_LEN, (pos + 1) * CHUNK_LEN))


def test_schunk_data_and_set_slice(storage, releasegil):
    data = np.arange(3 * CHUNK_LEN + 123, dtype=np.int32)
    schunk = new_schunk(storage, data=data)
    assert schunk.nchunks == 4
    # Grows the partial last chunk, then appends whole and partial chunks
    more = -np.arange(2 * CHUNK_LEN + 7, dtype=np.int32)
    schunk[data.size : data.size + more.size] = more
    expected = np.concatenate([data, more])
    out = np.empty_like(expected)
    schunk.get_slice(0, expected.size, out=out)
    np.testing.assert_array_equal(out, expected)


def test_failed_data_writes_change_nothing(storage):
    schunk = new_schunk(storage, data=np.arange(2 * CHUNK_LEN, dtype=np.int32))
    before = chunks_of(schunk)
    buffer = np.ones(CHUNK_LEN, dtype=np.int32)
    with pytest.raises(RuntimeError):
        schunk.update_data(2, buffer, copy=False)
    with pytest.raises(RuntimeError):
        schunk.insert_data(3, buffer, copy=False)
    assert schunk.nchunks == 2
    assert schunk.nbytes == 2 * CHUNK_LEN * 4
    for got, exp in zip(chunks_of(schunk), before, strict=True):
        np.testing.assert_array_equal(got, exp)


@pytest.mark.parametrize("locking", [False, True])
def test_insert_after_another_handle_appended(tmp_path, locking):
    # The position is checked against the file, not this handle's possibly stale view of it
    urlpath = str(tmp_path / "two_handles.b2frame")
    a = new_schunk({"contiguous": True, "urlpath": urlpath, "mode": "w", "locking": locking})
    a.append_data(np.zeros(CHUNK_LEN, dtype=np.int32))
    b = blosc2.open(urlpath, mode="a", locking=locking)
    assert b.nchunks == 1
    a.append_data(np.ones(CHUNK_LEN, dtype=np.int32))
    assert b.insert_data(2, np.full(CHUNK_LEN, 7, dtype=np.int32), copy=False) == 3
    reopened = blosc2.open(urlpath)
    for got, value in zip(chunks_of(reopened), [0, 1, 7], strict=True):
        np.testing.assert_array_equal(got, np.full(CHUNK_LEN, value, dtype=np.int32))


@pytest.mark.skipif(blosc2.IS_WASM, reason="no threads on wasm32")
def test_data_writes_from_threads(tmp_path):
    # Each thread compresses into its own reused buffer; frames in memory and on disk
    old = blosc2.set_releasegil(True)
    try:
        nthreads = 4
        schunks = [
            new_schunk(
                {"contiguous": True, "urlpath": str(tmp_path / f"t{i}.b2frame")}
                if i % 2
                else {"contiguous": True}
            )
            for i in range(nthreads)
        ]
        errors = []

        def write(i):
            try:
                for n in range(6):
                    schunks[i].append_data(np.full(CHUNK_LEN, 100 * i + n, dtype=np.int32))
                for n in range(6):
                    schunks[i].update_data(n, np.full(CHUNK_LEN, -(100 * i + n), dtype=np.int32), copy=False)
            except Exception as e:  # pragma: no cover - reported below
                errors.append(e)

        threads = [threading.Thread(target=write, args=(i,)) for i in range(nthreads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        for i, schunk in enumerate(schunks):
            for n, got in enumerate(chunks_of(schunk)):
                np.testing.assert_array_equal(got, np.full(CHUNK_LEN, -(100 * i + n), dtype=np.int32))
    finally:
        blosc2.set_releasegil(old)
