#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# NDArray.get_slice_numpy() decompresses without the GIL.  These tests check that other threads
# really run meanwhile, and that every other operation on the same array still waits for it.

import threading
import time

import numpy as np
import pytest

import blosc2

SHAPE, CHUNKS, BLOCKS = (64, 256, 256), (8, 256, 256), (2, 64, 256)


def run_for(seconds, *targets):
    """Run each target in a loop in its own thread for `seconds`; return the errors they raised."""
    stop = time.monotonic() + seconds
    errors = []

    def loop(target):
        try:
            while time.monotonic() < stop and not errors:
                target()
        except Exception as e:  # report anything a racing thread hits
            errors.append(e)

    threads = [threading.Thread(target=loop, args=(t,)) for t in targets]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=seconds + 60)
    assert not any(t.is_alive() for t in threads), "threads did not finish (deadlock?)"
    return errors


@pytest.fixture
def array_and_values(tmp_path, request):
    values = np.arange(np.prod(SHAPE), dtype=np.float32).reshape(SHAPE)
    kwargs = {"urlpath": tmp_path / "a.b2nd", "mode": "w"} if request.param == "disk" else {}
    return blosc2.asarray(values, chunks=CHUNKS, blocks=BLOCKS, **kwargs), values


def check_region(array, values):
    """Read [8:24, 10:200]; a concurrent writer only ever writes values or values + 1 into
    [8:24, 0:100], one chunk (8 rows) at a time."""
    got, want = array[8:24, 10:200], values[8:24, 10:200]
    for c in (0, 8):
        g, w = got[c : c + 8], want[c : c + 8]
        np.testing.assert_array_equal(g[:, 90:], w[:, 90:])
        assert np.array_equal(g[:, :90], w[:, :90]) or np.array_equal(g[:, :90], w[:, :90] + 1)


def test_handle_lock_is_shared():
    a = blosc2.asarray(np.arange(100).reshape(10, 10), chunks=(5, 5))
    view = blosc2.expand_dims(a, axis=0)
    for other in (a.schunk, a.schunk.vlmeta, view, view.schunk):
        assert other._handle_lock is a._handle_lock


def test_get_slice_releases_the_gil(monkeypatch):
    # Python-registered codecs or filters (from other tests in this process) keep the GIL
    monkeypatch.setattr(blosc2, "ucodecs_registry", {})
    monkeypatch.setattr(blosc2, "ufilters_registry", {})
    a = blosc2.linspace(
        0,
        1,
        32 * 1024 * 1024,
        shape=(4096, 8192),
        dtype=np.float32,
        cparams={"codec": blosc2.Codec.ZSTD, "clevel": 1},
        dparams={"nthreads": 1},
    )
    stamps, stop = [], threading.Event()

    def ticker():
        while not stop.is_set():
            stamps.append(time.perf_counter())
            time.sleep(0)

    t = threading.Thread(target=ticker)
    t.start()
    try:
        t0 = time.perf_counter()
        _ = a[:]
        t1 = time.perf_counter()
    finally:
        stop.set()
        t.join()
    inside = [s for s in stamps if t0 <= s <= t1]
    gaps = np.diff([t0, *inside, t1])
    # Holding the GIL while decompressing leaves one gap as long as the whole read
    assert gaps.max() < 0.5 * (t1 - t0), f"longest gap {gaps.max():.3f} s of a {t1 - t0:.3f} s read"


@pytest.mark.parametrize("array_and_values", ["memory", "disk"], indirect=True)
def test_read_while_writing_same_handle(array_and_values):
    a, values = array_and_values
    k = [0]

    def write():
        k[0] += 1
        a[8:24, 0:100] = values[8:24, 0:100] + (k[0] % 2)

    assert run_for(1.0, lambda: check_region(a, values), write) == []


@pytest.mark.parametrize("array_and_values", ["memory"], indirect=True)
@pytest.mark.parametrize("other", ["schunk", "vlmeta", "dparams", "resize"])
def test_read_while_other_access_same_handle(array_and_values, other):
    a, values = array_and_values
    k = [0]

    def access():
        k[0] += 1
        if other == "schunk":
            n = 1 + k[0] % 6
            a.schunk.update_chunk(n, a.schunk.get_chunk(n))
            a.schunk.decompress_chunk(k[0] % 8)
        elif other == "vlmeta":
            a.schunk.vlmeta[f"k{k[0] % 5}"] = list(range(k[0] % 50))
        elif other == "dparams":
            a.schunk.dparams = blosc2.DParams(nthreads=1 + k[0] % 2)
        elif other == "resize":
            # grow and shrink past the region the reader checks
            a.resize((SHAPE[0] + 8 * (k[0] % 2), *SHAPE[1:]))

    assert run_for(1.0, lambda: check_region(a, values), access) == []


def test_concurrent_reads_with_python_postfilter():
    values = np.arange(np.prod(SHAPE), dtype=np.float32).reshape(SHAPE)
    a = blosc2.asarray(values, chunks=CHUNKS, blocks=BLOCKS, dparams={"nthreads": 1})

    @a.schunk.postfilter(np.float32, np.float32)
    def add_one(inp, out, offset):
        out[:] = inp + 1

    def read():
        np.testing.assert_array_equal(a[8:24, 10:200], values[8:24, 10:200] + 1)

    assert run_for(1.0, read, read) == []


def test_concat_both_orders_does_not_deadlock():
    a = blosc2.asarray(np.zeros((16, 64), dtype=np.int32), chunks=(8, 64))
    b = blosc2.asarray(np.ones((16, 64), dtype=np.int32), chunks=(8, 64))
    errors = run_for(
        1.0,
        lambda: blosc2.concat([a, b], axis=0),
        lambda: blosc2.concat([b, a], axis=0),
        lambda: a[:],
        lambda: b[:],
    )
    assert errors == []
