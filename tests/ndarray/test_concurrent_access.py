#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# NDArray.get_slice_numpy() decompresses without the GIL.  These tests check that other threads
# really run meanwhile, and that every other operation on the same array still waits for it.

import itertools
import os
import signal
import threading
import time

import numpy as np
import pytest

import blosc2

pytestmark = pytest.mark.skipif(blosc2.IS_WASM, reason="Pyodide cannot start threads")

SHAPE, CHUNKS, BLOCKS = (64, 256, 256), (8, 256, 256), (2, 64, 256)
VALUES = np.arange(np.prod(SHAPE), dtype=np.float32).reshape(SHAPE)
# Per stress test; without the locking, the races show up within the first few reads
SECONDS = 0.5


def make_array(**kwargs):
    return blosc2.asarray(VALUES, chunks=CHUNKS, blocks=BLOCKS, **kwargs)


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


def check_region(array):
    """Read [8:24, 10:200]; a concurrent write_region() only ever writes VALUES or VALUES + 1 into
    [8:24, 0:100], one chunk (8 rows) at a time."""
    got, want = array[8:24, 10:200], VALUES[8:24, 10:200]
    for chunk in (slice(0, 8), slice(8, 16)):
        got_rows, want_rows = got[chunk], want[chunk]
        np.testing.assert_array_equal(got_rows[:, 90:], want_rows[:, 90:])
        assert np.array_equal(got_rows[:, :90], want_rows[:, :90]) or np.array_equal(
            got_rows[:, :90], want_rows[:, :90] + 1
        )


def write_region(array, k):
    array[8:24, 0:100] = VALUES[8:24, 0:100] + (k % 2)


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

    def longest_gap():
        """Longest time a ticker thread did not run during one read, as a fraction of the read."""
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
        return np.diff([t0, *inside, t1]).max() / (t1 - t0)

    # Holding the GIL while decompressing leaves one gap as long as nearly the whole read, every
    # time; a busy machine can delay the ticker now and then, hence a few attempts
    gaps = []
    while len(gaps) < 5 and (not gaps or gaps[-1] >= 0.5):
        gaps.append(longest_gap())
    assert gaps[-1] < 0.5, f"longest gaps, as fractions of the read: {gaps}"


@pytest.mark.parametrize("on_disk", [False, True], ids=["memory", "disk"])
def test_read_while_writing_same_handle(tmp_path, on_disk):
    a = make_array(urlpath=tmp_path / "a.b2nd", mode="w") if on_disk else make_array()
    k = itertools.count()
    assert run_for(SECONDS, lambda: check_region(a), lambda: write_region(a, next(k))) == []


def update_chunks(a, k):
    n = 1 + k % 6
    a.schunk.update_chunk(n, a.schunk.get_chunk(n))
    a.schunk.decompress_chunk(k % 8)


def write_vlmeta(a, k):
    a.schunk.vlmeta[f"k{k % 5}"] = list(range(k % 50))


def swap_dparams(a, k):
    a.schunk.dparams = blosc2.DParams(nthreads=1 + k % 2)


def resize(a, k):
    # grow and shrink past the region check_region() reads
    a.resize((SHAPE[0] + 8 * (k % 2), *SHAPE[1:]))


@pytest.mark.parametrize(
    "access", [update_chunks, write_vlmeta, swap_dparams, resize], ids=lambda f: f.__name__
)
def test_read_while_other_access_same_handle(access):
    a = make_array()
    k = itertools.count()
    assert run_for(SECONDS, lambda: check_region(a), lambda: access(a, next(k))) == []


def test_concurrent_reads_with_python_postfilter():
    a = make_array(dparams={"nthreads": 1})

    @a.schunk.postfilter(np.float32, np.float32)
    def add_one(inp, out, offset):
        out[:] = inp + 1

    def read():
        np.testing.assert_array_equal(a[8:24, 10:200], VALUES[8:24, 10:200] + 1)

    assert run_for(SECONDS, read, read) == []


def test_lazy_expression_while_reading_operands(tmp_path):
    # The miniexpr prefilter reads on-disk operands from C, so it must hold their locks too
    a = make_array(urlpath=tmp_path / "a.b2nd", mode="w")
    b = make_array(urlpath=tmp_path / "b.b2nd", mode="w")

    def compute():
        np.testing.assert_array_equal((a + b).compute()[8:24], 2 * VALUES[8:24])

    assert run_for(SECONDS, compute, compute, lambda: check_region(a), lambda: check_region(b)) == []


def test_concat_both_orders_does_not_deadlock():
    a = blosc2.asarray(np.zeros((16, 64), dtype=np.int32), chunks=(8, 64))
    b = blosc2.asarray(np.ones((16, 64), dtype=np.int32), chunks=(8, 64))
    errors = run_for(
        SECONDS,
        lambda: blosc2.concat([a, b], axis=0),
        lambda: blosc2.concat([b, a], axis=0),
        lambda: a[:],
        lambda: b[:],
    )
    assert errors == []


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")  # fork() with threads running
def test_fork_while_another_thread_holds_the_lock():
    # As when forking while another thread reads without the GIL: the child must not inherit
    # a lock that no thread of its own will ever release
    a = make_array(dparams={"nthreads": 1})
    held, release = threading.Event(), threading.Event()

    def hold():
        with a._handle_lock:
            held.set()
            release.wait()

    t = threading.Thread(target=hold)
    t.start()
    try:
        assert held.wait(timeout=10)
        pid = os.fork()
        if pid == 0:
            ok = False
            try:
                ok = np.array_equal(a[8:24, 10:200], VALUES[8:24, 10:200])
            finally:
                os._exit(0 if ok else 1)
        deadline = time.monotonic() + 30
        while (status := os.waitpid(pid, os.WNOHANG))[0] == 0:
            if time.monotonic() > deadline:
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)
                pytest.fail("the child hung on a lock held by a thread of the parent")
            time.sleep(0.01)
        assert os.waitstatus_to_exitcode(status[1]) == 0
    finally:
        release.set()
        t.join()
