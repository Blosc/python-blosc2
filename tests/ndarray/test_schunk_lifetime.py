"""Lifetime contracts around NDArray/SChunk/vlmeta and zero-copy cframes.

NDArray.schunk is a view over the NDArray's C super-chunk. The NDArray and
the view share a small owner object that frees the C array when both are
gone, so the view stays valid after the NDArray. A keep-alive back-reference
from the view to the NDArray was tried and reverted: the NDArray<->SChunk
cycle deferred finalization from refcount time to GC time, exhausting file
descriptors and breaking the indexing machinery's dropped-on-gc caches."""

import gc
import os
import sys
import weakref

import numpy as np
import pytest

import blosc2


@pytest.fixture
def gc_disabled():
    enabled = gc.isenabled()
    gc.disable()
    yield
    if enabled:
        gc.enable()


def test_ndarray_dies_promptly_without_gc(gc_disabled):
    # Refcount-prompt finalization is load-bearing (fd release, indexing's
    # weakref caches): dropping an array and its schunk handle must free the
    # pair immediately, with no cycle collection needed.
    a = blosc2.asarray(np.arange(100))
    sc = a.schunk
    refs = [weakref.ref(a), weakref.ref(sc)]
    del a, sc
    assert all(ref() is None for ref in refs)  # no gc.collect(): refcount alone must do it


def test_fresh_ndarray_has_no_extra_refs():
    # No hidden reference structures on a fresh array (same invariant as
    # test_expand_dims's leak checks).
    a = blosc2.asarray(np.arange(4))
    assert sys.getrefcount(a) == 2 - (sys.version_info >= (3, 14))


def test_from_cframe_zero_copy_pins_buffer():
    # copy=False (default) points the C schunk into the bytes buffer: the
    # returned object must keep that buffer alive even when the caller's
    # cframe was a temporary.
    ref = np.arange(1000, dtype="i8")
    arr = blosc2.ndarray_from_cframe(bytes(blosc2.asarray(ref).to_cframe()))
    _churn = [np.random.default_rng().random(1000).tobytes() for _ in range(200)]
    gc.collect()
    np.testing.assert_array_equal(arr[:], ref)

    schunk = blosc2.schunk_from_cframe(bytes(blosc2.SChunk(data=ref.tobytes()).to_cframe()))
    _churn = [np.random.default_rng().random(1000).tobytes() for _ in range(200)]
    gc.collect()
    out = np.frombuffer(schunk.decompress_chunk(0), dtype="i8")
    np.testing.assert_array_equal(out[: len(ref)], ref)


def test_orphan_vlmeta_raises_not_segfaults():
    # vlmeta deliberately weak-refs its owner; every operation must then
    # fail with ReferenceError once the owner is gone, never touch the
    # dangling C pointer (this used to segfault on the read paths).
    def orphan():
        a = blosc2.asarray(np.arange(100))
        a.schunk.vlmeta["tag"] = "hello"
        return a.schunk.vlmeta

    vm = orphan()
    gc.collect()
    for op in (
        lambda: vm["tag"],
        lambda: vm[:],
        lambda: len(vm),
        lambda: "tag" in vm,
        lambda: vm.__setitem__("x", 1),
    ):
        with pytest.raises(ReferenceError):
            op()


_DATA = np.arange(1000, dtype=np.int64)


def _make_array_file(tmp_path):
    urlpath = str(tmp_path / "a.b2nd")
    a = blosc2.asarray(_DATA, urlpath=urlpath, mode="w", chunks=(100,), blocks=(50,), meta={"m": 1})
    a.vlmeta["v"] = "x"
    return urlpath


def _make_2d():
    return blosc2.asarray(_DATA.reshape(1, 1000), chunks=(1, 100), blocks=(1, 50))


def _reuse_freed_memory():
    # Buffers the size of a blosc2_schunk struct (~64 KiB) and of a small cframe
    gc.collect()
    return [bytearray(size) for size in (1 << 16, 1 << 11) for _ in range(200)]


# The schunk of a temporary NDArray; views share their base's chunks, so the base must stay alive too
_SCHUNK_OF_TEMPORARY = {
    "open": lambda tmp_path: blosc2.open(_make_array_file(tmp_path)).schunk,
    "open_mmap": lambda tmp_path: blosc2.open(_make_array_file(tmp_path), mmap_mode="r").schunk,
    "asarray": lambda tmp_path: blosc2.asarray(_DATA, chunks=(100,), blocks=(50,)).schunk,
    "expand_dims": lambda tmp_path: blosc2.expand_dims(_make_2d(), 0).schunk,
    "squeeze": lambda tmp_path: blosc2.squeeze(_make_2d(), 0).schunk,
    "squeeze_zero_copy": lambda tmp_path: (
        blosc2.squeeze(blosc2.ndarray_from_cframe(bytes(_make_2d().to_cframe())), 0).schunk
    ),
}


@pytest.mark.parametrize("case", _SCHUNK_OF_TEMPORARY)
def test_schunk_outlives_ndarray(tmp_path, case):
    # This used to read freed memory (nchunks 0, no metalayers) or crash
    if case == "open_mmap" and blosc2.IS_WASM:
        pytest.skip("mmap_mode is not supported reliably on wasm32")
    schunk = _SCHUNK_OF_TEMPORARY[case](tmp_path)
    _ = _reuse_freed_memory()
    assert schunk.nchunks == 10
    assert "b2nd" in schunk.meta
    np.testing.assert_array_equal(np.frombuffer(schunk.decompress_chunk(0), dtype=np.int64), _DATA[:100])


def test_meta_and_vlmeta_outlive_ndarray(tmp_path):
    # NDArray.meta holds the schunk; a kept schunk keeps its vlmeta valid
    urlpath = _make_array_file(tmp_path)
    meta = blosc2.open(urlpath).meta
    schunk = blosc2.open(urlpath).schunk
    _ = _reuse_freed_memory()
    assert list(meta.keys()) == ["b2nd", "m"]
    assert meta["m"] == 1
    assert schunk.vlmeta["v"] == "x"


@pytest.mark.skipif(sys.version_info < (3, 12), reason="__buffer__ needs Python 3.12")
def test_zero_copy_cycle_is_collected():
    # The array keeps its cframe alive: a cycle through it (a buffer exporter that keeps the array)
    # must stay visible to the cyclic GC
    class Exporter:
        def __init__(self, data):
            self.data = data

        def __buffer__(self, flags):
            return memoryview(self.data)

    exporter = Exporter(bytes(blosc2.asarray(np.arange(100)).to_cframe()))
    exporter.array = blosc2.ndarray_from_cframe(exporter)
    ref = weakref.ref(exporter)
    del exporter
    gc.collect()
    assert ref() is None


def _open_fds():
    # None where open file descriptors cannot be listed (e.g. Windows)
    for fd_dir in ("/proc/self/fd", "/dev/fd"):
        if os.path.isdir(fd_dir):
            return len(os.listdir(fd_dir))
    return None


def test_file_closed_at_once_without_gc(tmp_path, gc_disabled):
    # No cycle: the file is closed as soon as the last of the NDArray and its schunk goes, also
    # after the schunk, meta and vlmeta were used
    urlpath = _make_array_file(tmp_path)
    fds = _open_fds()

    def files_opened():
        return None if fds is None else _open_fds() - fds

    a = blosc2.open(urlpath)
    assert (a.schunk.nchunks, a.meta["m"], a.vlmeta["v"]) == (10, 1, "x")
    assert files_opened() in (None, 1)
    del a
    assert files_opened() in (None, 0)
    schunk = blosc2.open(urlpath).schunk
    assert files_opened() in (None, 1)  # the kept schunk keeps the file open
    assert schunk.nchunks == 10
    del schunk
    assert files_opened() in (None, 0)
    os.remove(urlpath)  # fails on Windows while the file is still open
