"""Lifetime contracts around NDArray/SChunk/vlmeta and zero-copy cframes.

NDArray.schunk is a view over C memory owned by the NDArray; the documented
contract is "keep the array alive while using its schunk". A keep-alive
back-reference was tried and reverted: the NDArray<->SChunk cycle deferred
finalization from refcount time to GC time, exhausting file descriptors and
breaking the indexing machinery's dropped-on-gc caches."""

import gc
import sys
import weakref

import numpy as np
import pytest

import blosc2


def test_ndarray_dies_promptly_without_gc():
    # Refcount-prompt finalization is load-bearing (fd release, indexing's
    # weakref caches): dropping an array and its schunk handle must free the
    # pair immediately, with no cycle collection needed.
    a = blosc2.asarray(np.arange(100))
    sc = a.schunk
    ref = weakref.ref(a)
    del a, sc
    assert ref() is None  # no gc.collect(): refcount alone must do it


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
