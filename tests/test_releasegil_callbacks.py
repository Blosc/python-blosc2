#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Python callbacks (prefilters, fillers, postfilters, codecs and filters registered from Python) need the
# GIL, so the calls that run them must keep it even when blosc2.set_releasegil(True) asks to release it.
# An exception raised in a callback must reach the caller as it is.

import collections
import contextlib
import threading
import time
import zlib

import numpy as np
import pytest

import blosc2

# Ids that no other test registers: registering is global and lasts for the whole process
CODEC_ID = 170
FILTER_ID = 171

N = 1000  # items per chunk
DTYPE = np.dtype(np.int32)
CHUNKSIZE = N * DTYPE.itemsize
# Several blocks per chunk, so that every callback runs more than once per chunk
CPARAMS = {"typesize": DTYPE.itemsize, "nthreads": 1, "blocksize": CHUNKSIZE // 4}
DPARAMS = {"nthreads": 1}

calls = collections.Counter()
failing = set()  # the callbacks that raise KeyError instead (see fail())


@pytest.fixture(params=[False, True], ids=["gil-held", "gil-released"])
def releasegil(request):
    old = blosc2.set_releasegil(request.param)
    yield
    blosc2.set_releasegil(old)


def called(name):
    calls[name] += 1
    if name in failing:
        raise KeyError(name)


@contextlib.contextmanager
def fail(name):
    """Make the callback `name` raise KeyError(name) meanwhile.  The KeyError must reach the caller, and
    the callback must run only once: C-Blosc2 stops at the first failure, so no Python code runs with the
    exception set."""
    ncalls = calls[name]
    failing.add(name)
    try:
        with pytest.raises(KeyError, match=name):
            yield
    finally:
        failing.discard(name)
    assert calls[name] == ncalls + 1


def encoder(input, output, meta, schunk):
    called("encoder")
    out = zlib.compress(input.tobytes())
    if len(out) >= output.size:
        return 0  # incompressible: C-Blosc2 stores the block as is
    output[: len(out)] = np.frombuffer(out, dtype=np.uint8)
    return len(out)


def decoder(input, output, meta, schunk):
    called("decoder")
    out = zlib.decompress(input.tobytes())
    output[: len(out)] = np.frombuffer(out, dtype=np.uint8)
    return len(out)


def forward(input, output, meta, schunk):
    called("forward")
    np.bitwise_xor(input, 0x5A, out=output)


def backward(input, output, meta, schunk):
    called("backward")
    np.bitwise_xor(input, 0x5A, out=output)


@pytest.fixture(scope="module")
def registered():
    if CODEC_ID not in blosc2.ucodecs_registry:
        blosc2.register_codec("test_releasegil", CODEC_ID, encoder, decoder)
    if FILTER_ID not in blosc2.ufilters_registry:
        blosc2.register_filter(FILTER_ID, forward, backward, "test_releasegil")


@pytest.fixture(params=["codec", "filter"])
def python_cparams(request, registered):
    """cparams with the Python codec or filter, and the names of its compressing and decompressing
    callbacks."""
    if request.param == "codec":
        cparams = {"codec": CODEC_ID, "filters": [blosc2.Filter.NOFILTER], "filters_meta": [0]}
        return {**CPARAMS, **cparams}, "encoder", "decoder"
    return {**CPARAMS, "filters": [FILTER_ID], "filters_meta": [0]}, "forward", "backward"


def data(nitems, start=0):
    return (np.arange(start, start + nitems) % 200).astype(DTYPE)


def new_schunk(cparams=CPARAMS, **kwargs):
    return blosc2.SChunk(chunksize=CHUNKSIZE, cparams=cparams, dparams=DPARAMS, **kwargs)


def read(schunk):
    out = np.empty(schunk.nbytes // DTYPE.itemsize, dtype=DTYPE)
    schunk.get_slice(out=out)
    return out


WRITE_PATHS = ["append", "insert", "update", "setitem"]


def write(schunk, path):
    """Write 2 * N items to the empty `schunk` through `path`, return them."""
    values = data(2 * N)
    if path == "append":
        schunk.append_data(values[:N])
        schunk.append_data(values[N:])
    elif path == "insert":
        schunk.append_data(values[N:])
        schunk.insert_data(0, values[:N], copy=False)
    elif path == "update":
        schunk.append_data(np.zeros(N, dtype=DTYPE))
        schunk.append_data(np.zeros(N, dtype=DTYPE))
        schunk.update_data(0, values[:N], copy=True)
        schunk.update_data(1, values[N:], copy=False)
    else:
        # Growing: appends a partial chunk, then recompresses it in full and appends one more
        schunk[: N // 2] = values[: N // 2]
        schunk[N // 2 : 2 * N] = values[N // 2 :]
    return values


@pytest.mark.parametrize("path", WRITE_PATHS)
def test_prefilter_writes(releasegil, path):
    schunk = new_schunk()

    @schunk.prefilter(DTYPE, DTYPE)
    def pf_test_releasegil(input, output, offset):
        output[:] = input | 1  # idempotent: the growing setitem prefilters some items twice

    values = write(schunk, path)
    np.testing.assert_array_equal(read(schunk), values | 1)


def test_filler(releasegil):
    schunk = new_schunk()
    values = data(2 * N + N // 2)

    @schunk.filler(((values, DTYPE),), DTYPE)
    def fill_test_releasegil(inputs, output, offset):
        output[:] = inputs[0] + 1

    np.testing.assert_array_equal(read(schunk), values + 1)


@pytest.mark.parametrize("path", ["init", *WRITE_PATHS])
def test_python_codec_filter_writes(releasegil, python_cparams, path):
    cparams, encode, decode = python_cparams
    before = calls.copy()
    if path == "init":
        values = data(2 * N + N // 2)
        schunk = new_schunk(cparams, data=values)
    else:
        schunk = new_schunk(cparams)
        values = write(schunk, path)
    np.testing.assert_array_equal(read(schunk), values)
    assert calls[encode] > before[encode]
    assert calls[decode] > before[decode]


def test_to_cframe_postfilter(releasegil):
    # Serializing a sparse super-chunk decompresses its chunks with its dparams, postfilter included
    values = data(2 * N)
    schunk = new_schunk(data=values, contiguous=False)

    @schunk.postfilter(DTYPE, DTYPE)
    def post_test_releasegil(input, output, offset):
        output[:] = input

    copy = blosc2.schunk_from_cframe(schunk.to_cframe(), copy=True)
    np.testing.assert_array_equal(read(copy), values)


def longest_gap(func):
    """Longest time a ticker thread did not run during func(), as a fraction of that call."""
    stamps, stop = [], threading.Event()

    def ticker():
        while not stop.is_set():
            stamps.append(time.perf_counter())
            time.sleep(0)

    t = threading.Thread(target=ticker)
    t.start()
    try:
        t0 = time.perf_counter()
        func()
        t1 = time.perf_counter()
    finally:
        stop.set()
        t.join()
    inside = [s for s in stamps if t0 <= s <= t1]
    return np.diff([t0, *inside, t1]).max() / (t1 - t0)


@pytest.mark.skipif(blosc2.IS_WASM, reason="set_releasegil() does nothing on wasm32")
def test_write_releases_the_gil_next_to_python_codecs(registered):
    # A codec or filter registered from Python only keeps the GIL for the compressions that use it
    values = np.linspace(0, 1, 8 * 1024 * 1024, dtype=np.float32)
    schunk = blosc2.SChunk(
        chunksize=values.nbytes, cparams={"codec": blosc2.Codec.ZSTD, "clevel": 1, "nthreads": 1}
    )
    old = blosc2.set_releasegil(True)
    try:
        # Holding the GIL leaves one gap as long as nearly the whole write, every time; a busy machine can
        # delay the ticker now and then, hence a few attempts (as in test_get_slice_releases_the_gil)
        gaps = []
        while len(gaps) < 5 and (not gaps or gaps[-1] >= 0.5):
            gaps.append(longest_gap(lambda: schunk.append_data(values)))
    finally:
        blosc2.set_releasegil(old)
    assert gaps[-1] < 0.5, f"longest gaps, as fractions of the write: {gaps}"


# The ways of writing one chunk to a super-chunk holding a chunk (or half of one, for setitem-append)
FAILING_WRITE_PATHS = ["append", "insert", "update", "setitem", "setitem-grow", "setitem-append"]


def schunk_for(path, cparams=CPARAMS):
    return new_schunk(cparams, data=data(N // 2 if path == "setitem-append" else N))


def write_chunk(schunk, path):
    values = data(N, start=1)
    if path == "append":
        schunk.append_data(values)
    elif path == "insert":
        schunk.insert_data(0, values, copy=False)
    elif path == "update":
        schunk.update_data(0, values, copy=True)
    elif path == "setitem":
        schunk[: N // 2] = values[: N // 2]
    else:
        # setitem-grow updates the second half of the full chunk, setitem-append recompresses the half
        # chunk in full; both append a chunk then
        schunk[N // 2 : 2 * N] = data(3 * N // 2, start=1)


@pytest.mark.parametrize("path", FAILING_WRITE_PATHS)
def test_prefilter_raises(releasegil, path):
    schunk = schunk_for(path)
    before = read(schunk)

    @schunk.prefilter(DTYPE, DTYPE)
    def pf_raises_test_releasegil(input, output, offset):
        called("prefilter")
        output[:] = input

    with fail("prefilter"):
        write_chunk(schunk, path)
    np.testing.assert_array_equal(read(schunk), before)


def test_filler_raises(releasegil):
    schunk = new_schunk()
    with fail("filler"):

        @schunk.filler(((data(2 * N), DTYPE),), DTYPE)
        def fill_raises_test_releasegil(inputs, output, offset):
            called("filler")

    assert schunk.nchunks == 0


@pytest.mark.parametrize("path", ["init", *FAILING_WRITE_PATHS])
def test_python_codec_filter_raises_on_write(releasegil, python_cparams, path):
    cparams, encode, _ = python_cparams
    if path == "init":
        with fail(encode):
            new_schunk(cparams, data=data(2 * N))
        return
    schunk = schunk_for(path, cparams)
    before = read(schunk)
    with fail(encode):
        write_chunk(schunk, path)
    np.testing.assert_array_equal(read(schunk), before)


READ_PATHS = ["decompress_chunk", "decompress_chunk-dst", "get_slice", "get_slice-out", "getitem"]


def read_chunk(schunk, path):
    if path == "decompress_chunk":
        schunk.decompress_chunk(0)
    elif path == "decompress_chunk-dst":
        schunk.decompress_chunk(0, np.empty(N, dtype=DTYPE))
    elif path == "get_slice":
        schunk.get_slice(N // 4, N)
    elif path == "get_slice-out":
        schunk.get_slice(N // 4, N, out=np.empty(3 * N // 4, dtype=DTYPE))
    else:
        schunk[N // 4 : N]


@pytest.mark.parametrize("path", READ_PATHS)
def test_postfilter_raises(releasegil, path):
    schunk = new_schunk(data=data(2 * N))

    @schunk.postfilter(DTYPE, DTYPE)
    def post_raises_test_releasegil(input, output, offset):
        called("postfilter")
        output[:] = input

    with fail("postfilter"):
        read_chunk(schunk, path)


@pytest.mark.parametrize("path", READ_PATHS)
def test_python_codec_filter_raises_on_read(releasegil, python_cparams, path):
    cparams, _, decode = python_cparams
    schunk = new_schunk(cparams, data=data(2 * N))
    with fail(decode):
        read_chunk(schunk, path)


def test_python_codec_filter_raises_ndarray(releasegil, python_cparams):
    cparams, encode, decode = python_cparams
    kwargs = {
        "cparams": blosc2.CParams(**cparams),
        "dparams": blosc2.DParams(**DPARAMS),
        "chunks": (20, 50),
        "blocks": (10, 50),
    }
    values = data(40 * 50).reshape(40, 50)
    with fail(encode):
        blosc2.asarray(values, **kwargs)
    array = blosc2.asarray(values, **kwargs)
    with fail(encode):
        array[5:25] = 0
    with fail(decode):
        array[5:25]
    with fail(decode):
        array.read_into(np.empty((20, 50), dtype=DTYPE), slice(5, 25))
    with fail(decode):
        array[[1, 3]]
    with fail(decode):
        blosc2.concat([array, blosc2.asarray(values, **kwargs)], axis=1)
    np.testing.assert_array_equal(array[:], values)


def test_python_codec_filter_without_schunk(releasegil, python_cparams):
    cparams = python_cparams[0]
    # The callbacks raise this themselves: Python codecs and filters only work in a super-chunk
    with pytest.raises(RuntimeError, match="without an SChunk"):
        blosc2.compress2(data(N), **cparams)
    with pytest.raises(RuntimeError, match="without an SChunk"):
        blosc2.blosc2_ext.vlcompress([data(N).tobytes()], **cparams)
    chunk = new_schunk(cparams, data=data(N)).get_chunk(0)
    with pytest.raises(RuntimeError, match="without an SChunk"):
        blosc2.decompress2(chunk, **DPARAMS)
    old_nthreads = blosc2.set_nthreads(1)  # decompress() uses the global number of threads
    try:
        for as_bytearray in (False, True):
            with pytest.raises(RuntimeError, match="without an SChunk"):
                blosc2.decompress(chunk, as_bytearray=as_bytearray)
    finally:
        blosc2.set_nthreads(old_nthreads)
