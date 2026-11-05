#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Python callbacks (prefilters, fillers, postfilters, codecs and filters registered from Python) need the
# GIL, so the calls that run them must keep it even when blosc2.set_releasegil(True) asks to release it.

import collections
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


@pytest.fixture(params=[False, True], ids=["gil-held", "gil-released"])
def releasegil(request):
    old = blosc2.set_releasegil(request.param)
    yield
    blosc2.set_releasegil(old)


def called(name):
    calls[name] += 1


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
