#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Codecs and filters registered from Python need the GIL, so C-Blosc2 must use one thread for them,
# also where it picks the number of threads itself: containers made from frames or opened from
# files start with the global number of threads, and BLOSC_NTHREADS replaces the nthreads of every
# context.  On failure the callbacks run on C-Blosc2's worker threads and crash the interpreter, so
# every case runs in a child interpreter.

import os
import subprocess
import sys
import textwrap

import pytest

import blosc2

pytestmark = pytest.mark.skipif(blosc2.IS_WASM, reason="no subprocesses and threads on emscripten/wasm32")

CHILD = textwrap.dedent(
    """
    import os
    import sys
    import threading

    import numpy as np

    import blosc2

    kind, source, urlpath = sys.argv[1:4]
    ID = 170  # of the codec or the filter, in the range for user ids
    MAIN = threading.get_ident()


    def check_thread():
        # Python code on a C-Blosc2 worker thread runs without the GIL: it often crashes, but not always
        if threading.get_ident() != MAIN:
            os._exit(70)


    def encoder(input, output, meta, schunk):
        check_thread()
        # 2 bytes per item: the values are below 2 ** 16
        n = input.size // 4
        output[: 2 * n] = input.view(np.int32).astype(np.uint16).view(np.uint8)
        return 2 * n


    def decoder(input, output, meta, schunk):
        check_thread()
        n = output.size // 4
        output[:] = input[: 2 * n].view(np.uint16).astype(np.int32).view(np.uint8)
        return output.size


    def reverse(input, output, meta, schunk):
        check_thread()
        output[:] = input[::-1]


    try:
        if kind == "codec":
            blosc2.register_codec("nthreads_codec", ID, encoder, decoder)
            cparams = {"codec": ID, "filters": [blosc2.Filter.NOFILTER], "filters_meta": [0]}
        else:
            blosc2.register_filter(ID, reverse, reverse, "nthreads_filter")
            cparams = {"codec": blosc2.Codec.LZ4, "filters": [ID], "filters_meta": [0]}
    except ValueError as e:
        print("refused:", e)
        sys.exit(0)
    # 16 blocks per chunk, so that C-Blosc2 would use its worker threads
    N = 16 * 1000
    cparams.update(typesize=4, nthreads=1, blocksize=4000, splitmode=blosc2.SplitMode.NEVER_SPLIT)
    values = (np.arange(3 * N) % 1000).astype(np.int32)
    if source == "env_later":
        # BLOSC_NTHREADS set after the registration: C-Blosc2 reads it for every new context
        dparams = {"nthreads": 1}
        s = blosc2.SChunk(chunksize=4 * N, data=values, cparams=cparams, dparams=dparams, urlpath=urlpath)
        cframe = s.to_cframe()
        os.environ["BLOSC_NTHREADS"] = "4"
        if blosc2.blosc2_ext.context_nthreads(1) != 4:
            # This C-Blosc2 does not see variables set at run time (another C runtime), so it cannot use it
            print("ok")
            sys.exit(0)

        def set_dparams():
            s.dparams = blosc2.DParams(nthreads=1)
            return s

        for make in (lambda: blosc2.SChunk(chunksize=4 * N, data=values, cparams=cparams, dparams=dparams),
                     lambda: blosc2.schunk_from_cframe(cframe, copy=True),
                     lambda: blosc2.open(urlpath),
                     set_dparams):
            try:
                make().decompress_chunk(0)
                print("not refused")
            except ValueError as e:
                assert "BLOSC_NTHREADS=4" in str(e), e
    elif source.startswith("ndarray"):
        a = blosc2.asarray(values, chunks=(N,), blocks=(N // 16,), cparams=cparams, dparams={"nthreads": 1},
                           urlpath=urlpath)
        blosc2.set_nthreads(4)
        if source == "ndarray_frame":
            a = blosc2.ndarray_from_cframe(a.to_cframe(), copy=True)
        else:
            a = blosc2.open(urlpath, mode="a")
        assert np.array_equal(a[:], values)
        a[:N] = values[N : 2 * N]
        assert np.array_equal(a[:N], values[N : 2 * N])
    else:
        s = blosc2.SChunk(chunksize=4 * N, data=values, cparams=cparams, dparams={"nthreads": 1},
                          urlpath=urlpath)
        blosc2.set_nthreads(4)
        if source == "frame":
            s = blosc2.schunk_from_cframe(s.to_cframe(), copy=True)
        elif source == "open":
            s = blosc2.open(urlpath, mode="a")
        elif source == "read_mode":
            s = blosc2.SChunk(urlpath=urlpath, mode="r")
        for i in range(3):
            chunk = np.frombuffer(s.decompress_chunk(i), np.int32)
            assert np.array_equal(chunk, values[i * N : (i + 1) * N])
        if source != "read_mode":  # unrelated bug: that SChunk gets chunksize 2 ** 24, get_slice() fails
            out = np.empty_like(values)
            s.get_slice(out=out)
            assert np.array_equal(out, values)
        if source in ("frame", "open"):
            s.append_data(values[:N])
            assert np.array_equal(np.frombuffer(s.decompress_chunk(3), np.int32), values[:N])
    print("ok")
    """
)


@pytest.mark.parametrize(
    ("kind", "source"),
    [
        # Every source with a codec, a filter where the check runs in another place
        ("codec", "frame"),
        ("codec", "open"),
        ("codec", "read_mode"),
        ("codec", "ndarray_frame"),
        ("codec", "ndarray_open"),
        ("filter", "frame"),
        ("filter", "open"),
        ("codec", "env_later"),
        ("filter", "env_later"),
    ],
)
def test_global_nthreads(kind, source, tmp_path):
    # Containers made from frames get C-Blosc2's global number of threads, opened ones blosc2.nthreads
    # (here both 4): they must use one
    env = os.environ.copy()
    env.pop("BLOSC_NTHREADS", None)
    result = subprocess.run(
        [sys.executable, "-c", CHILD, kind, source, str(tmp_path / "nthreads.b2frame")],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    assert result.returncode == 0, (result.returncode, result.stderr[-2000:])
    assert result.stdout.split() == ["ok"]


@pytest.mark.parametrize("kind", ["codec", "filter"])
def test_register_with_blosc_nthreads(kind, tmp_path):
    # C-Blosc2 gives every context BLOSC_NTHREADS threads, also the ones asking for one: registering a
    # Python codec or filter must fail then, instead of crashing later (on main: in SChunk(data=...))
    env = os.environ.copy()
    env["BLOSC_NTHREADS"] = "4"
    result = subprocess.run(
        [sys.executable, "-c", CHILD, kind, "frame", str(tmp_path / "nthreads.b2frame")],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    assert result.returncode == 0, (result.returncode, result.stderr[-2000:])
    assert result.stdout.startswith("refused:"), result.stdout
    assert "BLOSC_NTHREADS=4" in result.stdout
