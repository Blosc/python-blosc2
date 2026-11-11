#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Codecs and filters registered from Python need the GIL, which C-Blosc2's worker threads never hold, so
# chunks that use them must decompress on the calling thread, whatever number of threads is asked for.
# Running them on a worker thread crashes the interpreter: the decompressions run in a subprocess.

import json
import os
import subprocess
import sys
import zlib

import numpy as np
import pytest

import blosc2

CODEC_ID = 170
FILTER_ID = 171
N = 1000  # int32 items per chunk, in 4 blocks, so that several threads can share a chunk
# Offsets in a chunk header: flags (BLOSC2_CHUNK_FLAGS), filter ids (BLOSC2_CHUNK_FILTER_CODES) and the
# id of a user-defined codec, which the codec bits of the flags (the top 3) name with BLOSC_UDCODEC_FORMAT
CHUNK_FLAGS = 0x2
CHUNK_FILTER_CODES = 0x10
CHUNK_UDCODEC = 0x16
UDCODEC_FORMAT = 6


def encoder(input, output, meta, schunk):
    out = zlib.compress(input.tobytes())
    if len(out) >= output.size:
        return 0
    output[: len(out)] = np.frombuffer(out, dtype=np.uint8)
    return len(out)


def decoder(input, output, meta, schunk):
    out = zlib.decompress(input.tobytes())
    output[: len(out)] = np.frombuffer(out, dtype=np.uint8)
    return len(out)


def xor(input, output, meta, schunk):
    np.bitwise_xor(input, 0x5A, out=output)


def outcome(func, *args, **kwargs):
    try:
        func(*args, **kwargs)
    except Exception as e:
        return f"{type(e).__name__}: {e} (from {e.__cause__ or e.__context__!r})"
    return "ok"


def decompression_outcomes(nthreads):
    """How decompressing chunks that use the Python codec or filter outside of a super-chunk ends, with
    `nthreads` threads; and whether a chunk that uses neither still decompresses."""
    blosc2.register_codec("test_threads", CODEC_ID, encoder, decoder)
    blosc2.register_filter(FILTER_ID, xor, xor, "test_threads")
    blosc2.set_nthreads(nthreads)  # decompress() uses the global number of threads
    values = (np.arange(N) % 200).astype(np.int32)
    cparams = {"typesize": values.itemsize, "nthreads": 1, "blocksize": values.nbytes // 4}
    dparams = {"nthreads": 1}  # super-chunks refuse more with Python codecs and filters
    python_cparams = {
        "codec": {**cparams, "codec": CODEC_ID, "filters": [blosc2.Filter.NOFILTER], "filters_meta": [0]},
        "filter": {**cparams, "filters": [FILTER_ID], "filters_meta": [0]},
    }
    results = {}
    for kind, kind_cparams in python_cparams.items():
        schunk = blosc2.SChunk(chunksize=values.nbytes, data=values, cparams=kind_cparams, dparams=dparams)
        chunk = schunk.get_chunk(0)
        results[f"decompress2 {kind}"] = outcome(blosc2.decompress2, chunk, nthreads=nthreads)
        results[f"decompress {kind}"] = outcome(blosc2.decompress, chunk)
    # vlcompress() cannot run the Python codec or filter (no super-chunk): name them in the header of a
    # plain VL-block chunk instead
    vlchunk = blosc2.blosc2_ext.vlcompress([values.tobytes()] * 4, **cparams)
    vlchunks = {"codec": bytearray(vlchunk), "filter": bytearray(vlchunk)}
    vlchunks["codec"][CHUNK_FLAGS] = vlchunk[CHUNK_FLAGS] & 0x1F | UDCODEC_FORMAT << 5
    vlchunks["codec"][CHUNK_UDCODEC] = CODEC_ID
    vlchunks["filter"][CHUNK_FILTER_CODES] = FILTER_ID
    for kind, chunk in vlchunks.items():
        results[f"vldecompress {kind}"] = outcome(
            blosc2.blosc2_ext.vldecompress, bytes(chunk), nthreads=nthreads
        )
    plain = blosc2.compress2(values, **cparams)
    results["decompress2 plain"] = blosc2.decompress2(plain, nthreads=nthreads) == values.tobytes()
    return results


@pytest.mark.skipif(blosc2.IS_WASM, reason="no subprocesses and no threads on wasm32")
def test_python_codec_filter_decompress_threads():
    env = {name: value for name, value in os.environ.items() if name != "BLOSC_NTHREADS"}
    runs = {}
    for nthreads in (1, 4):
        proc = subprocess.run(
            [sys.executable, __file__, str(nthreads)],
            capture_output=True,
            text=True,
            env=env,
            check=False,
            timeout=300,
        )
        assert proc.returncode == 0, f"{nthreads} threads: exit code {proc.returncode}\n{proc.stderr}"
        runs[nthreads] = json.loads(proc.stdout.splitlines()[-1])
    # The callbacks raise this themselves: Python codecs and filters only work in a super-chunk
    assert all("without an SChunk" in result for call, result in runs[1].items() if "plain" not in call), (
        runs[1]
    )
    assert runs[1]["decompress2 plain"] is True, runs[1]
    assert runs[4] == runs[1]


def late_registration(phase, kind, path, loader, read_path):
    values = (np.arange(N) % 200).astype(np.int32)
    blosc2.set_nthreads(4)
    if phase == "write":
        blosc2.register_codec("test_threads", CODEC_ID, encoder, decoder)
        blosc2.register_filter(FILTER_ID, xor, xor, "test_threads")
        cparams = {"typesize": 4, "nthreads": 1, "blocksize": values.nbytes // 4}
        if kind == "codec":
            cparams["codec"] = CODEC_ID
        else:
            cparams["filters"] = [FILTER_ID]
            cparams["filters_meta"] = [0]
        schunk = blosc2.SChunk(
            chunksize=values.nbytes,
            data=values,
            cparams=cparams,
            dparams={"nthreads": 1},
            urlpath=path,
            contiguous=True,
        )
        del schunk
        return

    if loader == "cframe":
        with open(path, "rb") as frame:
            schunk = blosc2.schunk_from_cframe(frame.read(), copy=True)
    else:
        schunk = blosc2.open(path, mode="r")
    assert schunk.dparams.nthreads == 4
    blosc2.register_codec("test_threads", CODEC_ID, encoder, decoder)
    blosc2.register_filter(FILTER_ID, xor, xor, "test_threads")
    for releasegil in (False, True):
        blosc2.set_releasegil(releasegil)
        if read_path == "decompress_chunk":
            assert schunk.decompress_chunk(0) == values.tobytes()
            dst = np.empty_like(values)
            schunk.decompress_chunk(0, dst)
            np.testing.assert_array_equal(dst, values)
        else:
            assert schunk.get_slice() == values.tobytes()


def span_postfilter():
    import threading

    assert not blosc2.ucodecs_registry
    assert not blosc2.ufilters_registry
    values = np.arange(N, dtype=np.int32)
    array = blosc2.asarray(
        values,
        chunks=(N,),
        blocks=(N // 4,),
        dparams={"nthreads": 1},
    )
    caller = threading.get_ident()
    calls = []

    @array.schunk.postfilter(np.int32, np.int32)
    def post_span_threads(input, output, offset):
        assert threading.get_ident() == caller
        assert blosc2.blosc2_ext._wait_for_nogil_calls(timeout=0)
        calls.append(offset)
        output[:] = input + 7

    for releasegil in (False, True):
        blosc2.set_releasegil(releasegil)
        for nthreads in (0, 4):
            dst = np.empty(N // 2, dtype=np.int32)
            array.get_1d_span_numpy(dst, 0, N // 4, dst.size, nthreads=nthreads)
            np.testing.assert_array_equal(dst, values[N // 4 : 3 * N // 4] + 7)
    assert calls


def run_isolated(function, *args):
    env = {name: value for name, value in os.environ.items() if name != "BLOSC_NTHREADS"}
    script = f"import runpy; runpy.run_path({__file__!r})[{function!r}](*{args!r})"
    # Preserve an explicitly isolated test package without loading the shared editable install.
    python = [sys.executable]
    if sys.flags.no_site:
        python.append("-S")
        script = f"import sys; sys.path = {sys.path!r}; " + script
    proc = subprocess.run(
        [*python, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        check=False,
        timeout=60,
    )
    assert proc.returncode == 0, f"exit code {proc.returncode}\n{proc.stdout}\n{proc.stderr}"


@pytest.mark.skipif(blosc2.IS_WASM, reason="no subprocesses and no threads on wasm32")
@pytest.mark.parametrize("kind", ["codec", "filter"])
@pytest.mark.parametrize("loader", ["cframe", "open"])
@pytest.mark.parametrize("read_path", ["decompress_chunk", "get_slice"])
def test_registration_after_open(tmp_path, kind, loader, read_path):
    path = str(tmp_path / "callbacks.b2frame")
    run_isolated("late_registration", "write", kind, path, loader, read_path)
    run_isolated("late_registration", "read", kind, path, loader, read_path)


@pytest.mark.skipif(blosc2.IS_WASM, reason="no subprocesses and no threads on wasm32")
def test_span_postfilter_threads():
    run_isolated("span_postfilter")


if __name__ == "__main__":
    print(json.dumps(decompression_outcomes(int(sys.argv[1]))))
