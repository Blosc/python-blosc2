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


if __name__ == "__main__":
    print(json.dumps(decompression_outcomes(int(sys.argv[1]))))
