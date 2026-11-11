#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Span reads (NDArray.get_1d_span_numpy() and the native span readers of graph executions) decompress
# without the GIL.  Chunks of a codec or filter registered from Python need it: their callbacks run Python.
# Each case runs in a child interpreter, because the bug crashes it.

import subprocess
import sys

import pytest

import blosc2

pytestmark = pytest.mark.skipif(blosc2.IS_WASM, reason="Pyodide cannot start processes")

SETUP = """
import zlib

import numpy as np
import blosc2


def encoder(src, dst, meta, schunk):
    out = zlib.compress(src.tobytes(), 1)
    dst[: len(out)] = np.frombuffer(out, np.uint8)
    return len(out)


def decoder(src, dst, meta, schunk):
    out = zlib.decompress(src.tobytes())
    dst[: len(out)] = np.frombuffer(out, np.uint8)
    return len(out)


def reverse(src, dst, meta, schunk):
    dst[:] = src[::-1]


if KIND == "codec":
    blosc2.register_codec("span_zlib", 180, encoder, decoder)
    cparams = {"codec": 180, "nthreads": 1, "filters": [], "splitmode": blosc2.SplitMode.NEVER_SPLIT}
else:
    blosc2.register_filter(181, reverse, reverse, "span_reverse")
    cparams = {"filters": [181], "filters_meta": [0], "nthreads": 1}
"""

SPAN_SCRIPT = """
a = blosc2.asarray(np.arange(2**16, dtype=np.int32), chunks=(2**14,), blocks=(2**12,), cparams=cparams,
                   dparams={"nthreads": 1})
out = np.empty(2**13, dtype=np.int32)
for _ in range(20):
    a.get_1d_span_numpy(out, 1, 0, out.size, NTHREADS)
assert (out == np.arange(2**14, 2**14 + out.size)).all()
"""

GRAPH_SCRIPT = """
from blosc2.graph_plan import NativeGraph
from blosc2.storage_readers import ReaderState, StorageReader

values = np.arange(2**16, dtype=np.float64)
x = blosc2.asarray(values, chunks=(2**14,), blocks=(2**12,), cparams=cparams, dparams={"nthreads": 1})
reader = StorageReader(x, values.shape)
reader.start = 0
schedule = NativeGraph.from_expression("x + 1", {"x": "float64"}).specialize(
    {"x": (values.dtype, values.shape)}, tile_items=2**12
)
for _ in range(5):
    result = schedule.execute_sources({"x": x}, {"x": ReaderState(reader)}, native_spans=True)[0]
assert (result == values + 1).all()
"""


def run_child(kind, script, nthreads=0):
    code = f"KIND = {kind!r}\nNTHREADS = {nthreads}\n" + SETUP + script
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("kind", ["codec", "filter"])
@pytest.mark.parametrize("nthreads", [0, 4])
def test_get_1d_span_numpy_python_callbacks(kind, nthreads):
    run_child(kind, SPAN_SCRIPT, nthreads)


@pytest.mark.parametrize("kind", ["codec", "filter"])
def test_graph_native_spans_python_callbacks(kind, native_graph_runtime):
    if not blosc2.blosc2_ext.native_graph_sources_available():
        pytest.skip("Installed extension does not implement graph storage readers")
    run_child(kind, GRAPH_SCRIPT)
