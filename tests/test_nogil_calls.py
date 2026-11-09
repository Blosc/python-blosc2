#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Every C call that runs without the GIL is counted, so that the exit (blosc2_destroy() frees the global
# context and the thread pools) and register_codec()/register_filter() wait for it.  Both tests run in a
# new interpreter: no codec or filter registered from Python by other tests may keep the GIL.

import json
import subprocess
import sys

import pytest

import blosc2

pytestmark = pytest.mark.skipif(blosc2.IS_WASM, reason="Pyodide cannot start threads or processes")

COUNTED_SCRIPT = """
import json
import threading
import time

import numpy as np
import blosc2
from blosc2 import blosc2_ext

blosc2.set_releasegil(True)
data = (np.arange(2**20) % 1000).astype(np.int32)  # 4 MiB
cparams = {"typesize": 4, "nthreads": 1, "codec": blosc2.Codec.ZSTD, "clevel": 5}
chunk = blosc2.compress2(data, **cparams)
schunk = blosc2.SChunk(chunksize=data.nbytes, cparams=cparams)
sparse = blosc2.SChunk(chunksize=data.nbytes // 8, data=data, cparams=cparams, contiguous=False)
array = blosc2.asarray(data.reshape(1024, 1024), chunks=(256, 1024), cparams=cparams)
flat = blosc2.asarray(data, chunks=(2**18,), cparams=cparams)
span = np.empty(2**18, dtype=np.int32)


def append():
    schunk.append_data(data)
    schunk.delete_chunk(0)


calls = {
    "compress": lambda: blosc2.compress(data, typesize=4, clevel=5, codec=blosc2.Codec.ZSTD),
    "compress2": lambda: blosc2.compress2(data, **cparams),
    "decompress2": lambda: blosc2.decompress2(chunk),
    "vlcompress": lambda: blosc2_ext.vlcompress([data[:1000], data[1000:]], **cparams),
    "append_data": append,
    "to_cframe": sparse.to_cframe,
    "NDArray slice": lambda: array[:],
    "get_1d_span_numpy": lambda: flat.get_1d_span_numpy(span, 0, 0, span.size),
}

# Native graph and Menudet routes, where this build and NumPy support them
values = np.arange(2**20, dtype=np.float64)
try:
    from blosc2.graph_plan import NativeGraph
    from blosc2.storage_readers import ReaderState, StorageReader

    if not blosc2_ext.native_graph_sources_available():
        raise NotImplementedError
    compressed = blosc2.asarray(values, chunks=(2**18,), blocks=(2**14,))
    schedule = NativeGraph.from_expression("x + 1", {"x": "float64"}).specialize(
        {"x": (values.dtype, values.shape)}, tile_items=2**14
    )

    def graph_with_native_spans():
        reader = StorageReader(compressed, values.shape)
        reader.start = 0
        schedule.execute_sources({"x": compressed}, {"x": ReaderState(reader)}, native_spans=True)

    calls["graph with native spans"] = graph_with_native_spans
except Exception:
    pass
try:
    from blosc2.native_graph import prepare_resident

    if not blosc2_ext.menudet_resident_prefilter_available():
        raise NotImplementedError
    with blosc2.jit_options(jit=False, jit_backend="cc", math_backend="system"):
        resident = prepare_resident("x*x - 1", {"x": values}, jit=False, optimization="aggressive")[2]
    executor = resident.kernel._handle.resident_prefilter(
        values.shape, workers=2, block_items=2**14, chunk_items=2**18, graph=resident.plan._handle
    )
    calls["resident prefilter"] = lambda: executor.evaluate({"x": values})
except Exception:
    pass
problems = []
for name, call in calls.items():
    stop = threading.Event()

    def loop():
        while not stop.is_set():
            call()

    thread = threading.Thread(target=loop)
    thread.start()
    deadline = time.monotonic() + 10
    seen = False
    try:
        while not seen and time.monotonic() < deadline:
            time.sleep(0.001)
            # With the GIL held here, the other thread is either inside the call without it, or not in it
            seen = not blosc2_ext._wait_for_nogil_calls(timeout=0)
    finally:
        stop.set()
        thread.join()
    if not seen:
        problems.append(f"{name}: not counted")
    if not blosc2_ext._wait_for_nogil_calls(timeout=0):
        problems.append(f"{name}: still counted after it returned")
print(json.dumps(problems))
"""

# Two daemon threads compress() (the global context) while the interpreter exits
EXIT_SCRIPT = """
import threading
import time

import numpy as np
import blosc2

blosc2.set_releasegil(True)
# Blocks this large make the global context's temporaries big enough to be mmap()ed: freeing them under
# the compressing thread crashes it instead of going unnoticed
blosc2.set_blocksize(4 * 2**20)
data = (np.arange(4 * 2**20) % 1000).astype(np.int32)  # 16 MiB


def compress():
    while True:
        blosc2.compress(data, typesize=4, codec=blosc2.Codec.ZSTD, clevel=3)


for _ in range(2):
    threading.Thread(target=compress, daemon=True).start()
time.sleep(0.2)
"""


def run_python(code):
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)


def test_calls_without_the_gil_are_counted():
    result = run_python(COUNTED_SCRIPT)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == []


def test_exit_waits_for_daemon_threads_in_calls_without_the_gil():
    # Without the waiting, 4 out of 5 runs crashed with SIGSEGV on Linux
    for _ in range(3):
        result = run_python(EXIT_SCRIPT)
        assert result.returncode == 0, result.stderr
