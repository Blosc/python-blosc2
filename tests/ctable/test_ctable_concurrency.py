"""Bounded native-read probes; unsupported races run only via the explicit CLI.

Run this file with MODE MUTATION unsafe to investigate an unprotected
race in a disposable subprocess. Default pytest tests only caller serialization.
"""

from __future__ import annotations

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Lock, get_ident

import numpy as np
import pytest

import blosc2


def native_read_probe(mode, mutation, serialized=True):
    @dataclass
    class Row:
        x: int = blosc2.field(blosc2.int64(), chunks=(32,), blocks=(8,))

    table = blosc2.CTable(Row, [(i,) for i in range(64)], expected_size=64)
    # Fill the actual capacity: append must enter native resize, not spare space.
    while len(table) < len(table._valid_rows):
        table.append([len(table)])
    n = len(table)
    raw = table._cols["x"]
    old_shape = raw.shape
    expr = blosc2.lazyexpr("x * 2", {"x": table["x"]}, evaluation=mode)
    entered, resume, attempted, written = Event(), Event(), Event(), Event()
    operation_lock = Lock()
    reader_ident = None
    raw.schunk.dparams = blosc2.DParams(nthreads=1)

    @raw.schunk.postfilter(np.int64, np.int64)
    def pause_inside_decompression(input, output, offset):
        output[:] = input
        if get_ident() == reader_ident and not entered.is_set():
            entered.set()
            assert resume.wait(10), "native read was not resumed"

    def read():
        nonlocal reader_ident
        reader_ident = get_ident()
        if serialized:
            with operation_lock:
                return expr[:]
        return expr[:]

    def update():
        if serialized:
            # Deterministic contention evidence, not a sleep-based assertion.
            assert not operation_lock.acquire(blocking=False)
        attempted.set()
        if serialized:
            operation_lock.acquire()
        try:
            if mutation == "overwrite":
                table["x"][0] = 91
            else:
                table.append([91])
                assert raw.shape[0] > old_shape[0]
            written.set()
        finally:
            if serialized:
                operation_lock.release()

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            reading = pool.submit(read)
            writing = None
            try:
                assert entered.wait(10), "no postfilter inside the native read"
                writing = pool.submit(update)
                assert attempted.wait(10), "writer did not attempt access"
                if serialized:
                    assert not written.is_set()
                else:
                    writing.result(timeout=10)
                    assert written.is_set()
                    assert not resume.is_set()
                    print("WRITE_COMPLETED_INSIDE_NATIVE_READ", flush=True)
            finally:
                resume.set()
            values = reading.result(timeout=10)
            if writing is not None:
                writing.result(timeout=10)
        if serialized:
            np.testing.assert_array_equal(values, np.arange(n) * 2)
        current = np.arange(n + int(mutation == "append"))
        current[0 if mutation == "overwrite" else -1] = 91
        # A full-mode graph can retain its original shape; use a fresh graph
        # for the post-append check rather than changing that existing contract.
        fresh = blosc2.lazyexpr("x * 2", {"x": table["x"]}, evaluation=mode)
        np.testing.assert_array_equal(fresh[:], current * 2)
        return {
            "native_overlap": not serialized,
            "rows_read": len(values),
            "rows_now": len(table),
            "read_matches_before": bool(np.array_equal(values, np.arange(n) * 2)),
            "read_matches_after_original_extent": bool(np.array_equal(values, current[:n] * 2)),
            "read_first_values": values[:8].tolist(),
        }
    finally:
        raw.schunk.remove_postfilter("pause_inside_decompression")
        table.close()


@pytest.mark.skipif(sys.platform in {"emscripten", "wasi"}, reason="Requires threads and subprocesses")
@pytest.mark.parametrize("mode", ["safe", "full"])
@pytest.mark.parametrize("mutation", ["overwrite", "append"])
def test_caller_lock_serializes_native_table_read_and_mutation(mode, mutation):
    result = subprocess.run(
        [sys.executable, "-X", "faulthandler", str(Path(__file__).resolve()), mode, mutation, "serialized"],
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout.strip())
    assert report["native_overlap"] is False


if __name__ == "__main__":
    print(json.dumps(native_read_probe(sys.argv[1], sys.argv[2], sys.argv[3] != "unsafe")))
