#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

import json
import os
import subprocess
import sys
import threading

import pytest

import blosc2


def test_numexpr_environment_does_not_control_blosc_threads():
    """Legacy NumExpr environment variables do not control Blosc2 workers."""
    # Inherit the current environment but set NUMEXPR_MAX_THREADS to a low value
    if blosc2.IS_WASM:
        pytest.skip("subprocess is not supported on emscripten/wasm32")

    env = os.environ.copy()
    env["NUMEXPR_MAX_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"

    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            "import sys,json,importlib.util; sys.path[:]=json.loads(sys.argv[1]); "
            "package=json.loads(sys.argv[2]); "
            "spec=importlib.util.spec_from_file_location('blosc2',package['origin'], "
            "submodule_search_locations=package['paths']); "
            "blosc2=importlib.util.module_from_spec(spec); sys.modules['blosc2']=blosc2; "
            "spec.loader.exec_module(blosc2); "
            "blosc2.set_nthreads(2); assert blosc2.nthreads==2; "
            "assert not hasattr(blosc2, 'numexpr'); assert 'numexpr' not in sys.modules",
            json.dumps(sys.path),
            json.dumps({"origin": blosc2.__file__, "paths": list(blosc2.__path__)}),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )

    # Check that no warning about NUMEXPR_MAX_THREADS was printed
    assert "NUMEXPR_MAX_THREADS" not in result.stderr, (
        f"Unexpected numexpr warning in stderr: {result.stderr}"
    )
    assert "nthreads cannot be larger" not in result.stderr, (
        f"Unexpected numexpr warning in stderr: {result.stderr}"
    )


def _import_blosc2_and_print(env_update, code):
    env = os.environ.copy()
    env.pop("BLOSC_NTHREADS", None)
    env.update(env_update)
    result = subprocess.run(
        [sys.executable, "-c", f"import blosc2; {code}"], capture_output=True, text=True, env=env, check=True
    )
    return result.stdout.split()


@pytest.mark.skipif(blosc2.IS_WASM, reason="subprocess is not supported on emscripten/wasm32")
def test_blosc_nthreads_env():
    # C-Blosc2 uses BLOSC_NTHREADS for every compression and decompression context; nthreads and
    # the parameter defaults must say so
    code = (
        "a = blosc2.zeros(10); "
        "print(blosc2.nthreads, blosc2.CParams().nthreads, blosc2.DParams().nthreads, a.cparams.nthreads)"
    )
    assert _import_blosc2_and_print({"BLOSC_NTHREADS": "3"}, code) == ["3"] * 4
    # C-Blosc2 reads the leading integer (strtol)
    assert _import_blosc2_and_print({"BLOSC_NTHREADS": "4abc"}, code) == ["4"] * 4
    default = _import_blosc2_and_print({}, "print(blosc2.nthreads)")
    # One nonpositive integer and one malformed value cover the two rejection
    # classes; repeated strtol spellings only pay for additional package imports.
    for invalid in ["0", "abc"]:
        assert _import_blosc2_and_print({"BLOSC_NTHREADS": invalid}, "print(blosc2.nthreads)") == default


@pytest.mark.skipif(blosc2.IS_WASM, reason="threads are not supported on emscripten/wasm32")
def test_set_nthreads_concurrent_sync():
    old = blosc2.nthreads
    try:
        for _ in range(10):
            barrier = threading.Barrier(4)
            errors = []

            def worker(tid, barrier=barrier, errors=errors):
                try:
                    barrier.wait()
                    for i in range(50):
                        blosc2.set_nthreads(1 + ((tid + i) % 4))
                except Exception as exc:
                    errors.append(exc)

            threads = [threading.Thread(target=worker, args=(tid,)) for tid in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert not errors
            py_nthreads = blosc2.nthreads
            c_nthreads = blosc2.set_nthreads(old)
            assert c_nthreads == py_nthreads
    finally:
        blosc2.set_nthreads(old)
