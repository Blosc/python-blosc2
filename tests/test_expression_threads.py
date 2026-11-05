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
    assert _import_blosc2_and_print({"BLOSC_NTHREADS": "2.5"}, code) == ["2"] * 4
    default = _import_blosc2_and_print({}, "print(blosc2.nthreads)")
    for invalid in ["0", "-2", "abc", ""]:
        assert _import_blosc2_and_print({"BLOSC_NTHREADS": invalid}, "print(blosc2.nthreads)") == default


@pytest.mark.skipif(blosc2.IS_WASM, reason="subprocess is not supported on emscripten/wasm32")
def test_numexpr_num_threads_env_is_kept():
    # set_nthreads() recommends NUMEXPR_NUM_THREADS for numexpr; importing blosc2 must not override it
    code = "import numexpr; print(numexpr.get_num_threads())"
    assert _import_blosc2_and_print({"NUMEXPR_NUM_THREADS": "2"}, code) == ["2"]
    assert _import_blosc2_and_print({"BLOSC_NTHREADS": "3"}, code) == ["3"]


@pytest.mark.skipif(blosc2.IS_WASM, reason="subprocess is not supported on emscripten/wasm32")
def test_blosc_nthreads_env_beyond_numexpr_max():
    # nthreads says what C-Blosc2 uses; numexpr gets as many threads as it takes, without errors
    env = os.environ.copy()
    for name in ("NUMEXPR_NUM_THREADS", "NUMEXPR_MAX_THREADS"):
        env.pop(name, None)
    env["BLOSC_NTHREADS"] = "100"
    code = "import blosc2, numexpr; print(blosc2.nthreads, numexpr.get_num_threads(), numexpr.MAX_THREADS)"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True
    )
    assert "nthreads cannot be larger" not in result.stdout + result.stderr
    nthreads, numexpr_nthreads, numexpr_max = map(int, result.stdout.split())
    assert nthreads == 100
    assert numexpr_nthreads == min(100, numexpr_max)
