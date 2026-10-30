#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

import os
import subprocess
import sys

import pytest

import blosc2


def test_numexpr_max_threads_no_warning():
    """Test that importing blosc2 with NUMEXPR_MAX_THREADS set does not produce a warning.

    When NUMEXPR_MAX_THREADS is set to a value lower than the number of threads
    blosc2 would use, we should NOT call numexpr.set_num_threads() to avoid
    the numexpr warning being printed to stderr.
    """
    # Inherit the current environment but set NUMEXPR_MAX_THREADS to a low value
    if blosc2.IS_WASM:
        pytest.skip("subprocess is not supported on emscripten/wasm32")

    env = os.environ.copy()
    env["NUMEXPR_MAX_THREADS"] = "1"

    result = subprocess.run(
        [sys.executable, "-c", "import blosc2; print(blosc2.__version__)"],
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
