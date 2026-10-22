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
            "import sys,json; sys.path[:]=json.loads(sys.argv[1]); import blosc2; "
            "blosc2.set_nthreads(2); assert blosc2.nthreads==2; "
            "assert blosc2.numexpr is None; assert 'numexpr' not in sys.modules",
            json.dumps(sys.path),
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
