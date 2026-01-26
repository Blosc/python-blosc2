#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

import os
import subprocess
import sys


def test_numexpr_max_threads_no_warning():
    """Test that importing blosc2 with NUMEXPR_MAX_THREADS set does not produce a warning.

    When NUMEXPR_MAX_THREADS is set to a value lower than the number of threads
    blosc2 would use, we should NOT call numexpr.set_num_threads() to avoid
    the numexpr warning being printed to stderr.
    """
    # Inherit the current environment but set NUMEXPR_MAX_THREADS to a low value
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
