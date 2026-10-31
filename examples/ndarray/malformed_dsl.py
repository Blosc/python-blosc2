########################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# SPDX-License-Identifier: BSD-3-Clause
########################################################################
"""Validate unsupported source and show strict Menudet execution errors."""

import numpy as np

import blosc2

report = blosc2.validate_menudet("def invalid(x):\n    return callback(x)\n", {"x": "float64"}, "float64")
assert not report["valid"]
print("Validation error:", report["error"])


@blosc2.menudet(jit=False)
def checked_cast(x):
    return int64(x)  # noqa: F821 -- native Menudet conversion, not a Python callback


try:
    checked_cast(np.array([np.inf]))[:]
except blosc2.MenudetError as error:
    print("Native execution error (no Python replay):", error)
else:
    raise AssertionError("Nonfinite integer conversion must fail")
