#######################################################################
# Copyright (c) 2026, Blosc Development Team <blosc@blosc.org>
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################
"""Snapshot a scalar capture and execute the portable artifact without its author."""

import numpy as np

import blosc2

SCALE = 2.0


@blosc2.dsl_kernel
def scaled(x):
    return x * SCALE


if __name__ == "__main__":
    artifact = scaled.export({"x": "float64"}, "float64")
    SCALE = 99.0
    imported = blosc2.PortableKernel.from_json(artifact, jit=False)
    values = imported.evaluate({"x": np.arange(4, dtype=np.float64)})
    np.testing.assert_array_equal(values, [0, 2, 4, 6])
    print(values)
