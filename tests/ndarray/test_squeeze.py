#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

import numpy as np
import pytest

import blosc2


@pytest.mark.parametrize(
    ("shape", "chunks", "blocks", "fill_value", "mask"),
    [
        ((1, 1230), (1, 100), (1, 3), b"0123", [True, False]),
        ((23, 1, 1, 34), (20, 1, 1, 20), None, 1234, [False, False, True, False]),
        ((80, 1, 51, 60, 1), None, (6, 1, 6, 26, 1), 3.333, [False] * 4 + [True]),
        ((1, 1, 1), None, None, True, [False, True, True]),
    ],
)
def test_squeeze(shape, chunks, blocks, fill_value, mask):
    a = blosc2.full(shape, fill_value=fill_value, chunks=chunks, blocks=blocks)

    b = np.squeeze(a[...], tuple(i for i, m in enumerate(mask) if m))
    a_ = a.squeeze(mask)

    assert a_.shape == b.shape
    # TODO: this would work if squeeze returns a view
    # assert a_.shape != a.shape
