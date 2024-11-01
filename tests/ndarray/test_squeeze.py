#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

import pytest

import blosc2


@pytest.mark.parametrize(
    "shape, chunks, blocks, fill_value",
    [
        ((1, 1230), (1, 100), (1, 3), b"0123"),
        ((23, 1, 1, 34), (20, 1, 1, 20), None, 1234),
        ((80, 1, 51, 60, 1), None, (6, 1, 6, 26, 1), 3.333),
        ((1, 1, 1), None, None, True),
    ],
)
def test_squeeze(shape, chunks, blocks, fill_value):
    a = blosc2.full(shape, fill_value=fill_value, chunks=chunks, blocks=blocks)

    a.squeeze()
    b = a[...]

    assert a.shape == b.shape
