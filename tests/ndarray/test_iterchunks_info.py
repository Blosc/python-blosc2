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
    "shape, chunks, dtype, fill_value",
    [
        ((401, 100), (200, 10), "S10", "Hola!"),  # repeated string
        ((1020, 100), (200, 20), np.bool_, False),  # zeros
        ((1000, 99), (200, 20), np.int32, 1),  # ones
        ((799, 99), (20, 20), np.float64, np.nan),  # repeated float
    ],
)
def test_iterchunks_info(shape, chunks, dtype, fill_value):
    a = blosc2.full(shape, fill_value=fill_value, chunks=chunks, dtype=dtype)
    slice_ = (slice(0, chunks[0]), slice(0, chunks[1]))
    a[slice_] = 0  # introduce a zeroed chunk (another type of special value)

    for i, info in enumerate(a.iterchunks_info()):
        # print(info)
        assert info.nchunk == i
        if info.special == blosc2.SpecialValue.NOT_SPECIAL:
            assert info.cratio >= 10
        else:
            assert info.cratio >= 50
