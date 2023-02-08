#######################################################################
# Copyright (C) 2019-present, Blosc Development team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

import pytest

import blosc2
import numpy as np


@pytest.mark.parametrize("shape, chunks, blocks, typesize",
                         [
                             ([450], [128], [25], 8),
                             ([20, 134, 13], [3, 13, 5], [3, 10, 5], 4),
                         ])
def test_buffer(shape, chunks, blocks, typesize):
    size = int(np.prod(shape))
    buffer = bytes(size * typesize)
    a = blosc2.from_buffer(buffer, shape, chunks, blocks, typesize)
    buffer2 = a.to_buffer()
    assert buffer == buffer2
