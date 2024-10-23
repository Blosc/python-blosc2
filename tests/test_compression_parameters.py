#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################


import blosc2
import pytest


@pytest.mark.parametrize("nthreads, blocksize", [(2, 0), (1, 30), (4, 5)])
def test_compression_parameters(nthreads, blocksize):
    blosc2.set_nthreads(nthreads)
    blosc2.set_blocksize(blocksize)
