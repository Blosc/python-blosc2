#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################


import pytest

import blosc2


@pytest.mark.parametrize(("nthreads", "blocksize"), [(2, 0), (1, 30), (4, 5)])
def test_compression_parameters(nthreads, blocksize):
    blosc2.set_nthreads(nthreads)
    blosc2.set_blocksize(blocksize)
