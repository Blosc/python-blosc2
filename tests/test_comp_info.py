#######################################################################
# Copyright (C) 2019-present, Blosc Development team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################


import pytest
import numpy as np
import blosc2

SIZE = 1_000_000

@pytest.mark.parametrize("codec", list(blosc2.Codec))
def test_comp_info(codec):
    blosc2.compressor_list()
    blosc2.clib_info(codec)
    blosc2.set_compressor(codec)
    assert codec.name.lower() == blosc2.get_compressor()
    src = blosc2.compress2(np.zeros(SIZE, dtype="i8"), clevel=9)
    nbytes, cbytes, blocksize = blosc2.get_cbuffer_sizes(src)
    assert nbytes == SIZE * 8
    assert cbytes == blosc2.MAX_OVERHEAD
    # When raising the next limit when this would fail in the future, one should raise the SIZE too
    assert blocksize <= 2 ** 22
    blosc2.print_versions()

#@pytest.mark.parametrize("clevel", [0, 1, 5, 9])
@pytest.mark.parametrize("clevel", [1])
@pytest.mark.parametrize("shape", [
                         (1000, 1000),
                         # (10, 10, 10),
                         # (10, 10, 10, 10),
                         # (10, 10, 10, 10, 10),
                         ]
                         )
#@pytest.mark.parametrize("dtype", ["u1", "i4", "f8"])
@pytest.mark.parametrize("dtype", ["i4", "i8"])
def test_compute_chunks_blocks(clevel, shape, dtype):
    cparams = blosc2.cparams_dflts.copy()
    cparams['clevel'] = clevel
    cparams['typesize'] = np.dtype(dtype).itemsize
    chunks, blocks = blosc2.compute_chunks_blocks(shape, **cparams)
    print(chunks, blocks)
