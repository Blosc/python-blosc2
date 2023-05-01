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

################# Automatic compute of optional chunks and blocks #################
# The exact outcome of these depends on many aspects, including CPUs cache sizes,
# so what is done here is mainly a shallow sanity check.  Enable the prints in
# case you want a detailed view of the guesses.
###################################################################################


@pytest.mark.parametrize("clevel", [0, 1, 5, 9])
@pytest.mark.parametrize("codec", [blosc2.Codec.BLOSCLZ, blosc2.Codec.ZSTD])
@pytest.mark.parametrize(
    "shape",
    [
        (0, 1000),
        (1000, 1000),
        (10, 20, 30),
        (10, 30, 50, 10),
        (10, 10, 10, 10, 10),
    ],
)
@pytest.mark.parametrize("dtype", ["u1", "i4", "f8"])
def test_compute_chunks_blocks(clevel, codec, shape: tuple, dtype):
    cparams = blosc2.cparams_dflts.copy()
    cparams["clevel"] = clevel
    cparams["codec"] = codec
    cparams["typesize"] = np.dtype(dtype).itemsize
    if 0 in shape:
        # shapes with 0 should be reported as invalid
        with pytest.raises(ValueError):
            blosc2.compute_chunks_blocks(shape, **cparams)
        return
    else:
        chunks, blocks = blosc2.compute_chunks_blocks(shape, **cparams)
    # print(chunks, blocks)
    for i in range(len(shape)):
        assert shape[i] >= chunks[i]
        assert chunks[i] >= blocks[i]


@pytest.mark.parametrize(
    "shape, blocks",
    [
        ((1000, 1000), (10, 10)),
        ((10, 20, 30), (1, 2, 3)),
        ((10, 30, 50, 10), (10, 30, 50, 10)),
        ((10, 10, 10, 10, 10), (10, 10, 10, 9, 10)),
        ((100, 10, 20, 100, 10), (10, 10, 10, 9, 10)),
        ((1000, 10, 20, 100, 10), (100, 10, 10, 90, 10)),
    ],
)
def test_compute_chunks(shape: tuple, blocks: tuple):
    chunks, blocks = blosc2.compute_chunks_blocks(shape, blocks=blocks)
    # print(chunks, blocks)
    for i in range(len(shape)):
        assert shape[i] >= chunks[i]
        assert chunks[i] >= blocks[i]


# Invalid blocks
@pytest.mark.parametrize(
    "shape, blocks",
    [
        ((10, 10), (100, 100)),
        ((1000, 1000), (0, 10)),
        ((10, 20, 30), (1, 2, 31)),
        ((10, 20, 30), (1, 2)),
        ((1000, 10, 20, 100, 10), (100, 11, 10, 90, 10)),
    ],
)
def test_compute_chunks_except(shape: tuple, blocks: tuple):
    with pytest.raises(ValueError):
        blosc2.compute_chunks_blocks(shape, blocks=blocks)


@pytest.mark.parametrize(
    "shape, chunks",
    [
        ((10, 10), (100, 100)),
        ((1000, 1000), (10, 10)),
        ((10, 20, 30), (1, 2, 3)),
        ((10, 30, 50, 10), (10, 30, 50, 10)),
        ((10, 10, 10, 10, 10), (10, 10, 10, 9, 10)),
        ((100, 10, 20, 100, 10), (10, 11, 10, 9, 10)),
        ((1000, 10, 20, 100, 10), (100, 11, 10, 90, 10)),
    ],
)
def test_compute_blocks(shape: tuple, chunks: tuple):
    chunks, blocks = blosc2.compute_chunks_blocks(shape, chunks=chunks)
    # print(chunks, blocks)
    for i in range(len(shape)):
        # assert shape[i] >= chunks[i]  # chunks can exceed shape if user wants to
        assert chunks[i] >= blocks[i]


@pytest.mark.parametrize(
    "shape, chunks",
    [
        ((1000, 1000), (0, 10)),
        ((1000, 1000), (10,)),
    ],
)
def test_compute_blocks_except(shape: tuple, chunks: tuple):
    with pytest.raises(ValueError):
        blosc2.compute_chunks_blocks(shape, chunks=chunks)
