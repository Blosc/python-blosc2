#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

import numpy as np
import pytest

import blosc2


@pytest.mark.parametrize("nthreads", [1, 4])
def test_chunk_range(tmp_path, nthreads):
    path = str(tmp_path / "chunk_range.b2nd")
    a = blosc2.zeros(
        (40, 50, 61),
        dtype=np.int32,
        chunks=(10, 20, 16),
        blocks=(5, 8, 8),
        urlpath=path,
        cparams={"nthreads": nthreads},
    )
    # data in part of the array only, so that the other chunks stay special values (zeros) in the index
    values = np.arange(40 * 30 * 61, dtype=np.int64).reshape(40, 30, 61) * 2654435761 % 1000003
    a[:, :30] = values.astype(np.int32)
    del a
    a = blosc2.open(path, mode="r")
    with open(path, "rb") as f:
        raw = f.read()
    nspecial = 0
    for nchunk in range(a.schunk.nchunks):
        offset, nbytes = a.schunk.get_chunk_range(nchunk)
        if nbytes == 0:
            nspecial += 1
            continue
        # written in order: exactly the chunk's bytes
        assert raw[offset : offset + nbytes] == a.schunk.get_chunk(nchunk)
    assert nspecial > 0
    with pytest.raises(RuntimeError):
        a.schunk.get_chunk_range(a.schunk.nchunks)
    with pytest.raises(RuntimeError):
        blosc2.SChunk(chunksize=100).get_chunk_range(0)
