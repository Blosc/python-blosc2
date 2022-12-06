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


@pytest.mark.parametrize("id, dtype",
                         [
                            (160, np.dtype(np.int32)),
                            (180, np.dtype(np.float64)),
                            (255, np.dtype(np.uint8)),
                         ])
@pytest.mark.parametrize(
    "nchunks, contiguous, urlpath",
    [
        (2, True, None),
        (1, True, "test_filter.b2frame"),
        (5, False, None),
        (3, False, "test_filter.b2frame"),
    ],
)
def test_ufilters(contiguous, urlpath, nchunks, id, dtype):
    blosc2.remove_urlpath(urlpath)

    cparams = {"nthreads": 1, "filters": [id], "filters_meta": [0]}
    dparams = {"nthreads": 1}
    chunk_len = 20 * 1000

    def forward(input, output, meta, schunk):
        nd_input = input.view(dtype)
        nd_output = output.view(dtype)

        nd_output[:] = nd_input + 1

    def backward(input, output, meta, schunk):
        nd_input = input.view(dtype)
        nd_output = output.view(dtype)

        nd_output[:] = nd_input - 1

    if id not in blosc2.ufilters_registry:
        blosc2.register_filter(id, forward, backward)
    if "f" in dtype.str:
        data = np.linspace(0, 50, chunk_len * nchunks, dtype=dtype)
    else:
        fill_value = 341 if dtype == np.int32 else 33
        data = np.full(chunk_len * nchunks, fill_value, dtype=dtype)

    schunk = blosc2.SChunk(chunksize=chunk_len * dtype.itemsize, data=data,
                           contiguous=contiguous, urlpath=urlpath, cparams=cparams, dparams=dparams)

    out = np.empty(chunk_len * nchunks, dtype=dtype)
    schunk.get_slice(0, chunk_len * nchunks, out=out)
    if "f" in dtype.str:
        assert np.allclose(data, out)
    else:
        assert np.array_equal(data, out)

    blosc2.remove_urlpath(urlpath)
