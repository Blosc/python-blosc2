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
    "contiguous, urlpath, cparams, nchunks, start, stop",
    [
        (True, None, {"typesize": 4}, 10, 0, 100),
        (True, "b2frame", {"typesize": 4}, 1, 7, 23),
        (
            False,
            None,
            {"splitmode": blosc2.SplitMode.ALWAYS_SPLIT, "nthreads": 5, "typesize": 4},
            5,
            21,
            200 * 2 * 100,
        ),
        (False, "b2frame", {"codec": blosc2.Codec.LZ4HC, "typesize": 4}, 7, None, None),
        (True, None, {"blocksize": 200 * 100, "typesize": 4}, 5, -2456, -234),
        (True, "b2frame", {"blocksize": 200 * 100, "typesize": 4}, 4, 2456, -234),
        (False, None, {"blocksize": 100 * 100, "typesize": 4}, 2, -200 * 100 + 234, 40000),
        (True, None, {"blocksize": 100 * 100, "typesize": 4}, 2, 0, None),
    ],
)
def test_schunk_get_slice(contiguous, urlpath, cparams, nchunks, start, stop):
    storage = {"contiguous": contiguous, "urlpath": urlpath, "cparams": cparams}
    schunk = blosc2.SChunk(chunksize=200 * 100 * 4, mode="w", **storage)
    for i in range(nchunks):
        chunk = np.full(schunk.chunksize // schunk.typesize, i, dtype=np.int32)
        schunk.append_data(chunk)

    aux = np.empty(200 * 100 * nchunks, dtype=np.int32)
    schunk.get_slice(start, stop, aux)
    if stop is None and start is not None:
        res = aux[start]
        np.array_equal(res, blosc2.get_slice_nchunks(schunk, start))
    else:
        res = aux[start:stop]
        np.array_equal(np.unique(res), blosc2.get_slice_nchunks(schunk, (start, stop)))
        # slice variant
        np.array_equal(np.unique(res), blosc2.get_slice_nchunks(schunk, slice(start, stop)))

    blosc2.remove_urlpath(urlpath)
