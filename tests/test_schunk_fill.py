#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

import os

import numpy as np
import pytest

import blosc2


@pytest.mark.parametrize("contiguous", [True, False])
@pytest.mark.parametrize("urlpath", [None, "b2frame"])
@pytest.mark.parametrize(
    "cparams, nitems",
    [
        ({"codec": blosc2.Codec.LZ4, "clevel": 6, "typesize": 4}, 0),
        ({"typesize": 4}, 200 * 1000),
        ({"splitmode": blosc2.SplitMode.ALWAYS_SPLIT, "nthreads": 5, "typesize": 4}, 200 * 1000 * 2 + 17),
    ],
)
@pytest.mark.parametrize("special_value, expected_value",
                         [
                             (blosc2.SpecialValue.ZERO, 0),
                             (blosc2.SpecialValue.NAN, np.nan),
                             (blosc2.SpecialValue.UNINIT, 0),
                             (blosc2.SpecialValue.VALUE, 34),
                             (blosc2.SpecialValue.VALUE, np.pi),
                             (blosc2.SpecialValue.VALUE, b"0123"),
                             (blosc2.SpecialValue.VALUE, True),
                         ],
)
def test_schunk_fill_special(contiguous, urlpath, cparams, nitems, special_value, expected_value):
    storage = {"contiguous": contiguous, "urlpath": urlpath, "cparams": cparams}
    blosc2.remove_urlpath(urlpath)

    chunk_len = 200 * 1000
    schunk = blosc2.SChunk(chunksize=chunk_len * 4, **storage)
    if special_value in [blosc2.SpecialValue.ZERO, blosc2.SpecialValue.NAN, blosc2.SpecialValue.UNINIT]:
        schunk.fill_special(nitems, special_value)
    else:
        schunk.fill_special(nitems, special_value, expected_value)
    assert len(schunk) == nitems

    if special_value != blosc2.SpecialValue.UNINIT:
        dtype = np.int32
        if isinstance(expected_value, float):
            dtype = np.float32
        elif isinstance(expected_value, bytes):
            dtype = np.dtype('|S' + str(len(expected_value)))
        array = np.full(nitems, expected_value, dtype=dtype)
        dest = np.empty(nitems, dtype=dtype)
        schunk.get_slice(out=dest)
        print(dest[:10])
        if dtype in [np.float32, np.float64]:
            np.testing.assert_allclose(dest, array)
        else:
            np.testing.assert_equal(dest, array)

    blosc2.remove_urlpath(urlpath)
