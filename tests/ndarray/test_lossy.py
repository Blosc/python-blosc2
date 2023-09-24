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
    "shape, dtype, cparams, urlpath, contiguous",
    [
        (
            (32, 18),
            np.float32,
            {"codec": blosc2.Codec.NDLZ, "codec_meta": 4},
            None,
            False,
        ),
        (
            (100, 1230),
            np.float64,
            {"codec": blosc2.Codec.ZFP_ACC, "codec_meta": 37},
            None,
            False,
        ),
        (
            (23, 34),
            np.float64,
            {"codec": blosc2.Codec.ZFP_PREC, "codec_meta": 37},
            "lossy.b2nd",
            True,
        ),
        (
            (80, 51, 60),
            np.float32,
            {"codec": blosc2.Codec.ZFP_RATE, "codec_meta": 37},
            "lossy.b2nd",
            False,
        ),
        (
            (13, 13),
            np.int32,
            {"filters": [blosc2.Filter.NDMEAN], "filters_meta": [4]},
            None,
            True,
        ),
        (
            (10, 10),
            np.int64,
            {"filters": [blosc2.Filter.NDCELL], "filters_meta": [4]},
            None,
            False,
        ),
    ],
)
def test_lossy(shape, cparams, dtype, urlpath, contiguous):
    if cparams.get("codec") == blosc2.Codec.NDLZ:
        dtype = np.uint8
    array = np.linspace(0, np.prod(shape), np.prod(shape), dtype=dtype).reshape(shape)
    a = blosc2.asarray(array, cparams=cparams, urlpath=urlpath, contiguous=contiguous, mode="w")

    if (
        a.schunk.cparams["codec"] in (blosc2.Codec.ZFP_RATE, blosc2.Codec.ZFP_PREC, blosc2.Codec.ZFP_ACC)
        or a.schunk.cparams["filters"][0] == blosc2.Filter.NDMEAN
    ):
        _ = a[...]
    else:
        tol = 1e-5
        if dtype in (np.float32, np.float64):
            np.testing.assert_allclose(a[...], array, rtol=tol, atol=tol)
        else:
            np.array_equal(a[...], array)

    blosc2.remove_urlpath(urlpath)
