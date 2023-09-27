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


@pytest.mark.parametrize("mode", ["r", "w", "a"])
@pytest.mark.parametrize("urlpath", ["test_mode.b2nd"])
@pytest.mark.parametrize(
    "shape, fill_value, dtype, cparams, dparams, contiguous",
    [
        (
            (80, 51, 60),
            3.14,
            np.float64,
            {"codec": blosc2.Codec.ZLIB, "clevel": 5, "use_dict": False, "nthreads": 2},
            {"nthreads": 1},
            False,
        ),
        (
            (13, 13),
            123456789,
            None,
            {"codec": blosc2.Codec.LZ4HC, "clevel": 8, "use_dict": False, "nthreads": 2},
            {"nthreads": 2},
            True,
        ),
    ],
)
def test_mode(shape, fill_value, cparams, dparams, dtype, urlpath, contiguous, mode):
    blosc2.remove_urlpath(urlpath)
    if mode == "r":
        with pytest.raises(ValueError):
            blosc2.full(
                shape,
                fill_value,
                dtype=dtype,
                cparams=cparams,
                dparams=dparams,
                urlpath=urlpath,
                contiguous=contiguous,
                mode=mode,
            )
    _ = blosc2.full(
        shape,
        fill_value,
        dtype=dtype,
        cparams=cparams,
        dparams=dparams,
        urlpath=urlpath,
        contiguous=contiguous,
    )

    a = blosc2.open(urlpath, mode=mode)
    if mode == "r":
        with pytest.raises(ValueError):
            a[...] = 0
        with pytest.raises(ValueError):
            a.resize([50] * a.ndim)
    else:
        a[...] = 0
        a.resize([50] * a.ndim)

    blosc2.remove_urlpath(urlpath)
