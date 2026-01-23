#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

import os

import numpy as np
import pytest

import blosc2


@pytest.mark.parametrize(
    "contiguous",
    [
        True,
        False,
    ],
)
@pytest.mark.parametrize(
    ("shape", "chunks", "blocks", "urlpath", "dtype"),
    [
        ([556], [221], [33], "testmeta00.b2nd", np.float64),
        ([20, 134, 13], [12, 66, 8], [3, 13, 5], "testmeta01.b2nd", np.int32),
        ([12, 13, 14, 15, 16], [8, 9, 4, 12, 9], [2, 6, 4, 5, 4], "testmeta02.b2nd", np.float32),
    ],
)
def test_metalayers(shape, chunks, blocks, urlpath, contiguous, dtype):
    blosc2.remove_urlpath(urlpath)

    numpy_meta = {b"dtype": str(np.dtype(dtype))}
    test_meta = {b"lorem": 1234}

    # Create an empty b2nd array (on disk)
    a = blosc2.empty(
        shape,
        chunks=chunks,
        blocks=blocks,
        dtype=dtype,
        storage=blosc2.Storage(
            urlpath=urlpath, contiguous=contiguous, meta={"numpy": numpy_meta, "test": test_meta}
        ),
    )
    assert os.path.exists(urlpath)

    assert "numpy" in a.schunk.meta
    assert "error" not in a.schunk.meta
    assert a.schunk.meta["numpy"] == numpy_meta
    assert "test" in a.schunk.meta
    assert a.schunk.meta["test"] == test_meta

    test_meta = {b"lorem": 4231}
    a.schunk.meta["test"] = test_meta
    assert a.schunk.meta["test"] == test_meta

    # Remove file on disk
    blosc2.remove_urlpath(urlpath)
