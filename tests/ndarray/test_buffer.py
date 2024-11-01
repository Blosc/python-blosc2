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
    "shape, chunks, blocks, dtype, urlpath, contiguous, meta",
    [
        ([450], [128], [25], "|S8", "frombuffer.b2nd", True, None),
        ([20, 134, 13], [3, 13, 5], [3, 10, 5], np.complex128, "frombuffer.b2nd", False, {"123": 123}),
        ([45], [12], [6], "|S4", None, True, None),
        ([30, 29], [15, 28], [5, 27], np.int16, None, False, {"2": 123, "meta2": "abcdef"}),
    ],
)
def test_buffer(shape, chunks, blocks, dtype, urlpath, contiguous, meta):
    blosc2.remove_urlpath(urlpath)

    dtype = np.dtype(dtype)
    typesize = dtype.itemsize
    size = int(np.prod(shape))
    buffer = bytes(size * typesize)
    a = blosc2.frombuffer(
        buffer,
        shape,
        chunks=chunks,
        blocks=blocks,
        dtype=dtype,
        urlpath=urlpath,
        contiguous=contiguous,
        meta=meta,
    )
    buffer2 = a.tobytes()
    assert buffer == buffer2

    blosc2.remove_urlpath(urlpath)


@pytest.mark.parametrize(
    "shape, dtype",
    [
        ([450], "|S8"),
        ([20, 134, 13], np.complex128),
        ([45], "|S4"),
        ([30, 29], np.int16),
    ],
)
def test_buffer_simple(shape, dtype):
    dtype = np.dtype(dtype)
    typesize = dtype.itemsize
    size = int(np.prod(shape))
    buffer = bytes(size * typesize)
    a = blosc2.frombuffer(buffer, shape, dtype=dtype)
    buffer2 = a.tobytes()
    assert buffer == buffer2
