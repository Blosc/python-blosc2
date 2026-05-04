#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

import numpy as np
import pytest

import blosc2


@pytest.mark.parametrize("contiguous", [True, False])
@pytest.mark.parametrize("urlpath", [None, "reorder_offsets.b2frame"])
@pytest.mark.parametrize("nchunks", [1, 5, 12])
def test_schunk_reorder_offsets(contiguous, urlpath, nchunks):
    blosc2.remove_urlpath(urlpath)
    schunk = blosc2.SChunk(
        chunksize=200 * 1000 * 4,
        contiguous=contiguous,
        urlpath=urlpath,
        cparams={"typesize": 4, "nthreads": 2},
        dparams={"nthreads": 2},
    )

    for i in range(nchunks):
        buffer = np.arange(200 * 1000, dtype=np.int32) + i * 200 * 1000
        assert schunk.append_data(buffer) == (i + 1)

    order = np.array([(i + 3) % nchunks for i in range(nchunks)], dtype=np.int64)
    schunk.reorder_offsets(order)

    for i in range(nchunks):
        expected = np.arange(200 * 1000, dtype=np.int32) + order[i] * 200 * 1000
        dest = np.empty(200 * 1000, dtype=np.int32)
        schunk.decompress_chunk(i, dest)
        assert np.array_equal(dest, expected)

    blosc2.remove_urlpath(urlpath)


@pytest.mark.parametrize(
    "order",
    [
        [[0, 1]],
        [0, 1],
        [0, 0, 1],
        [0, 1, 3],
    ],
)
def test_schunk_reorder_offsets_invalid_order(order):
    schunk = blosc2.SChunk(chunksize=16, cparams={"typesize": 1})
    for payload in (b"a" * 16, b"b" * 16, b"c" * 16):
        schunk.append_data(payload)

    if order == [[0, 1]] or order == [0, 1]:
        with pytest.raises(ValueError):
            schunk.reorder_offsets(order)
    else:
        with pytest.raises(RuntimeError):
            schunk.reorder_offsets(order)


def test_schunk_reorder_offsets_read_only(tmp_path):
    urlpath = tmp_path / "reorder_offsets_read_only.b2frame"
    schunk = blosc2.SChunk(chunksize=16, urlpath=urlpath, contiguous=True, cparams={"typesize": 1})
    schunk.append_data(b"a" * 16)
    schunk.append_data(b"b" * 16)

    reopened = blosc2.open(urlpath, mode="r")
    with pytest.raises(ValueError, match="reading mode"):
        reopened.reorder_offsets([1, 0])
