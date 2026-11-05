#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

import os
import sys

import numpy as np
import pytest

import blosc2
from conftest import orthogonal_cases


@pytest.mark.parametrize(
    ("storage_case", "counts", "gil"),
    orthogonal_cases(
        [(True, None), (False, None), (True, "b2frame"), (False, "b2frame")],
        [(7, 3), (1, 1), (0, 0)],
        [True, False],
    ),
)
@pytest.mark.parametrize(("create_chunk", "copy"), [(True, True), (False, True), (False, False)])
def test_schunk_update_numpy(storage_case, counts, copy, create_chunk, gil):
    contiguous, urlpath = storage_case
    nchunks, nupdates = counts
    blosc2.set_releasegil(gil)
    kwargs = {
        "contiguous": contiguous,
        "urlpath": urlpath,
        "cparams": {"nthreads": 2},
        "dparams": {"nthreads": 2},
    }
    blosc2.remove_urlpath(urlpath)

    schunk = blosc2.SChunk(chunksize=200 * 1000 * 4, **kwargs)
    expected_chunks = []
    for i in range(nchunks):
        buffer = i * np.arange(200 * 1000, dtype="int32")
        nchunks_ = schunk.append_data(buffer)
        assert nchunks_ == (i + 1)
        expected_chunks.append(buffer.tobytes())

    for i in range(nupdates):
        pos = (0, nchunks - 1, nchunks // 2)[i % 3]
        buffer = (nchunks + i + 1) * np.arange(200 * 1000, dtype="int32")
        if create_chunk:
            chunk = blosc2.compress2(buffer)
            schunk.update_chunk(pos, chunk)
        else:
            schunk.update_data(pos, buffer, copy)
        chunk_ = schunk.decompress_chunk(pos)
        bytes_obj = buffer.tobytes()
        expected_chunks[pos] = bytes_obj
        assert chunk_ == bytes_obj

        dest = np.empty(buffer.shape, buffer.dtype)
        schunk.decompress_chunk(pos, dest)
        assert np.array_equal(buffer, dest)

    for i in range(nchunks):
        assert schunk.decompress_chunk(i) == expected_chunks[i]
    blosc2.remove_urlpath(urlpath)


@pytest.mark.parametrize(
    ("storage_case", "counts", "gil"),
    orthogonal_cases(
        [(True, None), (False, None), (True, "b2frame"), (False, "b2frame")],
        [(7, 3), (1, 1), (0, 0)],
        [True, False],
    ),
)
@pytest.mark.parametrize(("create_chunk", "copy"), [(True, True), (False, True), (False, False)])
def test_update(storage_case, counts, copy, create_chunk, gil):
    contiguous, urlpath = storage_case
    nchunks, nupdates = counts
    blosc2.set_releasegil(gil)
    kwargs = {
        "contiguous": contiguous,
        "urlpath": urlpath,
        "cparams": {"nthreads": 2},
        "dparams": {"nthreads": 2},
    }

    blosc2.remove_urlpath(urlpath)
    nbytes = 23401

    schunk = blosc2.SChunk(chunksize=nbytes * 2, **kwargs)
    expected_chunks = []
    for i in range(nchunks):
        bytes_obj = b"i " * nbytes
        nchunks_ = schunk.append_data(bytes_obj)
        assert nchunks_ == (i + 1)
        expected_chunks.append(bytes_obj)

    for i in range(nupdates):
        pos = (0, nchunks - 1, nchunks // 2)[i % 3]
        bytes_obj = bytes([65 + i]) * (nbytes * 2)
        if create_chunk:
            chunk = blosc2.compress2(bytes_obj, typesize=1)
            schunk.update_chunk(pos, chunk)
        else:
            schunk.update_data(pos, bytes_obj, copy)
        res = schunk.decompress_chunk(pos)
        assert res == bytes_obj
        expected_chunks[pos] = bytes_obj

    for i in range(nchunks):
        assert schunk.decompress_chunk(i) == expected_chunks[i]
    blosc2.remove_urlpath(urlpath)


@pytest.mark.parametrize(
    ("contiguous", "urlpath"),
    [
        (False, None),
        (True, None),
        (True, "test_variable_append_chunk.b2frame"),
        (False, "test_variable_append_chunk_s.b2frame"),
    ],
)
def test_append_chunk_variable_sizes(contiguous, urlpath):
    blosc2.remove_urlpath(urlpath)

    schunk = blosc2.SChunk(chunksize=-1, contiguous=contiguous, urlpath=urlpath, cparams={"typesize": 1})
    payloads = [b"a" * 13, b"b" * 29, b"c" * 41]

    for i, payload in enumerate(payloads, start=1):
        chunk = blosc2.compress2(payload, typesize=1)
        assert schunk.append_chunk(chunk) == i
        assert schunk.decompress_chunk(i - 1) == payload

    assert schunk.chunksize == 0

    replacement = b"z" * 17
    schunk.update_chunk(1, blosc2.compress2(replacement, typesize=1))
    expected = [payloads[0], replacement, payloads[2]]
    assert [schunk.decompress_chunk(i) for i in range(schunk.nchunks)] == expected

    if urlpath is not None:
        reopened = blosc2.open(urlpath, mode="r")
        assert reopened.chunksize == 0
        assert [reopened.decompress_chunk(i) for i in range(reopened.nchunks)] == expected

    blosc2.remove_urlpath(urlpath)


def _rss_bytes():
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="needs /proc and unlink of open files")
@pytest.mark.parametrize("method", ["update_data", "insert_data"])
def test_failed_data_write_without_copy_frees_chunk(tmp_path, method):
    # A failed update_chunk/insert_chunk without a copy leaves the compressed
    # chunk with the caller, so update_data/insert_data must free it
    urlpath = str(tmp_path / "failed_write.b2frame")
    nbytes = 2 * 2**20
    rng = np.random.default_rng(0)
    data = rng.integers(0, 256, nbytes, dtype=np.uint8)  # incompressible, so chunks stay large
    schunk = blosc2.SChunk(
        chunksize=nbytes,
        data=data,
        urlpath=urlpath,
        contiguous=True,
        mode="w",
        cparams={"typesize": 1, "clevel": 1},
    )
    os.remove(urlpath)  # the frame keeps working through its cached reader, but cannot write
    write = getattr(schunk, method)
    with pytest.raises(RuntimeError):
        write(0, data, copy=False)  # warm-up
    rss0 = _rss_bytes()
    for _ in range(40):
        with pytest.raises(RuntimeError):
            write(0, data, copy=False)
    # 40 leaked chunks would be 80 MB
    assert _rss_bytes() - rss0 < 30 * 2**20
