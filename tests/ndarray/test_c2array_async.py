#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

import asyncio

import httpx
import numpy as np
import pytest

import blosc2
from blosc2 import c2array as c2array_mod


class _FakeResponse:
    def __init__(self, json_data):
        self._json = json_data

    def raise_for_status(self):
        pass

    def json(self):
        return self._json


class _FakeHttpx:
    HTTPStatusError = Exception

    def __init__(self, meta):
        self._meta = meta

    def get(self, url, params=None, headers=None, timeout=None):
        return _FakeResponse(self._meta)


@pytest.fixture
def fake_c2array(monkeypatch):
    """A C2Array whose sync 'info' HTTP call is faked, so no network is used."""
    array = blosc2.asarray(np.arange(20, dtype=np.int64).reshape(4, 5), chunks=(2, 5), blocks=(1, 5))
    meta = {
        "shape": list(array.shape),
        "chunks": list(array.chunks),
        "blocks": list(array.blocks),
        "dtype": str(array.dtype),
        "schunk": {"cparams": {"typesize": array.dtype.itemsize}},
    }
    monkeypatch.setattr(c2array_mod, "_httpx", lambda: _FakeHttpx(meta))
    c2 = blosc2.C2Array("@public/fake.b2nd", urlbase="http://fake-server/")
    c2._chunks_source = array  # stash the real array to serve chunk bytes from
    return c2


def _mock_transport(chunks_source, calls):
    def handler(request: httpx.Request) -> httpx.Response:
        nchunk = int(dict(request.url.params)["nchunk"])
        calls.append(nchunk)
        return httpx.Response(200, content=chunks_source.get_chunk(nchunk))

    return httpx.MockTransport(handler)


def test_aget_chunk(fake_c2array):
    calls = []
    fake_c2array._aclient = httpx.AsyncClient(transport=_mock_transport(fake_c2array._chunks_source, calls))

    async def run():
        chunk = await fake_c2array.aget_chunk(0)
        await fake_c2array.aclose()
        return chunk

    chunk = asyncio.run(run())
    assert chunk == fake_c2array._chunks_source.get_chunk(0)
    assert calls == [0]
    assert fake_c2array._aclient is None  # aclose() clears the cached client


def test_afetch_over_c2array_higher_concurrency(fake_c2array):
    calls = []
    fake_c2array._aclient = httpx.AsyncClient(transport=_mock_transport(fake_c2array._chunks_source, calls))
    proxy = blosc2.Proxy(fake_c2array)

    result = asyncio.run(proxy.afetch())
    asyncio.run(fake_c2array.aclose())

    np.testing.assert_array_equal(result[:], fake_c2array._chunks_source[:])
    assert sorted(calls) == list(range(fake_c2array._chunks_source.schunk.nchunks))
