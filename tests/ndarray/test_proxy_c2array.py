#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################
import pathlib

import numpy as np
import pytest

import blosc2

pytestmark = pytest.mark.network

NITEMS_SMALL = 1_000
ROOT = "@public"
DIR = "expr/"


def get_array(shape, chunks_blocks):
    dtype = np.float64
    urlpath = f"ds-0-10-linspace-{dtype.__name__}-{chunks_blocks}-a1-{shape}d.b2nd"
    path = pathlib.Path(f"{ROOT}/{DIR + urlpath}").as_posix()
    return blosc2.C2Array(path)


@pytest.mark.parametrize(
    "chunks_blocks",
    [
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    ],
)
@pytest.mark.parametrize(
    ("urlpath", "slices"),
    [
        (None, (slice(0, 23), slice(None))),
        ("proxy", (slice(None), slice(None))),
        (None, (slice(0, 5), slice(0, 60))),
        ("proxy", (slice(37, 53), slice(19, 233))),
    ],
)
def test_simple(chunks_blocks, cat2_context, urlpath, slices):
    shape = (60, 60)
    a = get_array(shape, chunks_blocks)
    b = blosc2.Proxy(a, urlpath=urlpath, mode="w")

    np.testing.assert_allclose(b[slices], a[slices])

    cache_slice = b.fetch(slices)
    assert cache_slice.schunk.urlpath == urlpath
    np.testing.assert_allclose(cache_slice[slices], a[slices])

    cache = b.fetch()
    assert cache.schunk.urlpath == urlpath
    np.testing.assert_allclose(cache[...], a[...])

    blosc2.remove_urlpath(urlpath)


def test_small(cat2_context):
    shape = (NITEMS_SMALL,)
    chunks_blocks = "default"
    a = get_array(shape, chunks_blocks)
    b = blosc2.Proxy(a)

    np.testing.assert_allclose(b[0:100], a[0:100])

    cache_slice = b.fetch(slice(0, 100))
    np.testing.assert_allclose(cache_slice[0:100], a[0:100])

    cache = b.fetch()
    np.testing.assert_allclose(cache[...], a[...])


def test_open(cat2_context):
    urlpath = "proxy.b2nd"
    shape = (NITEMS_SMALL,)
    chunks_blocks = "default"
    a = get_array(shape, chunks_blocks)
    b = blosc2.Proxy(a, urlpath=urlpath, mode="w")
    del a
    del b

    b = blosc2.open(urlpath)
    a = get_array(shape, chunks_blocks)

    np.testing.assert_allclose(b[...], a[...])

    blosc2.remove_urlpath(urlpath)
