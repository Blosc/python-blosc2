#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

import pathlib
import random

import numpy as np
import pytest
import requests

import blosc2

pytestmark = pytest.mark.network

NITEMS_SMALL = 1_000
ROOT = "@public"
DIR = "expr/"


def test_open_c2array(cat2_context):
    dtype = np.float64
    shape = (NITEMS_SMALL,)
    chunks_blocks = "default"
    path = f"ds-0-10-linspace-{dtype.__name__}-{chunks_blocks}-a1-{shape}d.b2nd"
    path = pathlib.Path(f"{ROOT}/{DIR + path}").as_posix()
    a1 = blosc2.C2Array(path)
    urlpath = blosc2.URLPath(path)
    a_open = blosc2.open(urlpath, mode="r", offset=0)
    np.testing.assert_allclose(a1[:], a_open[:])

    a_open = blosc2.open(urlpath, mode="r")
    np.testing.assert_allclose(a1[:], a_open[:])

    ## Test slicing
    np.testing.assert_allclose(a1[:10], a_open[:10])
    np.testing.assert_allclose(a1.slice(slice(1, 10, 1))[:], a_open.slice(slice(1, 10, 1))[:])

    ## Test metadata
    assert a1.cratio == a_open.cratio

    with pytest.raises(NotImplementedError):
        _ = blosc2.open(urlpath)

    with pytest.raises(NotImplementedError):
        _ = blosc2.open(urlpath, mode="r", offset=0, cparams={})


def test_open_c2array_args(cat2_context):  # instance args prevail
    dtype = np.float64
    shape = (NITEMS_SMALL,)
    chunks_blocks = "default"
    path = f"ds-0-10-linspace-{dtype.__name__}-{chunks_blocks}-a1-{shape}d.b2nd"
    path = pathlib.Path(f"{ROOT}/{DIR + path}").as_posix()

    with blosc2.c2context(urlbase="https://wrong.example.com/", auth_token="wrong-token"):
        urlbase = cat2_context["urlbase"]
        auth_token = blosc2.c2array.login(**cat2_context) if cat2_context["username"] else None
        a1 = blosc2.C2Array(path, urlbase=urlbase, auth_token=auth_token)
        urlpath = blosc2.URLPath(path, urlbase=urlbase, auth_token=auth_token)
        a_open = blosc2.open(urlpath, mode="r", offset=0)
        np.testing.assert_allclose(a1[:], a_open[:])


@pytest.fixture(scope="session")
def c2sub_user():
    def rand32():
        return random.randint(0, 0x7FFFFFFF)

    urlbase = "https://cat2.cloud/testing/"
    username = f"user+{rand32():x}@example.com"
    password = hex(rand32())

    for _ in range(3):
        resp = requests.post(
            f"{urlbase}auth/register", json={"email": username, "password": password}, timeout=15
        )
        if resp.status_code != 400:
            break
        # Retry on possible username collision.
    resp.raise_for_status()

    return {"urlbase": urlbase, "username": username, "password": password}


def test_open_c2array_auth(c2sub_user):
    dtype = np.float64
    shape = (NITEMS_SMALL,)
    chunks_blocks = "default"
    path = f"ds-0-10-linspace-{dtype.__name__}-{chunks_blocks}-a1-{shape}d.b2nd"
    path = pathlib.Path(f"{ROOT}/{DIR + path}").as_posix()

    with blosc2.c2context(**c2sub_user):
        a1 = blosc2.C2Array(path)
        assert a1.dtype == dtype
        assert a1.shape == shape
