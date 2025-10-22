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

ROOT = "@public"
DIR = "expr/"

pytestmark = pytest.mark.network


def udf1p(inputs_tuple, output, offset):
    x = inputs_tuple[0]
    output[:] = x + 1


@pytest.mark.parametrize("chunked_eval", [True, False])
@pytest.mark.parametrize(
    ("chunks", "blocks"),
    [
        pytest.param((30, 30), (30, 30), marks=pytest.mark.heavy),
        (
            (50, 50),
            (30, 50),
        ),
    ],
)
def test_1p(chunks, blocks, chunked_eval, cat2_context):
    dtype = np.float64
    shape = (60, 60)
    urlpath = f"ds-0-10-linspace-{dtype.__name__}-(True, False)-a1-{shape}d.b2nd"
    path = pathlib.Path(f"{ROOT}/{DIR + urlpath}").as_posix()
    a = blosc2.C2Array(path)
    npa = a[:]
    npc = npa + 1

    expr = blosc2.lazyudf(
        udf1p, (a,), npa.dtype, chunked_eval=chunked_eval, chunks=chunks, blocks=blocks, dparams={}
    )
    res = expr.compute()
    assert res.chunks == chunks
    assert res.blocks == blocks
    assert res.dtype == npa.dtype

    tol = 1e-5 if res.dtype is np.float32 else 1e-14
    np.testing.assert_allclose(res[...], npc, rtol=tol, atol=tol)
    np.testing.assert_allclose(expr[...], npc, rtol=tol, atol=tol)


def udf2p(inputs_tuple, output, offset):
    x = inputs_tuple[0]
    y = inputs_tuple[1]
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            output[i, j] = x[i, j] ** 2 + y[i, j] ** 2 + 2 * x[i, j] * y[i, j] + 1


@pytest.mark.parametrize("chunked_eval", [True, False])
@pytest.mark.parametrize(
    ("chunks", "blocks", "slices", "urlpath", "contiguous"),
    [
        pytest.param((53, 20), (10, 13), (slice(3, 8), slice(9, 12)), None, False),
    ],
)
def test_getitem(chunks, blocks, slices, urlpath, contiguous, chunked_eval, cat2_context):
    dtype = np.float64
    shape = (60, 60)
    blosc2.remove_urlpath(urlpath)

    urlpath_a = f"ds-0-10-linspace-{dtype.__name__}-(True, False)-a1-{shape}d.b2nd"
    path = pathlib.Path(f"{ROOT}/{DIR + urlpath_a}").as_posix()
    a = blosc2.C2Array(path)

    urlpath_b = f"ds-0-10-linspace-{dtype.__name__}-(False, False)-a3-{shape}d.b2nd"
    path = pathlib.Path(f"{ROOT}/{DIR + urlpath_b}").as_posix()
    b = blosc2.C2Array(path)
    npa = a[:]
    npb = b[:]
    npc = npa**2 + npb**2 + 2 * npa * npb + 1
    dparams = {"nthreads": 4}

    expr = blosc2.lazyudf(
        udf2p,
        (npa, b),
        npa.dtype,
        chunked_eval=chunked_eval,
        chunks=chunks,
        blocks=blocks,
        storage=blosc2.Storage(urlpath=urlpath, contiguous=contiguous),
        dparams=dparams,
    )
    lazy_eval = expr[slices]
    np.testing.assert_allclose(lazy_eval, npc[slices])

    res = expr.compute(item=slices)
    np.testing.assert_allclose(res[...], npc[slices])
    assert res.schunk.urlpath is None
    assert res.schunk.contiguous == contiguous
    # Check dparams after a getitem and an eval
    assert res.schunk.dparams.nthreads == dparams["nthreads"]

    blosc2.remove_urlpath(urlpath)
