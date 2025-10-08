#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################
import math

import numpy as np
import pytest

import blosc2


def udf1p(inputs_tuple, output, offset):
    x = inputs_tuple[0]
    output[:] = x + 1


@pytest.mark.parametrize("chunked_eval", [True, False])
@pytest.mark.parametrize(
    ("shape", "chunks", "blocks"),
    [
        # Test different shapes with and without padding
        (
            (10, 10),
            (10, 10),
            (10, 10),
        ),
        (
            (20, 20),
            (10, 10),
            (10, 10),
        ),
        (
            (20, 20),
            (10, 10),
            (5, 5),
        ),
        (
            (13, 13),
            (10, 10),
            (10, 10),
        ),
        (
            (13, 13),
            (10, 10),
            (5, 5),
        ),
        (
            (10, 10),
            (10, 10),
            (4, 4),
        ),
        (
            (13, 13),
            (10, 10),
            (4, 4),
        ),
    ],
)
def test_1p(shape, chunks, blocks, chunked_eval):
    npa = np.linspace(0, 1, np.prod(shape)).reshape(shape)
    npc = npa + 1

    expr = blosc2.lazyudf(
        udf1p, (npa,), npa.dtype, chunked_eval=chunked_eval, chunks=chunks, blocks=blocks, dparams={}
    )
    res = expr.compute()
    assert res.shape == shape
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
    ("shape", "chunks", "blocks"),
    [
        (
            (20, 20),
            (10, 10),
            (5, 5),
        ),
        (
            (13, 13, 10),
            (10, 10, 5),
            (5, 5, 3),
        ),
        (
            (13, 13),
            (10, 10),
            (5, 5),
        ),
    ],
)
def test_2p(shape, chunks, blocks, chunked_eval):
    npa = np.arange(0, np.prod(shape)).reshape(shape)
    npb = np.arange(1, np.prod(shape) + 1).reshape(shape)
    npc = npa**2 + npb**2 + 2 * npa * npb + 1

    b = blosc2.asarray(npb)
    expr = blosc2.lazyudf(
        udf2p, (npa, b), npa.dtype, chunked_eval=chunked_eval, chunks=chunks, blocks=blocks
    )
    res = expr.compute()

    np.testing.assert_allclose(res[...], npc)


def udf0p(inputs_tuple, output, offset):
    output[:] = 1


@pytest.mark.parametrize("chunked_eval", [True, False])
@pytest.mark.parametrize(
    ("shape", "chunks", "blocks"),
    [
        (
            (20, 20),
            (10, 10),
            (5, 5),
        ),
        (
            (13, 13, 10),
            (10, 10, 5),
            (5, 5, 3),
        ),
        (
            (13, 13),
            (10, 10),
            (5, 5),
        ),
    ],
)
def test_0p(shape, chunks, blocks, chunked_eval):
    npa = np.ones(shape)

    expr = blosc2.lazyudf(
        udf0p, (), npa.dtype, shape=shape, chunked_eval=chunked_eval, chunks=chunks, blocks=blocks
    )
    res = expr.compute()

    np.testing.assert_allclose(res[...], npa)


def udf_1dim(inputs_tuple, output, offset):
    x = inputs_tuple[0]
    y = inputs_tuple[1]
    z = inputs_tuple[2]
    output[:] = x + y + z


@pytest.mark.parametrize("chunked_eval", [True, False])
@pytest.mark.parametrize(
    ("shape", "chunks", "blocks"),
    [
        (
            (20,),
            (10,),
            (5,),
        ),
        (
            (23,),
            (10,),
            (3,),
        ),
    ],
)
def test_1dim(shape, chunks, blocks, chunked_eval):
    npa = np.arange(start=0, stop=np.prod(shape)).reshape(shape)
    npb = np.linspace(1, 2, np.prod(shape)).reshape(shape)
    py_scalar = np.e
    npc = npa + npb + py_scalar

    b = blosc2.asarray(npb)
    expr = blosc2.lazyudf(
        udf_1dim,
        (npa, b, py_scalar),
        np.float64,
        chunked_eval=chunked_eval,
        chunks=chunks,
        blocks=blocks,
    )
    res = expr.compute()

    tol = 1e-5 if res.dtype is np.float32 else 1e-14
    np.testing.assert_allclose(res[...], npc, rtol=tol, atol=tol)


@pytest.mark.parametrize("chunked_eval", [True, False])
def test_params(chunked_eval):
    shape = (23,)
    npa = np.arange(start=0, stop=np.prod(shape)).reshape(shape)
    array = blosc2.asarray(npa)

    # Assert that shape is computed correctly
    npc = npa + 1
    cparams = {"nthreads": 4}
    urlpath = "lazyarray.b2nd"
    urlpath2 = "eval.b2nd"
    blosc2.remove_urlpath(urlpath)
    blosc2.remove_urlpath(urlpath2)

    expr = blosc2.lazyudf(
        udf1p, (array,), np.float64, chunked_eval=chunked_eval, urlpath=urlpath, cparams=cparams
    )
    with pytest.raises(ValueError):
        _ = expr.compute(urlpath=urlpath)

    res = expr.compute(urlpath=urlpath2, chunks=(10,))
    np.testing.assert_allclose(res[...], npc)
    assert res.shape == npa.shape
    assert res.schunk.cparams.nthreads == cparams["nthreads"]
    assert res.schunk.urlpath == urlpath2
    assert res.chunks == (10,)

    res = expr.compute()
    np.testing.assert_allclose(res[...], npc)
    assert res.schunk.urlpath is None

    blosc2.remove_urlpath(urlpath)
    blosc2.remove_urlpath(urlpath2)

    # Pass list
    lnumbers = [1, 2, 3, 4, 5]
    expr = blosc2.lazyudf(udf1p, (lnumbers,), np.float64)
    res = expr.compute()
    npc = np.array(lnumbers) + 1
    np.testing.assert_allclose(res[...], npc)


@pytest.mark.parametrize("chunked_eval", [True, False])
@pytest.mark.parametrize(
    ("shape", "chunks", "blocks", "slices", "urlpath", "contiguous"),
    [
        ((40, 20), (30, 10), (5, 5), (slice(0, 5), slice(5, 20)), "eval.b2nd", False),
        ((13, 13, 10), (10, 10, 5), (5, 5, 3), (slice(0, 12), slice(3, 13), ...), "eval.b2nd", True),
        ((13, 13), (10, 10), (5, 5), (slice(3, 8), slice(9, 12)), None, False),
    ],
)
def test_getitem(shape, chunks, blocks, slices, urlpath, contiguous, chunked_eval):
    blosc2.remove_urlpath(urlpath)
    npa = np.arange(0, np.prod(shape)).reshape(shape)
    npb = np.arange(1, np.prod(shape) + 1).reshape(shape)
    npc = npa**2 + npb**2 + 2 * npa * npb + 1
    dparams = {"nthreads": 4}

    b = blosc2.asarray(npb)
    expr = blosc2.lazyudf(
        udf2p,
        (npa, b),
        npa.dtype,
        chunked_eval=chunked_eval,
        chunks=chunks,
        blocks=blocks,
        urlpath=urlpath,
        contiguous=contiguous,
        dparams=dparams,
    )
    lazy_eval = expr[slices]
    np.testing.assert_allclose(lazy_eval, npc[slices])

    res = expr.compute()
    np.testing.assert_allclose(res[...], npc)
    assert res.schunk.urlpath is None
    assert res.schunk.contiguous == contiguous
    # Check dparams after a getitem and an eval
    assert res.schunk.dparams.nthreads == dparams["nthreads"]

    lazy_eval = expr[slices]
    np.testing.assert_allclose(lazy_eval, npc[slices])

    blosc2.remove_urlpath(urlpath)


@pytest.mark.parametrize("chunked_eval", [True, False])
@pytest.mark.parametrize(
    ("shape", "chunks", "blocks", "slices", "urlpath", "contiguous"),
    [
        ((40, 20), (30, 10), (5, 5), (slice(0, 5), slice(5, 20)), "slice_eval.b2nd", False),
        ((13, 13, 10), (10, 10, 5), (5, 5, 3), (slice(0, 12), slice(3, 13), ...), "slice_eval.b2nd", True),
        ((13, 13), (10, 10), (5, 5), (slice(3, 8), slice(9, 12)), None, False),
    ],
)
def test_eval_slice(shape, chunks, blocks, slices, urlpath, contiguous, chunked_eval):
    blosc2.remove_urlpath(urlpath)
    npa = np.arange(0, np.prod(shape)).reshape(shape)
    npb = np.arange(1, np.prod(shape) + 1).reshape(shape)
    npc = npa**2 + npb**2 + 2 * npa * npb + 1
    dparams = {"nthreads": 4}
    b = blosc2.asarray(npb)
    expr = blosc2.lazyudf(
        udf2p,
        (npa, b),
        npa.dtype,
        chunked_eval=chunked_eval,
        chunks=chunks,
        blocks=blocks,
        urlpath=urlpath,
        contiguous=contiguous,
        dparams=dparams,
    )
    res = expr.compute(item=slices, chunks=None, blocks=None)
    np.testing.assert_allclose(res[...], npc[slices])
    assert res.schunk.urlpath is None
    assert res.schunk.contiguous == contiguous
    assert res.schunk.dparams.nthreads == dparams["nthreads"]
    assert res.schunk.cparams.nthreads == blosc2.nthreads
    assert res.shape == npc[slices].shape

    cparams = {"nthreads": 6}
    urlpath2 = "slice_eval2.b2nd"
    blosc2.remove_urlpath(urlpath2)

    res = expr.compute(item=slices, chunks=None, blocks=None, cparams=cparams, urlpath=urlpath2)
    np.testing.assert_allclose(res[...], npc[slices])
    assert res.schunk.urlpath == urlpath2
    assert res.schunk.contiguous == contiguous
    assert res.schunk.dparams.nthreads == dparams["nthreads"]
    assert res.schunk.cparams.nthreads == cparams["nthreads"]
    assert res.shape == npc[slices].shape

    blosc2.remove_urlpath(urlpath)
    blosc2.remove_urlpath(urlpath2)


def udf_offset(inputs_tuple, output, offset):
    _ = inputs_tuple[0]
    output[:] = sum(offset)


@pytest.mark.parametrize("eval_mode", ["eval", "getitem"])
@pytest.mark.parametrize("chunked_eval", [True, False])
@pytest.mark.parametrize(
    ("shape", "chunks", "blocks", "slices"),
    [
        ((10,), (4,), (3,), ()),
        # ((10,), (4,), (3,), None),  # TODO: make this work (None is equivalent to newaxis)
        ((10,), (4,), (3,), (slice(None),)),
        ((10,), (4,), (3,), (slice(5),)),
        ((8, 8), (4, 4), (2, 2), (slice(None), slice(None))),
        ((9, 8), (4, 4), (2, 3), (slice(None), slice(None))),
        ((13, 13), (10, 10), (4, 3), (slice(None), slice(None))),
        # TODO: make this work
        # I think the issue is in how the offsets are calculated here, in the test.
        # Other tests involving offset with slices are working fine
        # (e.g. test_ndarray::test_arange)
        # ((8, 8), (4, 4), (2, 2), (slice(0, 5), slice(5, 8))),
        # ((9, 8), (4, 4), (2, 3), (slice(0, 5), slice(5, 8))),
        # ((40, 20), (30, 10), (5, 5), (slice(0, 5), slice(5, 20))),
        # ((13, 13), (10, 10), (4, 3), (slice(3, 8), slice(9, 12))),
        # ((13, 13, 10), (10, 10, 5), (5, 5, 3), (slice(0, 12), slice(3, 13), ...)),
    ],
)
def test_offset(shape, chunks, blocks, slices, chunked_eval, eval_mode):
    x = np.zeros(shape)
    bx = blosc2.asarray(x, chunks=chunks, blocks=blocks)

    # Compute the desired output
    out = np.zeros_like(x)
    # Calculate the number of chunks in each dimension
    if not chunked_eval:
        # When using prefilters/postfilters, the computation is split in blocks, not chunks
        chunks = blocks
    nchunks = tuple(math.ceil(x.shape[i] / blocks[i]) for i in range(len(x.shape)))

    # Iterate over the chunks for computing the output
    for index in np.ndindex(nchunks):
        # Calculate the offset for the current chunk
        offset = [index[i] * chunks[i] for i in range(len(index))]
        # Apply the offset to the chunk and store the result in the output array
        out_slice = tuple(slice(index[i] * chunks[i], (index[i] + 1) * chunks[i]) for i in range(len(index)))
        out[out_slice] = sum(offset)

    expr = blosc2.lazyudf(
        udf_offset,
        (bx,),
        bx.dtype,
        chunked_eval=chunked_eval,
        chunks=chunks,
        blocks=blocks,
    )
    if eval_mode == "eval":
        res = expr.compute(slices)
        res = res[:]
    else:
        res = expr[slices]
    np.testing.assert_allclose(res, out[slices])


@pytest.mark.parametrize(
    ("shape", "chunks", "blocks", "slices"),
    [
        ((40, 20), (30, 10), (5, 5), (slice(0, 5), slice(5, 20))),
        ((13, 13, 10), (10, 10, 5), (5, 5, 3), (slice(0, 12), slice(3, 13), ...)),
        ((13, 13), (10, 10), (5, 5), (slice(3, 8), slice(9, 12))),
    ],
)
def test_clip_logaddexp(shape, chunks, blocks, slices):
    npa = np.arange(0, np.prod(shape), dtype=np.float64).reshape(shape)
    npb = np.arange(1, np.prod(shape) + 1, dtype=np.int64).reshape(shape)
    b = blosc2.asarray(npb)
    a = blosc2.asarray(npa)

    npc = np.clip(npb, np.prod(shape) // 3, npb - 10)
    expr = blosc2.clip(b, np.prod(shape) // 3, npb - 10)
    res = expr.compute(item=slices)
    np.testing.assert_allclose(res[...], npc[slices])
    # clip is not a ufunc so will return np.ndarray
    expr = np.clip(b, np.prod(shape) // 3, npb - 10)
    assert isinstance(expr, np.ndarray)
    # test lazyexpr interface
    expr = blosc2.lazyexpr("clip(b, np.prod(shape) // 3, npb - 10)")
    res = expr.compute(item=slices)
    np.testing.assert_allclose(res[...], npc[slices])

    npc = np.logaddexp(npb, npa)
    expr = blosc2.logaddexp(b, a)
    res = expr.compute(item=slices)
    np.testing.assert_allclose(res[...], npc[slices])
    # test that ufunc has been overwritten successfully
    # (i.e. doesn't return np.ndarray)
    expr = np.logaddexp(b, a)
    assert isinstance(expr, blosc2.LazyArray)

    # test lazyexpr interface
    expr = blosc2.lazyexpr("logaddexp(a, b)")
    res = expr.compute(item=slices)
    np.testing.assert_allclose(res[...], npc[slices])

    # Test LazyUDF has inherited __add__ from Operand class
    expr = blosc2.logaddexp(b, a) + blosc2.clip(b, np.prod(shape) // 3, npb - 10)
    npc = np.logaddexp(npb, npa) + np.clip(npb, np.prod(shape) // 3, npb - 10)
    res = expr.compute(item=slices)
    np.testing.assert_allclose(res[...], npc[slices])

    # Test LazyUDF more
    expr = blosc2.evaluate("logaddexp(b, a) + clip(b, np.prod(shape) // 3, npb - 10)")
    np.testing.assert_allclose(expr, npc)
    expr = blosc2.evaluate("sin(logaddexp(b, a))")
    np.testing.assert_allclose(expr, np.sin(np.logaddexp(npb, npa)))
    expr = blosc2.evaluate("clip(logaddexp(b, a), 6, 12)")
    np.testing.assert_allclose(expr, np.clip(np.logaddexp(npb, npa), 6, 12))
