#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################
import pathlib

import numexpr as ne
import numpy as np
import pytest

import blosc2

NITEMS_SMALL = 1_000
ROOT = "b2tests"
DIR = "expr/"


def get_arrays(shape, chunks_blocks):
    dtype = np.float64
    urlpath = f"ds-0-10-linspace-{dtype.__name__}-{chunks_blocks}-a1-{shape}d.b2nd"
    path = pathlib.Path(f"{ROOT}/{DIR + urlpath}").as_posix()
    a1 = blosc2.C2Array(path)
    return a1


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
    "slices",
    [
        (slice(0, 23), slice(None))
    ],
)
def test_simple(chunks_blocks, c2sub_context, slices):
    shape = (60, 60)
    a = get_arrays(shape, chunks_blocks)
    b = blosc2.SChunkProxy(a)

    np.testing.assert_allclose(b[slices], a[slices])

    cache_slice = b.eval(slices)
    np.testing.assert_allclose(cache_slice[slices], a[slices])

    cache = b.eval()
    np.testing.assert_allclose(cache[...], a[...])

# def test_complex(c2sub_context):
#     shape = (NITEMS_SMALL,)
#     chunks_blocks = "default"
#     a1, a2, a3, a4, na1, na2, na3, na4 = get_arrays(shape, chunks_blocks)
#     expr = blosc2.tan(a1) * blosc2.sin(a2) + (blosc2.sqrt(a4) * 2)
#     expr += 2
#     nres = ne.evaluate("tan(na1) * sin(na2) + (sqrt(na4) * 2) + 2")
#     # eval
#     res = expr.eval()
#     np.testing.assert_allclose(res[:], nres)
#     # __getitem__
#     res = expr[:]
#     np.testing.assert_allclose(res, nres)
#     # slice
#     sl = slice(10)
#     res = expr[sl]
#     np.testing.assert_allclose(res, nres[sl])

#
# @pytest.fixture(
#     params=[
#         ((2, 5), (5,)),
#         pytest.param(((2, 1), (5,)), marks=pytest.mark.heavy),
#         pytest.param(((2, 5, 3), (5, 1)), marks=pytest.mark.heavy),
#         ((2, 1, 3), (5, 3)),
#         pytest.param(((2, 5, 3, 2), (5, 3, 1)), marks=pytest.mark.heavy),
#         ((2, 5, 3, 2), (5, 1, 2)),
#         pytest.param(((2, 5, 3, 2, 2), (5, 3, 2, 2)), marks=pytest.mark.heavy),
#     ]
# )
# def broadcast_shape(request):
#     return request.param
#
#
# @pytest.fixture
# def broadcast_fixture(broadcast_shape, c2sub_context):
#     shape1, shape2 = broadcast_shape
#     dtype = np.float64
#     na1 = np.linspace(0, 1, np.prod(shape1), dtype=dtype).reshape(shape1)
#     na2 = np.linspace(1, 2, np.prod(shape2), dtype=dtype).reshape(shape2)
#     urlpath = f"ds-0-1-linspace-{dtype.__name__}-b1-{shape1}d.b2nd"
#     path = pathlib.Path(f"{ROOT}/{DIR + urlpath}").as_posix()
#     b1 = blosc2.C2Array(path)
#     urlpath = f"ds-1-2-linspace-{dtype.__name__}-b2-{shape2}d.b2nd"
#     path = pathlib.Path(f"{ROOT}/{DIR + urlpath}").as_posix()
#     b2 = blosc2.C2Array(path)
#
#     return b1, b2, na1, na2
#
#
# def test_broadcasting(broadcast_fixture):
#     a1, a2, na1, na2 = broadcast_fixture
#     expr1 = a1 + a2
#     assert expr1.shape == a1.shape
#     expr2 = a1 * a2 + 1
#     assert expr2.shape == a1.shape
#     expr = expr1 - expr2
#     assert expr.shape == a1.shape
#     nres = ne.evaluate("na1 + na2 - (na1 * na2 + 1)")
#     res = expr.eval()
#     np.testing.assert_allclose(res[:], nres)
#     res = expr[:]
#     np.testing.assert_allclose(res, nres)
