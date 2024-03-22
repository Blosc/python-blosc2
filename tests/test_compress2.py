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


@pytest.mark.parametrize("gil", [True, False])
@pytest.mark.parametrize(
    "obj, cparams, dparams",
    [
        (np.random.randint(0, 10, 10), {"codec": blosc2.Codec.LZ4, "clevel": 6}, {}),
        (
            np.arange(10, dtype="float32"),
            # Select an absolute precision of 10 bits in mantissa
            {
                "filters": [blosc2.Filter.TRUNC_PREC, blosc2.Filter.BITSHUFFLE],
                "filters_meta": [10, 0],
                "typesize": 4,
            },
            {"nthreads": 4},
        ),
        (
            np.arange(10, dtype="float32"),
            # Do a reduction of precision of 10 bits in mantissa
            {
                "filters": [blosc2.Filter.TRUNC_PREC, blosc2.Filter.BITSHUFFLE],
                "filters_meta": [-10, 0],
                "typesize": 4,
            },
            {"nthreads": 4},
        ),
        (
            np.random.randint(0, 1000 + 1, 1000),
            {"splitmode": blosc2.SplitMode.ALWAYS_SPLIT, "nthreads": 5, "typesize": 4},
            {},
        ),
        (np.arange(45, dtype=np.float64), {"codec": blosc2.Codec.LZ4HC, "typesize": 4}, {}),
        (np.arange(50, dtype=np.int64), {"typesize": 4}, blosc2.dparams_dflts),
    ],
)
def test_compress2_numpy(obj, cparams, dparams, gil):
    blosc2.set_releasegil(gil)
    bytes_obj = obj.tobytes()
    c = blosc2.compress2(obj, **cparams)

    dest = bytearray(obj)
    blosc2.decompress2(c, dst=dest, **dparams)
    assert dest == bytes_obj

    dest2 = np.empty(obj.shape, obj.dtype)
    blosc2.decompress2(c, dst=dest2, **dparams)
    assert np.array_equal(dest2, obj)

    dest3 = blosc2.decompress2(c, **dparams)
    assert dest3 == bytes_obj

    dest4 = np.empty(obj.shape, obj.dtype)
    blosc2.decompress2(c, dst=memoryview(dest4), **dparams)
    assert np.array_equal(dest4, obj)


@pytest.mark.parametrize("gil", [True, False])
@pytest.mark.parametrize(
    "obj, cparams, dparams",
    [
        (
            np.random.randint(0, 10, 10, dtype=np.int64),
            {"codec": blosc2.Codec.LZ4, "clevel": 6, "filters_meta": [-50]},
            {},
        ),
        (
            np.arange(10, dtype="int32"),
            {"filters_meta": [-20]},
            {"nthreads": 4},
        ),
        (np.arange(45, dtype=np.int16), {"codec": blosc2.Codec.LZ4HC, "filters_meta": [-10]}, {}),
        (np.arange(50, dtype=np.int8), {"filters_meta": [-5]}, blosc2.dparams_dflts),
    ],
)
def test_compress2_int_trunc(obj, cparams, dparams, gil):
    blosc2.set_releasegil(gil)
    cparams["filters"] = [blosc2.Filter.INT_TRUNC]
    cparams["typesize"] = obj.dtype.itemsize
    c = blosc2.compress2(obj, **cparams)

    dest = np.empty(obj.shape, obj.dtype)
    blosc2.decompress2(c, dst=dest, **dparams)

    for i in range(obj.shape[0]):
        assert (obj[i] - dest[i]) <= (2 ** ((-1) * cparams["filters_meta"][0]))


@pytest.mark.parametrize("gil", [True, False])
@pytest.mark.parametrize(
    "nbytes, cparams, dparams",
    [
        (7, {"codec": blosc2.Codec.LZ4, "clevel": 6, "typesize": 1}, {}),
        (641091, {"typesize": 1}, {"nthreads": 4}),
        (136, {"typesize": 1}, {}),
        (1231, {"typesize": 4}, blosc2.dparams_dflts),
    ],
)
def test_compress2(nbytes, cparams, dparams, gil):
    blosc2.set_releasegil(gil)
    bytes_obj = b" " * nbytes
    c = blosc2.compress2(bytes_obj, **cparams)

    dest = bytearray(bytes_obj)
    blosc2.decompress2(c, dst=dest, **dparams)
    assert dest == bytes_obj

    dest2 = blosc2.decompress2(c, **dparams)
    assert dest2 == bytes_obj

    dest3 = bytearray(bytes_obj)
    blosc2.decompress2(np.array([c]), dst=dest3, **dparams)
    assert dest3 == bytes_obj


@pytest.mark.parametrize("gil", [True, False])
@pytest.mark.parametrize(
    "object, cparams, dparams",
    [(np.arange(0), {"codec": blosc2.Codec.LZ4, "clevel": 6}, {}), (b"", {}, {"nthreads": 3})],
)
def test_raise_error(object, cparams, dparams, gil):
    blosc2.set_releasegil(gil)
    c = blosc2.compress2(object, **cparams, **dparams)

    dest = bytearray(object)
    with pytest.raises(ValueError):
        blosc2.decompress2(c, dst=dest)

    dest3 = blosc2.decompress2(c)
    if isinstance(object, bytes):
        assert dest3 == object
    else:
        assert dest3 == object.tobytes()

    dest5 = bytearray(object)
    with pytest.raises(ValueError):
        blosc2.decompress2(np.array([c]), dst=dest5)

    with pytest.raises(ValueError):
        blosc2.decompress2(b"")
