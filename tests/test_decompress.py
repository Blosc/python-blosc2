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
    "object, codec",
    [
        (np.random.randint(0, 10, 10), blosc2.Codec.LZ4),
        (np.arange(10), blosc2.Codec.BLOSCLZ),
        (np.random.randint(0, 1000 + 1, 1000), blosc2.Codec.LZ4HC),
        (np.arange(45, dtype=np.float64), blosc2.Codec.ZLIB),
        (np.arange(50, dtype=np.int64), blosc2.Codec.ZSTD),
    ],
)
def test_decompress_numpy(object, codec, gil):
    blosc2.set_releasegil(gil)
    c = blosc2.compress(object, codec=codec)

    dest = bytearray(object)
    blosc2.decompress(c, dst=dest)
    assert dest == object.tobytes()

    dest2 = np.empty(object.shape, object.dtype)
    blosc2.decompress(c, dst=dest2)
    assert np.array_equal(dest2, object)

    dest3 = blosc2.decompress(c)
    assert dest3 == object.tobytes()

    dest4 = blosc2.decompress(c, as_bytearray=True)
    assert dest4 == object.tobytes()

    dest5 = np.empty(object.shape, object.dtype)
    blosc2.decompress(c, dst=memoryview(dest5))
    assert np.array_equal(dest5, object)


@pytest.mark.parametrize(
    "object, codec",
    [
        (bytearray([0, 12, 24, 33]), blosc2.Codec.LZ4),
        (bytearray([2, 45, 6, 12, 78, 43, 23, 234]), blosc2.Codec.BLOSCLZ),
        (b"A string", blosc2.Codec.LZ4HC),
        (bytearray("Another string" * 100, encoding="utf-8"), blosc2.Codec.ZSTD),
    ],
)
def test_decompress(object, codec):
    c = blosc2.compress(object, codec=codec)

    dest = bytearray(object)
    blosc2.decompress(c, dst=dest)
    assert dest == object

    dest3 = blosc2.decompress(c)
    assert dest3 == object

    dest4 = blosc2.decompress(c, as_bytearray=True)
    assert dest4 == object

    dest5 = bytearray(object)
    blosc2.decompress(np.array([c]), dst=dest5)
    assert dest5 == object


@pytest.mark.parametrize("object, codec", [(np.arange(0), blosc2.Codec.LZ4), (b"", blosc2.Codec.ZLIB)])
def test_raise_error(object, codec):
    c = blosc2.compress(object, codec=codec)

    dest = bytearray(object)
    with pytest.raises(ValueError):
        blosc2.decompress(c, dst=dest)

    dest3 = blosc2.decompress(c)
    if isinstance(object, bytes):
        assert dest3 == object
    else:
        assert dest3 == object.tobytes()

    dest4 = blosc2.decompress(c, as_bytearray=True)
    if isinstance(object, bytes):
        assert dest4 == object
    else:
        assert dest4 == object.tobytes()

    dest5 = bytearray(object)
    with pytest.raises(ValueError):
        blosc2.decompress(np.array([c]), dst=dest5)

    with pytest.raises(ValueError):
        blosc2.decompress(b"")
