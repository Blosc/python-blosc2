########################################################################
#
#       Author:  The Blosc development team - blosc@blosc.org
#
########################################################################


import numpy
import pytest

import blosc2


@pytest.mark.parametrize(
    "object, cname",
    [
        (numpy.random.randint(0, 10, 10), "lz4"),
        (numpy.arange(10), "blosclz"),
        (numpy.random.randint(0, 1000 + 1, 1000), "lz4hc"),
        (numpy.arange(45, dtype=numpy.float64), "zlib"),
        (numpy.arange(50, dtype=numpy.int64), "zstd"),
    ],
)
def test_decompress_numpy(object, cname):
    c = blosc2.compress(object, cname=cname)

    dest = bytearray(object)
    blosc2.decompress(c, dst=dest)
    assert dest == object.tobytes()

    dest2 = numpy.empty(object.shape, object.dtype)
    blosc2.decompress(c, dst=dest2)
    assert numpy.array_equal(dest2, object)

    dest3 = blosc2.decompress(c)
    assert dest3 == object.tobytes()

    dest4 = blosc2.decompress(c, as_bytearray=True)
    assert dest4 == object.tobytes()

    dest5 = numpy.empty(object.shape, object.dtype)
    blosc2.decompress(c, dst=memoryview(dest5))
    assert numpy.array_equal(dest5, object)


@pytest.mark.parametrize(
    "object, cname",
    [
        (bytearray([0, 12, 24, 33]), "lz4"),
        (bytearray([2, 45, 6, 12, 78, 43, 23, 234]), "blosclz"),
        (b"A string", "lz4hc"),
        (bytearray("Another string" * 100, encoding="utf-8"), "zstd"),
    ],
)
def test_decompress(object, cname):
    c = blosc2.compress(object, cname=cname)

    dest = bytearray(object)
    blosc2.decompress(c, dst=dest)
    assert dest == object

    dest3 = blosc2.decompress(c)
    assert dest3 == object

    dest4 = blosc2.decompress(c, as_bytearray=True)
    assert dest4 == object

    dest5 = bytearray(object)
    blosc2.decompress(numpy.array([c]), dst=dest5)
    assert dest5 == object


@pytest.mark.parametrize("object, cname", [(numpy.arange(0), "lz4"), (b"", "zlib")])
def test_raise_error(object, cname):
    c = blosc2.compress(object, cname=cname)

    dest = bytearray(object)
    with pytest.raises(ValueError):
        blosc2.decompress(c, dst=dest)

    dest3 = blosc2.decompress(c)
    if type(object) is bytes:
        assert dest3 == object
    else:
        assert dest3 == object.tobytes()

    dest4 = blosc2.decompress(c, as_bytearray=True)
    if type(object) is bytes:
        assert dest4 == object
    else:
        assert dest4 == object.tobytes()

    dest5 = bytearray(object)
    with pytest.raises(ValueError):
        blosc2.decompress(numpy.array([c]), dst=dest5)

    with pytest.raises(ValueError):
        blosc2.decompress(b"")
