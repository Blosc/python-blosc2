########################################################################
#
#       Created: May 13, 2021
#       Author:  The Blosc development team - blosc@blosc.org
#
########################################################################


import numpy
import pytest

import blosc2


@pytest.mark.parametrize(
    "obj, cparams, dparams",
    [
        (numpy.random.randint(0, 10, 10), {"compcode": blosc2.LZ4, "clevel": 6}, {}),
        (numpy.arange(10), {}, {"nthreads": 4}),
        (
            numpy.random.randint(0, 1000 + 1, 1000),
            {"splitmode": blosc2.ALWAYS_SPLIT, "nthreads": 5},
            {"schunk": None},
        ),
        (numpy.arange(45, dtype=numpy.float64), {"compcode": blosc2.LZ4HC, "typesize": 9}, {}),
        (numpy.arange(50, dtype=numpy.int64), blosc2.cparams_dflts, blosc2.dparams_dflts),
    ],
)
def test_compress2_numpy(obj, cparams, dparams):
    bytes_obj = obj.tobytes()
    c = blosc2.compress2(obj, **cparams)

    dest = bytearray(obj)
    blosc2.decompress2(c, dst=dest, **dparams)
    assert dest == bytes_obj

    dest2 = numpy.empty(obj.shape, obj.dtype)
    blosc2.decompress2(c, dst=dest2, **dparams)
    assert numpy.array_equal(dest2, obj)

    dest3 = blosc2.decompress2(c, **dparams)
    assert dest3 == bytes_obj

    dest4 = numpy.empty(obj.shape, obj.dtype)
    blosc2.decompress2(c, dst=memoryview(dest4), **dparams)
    assert numpy.array_equal(dest4, obj)


@pytest.mark.parametrize(
    "nbytes, cparams, dparams",
    [
        (7, {"compcode": blosc2.LZ4, "clevel": 6}, {}),
        (641091, {}, {"nthreads": 4}),
        (136, {}, {}),
        (1231, blosc2.cparams_dflts, blosc2.dparams_dflts),
    ],
)
def test_compress2(nbytes, cparams, dparams):
    bytes_obj = b" " * nbytes
    c = blosc2.compress2(bytes_obj, **cparams)

    dest = bytearray(bytes_obj)
    blosc2.decompress2(c, dst=dest, **dparams)
    assert dest == bytes_obj

    dest2 = blosc2.decompress2(c, **dparams)
    assert dest2 == bytes_obj

    dest3 = bytearray(bytes_obj)
    blosc2.decompress2(numpy.array([c]), dst=dest3, **dparams)
    assert dest3 == bytes_obj


@pytest.mark.parametrize(
    "object, cparams, dparams",
    [(numpy.arange(0), {"compcode": blosc2.LZ4, "clevel": 6}, {}), (b"", {}, {"nthreads": 3})],
)
def test_raise_error(object, cparams, dparams):
    c = blosc2.compress2(object, **cparams, **dparams)

    dest = bytearray(object)
    with pytest.raises(ValueError):
        blosc2.decompress2(c, dst=dest)

    dest3 = blosc2.decompress2(c)
    if type(object) is bytes:
        assert dest3 == object
    else:
        assert dest3 == object.tobytes()

    dest5 = bytearray(object)
    with pytest.raises(ValueError):
        blosc2.decompress2(numpy.array([c]), dst=dest5)

    with pytest.raises(ValueError):
        blosc2.decompress2(b"")
