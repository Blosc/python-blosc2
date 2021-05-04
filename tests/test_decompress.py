########################################################################
#
#       Created: May 4, 2021
#       Author:  The Blosc development team - blosc@blosc.org
#
########################################################################


import blosc2
import pytest
import numpy


@pytest.mark.parametrize("object",
                         [
                             numpy.random.randint(0, 10, 10),
                             numpy.arange(10),
                             numpy.random.randint(0, 1000 + 1, 1000)

                         ])
@pytest.mark.parametrize("cname",
                         [
                             'lz4',
                             'blosclz',
                             'lz4hc',
                             'zlib',
                             'zstd'
                         ])
def test_decompress(object, cname):
    c = blosc2.compress(object, cname=cname)

    dest = bytearray(object)
    blosc2.decompress(c, dst=dest)
    assert(dest == object.tobytes())

    dest2 = numpy.empty(object.shape, object.dtype)
    blosc2.decompress(c, dst=dest2)
    assert (numpy.array_equal(dest2, object))
