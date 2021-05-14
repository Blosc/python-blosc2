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
        (numpy.random.randint(0, 1000 + 1, 1000), {"nthreads": 5}, {"schunk": None}),
        (numpy.arange(45, dtype=numpy.float64), {"compcode": blosc2.LZ4HC, "typesize": 9}, {}),
        (numpy.arange(50, dtype=numpy.int64), blosc2.cparams_dflts, blosc2.dparams_dflts),
    ],
)
def test_context_numpy(obj, cparams, dparams):
    bytes_obj = obj.tobytes()
    c = blosc2.compress_ctx(bytes_obj, **cparams)
    d = blosc2.decompress_ctx(c, **dparams)
    assert bytes_obj == d


@pytest.mark.parametrize(
    "nbytes, cparams, dparams",
    [
        (7, {"compcode": blosc2.LZ4, "clevel": 6}, {}),
        (641091, {}, {"nthreads": 4}),
        (136, {}, {}),
        (1231, blosc2.cparams_dflts, blosc2.dparams_dflts),
    ],
)
def test_context(nbytes, cparams, dparams):
    bytes_obj = b" " * nbytes
    c = blosc2.compress_ctx(bytes_obj, **cparams)
    d = blosc2.decompress_ctx(c, **dparams)
    assert bytes_obj == d
