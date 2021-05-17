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
@pytest.mark.parametrize(
    "gilstate",
    [
        True,
        False,
    ]
)
def test_compress2_numpy(obj, cparams, dparams, gilstate):
    blosc2.set_releasegil(gilstate)
    bytes_obj = obj.tobytes()
    c = blosc2.compress2(bytes_obj, **cparams)
    d = blosc2.decompress2(c, **dparams)
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
@pytest.mark.parametrize(
    "gilstate",
    [
        True,
        False,
    ]
)
def test_compress2(nbytes, cparams, dparams, gilstate):
    blosc2.set_releasegil(gilstate)
    bytes_obj = b" " * nbytes
    c = blosc2.compress2(bytes_obj, **cparams)
    d = blosc2.decompress2(c, **dparams)
    assert bytes_obj == d
