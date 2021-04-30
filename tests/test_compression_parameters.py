########################################################################
#
#       Created: April 30, 2021
#       Author:  The Blosc development team - blosc@blosc.org
#
########################################################################


import blosc2
import pytest


@pytest.mark.parametrize("nthreads, blocksize",
                         [
                             (2, 0),
                             (1, 30),
                             (4, 5)
                         ]
                         )
def test_compression_parameters(nthreads, blocksize):
    blosc2.set_nthreads(nthreads)
    blosc2.set_blocksize(blocksize)
