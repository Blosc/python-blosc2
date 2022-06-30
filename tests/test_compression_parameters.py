########################################################################
#
#       Author:  The Blosc development team - blosc@blosc.org
#
########################################################################


import pytest

import blosc2


@pytest.mark.parametrize("nthreads, blocksize", [(2, 0), (1, 30), (4, 5)])
def test_compression_parameters(nthreads, blocksize):
    blosc2.set_nthreads(nthreads)
    blosc2.set_blocksize(blocksize)
