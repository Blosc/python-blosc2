########################################################################
#
#       Author:  The Blosc development team - blosc@blosc.org
#
########################################################################


import pytest

import blosc2


@pytest.mark.parametrize("arr", [b"", b"1" * 7])
@pytest.mark.parametrize("gil", [True, False])
def test_bytes_array(arr, gil):
    blosc2.set_releasegil(gil)
    dest = blosc2.compress(arr, 1)
    assert arr == blosc2.decompress(dest)
