########################################################################
#
#       Created: April 30, 2021
#       Author:  The Blosc development team - blosc@blosc.org
#
########################################################################


import pytest

import blosc2


@pytest.mark.parametrize("cname", ["lz4", "blosclz", "lz4hc", "zlib", "zstd"])
def test_comp_info(cname):
    blosc2.compressor_list()
    blosc2.clib_info(cname)
    blosc2.set_compressor(cname)
    assert cname == blosc2.get_compressor()
    blosc2.print_versions()
