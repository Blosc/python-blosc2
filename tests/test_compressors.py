########################################################################
#
#       Created: April 30, 2021
#       Author:  The Blosc development team - blosc@blosc.org
#
########################################################################


import pytest

import blosc2


@pytest.mark.parametrize(
    "typesize, clevel, cname",
    [(1, 8, "blosclz"), (1, 9, "lz4"), (1, 3, "lz4hc"), (1, 5, "zlib"), (1, 2, "zstd")],
)
@pytest.mark.parametrize(
    "filt", [blosc2.BITSHUFFLE, blosc2.SHUFFLE, blosc2.NOFILTER, blosc2.DELTA, blosc2.TRUNC_PREC]
)
def test_compressors(typesize, clevel, filt, cname):
    src = b"Something to be compressed" * 100
    dest = blosc2.compress(src, typesize, clevel, filt, cname)
    src2 = blosc2.decompress(dest)
    assert src == src2
    if cname == "lz4hc":
        assert blosc2.get_clib(dest).lower() == b"lz4"
    else:
        assert blosc2.get_clib(dest).lower() == cname.encode("utf-8").lower()
    blosc2.free_resources()
