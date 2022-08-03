########################################################################
#
#       Author:  The Blosc development team - blosc@blosc.org
#
########################################################################


import pytest

import blosc2


@pytest.mark.parametrize(
    "clevel, cname",
    [(8, "blosclz"), (9, "lz4"), (3, "lz4hc"), (5, "zlib"), (2, "zstd")],
)
@pytest.mark.parametrize(
    "filt", list(blosc2.Filter)
)
def test_compressors(clevel, filt, cname):
    src = b"Something to be compressed" * 100
    dest = blosc2.compress(src, 1, clevel, filt, cname)
    src2 = blosc2.decompress(dest)
    assert src == src2
    if cname == "lz4hc":
        assert blosc2.get_clib(dest).lower() == b"lz4"
    else:
        assert blosc2.get_clib(dest).lower() == cname.encode("utf-8").lower()
    blosc2.free_resources()
