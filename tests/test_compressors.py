#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################


import pytest

import blosc2


@pytest.mark.parametrize("gil", [True, False])
@pytest.mark.parametrize(
    "clevel, codec",
    [
        (8, blosc2.Codec.BLOSCLZ),
        (9, blosc2.Codec.LZ4),
        (3, blosc2.Codec.LZ4HC),
        (5, blosc2.Codec.ZLIB),
        (2, blosc2.Codec.ZSTD),
    ],
)
@pytest.mark.parametrize("filt", list(blosc2.Filter))
def test_compressors(clevel, filt, codec, gil):
    blosc2.set_releasegil(gil)
    src = b"Something to be compressed" * 100
    dest = blosc2.compress(src, 1, clevel, filt, codec)
    src2 = blosc2.decompress(dest)
    assert src == src2
    if codec == blosc2.Codec.LZ4HC:
        assert blosc2.get_clib(dest).lower() == "lz4"
    else:
        assert blosc2.get_clib(dest).lower() == codec.name.lower()
    blosc2.free_resources()
