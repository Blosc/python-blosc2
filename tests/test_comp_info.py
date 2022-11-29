#######################################################################
# Copyright (C) 2019-present, Blosc Development team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################


import pytest

import blosc2


@pytest.mark.parametrize("codec", list(blosc2.Codec))
def test_comp_info(codec):
    blosc2.compressor_list()
    blosc2.clib_info(codec)
    blosc2.set_compressor(codec)
    assert codec.name.lower() == blosc2.get_compressor()
    blosc2.print_versions()
