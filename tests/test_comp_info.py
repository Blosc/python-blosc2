#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

import numpy as np
import pytest

import blosc2


def test_print_versions(capsys):
    version = blosc2.blosc2_ext.MINIEXPR_VERSION_STRING
    assert isinstance(version, str)
    assert version
    assert blosc2.print_versions() is None
    lines = capsys.readouterr().out.splitlines()
    version_line = lines.index(f"miniexpr version: {version}")
    assert lines[version_line + 1] == f"NumPy version: {np.__version__}"


@pytest.mark.parametrize("codec", blosc2.compressor_list())
def test_comp_info(codec):
    blosc2.clib_info(codec)
    blosc2.set_compressor(codec)
    assert codec.name.lower() == blosc2.get_compressor()

    arr = np.zeros(1_000_000, dtype="V8")
    src = blosc2.compress2(arr)
    nbytes, cbytes, blocksize = blosc2.get_cbuffer_sizes(src)
    assert nbytes == arr.size * arr.dtype.itemsize
    assert cbytes == blosc2.MAX_OVERHEAD
    # When raising the next limit when this would fail in the future, one should raise the SIZE too
    assert blocksize <= 2**23
    blosc2.print_versions()
