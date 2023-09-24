#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

import numpy as np

import blosc2

a = np.linspace(0, 1, 1_000_000, dtype=np.float64)
typesize = a.dtype.itemsize
c_bytesobj = blosc2.compress2(
    a,
    typesize=typesize,
    codec=blosc2.Codec.ZSTD,
    filters=[blosc2.Filter.TRUNC_PREC, blosc2.Filter.SHUFFLE],
    filters_meta=[20, 0],
)
assert len(c_bytesobj) < (len(a) * typesize)
cratio = (len(a) * typesize) / len(c_bytesobj)
print(f"cratio: {cratio:.3f}")

a_bytesobj2 = blosc2.decompress2(c_bytesobj)
# The next check does not work when using truncation (obviously)
# assert a_bytesobj == a_bytesobj2
