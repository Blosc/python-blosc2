#######################################################################
# Copyright (C) 2019-present, Blosc Development team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################


"""
Small benchmark that exercises packaging of arrays larger than 2 GB.
"""

import sys
import time

import blosc2
import numpy as np

NREP = 1
N = int(4e8 - 2**27)  # larger than 2 GB
Nexp = np.log10(N)

store = False
if len(sys.argv) > 1:
    store = True

# blosc2.set_nthreads(2)

print(f"Creating NumPy array with {float(N):.3g} int64 elements...")
in_ = np.arange(N, dtype=np.int64)

if store:
    codec = blosc2.Codec.BLOSCLZ
    print(f"Storing with codec {codec}")
    cparams = {"codec": codec, "clevel": 9}

    c = None
    ctic = time.time()
    for i in range(NREP):
        c = blosc2.pack_tensor(in_, cparams=cparams)
    ctoc = time.time()
    tc = (ctoc - ctic) / NREP
    print(
        "  Time for pack_tensor:   %.3f (%.2f GB/s)) "
        % (tc, ((N * 8 / tc) / 2 ** 30)),
    )
    print("\tcr: %5.1fx" % (in_.size * in_.dtype.itemsize * 1.0 / len(c)))

    with open("pack_compress2.bl2", 'wb') as f:
        f.write(c)

else:
    with open("pack_compress2.bl2", 'rb') as f:
        c = f.read()

    out = None
    dtic = time.time()
    for i in range(NREP):
        out = blosc2.unpack_tensor(c)
    dtoc = time.time()

    td = (dtoc - dtic) / NREP
    print(
        "  Time for unpack_tensor:   %.3f s (%.2f GB/s)) "
        % (td, ((N * 8 / td) / 2 ** 30)),
    )
    assert np.array_equal(in_, out)
