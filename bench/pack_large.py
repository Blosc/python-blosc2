#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################


"""
Small benchmark that exercises packaging of arrays larger than 2 GB.
"""

import time

import numpy as np

import blosc2

NREP = 1
N = int(4e8 - 2**27)  # larger than 2 GB
Nexp = np.log10(N)

print(f"Creating NumPy array with {float(N):.3g} int64 elements...")
in_ = np.arange(N, dtype=np.int64)

if __name__ == "__main__":
    cparams = {
        "codec": blosc2.Codec.BLOSCLZ,
        "clevel": 9,
        # "filters": [blosc2.Filter.NOFILTER] * 4 + [blosc2.Filter.SHUFFLE, blosc2.Filter.BYTEDELTA],
        # "filters_meta": [0] * 6,
        # "splitmode": blosc2.SplitMode.NEVER_SPLIT,
    }
    print(f"Storing with {cparams=}")

    c = None
    ctic = time.time()
    for i in range(NREP):
        c = blosc2.pack_tensor(in_, cparams=cparams)
    ctoc = time.time()
    tc = (ctoc - ctic) / NREP
    print(
        "  Time for pack_tensor:   {:.3f} ({:.2f} GB/s)) ".format(tc, ((N * 8 / tc) / 2**30)),
    )
    print("\tcr: {:5.1f}x".format(in_.size * in_.dtype.itemsize * 1.0 / len(c)))

    out = None
    dtic = time.time()
    for i in range(NREP):
        out = blosc2.unpack_tensor(c)
    dtoc = time.time()

    td = (dtoc - dtic) / NREP
    print(
        "  Time for unpack_tensor:   {:.3f} s ({:.2f} GB/s)) ".format(td, ((N * 8 / td) / 2**30)),
    )
    assert np.array_equal(in_, out)
