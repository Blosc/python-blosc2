#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Create an empty array with different compression parameters and set some values on it

import blosc2

cparams = blosc2.CParams(
    codec=blosc2.Codec.LZ4,
    clevel=5,
    nthreads=4,
    filters=[blosc2.Filter.DELTA, blosc2.Filter.TRUNC_PREC, blosc2.Filter.BITSHUFFLE],
    filters_meta=[0, 3, 0],  # keep just 3 bits in mantissa
)
a = blosc2.empty(shape=(40, 401), blocks=(6, 26), dtype="f8", cparams=cparams)

a[...] = 222
print(a.info)

print(a[:, 0])  # note the truncation filter at work
