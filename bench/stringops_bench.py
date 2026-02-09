#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""
Compare miniexpr and non-miniexpr paths for string ops.
"""

import time
import numpy as np
import blosc2

N = 3 * int(1e6)
nparr = np.asarray(['pepe','josé','francisco'])
nparr = np.repeat(nparr, N//3)

arr1 = blosc2.asarray(nparr)
arr2 = blosc2.full(arr1.shape, 'francisco', blocks=arr1.blocks, chunks=arr1.chunks)
arr3 = blosc2.asarray(nparr, chunks=tuple(a - 1 for a in arr1.chunks))

names = ['==', 'contains', 'startswith', 'endswith']
functuple = (lambda a, b : a==b, blosc2.contains, blosc2.startswith, blosc2.endswith)
for name, func in zip(names, functuple):
    expr = func(arr1, arr2)
    dtic = time.time()
    res = expr[()]
    dtoc = time.time()
    print(f'{name} took {dtoc-dtic}s for miniexpr')
    expr = func(arr3, arr2)
    dtic = time.time()
    res = expr[()]
    dtoc = time.time()
    print(f'{name} took {dtoc-dtic}s for normal path')
