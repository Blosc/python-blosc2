#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Benchmark for extending empty array

import blosc2
from time import time
import numpy as np
dtype=np.float32
N = 1000_000_000
c, b = blosc2.compute_chunks_blocks((N,), dtype=dtype)
print(f'Setting array chunks, blocks to {c, b}')
data = np.linspace(0, 1, N, dtype=dtype)
arr = blosc2.zeros((0,), chunks=c, blocks=b, dtype=dtype)
tic = time()
arr.resize((N,))
toc = time()
rsize_time = toc-tic
tic = time()
arr[:] = data
toc = time()
fill_time = toc-tic
np.testing.assert_array_equal(arr, data)
print(f'Filled array chunks, blocks: {arr.chunks, arr.blocks}')
print(f'Resize took {rsize_time} s')
print(f'Fill took {fill_time} s')
