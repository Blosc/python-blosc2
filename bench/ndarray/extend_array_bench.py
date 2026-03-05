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
import matplotlib.pyplot as plt

dtype=np.float32
Ns = [100_000, 1_000_000, 10_000_000, 100_000_000]
rsize_times = []
fill_times = []
for N in Ns:
    c, b = blosc2.compute_chunks_blocks((N,), dtype=dtype)
    tic = time()
    data = np.linspace(0, 1, num=N, dtype=dtype)
    toc = time()
    bufgen_time = toc-tic
    print(f"Time to generate buffer of data: {bufgen_time} s")
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
    assert c == arr.chunks and b == arr.blocks
    rsize_times += [rsize_time]
    fill_times += [fill_time]
    del data
    del arr

x = np.arange(len(Ns))
w = 0.2
fig = plt.figure()
ax = plt.gca()
ax.bar(x, rsize_times, color='r', label='Resize', width=w)
ax.set_ylabel('Resize time (s)', color='r')
ax.tick_params(axis='y', labelcolor='r')
# ax.set_yscale('log')
ax2 = ax.twinx()
ax2.bar(x+w, fill_times, color='b', label='Fill', width=w)
ax2.set_ylabel('Fill time (s)', color='b')
ax2.tick_params(axis='y', labelcolor='b')
# ax2.set_yscale('log')
ax.set_xticks(x + w/2, [f'$10^{i}$' for i in np.int64(np.log10(Ns))])
ax.set_xlabel('Array length $N$')
fig.tight_layout()
fig.savefig('extend_array_bench.png', format='png', bbox_inches='tight')
