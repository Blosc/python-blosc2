#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Benchmark to compare NumPy and Blosc2 cumulative_sum for large arrays

import blosc2
import numpy as np
from time import time
import matplotlib.pyplot as plt

blosc2_dt = []
np_dt = []
arr_size = []
sizes = (np.array([1, 2, 4, 8, 16]) * 1024 ** 3 / 8)**(1/3)
for N in sizes:
    shape = (int(N),) * 3
    arr = blosc2.arange(0, np.prod(shape), shape=shape, dtype=np.float64)
    dt = 0
    for axis in (0, 1, 2):
        tic = time()
        res = blosc2.cumulative_sum(arr, axis=axis)
        toc = time()
        dt += (toc-tic) / 3
    blosc2_dt += [dt]

    arr = arr[()]
    dt = 0
    for axis in (0, 1, 2):
        tic = time()
        res = np.cumulative_sum(arr, axis=axis)
        toc = time()
        dt += (toc-tic) / 3
    np_dt += [dt]
    arr_size += [round(arr.dtype.itemsize * np.prod(shape) / 1024**3, 1)]

results = {'blosc2': blosc2_dt, 'numpy': np_dt, 'sizes': arr_size}


blosc2_dt = results['blosc2']
np_dt = results['numpy']
arr_size = results['sizes']
w = 0.2
x = np.arange(len(arr_size))
plt.bar(x, blosc2_dt, width = w, label='Blosc2')
plt.bar(x + w, np_dt, width=w, label='Numpy')
plt.gca().set_yscale('log')
plt.xticks(x, arr_size)
plt.xlabel('Array size (GB)')
plt.ylabel('Average Time (s)')
plt.title(f'Cumulative_sum for 3D array')
plt.legend()
plt.savefig('cumsumbench.png', format='png')
