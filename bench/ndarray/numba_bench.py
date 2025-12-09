#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

# Compare Numba-compiled UDF with standard UDF

import blosc2
import numpy as np
import numba
import time
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})
nios = 4
intensity_val = 147 / nios
expr = "exp(sqrt((sin(a) ** 2 + (cos(b) + arctan(c)) ** 3) * (1 + sin(b) ** 2 + cos(c) ** 2)))"
dtype = np.float64()

sizes = np.sqrt(1024 ** 3 * np.array([1 / 2**5, 1 / 2**4, 1 / 2**3, 1 / 2**2, 1 / 2, 1]) / dtype.itemsize)  # operand size up to 1GB
@numba.jit(nopython=True, parallel=True)
def myudf_numba(inputs, output, offset):
    a, b, c = inputs
    output[:] = np.exp(np.sqrt((np.sin(a) ** 2 + (np.cos(b) + np.arctan(c)) ** 3) * (1 + np.sin(b) ** 2 + np.cos(c) ** 2)))

def myudf(inputs, output, offset):
    a, b, c = inputs
    output[:] = np.exp(np.sqrt((np.sin(a) ** 2 + (np.cos(b) + np.arctan(c)) ** 3) * (1 + np.sin(b) ** 2 + np.cos(c) ** 2)))

n = 10
n = int(n)
a = blosc2.arange(0, n**2, shape=(n, n), dtype=dtype)
b = blosc2.arange(0, n**2, shape=(n, n), dtype=dtype)
c = blosc2.arange(0, n**2, shape=(n, n), dtype=dtype)

larray_nb = blosc2.lazyudf(myudf_numba, (a, b, c), c.dtype)
t0 = time.time()
res = larray_nb.compute()
dt = time.time() - t0

MAX_THREADS = numba.get_num_threads()

for nthreads, c_ in zip([MAX_THREADS], ['r']):
    numba.set_num_threads(nthreads)

    blosc2_parallel_times = []
    np_parallel_times = []
    blosc2_times = []
    for n in sizes:
        n = int(n)
        a = blosc2.arange(0, n**2, shape=(n, n), dtype=dtype)
        b = blosc2.arange(0, n**2, shape=(n, n), dtype=dtype)
        c = blosc2.arange(0, n**2, shape=(n, n), dtype=dtype)

        larray_nb = blosc2.lazyudf(myudf_numba, (a, b, c), c.dtype)
        t0 = time.time()
        res = larray_nb.compute()
        dt = time.time() - t0
        blosc2_parallel_times += [intensity_val * n ** 2 / dt / 1e9]
        if nthreads == MAX_THREADS:
            larray_nb = blosc2.lazyudf(myudf, (a, b, c), c.dtype)
            t0 = time.time()
            res = larray_nb.compute()
            dt = time.time() - t0
            blosc2_times += [intensity_val * n ** 2 / dt / 1e9]

        # a, b, c, res = a[:], b[:], c[:], res[:]
        # t0 = time.time()
        # myudf((a, b, c), res, ())
        # dt = time.time() - t0
        # np_parallel_times += [intensity_val * n ** 2 / dt / 1e9]

    # plt.loglog(4 * sizes**2 / 1024**3 * dtype.itemsize, np_parallel_times, color=c_, ls='--')

gigas = 4 * sizes**2 / 1024**3 * dtype.itemsize
if nthreads == MAX_THREADS:
    plt.loglog(gigas, blosc2_times, color='b', ls='-', label=f'Blosc2', lw=3)
boost = np.mean(np.divide(blosc2_parallel_times, blosc2_times))
plt.loglog(gigas, blosc2_parallel_times, color=c_, ls='-', label=f'Blosc2 + Numba', lw=3)

plt.xlabel('Working set size (GB)')
plt.ylabel("GFLOPS / s")
plt.xticks([.1, .5, 1, 2, 4], [.1, .5, 1, 2, 4])
plt.yticks([1, 2, 4, 8], [1, 2, 4, 8])
# plt.plot([], [], 'k-', label='blosc2 + numba')
# plt.plot([], [], 'k--', label='NumPy + numba')
# plt.plot([], [], 'k:', label='blosc2')

plt.legend()
plt.title('Accelerate with Blosc2 + Numba!')
plt.annotate(f'Performance boost: {round(boost, 1)}x !', (0.31, .6), xycoords='figure fraction', bbox=dict(boxstyle="round", fc="0.8", color='b', alpha=.5))
idx = len(gigas)//4
plt.annotate("", xytext=(gigas[idx], blosc2_times[idx]), xy=(gigas[idx], blosc2_parallel_times[idx]),
            arrowprops=dict(arrowstyle="<->", lw=3))
plt.tight_layout()
plt.savefig('temp.png', format='png', bbox_inches='tight')
