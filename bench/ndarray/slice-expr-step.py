#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Benchmark for computing a slice with non-unit steps of a expression in a ND array.

import blosc2
import numpy as np
import matplotlib.pyplot as plt
from memory_profiler import profile, memory_usage

N = 50_000
LARGE_SLICE = False
ndim = 2
shape = (N, ) * ndim
a = blosc2.linspace(start=0, stop=np.prod(shape), num=np.prod(shape), dtype=np.float64, shape=shape)
_slice = (slice(0, N, 2),) if LARGE_SLICE else (slice(0, N, N//4),)
expr = 2 * a ** 2

@profile
def _slice_():
    res1 = expr.slice(_slice)
    print(f'Result of slice occupies {res1.schunk.cbytes / 1024**2:.2f} MiB')
    return res1

@profile
def _gitem():
    res2 = expr[_slice]
    print(f'Result of _getitem_ occupies {np.prod(res2.shape) * res2.itemsize / 1024**2:.2f} MiB')
    return res2

interval = 0.001
offset = 0
for f in [_slice_, _gitem]:
    mem = memory_usage((f,), interval=interval)
    times = offset + interval * np.arange(len(mem))
    offset = times[-1]
    plt.plot(times, mem)

plt.xlabel('Time (s)')
plt.ylabel('Memory usage (MiB)')
lab = 'LARGE' if LARGE_SLICE else 'SMALL'
plt.title(f'{lab} slice w/steps, Linux Blosc2 {blosc2.__version__}')
plt.legend([f'expr.slice({_slice}', f'expr[{_slice}]'])
plt.savefig(f'sliceexpr_{lab}_Blosc{blosc2.__version__.replace(".","_")}.png', format="png")
