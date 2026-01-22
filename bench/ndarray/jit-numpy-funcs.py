#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Benchmarks of using the jit decorator with arbitrary NumPy functions.

import numpy as np
from time import time
import numba

import blosc2

N = 30_000   # working size is N * N * 4 * 2 bytes ~ 6.7 GB
# N = 65_000   # working size is N * N * 4 * 2 bytes ~ 32 GB

# Create some sample data
t0 = time()
na = np.linspace(0, 1, N * N, dtype="float32").reshape(N, N)
nb = np.linspace(1, 2, N * N, dtype="float32").reshape(N, N)
nc = np.linspace(-10, 10, N, dtype="float32")
print(f"Time to create data (np.ndarray): {time() - t0:.3f} s")

t0 = time()
a = blosc2.linspace(0, 1, N * N, dtype="float32", shape=(N, N))
b = blosc2.linspace(1, 2, N * N, dtype="float32", shape=(N, N))
c = blosc2.linspace(-10, 10, N, dtype="float32", shape=(N,))
print(f"Time to create data (NDArray): {time() - t0:.3f} s")
#print("a.chunks: ", a.chunks, "a.blocks: ", a.blocks)

# Take NumPy as reference
def expr_numpy(a, b, c):
    # return np.cumsum(((na**3 + np.sin(na * 2)) < nc) & (nb > 0), axis=0)
    # The next is equally illustrative, but can achieve better speedups
    return np.sum(((na**3 + np.sin(na * 2)) < np.cumulative_sum(nc)) & (nb > 0), axis=1)

@blosc2.jit
def expr_jit(a, b, c):
    # return np.cumsum(((a**3 + np.sin(a * 2)) < c) & (b > 0), axis=0)
    return np.sum(((a**3 + np.sin(a * 2)) < np.cumulative_sum(c)) & (b > 0), axis=1)

@numba.jit
def expr_numba(a, b, c):
    # numba fails with the next with:
    # """No implementation of function Function(<function cumsum at 0x101a30720>) found for signature:
    #  >>> cumsum(array(bool, 2d, C), axis=Literal[int](0))"""
    # return np.cumsum(((a**3 + np.sin(a * 2)) < c) & (b > 0), axis=0)
    # The np.cumulative_sum() is not supported yet by numba
    # return np.sum(((a**3 + np.sin(a * 2)) < np.cumulative_sum(c)) & (b > 0), axis=1)
    return np.sum(((a**3 + np.sin(a * 2)) < np.cumsum(c)) & (b > 0), axis=1)

times = []
# Call the NumPy function natively on NumPy containers
t0 = time()
result = expr_numpy(a, b, c)
tref = time() - t0
times.append(tref)
print(f"Time for native NumPy: {tref:.3f} s")

# Call the function with the blosc2.jit decorator, using NumPy containers
t0 = time()
result = expr_jit(na, nb, nc)
times.append(time() - t0)
print(f"Time for blosc2.jit (np.ndarray): {times[-1]:.3f} s, speedup: {tref / times[-1]:.2f}x")

# Call the function with the blosc2.jit decorator, using Blosc2 containers
t0 = time()
result = expr_jit(a, b, c)
times.append(time() - t0)
print(f"Time for blosc2.jit (blosc2.NDArray): {times[-1]:.3f} s, speedup: {tref / times[-1]:.2f}x")

# Call the function with the jit decorator, using NumPy containers
t0 = time()
result = expr_numba(na, nb, nc)
times.append(time() - t0)
print(f"Time for numba.jit (np.ndarray, first run): {times[-1]:.3f} s, speedup: {tref / times[-1]:.2f}x")
t0 = time()
result = expr_numba(na, nb, nc)
times.append(time() - t0)
print(f"Time for numba.jit (np.ndarray): {times[-1]:.3f} s, speedup: {tref / times[-1]:.2f}x")


# Plot the results using an horizontal bar chart
import matplotlib.pyplot as plt

labels = ['NumPy', 'blosc2.jit (np.ndarray)', 'blosc2.jit (blosc2.NDArray)', 'numba.jit (first run)', 'numba.jit (cached)']
# Reverse the labels and times arrays
labels_rev = labels[::-1]
times_rev = times[::-1]

# Create position indices for the reversed data
x = np.arange(len(labels_rev))

fig, ax = plt.subplots(figsize=(10, 6))

# Define colors for different categories
colors = ['#FF9999', '#66B2FF', '#66B2FF', '#99CC99', '#99CC99']  # Red for NumPy, Blue for blosc2, Green for numba
# Note: colors are in reverse order to match the reversed data
colors_rev = colors[::-1]

bars = ax.barh(x, times_rev, height=0.35, color=colors_rev, label='Time (s)')

# Add speedup annotations at the end of each bar
# NumPy is our reference (the first element in original array, last in reversed)
numpy_time = tref  # Reference time for NumPy
for i, (bar, time) in enumerate(zip(bars, times_rev)):
    # Skip the NumPy bar since it's our reference
    if i < len(times_rev) - 1:  # Skip the last bar (NumPy)
        speedup = numpy_time / time
        ax.annotate(f'({speedup:.1f}x)',
                   (bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2),
                   va='center')

ax.set_xlabel('Time (s)')
ax.set_title("""Compute: np.sum(((a**3 + np.sin(a * 2)) < np.cumsum(c)) & (b > 0), axis=1)
             (Execution time for different decorators)""")
ax.set_yticks(x)
ax.set_yticklabels(labels_rev)

# Create custom legend with only one entry per category
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#FF9999', label='NumPy'),
    Patch(facecolor='#66B2FF', label='blosc2.jit'),
    Patch(facecolor='#99CC99', label='numba.jit')
]
ax.legend(handles=legend_elements, loc='best')

plt.tight_layout()
plt.savefig('jit_benchmark_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
