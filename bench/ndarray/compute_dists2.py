#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

# Benchmark for comparing compute speeds of Blosc2 and Numexpr.
# This version compares across different distributions of data:
# constant, arange, linspace, or random
# The expression can be any valid Numexpr expression.

import blosc2
from time import time
import numpy as np
import numexpr as ne
import matplotlib.pyplot as plt
import seaborn as sns
import sys


# Bench params
N = 10_000
step = 3000
dtype = np.dtype(np.float64)
persistent = False
distributions = ["constant", "arange", "linspace", "random"]
expr = "(a - b)"
#expr = "sum(a - b)"
#expr = "cos(a)**2 + sin(b)**2 - 1"
#expr = "sum(cos(a)**2 + sin(b)**2 - 1)"

# Params for large memory machines
if len(sys.argv) > 1 and sys.argv[1] == "large":
    N = 30_000  # For large memory machines
    distributions = ["constant", "arange", "linspace"]

# Set default compression params
cparams = blosc2.CParams(clevel=1, codec=blosc2.Codec.BLOSCLZ)
blosc2.cparams_dflts["codec"] = cparams.codec
blosc2.cparams_dflts["clevel"] = cparams.clevel
# Set default storage params
storage = blosc2.Storage(contiguous=True, mode="w")
blosc2.storage_dflts["contiguous"] = storage.contiguous
blosc2.storage_dflts["mode"] = storage.mode

# Create dictionaries to store results for each distribution
blosc2_speeds = {dist: [] for dist in distributions}
numexpr_speeds = {dist: [] for dist in distributions}
ws_sizes = []

# Generate working set sizes once
sizes = list(range(step, N + step, step))
for i in sizes:
    ws_sizes.append((i * i * 3 * np.dtype(dtype).itemsize) / 2**30)  # Approximate size in GB

# Loop through different distributions for benchmarking
for dist in distributions:
    print(f"\nBenchmarking {dist} distribution...")

    # Evaluate using Blosc2
    for i in sizes:
        shape = (i, i)
        urlpath = {name: None for name in ("a", "b", "c")}

        if dist == "constant":
            a = blosc2.ones(shape, dtype=dtype, urlpath=urlpath['a'])
            b = blosc2.full(shape, 2, dtype=dtype, urlpath=urlpath['b'])
        elif dist == "arange":
            a = blosc2.arange(0, i * i, dtype=dtype, shape=shape, urlpath=urlpath['a'])
            b = blosc2.arange(i * i, 2* i * i, dtype=dtype, shape=shape, urlpath=urlpath['b'])
        elif dist == "linspace":
            a = blosc2.linspace(0, 1, dtype=dtype, shape=shape, urlpath=urlpath['a'])
            b = blosc2.linspace(1, 2, dtype=dtype, shape=shape, urlpath=urlpath['b'])
        elif dist == "random":
            _ = np.random.random(shape)
            a = blosc2.fromiter(np.nditer(_), dtype=dtype, shape=shape, urlpath=urlpath['a'])
            # b = a.copy(urlpath=urlpath['b'])  # faster, but output is not random
            _ = np.random.random(shape)
            b = blosc2.fromiter(np.nditer(_), dtype=dtype, shape=shape, urlpath=urlpath['b'])

        t0 = time()
        c = blosc2.lazyexpr(expr).compute(urlpath=urlpath['c'])
        t = time() - t0
        speed = (a.schunk.nbytes + b.schunk.nbytes + c.schunk.nbytes) / 2**30 / t
        print(f"Blosc2 - {dist} - Size {i}x{i}: {speed:.2f} GB/s - cratio: {c.schunk.cratio:.1f}x")
        blosc2_speeds[dist].append(speed)

    # Evaluate using Numexpr
    for i in sizes:
        shape = (i, i)

        if dist == "constant":
            a = np.ones(shape, dtype=dtype)
            b = np.full(shape, 2, dtype=dtype)
        elif dist == "arange":
            a = np.arange(0, i * i, dtype=dtype).reshape(shape)
            b = np.arange(i * i, 2 * i * i, dtype=dtype).reshape(shape)
        elif dist == "linspace":
            a = np.linspace(0, 1, num=i * i, dtype=dtype).reshape(shape)
            b = np.linspace(1, 2, num=i * i, dtype=dtype).reshape(shape)
        elif dist == "random":
            a = np.random.random(shape)
            b = np.random.random(shape)

        t0 = time()
        c = ne.evaluate(expr)
        t = time() - t0
        speed = (a.nbytes + b.nbytes + c.nbytes) / 2**30 / t
        print(f"Numexpr - {dist} - Size {i}x{i}: {speed:.2f} GB/s")
        numexpr_speeds[dist].append(speed)

# Create a figure with four subplots (2x2 grid)
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)

# Flatten axes for easier iteration
axes = axes.flatten()

# Plot each distribution in its own subplot
for i, dist in enumerate(distributions):
    axes[i].plot(ws_sizes, blosc2_speeds[dist], marker='o', linestyle='-', label="Blosc2")
    axes[i].plot(ws_sizes, numexpr_speeds[dist], marker='s', linestyle='--', label="Numexpr")
    axes[i].set_title(f"{dist.capitalize()} Distribution")
    axes[i].set_ylabel("Speed (GB/s)")
    axes[i].grid(True)
    axes[i].legend()
    if i >= 2:  # Add x-label only to bottom subplots
        axes[i].set_xlabel("Working set size (GB)")

# Add a shared title
fig.suptitle(f"Blosc2 vs Numexpr Performance Across Different Data Distributions ({expr=})", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust the rect parameter to make room for the suptitle

# Save the unified plot with subplots
plt.savefig("blosc2_vs_numexpr_subplots.png", dpi=300, bbox_inches='tight')
plt.show()
