#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

# Benchmark for comparing compute speeds of Blosc2 and Numexpr.
# One can use different distributions of data:
# constant, arange, linspace, or random
# The expression can be any valid Numexpr expression.

import blosc2
from time import time
import numpy as np
import numexpr as ne

# Bench params
N = 30_000
step = 3000
dtype = np.dtype(np.float64)
persistent = False
dist = "constant"  # "arange" or "linspace" or "constant" or "random"
expr = "(a - b)"
#expr = "sum(a - b)"
#expr = "cos(a)**2 + sin(b)**2 - 1"
#expr = "sum(cos(a)**2 + sin(b)**2 - 1)"

# Set default compression params
cparams = blosc2.CParams(clevel=1, codec=blosc2.Codec.BLOSCLZ)
blosc2.cparams_dflts["codec"] = cparams.codec
blosc2.cparams_dflts["clevel"] = cparams.clevel
# Set default storage params
storage = blosc2.Storage(contiguous=True, mode="w")
blosc2.storage_dflts["contiguous"] = storage.contiguous
blosc2.storage_dflts["mode"] = storage.mode

urlpath = dict((aname, None) for aname in ("a", "b", "c"))
if persistent:
    urlpath = dict((aname, f"{aname}.b2nd") for aname in ("a", "b", "c"))

btimes = []
bspeeds = []
ws_sizes = []
rng = np.random.default_rng()
for i in range(step, N + step, step):
    shape = (i, i)
    # shape = (i * i,)
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
        t0 = time()
        _ = np.random.random(shape)
        a = blosc2.fromiter(np.nditer(_), dtype=dtype, shape=shape, urlpath=urlpath['a'])
        b = a.copy(urlpath=urlpath['b'])
        # This uses less memory, but it is 2x-3x slower
        # iter_ = (rng.random() for _ in range(i**2 * 2))
        # a = blosc2.fromiter(iter_, dtype=dtype, shape=shape, urlpath=urlpath['a'])
        # b = blosc2.fromiter(iter_, dtype=dtype, shape=shape, urlpath=urlpath['b'])
        t = time() - t0
        #print(f"Time to create data: {t:.5f} s - {a.schunk.nbytes/t / 1e9:.2f} GB/s")
    else:
        raise ValueError("Invalid distribution type")

    t0 = time()
    c = blosc2.lazyexpr(expr).compute(urlpath=urlpath['c'])
    t = time() - t0
    ws_sizes.append((a.schunk.nbytes + b.schunk.nbytes + c.schunk.nbytes) / 2**30)
    speed = ws_sizes[-1] / t
    print(f"Time to compute a - b: {t:.5f} s -- {speed:.2f} GB/s -- cratio: {c.schunk.cratio:.1f}x")
    #print(f"result: {c[()]}")
    btimes.append(t)
    bspeeds.append(speed)

# Evaluate using Numexpr compute engine
ntimes = []
nspeeds = []
for i in range(step, N + step, step):
    shape = (i, i)
    # shape = (i * i,)
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
    else:
        raise ValueError("Invalid distribution type")

    t0 = time()
    c = ne.evaluate(expr)
    t = time() - t0
    ws_size = (a.nbytes + b.nbytes + c.nbytes) / 2**30
    speed = ws_size / t
    print(f"Time to compute with Numexpr: {t:.5f} s - {speed:.2f} GB/s")
    #print(f"result: {c}")
    ntimes.append(t)
    nspeeds.append(speed)

# Plot
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))
plt.plot(ws_sizes, bspeeds, label="Blosc2", marker='o')
plt.plot(ws_sizes, nspeeds, label="Numexpr", marker='o')
# Set y-axis to start from 0
plt.ylim(bottom=0)
plt.xlabel("Working set (GB)")
#plt.ylabel("Time (s)")
plt.ylabel("Speed (GB/s)")
plt.title(f"Blosc2 vs Numexpr performance -- {dist} distribution")
plt.legend()
#plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
#plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.2f}'))
plt.grid()
plt.show()
# Save the figure
plt.savefig("blosc2_vs_numexpr.png", dpi=300, bbox_inches='tight')
plt.close()
