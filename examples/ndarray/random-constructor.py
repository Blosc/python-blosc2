#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# This example shows how to use `blosc2.random` to build blosc2 arrays with the same
# API shape as `numpy.random.Generator`, but generated chunk-by-chunk in parallel.

from time import time

import numpy as np

import blosc2

# `default_rng` mirrors `numpy.random.default_rng`: same seeding, same distributions.
rng = blosc2.random.default_rng(42)

# Every method returns a blosc2.NDArray, not a numpy array. `shape` replaces numpy's
# `size` and is required (no implicit scalar draws) and keyword-only for every
# distribution except `random`, where it stays positional to mirror numpy's own order.
a = rng.random((4, 4))
print("*** rng.random((4, 4)) ***")
print(a[:])

b = rng.integers(0, 10, shape=(10,), dtype=np.int32)
print("\n*** rng.integers(0, 10, shape=(10,), dtype=np.int32) ***")
print(b[:])

# Reproducibility: same seed -> same array. Successive calls on the *same* generator
# draw different arrays (the "generator state advances" semantics numpy users expect).
c1 = blosc2.random.default_rng(7).normal(shape=(1000,))
c2 = blosc2.random.default_rng(7).normal(shape=(1000,))
same_seed = blosc2.random.default_rng(7)
d1 = same_seed.normal(shape=(1000,))
d2 = same_seed.normal(shape=(1000,))
print("\n*** Reproducibility ***")
print(f"Two fresh generators, same seed, first call:  equal = {np.array_equal(c1[:], d1[:])}")
print(f"One generator, two successive calls:          equal = {np.array_equal(d1[:], d2[:])}")

# Extra kwargs are forwarded to blosc2.empty(): chunks, cparams, urlpath, ...
e = rng.uniform(-1, 1, shape=(1_000, 1_000), chunks=(100, 1_000), cparams={"clevel": 5})
print(f"\n*** rng.uniform(..., chunks=(100, 1000), cparams=clevel5) -> cratio {e.schunk.cratio:.1f}x ***")

# Most of numpy.random.Generator's other distributions are covered too.
poisson = rng.poisson(4.0, shape=(10,))
gamma = rng.gamma(2.0, scale=1.5, shape=(10,))  # shape_param renamed, see docs
choice = rng.choice(["red", "green", "blue"], shape=(10,))
print("\n*** A few more distributions ***")
print("poisson:", poisson[:])
print("gamma:  ", gamma[:])
print("choice: ", choice[:])

# Vector-valued distributions draw a whole length-k vector per element: output shape
# is `shape + (k,)`, and that trailing dimension is always kept whole within a chunk.
dirichlet = rng.dirichlet([1.0, 2.0, 3.0], shape=(5,))
print("\n*** rng.dirichlet([1, 2, 3], shape=(5,)) -> shape", dirichlet.shape, "***")
print(dirichlet[:])
print("rows sum to 1:", np.allclose(dirichlet[:].sum(axis=-1), 1.0))

# Parallel generation vs. building the array via numpy + asarray().
N = 100_000_000
print(f"\n*** Timing: {N:_} float64 uniform draws ***")
t0 = time()
blosc2.asarray(np.random.default_rng(42).random(N), cparams={"clevel": 1})
t_numpy = time() - t0
print(f"asarray(np.random...):        {t_numpy:.3f} s")

t0 = time()
blosc2.random.default_rng(42).random((N,), cparams={"clevel": 1})
t_blosc2 = time() - t0
print(f"blosc2.random.default_rng...: {t_blosc2:.3f} s  ({t_numpy / t_blosc2:.1f}x faster)")
