#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Tip 11: generate arrays with DSL kernels. A stateless counter-based
# pseudo-random generator (value = hash(flat index)) written as a
# @blosc2.dsl_kernel fills the NDArray chunk by chunk, in parallel,
# instead of materializing the full array with np.random first and
# compressing it via asarray().

import numpy as np

import blosc2
from common import fmt_bytes, measure, save_plot

N = 200_000_000  # 200M int32 = ~0.75 GiB as a plain NumPy array

# Random data is incompressible: skip the codec entirely.
CPARAMS = {"clevel": 0}


# Classic 2-round xorshift-multiply integer hash. Two identical mixing
# rounds are needed: a single multiply is nearly linear in the index,
# leaving consecutive values strongly correlated (lag-1 ~0.75). The masks
# emulate uint32 truncation and keep every product below 2^63 (0x45d9f3b
# < 2^27, so a 32-bit state times it stays < 2^59, exact in int64
# arithmetic). The final 32-bit value wraps two's-complement into the
# int32 output dtype, covering the full [-2^31, 2^31) range.
#
# The integer output dtype matters: an integer output makes the DSL
# evaluate the whole kernel in exact int64 arithmetic, while a float
# output dtype would compute everything in float64, where integer
# operations are only exact below 2^53. See the DSL syntax reference
# (doc/reference/dsl_syntax.md) for the full rules.
#
# One statistical quirk to be aware of: the hash is near-bijective in the
# index, so it samples without replacement — expect ~0 duplicates (a true
# random sample of 1M draws expects ~116) and a 256-bin chi-square a bit
# below the i.i.d. 255+-23 band (bins come out "too even").
@blosc2.dsl_kernel
def random_int32(seed):
    x = _flat_idx ^ seed  # noqa: F821
    x = (((x >> 16) ^ x) * 0x45D9F3B) & 0xFFFFFFFF
    x = (((x >> 16) ^ x) * 0x45D9F3B) & 0xFFFFFFFF
    return (x >> 16) ^ x


def naive():
    rng = np.random.default_rng(42)
    return blosc2.asarray(rng.integers(-(2**31), 2**31, size=N, dtype=np.int32), cparams=CPARAMS)


def tip():
    lazy = blosc2.lazyudf(random_int32, (42,), dtype=np.int32, shape=(N,))
    return lazy.compute(cparams=CPARAMS)


def quality_stats(v, label, n_dups=1_000_000):
    """Light uniformity checks over a sample; quoted in the doc tip."""
    f = v.astype(np.float64)
    h = np.histogram(v, bins=256, range=(-(2**31), 2**31))[0]
    e = len(v) / 256
    chi2 = ((h - e) ** 2 / e).sum()
    lag1 = np.corrcoef(f[:-1], f[1:])[0, 1]
    dups = n_dups - len(np.unique(v[:n_dups]))
    print(
        f"{label:22s} mean={f.mean():12.1f}  std/2^31={f.std() / 2**31:.4f}  "
        f"chi2(255 dof)={chi2:6.1f}  lag1={lag1: .2e}  dups/1M={dups}"
    )


if __name__ == "__main__":
    naive_t, naive_m = measure(__file__, "naive")
    tip_t, tip_m = measure(__file__, "tip")

    print(f"naive  asarray(rng.integers(N={N:,})): {naive_t:.3f}s  peak {fmt_bytes(naive_m)}")
    print(f"tip    lazyudf(random_int32, N={N:,}): {tip_t:.3f}s  peak {fmt_bytes(tip_m)}")
    print(f"speedup: {naive_t / tip_t:.1f}x   memory: {naive_m / tip_m:.1f}x less")

    n_check = 20_000_000
    print(f"\nquality (light checks, {n_check // 10**6}M samples; expect mean~0, "
          f"std/2^31~0.5774, chi2~255+-23, lag1~0, dups~116):")
    lazy = blosc2.lazyudf(random_int32, (42,), dtype=np.int32, shape=(n_check,))
    quality_stats(lazy.compute(cparams=CPARAMS)[:], "dsl random_int32")
    rng = np.random.default_rng(42)
    quality_stats(rng.integers(-(2**31), 2**31, size=n_check, dtype=np.int32), "np.random (PCG64)")

    save_plot(
        "tip_11_dsl_random.png",
        "DSL random kernel vs asarray(rng.integers()) — 200M int32 elements",
        "asarray(rng.integers)",
        "lazyudf(random_int32)",
        naive_t,
        tip_t,
        naive_m,
        tip_m,
    )
