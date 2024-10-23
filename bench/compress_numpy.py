#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################


"""
Small benchmark that compares a plain NumPy array copy against
compression through different compressors in blosc2.
"""

import time

import blosc2
import numpy as np

NREP = 4
N = int(1e8)
Nexp = np.log10(N)

blosc2.print_versions()

print(f"Creating NumPy arrays with 10**{Nexp} int64/float64 elements:")
arrays = (
    (np.arange(N, dtype=np.int64), "the arange linear distribution"),
    (np.linspace(0, 10_000, N), "the linspace linear distribution"),
    (np.random.randint(0, 10_000, N), "the random distribution"),  # noqa: NPY002
)

in_ = arrays[0][0]
# Cause a page fault here
out_ = np.full_like(in_, fill_value=0)
t0 = time.time()
for _i in range(NREP):
    np.copyto(out_, in_)
tcpy = (time.time() - t0) / NREP
print(
    f"  *** np.copyto() *** Time for memcpy():\t{tcpy:.3f} s\t({(N * 8 / tcpy) / 2**30:.2f} GB/s)"
)

print("\nTimes for compressing/decompressing:")
for in_, label in arrays:
    print(f"\n*** {label} ***")
    for codec in blosc2.compressor_list():
        for filter in (
            blosc2.Filter.NOFILTER,
            blosc2.Filter.SHUFFLE,
            blosc2.Filter.BITSHUFFLE,
        ):
            clevel = 6
            t0 = time.time()
            c = blosc2.compress(in_, in_.itemsize, clevel=clevel, filter=filter, codec=codec)
            tc = time.time() - t0
            # Cause a page fault here
            out = np.full_like(in_, fill_value=0)
            t0 = time.time()
            for _i in range(NREP):
                blosc2.decompress(c, dst=out)
            td = (time.time() - t0) / NREP
            assert np.array_equal(in_, out)
            print(
                f"  *** {codec:15s}, {filter:20s} *** {tc:6.3f} s ({(N * 8 / tc) / 2**30:.2f} GB/s) / {td:5.3f} s ({(N * 8 / td) / 2**30:.2f} GB/s)",
                end="",
            )
            print(f"\tcr: {N * 8.0 / len(c):5.1f}x")
