########################################################################
#
#       Created: April 30, 2021
#       Author:  The Blosc development team - blosc@blosc.org
#
########################################################################


"""
Small benchmark that compares a plain NumPy array copy against
compression through different compressors in blosc2.
"""

from __future__ import print_function

import time

import numpy as np

import blosc2

NREP = 4
N = int(1e8)
Nexp = np.log10(N)

blosc2.print_versions()

print("Creating NumPy arrays with 10**%d int64/float64 elements:" % Nexp)
arrays = (
    (np.arange(N, dtype=np.int64), "the arange linear distribution"),
    (np.linspace(0, 10_000, N), "the linspace linear distribution"),
    (np.random.randint(0, 10_000, N), "the random distribution"),
)

in_ = arrays[0][0]
# Cause a page fault here
out_ = np.full_like(in_, fill_value=0)
t0 = time.time()
for i in range(NREP):
    np.copyto(out_, in_)
tcpy = (time.time() - t0) / NREP
print("  *** np.copyto() *** Time for memcpy():\t%.3f s\t(%.2f GB/s)" % (tcpy, (N * 8 / tcpy) / 2 ** 30))

print("\nTimes for compressing/decompressing:")
for (in_, label) in arrays:
    print("\n*** %s ***" % label)
    for cname in blosc2.compressor_list():
        for filter in [blosc2.NOFILTER, blosc2.SHUFFLE, blosc2.BITSHUFFLE]:
            # clevel 9 is usually the best setting for fast compressors
            clevel = 9 if cname in ["lz4", "blosclz"] else 6
            t0 = time.time()
            c = blosc2.compress(in_, in_.itemsize, clevel=clevel, shuffle=filter, cname=cname)
            tc = time.time() - t0
            # Cause a page fault here
            out = np.full_like(in_, fill_value=0)
            t0 = time.time()
            for i in range(NREP):
                blosc2.decompress(c, dst=out)
            td = (time.time() - t0) / NREP
            assert np.array_equal(in_, out)
            filter_name = blosc2.filter_names[filter]
            print(
                "  *** %-7s, %-10s *** %6.3f s (%.2f GB/s) / %5.3f s (%.2f GB/s)"
                % (cname, filter_name, tc, ((N * 8 / tc) / 2 ** 30), td, ((N * 8 / td) / 2 ** 30)),
                end="",
            )
            print("\tcr: %5.1fx" % (N * 8.0 / len(c)))
