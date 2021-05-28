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
tic = time.time()
for i in range(NREP):
    out_ = np.copy(in_)
toc = time.time()
tcpy = (toc - tic) / NREP
print("  Time for copying array with np.copy:                  %.3f s (%.2f GB/s))" %
      (tcpy, ((N * 8 / tcpy) / 2 ** 30)))

out_ = np.empty_like(in_)
tic = time.time()
for i in range(NREP):
    np.copyto(out_, in_)
toc = time.time()
tcpy = (toc - tic) / NREP
print("  Time for copying array with np.copyto and empty_like: %.3f s (%.2f GB/s))" %
      (tcpy, ((N * 8 / tcpy) / 2 ** 30)))

# Unlike numpy.zeros, numpy.zeros_like doens't use calloc, but instead uses
# empty_like and explicitely assigns zeros, which is basically like calling
# full like
# Here we benchmark what happens when we allocate memory using calloc
out_ = np.zeros(in_.shape, dtype=in_.dtype)
tic = time.time()
for i in range(NREP):
    np.copyto(out_, in_)
toc = time.time()
tcpy = (toc - tic) / NREP
print("  Time for copying array with np.copyto and zeros:      %.3f s (%.2f GB/s))" %
      (tcpy, ((N * 8 / tcpy) / 2 ** 30)))

# Cause a page fault before the benchmark
out_ = np.full_like(in_, fill_value=0)
tic = time.time()
for i in range(NREP):
    np.copyto(out_, in_)
toc = time.time()
tcpy = (toc - tic) / NREP
print("  Time for copying array with np.copyto and full_like:  %.3f s (%.2f GB/s))" %
      (tcpy, ((N * 8 / tcpy) / 2 ** 30)))

out_ = np.full_like(in_, fill_value=0)
tic = time.time()
for i in range(NREP):
    out_[...] = in_
toc = time.time()
tcpy = (toc - tic) / NREP
print("  Time for copying array with numpy assignment:         %.3f s (%.2f GB/s))" %
      (tcpy, ((N * 8 / tcpy) / 2 ** 30)))
print()

for (in_, label) in arrays:
    print("\n*** %s ***" % label)
    for cname in blosc2.compressor_list():
        print("Using *** %s *** compressor:" % cname)
        # clevel 9 is usually the best setting for fast compressors
        clevel = 9 if cname in ["lz4", "blosclz"] else 6

        ctic = time.time()
        for i in range(NREP):
            c = blosc2.pack_array(in_, clevel=clevel, shuffle=True, cname=cname)
        ctoc = time.time()
        dtic = time.time()
        for i in range(NREP):
            out = blosc2.unpack_array(c)
        dtoc = time.time()

        assert np.array_equal(in_, out)
        tc = (ctoc - ctic) / NREP
        td = (dtoc - dtic) / NREP
        print("  Time for pack_array/unpack_array:     %.3f/%.3f s (%.2f/%.2f GB/s)) " %
              (tc, td, ((N * 8 / tc) / 2 ** 30), ((N * 8 / td) / 2 ** 30)), end="")
        print("\tcr: %5.1fx" % (in_.size * in_.dtype.itemsize * 1.0 / len(c)))

        ctic = time.time()
        for i in range(NREP):
            c = blosc2.compress(in_, clevel=clevel, shuffle=True, cname=cname)
        ctoc = time.time()
        out = np.full_like(in_, fill_value=0)
        dtic = time.time()
        for i in range(NREP):
            blosc2.decompress(c, dst=out)
        dtoc = time.time()

        assert np.array_equal(in_, out)
        tc = (ctoc - ctic) / NREP
        td = (dtoc - dtic) / NREP
        print("  Time for compress/decompress:         %.3f/%.3f s (%.2f/%.2f GB/s)) " %
              (tc, td, ((N * 8 / tc) / 2 ** 30), ((N * 8 / td) / 2 ** 30)), end="")
        print("\tcr: %5.1fx" % (in_.size * in_.dtype.itemsize * 1.0 / len(c)))
