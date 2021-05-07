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
import numpy as np
import time
import blosc2

N = 1e8
clevel = 5
Nexp = np.log10(N)

blosc2.print_versions()
print("Creating a large NumPy array with 10**%d int64 elements:" % Nexp)
in_ = np.arange(N, dtype=np.int64)  # the trivial linear distribution
# in_ = np.linspace(0, 100, N)  # another linear distribution
# in_ = np.random.random_integers(0, 100, N)  # random distribution
print(" ", in_)

tic = time.time()
out_ = np.copy(in_)
toc = time.time()
print("  Time for copying array with np.copy():                    %.3f s" % (toc-tic,))

out_ = np.empty_like(in_)
tic = time.time()
np.copyto(out_, in_)
toc = time.time()
print("  Time for copying array with np.copyto and empty_like:     %.3f s" % (toc-tic,))

# Unlike numpy.zeros, numpy.zeros_like doens't use calloc, but instead uses
# empty_like and explicitely assigns zeros, which is basically like calling
# full like
# Here we benchmark what happens when we allocate memory using calloc
out_ = np.zeros(in_.shape, dtype=in_.dtype)
tic = time.time()
np.copyto(out_, in_)
toc = time.time()
print("  Time for copying array with np.copyto and zeros     :     %.3f s" % (toc-tic,))

# Cause a page faults before the benchmark
out_ = np.full_like(in_, fill_value=0)
tic = time.time()
np.copyto(out_, in_)
toc = time.time()
print("  Time for copying array with np.copyto and full_like:      %.3f s" % (toc-tic,))

out_ = np.full_like(in_, fill_value=0)
tic = time.time()
out_[...] = in_
toc = time.time()
print("  Time for copying array with numpy assignment:             %.3f s" % (toc-tic,))
print()

for cname in blosc2.compressor_list():
    print("Using *** %s *** compressor::" % cname)
    ctic = time.time()
    c = blosc2.pack_array(in_, clevel=clevel, shuffle=True, cname=cname)
    ctoc = time.time()
    dtic = time.time()
    out = blosc2.unpack_array(c)
    dtoc = time.time()

    assert (np.array_equal(in_, out))
    print("  Time for pack_array/unpack_array:     %.3f/%.3f s." %
          (ctoc-ctic, dtoc-dtic), end='')
    print("\tCompr ratio: %.2f" % (in_.size*in_.dtype.itemsize*1. / len(c)))

    ctic = time.time()
    c = blosc2.compress(in_, clevel=clevel, shuffle=True, cname=cname)
    ctoc = time.time()
    out = np.empty(in_.shape, in_.dtype)
    dtic = time.time()
    blosc2.decompress(c, dst=out)
    dtoc = time.time()

    assert(np.array_equal(in_, out))
    print("  Time for compress/decompress:         %.3f/%.3f s." %
          (ctoc-ctic, dtoc-dtic), end='')
    print("\tCompr ratio: %.2f" % (in_.size*in_.dtype.itemsize*1. / len(c)))
