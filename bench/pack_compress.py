#######################################################################
# Copyright (C) 2019-present, Blosc Development team <blosc@blosc.org>
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

NREP = 3
N = int(2e8)
Nexp = np.log10(N)

comprehensive_copy_timing = False

blosc2.print_versions()
print(f"Creating NumPy arrays with 10 ** {Nexp:.2f} int64/float64 elements:")
arrays = (
    (np.arange(N), "the arange linear distribution"),
    (np.linspace(0, 10_000, N), "the linspace linear distribution"),
    (np.random.randint(0, 10_000, N), "the random distribution"),
)

in_ = arrays[0][0]
tic = time.time()
for i in range(NREP):
    out_ = np.copy(in_)
toc = time.time()
tcpy = (toc - tic) / NREP
print(
    "  Time for copying array with np.copy:                  %.3f s (%.2f GB/s))"
    % (tcpy, ((N * 8 / tcpy) / 2 ** 30))
)

if comprehensive_copy_timing:
    tic = time.time()
    out_ = np.empty_like(in_)
    for i in range(NREP):
        np.copyto(out_, in_)
    toc = time.time()
    tcpy = (toc - tic) / NREP
    print(
        "  Time for copying array with np.copyto and empty_like: %.3f s (%.2f GB/s))"
        % (tcpy, ((N * 8 / tcpy) / 2 ** 30))
    )

    # Unlike numpy.zeros, numpy.zeros_like doesn't use calloc, but instead uses
    # empty_like and explicitly assigns zeros, which is basically like calling
    # full_like
    # Here we benchmark what happens when we allocate memory using calloc
    tic = time.time()
    out_ = np.zeros(in_.shape, dtype=in_.dtype)
    for i in range(NREP):
        np.copyto(out_, in_)
    toc = time.time()
    tcpy = (toc - tic) / NREP
    print(
        "  Time for copying array with np.copyto and zeros:      %.3f s (%.2f GB/s))"
        % (tcpy, ((N * 8 / tcpy) / 2 ** 30))
    )

    # Cause a page fault before the benchmark
    tic = time.time()
    out_ = np.full_like(in_, fill_value=0)
    for i in range(NREP):
        np.copyto(out_, in_)
    toc = time.time()
    tcpy = (toc - tic) / NREP
    print(
        "  Time for copying array with np.copyto and full_like:  %.3f s (%.2f GB/s))"
        % (tcpy, ((N * 8 / tcpy) / 2 ** 30))
    )

    tic = time.time()
    out_ = np.full_like(in_, fill_value=0)
    for i in range(NREP):
        out_[...] = in_
    toc = time.time()
    tcpy = (toc - tic) / NREP
    print(
        "  Time for copying array with numpy assignment:         %.3f s (%.2f GB/s))"
        % (tcpy, ((N * 8 / tcpy) / 2 ** 30))
    )

print()

for (in_, label) in arrays:
    print("\n*** %s ***" % label)
    for codec in blosc2.Codec:
        print("Using *** %s *** compressor:" % codec)
        clevel = 6
        cparams = {"codec": codec, "clevel": clevel}

        ctic = time.time()
        for i in range(NREP):
            c = blosc2.compress(in_, clevel=clevel, codec=codec)
        ctoc = time.time()
        dtic = time.time()
        out = np.empty_like(in_)
        for i in range(NREP):
            blosc2.decompress(c, dst=out)
        dtoc = time.time()

        assert np.array_equal(in_, out)
        tc = (ctoc - ctic) / NREP
        td = (dtoc - dtic) / NREP
        print(
            "  Time for compress/decompress:         %.3f/%.3f s (%.2f/%.2f GB/s)) "
            % (tc, td, ((N * 8 / tc) / 2 ** 30), ((N * 8 / td) / 2 ** 30)),
            end="",
        )
        print("\tcr: %5.1fx" % (in_.size * in_.dtype.itemsize * 1.0 / len(c)))

        ctic = time.time()
        for i in range(NREP):
            c = blosc2.pack_array(in_, clevel=clevel, codec=codec)
        ctoc = time.time()
        dtic = time.time()
        for i in range(NREP):
            out = blosc2.unpack_array(c)
        dtoc = time.time()

        assert np.array_equal(in_, out)
        tc = (ctoc - ctic) / NREP
        td = (dtoc - dtic) / NREP
        print(
            "  Time for pack_array/unpack_array:     %.3f/%.3f s (%.2f/%.2f GB/s)) "
            % (tc, td, ((N * 8 / tc) / 2 ** 30), ((N * 8 / td) / 2 ** 30)),
            end="",
        )
        print("\tcr: %5.1fx" % (in_.size * in_.dtype.itemsize * 1.0 / len(c)))

        ctic = time.time()
        for i in range(NREP):
            c = blosc2.pack_tensor(in_, cparams=cparams)
        ctoc = time.time()
        dtic = time.time()
        for i in range(NREP):
            out = blosc2.unpack_tensor(c)
        dtoc = time.time()

        assert np.array_equal(in_, out)
        tc = (ctoc - ctic) / NREP
        td = (dtoc - dtic) / NREP
        print(
            "  Time for pack_tensor/unpack_tensor:   %.3f/%.3f s (%.2f/%.2f GB/s)) "
            % (tc, td, ((N * 8 / tc) / 2 ** 30), ((N * 8 / td) / 2 ** 30)),
            end="",
        )
        print("\tcr: %5.1fx" % (in_.size * in_.dtype.itemsize * 1.0 / len(c)))
