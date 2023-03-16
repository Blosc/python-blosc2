#######################################################################
# Copyright (C) 2019-present, Blosc Development team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################


"""
Benchmark that compares compressing real data copy using different filters
and codecs in Blosc2.  You need to download the data first by using the
companion download_data.py script.
"""

import copy
import os
from time import time

import blosc2

# Number of repetitions for each time measurement.  The minimum will be taken.
NREP = 3
# The directory where the data is (see download_data.py)
dir_path = "era5-pds"

# The threads for compression / decompression
# For some reason, decompression benefits more from using more threads
nthreads_comp = blosc2.nthreads  # 24
nthreads_decomp = blosc2.nthreads  # 32

# put here your desired codec and clevel
codecs = [(blosc2.Codec.LZ4, 9), (blosc2.Codec.BLOSCLZ, 9)]
# codecs = [(blosc2.Codec.BLOSCLZ, 9)]
# codecs = [(codec, (9 if codec.value <= blosc2.Codec.LZ4.value else 6))
#          for codec in blosc2.Codec if codec.value <= blosc2.Codec.ZSTD.value]

cparams = {}
cparams["nofilter"] = {
    "filters": [blosc2.Filter.NOFILTER],
    "filters_meta": [0],
    "nthreads": nthreads_comp,
}
cparams["shuffle"] = {
    "filters": [blosc2.Filter.SHUFFLE],
    "filters_meta": [0],
    "nthreads": nthreads_comp,
}
cparams["bitshuffle"] = {
    "filters": [blosc2.Filter.BITSHUFFLE],
    "filters_meta": [0],
    "nthreads": nthreads_comp,
}
cparams["bytedelta"] = {
    "filters": [blosc2.Filter.SHUFFLE, blosc2.Filter.BYTEDELTA],
    "filters_meta": [0, 0],
    "nthreads": nthreads_comp,
}

dparams = {
    "nthreads": nthreads_decomp,
}

if not os.path.isdir(dir_path):
    os.mkdir(dir_path)

for fname in os.listdir(dir_path):
    path = os.path.join(dir_path, fname)
    if not path.endswith(".b2nd"):
        continue
    finput = blosc2.open(path)
    mcpy = finput.copy(dparams=dparams)  # copy in memory
    # Compute decompression time for subtracting from copy later
    lt = []
    for rep in range(NREP):
        t0 = time()
        for chunk in mcpy.schunk.iterchunks(dtype=mcpy.dtype):
            pass
        lt.append(time() - t0)
    tdecomp0 = min(lt)
    print(f"Transcoding {path} (shape: {mcpy.shape}, dtype: {mcpy.dtype})")
    for codec in codecs:
        print("Using codec: ", codec)
        for filter in cparams:
            cparams2 = copy.deepcopy(cparams[filter])
            cparams2["codec"] = codec[0]
            cparams2["clevel"] = codec[1]

            # Compression.  Do a copy and subtract the time for decompression.
            lt = []
            for rep in range(NREP):
                t0 = time()
                fout = mcpy.copy(cparams=cparams2, dparams=dparams)
                lt.append(time() - t0)
            tcomp = min(lt) - tdecomp0
            schunk = fout.schunk

            # Decompression
            lt = []
            for rep in range(NREP):
                t0 = time()
                for chunk in schunk.iterchunks(dtype=mcpy.dtype):
                    pass
                lt.append(time() - t0)
            tdecomp = min(lt)
            print(
                f"  Using {filter};\t compr time: {tcomp:.2f}s ({schunk.nbytes / (tcomp * 2**30):.3f} GB/s)"
                f"; decompr time: {tdecomp:.2f}s ({schunk.nbytes / (tdecomp * 2**30):.3f} GB/s)"
                f" / cratio: {schunk.cratio:.2f} x"
            )

print("All done!")
