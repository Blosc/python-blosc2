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
companion donwload_data.py script.
"""

import copy
import os
from time import time

import rich

import blosc2

# The directory where the data is (see download_data.py)
dir_path = "era5-pds"
# put here your desired codec and clevel
codec = blosc2.Codec.BLOSCLZ
clevel = 9
# codec = blosc2.Codec.ZSTD
# clevel = 6

cparams = {}

cparams["shuffle"] = {
    "filters": [blosc2.Filter.SHUFFLE],
    "filters_meta": [0],
}

cparams["bitshuffle"] = {
    "filters": [blosc2.Filter.BITSHUFFLE],
    "filters_meta": [0],
}

# bytedelta seems to work best when one does not split
cparams["bytedelta"] = {
    "filters": [blosc2.Filter.SHUFFLE, blosc2.Filter.BYTEDELTA],
    "filters_meta": [0, 0],
    "splitmode": blosc2.SplitMode.NEVER_SPLIT,
}

if not os.path.isdir(dir_path):
    os.mkdir(dir_path)

for filter in cparams:
    cparams2 = copy.deepcopy(cparams[filter])
    cparams2["codec"] = codec
    cparams2["clevel"] = clevel
    rich.print("Using cparams: ", cparams2)
    for fname in os.listdir(dir_path):
        path = os.path.join(dir_path, fname)
        if not path.endswith(".b2nd"):
            continue
        path2 = path[: -len(".b2nd")] + f"-{filter}.b2nd_"
        fin = blosc2.open(path)

        print(f"Transcoding {path} (shape: {fin.shape}, dtype: {fin.dtype}) to {path2}")
        data = fin[:]
        t0 = time()
        fout = blosc2.empty(shape=fin.shape, dtype=fin.dtype, cparams=cparams2, urlpath=path2, mode="w")
        fout[:] = data
        tcomp = time() - t0
        t0 = time()
        data = fout[:]
        tdecomp = time() - t0
        # print(fout.info)
        print(
            f"  compr time: {tcomp:.2f}s ({fout.schunk.nbytes / (tcomp * 2**30):.3f} GB/s)"
            f"; decompr time: {tdecomp:.2f}s ({fout.schunk.nbytes / (tdecomp * 2**30):.3f} GB/s)"
            f" / cratio: {fout.schunk.cratio:.2f}x"
        )

print("All done!")
