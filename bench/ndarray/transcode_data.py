#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
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
from pathlib import Path
from time import time

import pandas as pd

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
# codecs = [(blosc2.Codec.LZ4, 9)]
# codecs = [(blosc2.Codec.BLOSCLZ, clevel) for clevel in (0, 1, 3, 6, 9)]
# codecs = [(codec, (9 if codec.value <= blosc2.Codec.LZ4.value else 6))
#           for codec in blosc2.Codec if codec.value <= blosc2.Codec.ZSTD.value]
codecs = [
    (codec, clevel)
    for codec in blosc2.Codec
    if codec.value <= blosc2.Codec.ZSTD.value
    for clevel in (0, 1, 3, 6, 9)
]

# measurements
meas = {
    "dset": [],
    "codec": [],
    "clevel": [],
    "filter": [],
    "cspeed": [],
    "dspeed": [],
    "cratio": [],
}

filters = {
    "nofilter": {
        "filters": [blosc2.Filter.NOFILTER],
    },
    "shuffle": {
        "filters": [blosc2.Filter.SHUFFLE],
    },
    "bitshuffle": {
        "filters": [blosc2.Filter.BITSHUFFLE],
    },
    "bytedelta": {
        "filters": [blosc2.Filter.SHUFFLE, blosc2.Filter.BYTEDELTA],
    },
}

dparams = {
    "nthreads": nthreads_decomp,
}

dir_path = Path(dir_path)
if not dir_path.is_dir():
    raise OSError(f"{dir_path} must be the directory with datasets")

for fname in dir_path.iterdir():
    path = str(fname)
    if not path.endswith(".b2nd"):
        continue
    finput = blosc2.open(path)
    # 64 KB is a good balance for both compression and decompression speeds
    mcpy = finput.copy(blocks=(16, 32, 32), dparams=dparams)  # copy in memory
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
        for filter in filters:
            cparams2 = copy.deepcopy(filters[filter])
            codec_, clevel = codec
            cparams2["codec"] = codec_
            cparams2["clevel"] = clevel
            cparams2["nthreads"] = nthreads_comp

            # Compression.  Do a copy and subtract the time for decompression.
            lt = []
            # Do not spend too much time performing costly compression settings
            nrep = 1 if codec_.value >= blosc2.Codec.LZ4HC.value and clevel == 9 else NREP
            for rep in range(nrep):
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
            cspeed = schunk.nbytes / (tcomp * 2**30)
            dspeed = schunk.nbytes / (tdecomp * 2**30)
            print(
                f"  Using {filter};\t compr time: {tcomp:.2f}s ({cspeed:.3f} GB/s)"
                f"; decompr time: {tdecomp:.2f}s ({dspeed:.3f} GB/s)"
                f" / cratio: {schunk.cratio:.2f} x"
            )

            # Fill measurements
            fname_ = fname.name
            dset = fname_[: fname_.find(".")]
            this_meas = {
                "dset": dset,
                "codec": codec[0].name,
                "clevel": codec[1],
                "filter": filter,
                "cspeed": cspeed,
                "dspeed": dspeed,
                "cratio": schunk.cratio,
            }
            for k, v in meas.items():
                v.append(this_meas[k])

            # Skip the other filters when no compression is going on
            if clevel == 0:
                break

meas_df = pd.DataFrame.from_dict(meas)
print("measurements:\n", meas_df)
fdest = dir_path / "measurements.parquet"
meas_df.to_parquet(fdest)
print("measurements stored at:", fdest)
print("All done!")
