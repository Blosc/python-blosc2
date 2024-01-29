##############################################################################
# blosc2_grok: Grok (JPEG2000 codec) plugin for Blosc2
#
# Copyright (c) 2023  The Blosc Development Team <blosc@blosc.org>
# https://blosc.org
# License: GNU Affero General Public License v3.0 (see LICENSE.txt)
##############################################################################

"""
Benchmark for compressing a dataset of images with the grok codec and store it into Blosc2.

Data can be downloaded from: http://www.silx.org/pub/nabu/data/compression/lung_raw_2000-2100.h5
"""

from time import time

import blosc2_grok
import h5py
import numpy as np
from skimage.metrics import structural_similarity as ssim

import blosc2

IMAGES_PER_CHUNK = 64

if __name__ == "__main__":
    # Define the compression and decompression parameters. Disable the filters and the
    # splitmode, because these don't work with the codec.
    cparams = {
        "codec": blosc2.Codec.GROK,
        "nthreads": 8,
        "filters": [],
        "splitmode": blosc2.SplitMode.NEVER_SPLIT,
    }

    # Open the dataset
    f = h5py.File("/Users/faltet/Downloads/lung_raw_2000-2100.h5", "r")
    dset = f["/data"]
    print(f"Compressing dataset of {dset.shape} images ...")

    # for cratio in range(1, 11):
    # for cratio in [4, 6, 8, 10, 20]:
    for cratio in [7]:
        # Set the parameters that will be used by grok
        kwargs = {
            "cod_format": blosc2_grok.GrkFileFmt.GRK_FMT_JP2,
            "quality_mode": "rates",
            "quality_layers": np.array([cratio], dtype=np.float64),
        }
        blosc2_grok.set_params_defaults(**kwargs)

        # Open the output file
        chunks = (IMAGES_PER_CHUNK,) + dset.shape[1:]  # IMAGES_PER_CHUNK images per chunk
        blocks = (1,) + dset.shape[1:]
        # blocks = (IMAGES_PER_CHUNK,) + (dset.shape[1] // 8, dset.shape[2] // 8)
        # chunks = (100,) + dset.shape[1:]   # IMAGES_PER_CHUNK images per chunk
        # blocks = (100,) + (dset.shape[1] // 10, dset.shape[2] // 10)
        # chunks = (IMAGES_PER_CHUNK,) + dset.shape[1:]   # IMAGES_PER_CHUNK images per chunk
        # blocks = (IMAGES_PER_CHUNK,) + (dset.shape[1] // 8, dset.shape[2] // 8)
        cparams2 = blosc2.cparams_dflts.copy()
        # cparams2['splitmode'] = blosc2.SplitMode.NEVER_SPLIT
        cparams2["codec"] = blosc2.Codec.ZSTD
        # cparams2['typesize'] = 2
        # cparams2['codec'] = blosc2.Codec.LZ4
        cparams2["clevel"] = 6
        cparams2["filters"] = [blosc2.Filter.INT_TRUNC, blosc2.Filter.TRANSPOSE, blosc2.Filter.SHUFFLE]
        cparams2["filters_meta"] = [8, 0, 1]
        cparams2["use_dict"] = 0
        fout = blosc2.uninit(
            dset.shape,
            dset.dtype,
            chunks=chunks,
            blocks=blocks,
            cparams=cparams2,
            urlpath=f"/Users/faltet/Downloads/lung_zstd_2000-2100-{cratio}x.b2nd",
            mode="w",
        )

        # Do the actual transcoding
        # for i in tqdm(range(0, dset.shape[0], IMAGES_PER_CHUNK)):
        for i in range(0, dset.shape[0], IMAGES_PER_CHUNK):
            im = dset[i : i + IMAGES_PER_CHUNK, ...]
            # Transform the numpy array to a blosc2 array. This is where compression happens.
            t0 = time()
            fout[i : i + IMAGES_PER_CHUNK] = im
            time_ = time() - t0
            print(f"time: {time_}")
            for j in range(IMAGES_PER_CHUNK):
                if i + j >= dset.shape[0]:
                    break
                im1 = im[j]
                im2 = fout[i + j]
                # Compare lossy image with original
                ssim_ = ssim(im1[0], im2[0], data_range=im1.max() - im1.min())
                print(f"SSIM: {ssim_}")
        print(f"Compression with cratio={fout.schunk.cratio:.2f}x ...")

    f.close()
