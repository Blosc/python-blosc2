#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#
# This example can only be run if blosc2-btune is installed. You can
# get it from https://pypi.org/project/blosc2-btune/
# For more info on this tuner plugin see
# https://github.com/Blosc/blosc2_btune/blob/main/README.md
#######################################################################

import blosc2_btune
import numpy as np

import blosc2

nchunks = 10
# Set the compression and decompression parameters, use BTUNE tuner
cparams = {"codec": blosc2.Codec.LZ4HC, "typesize": 4, "tuner": blosc2.Tuner.BTUNE}
dparams = {}
contiguous = True
urlpath = "filename"

storage = {"contiguous": contiguous, "urlpath": urlpath, "cparams": cparams, "dparams": dparams}
blosc2.remove_urlpath(urlpath)

# Set the Btune configuration to use
btune_conf = {"tradeoff": 0.3, "perf_mode": blosc2_btune.PerformanceMode.DECOMP}
blosc2_btune.set_params_defaults(**btune_conf)

# Create the SChunk
data = np.arange(200 * 1000 * nchunks)
schunk = blosc2.SChunk(chunksize=200 * 1000 * 4, data=data, **storage)

# Check data can be retrieved correctly
data2 = np.empty(data.shape, dtype=data.dtype)
schunk.get_slice(out=data2)
assert np.array_equal(data, data2)

blosc2.remove_urlpath(urlpath)
