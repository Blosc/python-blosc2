#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

import numpy as np

import blosc2

nchunks = 10
# Set the compression and decompression parameters
cparams = {"codec": blosc2.Codec.LZ4HC, "typesize": 4}
dparams = {}
contiguous = True
urlpath = "filename"

storage = {"contiguous": contiguous, "urlpath": urlpath, "cparams": cparams, "dparams": dparams}
blosc2.remove_urlpath(urlpath)

# Create the SChunk
data = np.arange(200 * 1000 * nchunks)
schunk = blosc2.SChunk(chunksize=200 * 1000 * 4, data=data, **storage)

cframe = schunk.to_cframe()

schunk2 = blosc2.schunk_from_cframe(cframe, False)
data2 = np.empty(data.shape, dtype=data.dtype)
schunk2.get_slice(out=data2)
assert np.array_equal(data, data2)

blosc2.remove_urlpath(urlpath)
