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
schunk = blosc2.SChunk(chunksize=200 * 1000 * 4)
for i in range(nchunks):
    buffer = i * np.arange(200 * 1000, dtype="int32")
    nchunks_ = schunk.append_data(buffer)
    assert nchunks_ == (i + 1)

# Initially the vlmeta is empty
print(len(schunk.vlmeta))
# Add a vlmeta
schunk.vlmeta["meta1"] = "first vlmetalayer"
print(schunk.vlmeta.getall())
# Update the vlmeta
schunk.vlmeta["meta1"] = "new vlmetalayer"
print(schunk.vlmeta.getall())
# Add another vlmeta
schunk.vlmeta["vlmeta2"] = "second vlmeta"
# Check that it has been added
assert "vlmeta2" in schunk.vlmeta

# Delete a vlmeta
del schunk.vlmeta["vlmeta2"]
assert "vlmeta2" not in schunk.vlmeta
