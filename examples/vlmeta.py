#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
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
print(len(schunk.attrs))
# Add a vlmeta
schunk.attrs["meta1"] = "first vlmetalayer"
print(schunk.attrs.getall())
# Update the vlmeta
schunk.attrs["meta1"] = "new vlmetalayer"
print(schunk.attrs.getall())
# Add another vlmeta
schunk.attrs["vlmeta2"] = "second vlmeta"
# Check that it has been added
assert "vlmeta2" in schunk.attrs

# Delete a vlmeta
del schunk.attrs["vlmeta2"]
assert "vlmeta2" not in schunk.attrs
