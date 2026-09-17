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

# Initially attrs is empty
print(len(schunk.attrs))
# Add an attribute
schunk.attrs["meta1"] = "first metadata value"
print(schunk.attrs.getall())
# Update the attribute
schunk.attrs["meta1"] = "new metadata value"
print(schunk.attrs.getall())
# Add another attribute
schunk.attrs["meta2"] = "second metadata value"
# Check that it has been added
assert "meta2" in schunk.attrs

# Delete an attribute
del schunk.attrs["meta2"]
assert "meta2" not in schunk.attrs
