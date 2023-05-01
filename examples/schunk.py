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
numpy_meta = {b"dtype": str(np.dtype("int32"))}
test_meta = {b"lorem": 1234}
meta = {"numpy": numpy_meta, "test": test_meta}

# Create the empty SChunk
schunk = blosc2.SChunk(chunksize=200 * 1000 * 4, meta=meta, **storage)
# Append some chunks
for i in range(nchunks):
    buffer = i * np.arange(200 * 1000, dtype="int32")
    nchunks_ = schunk.append_data(buffer)
    assert nchunks_ == (i + 1)

# Decompress the second chunk in different ways
buffer = 1 * np.arange(200 * 1000, dtype="int32")
bytes_obj = buffer.tobytes()
res = schunk.decompress_chunk(1)
assert res == bytes_obj

dest = np.empty(buffer.shape, buffer.dtype)
schunk.decompress_chunk(1, dest)
assert np.array_equal(buffer, dest)

schunk.decompress_chunk(1, memoryview(dest))
assert np.array_equal(buffer, dest)

dest = bytearray(buffer)
schunk.decompress_chunk(1, dest)
assert dest == bytes_obj

# Insert a chunk in the 5th position
buffer = 10 * np.arange(200 * 1000, dtype="int32")
schunk.insert_data(5, buffer, False)

# Update a chunk compressing the data first
buffer = 11 * np.arange(200 * 1000, dtype="int32")
chunk = blosc2.compress2(buffer, **cparams)
schunk.update_chunk(7, chunk)

# Delete the 4th chunk
schunk.delete_chunk(4)

# Get the compressed chunk
schunk.get_chunk(1)

# Set a slice from the SChunk
start = 5 * 200 * 1000 + 47
stop = start + 200 * 1000 + 4
val = nchunks * np.arange(stop - start, dtype="int32")
schunk[start:stop] = val

# Get the modified slice
out = np.empty(val.shape, dtype="int32")
schunk.get_slice(start, stop, out)
assert np.array_equal(val, out)

# Expand the SChunk with __setitem__
# When a part of the slice section overflows the SChunk size,
# the remaining data is appended until stop is reached
start = nchunks * 200 * 1000 - 40
stop = start + 200 * 1000
val = nchunks * np.arange(stop - start, dtype="int32")
schunk[start:stop] = val

blosc2.remove_urlpath(urlpath)
