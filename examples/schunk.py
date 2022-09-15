########################################################################
#
#       Author:  The Blosc development team - blosc@blosc.org
#
########################################################################

import numpy

import blosc2

nchunks = 10
# Set the compression and decompression parameters
cparams = {"compcode": blosc2.Codec.LZ4HC, "typesize": 4}
dparams = {}
contiguous = True
urlpath = "filename"

storage = {"contiguous": contiguous, "urlpath": urlpath, "cparams": cparams, "dparams": dparams}
blosc2.remove_urlpath(urlpath)

# Create the empty SChunk
schunk = blosc2.SChunk(chunksize=200 * 1000 * 4, **storage)
# Append some chunks
for i in range(nchunks):
    buffer = i * numpy.arange(200 * 1000, dtype="int32")
    nchunks_ = schunk.append_data(buffer)
    assert nchunks_ == (i + 1)

# Decompress the second chunk in different ways
buffer = 1 * numpy.arange(200 * 1000, dtype="int32")
bytes_obj = buffer.tobytes()
res = schunk.decompress_chunk(1)
assert res == bytes_obj

dest = numpy.empty(buffer.shape, buffer.dtype)
schunk.decompress_chunk(1, dest)
assert numpy.array_equal(buffer, dest)

schunk.decompress_chunk(1, memoryview(dest))
assert numpy.array_equal(buffer, dest)

dest = bytearray(buffer)
schunk.decompress_chunk(1, dest)
assert dest == bytes_obj

# Insert a chunk in the 5th position
buffer = 10 * numpy.arange(200 * 1000, dtype="int32")
schunk.insert_data(5, buffer, False)

# Update a chunk compressing the data first
buffer = 11 * numpy.arange(200 * 1000, dtype="int32")
chunk = blosc2.compress2(buffer, **cparams)
schunk.update_chunk(7, chunk)

# Delete the 4th chunk
schunk.delete_chunk(4)

# Get the compressed chunk
schunk.get_chunk(1)

# Set a slice from the SChunk
start = 5 * 200 * 1000 + 47
stop = start + 200 * 1000 + 4
val = nchunks * numpy.arange(stop - start, dtype="int32")
schunk[start:stop] = val

# Get the modified slice
out = numpy.empty(val.shape, dtype="int32")
schunk.get_slice(start, stop, out)
assert numpy.array_equal(val, out)

# Expand the SChunk with __setitem__
# When a part of the slice section overflows the SChunk size, the remaining data is appended until stop is reached
start = nchunks * 200 * 1000 - 40
stop = start + 200 * 1000
val = nchunks * numpy.arange(stop - start, dtype="int32")
schunk[start:stop] = val

blosc2.remove_urlpath(urlpath)
