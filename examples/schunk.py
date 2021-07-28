########################################################################
#
#       Created: July 28, 2021
#       Author:  The Blosc development team - blosc@blosc.org
#
########################################################################

import blosc2
import numpy

nchunks = 10
# Set the compression and decompression parameters
cparams = {"compcode": blosc2.LZ4HC, "typesize": 4}
dparams = {}
contiguous = True
urlpath = "filename"

storage = {"contiguous": contiguous, "urlpath": urlpath, "cparams": cparams, "dparams": dparams}

# Create the empty SChunk
schunk = blosc2.SChunk(chunksize=200 * 1000 * 4, **storage)
# Append some chunks
for i in range(nchunks):
    buffer = i * numpy.arange(200 * 1000, dtype="int32")
    nchunks_ = schunk.append_data(buffer)
    assert nchunks_ == (i + 1)

# Decompress second the chunk in different ways
buffer = 1 * numpy.arange(200 * 1000)
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

# Get the compressed chunk
schunk.get_chunk(1)

blosc2.remove_urlpath(urlpath)