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

# Create the SChunk
data = numpy.arange(200 * 1000 * nchunks)
schunk = blosc2.SChunk(chunksize=200 * 1000 * 4, data=data, **storage)

cframe = schunk.to_cframe()

schunk2 = blosc2.schunk_from_cframe(cframe, False)
data2 = numpy.empty(data.shape, dtype=data.dtype)
schunk2.get_slice(out=data2)
assert numpy.array_equal(data, data2)

blosc2.remove_urlpath(urlpath)
