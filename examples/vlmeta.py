########################################################################
#
#       Created: July 28, 2021
#       Author:  The Blosc development team - blosc@blosc.org
#
########################################################################

import numpy

import blosc2

nchunks = 10
schunk = blosc2.SChunk(chunksize=200 * 1000 * 4)
for i in range(nchunks):
    buffer = i * numpy.arange(200 * 1000, dtype="int32")
    nchunks_ = schunk.append_data(buffer)
    assert nchunks_ == (i + 1)

# Initially the vlmeta is empty
print(schunk.vlmeta.vlmeta)
# Add a vlmeta
schunk.vlmeta["meta1"] = "first vlmetalayer"
print(schunk.vlmeta.vlmeta)
# Update the vlmeta
schunk.vlmeta["meta1"] = "new vlmetalayer"
print(schunk.vlmeta.vlmeta)
# Add another vlmeta
schunk.vlmeta["vlmeta2"] = "second vlmeta"
# Check that it has been added
assert "vlmeta2" in schunk.vlmeta

# Delete a vlmeta
del schunk.vlmeta['vlmeta2']
assert 'vlmeta2' not in schunk.vlmeta
