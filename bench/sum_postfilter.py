import blosc2
import numpy as np
from time import time

nchunks = 1_000
# Set the compression and decompression parameters
cparams = {"codec": blosc2.Codec.LZ4, "typesize": 4, "nthreads": 1}
dparams = {"nthreads": 4}
contiguous = True
urlpath = "filename"

storage = {"contiguous": contiguous, "urlpath": urlpath, "cparams": cparams, "dparams": dparams}
blosc2.remove_urlpath(urlpath)

chunkshape = 500 * 1000
schunk = blosc2.SChunk(chunksize=chunkshape * 4, **storage)
data = np.arange(chunkshape, dtype=np.int32)

t0 = time()
for i in range(nchunks):
    schunk.append_data(data)
print(f"time append: {time() - t0:.2f} s")


@blosc2.postfilter(schunk, np.dtype(np.int32), np.dtype(np.int32))
def py_postfilter(input, output):
    output[:] = input + 1

t0 = time()
sum = 0
for chunk in schunk.iterchunks(np.int32):
    sum += chunk.sum()
print(f"time sum: {time() - t0:.2f} s")
print(sum)
