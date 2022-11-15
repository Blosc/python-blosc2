from time import time

import blosc2
import numpy as np


# Size and dtype of super-chunks
nchunks = 20_000
chunkshape = 50_000
dtype = np.dtype(np.int32)
chunksize = chunkshape * dtype.itemsize

# Set the compression and decompression parameters
cparams = {"clevel": 9, "codec": blosc2.Codec.BLOSCLZ, "typesize": 4, "nthreads": 1}
dparams = {"nthreads": 1}
storage = {"cparams": cparams, "dparams": dparams}

# Create super-chunks
schunk0 = blosc2.SChunk(chunksize=chunksize, **storage)
schunk = blosc2.SChunk(chunksize=chunksize, **storage)

data = np.arange(chunkshape, dtype=dtype)
t0 = time()
for i in range(nchunks):
    schunk.append_data(data)
    schunk0.append_data(data)
print(f"time append: {time() - t0:.2f}s")
# print(f"cratio: {schunk.nbytes / schunk.cbytes:.2f}x")

# Associate a postfilter to schunk
@blosc2.postfilter(schunk, np.dtype(dtype))
def py_postfilter(input, output):
    output[:] = input + 1


t0 = time()
sum = 0
for chunk in schunk0.iterchunks(dtype):
    sum += chunk.sum()
print(f"time sum (no postfilter): {time() - t0:.2f}s")
print(sum)

t0 = time()
sum = 0
for chunk in schunk.iterchunks(dtype):
    sum += chunk.sum()
print(f"time sum (postfilter): {time() - t0:.2f}s")
print(sum)
