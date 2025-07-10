#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

# Benchmark for computing a fancy index of a blosc2 array

import numpy as np
import ndindex
import blosc2
import time
from memory_profiler import memory_usage, profile
import matplotlib.pyplot as plt

d = 160
shape = (d,) * 4
chunks = (d // 4,) * 4
blocks = (d // 10,) * 4
print(f"Creating a 4D array of shape {shape} with chunks {chunks} and blocks {blocks}...")
t = time.time()
arr = blosc2.ones(shape=shape, chunks=chunks, blocks=blocks) #, urlpath=file, mode="w")
t = time.time() - t
print(f"Time to create array: {t:.6f} seconds")
t = time.time()
idx = np.random.randint(low=0, high=d, size=(d//4,))
fancyIdx = np.s_[idx, 40, :d//2, d//2:]
ans = arr[:][fancyIdx]

# dim0
# @profile
def slice_dim1():
    t = time.time()
    _slice = ndindex.ndindex(fancyIdx).expand(shape)
    chunk_size = ndindex.ChunkSize(chunks)

    # repeated indices are grouped together
    intersecting_chunks = chunk_size.as_subchunks(_slice, shape)  # if _slice is (), returns all chunks
    out_shape = _slice.newshape(shape)
    print(out_shape)

    # Attempt 1
    out = np.empty(out_shape, dtype=arr.dtype)
    for n, c in enumerate(chunk_size.indices(shape)):
        try:
            sub_idx = _slice.as_subindex(c).raw
            sel_idx = c.as_subindex(_slice).raw
            chunk = blosc2.empty(
                shape=arr.chunks, chunks=arr.chunks, blocks=arr.blocks, dtype=arr.dtype
            )  # very cheap memory allocation
            chunk.schunk.insert_chunk(0, arr.get_chunk(n))
            out[sel_idx] = chunk[:][sub_idx]
        except ValueError:
            # This happens when the _slice and chunk do not intersect
            continue
    t0 = time.time() - t
    np.testing.assert_allclose(out, ans)
    print(f"Time to access blosc2, method 1: {t0:.6f} seconds")

# @profile
def slice_dim2():
    t = time.time()
    _slice = ndindex.ndindex(fancyIdx).expand(shape)
    chunk_size = ndindex.ChunkSize(chunks)

    # repeated indices are grouped together
    intersecting_chunks = chunk_size.as_subchunks(_slice, shape)  # if _slice is (), returns all chunks
    out_shape = _slice.newshape(shape)
    ## Attempt 2
    out = np.empty(out_shape, dtype=arr.dtype)
    for c in intersecting_chunks:
        sub_idx = _slice.as_subindex(c).raw
        sel_idx = c.as_subindex(_slice).raw
        chunk = arr[c.raw]
        out[sel_idx] = chunk[sub_idx]
    t0 = time.time() - t
    np.testing.assert_allclose(out, ans)
    print(f"Time to access blosc2, method 2: {t0:.6f} seconds")

# @profile
def slice_dimNumpy():
    t = time.time()
    res = arr[:][fancyIdx]
    t0 = time.time() - t
    print(f"Time to access numpy: {t0:.6f} seconds")
    print(f"dim0 slice size: {np.prod(res.shape) * res.dtype.itemsize / 2**30:.6f} GB")

if __name__ == '__main__':
    interval = 0.001
    offset = 0
    for f in [slice_dim1, slice_dim2, slice_dimNumpy]:
        # mem = memory_usage((f,), interval=interval)
        f()
    #     times = offset + interval * np.arange(len(mem))
    #     offset = times[-1]
    #     plt.plot(times, mem)
    #
    # plt.xlabel('Time (s)')
    # plt.ylabel('Memory usage (MiB)')
    # plt.title('Memory usage fancy indexing')
    # plt.legend(['method1', 'method2', 'numpy'])
    # plt.savefig('plots/fancyIdx.png', format="png")
