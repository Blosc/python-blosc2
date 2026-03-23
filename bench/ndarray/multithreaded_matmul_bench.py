import blosc2
import numpy as np
import time

N = 10000
ndim = 2
ashape = (N,) * ndim
bshape = ashape
dtype = np.float64

achunks = (1000, 1000)
bchunks = (achunks[1], achunks[0])
ablocks = (200, 200)
bblocks = (ablocks[1], ablocks[0])
outblocks = (ablocks[0], bblocks[1])
outchunks = (achunks[0], bchunks[1])
# a = blosc2.linspace(0, 1, dtype=dtype, shape=ashape, chunks=achunks, blocks=ablocks)
# b = blosc2.linspace(0, 1, dtype=dtype, shape=bshape, chunks=bchunks, blocks=bblocks)
a = blosc2.ones(dtype=dtype, shape=ashape, chunks=achunks, blocks=ablocks)
b = blosc2.full(fill_value=2, dtype=dtype, shape=bshape, chunks=bchunks, blocks=bblocks)

a_np = a[:]
b_np = b[:]
tic = time.time()
np_res = np.matmul(a_np, b_np)
print(f'numpy finished in {time.time()-tic} s')

tic = time.time()
b2_res = blosc2.matmul(a, b, blocks=outblocks, chunks=outchunks)
print(f'blosc2 multithreaded finished in {time.time()-tic} s')

tic = time.time()
b2_res = blosc2.matmul(a, b)
print(f'blosc2 normal finished in {time.time()-tic} s')

achunks = None #(1000, 1000)
bchunks = None #(achunks[1], achunks[0])
ablocks = None #(200, 200)
bblocks = None #(ablocks[1], ablocks[0])
outblocks = None #(ablocks[0], bblocks[1])
outchunks = None #(achunks[0], bchunks[1])
# a = blosc2.linspace(0, 1, dtype=dtype, shape=ashape, chunks=achunks, blocks=ablocks)
# b = blosc2.linspace(0, 1, dtype=dtype, shape=bshape, chunks=bchunks, blocks=bblocks)
a = blosc2.ones(dtype=dtype, shape=ashape, chunks=achunks, blocks=ablocks)
b = blosc2.full(fill_value=2, dtype=dtype, shape=bshape, chunks=bchunks, blocks=bblocks)
tic = time.time()
b2_res = blosc2.matmul(a, b, blocks=outblocks, chunks=outchunks)
print(f'blosc2 normal with default chunks etc. finished in {time.time()-tic} s')
