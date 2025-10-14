# Benchmark tensordot
import sys
from time import time

import numpy as np
import blosc2
import dask
import dask.array as da
import zarr
from numcodecs import Blosc
import h5py
import hdf5plugin
import b2h5py.auto
assert(b2h5py.is_fast_slicing_enabled())


# --- Experiment Setup ---
N = 600
shape_a = (N,) * 3
shape_b = (N,) * 3
shape_out = (N,) * 2
chunks = (150,) * 3
chunks_out = (150,) * 2
dtype = np.float64
cparams = blosc2.CParams(codec=blosc2.Codec.LZ4, clevel=1)
compressor = Blosc(cname='lz4', clevel=1, shuffle=Blosc.SHUFFLE)
h5compressor = hdf5plugin.Blosc2(cname='lz4', clevel=1, filters=hdf5plugin.Blosc2.SHUFFLE)
scheduler = "single-threaded" if blosc2.nthreads == 1 else "threads"
create = True

# --- Numpy array creation ---
if create:
    t0 = time()
    # matrix_numpy = np.linspace(0, 1, N**3).reshape(shape_a)
    matrix_numpy = np.ones(N**3).reshape(shape_a)
    print(f"N={N}, Numpy array creation = {time() - t0:.2f} s")

# --- Blosc2 array creation ---
if create:
    t0 = time()
    matrix_a_blosc2 = blosc2.asarray(matrix_numpy, cparams=cparams, chunks=chunks, urlpath="a.b2nd", mode="w")
    matrix_b_blosc2 = blosc2.asarray(matrix_numpy, cparams=cparams, chunks=chunks, urlpath="b.b2nd", mode="w")
    print(f"N={N}, Array creation = {time() - t0:.2f} s")

# Re-open the arrays
t0 = time()
matrix_a_blosc2 = blosc2.open("a.b2nd", mode="r")
matrix_b_blosc2 = blosc2.open("b.b2nd", mode="r")
print(f"N={N}, Blosc2 array opening = {time() - t0:.2f} s")

# --- Tensordot computation ---
for axis in ((0, 1), (1, 2), (2, 0)):
    t0 = time()
    lexpr = blosc2.lazyexpr("tensordot(matrix_a_blosc2, matrix_b_blosc2, axes=(axis, axis))")
    out_blosc2 = lexpr.compute(urlpath="out.b2nd", mode="w", chunks=chunks_out)
    print(f"axes={axis}, Blosc2 Performance = {time() - t0:.2f} s")

# --- HDF5 array creation ---
if create:
    t0 = time()
    f = h5py.File("a_b_out.h5", "w")
    f.create_dataset("a", data=matrix_numpy, chunks=chunks, **h5compressor)
    f.create_dataset("b", data=matrix_numpy, chunks=chunks, **h5compressor)
    f.create_dataset("out", shape=shape_out, dtype=dtype, chunks=chunks_out, **h5compressor)
    print(f"N={N}, HDF5 array creation = {time() - t0:.2f} s")
    f.close()

# Re-open the HDF5 arrays
t0 = time()
f = h5py.File("a_b_out.h5", "a")
matrix_a_hdf5 = f["a"]
matrix_b_hdf5 = f["b"]
out_hdf5 = f["out"]
print(f"N={N}, HDF5 array opening = {time() - t0:.2f} s")

# --- Tensordot computation with HDF5 ---
for axis in ((0, 1), (1, 2), (2, 0)):
    t0 = time()
    blosc2.evaluate("tensordot(matrix_a_hdf5, matrix_b_hdf5, axes=(axis, axis))", out=out_hdf5)
    print(f"axes={axis}, HDF5 Performance = {time() - t0:.2f} s")
f.close()

# --- Zarr array creation ---
if create:
    t0 = time()
    matrix_a_zarr = zarr.open_array("a.zarr", mode="w", shape=shape_a, chunks=chunks,
                                    dtype=dtype, compressor=compressor, zarr_format=2)
    matrix_a_zarr[:] = matrix_numpy

    matrix_b_zarr = zarr.open_array("b.zarr", mode="w", shape=shape_b, chunks=chunks,
                                    dtype=dtype, compressor=compressor, zarr_format=2)
    matrix_b_zarr[:] = matrix_numpy
    print(f"N={N}, Zarr array creation = {time() - t0:.2f} s")

# --- Re-open the Zarr arrays ---
t0 = time()
matrix_a_zarr = zarr.open("a.zarr", mode="r")
matrix_b_zarr = zarr.open("b.zarr", mode="r")
matrix_a_dask = da.from_zarr(matrix_a_zarr)
matrix_b_dask = da.from_zarr(matrix_b_zarr)
print(f"N={N}, Dask + Zarr array opening = {time() - t0:.2f} s")

# --- Tensordot computation with Dask ---
zout = zarr.open_array("out.zarr", mode="w", shape=shape_out, chunks=chunks_out,
                       dtype=dtype, compressor=compressor, zarr_format=2)
with dask.config.set(scheduler=scheduler, num_workers=blosc2.nthreads):
    for axis in ((0, 1), (1, 2), (2, 0)):
        t0 = time()
        dexpr = da.tensordot(matrix_a_dask, matrix_b_dask, axes=(axis, axis))
        da.to_zarr(dexpr, zout)
        print(f"axes={axis}, Dask Performance = {time() - t0:.2f} s")

# --- Tensordot computation with Blosc2
zout2 = zarr.open_array("out2.zarr", mode="w", shape=shape_out, chunks=chunks_out,
                        dtype=dtype, compressor=compressor, zarr_format=2)
b2out = blosc2.empty(shape=shape_out, chunks=chunks_out, dtype=dtype, cparams=cparams, urlpath="out2.b2nd", mode="w")
for axis in ((0, 1), (1, 2), (2, 0)):
    t0 = time()
    blosc2.evaluate("tensordot(matrix_a_zarr, matrix_b_zarr, axes=(axis, axis))", out=zout2)
    print(f"axes={axis}, Blosc2 Performance = {time() - t0:.2f} s")
