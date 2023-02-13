#######################################################################
# Copyright (C) 2019-present, Blosc Development team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

# Benchmark for comparing speeds of getitem of hyperplanes on a
# multidimensional array and using different backends:
# blosc2, Zarr and HDF5
# In brief, each approach has its own strengths and weaknesses.
#
# Usage: pass any argument for testing the persistent backends.
# Else, only in-memory containers will be tested.

import os
import shutil
import sys
from time import time

import numcodecs
import tables
import zarr

import blosc2
import numpy as np

persistent = bool(sys.argv[1]) if len(sys.argv) > 1 else False
if persistent:
    print("Testing the persistent backends...")
else:
    print("Testing the in-memory backends...")

# Dimensions and type properties for the arrays
shape = (100, 5000, 250)
chunks = (20, 500, 50)
blocks = (10, 100, 25)
# This config generates containers of more than 2 GB in size
# shape = (250, 4000, 350)
# pshape = (200, 100, 100)
dtype = np.float64

# Compression properties
cparams = {"codec": blosc2.Codec.ZSTD, "clevel": 6, "filters": [blosc2.Filter.SHUFFLE],
           "filters_meta": [0], "nthreads": 1}
dparams = {"nthreads": 1}
cname = "zstd"
clevel = 6
filter = blosc2.Filter.SHUFFLE
zfilter = numcodecs.Blosc.SHUFFLE
nthreads = 1
blocksize = int(np.prod(blocks))

fname_b2nd = None
fname_zarr = None
fname_h5 = "whatever.h5"
if persistent:
    fname_b2nd = "compare_getslice.b2nd"
    blosc2.remove_urlpath(fname_b2nd)
    fname_zarr = "compare_getslice.zarr"
    blosc2.remove_urlpath(fname_zarr)
    fname_h5 = "compare_getslice.h5"
    blosc2.remove_urlpath(fname_h5)

# Create content for populating arrays
content = np.random.normal(0, 1, int(np.prod(shape))).reshape(shape)

# Create and fill a b2nd array using a block iterator
t0 = time()
a = blosc2.empty(shape, dtype=content.dtype, chunks=chunks, blocks=blocks,
                 urlpath=fname_b2nd, cparams=cparams)
a[:] = content
acratio = a.schunk.cratio
if persistent:
    del a
t1 = time()
print("Time for filling array (b2nd, iter): %.3fs ; CRatio: %.1fx" % ((t1 - t0), acratio))

# Create and fill a zarr array
t0 = time()
compressor = numcodecs.Blosc(cname=cname, clevel=clevel, shuffle=zfilter, blocksize=blocksize)
numcodecs.blosc.set_nthreads(nthreads)
if persistent:
    z = zarr.open(fname_zarr, mode='w', shape=shape, chunks=chunks, dtype=dtype, compressor=compressor)
else:
    z = zarr.empty(shape=shape, chunks=chunks, dtype=dtype, compressor=compressor)
z[:] = content
zratio = z.nbytes / z.nbytes_stored
if persistent:
    del z
t1 = time()
print("Time for filling array (zarr): %.3fs ; CRatio: %.1fx" % ((t1 - t0), zratio))

# Create and fill a hdf5 array
t0 = time()
filters = tables.Filters(complevel=clevel, complib="blosc:%s" % cname, shuffle=True)
tables.set_blosc_max_threads(nthreads)
if persistent:
    h5f = tables.open_file(fname_h5, 'w')
else:
    h5f = tables.open_file(fname_h5, 'w', driver='H5FD_CORE', driver_core_backing_store=0)
h5ca = h5f.create_carray(h5f.root, 'carray', filters=filters, chunkshape=chunks, obj=content)
h5f.flush()
h5ratio = h5ca.size_in_memory / h5ca.size_on_disk
if persistent:
    h5f.close()
t1 = time()
print("Time for filling array (hdf5): %.3fs ; CRatio: %.1fx" % ((t1 - t0), h5ratio))

# Setup the coordinates for random planes
planes_idx = np.random.randint(0, shape[1], 100)

# Time getitem with blosc2
t0 = time()
if persistent:
    a = blosc2.open(fname_b2nd)  # reopen
for i in planes_idx:
    rbytes = a[:, i, :]
del a
t1 = time()
print("Time for reading with getitem (blosc2): %.3fs" % (t1 - t0))

# Time getitem with zarr
t0 = time()
if persistent:
    z = zarr.open(fname_zarr, mode='r')
for i in planes_idx:
    block = z[:, i, :]
del z
t1 = time()
print("Time for reading with getitem (zarr): %.3fs" % (t1 - t0))

# Time getitem with hdf5
t0 = time()
if persistent:
    h5f = tables.open_file(fname_h5, 'r', filters=filters)
h5ca = h5f.root.carray
for i in planes_idx:
    block = h5ca[:, i, :]
h5f.close()
t1 = time()
print("Time for reading with getitem (hdf5): %.3fs" % (t1 - t0))


if persistent:
    os.remove(fname_b2nd)
    shutil.rmtree(fname_zarr)
    os.remove(fname_h5)
