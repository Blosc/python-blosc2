#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

# This shows how to convert a generic .h5 file to a custom blosc2-compressed .h5 file
# The blosc2 plugin in hdf5plugin doesn't support custom block shapes, and so one
# has to go a different route for more bespoke compression

import os

import h5py
import hdf5plugin

import blosc2

clevel = 5  # compression level, e.g., 0-9, where 0 is no compression and 9 is maximum compression
fname_in = "kevlar.h5"  # input file with the kevlar dataset
fname_out = "kevlar-blosc2.h5"
nframes = 1000
if not os.path.exists(fname_in):
    raise FileNotFoundError(
        f"Input file {fname_in} does not exist\n"
        "Please download it from the kevlar repository at:"
        " http://www.silx.org/pub/pyFAI/pyFAI_UM_2020/data_ID13/kevlar.h5"
    )

# Example 1
# hdf5plugin supports limited blosc2 compression with certain codecs
cname = "zstd"

if not os.path.exists("STD" + fname_out):
    with h5py.File(fname_in, "r") as fr:
        dset = fr["/entry/data/data"][:nframes]
    with h5py.File("STD" + fname_out, "w") as fw:
        g = fw.create_group("/data")
        b2comp = hdf5plugin.Blosc2(cname=cname, clevel=clevel, filters=hdf5plugin.Blosc2.BITSHUFFLE)
        dset_out = g.create_dataset(
            f"cname-{cname}",
            data=dset,
            dtype=dset.dtype,
            chunks=(1,) + dset.shape[1:],  # chunk size of 1 frame
            **b2comp,
        )
print("Successfully compressed file with hdf5plugin")

# Example 2
# For other codecs (e.g grok) or for more custom compression such as with user-defined block shapes, one
# has to use a more involved route
blocks = (50, 80, 80)
chunks = (100, 240, 240)
cparams = {
    "codec": blosc2.Codec.LZ4,
    "filters": [blosc2.Filter.BITSHUFFLE],
    "splitmode": blosc2.SplitMode.NEVER_SPLIT,
    "clevel": clevel,
}

if os.path.exists("dset.b2nd"):  # don't reload dset to blosc2 if already done so once
    b2im = blosc2.open(urlpath="dset.b2nd", mode="r")
else:
    with h5py.File(fname_in, "r") as fr:  # load file and process to blosc2 array
        dset = fr["/entry/data/data"][:nframes]
        b2im = blosc2.asarray(
            dset, chunks=chunks, blocks=blocks, cparams=cparams, urlpath="dset.b2nd", mode="w"
        )
        del dset

s, d = b2im.shape, b2im.dtype
# Write to .h5 file #
with h5py.File("Custom" + fname_out, "w") as fw:
    g = fw.create_group("/data")
    b2comp = hdf5plugin.Blosc2()  # just for identification, no compression algorithm specified
    dset_out = g.create_dataset(
        "cname-customlz4",
        s,
        d,
        chunks=chunks,  # chunk size of 1 frame
        **b2comp,
    )
    # Write individual blosc2 chunks directly to hdf5
    # hdf5 requires a cframe, which is only available via blosc2 schunks (not chunks)
    for info in b2im.iterchunks_info():
        ncoords = tuple(n * chunks[i] for i, n in enumerate(info.coords))
        aux = blosc2.empty(
            shape=b2im.chunks, chunks=b2im.chunks, blocks=b2im.blocks, dtype=b2im.dtype
        )  # very cheap memory allocation
        aux.schunk.insert_chunk(
            0, b2im.get_chunk(info.nchunk)
        )  # insert chunk into blosc2 array so we have schunk wrapper (no decompression required)
        dset_out.id.write_direct_chunk(
            ncoords, aux.schunk.to_cframe()
        )  # convert schunk to cframe and write to hdf5
    print("Successfully compressed file with custom parameters")
