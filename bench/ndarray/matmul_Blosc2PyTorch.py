#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

### Matmul performance comparison between Blosc2 and PyTorch with persistent storage
# For this bench to work, you first need to download the data file at:
# http://www.silx.org/pub/pyFAI/pyFAI_UM_2020/data_ID13/kevlar.h5

import numpy as np
import blosc2
import torch
import pickle
from time import time
import h5py
import hdf5plugin
from tqdm import tqdm  # progress bar

cparams = {
    "codec": blosc2.Codec.LZ4,
    "filters": [blosc2.Filter.SHUFFLE],
    "clevel": 1,
}
batch_size = 32
CREATE = True
dtype = np.float32

# Check what's available
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"CUDA available: {torch.cuda.is_available()}")

# GPU for PyTorch
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device("gpu" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")  # Force CPU usage
print(f"Using device: {device}")

if CREATE:
    def build_dense_rowwarp_matrix(out_h=2000, in_h=2167,
                                scale=1.0,
                                ripple_amplitude=30.0,
                                ripple_period=400.0,
                                blur_radius=1,
                                row_gain_amplitude=0.15):
        """
        Same function as before — builds a vertical warp matrix A of shape (out_h, in_h)
        that can be applied as A @ img.
        """
        A = np.zeros((out_h, in_h), dtype=dtype)
        i = np.arange(out_h, dtype=dtype)
        t = i / max(out_h - 1, 1)
        linear_src = t * (in_h - 1) * scale
        ripple = ripple_amplitude * np.sin(2.0 * np.pi * i / ripple_period)
        src = linear_src + ripple
        row_gain = 1.0 + row_gain_amplitude * np.cos(2.0 * np.pi * i / (ripple_period * 0.5))
        for out_r in range(out_h):
            s = src[out_r]
            k_min = int(np.floor(s)) - blur_radius
            k_max = int(np.floor(s)) + blur_radius + 1
            k_min_clamped = max(k_min, 0)
            k_max_clamped = min(k_max, in_h - 1) + 1
            ks = np.arange(k_min_clamped, k_max_clamped, dtype=np.int32)
            d = np.abs(ks - s)
            w = np.maximum(0.0, 1.0 - d / (blur_radius + 1e-6))
            if w.sum() > 0:
                w = w / w.sum()
            w = w * row_gain[out_r]
            A[out_r, ks] = w.astype(dtype)
        return A

    NUM_IMAGES = 2000
    IN_H, OUT_H, W = 2167, 2000, 2070

    out = blosc2.empty(shape=(NUM_IMAGES, OUT_H, IN_H), dtype=dtype, urlpath="transform.b2nd", mode='w', cparams=cparams)

    for i in tqdm(range(NUM_IMAGES), desc="Generating and saving transform matrices to Blosc2"):
        # Randomize warp parameters a little per image
        ripple_amp = 20 + np.random.uniform(-5, 5)
        ripple_period = 300 + np.random.uniform(-30, 30)
        row_gain_amp = 0.10 + np.random.uniform(-0.05, 0.05)
        blur_r = np.random.choice([0, 1, 2])

        # Build and apply matrix
        A = build_dense_rowwarp_matrix(out_h=OUT_H, in_h=IN_H,
                                        ripple_amplitude=ripple_amp,
                                        ripple_period=ripple_period,
                                        blur_radius=blur_r,
                                        row_gain_amplitude=row_gain_amp)
        out[i] = A

    fname_in = "kevlar.h5"  # input file with the kevlar dataset
    with h5py.File(fname_in, "r") as fr:  # load file and process to blosc2 array
        dset = fr["/entry/data/data"]
        b2im = blosc2.empty(shape=(2*len(dset), 2167, 2070), dtype=dtype, cparams=cparams, urlpath="kevlar.b2nd", mode="w")
        for i in tqdm(range(0, len(dset), batch_size), desc="Converting data matrices to Blosc2"):
            end = min((i+batch_size), len(dset))
            res = dset[i:end]
            res = np.where(res>10, 0, res)
            # For visibility, zero-out pixels
            b2im[i:end] = res
            b2im[i + 1000, end + 1000] = res
        del dset

    b2im = blosc2.open(urlpath="kevlar.b2nd", mode="r")
    b2im_trans = blosc2.open(urlpath="transform.b2nd", mode="r")
    s, d = b2im.shape, b2im.dtype
    fname_out = "my_kevlar.h5"
    # Write to .h5 file #
    with h5py.File(fname_out, "w") as fw:
        b2comp = hdf5plugin.Blosc2(cname='lz4', clevel=1, filters=hdf5plugin.Blosc2.SHUFFLE) # just for identification, no compression algorithm specified
        dset_out1 = fw.create_dataset(
            "data",
            b2im.shape, b2im.dtype,
            **b2comp,
        )
        dset_out2 = fw.create_dataset(
            "transform",
            b2im_trans.shape, b2im_trans.dtype,
            **b2comp,
        )
        for i in tqdm(range(0, len(b2im), batch_size), desc="Converting transform and data matrices to HDF5"):
            dset_out1[i:i+batch_size] = b2im[i:i+batch_size]
            dset_out2[i:i+batch_size] = b2im_trans[i:i+batch_size]


# Re-open the arrays
dset_a = blosc2.open("transform.b2nd", mode="r")
dset_b = blosc2.open("kevlar.b2nd", mode="r")
print(f'Total working set size: {round((np.prod(dset_a.shape)/ 2 ** 30 + np.prod(dset_a.shape[:-1]+dset_b.shape[-1:])/ 2 ** 30 + np.prod(dset_b.shape)/ 2 ** 30) * dset_b.dtype.itemsize, 1)} GB.')

# --- Matmul Blosc2 ---
t0 = time()
out_blosc = blosc2.matmul(dset_a, dset_b, urlpath='out.b2nd', mode="w", cparams=cparams)
blosc_time = time() - t0
chunks_blosc = [dset_a.chunks, dset_b.chunks]
chunks_blosc_out = out_blosc.chunks
in_shapes = [dset_a.shape, dset_b.shape]
print(f"Blosc2 Performance = {blosc_time:.2f} s")

h5compressor = hdf5plugin.Blosc2(cname='lz4', clevel=1, filters=hdf5plugin.Blosc2.SHUFFLE)
t0 = time()
f = h5py.File("my_kevlar.h5", "r+")
if not ("out" in f):
    f.create_dataset("out", shape=out_blosc.shape, dtype=out_blosc.dtype, **h5compressor)
# Re-open the HDF5 arrays
t0 = time()
with h5py.File("my_kevlar.h5", "r+") as f:
    dset_a = f["transform"]
    dset_b = f["data"]
    dset_out = f["out"]

    for i in range(0, len(dset_out), batch_size):
        batch_a = torch.from_numpy(dset_a[i:i+batch_size]).to(device)
        batch_b = torch.from_numpy(dset_b[i:i+batch_size]).to(device)
        dset_out[i:i+batch_size] = torch.matmul(batch_a, batch_b)
    hdf5_chunks = [dset_a.chunks, dset_b.chunks]
    hdf5_chunks_out = dset_out.chunks
torch_time = time() - t0
print(f"PyTorch Performance = {torch_time:.2f} s")

results = {'blosc_chunks_out': chunks_blosc_out, 'blosc_chunks': chunks_blosc,
           'hdf5_chunks_out': hdf5_chunks_out, 'hdf5_chunks': hdf5_chunks,
           'ABshape': in_shapes, 'dtype': out_blosc.dtype, 'PyTorch': torch_time, 'Blosc2': blosc_time}
fname = 'matmul_OOC'
with open(f'{fname}.pkl', 'wb') as f:
    pickle.dump(results, f)
