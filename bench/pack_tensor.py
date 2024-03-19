#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################


"""
Packaging tensors (PyTorch, TensorFlow) larger than 2 GB.
"""

import io
import sys
import time

import numpy as np
import tensorflow as tf
import torch

import blosc2

NREP = 1
# N = int(5e8 + 2**27)  # larger than 2 GB
# Using tensors > 2 GB makes tensorflow serialization to raise this error:
# [libprotobuf FATAL google/protobuf/io/coded_stream.cc:831] CHECK failed: overrun <= kSlopBytes:
N = int(1e8)

store = True
if len(sys.argv) > 1:
    store = True

# blosc2.set_nthreads(8)

print(f"Creating NumPy array with {float(N):.3g} float32 elements...")
# in_ = np.arange(N, dtype=np.float32)
in_ = np.linspace(0, 1, N, dtype=np.float32)

if store:
    tt = tf.constant(in_)
    th = torch.from_numpy(in_)

    # Standard TensorFlow serialization
    c = None
    ctic = time.time()
    for i in range(NREP):
        c = tf.io.serialize_tensor(tt).numpy()
    ctoc = time.time()
    tc = (ctoc - ctic) / NREP
    print(
        "  Time for tensorflow (tf.io.serialize):\t{:.3f} s ({:.2f} GB/s)) ".format(
            tc, ((N * 8 / tc) / 2**30)
        ),
        end="",
    )
    print("\tcr: {:5.1f}x".format(in_.size * in_.dtype.itemsize * 1.0 / len(c)))

    with open("serialize_tensorflow.bin", "wb") as f:
        f.write(c)

    # Standard PyTorch serialization
    c = None
    buff = io.BytesIO()
    ctic = time.time()
    for i in range(NREP):
        torch.save(th, buff)
    ctoc = time.time()
    tc = (ctoc - ctic) / NREP
    print(
        "  Time for torch (torch.save):\t\t\t{:.3f} s ({:.2f} GB/s)) ".format(tc, ((N * 8 / tc) / 2**30)),
        end="",
    )
    buff.seek(0)
    c = buff.read()
    print("\tcr: {:5.1f}x".format(in_.size * in_.dtype.itemsize * 1.0 / len(c)))

    with open("serialize_torch.bin", "wb") as f:
        f.write(c)

    codec = blosc2.Codec.LZ4
    # print(f"Storing with {codec}")
    cparams = {"codec": codec, "clevel": 9}

    c = None
    ctic = time.time()
    for i in range(NREP):
        c = blosc2.pack_tensor(in_, cparams=cparams)
    ctoc = time.time()
    tc = (ctoc - ctic) / NREP
    print(
        "  Time for tensorflow (blosc2.pack_tensor):\t{:.3f} s ({:.2f} GB/s)) ".format(
            tc, ((N * 8 / tc) / 2**30)
        ),
        end="",
    )
    print("\tcr: {:5.1f}x".format(in_.size * in_.dtype.itemsize * 1.0 / len(c)))

    with open("pack_tensorflow.bl2", "wb") as f:
        f.write(c)

    tt = torch.from_numpy(in_)
    c = None
    ctic = time.time()
    for i in range(NREP):
        c = blosc2.pack_tensor(in_, cparams=cparams)
    ctoc = time.time()
    tc = (ctoc - ctic) / NREP
    print(
        "  Time for torch (blosc2.pack_tensor):\t\t{:.3f} s ({:.2f} GB/s)) ".format(
            tc, ((N * 8 / tc) / 2**30)
        ),
        end="",
    )
    print("\tcr: {:5.1f}x".format(in_.size * in_.dtype.itemsize * 1.0 / len(c)))

    with open("pack_torch.bl2", "wb") as f:
        f.write(c)

if True:
    with open("serialize_tensorflow.bin", "rb") as f:
        c = f.read()

    out = None
    dtic = time.time()
    for i in range(NREP):
        out = tf.io.parse_tensor(c, out_type=in_.dtype)
    dtoc = time.time()
    td = (dtoc - dtic) / NREP
    print(
        "  Time for tensorflow (tf.io.parse_tensor):\t{:.3f} s ({:.2f} GB/s)) ".format(
            td, ((N * 8 / td) / 2**30)
        ),
    )

    with open("serialize_torch.bin", "rb") as f:
        buff = io.BytesIO(f.read())

    out = None
    dtic = time.time()
    for i in range(NREP):
        buff.seek(0)
        out = torch.load(buff)
    dtoc = time.time()
    td = (dtoc - dtic) / NREP
    print(
        "  Time for torch (torch.load):\t\t\t{:.3f} s ({:.2f} GB/s)) ".format(td, ((N * 8 / td) / 2**30)),
    )

    with open("pack_tensorflow.bl2", "rb") as f:
        c = f.read()

    out = None
    dtic = time.time()
    for i in range(NREP):
        out = blosc2.unpack_tensor(c)
    dtoc = time.time()
    td = (dtoc - dtic) / NREP
    print(
        "  Time for tensorflow (blosc2.unpack_tensor):\t{:.3f} s ({:.2f} GB/s)) ".format(
            td, ((N * 8 / td) / 2**30)
        ),
    )
    assert np.array_equal(in_, out)

    with open("pack_torch.bl2", "rb") as f:
        c = f.read()

    out = None
    dtic = time.time()
    for i in range(NREP):
        out = blosc2.unpack_tensor(c)

    dtoc = time.time()

    td = (dtoc - dtic) / NREP
    print(
        "  Time for torch (blosc2.unpack_tensor):\t{:.3f} s ({:.2f} GB/s)) ".format(
            td, ((N * 8 / td) / 2**30)
        ),
    )
    assert np.array_equal(in_, out)
