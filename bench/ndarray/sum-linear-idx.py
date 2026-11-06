#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Compare reduction performance on public Menudet kernels.
# This uses the special _flat_idx var.

from time import time

import numpy as np

import blosc2

dtype = np.int64
shape = (10_000, 10_000)
cparams = blosc2.CParams(codec=blosc2.Codec.BLOSCLZ, clevel=1)


@blosc2.menudet(ndim=2)
def kernel_ramp() -> np.int64:
    return _flat_idx  # noqa: F821  # Global logical index supplied by Menudet.


def make_lazy(shape, cparams):
    chunks, blocks = blosc2.compute_chunks_blocks(shape, dtype=dtype, cparams=cparams)
    storage = {"chunks": chunks, "blocks": blocks, "cparams": cparams}
    # Preserve global coordinates without re-evaluating the entire ramp per tile.
    lazy = kernel_ramp.specialize().lazy({}, shape=shape, partitions=chunks)
    return lazy, storage


def main():
    nitems = int(np.prod(shape))
    expected = nitems * (nitems - 1) // 2
    a, storage = make_lazy(shape, cparams)
    print(a.kernel.source)
    materialized = a.compute(**storage)
    t0 = time()
    result = materialized.sum()
    print("Blosc2 sum over NDArray:", round(time() - t0, 3), "s")
    assert result == expected

    t0 = time()
    a, storage = make_lazy(shape, cparams)
    result = a.sum(**storage)
    print("Blosc2 sum over Menudet LazyArray:", round(time() - t0, 3), "s")
    assert result == expected

    t0 = time()
    a, storage = make_lazy(shape, cparams)
    result = a.compute(**storage).sum()
    print("Menudet compute + sum:", round(time() - t0, 3), "s")
    assert result == expected

    t0 = time()
    a = blosc2.arange(nitems, dtype=dtype, shape=shape, **storage)
    result = a.sum()
    print("Blosc2 arange + sum:", round(time() - t0, 3), "s")
    assert result == expected

    t0 = time()
    npa = np.arange(nitems, dtype=dtype).reshape(shape)
    result = npa.sum()
    print("NumPy arange + sum:", round(time() - t0, 3), "s")
    assert result == expected


if __name__ == "__main__":
    main()
