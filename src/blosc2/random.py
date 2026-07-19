#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################
"""Chunk-parallel NumPy-quality random generators producing :class:`blosc2.NDArray`.

Each chunk gets its own independent stream, spawned from a root
:class:`numpy.random.SeedSequence` via :meth:`~numpy.random.SeedSequence.spawn`, and
generation runs in a thread pool (NumPy ``Generator`` fill loops release the GIL).
Results are reproducible for a given ``(seed, call order, shape, chunks)``; changing
the chunk layout changes the values, since the chunk grid determines the stream
assignment.
"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Any

import numpy as np

import blosc2


def _chunk_slices(a: blosc2.NDArray):
    for info in a.iterchunks_info():
        yield tuple(
            slice(c * s, min((c + 1) * s, sh))
            for c, s, sh in zip(info.coords, a.chunks, a.shape, strict=False)
        )


def _bounded_map(fn, n: int, max_workers: int):
    # ponytail: window bounds peak memory to O(workers * chunk); a plain
    # executor.map would submit all n chunks upfront and could buffer O(n).
    window = 2 * max_workers
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        pending = {ex.submit(fn, i): i for i in range(min(window, n))}
        next_i = len(pending)
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for fut in done:
                del pending[fut]
                yield fut.result()
                if next_i < n:
                    pending[ex.submit(fn, next_i)] = next_i
                    next_i += 1


class Generator:
    """Chunk-parallel counterpart of :class:`numpy.random.Generator`.

    Create via :func:`default_rng`. Each method call draws a full
    :class:`~blosc2.NDArray` in one shot, generating its chunks concurrently.
    """

    def __init__(self, seed=None):
        self._root = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)

    def _fill(self, method: str, args: tuple, kwargs: dict, shape, **b2_kwargs: Any) -> blosc2.NDArray:
        if shape is None:
            raise TypeError("shape is required")
        dtype = getattr(np.random.default_rng(0), method)(*args, size=1, **kwargs).dtype
        dst = blosc2.empty(shape, dtype=dtype, **b2_kwargs)

        slices = list(_chunk_slices(dst))
        call_ss = self._root.spawn(1)[0]
        chunk_streams = call_ss.spawn(len(slices))

        def gen(i):
            rng = np.random.default_rng(chunk_streams[i])
            chunk_shape = tuple(s.stop - s.start for s in slices[i])
            return i, getattr(rng, method)(*args, size=chunk_shape, **kwargs)

        if len(slices) <= 1 or blosc2.nthreads == 1:
            results = (gen(i) for i in range(len(slices)))
        else:
            results = _bounded_map(gen, len(slices), blosc2.nthreads)

        for i, buf in results:
            dst[slices[i]] = buf
        return dst

    def random(self, shape, dtype=np.float64, **kwargs: Any) -> blosc2.NDArray:
        """Draw uniform floats in [0, 1). Mirrors :meth:`numpy.random.Generator.random`."""
        return self._fill("random", (), {"dtype": dtype}, shape, **kwargs)

    def integers(
        self, low, high=None, *, shape, dtype=np.int64, endpoint: bool = False, **kwargs: Any
    ) -> blosc2.NDArray:
        """Draw random integers. Mirrors :meth:`numpy.random.Generator.integers`."""
        return self._fill("integers", (low, high), {"dtype": dtype, "endpoint": endpoint}, shape, **kwargs)

    def normal(self, loc=0.0, scale=1.0, *, shape, **kwargs: Any) -> blosc2.NDArray:
        """Draw samples from a normal distribution. Mirrors :meth:`numpy.random.Generator.normal`."""
        return self._fill("normal", (loc, scale), {}, shape, **kwargs)

    def uniform(self, low=0.0, high=1.0, *, shape, **kwargs: Any) -> blosc2.NDArray:
        """Draw samples from a uniform distribution. Mirrors :meth:`numpy.random.Generator.uniform`."""
        return self._fill("uniform", (low, high), {}, shape, **kwargs)


def default_rng(seed=None) -> Generator:
    """Construct a chunk-parallel :class:`Generator`, mirroring :func:`numpy.random.default_rng`.

    Parameters
    ----------
    seed: int, array-like, SeedSequence or None
        Seed for the root :class:`numpy.random.SeedSequence`. ``None`` (default) draws
        entropy from the OS.

    Returns
    -------
    out: :class:`Generator`
    """
    return Generator(seed)
