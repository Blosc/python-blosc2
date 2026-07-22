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


def _materialize(x) -> np.ndarray:
    return x[:] if isinstance(x, blosc2.NDArray) else np.asarray(x)


def _bounded_map(fn, n: int, max_workers: int):
    # ponytail: window bounds peak memory to O(workers * chunk); a plain
    # executor.map would submit all n chunks upfront and could buffer O(n).
    window = 2 * max_workers
    try:
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
    except RuntimeError:
        # ponytail: some sandboxes (e.g. wasm32/Pyodide without pthreads) report a
        # real nthreads count but can't actually start OS threads at all — submit()
        # then raises "can't start new thread" on the very first call, before
        # anything has been yielded. Fall back to serial generation rather than
        # crashing; upgrade path is none needed, this is the correct behavior.
        for i in range(n):
            yield fn(i)


class Generator:
    """Chunk-parallel counterpart of :class:`numpy.random.Generator`.

    Create via :func:`default_rng`. Each method call draws a full
    :class:`~blosc2.NDArray` in one shot, generating its chunks concurrently.
    """

    def __init__(self, seed=None):
        self._root = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)

    def _fill_chunks(self, dst, method, args, kwargs, chunk_shape_of) -> blosc2.NDArray:
        slices = list(_chunk_slices(dst))
        call_ss = self._root.spawn(1)[0]
        chunk_streams = call_ss.spawn(len(slices))

        def gen(i):
            rng = np.random.default_rng(chunk_streams[i])
            return i, getattr(rng, method)(*args, size=chunk_shape_of(slices[i]), **kwargs)

        if len(slices) <= 1 or blosc2.nthreads == 1:
            results = (gen(i) for i in range(len(slices)))
        else:
            results = _bounded_map(gen, len(slices), blosc2.nthreads)

        for i, buf in results:
            dst[slices[i]] = buf
        return dst

    def _fill(self, method: str, args: tuple, kwargs: dict, shape, **b2_kwargs: Any) -> blosc2.NDArray:
        if shape is None:
            raise TypeError("shape is required")
        dtype = getattr(np.random.default_rng(0), method)(*args, size=1, **kwargs).dtype
        dst = blosc2.empty(shape, dtype=dtype, **b2_kwargs)
        return self._fill_chunks(dst, method, args, kwargs, lambda sl: tuple(s.stop - s.start for s in sl))

    def _fill_vector(
        self, method: str, args: tuple, kwargs: dict, shape, k: int, **b2_kwargs: Any
    ) -> blosc2.NDArray:
        # Each draw produces a whole length-k vector, so the trailing dimension must
        # never be split across chunks (a chunk can only hold complete draws).
        if shape is None:
            raise TypeError("shape is required")
        shape = (shape,) if isinstance(shape, int | np.integer) else tuple(shape)
        full_shape = shape + (k,)
        dtype = getattr(np.random.default_rng(0), method)(*args, size=1, **kwargs).dtype

        chunks = b2_kwargs.get("chunks")
        if chunks is not None:
            if tuple(chunks)[-1] != k:
                raise ValueError(f"chunks[-1] must equal the trailing vector length {k}")
        else:
            # ponytail: reuse empty()'s own chunk-sizing heuristic instead of
            # reimplementing it, then pin the non-splittable trailing dim to its full length.
            b2_kwargs["chunks"] = blosc2.empty(full_shape, dtype=dtype).chunks[:-1] + (k,)

        dst = blosc2.empty(full_shape, dtype=dtype, **b2_kwargs)
        return self._fill_chunks(
            dst, method, args, kwargs, lambda sl: tuple(s.stop - s.start for s in sl[:-1])
        )

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

    def beta(self, a, b, *, shape, **kwargs: Any) -> blosc2.NDArray:
        """Draw samples from a beta distribution. Mirrors :meth:`numpy.random.Generator.beta`."""
        return self._fill("beta", (a, b), {}, shape, **kwargs)

    def binomial(self, n, p, *, shape, **kwargs: Any) -> blosc2.NDArray:
        """Draw samples from a binomial distribution. Mirrors :meth:`numpy.random.Generator.binomial`."""
        return self._fill("binomial", (n, p), {}, shape, **kwargs)

    def chisquare(self, df, *, shape, **kwargs: Any) -> blosc2.NDArray:
        """Draw samples from a chi-square distribution. Mirrors :meth:`numpy.random.Generator.chisquare`."""
        return self._fill("chisquare", (df,), {}, shape, **kwargs)

    def exponential(self, scale=1.0, *, shape, **kwargs: Any) -> blosc2.NDArray:
        """Draw samples from an exponential distribution. Mirrors :meth:`numpy.random.Generator.exponential`."""
        return self._fill("exponential", (scale,), {}, shape, **kwargs)

    def f(self, dfnum, dfden, *, shape, **kwargs: Any) -> blosc2.NDArray:
        """Draw samples from an F distribution. Mirrors :meth:`numpy.random.Generator.f`."""
        return self._fill("f", (dfnum, dfden), {}, shape, **kwargs)

    def gamma(self, shape_param, /, scale=1.0, *, shape, **kwargs: Any) -> blosc2.NDArray:
        """Draw samples from a gamma distribution. Mirrors :meth:`numpy.random.Generator.gamma`.

        ``shape_param`` is numpy's ``shape`` (the gamma distribution's shape parameter,
        usually called *k*); renamed here, and made positional-only, to avoid colliding
        with the array-shape ``shape`` keyword.
        """
        return self._fill("gamma", (shape_param, scale), {}, shape, **kwargs)

    def geometric(self, p, *, shape, **kwargs: Any) -> blosc2.NDArray:
        """Draw samples from a geometric distribution. Mirrors :meth:`numpy.random.Generator.geometric`."""
        return self._fill("geometric", (p,), {}, shape, **kwargs)

    def gumbel(self, loc=0.0, scale=1.0, *, shape, **kwargs: Any) -> blosc2.NDArray:
        """Draw samples from a Gumbel distribution. Mirrors :meth:`numpy.random.Generator.gumbel`."""
        return self._fill("gumbel", (loc, scale), {}, shape, **kwargs)

    def hypergeometric(self, ngood, nbad, nsample, *, shape, **kwargs: Any) -> blosc2.NDArray:
        """Draw samples from a hypergeometric distribution. Mirrors :meth:`numpy.random.Generator.hypergeometric`."""
        return self._fill("hypergeometric", (ngood, nbad, nsample), {}, shape, **kwargs)

    def laplace(self, loc=0.0, scale=1.0, *, shape, **kwargs: Any) -> blosc2.NDArray:
        """Draw samples from a Laplace distribution. Mirrors :meth:`numpy.random.Generator.laplace`."""
        return self._fill("laplace", (loc, scale), {}, shape, **kwargs)

    def logistic(self, loc=0.0, scale=1.0, *, shape, **kwargs: Any) -> blosc2.NDArray:
        """Draw samples from a logistic distribution. Mirrors :meth:`numpy.random.Generator.logistic`."""
        return self._fill("logistic", (loc, scale), {}, shape, **kwargs)

    def lognormal(self, mean=0.0, sigma=1.0, *, shape, **kwargs: Any) -> blosc2.NDArray:
        """Draw samples from a log-normal distribution. Mirrors :meth:`numpy.random.Generator.lognormal`."""
        return self._fill("lognormal", (mean, sigma), {}, shape, **kwargs)

    def logseries(self, p, *, shape, **kwargs: Any) -> blosc2.NDArray:
        """Draw samples from a logarithmic series distribution. Mirrors :meth:`numpy.random.Generator.logseries`."""
        return self._fill("logseries", (p,), {}, shape, **kwargs)

    def negative_binomial(self, n, p, *, shape, **kwargs: Any) -> blosc2.NDArray:
        """Draw samples from a negative binomial distribution. Mirrors :meth:`numpy.random.Generator.negative_binomial`."""
        return self._fill("negative_binomial", (n, p), {}, shape, **kwargs)

    def noncentral_chisquare(self, df, nonc, *, shape, **kwargs: Any) -> blosc2.NDArray:
        """Draw samples from a noncentral chi-square distribution. Mirrors :meth:`numpy.random.Generator.noncentral_chisquare`."""
        return self._fill("noncentral_chisquare", (df, nonc), {}, shape, **kwargs)

    def noncentral_f(self, dfnum, dfden, nonc, *, shape, **kwargs: Any) -> blosc2.NDArray:
        """Draw samples from a noncentral F distribution. Mirrors :meth:`numpy.random.Generator.noncentral_f`."""
        return self._fill("noncentral_f", (dfnum, dfden, nonc), {}, shape, **kwargs)

    def pareto(self, a, *, shape, **kwargs: Any) -> blosc2.NDArray:
        """Draw samples from a Pareto distribution. Mirrors :meth:`numpy.random.Generator.pareto`."""
        return self._fill("pareto", (a,), {}, shape, **kwargs)

    def poisson(self, lam=1.0, *, shape, **kwargs: Any) -> blosc2.NDArray:
        """Draw samples from a Poisson distribution. Mirrors :meth:`numpy.random.Generator.poisson`."""
        return self._fill("poisson", (lam,), {}, shape, **kwargs)

    def power(self, a, *, shape, **kwargs: Any) -> blosc2.NDArray:
        """Draw samples from a power distribution. Mirrors :meth:`numpy.random.Generator.power`."""
        return self._fill("power", (a,), {}, shape, **kwargs)

    def rayleigh(self, scale=1.0, *, shape, **kwargs: Any) -> blosc2.NDArray:
        """Draw samples from a Rayleigh distribution. Mirrors :meth:`numpy.random.Generator.rayleigh`."""
        return self._fill("rayleigh", (scale,), {}, shape, **kwargs)

    def standard_cauchy(self, *, shape, **kwargs: Any) -> blosc2.NDArray:
        """Draw samples from a standard Cauchy distribution. Mirrors :meth:`numpy.random.Generator.standard_cauchy`."""
        return self._fill("standard_cauchy", (), {}, shape, **kwargs)

    def standard_exponential(
        self, *, shape, dtype=np.float64, method="zig", **kwargs: Any
    ) -> blosc2.NDArray:
        """Draw samples from a standard exponential distribution.

        Mirrors :meth:`numpy.random.Generator.standard_exponential`.
        """
        return self._fill("standard_exponential", (), {"dtype": dtype, "method": method}, shape, **kwargs)

    def standard_gamma(self, shape_param, /, *, shape, dtype=np.float64, **kwargs: Any) -> blosc2.NDArray:
        """Draw samples from a standard gamma distribution.

        Mirrors :meth:`numpy.random.Generator.standard_gamma`; see :meth:`gamma` for why the
        distribution's ``shape`` parameter is renamed ``shape_param`` here.
        """
        return self._fill("standard_gamma", (shape_param,), {"dtype": dtype}, shape, **kwargs)

    def standard_normal(self, *, shape, dtype=np.float64, **kwargs: Any) -> blosc2.NDArray:
        """Draw samples from a standard normal distribution.

        Mirrors :meth:`numpy.random.Generator.standard_normal`.
        """
        return self._fill("standard_normal", (), {"dtype": dtype}, shape, **kwargs)

    def standard_t(self, df, *, shape, **kwargs: Any) -> blosc2.NDArray:
        """Draw samples from a standard Student's t distribution. Mirrors :meth:`numpy.random.Generator.standard_t`."""
        return self._fill("standard_t", (df,), {}, shape, **kwargs)

    def triangular(self, left, mode, right, *, shape, **kwargs: Any) -> blosc2.NDArray:
        """Draw samples from a triangular distribution. Mirrors :meth:`numpy.random.Generator.triangular`."""
        return self._fill("triangular", (left, mode, right), {}, shape, **kwargs)

    def vonmises(self, mu, kappa, *, shape, **kwargs: Any) -> blosc2.NDArray:
        """Draw samples from a von Mises distribution. Mirrors :meth:`numpy.random.Generator.vonmises`."""
        return self._fill("vonmises", (mu, kappa), {}, shape, **kwargs)

    def wald(self, mean, scale, *, shape, **kwargs: Any) -> blosc2.NDArray:
        """Draw samples from a Wald distribution. Mirrors :meth:`numpy.random.Generator.wald`."""
        return self._fill("wald", (mean, scale), {}, shape, **kwargs)

    def weibull(self, a, *, shape, **kwargs: Any) -> blosc2.NDArray:
        """Draw samples from a Weibull distribution. Mirrors :meth:`numpy.random.Generator.weibull`."""
        return self._fill("weibull", (a,), {}, shape, **kwargs)

    def zipf(self, a, *, shape, **kwargs: Any) -> blosc2.NDArray:
        """Draw samples from a Zipf distribution. Mirrors :meth:`numpy.random.Generator.zipf`."""
        return self._fill("zipf", (a,), {}, shape, **kwargs)

    def choice(self, a, *, shape, p=None, replace: bool = True, **kwargs: Any) -> blosc2.NDArray:
        """Draw samples from ``a`` with replacement. Mirrors :meth:`numpy.random.Generator.choice`.

        Only ``replace=True`` (numpy's default) and 1-D or scalar-int ``a`` are
        supported: sampling without replacement, or along an axis of a multi-dimensional
        ``a``, needs whole-array coordination across chunks that this module doesn't do.
        """
        if not replace:
            raise NotImplementedError(
                "blosc2.random.Generator.choice only supports replace=True: sampling "
                "without replacement needs whole-array coordination across chunks"
            )
        if np.ndim(a) > 1:
            raise NotImplementedError("blosc2.random.Generator.choice only supports 1-D or scalar `a`")
        return self._fill("choice", (a,), {"p": p, "replace": True}, shape, **kwargs)

    def dirichlet(self, alpha, *, shape, **kwargs: Any) -> blosc2.NDArray:
        """Draw samples from a Dirichlet distribution. Mirrors :meth:`numpy.random.Generator.dirichlet`.

        Output shape is ``shape + (len(alpha),)``; see :meth:`_fill_vector` — the
        trailing dimension holds one full draw and is never split across chunks.
        """
        return self._fill_vector("dirichlet", (alpha,), {}, shape, len(alpha), **kwargs)

    def multinomial(self, n, pvals, *, shape, **kwargs: Any) -> blosc2.NDArray:
        """Draw samples from a multinomial distribution. Mirrors :meth:`numpy.random.Generator.multinomial`.

        Output shape is ``shape + (len(pvals),)``; see :meth:`dirichlet` for the
        trailing-dimension note.
        """
        return self._fill_vector("multinomial", (n, pvals), {}, shape, len(pvals), **kwargs)

    def multivariate_hypergeometric(self, colors, nsample, *, shape, **kwargs: Any) -> blosc2.NDArray:
        """Draw samples from a multivariate hypergeometric distribution.

        Mirrors :meth:`numpy.random.Generator.multivariate_hypergeometric`. Output shape is
        ``shape + (len(colors),)``; see :meth:`dirichlet` for the trailing-dimension note.
        """
        return self._fill_vector(
            "multivariate_hypergeometric", (colors, nsample), {}, shape, len(colors), **kwargs
        )

    def multivariate_normal(self, mean, cov, *, shape, **kwargs: Any) -> blosc2.NDArray:
        """Draw samples from a multivariate normal distribution.

        Mirrors :meth:`numpy.random.Generator.multivariate_normal`. Output shape is
        ``shape + (len(mean),)``; see :meth:`dirichlet` for the trailing-dimension note.
        """
        return self._fill_vector("multivariate_normal", (mean, cov), {}, shape, len(mean), **kwargs)

    def permutation(self, x, *, axis: int = 0, **kwargs: Any) -> blosc2.NDArray:
        """Return a shuffled copy of ``x`` (or of ``arange(x)`` if ``x`` is an int).

        Mirrors :meth:`numpy.random.Generator.permutation`.

        Unlike the rest of this module, permuting is inherently sequential: it loads
        ``x`` fully into memory and shuffles it single-threaded, rather than generating
        chunks independently in parallel.
        """
        rng = np.random.default_rng(self._root.spawn(1)[0])
        arr = np.arange(x) if isinstance(x, int | np.integer) else _materialize(x)
        return blosc2.asarray(rng.permutation(arr, axis=axis), **kwargs)

    def permuted(self, x, *, axis: int | None = None, **kwargs: Any) -> blosc2.NDArray:
        """Return a copy of ``x`` with elements permuted along ``axis`` (flattened if ``None``).

        Mirrors :meth:`numpy.random.Generator.permuted`. See :meth:`permutation` for the
        single-threaded, full-materialization caveat.
        """
        rng = np.random.default_rng(self._root.spawn(1)[0])
        return blosc2.asarray(rng.permuted(_materialize(x), axis=axis), **kwargs)

    def shuffle(self, x: blosc2.NDArray, *, axis: int = 0) -> None:
        """Shuffle ``x`` in place along ``axis``. Mirrors :meth:`numpy.random.Generator.shuffle`.

        ``x`` must be a :class:`~blosc2.NDArray`: its full contents are read into memory,
        shuffled single-threaded (see :meth:`permutation`), and written back. Returns
        ``None``, matching numpy.
        """
        if not isinstance(x, blosc2.NDArray):
            raise TypeError("blosc2.random.Generator.shuffle requires a blosc2.NDArray to shuffle in place")
        rng = np.random.default_rng(self._root.spawn(1)[0])
        arr = x[:]
        rng.shuffle(arr, axis=axis)
        x[:] = arr


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
