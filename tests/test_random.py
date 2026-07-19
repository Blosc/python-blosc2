#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

import numpy as np
import pytest

import blosc2


def test_reproducible_same_seed():
    a = blosc2.random.default_rng(42).random((100, 100), chunks=(17, 17))
    b = blosc2.random.default_rng(42).random((100, 100), chunks=(17, 17))
    np.testing.assert_array_equal(a[:], b[:])


def test_successive_calls_differ():
    rng = blosc2.random.default_rng(42)
    a = rng.random((1000,))
    b = rng.random((1000,))
    assert not np.array_equal(a[:], b[:])


def test_different_seeds_differ():
    a = blosc2.random.default_rng(1).random((1000,))
    b = blosc2.random.default_rng(2).random((1000,))
    assert not np.array_equal(a[:], b[:])


def test_seed_none_works():
    a = blosc2.random.default_rng(None).random((10,))
    assert a.shape == (10,)


@pytest.mark.parametrize("nthreads", [1, 2])
def test_random_distribution_sanity(nthreads, monkeypatch):
    monkeypatch.setattr(blosc2, "nthreads", nthreads)
    a = blosc2.random.default_rng(0).random((200, 200), chunks=(31, 31))[:]
    assert a.dtype == np.float64
    assert a.min() >= 0.0
    assert a.max() < 1.0
    assert abs(a.mean() - 0.5) < 0.05


def test_normal_distribution_sanity():
    a = blosc2.random.default_rng(0).normal(loc=5.0, scale=2.0, shape=(500, 500))[:]
    assert abs(a.mean() - 5.0) < 0.1
    assert abs(a.std() - 2.0) < 0.1


def test_integers_bounds_dtype_endpoint():
    a = blosc2.random.default_rng(0).integers(
        0, 10, shape=(1000,), dtype=np.int32, endpoint=True, chunks=(100,)
    )[:]
    assert a.dtype == np.int32
    assert a.min() >= 0
    assert a.max() <= 10


def test_uniform_bounds():
    a = blosc2.random.default_rng(0).uniform(-3.0, 3.0, shape=(1000,))[:]
    assert a.min() >= -3.0
    assert a.max() < 3.0


def test_blosc2_kwargs_passthrough(tmp_path):
    urlpath = tmp_path / "r.b2nd"
    a = blosc2.random.default_rng(0).random(
        (100, 100), chunks=(20, 20), cparams={"clevel": 5}, urlpath=str(urlpath)
    )
    assert a.chunks == (20, 20)
    assert a.schunk.cparams.clevel == 5
    assert urlpath.exists()
    b = blosc2.open(str(urlpath))
    np.testing.assert_array_equal(a[:], b[:])


def test_nd_shape():
    a = blosc2.random.default_rng(0).random((4, 5, 6), chunks=(2, 3, 3))
    assert a.shape == (4, 5, 6)


def test_shape_required():
    with pytest.raises(TypeError):
        blosc2.random.default_rng(0).normal()


def test_scheduling_independent_of_nthreads(monkeypatch):
    monkeypatch.setattr(blosc2, "nthreads", 1)
    serial = blosc2.random.default_rng(7).random((300, 300), chunks=(17, 23))[:]
    monkeypatch.setattr(blosc2, "nthreads", 8)
    parallel = blosc2.random.default_rng(7).random((300, 300), chunks=(17, 23))[:]
    np.testing.assert_array_equal(serial, parallel)


_DIST_CASES = {
    "beta": (2.0, 5.0),
    "binomial": (10, 0.3),
    "chisquare": (3.0,),
    "exponential": (2.0,),
    "f": (5.0, 2.0),
    "gamma": (2.0, 3.0),
    "geometric": (0.3,),
    "gumbel": (0.0, 1.0),
    "hypergeometric": (10, 10, 5),
    "laplace": (0.0, 1.0),
    "logistic": (0.0, 1.0),
    "lognormal": (0.0, 1.0),
    "logseries": (0.5,),
    "negative_binomial": (5, 0.5),
    "noncentral_chisquare": (3.0, 2.0),
    "noncentral_f": (5.0, 2.0, 1.0),
    "pareto": (3.0,),
    "poisson": (4.0,),
    "power": (2.0,),
    "rayleigh": (1.0,),
    "standard_cauchy": (),
    "standard_exponential": (),
    "standard_gamma": (2.0,),
    "standard_normal": (),
    "standard_t": (5.0,),
    "triangular": (0.0, 0.5, 1.0),
    "vonmises": (0.0, 4.0),
    "wald": (1.0, 1.0),
    "weibull": (1.5,),
    "zipf": (2.0,),
}


@pytest.mark.parametrize(("method", "args"), _DIST_CASES.items(), ids=_DIST_CASES.keys())
def test_distribution_reproducible_and_finite(method, args):
    def draw():
        rng = getattr(blosc2.random.default_rng(0), method)
        return rng(*args, shape=(200,), chunks=(37,))[:]

    a, b = draw(), draw()
    np.testing.assert_array_equal(a, b)
    assert np.all(np.isfinite(a))


def test_poisson_mean_sanity():
    a = blosc2.random.default_rng(0).poisson(4.0, shape=(5000,))[:]
    assert abs(a.mean() - 4.0) < 0.2


def test_standard_normal_dtype():
    a = blosc2.random.default_rng(0).standard_normal(shape=(10,), dtype=np.float32)
    assert a.dtype == np.float32
