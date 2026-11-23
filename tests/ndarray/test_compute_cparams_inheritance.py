"""Tests for guarded operand cparams inheritance in LazyArray.compute()."""

import numpy as np
import pytest

import blosc2


def _active_filters(cparams):
    return [f for f in cparams.filters if f != blosc2.Filter.NOFILTER]


def test_compute_inherits_uniform_ramp_cparams():
    a = blosc2.linspace(0, 1, 1024)
    assert a.cparams.codec == blosc2.Codec.BLOSCLZ
    assert _active_filters(a.cparams) == [blosc2.Filter.SHUFFLE, blosc2.Filter.BYTEDELTA]

    res = blosc2.sin(a).compute()
    assert res.cparams.codec == blosc2.Codec.BLOSCLZ
    assert res.cparams.clevel == 1
    assert _active_filters(res.cparams) == [blosc2.Filter.SHUFFLE, blosc2.Filter.BYTEDELTA]
    np.testing.assert_allclose(res[:], np.sin(a[:]))

    # Scalar operands and repeated references keep uniformity.
    res2 = (blosc2.sin(a) + blosc2.cos(a) * 2.0).compute()
    assert res2.cparams.codec == blosc2.Codec.BLOSCLZ
    assert _active_filters(res2.cparams) == [blosc2.Filter.SHUFFLE, blosc2.Filter.BYTEDELTA]


def test_compute_inherits_uniform_custom_lossless_cparams():
    cp = {"codec": blosc2.Codec.LZ4, "clevel": 3, "filters": [blosc2.Filter.BITSHUFFLE]}
    a = blosc2.asarray(np.arange(1024, dtype=np.float64), cparams=cp)
    b = blosc2.asarray(np.arange(1024, dtype=np.float64) * 0.5, cparams=cp)

    res = (a + b).compute()
    assert res.cparams.codec == blosc2.Codec.LZ4
    assert res.cparams.clevel == 3
    assert _active_filters(res.cparams) == [blosc2.Filter.BITSHUFFLE]
    np.testing.assert_allclose(res[:], a[:] + b[:])


def test_compute_inherits_in_menudet_and_lazyudf():
    a = blosc2.linspace(0, 1, 256)

    @blosc2.menudet
    def scaled(x):
        return x * 2.0

    res_menudet = scaled(a).compute()
    assert res_menudet.cparams.codec == blosc2.Codec.BLOSCLZ
    assert _active_filters(res_menudet.cparams) == [blosc2.Filter.SHUFFLE, blosc2.Filter.BYTEDELTA]
    np.testing.assert_allclose(res_menudet[:], a[:] * 2.0)

    def udf_double(inputs, output, offset):
        output[:] = inputs[0] * 2.0

    res_udf = blosc2.lazyudf(udf_double, (a,), dtype=a.dtype).compute()
    assert res_udf.cparams.codec == blosc2.Codec.BLOSCLZ
    assert _active_filters(res_udf.cparams) == [blosc2.Filter.SHUFFLE, blosc2.Filter.BYTEDELTA]
    np.testing.assert_allclose(res_udf[:], a[:] * 2.0)


def test_compute_mixed_or_numpy_operands_use_defaults():
    a = blosc2.linspace(0, 1, 256)
    b = blosc2.asarray(np.ones(256, dtype=np.float64))  # default ZSTD + SHUFFLE

    res_mixed = (a + b).compute()
    assert res_mixed.cparams.codec == blosc2.Codec.ZSTD
    assert _active_filters(res_mixed.cparams) == [blosc2.Filter.SHUFFLE]

    res_numpy = (a + np.ones(256, dtype=np.float64)).compute()
    assert res_numpy.cparams.codec == blosc2.Codec.ZSTD
    assert _active_filters(res_numpy.cparams) == [blosc2.Filter.SHUFFLE]


def test_compute_dtype_kind_change_uses_defaults():
    a = blosc2.linspace(0, 1, 256)
    mask = (a > 0.5).compute()
    assert mask.dtype == np.bool_
    assert mask.cparams.codec == blosc2.Codec.ZSTD
    assert _active_filters(mask.cparams) == [blosc2.Filter.SHUFFLE]

    ints = blosc2.arange(256, dtype=np.int64)
    floats = blosc2.sin(ints).compute()
    assert floats.dtype == np.float64
    assert floats.cparams.codec == blosc2.Codec.ZSTD
    assert _active_filters(floats.cparams) == [blosc2.Filter.SHUFFLE]


def test_compute_explicit_and_global_cparams_override_inheritance(monkeypatch):
    a = blosc2.linspace(0, 1, 256)

    res_dict = blosc2.sin(a).compute(cparams={"codec": blosc2.Codec.ZSTD})
    assert res_dict.cparams.codec == blosc2.Codec.ZSTD
    assert _active_filters(res_dict.cparams) == [blosc2.Filter.SHUFFLE]

    res_obj = blosc2.sin(a).compute(cparams=blosc2.CParams())
    assert res_obj.cparams.codec == blosc2.Codec.ZSTD

    monkeypatch.setitem(blosc2.cparams_dflts, "codec", blosc2.Codec.LZ4)
    res_global = blosc2.sin(a).compute()
    assert res_global.cparams.codec == blosc2.Codec.LZ4
    assert _active_filters(res_global.cparams) == [blosc2.Filter.SHUFFLE]


@pytest.mark.parametrize(
    "bad_cparams",
    [
        {"codec": blosc2.Codec.BLOSCLZ, "clevel": 0},
        {"codec": blosc2.Codec.ZSTD, "clevel": 3, "use_dict": True},
        {
            "codec": blosc2.Codec.BLOSCLZ,
            "clevel": 2,
            "filters": [blosc2.Filter.TRUNC_PREC, blosc2.Filter.SHUFFLE],
            "filters_meta": [10, 0],
        },
        {"codec": blosc2.Codec.ZFP_RATE, "codec_meta": 37, "filters": []},
    ],
)
def test_compute_does_not_inherit_uncompressed_dict_or_lossy_cparams(bad_cparams):
    a = blosc2.asarray(np.linspace(0, 1, 256, dtype=np.float64), cparams=bad_cparams)
    res = (a + 1.0).compute()
    assert res.cparams.codec == blosc2.Codec.ZSTD
    assert res.cparams.clevel == 1
    assert not res.cparams.use_dict
    assert _active_filters(res.cparams) == [blosc2.Filter.SHUFFLE]


def test_compute_does_not_inherit_runtime_nthreads_or_typesize():
    a = blosc2.linspace(0, 1, 256, dtype=np.float32, cparams={"nthreads": 1})
    b = blosc2.linspace(0, 1, 256, dtype=np.float64, cparams={"nthreads": 1})
    assert a.cparams.typesize == 4
    assert a.cparams.nthreads == 1
    res = (a + b).compute()
    assert res.dtype == np.float64
    assert res.cparams.codec == blosc2.Codec.BLOSCLZ
    assert _active_filters(res.cparams) == [blosc2.Filter.SHUFFLE, blosc2.Filter.BYTEDELTA]
    assert res.cparams.typesize == 8
    assert res.cparams.nthreads == blosc2.nthreads
