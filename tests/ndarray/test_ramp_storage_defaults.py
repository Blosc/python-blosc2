"""Constructor-specific compression policy, independent of ramp numerical semantics."""

import numpy as np
import pytest

import blosc2


def make_ramp(constructor, size=24, **kwargs):
    if constructor == "arange":
        return blosc2.arange(size, **kwargs)
    return blosc2.linspace(0, size - 1, size, **kwargs)


@pytest.mark.parametrize("constructor", ["arange", "linspace"])
@pytest.mark.parametrize("size", [0, 24])
@pytest.mark.parametrize("dtype", [np.float64, np.int16, np.complex128])
@pytest.mark.parametrize("c_order", [True, False])
def test_ramp_default_filters(constructor, size, dtype, c_order):
    array = make_ramp(constructor, size, dtype=dtype, c_order=c_order)
    assert array.cparams.codec == blosc2.Codec.ZSTD
    assert array.cparams.filters[:2] == [blosc2.Filter.SHUFFLE, blosc2.Filter.BYTEDELTA]
    np.testing.assert_array_equal(array[:], np.arange(size, dtype=dtype))


@pytest.mark.parametrize("constructor", ["arange", "linspace"])
def test_ramp_partial_cparams_preserve_user_settings(constructor):
    cparams = {"codec": blosc2.Codec.LZ4, "clevel": 2}
    array = make_ramp(constructor, cparams=cparams)
    assert array.cparams.codec == blosc2.Codec.LZ4
    assert array.cparams.clevel == 2
    assert array.cparams.filters[:2] == [blosc2.Filter.SHUFFLE, blosc2.Filter.BYTEDELTA]
    assert cparams == {"codec": blosc2.Codec.LZ4, "clevel": 2}


@pytest.mark.parametrize("constructor", ["arange", "linspace"])
@pytest.mark.parametrize("filters", [[], [blosc2.Filter.BITSHUFFLE]])
def test_ramp_explicit_filters(constructor, filters):
    cparams = {"filters": filters.copy()}
    array = make_ramp(constructor, cparams=cparams)
    active = [f for f in array.cparams.filters if f != blosc2.Filter.NOFILTER]
    assert active == filters
    assert cparams == {"filters": filters}


@pytest.mark.parametrize("constructor", ["arange", "linspace"])
def test_ramp_complete_cparams(constructor):
    cparams = blosc2.CParams(codec=blosc2.Codec.LZ4, filters=[blosc2.Filter.SHUFFLE])
    array = make_ramp(constructor, cparams=cparams)
    assert array.cparams.codec == cparams.codec
    assert [f for f in array.cparams.filters if f != blosc2.Filter.NOFILTER] == cparams.filters


@pytest.mark.parametrize("constructor", ["arange", "linspace"])
def test_ramp_complex_reshape_defaults(constructor):
    array = make_ramp(constructor, dtype=np.complex128, shape=(4, 6))
    assert array.cparams.filters[:2] == [blosc2.Filter.SHUFFLE, blosc2.Filter.BYTEDELTA]
    np.testing.assert_array_equal(array[:], np.arange(24).reshape(4, 6))


@pytest.mark.parametrize("constructor", ["arange", "linspace"])
def test_ramp_partitions_unchanged(constructor):
    array = make_ramp(constructor, size=1_000_000)
    reference = make_ramp(constructor, size=1_000_000, cparams={"filters": [blosc2.Filter.SHUFFLE]})
    assert array.chunks == reference.chunks
    assert array.blocks == reference.blocks


@pytest.mark.parametrize("constructor", ["arange", "linspace"])
def test_ramp_lazy_storage_remains_deferred(constructor):
    array = make_ramp(constructor, lazy=True).compute(cparams={"filters": [blosc2.Filter.BITSHUFFLE]})
    assert [f for f in array.cparams.filters if f != blosc2.Filter.NOFILTER] == [blosc2.Filter.BITSHUFFLE]
    with pytest.raises(TypeError, match="Storage options"):
        make_ramp(constructor, lazy=True, cparams={})


def test_other_constructor_filters_unchanged():
    for array in (blosc2.zeros(24), blosc2.ones(24), blosc2.eye(4)):
        assert array.cparams.filters[-1] == blosc2.Filter.SHUFFLE
        assert blosc2.Filter.BYTEDELTA not in array.cparams.filters
