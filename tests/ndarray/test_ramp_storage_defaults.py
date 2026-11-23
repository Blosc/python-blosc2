"""Constructor-specific compression policy, independent of ramp numerical semantics."""

import numpy as np
import pytest

import blosc2


def make_ramp(constructor, size=24, **kwargs):
    if constructor == "arange":
        return blosc2.arange(size, **kwargs)
    return blosc2.linspace(0, size - 1, size, **kwargs)


@pytest.mark.parametrize(
    ("constructor", "size", "dtype", "c_order"),
    [
        ("arange", 0, np.float64, True),
        ("linspace", 0, np.int16, False),
        ("arange", 24, np.float64, False),
        ("linspace", 24, np.float64, True),
        ("arange", 24, np.int16, True),
        ("linspace", 24, np.int16, False),
    ],
)
def test_ramp_default_filters(constructor, size, dtype, c_order):
    array = make_ramp(constructor, size, dtype=dtype, c_order=c_order)
    assert array.cparams.codec == blosc2.Codec.BLOSCLZ
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


@pytest.mark.parametrize(
    ("constructor", "filters"), [("arange", []), ("linspace", [blosc2.Filter.BITSHUFFLE])]
)
def test_ramp_explicit_filters(constructor, filters):
    cparams = {"filters": filters.copy()}
    array = make_ramp(constructor, cparams=cparams)
    active = [f for f in array.cparams.filters if f != blosc2.Filter.NOFILTER]
    assert active == filters
    assert cparams == {"filters": filters}


@pytest.mark.parametrize("constructor", ["arange", "linspace"])
@pytest.mark.parametrize(
    ("cparams", "filters"),
    [(None, []), ({}, [blosc2.Filter.SHUFFLE]), ({"codec": blosc2.Codec.LZ4}, [blosc2.Filter.BITSHUFFLE])],
)
def test_ramp_custom_global_filters(constructor, cparams, filters, monkeypatch):
    monkeypatch.setitem(blosc2.cparams_dflts, "filters", filters.copy())
    array = make_ramp(constructor, cparams=cparams)
    assert [f for f in array.cparams.filters if f != blosc2.Filter.NOFILTER] == filters
    assert blosc2.cparams_dflts["filters"] == filters
    np.testing.assert_array_equal(array[:], np.arange(24))
    if cparams is not None:
        assert "filters" not in cparams


@pytest.mark.parametrize("constructor", ["arange", "linspace"])
def test_ramp_global_filters_updated_between_calls(constructor, monkeypatch):
    filters = [blosc2.Filter.SHUFFLE, blosc2.Filter.BYTEDELTA]
    monkeypatch.setitem(blosc2.cparams_dflts, "filters", filters)
    array = make_ramp(constructor, size=1000, shape=(10, 10, 10))
    assert array.cparams.filters[:2] == filters
    monkeypatch.setitem(blosc2.cparams_dflts, "filters", [blosc2.Filter.SHUFFLE])
    array = make_ramp(constructor, size=1000, shape=(10, 10, 10))
    assert [f for f in array.cparams.filters if f != blosc2.Filter.NOFILTER] == [blosc2.Filter.SHUFFLE]


@pytest.mark.parametrize("constructor", ["arange", "linspace"])
def test_ramp_global_filters_inplace(constructor):
    filters = blosc2.cparams_dflts["filters"]
    original = filters.copy()
    try:
        filters[:] = [blosc2.Filter.SHUFFLE]
        array = make_ramp(constructor)
        assert [f for f in array.cparams.filters if f != blosc2.Filter.NOFILTER] == filters
    finally:
        filters[:] = original


@pytest.mark.parametrize("constructor", ["arange", "linspace"])
def test_ramp_global_filters_replaced_with_original_values(constructor, monkeypatch):
    filters = blosc2.cparams_dflts["filters"].copy()
    monkeypatch.setitem(blosc2.cparams_dflts, "filters", filters)
    array = make_ramp(constructor)
    assert array.cparams.filters == filters


@pytest.mark.parametrize("constructor", ["arange", "linspace"])
@pytest.mark.parametrize("complete", [False, True])
def test_ramp_explicit_filters_override_global(constructor, complete, monkeypatch):
    monkeypatch.setitem(blosc2.cparams_dflts, "filters", [blosc2.Filter.SHUFFLE])
    cparams = {"codec": blosc2.Codec.LZ4, "filters": [blosc2.Filter.BITSHUFFLE]}
    if complete:
        cparams = blosc2.CParams(**cparams)
    array = make_ramp(constructor, cparams=cparams)
    assert array.cparams.codec == blosc2.Codec.LZ4
    assert [f for f in array.cparams.filters if f != blosc2.Filter.NOFILTER] == [blosc2.Filter.BITSHUFFLE]


@pytest.mark.parametrize("constructor", ["arange", "linspace"])
def test_ramp_explicit_and_global_codec(constructor, monkeypatch):
    explicit = {"codec": blosc2.Codec.ZSTD}
    array = make_ramp(constructor, cparams=explicit)
    assert array.cparams.codec == blosc2.Codec.ZSTD
    assert array.cparams.filters[:2] == [blosc2.Filter.SHUFFLE, blosc2.Filter.BYTEDELTA]
    assert explicit == {"codec": blosc2.Codec.ZSTD}

    complete = blosc2.CParams()
    array = make_ramp(constructor, cparams=complete)
    assert array.cparams.codec == blosc2.Codec.ZSTD

    monkeypatch.setitem(blosc2.cparams_dflts, "codec", blosc2.Codec.LZ4)
    array = make_ramp(constructor)
    assert array.cparams.codec == blosc2.Codec.LZ4
    assert array.cparams.filters[:2] == [blosc2.Filter.SHUFFLE, blosc2.Filter.BYTEDELTA]


@pytest.mark.parametrize("constructor", ["arange", "linspace"])
def test_ramp_complex_reshape_defaults(constructor):
    array = make_ramp(constructor, dtype=np.complex128, shape=(4, 6))
    assert array.cparams.codec == blosc2.Codec.BLOSCLZ
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
        assert array.cparams.codec == blosc2.Codec.ZSTD
        assert array.cparams.filters[-1] == blosc2.Filter.SHUFFLE
        assert blosc2.Filter.BYTEDELTA not in array.cparams.filters
