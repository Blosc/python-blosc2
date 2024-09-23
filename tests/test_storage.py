#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

from dataclasses import asdict, fields

import pytest

import blosc2


@pytest.mark.parametrize(
    "urlpath, contiguous, mode, mmap_mode, cparams, dparams",
    [
        (None, None, "w", None, blosc2.CParams(codec=blosc2.Codec.LZ4, clevel=6, typesize=4), blosc2.DParams()),
        (None, False, "a", None, {"typesize": 4}, blosc2.DParams()),
        (None, None, "r", None, blosc2.CParams(codec=blosc2.Codec.LZ4, clevel=6, typesize=4), blosc2.DParams(nthreads=4)),
        (None, True, "a", None, blosc2.CParams(splitmode=blosc2.SplitMode.ALWAYS_SPLIT, nthreads=5, typesize=4), {}),
        ("b2frame", None, "r", None, {"codec": blosc2.Codec.LZ4HC, "typesize": 4}, blosc2.DParams()),
        ("b2frame", False, "a", None, blosc2.CParams(codec=blosc2.Codec.LZ4, clevel=6, typesize=4), blosc2.DParams(nthreads=4)),
        ("b2frame", True, "w", None, blosc2.CParams(splitmode=blosc2.SplitMode.ALWAYS_SPLIT, nthreads=5, typesize=4), {}),
        ("b2frame", True, "r", "r", blosc2.CParams(codec=blosc2.Codec.LZ4, clevel=6, typesize=4), blosc2.DParams()),
        ("b2frame", None, "w", "w+", {"typesize": 4}, {}),
    ],
)
def test_storage_values(contiguous, urlpath, mode, mmap_mode, cparams, dparams):
    storage = blosc2.Storage(contiguous=contiguous, urlpath=urlpath, mode=mode, mmap_mode=mmap_mode,
                             cparams=cparams, dparams=dparams)
    if contiguous is None:
        if urlpath is not None:
            assert storage.contiguous
        else:
            assert not storage.contiguous
    else:
        assert storage.contiguous == contiguous

    assert storage.urlpath == urlpath
    assert storage.mode == mode
    assert storage.mmap_mode == mmap_mode
    assert storage.cparams == cparams
    assert storage.dparams == dparams


def test_storage_defaults():
    storage = blosc2.Storage()
    assert storage.contiguous == False
    assert storage.urlpath is None
    assert storage.mode == "a"
    assert storage.mmap_mode is None
    assert storage.initial_mapping_size is None
    assert storage.cparams == blosc2.CParams()
    assert storage.dparams == blosc2.DParams()
    assert storage.meta is None


@pytest.mark.parametrize(
    "urlpath, contiguous, cparams, dparams",
    [
        (None, False, blosc2.CParams(codec=blosc2.Codec.LZ4, clevel=6, typesize=4), blosc2.DParams()),
        (None, True, {"typesize": 4}, blosc2.DParams(nthreads=4)),
        ("b2frame", False, blosc2.CParams(splitmode=blosc2.SplitMode.ALWAYS_SPLIT, nthreads=5, typesize=4), {}),
        ("b2frame", True, {"codec": blosc2.Codec.LZ4HC, "typesize": 4}, {}),
    ],
)
def test_raises_storage(contiguous, urlpath, cparams, dparams):
    storage = blosc2.Storage(contiguous=contiguous, urlpath=urlpath,
                             cparams=cparams, dparams=dparams)
    blosc2.remove_urlpath(urlpath)

    for field in fields(blosc2.Storage):
        with pytest.raises(AttributeError):
            _ = blosc2.SChunk(storage=storage, **{str(field.name): {}})
        with pytest.raises(TypeError):
            _ = blosc2.SChunk(**{str(field.name): {}}, **asdict(storage))


@pytest.mark.parametrize(
    "cparams",
    [
        blosc2.CParams(codec=blosc2.Codec.LZ4, filters=[blosc2.Filter.BITSHUFFLE], tuner=blosc2.Tuner.BTUNE),
        {"typesize": 4, 'filters': [blosc2.Filter.TRUNC_PREC, blosc2.Filter.DELTA], 'filters_meta': [0, 0]},
        blosc2.CParams(nthreads=5, filters=[blosc2.Filter.BITSHUFFLE, blosc2.Filter.BYTEDELTA], filters_meta=[0] * 3),
        {"codec": blosc2.Codec.LZ4HC, "typesize": 4, 'filters': [blosc2.Filter.BYTEDELTA], 'tuner': blosc2.Tuner.BTUNE},
    ],
)
def test_cparams_values(cparams):
    schunk = blosc2.SChunk(cparams=cparams)
    cparams_dataclass = cparams if isinstance(cparams, blosc2.CParams) else blosc2.CParams(**cparams)

    for field in fields(cparams_dataclass):
        if field.name in ['filters', 'filters_meta']:
            assert getattr(schunk.cparams, field.name)[:len(getattr(cparams_dataclass, field.name))] == getattr(cparams_dataclass, field.name)
        else:
            assert getattr(schunk.cparams, field.name) == getattr(cparams_dataclass, field.name)


def test_cparams_defaults():
    cparams = blosc2.CParams()
    assert cparams.codec == blosc2.Codec.ZSTD
    assert cparams.codec_meta == 0
    assert cparams.splitmode == blosc2.SplitMode.ALWAYS_SPLIT
    assert cparams.clevel == 1
    assert cparams.typesize == 8
    assert cparams.nthreads == blosc2.nthreads
    assert cparams.filters == [blosc2.Filter.NOFILTER] * 5 + [blosc2.Filter.SHUFFLE]
    assert cparams.filters_meta == [0] * 6
    assert not cparams.use_dict
    assert cparams.blocksize == 0
    assert cparams.tuner == blosc2.Tuner.STUNE


def test_raises_cparams():
    cparams = blosc2.CParams(codec=blosc2.Codec.LZ4, clevel=6, typesize=4)
    for field in fields(blosc2.CParams):
        with pytest.raises(ValueError):
            _ = blosc2.SChunk(cparams=cparams, **{str(field.name): {}})
        with pytest.raises(AttributeError):
            _ = blosc2.compress2(b"12345678" * 1000, cparams=cparams, **{str(field.name): {}})


@pytest.mark.parametrize(
    "dparams",
    [
        (blosc2.DParams()),
        (blosc2.DParams(nthreads=2)),
        ({}),
        ({'nthreads': 2}),
    ],
)
def test_dparams_values(dparams):
    schunk = blosc2.SChunk(dparams=dparams)
    dparams_dataclass = dparams if isinstance(dparams, blosc2.DParams) else blosc2.DParams(**dparams)

    for field in fields(dparams_dataclass):
        assert getattr(schunk.dparams, field.name) == getattr(dparams_dataclass, field.name)


def test_dparams_defaults():
    dparams = blosc2.DParams()
    assert dparams.nthreads == blosc2.nthreads


def test_raises_dparams():
    dparams = blosc2.DParams()
    for field in fields(blosc2.DParams):
        with pytest.raises(ValueError):
            _ = blosc2.SChunk(dparams=dparams, **{str(field.name): {}})
        with pytest.raises(AttributeError):
            _ = blosc2.decompress2(b"12345678" * 1000, dparams=dparams, **{str(field.name): {}})
