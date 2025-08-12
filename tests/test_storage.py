#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

from dataclasses import asdict, fields

import numpy as np
import pytest

import blosc2


@pytest.mark.parametrize(
    ("urlpath", "contiguous", "mode", "mmap_mode"),
    [
        (None, None, "w", None),
        (None, False, "a", None),
        (None, None, "r", None),
        (None, True, "a", None),
        ("b2frame", None, "r", None),
        ("b2frame", False, "a", None),
        ("b2frame", True, "w", None),
        ("b2frame", True, "r", "r"),
        ("b2frame", None, "w", "w+"),
    ],
)
def test_storage_values(contiguous, urlpath, mode, mmap_mode):
    storage = blosc2.Storage(contiguous=contiguous, urlpath=urlpath, mode=mode, mmap_mode=mmap_mode)
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


def test_storage_defaults():
    storage = blosc2.Storage()
    assert storage.contiguous is False
    assert storage.urlpath is None
    assert storage.mode == "a"
    assert storage.mmap_mode is None
    assert storage.initial_mapping_size is None
    assert storage.meta is None


@pytest.mark.parametrize(
    ("urlpath", "contiguous"),
    [
        (None, False),
        (None, True),
        ("b2frame", False),
        ("b2frame", True),
    ],
)
def test_raises_storage(contiguous, urlpath):
    storage = blosc2.Storage(contiguous=contiguous, urlpath=urlpath)
    blosc2.remove_urlpath(urlpath)

    for field in fields(blosc2.Storage):
        with pytest.raises(AttributeError):
            _ = blosc2.SChunk(storage=storage, **{str(field.name): {}})
        with pytest.raises(TypeError):
            _ = blosc2.SChunk(**{str(field.name): {}}, **asdict(storage))

        with pytest.raises(AttributeError):
            _ = blosc2.empty((30, 30), storage=storage, **{str(field.name): {}})
        with pytest.raises(TypeError):
            _ = blosc2.empty((30, 30), **{str(field.name): {}}, **asdict(storage))


@pytest.mark.parametrize(
    "cparams",
    [
        blosc2.CParams(codec=blosc2.Codec.LZ4, filters=[blosc2.Filter.BITSHUFFLE]),
        {"typesize": 4, "filters": [blosc2.Filter.TRUNC_PREC, blosc2.Filter.DELTA], "filters_meta": [0, 0]},
        blosc2.CParams(
            nthreads=5, filters=[blosc2.Filter.BITSHUFFLE, blosc2.Filter.BYTEDELTA], filters_meta=[0] * 3
        ),
        {"codec": blosc2.Codec.LZ4HC, "typesize": 4, "filters": [blosc2.Filter.BYTEDELTA]},
    ],
)
def test_cparams_values(cparams):
    schunk = blosc2.SChunk(cparams=cparams)
    cparams_dataclass = cparams if isinstance(cparams, blosc2.CParams) else blosc2.CParams(**cparams)
    for field in fields(cparams_dataclass):
        if field.name in ["filters", "filters_meta"]:
            assert getattr(schunk.cparams, field.name)[
                : len(getattr(cparams_dataclass, field.name))
            ] == getattr(cparams_dataclass, field.name)
        else:
            assert getattr(schunk.cparams, field.name) == getattr(cparams_dataclass, field.name)

    array = blosc2.empty((30, 30), np.int32, cparams=cparams)
    for field in fields(cparams_dataclass):
        if field.name in ["filters", "filters_meta"]:
            assert getattr(array.schunk.cparams, field.name)[
                : len(getattr(cparams_dataclass, field.name))
            ] == getattr(cparams_dataclass, field.name)
        elif field.name == "typesize":
            assert getattr(array.schunk.cparams, field.name) == array.dtype.itemsize
        elif field.name != "blocksize":
            assert getattr(array.schunk.cparams, field.name) == getattr(cparams_dataclass, field.name)

    blosc2.set_nthreads(10)
    schunk = blosc2.SChunk(cparams=cparams)
    cparams_dataclass = cparams if isinstance(cparams, blosc2.CParams) else blosc2.CParams(**cparams)
    assert schunk.cparams.nthreads == cparams_dataclass.nthreads

    array = blosc2.empty((30, 30), np.int32, cparams=cparams)
    assert array.schunk.cparams.nthreads == cparams_dataclass.nthreads


def test_cparams_defaults():
    cparams = blosc2.CParams()
    assert cparams.codec == blosc2.Codec.ZSTD
    assert cparams.codec_meta == 0
    assert cparams.splitmode == blosc2.SplitMode.AUTO_SPLIT
    assert cparams.clevel == 5
    assert cparams.typesize == 8
    assert cparams.nthreads == blosc2.nthreads
    assert cparams.filters == [blosc2.Filter.NOFILTER] * 5 + [blosc2.Filter.SHUFFLE]
    assert cparams.filters_meta == [0] * 6
    assert not cparams.use_dict
    assert cparams.blocksize == 0
    assert cparams.tuner == blosc2.Tuner.STUNE

    blosc2.set_nthreads(1)
    cparams = blosc2.CParams()
    assert cparams.nthreads == blosc2.nthreads


def test_raises_cparams():
    cparams = blosc2.CParams(codec=blosc2.Codec.LZ4, clevel=6, typesize=4)
    for field in fields(blosc2.CParams):
        with pytest.raises(ValueError):
            _ = blosc2.SChunk(cparams=cparams, **{str(field.name): {}})
        with pytest.raises(AttributeError):
            _ = blosc2.compress2(b"12345678" * 1000, cparams=cparams, **{str(field.name): {}})
        with pytest.raises(KeyError):
            _ = blosc2.empty((10, 10), cparams=cparams, **{str(field.name): {}})


@pytest.mark.parametrize(
    "dparams",
    [
        (blosc2.DParams()),
        (blosc2.DParams(nthreads=2)),
        ({}),
        ({"nthreads": 2}),
    ],
)
def test_dparams_values(dparams):
    schunk = blosc2.SChunk(dparams=dparams)
    dparams_dataclass = dparams if isinstance(dparams, blosc2.DParams) else blosc2.DParams(**dparams)
    array = blosc2.empty((30, 30), dparams=dparams)
    for field in fields(dparams_dataclass):
        assert getattr(schunk.dparams, field.name) == getattr(dparams_dataclass, field.name)
        assert getattr(array.schunk.dparams, field.name) == getattr(dparams_dataclass, field.name)

    blosc2.set_nthreads(3)
    schunk = blosc2.SChunk(dparams=dparams)
    dparams_dataclass = dparams if isinstance(dparams, blosc2.DParams) else blosc2.DParams(**dparams)
    array = blosc2.empty((30, 30), dparams=dparams)
    assert schunk.dparams.nthreads == dparams_dataclass.nthreads
    assert array.schunk.dparams.nthreads == dparams_dataclass.nthreads


def test_dparams_defaults():
    dparams = blosc2.DParams()
    assert dparams.nthreads == blosc2.nthreads

    blosc2.set_nthreads(1)
    dparams = blosc2.DParams()
    assert dparams.nthreads == blosc2.nthreads


def test_raises_dparams():
    dparams = blosc2.DParams()
    for field in fields(blosc2.DParams):
        with pytest.raises(ValueError):
            _ = blosc2.SChunk(dparams=dparams, **{str(field.name): {}})
        with pytest.raises(AttributeError):
            _ = blosc2.decompress2(b"12345678" * 1000, dparams=dparams, **{str(field.name): {}})
