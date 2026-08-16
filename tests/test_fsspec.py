#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

import numpy as np
import pytest

import blosc2

fsspec = pytest.importorskip("fsspec")


@pytest.fixture(autouse=True)
def clean_memory_fs():
    fsspec.filesystem("memory").store.clear()


def test_open_memory_url():
    a = blosc2.arange(10, dtype="i4")
    with fsspec.open("memory://x.b2nd", "wb") as f:
        f.write(a.to_cframe())

    b = blosc2.open("memory://x.b2nd")
    assert isinstance(b, blosc2.NDArray)
    assert np.array_equal(b[:], a[:])


def test_save_array_to_url():
    a = np.arange(100, dtype="f8").reshape(10, 10)
    nbytes = blosc2.save_array(a, "memory://y.b2nd")
    assert nbytes > 0
    assert np.array_equal(blosc2.load_array("memory://y.b2nd"), a)


def test_save_tensor_to_url():
    a = np.arange(50, dtype="f4")
    blosc2.save_tensor(a, "memory://z.b2nd")
    assert np.array_equal(blosc2.load_tensor("memory://z.b2nd"), a)


def test_schunk_roundtrip():
    schunk = blosc2.SChunk(chunksize=1000)
    schunk.append_data(np.arange(1000, dtype="u1"))
    with fsspec.open("memory://s.b2f", "wb") as f:
        f.write(schunk.to_cframe())

    sc = blosc2.open("memory://s.b2f")
    assert isinstance(sc, blosc2.SChunk)
    assert sc.nbytes == schunk.nbytes


def test_chained_url(tmp_path):
    # A container inside a local zip, reached through fsspec's chained syntax
    import zipfile

    a = blosc2.arange(20, dtype="i2")
    zippath = tmp_path / "archive.zip"
    with zipfile.ZipFile(zippath, "w") as zf:
        zf.writestr("inner.b2nd", a.to_cframe())

    b = blosc2.open(f"zip://inner.b2nd::file://{zippath}")
    assert np.array_equal(b[:], a[:])


@pytest.mark.parametrize("mode", ["a", "w"])
def test_mode_not_supported(mode):
    with pytest.raises(NotImplementedError):
        blosc2.open("memory://x.b2nd", mode=mode)


def test_offset_not_supported():
    with pytest.raises(NotImplementedError):
        blosc2.open("memory://x.b2nd", offset=32)


def test_dir_container_not_supported():
    with pytest.raises(NotImplementedError):
        blosc2.open("memory://store.b2d")


def test_unknown_protocol():
    # fsspec owns this error; we only check that we do not swallow it into a
    # misleading FileNotFoundError
    with pytest.raises(ValueError):
        blosc2.open("nosuchproto://bucket/key.b2nd")


def test_http_still_goes_to_c2array():
    # http(s) is reserved for Caterva2, so it must not reach fsspec
    with pytest.raises(FileNotFoundError):
        blosc2.open("http://localhost:1/foo.b2nd")


def test_local_path_untouched(tmp_path):
    urlpath = str(tmp_path / "local.b2nd")
    a = blosc2.arange(10, dtype="i4", urlpath=urlpath, mode="w")
    assert np.array_equal(blosc2.open(urlpath)[:], a[:])
