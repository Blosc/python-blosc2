#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

from __future__ import annotations

import numpy as np
import pytest

import blosc2
from blosc2.cli.b2nd_to_zarr import b2nd_to_zarr, blosc2_to_zarr, main

zarr = pytest.importorskip("zarr")


def test_b2nd_to_zarr_basic(tmp_path):
    src = tmp_path / "data.b2nd"
    dst = tmp_path / "data.zarr"

    shape = (100, 50)
    chunks = (20, 25)
    data = np.arange(5000, dtype=np.float32).reshape(shape)
    a = blosc2.asarray(data, chunks=chunks, urlpath=str(src), mode="w")
    a.schunk.vlmeta["custom_key"] = "custom_value"

    summary = b2nd_to_zarr(src, dst, verify=True)
    assert summary["shape"] == shape
    assert summary["dtype"] == np.float32
    assert summary["chunks"] == chunks
    assert dst.exists()

    z = zarr.open(store=str(dst), mode="r")
    assert z.shape == shape
    assert z.dtype == np.float32
    np.testing.assert_array_equal(z[:], data)
    assert z.attrs.get("custom_key") == "custom_value"


def test_b2nd_to_zarr_custom_chunks_and_codec(tmp_path):
    src = tmp_path / "cube.b2nd"
    dst = tmp_path / "cube.zarr"

    shape = (30, 40, 50)
    data = np.arange(np.prod(shape), dtype=np.int32).reshape(shape)
    blosc2.asarray(data, chunks=(10, 10, 10), urlpath=str(src), mode="w")

    new_chunks = (15, 20, 25)
    summary = b2nd_to_zarr(
        src,
        dst,
        chunks=new_chunks,
        codec="lz4",
        clevel=3,
        shuffle="shuffle",
        full_verify=True,
    )
    assert summary["chunks"] == new_chunks

    z = zarr.open(store=str(dst), mode="r")
    assert z.chunks == new_chunks
    np.testing.assert_array_equal(z[:], data)


def test_b2nd_to_zarr_sharded(tmp_path):
    src = tmp_path / "sharded.b2nd"
    dst = tmp_path / "sharded.zarr"

    shape = (40, 40)
    data = np.arange(1600, dtype=np.int64).reshape(shape)
    blosc2.asarray(data, chunks=(20, 20), blocks=(5, 5), urlpath=str(src), mode="w")

    summary = b2nd_to_zarr(src, dst, sharded=True, verify=True)
    assert summary["shards"] == (20, 20)
    assert summary["chunks"] == (5, 5)

    z = zarr.open(store=str(dst), mode="r")
    assert z.shards == (20, 20)
    assert z.chunks == (5, 5)
    np.testing.assert_array_equal(z[:], data)


def test_b2nd_to_zarr_zip_store(tmp_path):
    src = tmp_path / "arr.b2nd"
    dst = tmp_path / "arr.zarr.zip"

    data = np.linspace(0, 1, 1000, dtype=np.float64)
    blosc2.asarray(data, chunks=(100,), urlpath=str(src), mode="w")

    summary = blosc2_to_zarr(src, dst, verify=True)
    assert dst.exists()
    assert summary["dst_path"] == dst

    from zarr.storage import ZipStore

    store = ZipStore(str(dst), mode="r")
    z = zarr.open(store=store)
    np.testing.assert_allclose(z[:], data)
    store.close()


def test_b2nd_to_zarr_overwrite_protection(tmp_path):
    src = tmp_path / "data.b2nd"
    dst = tmp_path / "data.zarr"

    data = np.arange(10, dtype=np.int32)
    blosc2.asarray(data, urlpath=str(src), mode="w")

    b2nd_to_zarr(src, dst)
    assert dst.exists()

    with pytest.raises(FileExistsError):
        b2nd_to_zarr(src, dst, overwrite=False)

    b2nd_to_zarr(src, dst, overwrite=True)
    assert dst.exists()


def test_cli_main(tmp_path, capsys):
    src = tmp_path / "cli.b2nd"
    dst = tmp_path / "cli.zarr"

    data = np.arange(50, dtype=np.int16)
    blosc2.asarray(data, chunks=(10,), urlpath=str(src), mode="w")

    ret = main([str(src), str(dst), "--verify", "--chunks", "25", "--quiet"])
    assert ret == 0
    assert dst.exists()

    z = zarr.open(store=str(dst), mode="r")
    assert z.chunks == (25,)
    np.testing.assert_array_equal(z[:], data)


def test_cli_main_errors(tmp_path, capsys):
    ret = main(["/nonexistent/path/here.b2nd"])
    assert ret == 1
    captured = capsys.readouterr()
    assert "Error:" in captured.err
