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


@pytest.mark.parametrize("codec", ["lz4", "gzip", "none"])
def test_b2nd_to_zarr_v2_uses_numcodecs_compressor(tmp_path, codec):
    src = tmp_path / f"v2-{codec}.b2nd"
    dst = tmp_path / f"v2-{codec}.zarr"

    data = np.arange(1200, dtype=np.int32).reshape(30, 40)
    blosc2.asarray(data, chunks=(10, 10), urlpath=str(src), mode="w")

    summary = b2nd_to_zarr(src, dst, codec=codec, zarr_format=2, full_verify=True)
    z = zarr.open(store=str(dst), mode="r")
    assert z.metadata.zarr_format == 2
    compressor = z.metadata.compressor
    expected_id = {"lz4": "blosc", "gzip": "gzip", "none": None}[codec]
    assert (compressor is None and summary["codec"] is None) or compressor.get_config()["id"] == expected_id
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


@pytest.mark.parametrize("destination", ["source", "parent"])
def test_conversion_cannot_overwrite_source(tmp_path, destination):
    src = tmp_path / "input" / "data.b2nd"
    src.parent.mkdir()
    data = np.arange(10, dtype=np.int32)
    blosc2.asarray(data, urlpath=src, mode="w")
    dst = src if destination == "source" else src.parent
    with pytest.raises(ValueError, match="source"):
        b2nd_to_zarr(src, dst, overwrite=True)
    np.testing.assert_array_equal(blosc2.open(src)[:], data)


def test_invalid_conversion_preserves_destination(tmp_path):
    src = tmp_path / "data.b2nd"
    dst = tmp_path / "data.zarr"
    blosc2.asarray(np.arange(10), urlpath=src, mode="w")
    dst.mkdir()
    sentinel = dst / "sentinel"
    sentinel.write_bytes(b"preserve me")
    with pytest.raises(ValueError, match="Sharding"):
        b2nd_to_zarr(src, dst, overwrite=True, shards=(10,), zarr_format=2)
    assert sentinel.read_bytes() == b"preserve me"


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


def test_conversion_empty_array_with_zero_chunks(tmp_path):
    src = tmp_path / "empty.b2nd"
    dst = tmp_path / "empty.zarr"
    # Create empty Blosc2 array with zero chunk extent
    blosc2.asarray(
        np.empty((0, 100), dtype=np.float32),
        chunks=(0, 10),
        blocks=(0, 5),
        urlpath=str(src),
        mode="w",
    )

    result = b2nd_to_zarr(src, dst, verify=True)
    assert result["shape"] == (0, 100)
    assert all(c > 0 for c in result["chunks"])

    z = zarr.open(store=str(dst), mode="r")
    assert z.shape == (0, 100)
    assert all(c > 0 for c in z.chunks)


def test_conversion_with_nans(tmp_path):
    src = tmp_path / "nans.b2nd"
    dst = tmp_path / "nans.zarr"
    data = np.array([1.0, np.nan, 3.0, np.nan, 5.0], dtype=np.float32)
    blosc2.asarray(data, chunks=(2,), urlpath=str(src), mode="w")

    # Fast verify and full verify both verify slices with NaNs without raising
    result = b2nd_to_zarr(src, dst, full_verify=True)
    z = zarr.open(store=str(dst), mode="r")
    np.testing.assert_array_equal(z[:], data)
