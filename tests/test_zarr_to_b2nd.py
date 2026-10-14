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
from blosc2.cli.b2nd_to_zarr import b2nd_to_zarr
from blosc2.cli.zarr_to_b2nd import main, zarr_to_b2nd

zarr = pytest.importorskip("zarr")
numcodecs = pytest.importorskip("numcodecs")


def make_zarr(path, data, **kwargs):
    z = zarr.create_array(store=str(path), shape=data.shape, dtype=data.dtype, **kwargs)
    z[...] = data
    return z


def test_zarr_to_b2nd_basic(tmp_path):
    src = tmp_path / "data.zarr"
    dst = tmp_path / "data.b2nd"

    shape = (100, 50)
    data = np.arange(5000, dtype=np.float32).reshape(shape)
    z = make_zarr(src, data, chunks=(20, 50))
    z.attrs["custom_key"] = "custom_value"

    summary = zarr_to_b2nd(src, verify=True)
    assert summary["dst_path"] == dst
    assert summary["shape"] == shape
    assert summary["dtype"] == np.float32
    assert summary["chunks"] == (20, 50)

    b = blosc2.open(dst)
    assert isinstance(b, blosc2.NDArray)
    assert b.chunks == (20, 50)
    np.testing.assert_array_equal(b[:], data)
    assert b.schunk.vlmeta["custom_key"] == "custom_value"


def test_zarr_to_b2nd_v2_blosc_compressor_is_preserved(tmp_path):
    src = tmp_path / "v2.zarr"
    data = np.arange(1200, dtype=np.int32).reshape(30, 40)
    compressor = numcodecs.Blosc(cname="lz4", clevel=7, shuffle=numcodecs.Blosc.BITSHUFFLE)
    make_zarr(src, data, chunks=(10, 40), zarr_format=2, compressors=compressor)

    summary = zarr_to_b2nd(src, full_verify=True)
    b = blosc2.open(summary["dst_path"])
    assert b.cparams.codec == blosc2.Codec.LZ4
    assert b.cparams.clevel == 7
    assert b.cparams.filters[0] == blosc2.Filter.BITSHUFFLE
    np.testing.assert_array_equal(b[:], data)


@pytest.mark.parametrize(
    ("kwargs", "codec", "clevel", "filt"),
    [
        # Zarr's default is Zstd level 0, meaning "Zstd default", not "no compression".
        ({}, blosc2.Codec.ZSTD, 5, blosc2.Filter.SHUFFLE),
        ({"compressors": zarr.codecs.ZstdCodec(level=15)}, blosc2.Codec.ZSTD, 9, blosc2.Filter.SHUFFLE),
        ({"compressors": zarr.codecs.GzipCodec(level=4)}, blosc2.Codec.ZLIB, 4, blosc2.Filter.SHUFFLE),
        ({"compressors": None}, blosc2.Codec.ZSTD, 5, blosc2.Filter.SHUFFLE),
        (
            {"compressors": zarr.codecs.BloscCodec(cname="zstd", clevel=3, shuffle="noshuffle")},
            blosc2.Codec.ZSTD,
            3,
            blosc2.Filter.NOFILTER,
        ),
    ],
    ids=["zstd-default", "zstd-high-level", "gzip", "uncompressed", "blosc-v3"],
)
def test_zarr_to_b2nd_compressor_translation(tmp_path, kwargs, codec, clevel, filt):
    src = tmp_path / "data.zarr"
    data = np.arange(1000, dtype=np.float64)
    make_zarr(src, data, chunks=(100,), **kwargs)

    summary = zarr_to_b2nd(src)
    cparams = summary["cparams"]
    assert cparams.codec == codec
    assert cparams.clevel == clevel
    assert cparams.filters[0] == filt
    np.testing.assert_array_equal(blosc2.open(summary["dst_path"])[:], data)


def test_zarr_to_b2nd_v2_autoshuffle_depends_on_itemsize(tmp_path):
    compressor = numcodecs.Blosc(shuffle=numcodecs.Blosc.AUTOSHUFFLE)
    for dtype, expected in [(np.uint8, blosc2.Filter.BITSHUFFLE), (np.float64, blosc2.Filter.SHUFFLE)]:
        src = tmp_path / f"{np.dtype(dtype).name}.zarr"
        make_zarr(src, np.arange(100, dtype=dtype), zarr_format=2, compressors=compressor)
        assert zarr_to_b2nd(src)["cparams"].filters[0] == expected


def test_zarr_to_b2nd_sharded(tmp_path):
    src = tmp_path / "sharded.zarr"
    data = np.arange(1600, dtype=np.int64).reshape(40, 40)
    make_zarr(src, data, chunks=(5, 40), shards=(20, 40))

    summary = zarr_to_b2nd(src, verify=True)
    assert summary["chunks"] == (20, 40)
    assert summary["blocks"] == (5, 40)

    b = blosc2.open(summary["dst_path"])
    assert b.chunks == (20, 40)
    assert b.blocks == (5, 40)
    np.testing.assert_array_equal(b[:], data)


def test_zarr_to_b2nd_custom_options(tmp_path):
    src = tmp_path / "cube.zarr"
    dst = tmp_path / "custom.b2nd"
    data = np.arange(30 * 40 * 50, dtype=np.int32).reshape(30, 40, 50)
    make_zarr(src, data, chunks=(10, 10, 10))

    summary = zarr_to_b2nd(
        src,
        dst,
        chunks=(15, 20, 50),
        blocks=(5, 10, 50),
        codec="lz4",
        clevel=3,
        shuffle="bitshuffle",
        full_verify=True,
    )
    assert summary["dst_path"] == dst

    b = blosc2.open(dst)
    assert b.chunks == (15, 20, 50)
    assert b.blocks == (5, 10, 50)
    assert b.cparams.codec == blosc2.Codec.LZ4
    assert b.cparams.clevel == 3
    assert b.cparams.filters[0] == blosc2.Filter.BITSHUFFLE
    np.testing.assert_array_equal(b[:], data)


def test_roundtrip_b2nd_to_zarr_to_b2nd(tmp_path):
    original = tmp_path / "original.b2nd"
    zarr_path = tmp_path / "middle.zarr"
    roundtrip = tmp_path / "roundtrip.b2nd"

    data = np.linspace(0, 1, 40 * 60, dtype=np.float64).reshape(40, 60)
    a = blosc2.asarray(data, chunks=(20, 60), blocks=(5, 60), urlpath=original, mode="w")
    a.schunk.vlmeta["units"] = "K"

    b2nd_to_zarr(original, zarr_path, sharded=True)
    zarr_to_b2nd(zarr_path, roundtrip, full_verify=True)

    b = blosc2.open(roundtrip)
    assert b.chunks == a.chunks
    assert b.blocks == a.blocks
    assert b.schunk.vlmeta["units"] == "K"
    np.testing.assert_array_equal(b[:], data)


def test_zarr_to_b2nd_zip_input(tmp_path):
    original = tmp_path / "arr.b2nd"
    src = tmp_path / "arr.zarr.zip"
    data = np.arange(1000, dtype=np.float64)
    blosc2.asarray(data, chunks=(100,), urlpath=original, mode="w")
    b2nd_to_zarr(original, src)
    original.unlink()

    summary = zarr_to_b2nd(src, verify=True)
    # data.zarr.zip should become data.b2nd, not data.zarr.b2nd
    assert summary["dst_path"] == tmp_path / "arr.b2nd"
    np.testing.assert_array_equal(blosc2.open(summary["dst_path"])[:], data)


def test_zarr_to_b2nd_rejects_group(tmp_path):
    src = tmp_path / "group.zarr"
    group = zarr.open_group(str(src), mode="w")
    group.create_array("temperature", shape=(4,), dtype="i4")

    with pytest.raises(ValueError, match="Zarr group"):
        zarr_to_b2nd(src)
    assert not (tmp_path / "group.b2nd").exists()


def test_zarr_to_b2nd_overwrite_protection(tmp_path):
    src = tmp_path / "data.zarr"
    dst = tmp_path / "data.b2nd"
    make_zarr(src, np.arange(10, dtype=np.int32))

    zarr_to_b2nd(src, dst)
    with pytest.raises(FileExistsError):
        zarr_to_b2nd(src, dst, overwrite=False)
    zarr_to_b2nd(src, dst, overwrite=True)
    assert dst.exists()


def test_zarr_to_b2nd_cannot_write_inside_source(tmp_path):
    src = tmp_path / "data.zarr"
    data = np.arange(10, dtype=np.int32)
    make_zarr(src, data)

    with pytest.raises(ValueError, match="source"):
        zarr_to_b2nd(src, src / "inside.b2nd", overwrite=True)
    np.testing.assert_array_equal(zarr.open_array(str(src), mode="r")[:], data)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"chunks": (10,)}, "same length"),
        ({"blocks": (500, 20)}, "greater than chunks"),
        ({"clevel": 12}, "between 0 and 9"),
        ({"codec": "zstdd"}, "Unsupported codec"),
        ({"shuffle": "bytes"}, "Unsupported shuffle"),
    ],
)
def test_invalid_conversion_preserves_destination(tmp_path, kwargs, match):
    src = tmp_path / "data.zarr"
    dst = tmp_path / "data.b2nd"
    make_zarr(src, np.arange(400, dtype=np.int32).reshape(20, 20), chunks=(10, 20))
    dst.write_bytes(b"preserve me")

    with pytest.raises(ValueError, match=match):
        zarr_to_b2nd(src, dst, overwrite=True, **kwargs)
    assert dst.read_bytes() == b"preserve me"


def test_zarr_to_b2nd_empty_array(tmp_path):
    src = tmp_path / "empty.zarr"
    zarr.create_array(store=str(src), shape=(0, 100), dtype="f4")

    summary = zarr_to_b2nd(src, verify=True)
    assert summary["shape"] == (0, 100)
    assert all(c > 0 for c in summary["chunks"])
    assert blosc2.open(summary["dst_path"]).shape == (0, 100)


def test_zarr_to_b2nd_with_nans(tmp_path):
    src = tmp_path / "nans.zarr"
    data = np.array([1.0, np.nan, 3.0, np.nan, 5.0], dtype=np.float32)
    make_zarr(src, data, chunks=(2,))

    summary = zarr_to_b2nd(src, full_verify=True)
    np.testing.assert_array_equal(blosc2.open(summary["dst_path"])[:], data)


def test_zarr_to_b2nd_skips_unstorable_attrs(tmp_path):
    src = tmp_path / "attrs.zarr"
    z = make_zarr(src, np.arange(4, dtype=np.int32))
    # JSON allows integers of any size, but vlmeta only stores 64-bit integers.
    z.attrs.update({"units": "K", "huge": 2**70, "nested": {"a": [1, 2]}})

    summary = zarr_to_b2nd(src)
    b = blosc2.open(summary["dst_path"])  # keep the array alive while reading its vlmeta
    vlmeta = b.schunk.vlmeta
    assert vlmeta["units"] == "K"
    assert vlmeta["nested"] == {"a": [1, 2]}
    assert "huge" not in vlmeta


def test_cli_main(tmp_path):
    src = tmp_path / "cli.zarr"
    dst = tmp_path / "cli.b2nd"
    data = np.arange(50, dtype=np.int16)
    make_zarr(src, data, chunks=(10,))

    ret = main([str(src), str(dst), "--verify", "--chunks", "25", "--blocks", "5", "--quiet"])
    assert ret == 0

    b = blosc2.open(dst)
    assert b.chunks == (25,)
    assert b.blocks == (5,)
    np.testing.assert_array_equal(b[:], data)


def test_cli_main_plain_summary(tmp_path, capsys, monkeypatch):
    src = tmp_path / "plain.zarr"
    make_zarr(src, np.arange(50, dtype=np.int16))
    monkeypatch.setattr("blosc2.cli.conversion_utils.HAVE_RICH", False)

    assert main([str(src)]) == 0
    out = capsys.readouterr().out
    assert "--- Conversion Summary ---" in out
    assert "Blosc2 CRatio:" in out


def test_cli_main_errors(tmp_path, capsys):
    assert main([str(tmp_path / "missing.zarr")]) == 1
    assert "Error:" in capsys.readouterr().err

    with pytest.raises(SystemExit) as excinfo:
        main([str(tmp_path / "missing.zarr"), "--codec", "gzip"])
    assert excinfo.value.code == 2
