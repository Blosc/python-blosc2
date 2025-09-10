import re

import numpy as np
import pytest

import blosc2


@pytest.mark.parametrize("initial_mapping_size", [None, 1000])
def test_initial_mapping_size(tmp_path, monkeypatch, capfd, initial_mapping_size):
    monkeypatch.setenv("BLOSC_INFO", "true")
    expected_mapping_size = 2**30 if initial_mapping_size is None else initial_mapping_size
    urlpath = tmp_path / "schunk.b2frame"

    # Writing via SChunk
    storage = {"contiguous": True, "urlpath": urlpath}
    chunk_nitems = 10
    nchunks = 2
    dtype = np.dtype(np.int64)

    schunk = blosc2.SChunk(
        chunksize=chunk_nitems * dtype.itemsize,
        mmap_mode="w+",
        initial_mapping_size=initial_mapping_size,
        **storage,
    )
    for i in range(nchunks):
        buffer = i * np.arange(chunk_nitems, dtype=dtype)
        nchunks_ = schunk.append_data(buffer)
        assert nchunks_ == (i + 1)
    del schunk

    captured = capfd.readouterr()
    assert (
        re.search(
            r"Opened memory-mapped file .*schunk\.b2frame in mode w\+ with an mapping size of "
            + str(expected_mapping_size),
            captured.err,
        )
        is not None
    ), captured.err

    # Reading via open
    for mmap_mode in ["r", "r+", "c"]:
        open_mapping_size = None if mmap_mode == "r" else initial_mapping_size
        schunk_open = blosc2.open(urlpath, mmap_mode=mmap_mode, initial_mapping_size=open_mapping_size)
        for i in range(nchunks):
            buffer = i * np.arange(chunk_nitems, dtype=dtype)
            bytes_obj = buffer.tobytes()
            res = schunk_open.decompress_chunk(i)
            assert res == bytes_obj

        captured = capfd.readouterr()
        mode_mapping_size = urlpath.stat().st_size if mmap_mode == "r" else expected_mapping_size
        assert (
            re.search(
                r"Opened memory-mapped file .*schunk\.b2frame in mode "
                + re.escape(mmap_mode)
                + " with an mapping size of "
                + str(mode_mapping_size),
                captured.err,
            )
            is not None
        ), captured.err

    # Writing via asarray
    nparray = np.arange(3, dtype=np.float32)
    a = blosc2.asarray(
        nparray,
        urlpath=tmp_path / "schunk2.b2frame",
        mmap_mode="w+",
        initial_mapping_size=initial_mapping_size,
    )
    np.testing.assert_almost_equal(a[...], nparray)

    captured = capfd.readouterr()
    assert (
        re.search(
            r"Opened memory-mapped file .*schunk2\.b2frame in mode w\+ with an mapping size of "
            + str(expected_mapping_size),
            captured.err,
        )
        is not None
    ), captured.err

    # Error handling
    with pytest.raises(ValueError, match=r"w\+ mmap_mode cannot be used to open an existing file"):
        blosc2.open(urlpath, mmap_mode="w+")

    with pytest.raises(ValueError, match="initial_mapping_size can only be used with writing modes"):
        blosc2.open(urlpath, mmap_mode="r", initial_mapping_size=100)

    with pytest.raises(ValueError, match="initial_mapping_size can only be used with mmap_mode"):
        blosc2.open(urlpath, mmap_mode=None, initial_mapping_size=100)

    with pytest.raises(ValueError, match="initial_mapping_size can only be used with writing modes"):
        blosc2.SChunk(mmap_mode="r", initial_mapping_size=100, **storage)

    with pytest.raises(ValueError, match="initial_mapping_size can only be used with mmap_mode"):
        blosc2.SChunk(mmap_mode=None, initial_mapping_size=100, **storage)

    with pytest.raises(ValueError, match="Only contiguous storage is supported"):
        blosc2.SChunk(contiguous=False, urlpath="b2frame", mmap_mode="w+")

    with pytest.raises(ValueError, match="urlpath must be set"):
        blosc2.SChunk(contiguous=True, urlpath=None, mmap_mode="w+")
