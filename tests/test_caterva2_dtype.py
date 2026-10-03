"""Caterva2 dtype representations must agree with native b2nd decoding."""

import numpy as np
import pytest
from test_remote_caterva2 import caterva2_source  # noqa: F401

import blosc2
from blosc2.b2view.model import StoreBrowser


@pytest.mark.parametrize(
    "dtype",
    [
        np.dtype("int32"),
        np.dtype("S6"),
        np.dtype("U6"),
        np.dtype("complex128"),
        np.dtype([("a", "<i4"), ("b", "<f8"), ("c", "S10"), ("d", "?")]),
        np.dtype([("nested", [("x", "i4"), ("y", "f8")]), ("vector", "f4", (2, 3))]),
        np.dtype([("byte", "u1"), ("number", "f8")], align=True),
        np.dtype(("f8", (4,))),
    ],
)
def test_c2array_dtype_repr(dtype):
    array = object.__new__(blosc2.C2Array)
    array.meta = {"dtype": str(dtype)}
    assert array.dtype == dtype


@pytest.mark.parametrize("dtype", ["not-a-dtype", "[('bad', 'not-a-dtype')]", "__import__('os').getcwd()"])
def test_invalid_dtype_is_not_executed(dtype, monkeypatch):
    import os

    monkeypatch.setattr(os, "getcwd", lambda: pytest.fail("executed remote metadata"))
    array = object.__new__(blosc2.C2Array)
    array.meta = {"dtype": dtype}
    with pytest.raises((TypeError, ValueError, SyntaxError)):
        _ = array.dtype


@pytest.mark.parametrize("shape", [(13,), (5, 7)])
def test_structured_remote_array_and_browser(caterva2_source, shape):  # noqa: F811
    base, _, _, stats = caterva2_source
    dtype = np.dtype([("a", "<i4"), ("b", "<f8"), ("c", "S10"), ("d", "?")])
    data = np.zeros(shape, dtype=dtype)
    data["a"] = np.arange(data.size).reshape(shape)
    data["b"] = data["a"] / 10
    data["c"] = b"foobar"
    data["d"] = data["a"] % 2 == 0
    chunks = (5,) if len(shape) == 1 else (3, 4)
    blocks = (3,) if len(shape) == 1 else (2, 3)
    source = blosc2.asarray(data, chunks=chunks, blocks=blocks)
    key = "@public/structured"
    stats["arrays"][key] = source
    stats["groups"]["@public"].append("structured")
    # Legacy info does not have an explicit kind; recognition must still work.
    stats["array_metadata"][key] = {"kind": None}
    with blosc2.C2Array(key, urlbase=base) as c2:
        assert c2.dtype == dtype
        np.testing.assert_array_equal(c2[:], data)
    with blosc2.open(base + key) as remote:
        assert remote.dtype == dtype
        np.testing.assert_array_equal(remote[:], data)
        np.testing.assert_array_equal(remote[2:8], data[2:8])
    with StoreBrowser(base) as browser:
        assert browser.get_info("/structured").metadata["dtype"] == str(dtype)
        np.testing.assert_array_equal(browser._get_object("/structured")[:], data)
        preview = browser.preview("/structured", max_rows=2, max_cols=2)
        if len(shape) == 1:
            np.testing.assert_array_equal(preview["data"]["value"], data[:2])
        else:
            np.testing.assert_array_equal(preview["data"]["0"], data[:2, 0])


@pytest.mark.parametrize(("shape", "inner_shape"), [((13,), (4,)), ((5, 7), (2, 3))])
def test_subarray_chunk_geometry(caterva2_source, shape, inner_shape):  # noqa: F811
    base, _, _, stats = caterva2_source
    chunks = (5,) if len(shape) == 1 else (3, 4)
    blocks = (3,) if len(shape) == 1 else (2, 3)
    expanded_shape = (*shape, *inner_shape)
    data = np.arange(np.prod(expanded_shape), dtype=np.float64).reshape(expanded_shape)
    # HDF5 serves an expanded NDArray slice but describes a logical subarray dtype.
    source = blosc2.asarray(data, chunks=(*chunks, *inner_shape), blocks=(*blocks, *inner_shape))
    key = "@public/subarray"
    stats["arrays"][key] = source
    stats["array_metadata"][key] = {
        "dtype": str(np.dtype(("f8", inner_shape))),
        "shape": shape,
        "chunks": chunks,
        "blocks": blocks,
    }
    stats["groups"]["@public"].append("subarray")
    with blosc2.open(base + key) as remote:
        assert remote.shape == shape
        assert remote.dtype == np.dtype(("f8", inner_shape))
        np.testing.assert_array_equal(remote[:], data)
        np.testing.assert_array_equal(remote[2:4], data[2:4])
    with StoreBrowser(base) as browser:
        np.testing.assert_array_equal(browser._get_object("/subarray")[:], data)
        preview = browser.preview("/subarray", max_rows=2, max_cols=2)
        if len(shape) == 1:
            np.testing.assert_array_equal(preview["data"]["value"], data[:2])
        else:
            np.testing.assert_array_equal(preview["data"]["0"], data[:2, 0])


@pytest.mark.network
@pytest.mark.parametrize(
    "path",
    [
        "examples/ds-1d-fields.b2nd",
        "examples/ds-2d-fields.b2nd",
        "examples/sa-1M.b2nd",
        "examples/hdf5root-example.h5/unsupported/compound-dtype",
        "examples/hdf5root-example.h5/unsupported/array-dtype",
    ],
)
def test_live_demo_dtype_previews(path):
    with StoreBrowser("https://cat2.cloud/demo") as browser:
        info = browser.get_info("/" + path)
        assert info.kind == "ndarray"
        preview = browser.preview("/" + path, max_rows=2, max_cols=2)
        assert preview["stop"] == 2
        assert len(preview["data"][preview["columns"][0]]) == 2
