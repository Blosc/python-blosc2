#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

from __future__ import annotations

from pathlib import Path

import numpy as np

import blosc2
import blosc2.c2array as blosc2_c2array


@blosc2.dsl_kernel
def kernel_add_square(x, y):
    return x * x + y * y + 2 * x * y


def _make_c2array(
    monkeypatch,
    path="@public/examples/ds-1d.b2nd",
    urlbase="https://cat2.cloud/demo/",
    auth_token=None,
    shape=(5,),
    chunks=(5,),
    blocks=(5,),
    dtype=np.int64,
):
    dtype = np.dtype(dtype)

    def fake_info(path_, urlbase_, params=None, headers=None, model=None, auth_token=None):
        return {
            "shape": list(shape),
            "chunks": list(chunks),
            "blocks": list(blocks),
            "dtype": np.dtype(dtype).str,
            "schunk": {
                "cparams": dict(blosc2.cparams_dflts),
                "nbytes": int(np.prod(shape)) * np.dtype(dtype).itemsize,
                "cbytes": int(np.prod(shape)) * np.dtype(dtype).itemsize,
                "cratio": 1.0,
                "blocksize": int(np.prod(blocks)) * np.dtype(dtype).itemsize,
                "vlmeta": {},
            },
        }

    monkeypatch.setattr(blosc2_c2array, "info", fake_info)
    return blosc2.C2Array(path, urlbase=urlbase, auth_token=auth_token)


def test_c2array_from_cframe_roundtrip(monkeypatch):
    original = _make_c2array(monkeypatch, auth_token="secret-token")
    carrier = blosc2.ndarray_from_cframe(original.to_cframe())

    assert carrier.schunk.meta["b2o"] == {"kind": "c2array", "version": 1}
    assert carrier.schunk.vlmeta["b2o"] == {
        "kind": "c2array",
        "version": 1,
        "path": original.path,
        "urlbase": original.urlbase,
    }

    restored = blosc2.from_cframe(original.to_cframe())

    assert isinstance(restored, blosc2.C2Array)
    assert restored.path == original.path
    assert restored.urlbase == original.urlbase
    assert restored.auth_token is None
    assert restored.shape == original.shape
    assert restored.dtype == original.dtype


def test_c2array_open_roundtrip(tmp_path, monkeypatch):
    original = _make_c2array(monkeypatch, shape=(8,), chunks=(4,), blocks=(2,))
    urlpath = tmp_path / "remote-array.b2nd"

    original.save(urlpath)
    restored = blosc2.open(urlpath, mode="r")

    assert isinstance(restored, blosc2.C2Array)
    assert restored.path == original.path
    assert restored.urlbase == original.urlbase
    assert restored.auth_token is None
    assert restored.shape == original.shape
    assert restored.chunks == original.chunks
    assert restored.blocks == original.blocks


def test_lazyexpr_from_cframe_roundtrip(tmp_path):
    a = blosc2.asarray(np.arange(5, dtype=np.int64), urlpath=tmp_path / "a.b2nd", mode="w")
    b = blosc2.asarray(np.arange(5, dtype=np.int64) * 2, urlpath=tmp_path / "b.b2nd", mode="w")
    expr = blosc2.lazyexpr("a + b", operands={"a": a, "b": b})
    carrier = blosc2.ndarray_from_cframe(expr.to_cframe())

    assert carrier.schunk.meta["b2o"] == {"kind": "lazyexpr", "version": 1}
    assert carrier.schunk.vlmeta["b2o"] == {
        "kind": "lazyexpr",
        "version": 1,
        "expression": "a + b",
        "operands": {
            "a": {"kind": "urlpath", "version": 1, "urlpath": (tmp_path / "a.b2nd").as_posix()},
            "b": {"kind": "urlpath", "version": 1, "urlpath": (tmp_path / "b.b2nd").as_posix()},
        },
    }

    restored = blosc2.from_cframe(expr.to_cframe())

    assert isinstance(restored, blosc2.LazyExpr)
    np.testing.assert_array_equal(restored[:], np.arange(5, dtype=np.int64) * 3)


def test_lazyexpr_open_roundtrip(tmp_path):
    a = blosc2.asarray(np.arange(5, dtype=np.int64), urlpath=tmp_path / "a.b2nd", mode="w")
    b = blosc2.asarray(np.arange(5, dtype=np.int64) * 2, urlpath=tmp_path / "b.b2nd", mode="w")
    expr = blosc2.lazyexpr("a + b", operands={"a": a, "b": b})
    urlpath = tmp_path / "expr.b2nd"

    expr.save(urlpath)
    restored = blosc2.open(urlpath, mode="r")

    assert isinstance(restored, blosc2.LazyExpr)
    np.testing.assert_array_equal(restored[:], np.arange(5, dtype=np.int64) * 3)


def test_legacy_lazyexpr_open_backward_compat():
    fixture = Path(__file__).parent / "data" / "legacy_lazyexpr_v1" / "expr.b2nd"

    restored = blosc2.open(fixture, mode="r")

    assert isinstance(restored, blosc2.LazyExpr)
    np.testing.assert_array_equal(restored[:], np.arange(5, dtype=np.int64) * 3)


def test_legacy_lazyudf_open_backward_compat():
    fixture = Path(__file__).parent / "data" / "legacy_lazyudf_v1" / "expr.b2nd"

    restored = blosc2.open(fixture, mode="r")

    assert isinstance(restored, blosc2.LazyUDF)
    np.testing.assert_allclose(restored.compute()[:], (np.arange(5, dtype=np.float64) * 3) ** 2)


def test_lazyudf_from_cframe_roundtrip(tmp_path):
    a = blosc2.asarray(np.arange(5, dtype=np.float64), urlpath=tmp_path / "a.b2nd", mode="w")
    b = blosc2.asarray(np.arange(5, dtype=np.float64) * 2, urlpath=tmp_path / "b.b2nd", mode="w")
    expr = blosc2.lazyudf(kernel_add_square, (a, b), dtype=np.float64)
    carrier = blosc2.ndarray_from_cframe(expr.to_cframe())

    assert carrier.schunk.meta["b2o"] == {"kind": "lazyudf", "version": 1}
    payload = carrier.schunk.vlmeta["b2o"]
    assert payload["kind"] == "lazyudf"
    assert payload["version"] == 1
    assert payload["function_kind"] == "dsl"
    assert payload["dsl_version"] == 1
    assert payload["name"] == "kernel_add_square"
    assert "kernel_add_square" in payload["udf_source"]
    assert payload["dtype"] == np.dtype(np.float64).str
    assert payload["shape"] == [5]
    assert payload["operands"] == {
        "o0": {"kind": "urlpath", "version": 1, "urlpath": (tmp_path / "a.b2nd").as_posix()},
        "o1": {"kind": "urlpath", "version": 1, "urlpath": (tmp_path / "b.b2nd").as_posix()},
    }

    restored = blosc2.from_cframe(expr.to_cframe())

    assert isinstance(restored, blosc2.LazyUDF)
    np.testing.assert_allclose(restored[:], (np.arange(5, dtype=np.float64) * 3) ** 2)


def test_lazyudf_open_roundtrip(tmp_path):
    a = blosc2.asarray(np.arange(5, dtype=np.float64), urlpath=tmp_path / "a.b2nd", mode="w")
    b = blosc2.asarray(np.arange(5, dtype=np.float64) * 2, urlpath=tmp_path / "b.b2nd", mode="w")
    expr = blosc2.lazyudf(kernel_add_square, (a, b), dtype=np.float64)
    urlpath = tmp_path / "expr.b2nd"

    expr.save(urlpath)
    restored = blosc2.open(urlpath, mode="r")

    assert isinstance(restored, blosc2.LazyUDF)
    np.testing.assert_allclose(restored[:], (np.arange(5, dtype=np.float64) * 3) ** 2)


def test_b2z_bundle_with_lazy_recipes_opens_read_only(tmp_path):
    bundle_path = tmp_path / "bundle.b2z"

    with blosc2.DictStore(str(bundle_path), mode="w", threshold=1) as store:
        store["/data/a"] = np.arange(5, dtype=np.float64)
        store["/data/b"] = np.arange(5, dtype=np.float64) * 2

        a = store["/data/a"]
        b = store["/data/b"]
        expr = blosc2.lazyexpr("a + b", operands={"a": a, "b": b})
        udf = blosc2.lazyudf(kernel_add_square, (a, b), dtype=np.float64, shape=a.shape)

        store["/recipes/expr"] = blosc2.ndarray_from_cframe(expr.to_cframe())
        store["/recipes/udf"] = blosc2.ndarray_from_cframe(udf.to_cframe())

    with blosc2.open(str(bundle_path), mode="r") as store:
        restored_expr = store["/recipes/expr"]
        restored_udf = store["/recipes/udf"]

        assert isinstance(restored_expr, blosc2.LazyExpr)
        assert isinstance(restored_udf, blosc2.LazyUDF)
        np.testing.assert_allclose(restored_expr.compute()[:], np.arange(5, dtype=np.float64) * 3)
        np.testing.assert_allclose(restored_udf.compute()[:], (np.arange(5, dtype=np.float64) * 3) ** 2)
