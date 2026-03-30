#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

from __future__ import annotations

import numpy as np

import blosc2
import blosc2.c2array as blosc2_c2array


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
    urlpath = tmp_path / "remote-array.b2o"

    original.save(urlpath)
    restored = blosc2.open(urlpath, mode="r")

    assert isinstance(restored, blosc2.C2Array)
    assert restored.path == original.path
    assert restored.urlbase == original.urlbase
    assert restored.auth_token is None
    assert restored.shape == original.shape
    assert restored.chunks == original.chunks
    assert restored.blocks == original.blocks
