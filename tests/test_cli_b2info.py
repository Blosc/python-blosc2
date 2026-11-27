#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

from __future__ import annotations

import base64
import datetime
import json
import runpy
import sys
from dataclasses import dataclass

import numpy as np
import pytest

import blosc2
from blosc2.cli.b2info import attrs_to_json, main

# --------------------------------------------------------------------------
# Fixtures: every container is built inside tmp_path
# --------------------------------------------------------------------------


@pytest.fixture
def ndarray_path(tmp_path):
    path = tmp_path / "array.b2nd"
    arr = blosc2.asarray(
        np.linspace(0, 1, 40 * 50).reshape(40, 50),
        urlpath=str(path),
        mode="w",
        chunks=(20, 25),
        blocks=(10, 5),
        cparams=blosc2.CParams(codec=blosc2.Codec.ZSTD, clevel=5),
    )
    arr.attrs["units"] = "metres"
    return str(path)


@pytest.fixture
def schunk_path(tmp_path):
    path = tmp_path / "frame.b2frame"
    schunk = blosc2.SChunk(
        chunksize=100 * 4,
        urlpath=str(path),
        mode="w",
        cparams=blosc2.CParams(
            codec=blosc2.Codec.LZ4, clevel=7, typesize=4, filters=[blosc2.Filter.BITSHUFFLE]
        ),
    )
    for i in range(3):
        schunk.append_data(np.arange(i * 100, (i + 1) * 100, dtype=np.int32))
    schunk.attrs["source"] = "test"
    return str(path)


@pytest.fixture
def dictstore_path(tmp_path):
    path = tmp_path / "store.b2d"
    with blosc2.DictStore(str(path), mode="w") as store:
        store["/a"] = np.arange(3)
        store["/b"] = np.arange(4)
    return str(path)


@pytest.fixture
def treestore_path(tmp_path):
    path = tmp_path / "tree.b2z"
    with blosc2.TreeStore(str(path), mode="w") as store:
        store["/x"] = np.arange(3)
        store["/grp/y"] = np.arange(2)
        store.attrs["experiment"] = "t"
    return str(path)


# One value per encoding scenario, read back by --json --attrs.
# Kept under 16 keys: blosc2 refuses a 17th variable-length metalayer.
ATTRS_VALUES = {
    "units": "kelvin",
    "plain": {"count": 100, "ok": True, "nothing": None, "numbers": [1, 2]},
    "config": {"rate_hz": 100, "mode": "fast"},
    "pair": (1, 2),
    "checksum": b"\x9f\x12\xab",
    "calibration": np.array([0.98, 1.02]),
    "byte_array": np.array([b"ab", b"cd"]),
    "tags": {"raw", "night"},
    "mixed_set": {1, "a"},
    "fill_value": float("nan"),
    "limit": float("inf"),
    "floor": float("-inf"),
    "header": {b"SIMPLE": True},
    "nested": {"inner": [b"\x01", float("nan"), (3, 4)]},
}


@pytest.fixture
def attrs_path(tmp_path):
    path = tmp_path / "attrs.b2nd"
    arr = blosc2.zeros(4, urlpath=str(path), mode="w")
    for key, value in ATTRS_VALUES.items():
        arr.attrs[key] = value
    return str(path)


def run(capsys, *args):
    """Run b2info and return (exit code, stdout, stderr)."""
    code = main([*args])
    out, err = capsys.readouterr()
    return code, out, err


def run_json(capsys, *args):
    code, out, err = run(capsys, *args, "--json")
    assert code == 0, err
    return json.loads(out)


def no_nan_constants(name):
    raise ValueError(f"invalid JSON constant {name}")


# --------------------------------------------------------------------------
# A. Default text output
# --------------------------------------------------------------------------


def test_text_ndarray(capsys, ndarray_path):
    code, out, err = run(capsys, ndarray_path)
    assert code == 0
    assert err == ""
    assert out.startswith("type    : NDArray\n")
    assert "shape   : (40, 50)" in out
    assert "cparams : " in out


def test_text_schunk(capsys, schunk_path):
    code, out, _ = run(capsys, schunk_path)
    assert code == 0
    assert out.startswith("type      : SChunk\n")
    assert "chunksize : 400" in out


@pytest.mark.parametrize(
    ("fixture", "type_name", "entries"),
    [("dictstore_path", "DictStore", 2), ("treestore_path", "TreeStore", 3)],
)
def test_text_store_summary_has_no_contents(request, capsys, fixture, type_name, entries):
    code, out, _ = run(capsys, request.getfixturevalue(fixture))
    assert code == 0
    assert out.startswith(f"type    : {type_name}\n")
    assert f"entries : {entries}" in out
    assert "contents" not in out


def test_text_empty_store(capsys, tmp_path):
    path = tmp_path / "empty.b2d"
    blosc2.DictStore(str(path), mode="w").close()
    code, out, _ = run(capsys, str(path))
    assert code == 0
    assert "entries : 0" in out


def test_text_matches_info(capsys, ndarray_path):
    # The default summary is exactly what `.info` prints.
    arr = blosc2.open(ndarray_path)
    code, out, _ = run(capsys, ndarray_path)
    assert code == 0
    assert out == repr(arr.info)


# --------------------------------------------------------------------------
# B. --tree
# --------------------------------------------------------------------------


def test_tree_treestore(capsys, treestore_path):
    code, out, _ = run(capsys, treestore_path, "--tree")
    assert code == 0
    assert "contents : /\n" in out
    assert "├── grp" in out
    assert "└── y" in out


def test_tree_dictstore_is_flat(capsys, dictstore_path):
    code, out, _ = run(capsys, dictstore_path, "--tree")
    assert code == 0
    assert "contents : /a\n" in out
    assert "/b" in out
    assert "├" not in out
    assert "└" not in out


@pytest.mark.parametrize(("fixture", "type_name"), [("ndarray_path", "NDArray"), ("schunk_path", "SChunk")])
@pytest.mark.parametrize("extra", [[], ["--json"]])
def test_tree_needs_a_store(request, capsys, fixture, type_name, extra):
    code, out, err = run(capsys, request.getfixturevalue(fixture), "--tree", *extra)
    assert code == 1
    assert out == ""
    assert err == f"b2info: error: --tree needs a store (.b2d or .b2z), not {type_name}\n"


# --------------------------------------------------------------------------
# C. --json
# --------------------------------------------------------------------------


def test_json_ndarray(capsys, ndarray_path):
    data = run_json(capsys, ndarray_path)
    assert data["type"] == "NDArray"
    assert data["shape"] == [40, 50]
    assert data["chunks"] == [20, 25]
    assert data["blocks"] == [10, 5]
    assert data["dtype"] == "float64"
    assert data["nbytes"] == 40 * 50 * 8
    assert isinstance(data["cbytes"], int)
    assert data["cratio"] == pytest.approx(data["nbytes"] / data["cbytes"])
    cparams = data["cparams"]
    assert cparams["codec"] == "ZSTD"
    assert cparams["clevel"] == 5
    assert "SHUFFLE" in cparams["filters"]
    assert all(isinstance(f, str) for f in cparams["filters"])
    assert isinstance(cparams["splitmode"], str)
    assert isinstance(cparams["tuner"], str)
    assert "nthreads" in data["dparams"]
    assert "attrs" not in data


def test_json_schunk(capsys, schunk_path):
    data = run_json(capsys, schunk_path)
    assert data["type"] == "SChunk"
    assert data["chunksize"] == 400
    assert data["typesize"] == 4
    assert data["nbytes"] == 300 * 4
    assert "shape" not in data
    assert data["cparams"]["codec"] == "LZ4"
    assert data["cparams"]["filters"][0] == "BITSHUFFLE"


def test_json_dictstore(capsys, dictstore_path):
    data = run_json(capsys, dictstore_path)
    assert data["type"] == "DictStore"
    assert data["format"] == "b2d"
    assert data["mode"] == "r"
    assert data["entries"] == 2
    assert data["contents"] == ["/a", "/b"]


def test_json_treestore(capsys, treestore_path):
    data = run_json(capsys, treestore_path)
    assert data["type"] == "TreeStore"
    assert data["format"] == "b2z"
    assert data["contents"] == ["/grp", "/grp/y", "/x"]
    assert data["entries"] == len(data["contents"])


def test_json_tree_adds_nothing_for_stores(capsys, treestore_path):
    _, plain, _ = run(capsys, treestore_path, "--json")
    _, with_tree, _ = run(capsys, treestore_path, "--json", "--tree")
    assert plain == with_tree


def test_json_cbytes_zero_gives_null_cratio(capsys, tmp_path):
    # All-zero data is stored as special chunks with no compressed bytes on disk.
    path = tmp_path / "zeros.b2nd"
    blosc2.asarray(np.zeros((50, 50), dtype=np.float32), urlpath=str(path), mode="w")
    data = run_json(capsys, str(path))
    assert data["cbytes"] == 0
    assert data["cratio"] is None


# --------------------------------------------------------------------------
# D. --attrs (text)
# --------------------------------------------------------------------------


def test_attrs_text_ndarray(capsys, ndarray_path):
    code, out, _ = run(capsys, ndarray_path, "--attrs")
    assert code == 0
    assert out.endswith("attrs   : {'units': 'metres'}\n")


def test_attrs_text_dictstore_has_none(capsys, dictstore_path):
    code, out, _ = run(capsys, dictstore_path, "--attrs")
    assert code == 0
    assert "attrs   : (none: DictStore has no user attributes)" in out


def test_attrs_text_treestore_root_only(capsys, treestore_path):
    code, out, _ = run(capsys, treestore_path, "--attrs")
    assert code == 0
    assert out.endswith("attrs   : {'experiment': 't'}\n")


def test_attrs_text_with_tree(capsys, treestore_path):
    code, out, _ = run(capsys, treestore_path, "--tree", "--attrs")
    assert code == 0
    assert "contents : /" in out
    assert "attrs    : {'experiment': 't'}" in out


def test_attrs_text_multiline_is_indented(capsys, attrs_path):
    code, out, _ = run(capsys, attrs_path, "--attrs")
    assert code == 0
    lines = out.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("attrs"))
    continuation = lines[start + 1 :]
    assert continuation  # the attrs dict needs several lines
    indent = " " * lines[start].index("{")
    assert all(line.startswith(indent) for line in continuation)


def test_no_attrs_without_flag(capsys, ndarray_path):
    _, out, _ = run(capsys, ndarray_path)
    assert "attrs" not in out
    assert "attrs" not in run_json(capsys, ndarray_path)


# --------------------------------------------------------------------------
# E. --json --attrs encoding
# --------------------------------------------------------------------------


@pytest.fixture
def attrs_json(capsys, attrs_path):
    return run_json(capsys, attrs_path, "--attrs")["attrs"]


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("units", "kelvin"),
        ("plain", {"count": 100, "ok": True, "nothing": None, "numbers": [1, 2]}),
        ("config", {"rate_hz": 100, "mode": "fast"}),
        ("pair", {"__type__": "tuple", "items": [1, 2]}),
        ("checksum", {"__type__": "bytes", "base64": "nxKr"}),
        (
            "calibration",
            {"__type__": "ndarray", "dtype": "float64", "shape": [2], "data": [0.98, 1.02]},
        ),
        (
            "byte_array",
            {
                "__type__": "ndarray",
                "dtype": "|S2",
                "shape": [2],
                "data": [
                    {"__type__": "bytes", "base64": "YWI="},
                    {"__type__": "bytes", "base64": "Y2Q="},
                ],
            },
        ),
        ("tags", {"__type__": "set", "items": ["night", "raw"]}),
        ("mixed_set", {"__type__": "set", "items": ["a", 1]}),
        ("fill_value", {"__type__": "float", "value": "NaN"}),
        ("limit", {"__type__": "float", "value": "Infinity"}),
        ("floor", {"__type__": "float", "value": "-Infinity"}),
        (
            "header",
            {"__type__": "dict", "items": [[{"__type__": "bytes", "base64": "U0lNUExF"}, True]]},
        ),
        (
            "nested",
            {
                "inner": [
                    {"__type__": "bytes", "base64": "AQ=="},
                    {"__type__": "float", "value": "NaN"},
                    {"__type__": "tuple", "items": [3, 4]},
                ]
            },
        ),
    ],
)
def test_attrs_json_encoding(attrs_json, key, expected):
    assert attrs_json[key] == expected


def test_attrs_json_bytes_round_trip(attrs_json):
    assert base64.b64decode(attrs_json["checksum"]["base64"]) == ATTRS_VALUES["checksum"]


def test_attrs_json_is_strict_json(capsys, attrs_path):
    code, out, _ = run(capsys, attrs_path, "--json", "--attrs")
    assert code == 0
    json.loads(out, parse_constant=no_nan_constants)  # no bare NaN / Infinity


def test_attrs_json_dictstore_is_null(capsys, dictstore_path):
    assert run_json(capsys, dictstore_path, "--attrs")["attrs"] is None


def test_attrs_json_treestore_root_only(capsys, treestore_path):
    assert run_json(capsys, treestore_path, "--attrs")["attrs"] == {"experiment": "t"}


# Types blosc2 converts when reading attrs back, so only reachable directly.
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (np.int64(7), 7),
        (np.float32(1.5), 1.5),
        (bytearray(b"ab"), {"__type__": "bytes", "base64": "YWI="}),
        (frozenset({3, 1}), {"__type__": "set", "items": [1, 3]}),
        (datetime.date(2026, 1, 1), {"__type__": "repr", "repr": "datetime.date(2026, 1, 1)"}),
    ],
)
def test_attrs_to_json_direct(value, expected):
    assert attrs_to_json(value) == expected


# --------------------------------------------------------------------------
# F. Errors
# --------------------------------------------------------------------------


def test_error_missing_path(capsys, tmp_path):
    path = str(tmp_path / "nope.b2nd")
    code, out, err = run(capsys, path)
    assert code == 1
    assert out == ""
    assert err.startswith("b2info: error: No such file or directory")
    assert "Traceback" not in err


@pytest.mark.parametrize("kind", ["text_file", "empty_dir"])
def test_error_not_a_container(capsys, tmp_path, kind):
    path = tmp_path / kind
    if kind == "text_file":
        path.write_text("hello, not blosc2\n")
    else:
        path.mkdir()
    code, out, err = run(capsys, str(path))
    assert code == 1
    assert out == ""
    assert err == (
        f"b2info: error: cannot open {str(path)!r}: not a valid Blosc2 container or not readable\n"
    )


@dataclass
class Row:
    id: int = blosc2.field(blosc2.int64())


def make_embedstore(path):
    store = blosc2.EmbedStore(urlpath=str(path), mode="w")
    store["/a"] = np.arange(3)


def make_ctable(path):
    table = blosc2.CTable(Row, urlpath=str(path), mode="w")
    table.extend([(1,), (2,)])
    table.close()


@pytest.mark.parametrize(
    ("filename", "make", "type_name"),
    [("x.b2e", make_embedstore, "EmbedStore"), ("table.b2z", make_ctable, "CTable")],
)
def test_error_unsupported_type(capsys, tmp_path, filename, make, type_name):
    path = tmp_path / filename
    make(path)
    code, out, err = run(capsys, str(path))
    assert code == 1
    assert out == ""
    assert err == f"b2info: error: unsupported container type: {type_name}\n"


@pytest.mark.parametrize("url", ["https://example.com/data/x.b2z", "s3://bucket/x.b2nd"])
def test_error_remote_url(capsys, url):
    code, out, err = run(capsys, url)
    assert code == 1
    assert out == ""
    assert err == "b2info: error: remote URLs are not supported; pass a local path\n"


def test_error_no_arguments(capsys):
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2
    assert "the following arguments are required: path" in capsys.readouterr().err


def test_error_with_json_keeps_stdout_clean(capsys, tmp_path):
    code, out, err = run(capsys, str(tmp_path / "nope.b2nd"), "--json")
    assert code == 1
    assert out == ""
    assert err.startswith("b2info: error: ")


# --------------------------------------------------------------------------
# G. Entry point
# --------------------------------------------------------------------------


def test_python_dash_m(capsys, monkeypatch, ndarray_path):
    # `python -m blosc2.cli.b2info` runs main() and exits with its return code.
    monkeypatch.setattr(sys, "argv", ["b2info", ndarray_path])
    monkeypatch.delitem(sys.modules, "blosc2.cli.b2info")  # avoid runpy's re-import warning
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("blosc2.cli.b2info", run_name="__main__")
    assert exc.value.code == 0
    assert capsys.readouterr().out.startswith("type    : NDArray\n")
