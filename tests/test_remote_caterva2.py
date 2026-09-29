"""Deterministic Caterva2 API sources for remote arrays, stores, and tables."""

from __future__ import annotations

import dataclasses
import json
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest

import blosc2


@dataclasses.dataclass
class Reading:
    ident: int = blosc2.field(blosc2.int64(), chunks=(4,), blocks=(2,))
    value: int = blosc2.field(blosc2.int32(), chunks=(4,), blocks=(2,))


def safe(value):
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return safe(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [safe(item) for item in value]
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


@pytest.fixture
def caterva2_source():
    array = blosc2.asarray(np.arange(60, dtype=np.int32).reshape(6, 10), chunks=(3, 5), blocks=(1, 5))
    table = blosc2.CTable(Reading, [(index, index * 3) for index in range(12)], create_summary_index=False)
    stats = {"fetches": 0, "cookies": []}

    def array_info():
        return {
            "kind": "ndarray",
            "shape": array.shape,
            "chunks": array.chunks,
            "blocks": array.blocks,
            "dtype": str(array.dtype),
            "attrs": {},
            "schunk": {
                "nbytes": array.nbytes,
                "cbytes": array.cbytes,
                "nchunks": 4,
                "cparams": safe(array.cparams),
                "vlmeta": {},
            },
        }

    def table_info():
        schema = table.schema_dict()
        return {
            "kind": "ctable",
            "nrows": table.nrows,
            "ncols": table.ncols,
            "columns": [column["name"] for column in schema["columns"]],
            "schema_dict": schema,
            "chunks": (4,),
            "blocks": (2,),
            "nbytes": table.nbytes,
            "cbytes": table.cbytes,
            "attrs": {"fixture": "local"},
        }

    class Handler(BaseHTTPRequestHandler):
        def send(self, payload, content_type):
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            stats["cookies"].append(self.headers.get("Cookie"))
            parsed = urllib.parse.urlsplit(self.path)
            path = urllib.parse.unquote(parsed.path)
            if path.startswith("/api/info/"):
                key = path.removeprefix("/api/info/")
                if key == "@public/group":
                    info = {"kind": "group", "attrs": {"name": "fixture"}}
                elif key == "@public/group/array":
                    info = array_info()
                elif key in {"@public/group/table", "@public/table"}:
                    info = table_info()
                else:
                    self.send_error(404)
                    return
                self.send(json.dumps(info).encode(), "application/json")
                return
            if path == "/api/list/@public/group":
                self.send(json.dumps(["array", "table"]).encode(), "application/json")
                return
            if path.startswith("/api/fetch/"):
                stats["fetches"] += 1
                key = path.removeprefix("/api/fetch/")
                query = urllib.parse.parse_qs(parsed.query)
                selection = query.get("slice_", [""])[0]
                if key == "@public/group/array":
                    axes = tuple(
                        slice(*(int(part) if part else None for part in axis.split(":")))
                        for axis in selection.split(",")
                    )
                    self.send(array.slice(axes).to_cframe(), "application/octet-stream")
                    return
                if key in {"@public/group/table", "@public/table"}:
                    start, stop = (int(part) for part in selection.split(":"))
                    result = table.slice(start, stop)
                    field = query.get("field", [None])[0]
                    if field is not None:
                        result = result.select([field])
                    self.send(result.to_cframe(), "application/octet-stream")
                    return
            self.send_error(404)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/", array, table, stats
    finally:
        server.shutdown()
        server.server_close()


def test_caterva2_group_arrays_tables_and_roundtrip(caterva2_source, tmp_path):
    urlbase, array, _, stats = caterva2_source
    source = blosc2.URLPath("@public/group", urlbase=urlbase)
    with blosc2.RemoteStore(source) as store:
        assert store.source["kind"] == "caterva2"
        assert store.keys() == ["array", "table"]
        with store["array"] as remote:
            np.testing.assert_array_equal(remote[1:3, 2:5], array[1:3, 2:5])
        with store["table"] as remote:
            np.testing.assert_array_equal(remote.slice(3, 6).ident[:], [3, 4, 5])
            np.testing.assert_array_equal(remote.value[2:4], [6, 9])
        reference = tmp_path / "group.b2z"
        store.save(reference, include_cache=False, mutable=False)

    stats["cookies"].clear()
    with blosc2.c2context(auth_token="ambient-secret"):
        with blosc2.open(reference) as reopened:
            assert reopened.source == {
                "kind": "caterva2",
                "version": 1,
                "urlbase": urlbase,
                "path": "@public/group",
                "assume_immutable": True,
            }
            with reopened["array"] as remote:
                np.testing.assert_array_equal(remote[0:1, 0:3], array[0:1, 0:3])
    assert stats["cookies"]
    assert set(stats["cookies"]) == {None}


def test_caterva2_table_dispatch_projection_and_roundtrip(caterva2_source, tmp_path):
    urlbase, _, _, stats = caterva2_source
    source = blosc2.URLPath("@public/table", urlbase=urlbase)
    with blosc2.open(source, lazy=True) as table:
        assert isinstance(table, blosc2.RemoteCTable)
        assert table.nrows == 12
        np.testing.assert_array_equal(table["value"][1:3], [3, 6])
        np.testing.assert_array_equal(table.select(["value"]).slice(4, 7).value[:], [12, 15, 18])
        with pytest.raises(NotImplementedError, match="bounded"):
            table.to_cframe()
        warm = stats["fetches"]
        np.testing.assert_array_equal(table.select(["value"]).slice(4, 7).value[:], [12, 15, 18])
        assert stats["fetches"] == warm
        reference = tmp_path / "table.b2z"
        table.save(reference, include_cache=False, mutable=False)

    with blosc2.open(reference) as reopened:
        assert isinstance(reopened, blosc2.RemoteCTable)
        np.testing.assert_array_equal(reopened.ident[1:3], [1, 2])


def test_caterva2_table_sparse_cache_survives_restart(caterva2_source, tmp_path):
    urlbase, _, _, stats = caterva2_source
    source = blosc2.URLPath("@public/table", urlbase=urlbase)
    cache = tmp_path / "cache"
    with blosc2.RemoteCTable.with_sparse_cache(source, cache) as table:
        np.testing.assert_array_equal(table.slice(2, 5).ident[:], [2, 3, 4])
    warm = stats["fetches"]
    with blosc2.RemoteCTable.with_sparse_cache(source, cache) as table:
        np.testing.assert_array_equal(table.slice(2, 5).ident[:], [2, 3, 4])
    # Discovery persisted the schema-preserving empty frame, and the first
    # handle persisted the nonempty result, so reopening performs no fetch.
    assert stats["fetches"] == warm


def test_caterva2_cache_identity_separates_credentials(caterva2_source, tmp_path):
    urlbase, _, _, _ = caterva2_source
    public = blosc2.URLPath("@public/group", urlbase=urlbase, auth_token="")
    private = blosc2.URLPath("@public/group", urlbase=urlbase, auth_token="secret")
    with blosc2.RemoteStore(public, cache_dir=tmp_path, cache_policy=blosc2.CachePolicy.DISK):
        pass
    with blosc2.RemoteStore(private, cache_dir=tmp_path, cache_policy=blosc2.CachePolicy.DISK):
        pass
    assert len([path for path in tmp_path.iterdir() if path.is_dir()]) == 2
