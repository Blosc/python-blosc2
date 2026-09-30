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


@dataclasses.dataclass
class ReadingBig:
    ident: int = blosc2.field(blosc2.int64(), chunks=(256,), blocks=(64,))
    value: int = blosc2.field(blosc2.int32(), chunks=(256,), blocks=(64,))


@dataclasses.dataclass
class ReadingRich:
    ident: int = blosc2.field(blosc2.int64())
    text: str = blosc2.field(blosc2.utf8(null_storage="mask"))
    category: str = blosc2.field(blosc2.dictionary(nullable=True))
    tags: list[int] = blosc2.field(blosc2.list(blosc2.int32(), nullable=True, batch_rows=128))  # noqa: RUF009
    when: object = blosc2.field(blosc2.timestamp(unit="ns", null_storage="mask"))


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
def caterva2_source(request):
    array = blosc2.asarray(np.arange(60, dtype=np.int32).reshape(6, 10), chunks=(3, 5), blocks=(1, 5))
    nrows = getattr(request, "param", 12)
    if nrows in ("rich", "rich-large"):
        rich_rows = 31 if nrows == "rich" else 1031
        category_boundary = 16 if nrows == "rich" else 1024
        table = blosc2.CTable(
            ReadingRich,
            [
                (
                    i,
                    None if i % 7 == 0 else "" if i % 7 == 1 else f"row-{i}",
                    None if i % 5 == 0 else "a" if i < category_boundary else "b",
                    None if i % 11 == 0 else [i, i + 1],
                    None if i % 13 == 0 else np.datetime64("2025-01-01", "ns") + np.timedelta64(i, "ns"),
                )
                for i in range(rich_rows)
            ],
            create_summary_index=False,
        )
    else:
        table = blosc2.CTable(
            Reading if nrows == 12 else ReadingBig,
            [(index, index * 3) for index in range(nrows)],
            create_summary_index=False,
        )
    stats = {
        "fetches": 0,
        "cookies": [],
        "fields": [],
        "ranges": [],
        "limit_rows": None,
        "fail_start": None,
        "short_start": None,
    }

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
                    stats["ranges"].append((start, stop))
                    if stats["fail_start"] == start:
                        self.send_error(503, "test batch failure")
                        return
                    if stats["limit_rows"] is not None and stop - start > stats["limit_rows"]:
                        self.send_error(400, "table selection exceeds configured slice byte limit")
                        return
                    result = table.slice(start, stop - (stats["short_start"] == start))
                    field = query.get("field", [None])[0]
                    stats["fields"].append(field)
                    if field is not None:
                        result = result.select([field])
                    self.send(result.to_cframe(), "application/octet-stream")
                    return
            self.send_error(404)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/", array, table, stats
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.network
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


@pytest.mark.network
def test_caterva2_table_dispatch_projection_and_roundtrip(caterva2_source, tmp_path):
    urlbase, _, _, stats = caterva2_source
    source = blosc2.URLPath("@public/table", urlbase=urlbase)
    with blosc2.open(source, lazy=True) as table:
        assert isinstance(table, blosc2.RemoteCTable)
        assert table.nrows == 12
        np.testing.assert_array_equal(table["value"][1:3], [3, 6])
        assert table[0].ident == 0
        assert "\n0" in str(table)
        assert [row.ident for row in table[:2]] == [0, 1]
        assert [row.ident for row in table] == list(range(12))
        assert table.slice(3, 1).nrows == 0
        np.testing.assert_array_equal(table.slice(slice(-3, None)).ident[:], [9, 10, 11])
        with table.select(["value"]) as projected:
            assert [column["name"] for column in projected.schema_dict()["columns"]] == ["value"]
            np.testing.assert_array_equal(projected.slice(4, 7).value[:], [12, 15, 18])
        assert stats["fields"][-1] == "value"
        local = table.materialize()
        assert isinstance(local, blosc2.CTable)
        assert local.nrows == table.nrows
        np.testing.assert_array_equal(local.value[:], np.arange(12) * 3)
        assert local.attrs["fixture"] == "local"
        with pytest.raises(NotImplementedError, match="bounded"):
            table.value.sum()
        with pytest.raises(NotImplementedError, match="bounded"):
            table.to_string()
        for suffix, export in (("b2z", table.to_b2z), ("b2d", table.to_b2d)):
            destination = tmp_path / f"materialized.{suffix}"
            export(destination)
            with blosc2.CTable.open(destination, mode="r") as persisted:
                np.testing.assert_array_equal(persisted.value[:], np.arange(12) * 3)
        with table.materialize(urlpath=tmp_path / "layout.b2d", chunks=8, blocks=4) as laid_out:
            assert laid_out._cols["ident"].chunks[0] == 8
            assert laid_out._cols["ident"].blocks[0] == 4
        warm = stats["fetches"]
        np.testing.assert_array_equal(table.select(["value"]).slice(4, 7).value[:], [12, 15, 18])
        assert stats["fetches"] == warm
        stale = table.select(["value"])
        table.refresh()
        with pytest.raises(RuntimeError, match="stale"):
            stale.slice(0, 1)
        stale.close()
        assert table[0].ident == 0
        reference = tmp_path / "table.b2z"
        table.save(reference, include_cache=False, mutable=False)

    with blosc2.open(reference) as reopened:
        assert isinstance(reopened, blosc2.RemoteCTable)
        np.testing.assert_array_equal(reopened.ident[1:3], [1, 2])


@pytest.mark.network
def test_caterva2_table_sparse_cache_survives_restart(caterva2_source, tmp_path):
    urlbase, _, _, stats = caterva2_source
    source = blosc2.URLPath("@public/table", urlbase=urlbase)
    cache = tmp_path / "cache"
    with blosc2.RemoteCTable.with_sparse_cache(source, cache) as table:
        np.testing.assert_array_equal(table.slice(2, 5).ident[:], [2, 3, 4])
    warm = stats["fetches"]
    with blosc2.RemoteCTable.with_sparse_cache(source, cache) as table:
        np.testing.assert_array_equal(table.slice(2, 5).ident[:], [2, 3, 4])
        assert stats["fetches"] == warm
        table.refresh()
        assert table.slice(2, 5).nrows == 3
    warm = stats["fetches"]
    with blosc2.RemoteCTable.with_sparse_cache(source, cache) as table:
        assert table.slice(2, 5).nrows == 3
    assert stats["fetches"] == warm


@pytest.mark.network
def test_caterva2_cache_identity_separates_credentials(caterva2_source, tmp_path):
    urlbase, _, _, _ = caterva2_source
    public = blosc2.URLPath("@public/group", urlbase=urlbase, auth_token="")
    private = blosc2.URLPath("@public/group", urlbase=urlbase, auth_token="secret")
    with blosc2.RemoteStore(public, cache_dir=tmp_path, cache_policy=blosc2.CachePolicy.DISK):
        pass
    with blosc2.RemoteStore(private, cache_dir=tmp_path, cache_policy=blosc2.CachePolicy.DISK):
        pass
    assert len([path for path in tmp_path.iterdir() if path.is_dir()]) == 2


@pytest.mark.network
def test_caterva2_table_export_keeps_cached_rows(caterva2_source, tmp_path):
    urlbase, _, _, stats = caterva2_source
    source = blosc2.URLPath("@public/table", urlbase=urlbase)
    artifact = tmp_path / "table.b2z"
    with blosc2.RemoteCTable(source) as table:
        table.slice(2, 4)
        with table.select(["value"]) as projected:
            projected.slice(4, 7)
        table.save(artifact, include_cache=True, mutable=False)
    manifest, _ = blosc2.RemoteStore._load_artifact_manifest(str(artifact))
    manifest["caterva2_frames"] = [{}]
    with pytest.raises(ValueError, match="frame key"):
        blosc2.RemoteStore._validate_artifact_manifest(manifest)
    warm = stats["fetches"]
    with blosc2.open(artifact) as reopened:
        assert reopened.slice(2, 4).nrows == 2
        with reopened.select(["value"]) as projected:
            assert projected.slice(4, 7).nrows == 3
    assert stats["fetches"] == warm
    with blosc2.RemoteCTable.with_sparse_cache(source, tmp_path / "cache", carrier=artifact) as attached:
        assert attached.slice(2, 4).nrows == 2
        with attached.select(["value"]) as projected:
            assert projected.slice(4, 7).nrows == 3
    assert stats["fetches"] == warm


@pytest.mark.network
def test_caterva2_cache_identity_separates_inherited_credentials(caterva2_source, tmp_path):
    urlbase, _, _, stats = caterva2_source
    source = blosc2.URLPath("@public/table", urlbase=urlbase)
    for token in ("account-A", "account-B"):
        before = stats["fetches"]
        with blosc2.c2context(auth_token=token):
            with blosc2.RemoteCTable(
                source, cache_dir=tmp_path, cache_policy=blosc2.CachePolicy.DISK
            ) as table:
                table.slice(0, 2)
                with blosc2.c2context(auth_token="another-account"):
                    table.slice(2, 3)
                assert stats["cookies"][-1] == token
        assert stats["fetches"] > before
        assert stats["cookies"][-1] == token
    assert len([path for path in tmp_path.iterdir() if path.is_dir()]) == 2


@pytest.mark.network
def test_caterva2_table_discovery_uses_injected_transport(caterva2_source, tmp_path, monkeypatch):
    import httpx

    urlbase, _, _, _ = caterva2_source
    source = blosc2.URLPath("@public/table", urlbase=urlbase)

    def forbidden_default():
        raise AssertionError("default HTTP transport used")

    monkeypatch.setattr(blosc2.c2array, "_sync_client", forbidden_default)
    with httpx.Client() as transport:
        with blosc2.RemoteStore.with_sparse_cache(source, tmp_path / "cache", _transport=transport) as store:
            with store[""] as table:
                assert table.nrows == 12
                assert table.slice(0, 1).nrows == 1


@pytest.mark.parametrize("caterva2_source", [41, pytest.param(4101, marks=pytest.mark.heavy)], indirect=True)
@pytest.mark.network
def test_caterva2_batches_large_slices_and_materialization(caterva2_source, tmp_path, monkeypatch):
    from blosc2 import remote_ctable

    urlbase, _, original, stats = caterva2_source
    nrows = original.nrows
    if nrows == 41:
        monkeypatch.setattr(remote_ctable, "CATERVA2_BATCH_ROWS", 32)
    start, stop = (5, 41) if nrows == 41 else (100, 1501)
    stats["limit_rows"] = limit = 16 if nrows == 41 else 600
    source = blosc2.URLPath("@public/table", urlbase=urlbase)
    with blosc2.RemoteCTable(source) as remote:
        selected = remote.slice(start, stop)
        np.testing.assert_array_equal(selected.ident[:], np.arange(start, stop))
        np.testing.assert_array_equal(remote.value[start:stop], np.arange(start, stop) * 3)
        assert all(hi - lo <= 1024 for lo, hi in stats["ranges"])
        assert any(hi - lo > limit for lo, hi in stats["ranges"])
        with remote.select(["value"]) as projected:
            local = projected.materialize()
            assert local.col_names == ["value"]
            np.testing.assert_array_equal(local.value[:], np.arange(nrows) * 3)
        destination = tmp_path / "full.b2z"
        with remote.materialize(urlpath=destination) as local:
            assert local.nrows == nrows
            np.testing.assert_array_equal(local.ident[nrows - 11 :], np.arange(nrows - 11, nrows))
        assert stats["ranges"][-1][1] == nrows
        rows = iter(remote)
        assert next(rows).ident == 0
        before = stats["fetches"]
        rows.close()
        assert stats["fetches"] == before
    with blosc2.CTable.open(destination, mode="r") as offline:
        assert offline.nrows == nrows


@pytest.mark.parametrize(
    "caterva2_source", ["rich", pytest.param("rich-large", marks=pytest.mark.heavy)], indirect=True
)
@pytest.mark.network
def test_caterva2_batches_preserve_types_and_nulls(caterva2_source, tmp_path):
    def assert_same_table(actual, expected):
        assert actual.nrows == expected.nrows
        assert actual.schema_dict()["columns"] == expected.schema_dict()["columns"]
        for name in expected.col_names:
            actual_values, expected_values = actual[name][:], expected[name][:]
            if isinstance(expected_values, np.ndarray):
                np.testing.assert_array_equal(actual_values, expected_values)
            else:
                assert actual_values == expected_values
        # Full row iteration repeatedly scans the validity mask. Compare all
        # values column-wise above, and sample row reconstruction at boundaries.
        for index in sorted(
            {0, expected.nrows - 1, min(1023, expected.nrows - 1), min(1024, expected.nrows - 1)}
        ):
            assert actual[index] == expected[index]

    urlbase, _, original, _ = caterva2_source
    source = blosc2.URLPath("@public/table", urlbase=urlbase)
    with blosc2.RemoteCTable(source) as remote:
        materialized = remote.materialize()
        assert_same_table(materialized, original)
        with remote.select(["category", "text"]) as projected:
            start, stop = original.nrows - 21, original.nrows
            result = projected.slice(start, stop)
            assert result.col_names == ["category", "text"]
            assert_same_table(result, original.select(["category", "text"]).slice(start, stop))
        destination = tmp_path / "rich.b2d"
        with remote.materialize(urlpath=destination) as disk:
            assert_same_table(disk, original)


@pytest.mark.network
def test_caterva2_failed_batch_preserves_destination(caterva2_source, tmp_path):
    import httpx

    urlbase, _, _, stats = caterva2_source
    stats["fail_start"] = 3
    stats["limit_rows"] = 4
    destination = tmp_path / "existing.b2z"
    destination.write_bytes(b"previous result")
    with blosc2.RemoteCTable(blosc2.URLPath("@public/table", urlbase=urlbase)) as remote:
        with pytest.raises(httpx.HTTPStatusError):
            remote.materialize(urlpath=destination, overwrite=True)
    assert destination.read_bytes() == b"previous result"


@pytest.mark.network
def test_caterva2_rejects_short_batch_and_single_row_limit(caterva2_source):
    import httpx

    urlbase, _, _, stats = caterva2_source
    with blosc2.RemoteCTable(blosc2.URLPath("@public/table", urlbase=urlbase)) as remote:
        stats["short_start"] = 0
        with pytest.raises(ValueError, match="incompatible row batch"):
            remote.slice(0, 2)
        stats["short_start"] = None
        stats["limit_rows"] = 0
        with pytest.raises(httpx.HTTPStatusError, match="400"):
            remote.materialize()
        assert stats["ranges"][-1] == (0, 1)


@pytest.mark.network
def test_caterva2_hierarchy_materialization(caterva2_source, tmp_path):
    urlbase, _, table, _ = caterva2_source
    source = blosc2.URLPath("@public/group", urlbase=urlbase)
    destination = tmp_path / "local-tree.b2z"
    with blosc2.RemoteStore(source) as store:
        store.materialize(destination)
    with blosc2.TreeStore(destination, mode="r") as local:
        assert local.attrs["name"] == "fixture"
        with local["table"] as copied:
            np.testing.assert_array_equal(copied.ident[:], table.ident[:])
        np.testing.assert_array_equal(local["array"][:], np.arange(60).reshape(6, 10))


def test_c2array_only_maps_http_404_to_missing():
    import httpx

    for status, expected in (
        (404, FileNotFoundError),
        (403, httpx.HTTPStatusError),
        (503, httpx.HTTPStatusError),
    ):

        def respond(request, status=status):
            return httpx.Response(status, request=request)

        client = httpx.Client(transport=httpx.MockTransport(respond))
        with client, pytest.raises(expected):
            blosc2.C2Array("@public/missing", urlbase="https://example.invalid/", _transport=client)
