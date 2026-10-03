"""Opt-in cross-project acceptance against an actual cat2lite server.

Run with CAT2LITE_SERVER=/absolute/path/to/cat2lite-server in the blosc2 env.
All upstream data is deterministic and served on loopback; no public network.
"""

import hashlib
import json
import os
import select
import subprocess
import sys
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

import numpy as np
import pytest

import blosc2

pytestmark = pytest.mark.skipif(
    not os.environ.get("CAT2LITE_SERVER"), reason="set CAT2LITE_SERVER for gateway acceptance"
)


@contextmanager
def gateway(base, catalog):
    config = base / "server.toml"
    config.write_text(
        f'[server]\nlisten = "127.0.0.1:0"\npython = {json.dumps(sys.executable)}\n'
        f"[remote]\ncache_dir = {json.dumps(str(base / 'server-cache'))}\n"
    )
    source_args = ["--data-dir", str(catalog)] if catalog.is_dir() else [str(catalog)]
    process = subprocess.Popen(
        [os.environ["CAT2LITE_SERVER"], "--config", str(config), *source_args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert select.select([process.stdout], [], [], 20)[0], "cat2lite startup timed out"
        line = process.stdout.readline()
        while line and not line.startswith("listening on "):
            line = process.stdout.readline()
        assert line.startswith("listening on "), line or process.stderr.read()
        address = line.removeprefix("listening on ").strip()
        yield address if address.startswith("http") else "http://" + address
    finally:
        process.terminate()
        process.wait(timeout=15)
        for stream in (process.stdout, process.stderr):
            stream.close()


def test_real_cat2lite_schunk_file_download(tmp_path):
    from blosc2.b2view.model import StoreBrowser

    data = tmp_path / "files"
    data.mkdir()
    payload = b"ordinary file bytes\n" * 10
    stream = blosc2.SChunk(chunksize=64, cparams={"typesize": 1})
    for offset in range(0, len(payload), 64):
        stream.append_data(payload[offset : offset + 64])
    (data / "bytes.b2frame").write_bytes(stream.to_cframe())
    with gateway(tmp_path, data) as url:
        with StoreBrowser(url) as browser:
            assert browser.kind("/bytes.b2frame") == "file"
            assert browser.get_info("/bytes.b2frame").metadata["nbytes"] == len(payload)
        with blosc2.open(url + "/@public/bytes.b2frame") as file:
            assert file.read_bytes(61, 135) == payload[61:135]
            file.download(tmp_path / "original.bin")
            assert (tmp_path / "original.bin").read_bytes() == payload


def test_real_catalog_browsing_and_shared_server_cache(tmp_path):
    h5py = pytest.importorskip("h5py")
    zarr = pytest.importorskip("zarr")
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    from blosc2.b2view.model import StoreBrowser

    data = tmp_path / "upstream"
    data.mkdir()
    expected = np.arange(60, dtype="i4").reshape(6, 10)
    with h5py.File(data / "hierarchy.h5", "w") as file:
        file.create_dataset("group/array", data=expected)
        file.create_dataset("scalar", data=np.int32(42))
    group = zarr.open_group(data / "hierarchy.zarr", mode="w", zarr_format=2)
    group.create_array("group/array", data=expected, chunks=(3, 5))
    zarr.consolidate_metadata(data / "hierarchy.zarr")
    pq.write_table(
        pa.table({"ident": list(range(20)), "value": [i * 3 for i in range(20)]}), data / "readings.parquet"
    )
    # Larger than small-source prefetch thresholds, with incompressible chunks.
    large = np.random.default_rng(7).integers(0, 2**31, (1024, 1024), dtype="i4")
    frame = blosc2.asarray(large, chunks=(128, 128), blocks=(32, 32)).to_cframe()
    (data / "large.b2nd").write_bytes(frame)
    files = {
        "/" + path.relative_to(data).as_posix(): path.read_bytes()
        for path in data.rglob("*")
        if path.is_file()
    }
    stats = {"large_bytes": 0, "large_gets": 0}

    class Source(BaseHTTPRequestHandler):
        def serve(self, body):
            path = urlsplit(self.path).path
            if path not in files:
                self.send_error(404)
                return
            payload = files[path]
            length = len(payload)
            offset, stop = 0, length
            requested = self.headers.get("Range")
            if requested:
                start, _, end = requested.removeprefix("bytes=").partition("-")
                offset = int(start) if start else max(0, length - int(end))
                stop = min(length, int(end) + 1) if start and end else length
            self.send_response(206 if requested else 200)
            self.send_header("Content-Length", str(stop - offset))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("ETag", '"' + hashlib.sha256(payload).hexdigest() + '"')
            if requested:
                self.send_header("Content-Range", f"bytes {offset}-{stop - 1}/{length}")
            self.end_headers()
            if body:
                if path == "/large.b2nd":
                    stats["large_bytes"] += stop - offset
                    stats["large_gets"] += 1
                self.wfile.write(payload[offset:stop])

        def do_GET(self):
            self.serve(True)

        def do_HEAD(self):
            self.serve(False)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Source)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        upstream = f"http://127.0.0.1:{server.server_port}"
        catalog = tmp_path / "repo.catl"
        catalog.write_text(
            "entries:\n"
            + "".join(
                f"  {name}: {upstream}/{source}\n"
                for name, source in [
                    ("hdf5", "hierarchy.h5"),
                    ("zarr", "hierarchy.zarr"),
                    ("table", "readings.parquet"),
                    ("large", "large.b2nd"),
                ]
            )
        )
        with gateway(tmp_path, catalog) as url:
            with StoreBrowser(url) as browser:
                assert [node.name for node in browser.list_children()] == ["hdf5", "large", "table", "zarr"]
                assert browser.list_children("/hdf5")[0].name == "group"
                assert browser.get_info("/hdf5/scalar").metadata["shape"] == ()
                np.testing.assert_array_equal(
                    browser.preview("/hdf5/group/array", slices=(slice(2), slice(3))), expected[:2, :3]
                )
                np.testing.assert_array_equal(
                    browser.preview("/zarr/group/array", slices=(slice(2), slice(3))), expected[:2, :3]
                )
                assert list(browser.preview("/table", stop=3)["data"]["ident"]) == [0, 1, 2]
            import asyncio

            from blosc2.b2view.app import B2ViewApp

            async def viewer_acceptance():
                app = B2ViewApp(url, start_path="/hdf5/group/array")
                async with app.run_test(size=(120, 40)) as pilot:
                    for _ in range(200):
                        if app.table_page and app.table_page["columns"]:
                            break
                        await pilot.pause(0.02)
                    assert app.selected_path == "/hdf5/group/array"
                    assert app.table_page
                    assert app.table_page["columns"]
                    np.testing.assert_array_equal(
                        app.table_page["data"]["0"], expected[: app.table_page["stop"], 0]
                    )
                app.wait_for_close()

            asyncio.run(viewer_acceptance())
            code = """
import sys
import numpy as np
import blosc2
with blosc2.open(sys.argv[1] + '/@public/large', cache_policy=blosc2.CachePolicy.NONE) as array:
    value = array[:2, :3]
    assert value.shape == (2, 3)
    print(value.tolist())
"""

            def independent_read():
                result = subprocess.run(
                    [sys.executable, "-c", code, url], capture_output=True, text=True, timeout=30
                )
                assert result.returncode == 0, result.stderr
                assert json.loads(result.stdout) == large[:2, :3].tolist()

            independent_read()
            warm = stats.copy()
            independent_read()
            assert stats == warm, "second no-cache process should reuse server payload"
            assert warm["large_bytes"] < large.nbytes // 2
        with gateway(tmp_path, catalog) as url:
            independent_read()
            # A restart may re-read source headers, but not the warm data chunk.
            assert stats["large_bytes"] - warm["large_bytes"] < 16 << 10
        print("gateway cache evidence:", warm, "after restart:", stats)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
