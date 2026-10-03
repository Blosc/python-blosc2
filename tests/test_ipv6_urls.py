"""IPv6 authorities must not be mistaken for fsspec chains or member selectors."""

import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

import blosc2
from blosc2.core import find_url_separator, fsspec_filesystem, parse_container_url
from blosc2.remote_array import validate_persistable_url


@pytest.mark.parametrize("host", ["[::1]", "[2001:db8::1]", "[::ffff:127.0.0.1]", "[fe80::1%25en0]"])
def test_ipv6_persistable_url_and_ref(host):
    url = f"https://{host}:8000/array.b2nd?version=1"
    assert find_url_separator(url) == -1
    validate_persistable_url(url)
    assert blosc2.Ref.fsspec_ref(url).urlpath == url


@pytest.mark.parametrize(
    "url",
    [
        "zip://a.b2nd::https://[::1]/archive.zip",
        "simplecache::https://[::1]/a.b2nd",
        "https://[::1]/a.b2nd::memory://other",
        "https://[::1]/a.b2nd::/member",
        "https://[::1]/a::b.b2nd",
    ],
)
def test_real_separator_still_rejected(url):
    assert find_url_separator(url) != -1
    with pytest.raises(ValueError, match="chained fsspec"):
        validate_persistable_url(url)


@pytest.mark.parametrize("suffix", ["#fragment", "?token=secret", "?sig=secret"])
def test_ipv6_does_not_bypass_sensitive_url_checks(suffix):
    with pytest.raises(ValueError):
        validate_persistable_url("https://[::1]/a.b2nd" + suffix)
    with pytest.raises(ValueError, match="user information"):
        validate_persistable_url("https://user:password@[::1]/a.b2nd")


@pytest.mark.parametrize(("extension", "kind"), [("b2z", "b2z"), ("h5", "hdf5"), ("hdf5", "hdf5")])
def test_ipv6_container_selectors(extension, kind):
    base = f"http://[::1]:8000/store.{extension}"
    assert parse_container_url(base) == (base, None, kind)
    assert parse_container_url(base + "::/group/array") == (base, "group/array", kind)
    assert parse_container_url(base + "/group/array?version=1") == (base + "?version=1", "group/array", kind)
    assert parse_container_url(base, dataset="group/array") == (base, "group/array", kind)
    with pytest.raises(ValueError, match="both URL path"):
        parse_container_url(base + "::/group/array", dataset="other")


def test_ipv6_hdf5_direct_parser():
    from blosc2.hdf5_source import HDF5NDSource

    base = "http://[::1]:8000/store.h5"
    assert HDF5NDSource._parse_url(base, "group/array") == (base, "group/array")
    assert HDF5NDSource._parse_url(base + "::/group/array", None) == (base, "group/array")


def test_ipv6_filesystem_preserves_url_and_options():
    pytest.importorskip("fsspec")
    url = "http://[::1]:8000/array.b2nd?version=1"
    options = {"http": {"headers": {"X-Test": "ipv6"}}, "skip_instance_cache": True}
    fs, path = fsspec_filesystem(url, options)
    assert path == url
    assert fs.kwargs["headers"] == {"X-Test": "ipv6"}
    assert fsspec_filesystem(url, options)[0] is not fs
    assert fsspec_filesystem("memory://ipv6-test/array.b2nd")[1] == "/ipv6-test/array.b2nd"


@pytest.mark.network
def test_ipv6_http_array_and_store_roundtrip(tmp_path):
    pytest.importorskip("fsspec")
    pytest.importorskip("aiohttp")

    class Server(ThreadingHTTPServer):
        address_family = socket.AF_INET6

    data = np.arange(100, dtype=np.int32)
    archive = tmp_path / "source.b2z"
    with blosc2.TreeStore(archive, mode="w", threshold=0) as tree:
        tree["group/array"] = blosc2.asarray(data)
    files = {"/array.b2nd": blosc2.asarray(data).to_cframe(), "/source.b2z": archive.read_bytes()}

    class Handler(BaseHTTPRequestHandler):
        def respond(self, body):
            payload = files[self.path]
            range_ = self.headers.get("Range")
            if range_:
                start, end = range_.removeprefix("bytes=").split("-")
                first = int(start) if start else max(0, len(payload) - int(end))
                last = int(end) if start and end else len(payload) - 1
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {first}-{last}/{len(payload)}")
                payload = payload[first : last + 1]
            else:
                self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            if body:
                self.wfile.write(payload)

        def do_HEAD(self):
            self.respond(False)

        def do_GET(self):
            self.respond(True)

        def log_message(self, *_args):
            pass

    try:
        server = Server(("::1", 0), Handler)
    except OSError as error:
        pytest.skip(f"IPv6 loopback unavailable: {error}")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://[::1]:{server.server_port}"
        with blosc2.RemoteArray(base + "/array.b2nd") as array:
            np.testing.assert_array_equal(array[1:4], data[1:4])
            array.save(tmp_path / "reference.b2nd")
        with blosc2.open(tmp_path / "reference.b2nd", deserialize="full") as reopened:
            np.testing.assert_array_equal(reopened[2:5], data[2:5])
        with blosc2.open(base + "/source.b2z::/group/array") as member:
            np.testing.assert_array_equal(member[1:4], data[1:4])
        with blosc2.RemoteStore(base + "/source.b2z") as store:
            with store["group/array"] as member:
                np.testing.assert_array_equal(member[2:5], data[2:5])
        with blosc2.core.fsspec_open(base + "/array.b2nd", "rb") as file:
            assert file.read() == files["/array.b2nd"]
        local = blosc2.core.localize_fsspec_url(base + "/source.b2z", tmp_path / "cache")
        assert Path(local).read_bytes() == files["/source.b2z"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
