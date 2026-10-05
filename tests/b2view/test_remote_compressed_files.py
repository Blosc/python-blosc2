"""Range-only remote document carriers share native previews and transfers."""

import http.server
import io
import json
import struct
import threading

import numpy as np
import pytest

import blosc2
from blosc2.b2view.compressed_file import CompressedFile
from blosc2.b2view.model import StoreBrowser
from blosc2.proxy_source import ByteRangeSChunkSource

fsspec = pytest.importorskip("fsspec")


def carrier(payload, chunksize=4096, **kwargs):
    stream = blosc2.SChunk(chunksize=chunksize, cparams={"typesize": 1}, **kwargs)
    for start in range(0, len(payload), chunksize):
        stream.append_data(payload[start : start + chunksize])
    return stream.to_cframe()


def test_remote_text_preview_reads_only_needed_chunks(monkeypatch, tmp_path):
    payload = b"a" * (256 << 10)
    fs = fsspec.filesystem("memory")
    fs.pipe("/remote-documents/README.md.b2", carrier(payload))
    calls = []
    original = ByteRangeSChunkSource.get_chunk

    def chunk(source, index):
        calls.append(index)
        return original(source, index)

    monkeypatch.setattr(ByteRangeSChunkSource, "get_chunk", chunk)
    with StoreBrowser("memory:///remote-documents") as directory:
        assert directory.list_children()[0].kind == "file"
        info = directory.get_info("/README.md.b2")
        assert info.metadata["carrier"] == "README.md.b2"
        assert info.metadata["name"] == "README.md"
        assert not calls
        assert len(directory.preview("/README.md.b2")["file_text"]) == 64 << 10
        assert calls == list(range(16))
    file = CompressedFile("memory:///remote-documents/README.md.b2")
    alias = file.alias()
    file.close()
    try:
        destination = tmp_path / "README.md"
        calls.clear()
        alias.download(destination)
        assert destination.read_bytes() == payload
        assert calls == list(range(64))
    finally:
        alias.close()


@pytest.mark.parametrize("name", ["empty.txt.b2", "doc.pdf.b2", "image.png.b2", "example.ipynb.b2"])
def test_remote_document_previews(name):
    if name.startswith("image"):
        pil = pytest.importorskip("PIL.Image")
        stream = io.BytesIO()
        pil.new("RGB", (20, 10), "red").save(stream, format="PNG")
        payload = stream.getvalue()
    elif name.startswith("example"):
        payload = json.dumps({"nbformat": 4, "cells": []}).encode()
    else:
        payload = b"%PDF-1.4\n" if name.startswith("doc") else b""
    source = "memory:///remote-types/" + name
    fsspec.filesystem("memory").pipe(source, carrier(payload))
    with StoreBrowser(source) as browser:
        preview = browser.preview("/")
        if name.startswith("image"):
            assert preview["file_image"].size == (20, 10)
        elif name.startswith("example"):
            assert preview["notebook_cells"] == []
        elif name.startswith("doc"):
            assert preview["preview_status"] == "Preview unavailable"
        else:
            assert preview["file_text"] == ""


def test_remote_zero_chunks_and_nonzero_byte_ranges():
    source = "memory:///zero.txt.b2"
    payload = b"\0" * 10000
    fsspec.filesystem("memory").pipe(source, carrier(payload))
    file = CompressedFile(source)
    try:
        assert file.read_bytes(4090, 9000) == payload[4090:9000]
    finally:
        file.close()


def test_remote_carriers_refuse_structural_metadata_and_large_chunks(monkeypatch):
    fs = fsspec.filesystem("memory")
    fs.pipe("/objects.txt.b2", carrier(b"bytes", meta={"b2o": {"kind": "object"}}))
    with pytest.raises(ValueError, match="serialized object"):
        CompressedFile("memory:///objects.txt.b2")
    fs.pipe("/array.txt.b2", blosc2.asarray(np.arange(4)).to_cframe())
    with pytest.raises(ValueError, match="SChunk"):
        CompressedFile("memory:///array.txt.b2")
    frame = bytearray(carrier(b"bytes"))
    header_len = struct.unpack_from(">i", frame, 11)[0]
    for position, value, limit in ((4, (256 << 20) + 1, "256 MiB"), (12, (32 << 20) + 1, "32 MiB")):
        broken = frame.copy()
        struct.pack_into("<I", broken, header_len + position, value)
        fs.pipe("/large.txt.b2", broken)
        file = CompressedFile("memory:///large.txt.b2")
        with monkeypatch.context() as patch:
            patch.setattr(
                ByteRangeSChunkSource, "get_chunk", lambda *args: pytest.fail("Oversized payload fetched")
            )
            with pytest.raises(ValueError, match=limit):
                file.read_bytes(0, 1)
        file.close()


@pytest.fixture
def document_http():
    if blosc2.IS_WASM:
        pytest.skip("emscripten cannot run a threaded HTTP server")
    state = {"frame": carrier(b"# Hello\n" * 10000), "mode": "range", "requests": []}

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_HEAD(self):
            self.send_response(200)
            self.send_header("Content-Length", str(len(state["frame"])))
            self.end_headers()

        def do_GET(self):
            value = self.headers.get("Range")
            state["requests"].append((value, self.headers.get("Authorization")))
            if state["mode"] == "ignore":
                self.send_response(200)
                self.send_header("Content-Length", str(len(state["frame"])))
                self.end_headers()
                return
            start, stop = map(int, value.removeprefix("bytes=").split("-"))
            data = state["frame"][start : stop + 1]
            self.send_response(206)
            span = f"bytes {start}-{stop}/{len(state['frame'])}"
            self.send_header("Content-Range", "bytes 0-0/1" if state["mode"] == "wrong" else span)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/README.md.b2?signature=test", state
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_http_document_ranges_and_transfer(document_http, tmp_path):
    source, state = document_http
    with StoreBrowser(source, storage_options={"headers": {"Authorization": "Bearer test"}}) as browser:
        assert len(state["requests"]) == 2
        assert browser.get_info("/").metadata["name"] == "README.md"
        assert not state["requests"][0][0].endswith(str(len(state["frame"]) - 1))
        assert browser.preview("/")["file_text"].startswith("# Hello")
        alias = browser.store.alias()
    try:
        destination = tmp_path / "README.md"
        alias.download(destination)
        assert destination.read_bytes() == b"# Hello\n" * 10000
    finally:
        alias.close()
    assert all(value and auth == "Bearer test" for value, auth in state["requests"])


@pytest.mark.parametrize("mode", ["ignore", "wrong"])
def test_http_document_refuses_non_range_servers(document_http, mode):
    source, state = document_http
    state["mode"] = mode
    with pytest.raises(ValueError, match="byte-range support"):
        StoreBrowser(source)
