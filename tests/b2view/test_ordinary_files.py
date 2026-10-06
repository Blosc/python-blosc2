"""Direct ordinary files share Caterva2's passive previews and explicit actions."""

import functools
import http.server
import io
import threading

import numpy as np
import pytest
from tui_wait import wait_until

import blosc2
from blosc2.b2view.model import StoreBrowser
from blosc2.b2view.ordinary_file import MAX_READ_BYTES, OrdinaryFile, local_path, ordinary_source


@pytest.mark.parametrize("name", ["README.md", "a space.md", "literal%20.md", "café.md"])
def test_local_file_url_roundtrip(tmp_path, name):
    path = tmp_path / name
    assert local_path(path.as_uri()) == path
    assert local_path(path.as_uri().replace("file:///", "file://localhost/", 1)) == path
    assert local_path(str(path)) == path


@pytest.fixture
def ordinary_http(tmp_path):
    if blosc2.IS_WASM:
        pytest.skip("emscripten cannot run a threaded HTTP server")

    class Handler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(Handler, directory=str(tmp_path))
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize("backend", ["local", "file", "memory", "http"])
def test_direct_file_metadata_preview_and_copy(tmp_path, ordinary_http, backend):
    payload = b"# Hello\n\nOrdinary **Markdown**.\n"
    path = tmp_path / "README.md"
    path.write_bytes(payload)
    if backend == "memory":
        fs = pytest.importorskip("fsspec").filesystem("memory")
        source = "memory://b2view-ordinary/README.md"
        fs.pipe(source, payload)
    else:
        source = (
            ordinary_http + path.name
            if backend == "http"
            else path.as_uri()
            if backend == "file"
            else str(path)
        )
    with StoreBrowser(source) as browser:
        assert not browser.is_tree
        assert browser.kind("/") == "file"
        metadata = browser.get_info("/").metadata
        assert metadata["name"] == "README.md"
        assert metadata["nbytes"] == len(payload)
        assert "cbytes" not in metadata
        assert browser.preview("/")["markdown"]
        assert not browser.preview("/", raw_text=True)["markdown"]
        alias = browser.store.alias()
    destination = tmp_path / "copy.md"
    progress = []
    alias.download(destination, progress=lambda done, total: progress.append((done, total)))
    assert destination.read_bytes() == payload
    assert progress[-1] == (len(payload), len(payload))
    with pytest.raises(FileExistsError):
        alias.download(destination)
    alias.close()
    with pytest.raises(RuntimeError, match="closed"):
        alias.read_bytes(0, 1)


def test_bounds_cancellation_and_size_changes(tmp_path):
    source = tmp_path / "large.txt"
    with source.open("wb") as stream:
        stream.truncate(MAX_READ_BYTES + 1)
    file = OrdinaryFile(source)
    with pytest.raises(ValueError, match="16 MiB"):
        file.read_bytes()
    with pytest.raises(ValueError, match="interval"):
        file.read_bytes(-1, 2)
    assert file.read_bytes(4, 8) == b"\0" * 4
    destination = tmp_path / "copy.txt"
    with pytest.raises(InterruptedError):
        file.download(destination, cancel=lambda: True)
    assert not destination.exists()
    assert not list(tmp_path.glob(".b2view-download-*"))
    source.write_bytes(b"short")
    with pytest.raises(OSError, match="short read"):
        file.download(destination)
    assert not destination.exists()
    assert not list(tmp_path.glob(".b2view-download-*"))
    with pytest.raises(ValueError, match="differ"):
        file.download(source, overwrite=True)


def test_preview_handles_partial_stream_reads(tmp_path, monkeypatch):
    path = tmp_path / "text.txt"
    path.write_bytes(b"hello world")
    file = OrdinaryFile(path)

    class Partial(io.BytesIO):
        def read(self, size=-1):
            return super().read(min(size, 2))

    monkeypatch.setattr(file, "_stream", lambda **kwargs: Partial(b"hello world"))
    assert file.read_bytes(0, 11) == b"hello world"


def test_http_large_text_preview_remains_bounded(tmp_path, ordinary_http):
    from blosc2.b2view.file_preview import TEXT_BYTES

    (tmp_path / "large.txt").write_bytes(b"a" * (TEXT_BYTES * 8))
    with StoreBrowser(ordinary_http + "large.txt") as browser:
        preview = browser.preview("/")
        assert len(preview["file_text"]) == TEXT_BYTES
        assert "truncated" in preview["notice"]


def test_empty_unknown_pdf_and_missing_files(tmp_path):
    for name in ("empty.txt", "data.bin", "brochure.pdf"):
        path = tmp_path / name
        path.touch()
        with StoreBrowser(str(path)) as browser:
            result = browser.preview("/")
            if name == "empty.txt":
                assert result["file_text"] == ""
            else:
                assert result["preview_status"] == "Preview unavailable"
            copy = tmp_path / ("copy-" + name)
            browser.store.download(copy)
            assert copy.read_bytes() == b""
    with pytest.raises(FileNotFoundError):
        StoreBrowser(str(tmp_path / "missing.txt"))
    with pytest.raises(ValueError, match="regular file"):
        OrdinaryFile(tmp_path)


def test_chained_fsspec_file_and_storage_options(tmp_path, monkeypatch):
    pytest.importorskip("fsspec")
    import zipfile

    from blosc2.b2view import ordinary_file

    payload = b"# Archive member\n"
    archive = tmp_path / "docs.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("notes/README.md", payload)
    with StoreBrowser(f"zip://notes/README.md::{archive.as_uri()}") as browser:
        assert browser.get_info("/").metadata["name"] == "README.md"
        assert browser.preview("/")["file_text"] == payload.decode()
        browser.store.download(tmp_path / "copied.md")
    fs = pytest.importorskip("fsspec").filesystem("memory")
    fs.pipe("memory://option-test/README.md", payload)
    original = ordinary_file.fsspec_filesystem
    seen = []

    def filesystem(source, options):
        seen.append(options)
        return original(source, options)

    monkeypatch.setattr(ordinary_file, "fsspec_filesystem", filesystem)
    with StoreBrowser("memory://option-test/README.md", storage_options={"custom": "value"}):
        assert seen
        assert all(options == {"custom": "value"} for options in seen)


def test_late_cancellation(tmp_path):
    source = tmp_path / "source.txt"
    source.write_bytes(b"hello")
    file = OrdinaryFile(source)
    destination = tmp_path / "destination.txt"
    cancelled = False

    def progress(done, total):
        nonlocal cancelled
        cancelled = True

    with pytest.raises(InterruptedError):
        file.download(destination, progress=progress, cancel=lambda: cancelled)
    assert not destination.exists()
    assert not list(tmp_path.glob(".b2view-download-*"))


@pytest.mark.skipif(blosc2.IS_WASM, reason="emscripten has no hard links")
def test_publication_race(tmp_path, monkeypatch):
    source = tmp_path / "source.txt"
    source.write_bytes(b"hello")
    file = OrdinaryFile(source)
    destination = tmp_path / "destination.txt"
    original = __import__("os").link

    def race(temporary, target):
        destination.write_bytes(b"existing")
        return original(temporary, target)

    monkeypatch.setattr("os.link", race)
    with pytest.raises(FileExistsError):
        file.download(destination)
    assert destination.read_bytes() == b"existing"
    assert not list(tmp_path.glob(".b2view-download-*"))


@pytest.mark.parametrize("competing_writer", [False, True])
def test_wasm_download_publication(tmp_path, monkeypatch, competing_writer):
    monkeypatch.setattr(blosc2, "IS_WASM", True)
    source = tmp_path / "source.txt"
    source.write_bytes(b"hello")
    destination = tmp_path / "destination.txt"
    file = OrdinaryFile(source)

    def progress(done, total):
        if competing_writer:
            destination.write_bytes(b"existing")

    if competing_writer:
        with pytest.raises(FileExistsError):
            file.download(destination, progress=progress)
        assert destination.read_bytes() == b"existing"
    else:
        file.download(destination, progress=progress)
        assert destination.read_bytes() == b"hello"
        with pytest.raises(FileExistsError):
            file.download(destination)
    assert not list(tmp_path.glob(".b2view-download-*"))


@pytest.mark.parametrize(
    "source",
    [
        "a.b2nd",
        "broken.b2z",
        "a.h5/group",
        "a.zarr",
        "a.parquet",
        "https://host/demo",
        "https://host/demo/@public/README.md",
        "memory://a.b2nd",
        "zip://a.b2nd::memory://archive.zip",
    ],
)
def test_dataset_and_service_routing_unchanged(source):
    assert not ordinary_source(source)


@pytest.mark.parametrize(
    "source",
    ["https://service.example.com", "https://service.example.com/", "http://service.example.com:8080"],
)
def test_bare_http_origins_keep_service_discovery(source):
    assert not ordinary_source(source)
    assert ordinary_source(source, remote_service="fsspec")
    assert ordinary_source(source + "/README.md")


def test_ipv6_http_and_explicit_fsspec_routing():
    assert ordinary_source("http://[::1]/README.md")
    assert ordinary_source("http://[::1]/no-extension", "fsspec")
    assert ordinary_source("https://host/@public/README.md", "fsspec")


def test_native_array_and_corrupt_dataset_keep_native_handling(tmp_path):
    path = tmp_path / "array.b2nd"
    blosc2.asarray(np.arange(10), urlpath=str(path))
    with StoreBrowser(str(path)) as browser:
        assert browser.kind("/") == "ndarray"
    broken = tmp_path / "broken.b2nd"
    broken.write_bytes(b"not a dataset")
    with pytest.raises(RuntimeError):
        StoreBrowser(str(broken))


@pytest.mark.parametrize("backend", ["local", "memory"])
def test_native_frames_with_nonstandard_names_still_open(tmp_path, backend):
    array = blosc2.asarray(np.arange(10))
    if backend == "memory":
        fs = pytest.importorskip("fsspec").filesystem("memory")
        source = "memory://b2view-ordinary/array.bin"
        fs.pipe(source, array.to_cframe())
    else:
        path = tmp_path / "array"
        path.write_bytes(array.to_cframe())
        source = str(path)
    with StoreBrowser(source) as browser:
        assert browser.kind("/") == "ndarray"


@pytest.mark.tui
@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["local", "memory"])
async def test_file_tui_toggle_copy_and_external_open(tmp_path, monkeypatch, backend):
    from textual.widgets import Input, Static

    from blosc2.b2view.app import B2ViewApp, FileTransferScreen

    payload = b"# Hello\n\nLocal and remote files share the viewer."
    if backend == "memory":
        fs = pytest.importorskip("fsspec").filesystem("memory")
        source = "memory://b2view-tui/README.md"
        fs.pipe(source, payload)
    else:
        path = tmp_path / "README.md"
        path.write_bytes(payload)
        source = str(path)
    launched = []
    monkeypatch.setattr("blosc2.b2view.file_preview.open_external", launched.append)
    app = B2ViewApp(source)
    async with app.run_test(size=(120, 40)) as pilot:
        await wait_until(
            pilot, lambda: "T: raw/Markdown" in str(app.query_one("#data-header", Static).render())
        )
        await pilot.press("T")
        await wait_until(pilot, lambda: "# Hello" in str(app.query_one("#preview", Static).render()))
        for external in (False, True):
            app._file_action(external)
            await wait_until(pilot, lambda: isinstance(app.screen, FileTransferScreen))
            destination = tmp_path / ("opened.md" if external else "saved.md")
            app.screen.query_one(Input).value = str(destination)
            await pilot.press("enter")
            await wait_until(
                pilot, lambda: "Saved" in str(app.screen.query_one("#file-status", Static).render())
            )
            assert destination.read_bytes() == payload
            await pilot.press("escape")
        assert len(launched) == 1
    app.wait_for_close()


@pytest.mark.tui
@pytest.mark.asyncio
async def test_local_image_uses_background_image_widget(tmp_path, monkeypatch):
    from textual.widgets import Static

    from blosc2.b2view.app import B2ViewApp

    image = pytest.importorskip("PIL.Image")
    path = tmp_path / "image.png"
    image.new("RGB", (20, 10), "red").save(path)
    monkeypatch.setattr("blosc2.b2view.app.TextualImage", lambda image: Static("Image"))
    app = B2ViewApp(str(path))
    async with app.run_test() as pilot:
        await wait_until(pilot, lambda: bool(app.query_one("#file-image").children))
        assert app._selected_info.kind == "file"
    app.wait_for_close()


@pytest.mark.tui
@pytest.mark.asyncio
async def test_missing_local_file_exits_gracefully(tmp_path):
    from blosc2.b2view.app import B2ViewApp

    app = B2ViewApp(str(tmp_path / "missing.txt"))
    async with app.run_test() as pilot:
        await wait_until(pilot, lambda: app.return_code == 1)
        assert "FileNotFoundError" in app.startup_error
    app.wait_for_close()
