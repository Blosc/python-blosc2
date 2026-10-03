"""Ordinary-file preview and explicit download/external-open UI tests."""

import io

import pytest
from test_remote_caterva2 import caterva2_source  # noqa: F401
from test_remote_file import file_source  # noqa: F401
from tui_wait import wait_until

from blosc2.b2view.file_preview import IMAGE_PIXELS, TEXT_BYTES, open_external, preview_file, safe_text
from blosc2.b2view.model import StoreBrowser


class BytesFile:
    def __init__(self, name, data):
        self.name, self.data = name, data
        self.nbytes = len(data)
        self.reads = []

    def read_bytes(self, start, stop):
        self.reads.append((start, stop))
        return self.data[start:stop]


def test_text_limits_controls_and_binary_fallback():
    file = BytesFile("README.md", b"a" * (TEXT_BYTES - 1) + "é".encode() + b"long content")
    result = preview_file(file)
    assert result["markdown"]
    assert "truncated" in result["notice"]
    assert "\ufffd" not in result["file_text"]
    assert file.reads == [(0, TEXT_BYTES)]
    assert safe_text("hello\x1b[31mred\x1b[0m\x1b]8;;https://evil\x07link\x1b]8;;\x07\0") == "helloredlink"
    assert "Binary" in preview_file(BytesFile("bad.md", b"a\0b"))["message"]
    assert "Invalid UTF-8" in preview_file(BytesFile("bad.txt", b"a\xffb"))["notice"]
    assert "truncated" in preview_file(BytesFile("many.txt", b"x\n" * 1001))["notice"]
    assert not preview_file(BytesFile("README.md", b"# Title"), raw=True)["markdown"]


def test_pdf_and_binary_do_not_fetch():
    for name in ("doc.pdf", "unknown.bin"):
        file = BytesFile(name, b"unknown payload")
        assert "download" in preview_file(file)["message"]
        assert file.reads == []


def test_missing_image_dependency_does_not_read(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "PIL", None)
    file = BytesFile("image.png", b"not needed")
    assert "needs Pillow" in preview_file(file)["message"]
    assert file.reads == []


def test_image_preview_and_limits(monkeypatch):
    pil = pytest.importorskip("PIL.Image")
    stream = io.BytesIO()
    pil.new("RGB", (20, 10), "blue").save(stream, format="PNG")
    result = preview_file(BytesFile("image.png", stream.getvalue()))
    assert result["file_image"].size == (20, 10)
    assert "20 × 10" in result["message"]
    assert "Invalid" in preview_file(BytesFile("image.jpg", b"not a JPEG"))["message"]
    monkeypatch.setattr("blosc2.b2view.file_preview.IMAGE_PIXELS", 10)
    assert "budget" in preview_file(BytesFile("image.png", stream.getvalue()))["message"]
    assert IMAGE_PIXELS <= (64 << 20) // 4


def test_external_open_is_shell_free_and_restricted(tmp_path, monkeypatch):
    import sys

    document = tmp_path / "--unsafe name.pdf"
    document.write_bytes(b"%PDF-1.4\n")
    seen = []
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr("subprocess.run", lambda args, **kw: seen.append((args, kw)))
    if sys.platform == "win32":
        pytest.skip("POSIX launcher contract")
    open_external(document)
    assert seen[0][0][1] == str(document.absolute())
    assert "shell" not in seen[0][1]
    script = tmp_path / "run.py"
    script.write_text("print('danger')")
    with pytest.raises(ValueError, match="restricted"):
        open_external(script)
    document.write_bytes(b"not pdf")
    with pytest.raises(ValueError, match="content"):
        open_external(document)


def test_browser_file_metadata_and_preview(file_source):  # noqa: F811
    base, _, stats = file_source
    with StoreBrowser(base) as browser:
        assert browser.kind("/README.md") == "file"
        info = browser.get_info("/README.md")
        assert info.display_path == "@public/README.md"
        assert info.metadata["name"] == "README.md"
        assert not stats["file_chunks"]
        assert browser.preview("/README.md")["markdown"]


@pytest.mark.tui
@pytest.mark.asyncio
async def test_tui_file_preview_and_download(file_source, tmp_path):  # noqa: F811
    from textual.widgets import Input, Static

    from blosc2.b2view.app import B2ViewApp, FileTransferScreen

    base, payload, _ = file_source
    app = B2ViewApp(base, start_path="/README.md")
    async with app.run_test(size=(120, 40)) as pilot:
        await wait_until(pilot, lambda: app._selected_info is not None and app._selected_info.kind == "file")
        assert app._selected_info.display_path == "@public/README.md"
        await pilot.press("T")
        assert app._file_raw
        await wait_until(pilot, lambda: not app._remote_page_pending)
        app.action_download_file()
        await wait_until(pilot, lambda: isinstance(app.screen, FileTransferScreen))
        destination = tmp_path / "README.md"
        app.screen.query_one(Input).value = str(destination)
        await pilot.press("enter")
        await wait_until(pilot, destination.exists)
        assert destination.read_bytes() == payload
        await wait_until(
            pilot, lambda: "Saved" in str(app.screen.query_one("#file-status", Static).render())
        )
        await pilot.press("escape")
    app.wait_for_close()


@pytest.mark.tui
@pytest.mark.asyncio
async def test_pdf_external_open_requires_consent(file_source, tmp_path, monkeypatch):  # noqa: F811
    from textual.widgets import Checkbox, Input, Static

    import blosc2
    from blosc2.b2view.app import B2ViewApp, FileTransferScreen

    base, _, stats = file_source
    data = b"%PDF-1.4\nDocument fixture"
    stream = blosc2.SChunk(chunksize=64, cparams={"typesize": 1})
    stream.append_data(data)
    stats["files"]["@public/doc.pdf"] = stream
    stats["groups"]["@public"].append("doc.pdf")
    launched = []
    monkeypatch.setattr("blosc2.b2view.file_preview.open_external", launched.append)
    app = B2ViewApp(base, start_path="/doc.pdf")
    async with app.run_test(size=(120, 40)) as pilot:
        await wait_until(pilot, lambda: app._selected_info is not None and app._selected_info.kind == "file")
        assert not stats["file_chunks"]
        app.action_open_file()
        await wait_until(pilot, lambda: isinstance(app.screen, FileTransferScreen))
        destination = tmp_path / "doc.pdf"
        app.screen.query_one(Input).value = str(destination)
        await pilot.press("enter")
        assert not destination.exists()
        assert not launched
        assert "consent" in str(app.screen.query_one("#file-status", Static).render())
        app.screen.query_one(Checkbox).value = True
        await pilot.press("enter")
        await wait_until(pilot, lambda: bool(launched))
        assert destination.read_bytes() == data
        await pilot.press("escape")
    app.wait_for_close()


@pytest.mark.tui
@pytest.mark.asyncio
@pytest.mark.parametrize("image_widget", [True, False])
async def test_image_widget_and_dependency_fallback(file_source, monkeypatch, image_widget):  # noqa: F811
    from textual.widgets import Static

    import blosc2
    from blosc2.b2view.app import B2ViewApp

    pil = pytest.importorskip("PIL.Image")
    base, _, stats = file_source
    encoded = io.BytesIO()
    pil.new("RGB", (20, 10), "red").save(encoded, format="PNG")
    stream = blosc2.SChunk(chunksize=1024, cparams={"typesize": 1})
    stream.append_data(encoded.getvalue())
    stats["files"]["@public/image.png"] = stream
    stats["groups"]["@public"].append("image.png")
    monkeypatch.setattr(
        "blosc2.b2view.app.TextualImage",
        (lambda image: Static(f"Image {image.size}")) if image_widget else None,
    )
    app = B2ViewApp(base, start_path="/image.png")
    async with app.run_test(size=(120, 40)) as pilot:
        if image_widget:
            await wait_until(pilot, lambda: bool(app.query_one("#file-image").children))
        else:
            await wait_until(
                pilot, lambda: "needs textual-image" in str(app.query_one("#preview", Static).render())
            )
        app.update_panels("/mount")
        await wait_until(
            pilot, lambda: app._selected_info is not None and app._selected_info.kind == "group"
        )
        await wait_until(pilot, lambda: not app.query_one("#file-image").children)
    app.wait_for_close()


@pytest.mark.tui
@pytest.mark.asyncio
async def test_slow_transfer_can_cancel_without_blocking_ui(file_source, tmp_path, monkeypatch):  # noqa: F811
    import threading

    from textual.widgets import Input

    import blosc2
    from blosc2.b2view.app import B2ViewApp, FileTransferScreen

    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    original = blosc2.RemoteFile._chunk

    def slow(file, index, cancel=None):
        if cancel is not None:
            entered.set()
            assert release.wait(5)
        try:
            return original(file, index, cancel)
        finally:
            if cancel is not None:
                finished.set()

    monkeypatch.setattr(blosc2.RemoteFile, "_chunk", slow)
    base, _, _ = file_source
    app = B2ViewApp(base, start_path="/README.md")
    destination = tmp_path / "cancelled.md"
    try:
        async with app.run_test(size=(120, 40)) as pilot:
            await wait_until(
                pilot, lambda: app._selected_info is not None and app._selected_info.kind == "file"
            )
            app.action_download_file()
            await wait_until(pilot, lambda: isinstance(app.screen, FileTransferScreen))
            app.screen.query_one(Input).value = str(destination)
            await pilot.press("enter")
            await wait_until(pilot, entered.is_set)
            await pilot.press("escape")
            assert not isinstance(app.screen, FileTransferScreen)
            await pilot.press("tab")
            release.set()
            await wait_until(pilot, finished.is_set)
            await wait_until(pilot, lambda: not list(tmp_path.glob(".b2view-download-*")))
            assert not destination.exists()
    finally:
        release.set()
        app.wait_for_close()
