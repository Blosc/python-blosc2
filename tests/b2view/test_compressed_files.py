"""Local .b2 document carriers use bounded chunk decoding, not eager reads."""

import io

import numpy as np
import pytest
from tui_wait import wait_until

import blosc2
from blosc2.b2view.compressed_file import CompressedFile
from blosc2.b2view.file_preview import TEXT_BYTES
from blosc2.b2view.model import StoreBrowser


def write_carrier(path, payload, chunksize=4096):
    schunk = blosc2.SChunk(
        urlpath=str(path), mode="w", chunksize=chunksize, cparams={"typesize": 1}, contiguous=True
    )
    for start in range(0, len(payload), chunksize):
        schunk.append_data(payload[start : start + chunksize])


@pytest.mark.parametrize("file_url", [False, True])
def test_text_carrier_preview_is_bounded(tmp_path, monkeypatch, file_url):
    path = tmp_path / "README.md.b2"
    payload = b"a" * (TEXT_BYTES * 4)
    write_carrier(path, payload)
    calls = []
    original = CompressedFile._chunk

    def chunk(file, index):
        calls.append(index)
        return original(file, index)

    monkeypatch.setattr(CompressedFile, "_chunk", chunk)
    source = path.as_uri() if file_url else str(path)
    with StoreBrowser(source) as browser:
        info = browser.get_info("/")
        assert info.kind == "file"
        assert info.metadata["name"] == "README.md"
        assert info.metadata["carrier"] == "README.md.b2"
        assert not calls
        preview = browser.preview("/")
        assert len(preview["file_text"]) == TEXT_BYTES
        assert "truncated" in preview["notice"]
        assert calls == list(range(TEXT_BYTES // 4096))


def test_streamed_download_and_alias_lifetime(tmp_path, monkeypatch):
    path = tmp_path / "notes.txt.b2"
    payload = b"hello world\n" * 100_000
    write_carrier(path, payload, chunksize=1 << 20)
    file = CompressedFile(path)
    alias = file.alias()
    file.close()
    calls = []
    original = alias._chunk

    def chunk(index):
        calls.append(index)
        return original(index)

    monkeypatch.setattr(alias, "_chunk", chunk)
    destination = tmp_path / "notes.txt"
    progress = []
    alias.download(destination, progress=lambda done, total: progress.append((done, total)))
    assert destination.read_bytes() == payload
    assert calls == list(range(alias.nchunks))
    assert progress[-1] == (len(payload), len(payload))
    with pytest.raises(FileExistsError):
        alias.download(destination)
    with pytest.raises(InterruptedError):
        alias.download(tmp_path / "cancelled.txt", cancel=lambda: True)
    assert not (tmp_path / "cancelled.txt").exists()
    assert not list(tmp_path.glob(".b2view-download-*"))
    alias.close()


def test_oversized_chunk_rejected_before_decompression(tmp_path, monkeypatch):
    path = tmp_path / "large.txt.b2"
    payload = b"a" * ((16 << 20) + 1)
    write_carrier(path, payload, chunksize=len(payload))
    with StoreBrowser(str(path)) as browser:

        def forbidden(*args, **kwargs):
            pytest.fail("Oversized chunk was fetched/decompressed")

        monkeypatch.setattr(blosc2.SChunk, "get_chunk", forbidden)
        preview = browser.preview("/")
        assert preview["preview_status"] == "Preview failed"
        assert "16 MiB" in str(preview["preview_error"])
        with pytest.raises(ValueError, match="16 MiB"):
            browser.store.download(tmp_path / "large.txt")
    assert not (tmp_path / "large.txt").exists()
    assert not list(tmp_path.glob(".b2view-download-*"))


def test_compressed_size_limit_before_payload_copy(tmp_path, monkeypatch):
    path = tmp_path / "random.txt.b2"
    payload = np.random.default_rng(42).integers(0, 256, size=9 << 20, dtype=np.uint8).tobytes()
    write_carrier(path, payload, chunksize=len(payload))
    file = CompressedFile(path)

    def forbidden(*args, **kwargs):
        pytest.fail("Oversized compressed chunk was copied")

    monkeypatch.setattr(blosc2.SChunk, "get_chunk", forbidden)
    with pytest.raises(ValueError, match="8 MiB compressed"):
        file.read_bytes(0, 1)
    file.close()


def test_nonzero_byte_ranges_and_closed_handle(tmp_path):
    path = tmp_path / "note.txt.b2"
    payload = b"abcdefghijklmnopqrstuvwxyz" * 1000
    write_carrier(path, payload, chunksize=1024)
    file = CompressedFile(path)
    assert file.read_bytes(1000, 2100) == payload[1000:2100]
    assert file.read_bytes(len(payload), len(payload)) == b""
    file.close()
    with pytest.raises(RuntimeError, match="closed"):
        file.read_bytes(0, 1)


def test_structural_metadata_is_not_treated_as_document_bytes(tmp_path):
    path = tmp_path / "object.txt.b2"
    schunk = blosc2.SChunk(
        urlpath=str(path), mode="w", contiguous=True, chunksize=8, meta={"b2o": {"kind": "object"}}
    )
    schunk.append_data(b"payload")
    del schunk
    with pytest.raises(ValueError, match="serialized object"):
        CompressedFile(path)


def test_pdf_image_empty_and_native_classification(tmp_path):
    pdf = tmp_path / "doc.pdf.b2"
    write_carrier(pdf, b"%PDF-1.4\nExample")
    with StoreBrowser(str(pdf)) as browser:
        assert browser.preview("/")["preview_status"] == "Preview unavailable"
    pil = pytest.importorskip("PIL.Image")
    stream = io.BytesIO()
    pil.new("RGB", (20, 10), "red").save(stream, format="PNG")
    image = tmp_path / "image.png.b2"
    write_carrier(image, stream.getvalue())
    with StoreBrowser(str(image)) as browser:
        assert browser.preview("/")["file_image"].size == (20, 10)
    empty = tmp_path / "empty.txt.b2"
    write_carrier(empty, b"")
    with StoreBrowser(str(empty)) as browser:
        assert browser.preview("/")["file_text"] == ""
    plain = tmp_path / "data.b2"
    write_carrier(plain, b"plain native data")
    with StoreBrowser(str(plain)) as browser:
        assert browser.kind("/") == "schunk"
    masquerade = tmp_path / "array.md.b2"
    blosc2.asarray(np.arange(4), urlpath=str(masquerade))
    with pytest.raises(ValueError, match="SChunk"):
        StoreBrowser(str(masquerade))
    with StoreBrowser(str(tmp_path)) as browser:
        kinds = {node.name: node.kind for node in browser.list_children()}
        assert kinds["doc.pdf.b2"] == "file"
        assert kinds["data.b2"] == "schunk"


@pytest.mark.tui
@pytest.mark.asyncio
async def test_carrier_tui_download_uses_original_name(tmp_path):
    from textual.widgets import Input, Static

    from blosc2.b2view.app import B2ViewApp, FileTransferScreen

    carrier = tmp_path / "README.md.b2"
    payload = b"# Hello\n\nFrom a compressed carrier."
    write_carrier(carrier, payload)
    app = B2ViewApp(str(tmp_path), start_path="/README.md.b2")
    async with app.run_test(size=(120, 40)) as pilot:
        await wait_until(pilot, lambda: app._selected_info is not None and app._selected_info.kind == "file")
        assert app._selected_info.metadata["name"] == "README.md"
        app.action_download_file()
        await wait_until(pilot, lambda: isinstance(app.screen, FileTransferScreen))
        assert app.screen.query_one(Input).value.endswith("README.md")
        destination = tmp_path / "README.md"
        app.screen.query_one(Input).value = str(destination)
        await pilot.press("enter")
        await wait_until(
            pilot, lambda: "Saved" in str(app.screen.query_one("#file-status", Static).render())
        )
        assert destination.read_bytes() == payload
        await pilot.press("escape")
    app.wait_for_close()
