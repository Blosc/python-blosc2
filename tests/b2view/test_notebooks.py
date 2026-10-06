"""Passive notebook previews share file transports without executing kernels."""

import base64
import io
import json

import pytest
from test_remote_caterva2 import caterva2_source  # noqa: F401
from test_remote_file import file_source  # noqa: F401
from tui_wait import wait_until

import blosc2
from blosc2.b2view.file_preview import TEXT_BYTES, TEXT_LINES, preview_file
from blosc2.b2view.model import StoreBrowser
from blosc2.b2view.notebook_preview import NOTEBOOK_BYTES, NOTEBOOK_CELLS
from blosc2.b2view.render import make_preview_renderables


class NotebookFile:
    name = "example.ipynb"

    def __init__(self, notebook):
        self.data = json.dumps(notebook).encode()
        self.nbytes = len(self.data)
        self.reads = []

    def read_bytes(self, start, stop):
        self.reads.append((start, stop))
        return self.data[start:stop]


def notebook(cells):
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {"language_info": {"name": "python"}},
        "cells": cells,
    }


def sample_notebook():
    return notebook(
        [
            {"cell_type": "markdown", "source": ["# Title\n", "Some **Markdown**."], "metadata": {}},
            {
                "cell_type": "code",
                "source": "raise RuntimeError('never execute')",
                "metadata": {},
                "outputs": [
                    {"output_type": "stream", "text": "hello\x1b[31m red\x1b[0m\n"},
                    {
                        "output_type": "display_data",
                        "data": {"text/plain": "result", "text/html": "<script>evil()</script>"},
                    },
                ],
            },
            {"cell_type": "raw", "source": "plain raw cell", "metadata": {}},
        ]
    )


def saved_plot(mime="image/png", size=(20, 10)):
    pil = pytest.importorskip("PIL.Image")
    stream = io.BytesIO()
    pil.new("RGB", size, "red").save(stream, format="PNG" if mime == "image/png" else "JPEG")
    return base64.b64encode(stream.getvalue()).decode("ascii")


def plot_notebook(data):
    return notebook(
        [
            {
                "cell_type": "code",
                "source": "# saved plot, never executed",
                "outputs": [
                    {"output_type": "display_data", "data": data},
                ],
            }
        ]
    )


@pytest.mark.parametrize("mime", ["image/png", "image/jpeg"])
def test_saved_notebook_images_are_passive(mime):
    encoded = saved_plot(mime)
    preview = preview_file(
        NotebookFile(
            plot_notebook({mime: [encoded[:20] + "\n", encoded[20:]], "text/plain": "Saved figure"})
        )
    )
    cell = preview["notebook_cells"][0]
    assert cell["images"][0].size == (20, 10)
    assert cell["output"] == "Saved figure"
    assert not preview["notice"]


def test_notebook_images_share_count_pixel_and_byte_limits(monkeypatch):
    from blosc2.b2view import notebook_preview

    encoded = saved_plot()
    cells = plot_notebook({"image/png": encoded})["cells"] * (notebook_preview.NOTEBOOK_IMAGES + 1)
    preview = preview_file(NotebookFile(notebook(cells)))
    assert sum(len(cell["images"]) for cell in preview["notebook_cells"]) == notebook_preview.NOTEBOOK_IMAGES
    assert "budget" in preview["notice"]
    monkeypatch.setattr(notebook_preview, "IMAGE_PIXELS", 300)
    preview = preview_file(NotebookFile(notebook(cells[:2])))
    assert [len(cell["images"]) for cell in preview["notebook_cells"]] == [1, 0]
    assert "pixel budget" in preview["notice"]
    monkeypatch.setattr(notebook_preview, "IMAGE_BYTES", 1)
    monkeypatch.setattr(
        notebook_preview,
        "preview_image_bytes",
        lambda *args, **kwargs: pytest.fail("Over-budget image reached decoder"),
    )
    preview = preview_file(NotebookFile(plot_notebook({"image/png": encoded})))
    assert not preview["notebook_cells"][0]["images"]
    assert "byte budget" in preview["notice"]


@pytest.mark.parametrize("value", ["not base64!", "", 42, base64.b64encode(b"not an image").decode()])
def test_invalid_saved_image_keeps_notebook_text(value):
    preview = preview_file(NotebookFile(plot_notebook({"image/png": value, "text/plain": "still readable"})))
    assert preview["notebook_cells"][0]["output"] == "still readable"
    assert not preview["notebook_cells"][0]["images"]
    assert "omitted" in preview["notice"]


def test_notebook_image_decoder_failure_preserves_cells(monkeypatch):
    monkeypatch.setattr(
        "blosc2.b2view.notebook_preview.preview_image_bytes",
        lambda *args, **kwargs: {"message": "Image preview needs Pillow."},
    )
    preview = preview_file(NotebookFile(plot_notebook({"image/png": saved_plot()})))
    assert not preview["notebook_cells"][0]["images"]
    assert "Pillow" in preview["notice"]


@pytest.mark.tui
@pytest.mark.asyncio
@pytest.mark.parametrize("widget", ["real", "missing", "broken"])
async def test_notebook_saved_plots_inline_and_raw_toggle(tmp_path, monkeypatch, widget):
    from textual.widgets import Static

    from blosc2.b2view.app import B2ViewApp

    if widget == "real":
        pytest.importorskip("textual_image.widget")
    elif widget == "missing":
        monkeypatch.setattr("blosc2.b2view.app.TextualImage", None)
    else:

        def broken(*args):
            raise RuntimeError("secret must not appear")

        monkeypatch.setattr("blosc2.b2view.app.TextualImage", broken)
    path = tmp_path / "plots.ipynb"
    path.write_text(json.dumps(plot_notebook({"image/png": saved_plot()})))
    app = B2ViewApp(str(path))
    async with app.run_test(size=(120, 40)) as pilot:
        await wait_until(pilot, lambda: bool(app.query_one("#file-image").children))
        container = app.query_one("#file-image")
        if widget == "real":
            assert len(container.children) == 2
            await wait_until(pilot, lambda: container.children[1].region.height > 0)
            assert "Cell 1" in container.children[0].content.title.plain
            assert app.query_one("#preview", Static).content == ""
        elif widget == "missing":
            assert "textual-image" in str(container.children[0].render())
            from rich.console import Console

            console = Console()
            with console.capture() as capture:
                console.print(app.query_one("#preview", Static).content)
            assert "Cell 1" in capture.get()
        else:
            assert "could not be displayed" in str(container.children[1].render())
            assert "secret" not in str(container.children[1].render())
        await pilot.press("T")
        await wait_until(pilot, lambda: not container.children)
        assert "nbformat" in str(app.query_one("#preview", Static).render())
        await pilot.press("T")
        await wait_until(pilot, lambda: bool(container.children))
    app.wait_for_close()


def test_cells_and_outputs_are_passive_and_safe():
    file = NotebookFile(sample_notebook())
    preview = preview_file(file)
    assert preview["language"] == "python"
    cells = preview["notebook_cells"]
    assert [cell["kind"] for cell in cells] == ["markdown", "code", "raw"]
    assert "never execute" in cells[1]["source"]
    assert "hello red" in cells[1]["output"]
    assert "script" not in cells[1]["output"]
    assert "omitted" in preview["notice"]
    assert "T: raw/notebook" in preview["message"]
    assert "O:" not in preview["message"]
    assert file.reads == [(0, file.nbytes)]
    from rich.console import Console

    console = Console(record=True, width=100)
    header, body = make_preview_renderables(preview)
    with console.capture() as capture:
        console.print(header, body)
    rendered = capture.get()
    assert "Title" in rendered
    assert "Cell 2" in rendered
    assert "evil()" not in rendered
    raw = preview_file(file, raw=True)
    assert "file_text" in raw
    assert not raw["markdown"]


def test_input_display_limits_and_unsupported_formats():
    large = NotebookFile(notebook([]))
    large.nbytes = NOTEBOOK_BYTES + 1
    assert preview_file(large)["preview_status"] == "Preview unavailable"
    assert not large.reads
    cells = [{"cell_type": "raw", "source": "small"}] * (NOTEBOOK_CELLS + 1)
    preview = preview_file(NotebookFile(notebook(cells)))
    assert len(preview["notebook_cells"]) == NOTEBOOK_CELLS
    assert "truncated" in preview["notice"]
    for source in ("x" * (TEXT_BYTES + 1), "x\n" * (TEXT_LINES + 1)):
        preview = preview_file(NotebookFile(notebook([{"cell_type": "raw", "source": source}])))
        text = preview["notebook_cells"][0]["source"]
        assert len(text.encode()) <= TEXT_BYTES
        assert len(text.splitlines()) <= TEXT_LINES
        assert "truncated" in preview["notice"]
    assert preview_file(NotebookFile({"nbformat": 3}))["preview_status"] == "Preview unavailable"
    assert preview_file(NotebookFile(notebook([])))["notebook_cells"] == []


def test_plain_outputs_share_text_budget_and_strip_error_controls():
    cells = [
        {
            "cell_type": "code",
            "source": "print('x')",
            "outputs": [{"output_type": "stream", "text": "y" * TEXT_BYTES}],
        }
    ]
    preview = preview_file(NotebookFile(notebook(cells)))
    cell = preview["notebook_cells"][0]
    assert len((cell["source"] + cell["output"]).encode()) <= TEXT_BYTES
    assert "truncated" in preview["notice"]
    error = {
        "cell_type": "code",
        "source": "",
        "outputs": [{"output_type": "error", "traceback": ["\x1b[31mValueError\x1b[0m", "details\0"]}],
    }
    preview = preview_file(NotebookFile(notebook([error])))
    assert preview["notebook_cells"][0]["output"] == "ValueError\ndetails"


def test_rich_only_outputs_and_unknown_language_are_not_interpreted():
    data = notebook(
        [
            {
                "cell_type": "code",
                "source": "x",
                "outputs": [
                    {
                        "output_type": "display_data",
                        "data": {"image/png": "invalid base64", "text/html": "<script>evil()</script>"},
                    }
                ],
            }
        ]
    )
    data["metadata"]["language_info"]["name"] = "untrusted/custom/lexer"
    preview = preview_file(NotebookFile(data))
    assert preview["language"] == "text"
    assert preview["notebook_cells"][0]["output"] == ""
    assert "omitted" in preview["notice"]


@pytest.mark.parametrize(
    "invalid",
    [
        b"not JSON",
        b'{"nbformat":4,"cells":"bad"}',
        b'{"nbformat":4,"cells":[{"cell_type":"code","source":42}]}',
    ],
)
def test_malformed_notebook_fails_without_crashing(invalid):
    file = NotebookFile({})
    file.data, file.nbytes = invalid, len(invalid)
    assert preview_file(file)["preview_status"] == "Preview failed"


@pytest.mark.parametrize("backend", ["local", "compressed", "memory"])
def test_notebooks_on_direct_backends(tmp_path, backend):
    payload = json.dumps(sample_notebook()).encode()
    if backend == "memory":
        fs = pytest.importorskip("fsspec").filesystem("memory")
        source = "memory://b2view-notebooks/example.ipynb"
        fs.pipe(source, payload)
    elif backend == "compressed":
        source = str(tmp_path / "example.ipynb.b2")
        schunk = blosc2.SChunk(urlpath=source, mode="w", chunksize=4096, contiguous=True)
        schunk.append_data(payload)
        del schunk
    else:
        path = tmp_path / "example.ipynb"
        path.write_bytes(payload)
        source = str(path)
    with StoreBrowser(source) as browser:
        assert browser.kind("/") == "file"
        assert len(browser.preview("/")["notebook_cells"]) == 3


@pytest.mark.tui
@pytest.mark.asyncio
async def test_caterva2_notebook_tui_and_raw_toggle(file_source):  # noqa: F811
    from textual.widgets import Static

    from blosc2.b2view.app import B2ViewApp

    base, _, stats = file_source
    payload = json.dumps(sample_notebook()).encode()
    stream = blosc2.SChunk(chunksize=4096, cparams={"typesize": 1})
    stream.append_data(payload)
    stats["files"]["@public/example.ipynb"] = stream
    stats["groups"]["@public"].append("example.ipynb")
    app = B2ViewApp(base, start_path="/example.ipynb")
    async with app.run_test(size=(120, 40)) as pilot:
        await wait_until(
            pilot, lambda: "T: raw/notebook" in str(app.query_one("#data-header", Static).render())
        )
        assert app._selected_info.kind == "file"
        assert not app._file_raw
        await pilot.press("T")
        await wait_until(pilot, lambda: "nbformat" in str(app.query_one("#preview", Static).render()))
        assert app._file_raw
        await pilot.press("T")
        await wait_until(
            pilot, lambda: "passive preview" in str(app.query_one("#data-header", Static).render())
        )
        assert not app._file_raw
    app.wait_for_close()
