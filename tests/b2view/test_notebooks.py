"""Passive notebook previews share file transports without executing kernels."""

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
