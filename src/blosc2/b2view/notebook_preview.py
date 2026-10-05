"""Bounded, passive notebook cells and plain-text outputs; no kernel or HTML."""

import json

from blosc2.b2view.file_preview import TEXT_BYTES, TEXT_LINES, file_actions, file_fallback, safe_text

NOTEBOOK_BYTES = 2 << 20
NOTEBOOK_CELLS = 100
LANGUAGES = {"python", "julia", "r", "javascript", "typescript", "bash", "sh", "sql", "c", "cpp", "text"}


def _text(value):
    if isinstance(value, list) and all(isinstance(part, str) for part in value):
        value = "".join(value)
    if not isinstance(value, str):
        raise ValueError("Notebook cell/output text must be a string or list of strings")
    return safe_text(value)


class _TextBudget:
    def __init__(self):
        self.bytes = TEXT_BYTES
        self.lines = TEXT_LINES
        self.truncated = False

    def take(self, text):
        lines = text.splitlines(keepends=True)
        self.truncated |= len(lines) > self.lines
        text = "".join(lines[: self.lines])
        encoded = text.encode("utf-8")
        self.truncated |= len(encoded) > self.bytes
        text = encoded[: self.bytes].decode("utf-8", errors="ignore")
        self.bytes -= len(text.encode("utf-8"))
        self.lines -= len(text.splitlines())
        return text


def _outputs(cell, budget):
    outputs = cell.get("outputs", [])
    if not isinstance(outputs, list):
        raise ValueError("Notebook outputs must be a list")
    text, skipped = [], 0
    for output in outputs:
        if not isinstance(output, dict):
            raise ValueError("Notebook output must be an object")
        kind = output.get("output_type")
        if kind == "stream":
            text.append(_text(output.get("text", "")))
        elif kind == "error":
            traceback = output.get("traceback", [])
            if not isinstance(traceback, list):
                raise ValueError("Notebook traceback must be a list")
            text.append("\n".join(_text(line) for line in traceback))
        elif kind in {"display_data", "execute_result"}:
            data = output.get("data", {})
            if not isinstance(data, dict):
                raise ValueError("Notebook output data must be an object")
            if "text/plain" in data:
                text.append(_text(data["text/plain"]))
            skipped += int(any(mime != "text/plain" for mime in data))
        else:
            skipped += 1
    return budget.take("\n".join(part for part in text if part)), skipped


def preview_notebook(file):
    """Parse a capped nbformat-4 JSON document without interpreting rich outputs."""
    if file.nbytes > NOTEBOOK_BYTES:
        return file_fallback(
            file.name,
            "Preview unavailable",
            "Notebook exceeds 2 MiB input limit; download to view it separately.",
        )
    notebook = json.loads(file.read_bytes(0, file.nbytes))
    if not isinstance(notebook, dict) or notebook.get("nbformat") != 4:
        return file_fallback(file.name, "Preview unavailable", "Only nbformat 4 notebooks are supported.")
    cells = notebook.get("cells")
    if not isinstance(cells, list):
        raise ValueError("Notebook cells must be a list")
    metadata = notebook.get("metadata", {})
    language = "text"
    if isinstance(metadata, dict) and isinstance(metadata.get("language_info"), dict):
        name = metadata["language_info"].get("name")
        if isinstance(name, str) and name.lower() in LANGUAGES:
            language = name.lower()
    budget = _TextBudget()
    rendered, skipped = [], 0
    for index, cell in enumerate(cells[:NOTEBOOK_CELLS]):
        if not isinstance(cell, dict):
            raise ValueError("Notebook cell must be an object")
        kind = cell.get("cell_type")
        if kind not in {"code", "markdown", "raw"}:
            skipped += 1
            continue
        source = budget.take(_text(cell.get("source", "")))
        output, omitted = _outputs(cell, budget) if kind == "code" else ("", 0)
        skipped += omitted
        rendered.append({"index": index + 1, "kind": kind, "source": source, "output": output})
        if not budget.bytes or not budget.lines:
            budget.truncated |= index + 1 < len(cells)
            break
    notice = []
    if budget.truncated or len(cells) > NOTEBOOK_CELLS:
        notice.append("Preview truncated (100 cells / 64 KiB text / 1,000 lines).")
    if skipped:
        notice.append(
            f"{skipped} unsupported cells/rich outputs omitted (HTML, scripts and media are not rendered)."
        )
    return {
        "notebook_cells": rendered,
        "language": language,
        "notice": " ".join(notice),
        "message": f"Notebook · {len(rendered)}/{len(cells)} cells · passive preview · {file_actions(file.name, markdown=True)}",
    }
