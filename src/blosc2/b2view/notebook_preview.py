"""Bounded, passive notebook cells, text and saved PNG/JPEGs; no kernel or HTML."""

import base64
import json

from blosc2.b2view.file_preview import (
    IMAGE_BYTES,
    IMAGE_PIXELS,
    TEXT_BYTES,
    TEXT_LINES,
    file_actions,
    file_fallback,
    preview_image_bytes,
    safe_text,
)

NOTEBOOK_BYTES = 2 << 20
NOTEBOOK_CELLS = 100
NOTEBOOK_IMAGES = 8
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


class _ImageBudget:
    def __init__(self):
        self.bytes = IMAGE_BYTES
        self.pixels = IMAGE_PIXELS
        self.count = 0
        self.notices = set()

    def take(self, data):
        mime = next((mime for mime in ("image/png", "image/jpeg") if mime in data), None)
        if mime is None:
            return None
        if self.count >= NOTEBOOK_IMAGES or self.pixels <= 0 or self.bytes <= 0:
            self.notices.add("Saved images omitted: notebook image budget reached.")
            return None
        try:
            value = data[mime]
            if isinstance(value, list) and all(isinstance(part, str) for part in value):
                value = "".join(value)
            if not isinstance(value, str):
                raise ValueError("Invalid image data")
            encoded = "".join(value.split())
            if len(encoded) > 4 * ((self.bytes + 2) // 3):
                self.notices.add("Saved image omitted: encoded byte budget exceeded.")
                return None
            payload = base64.b64decode(encoded, validate=True)
            if len(payload) > self.bytes:
                self.notices.add("Saved image omitted: encoded byte budget exceeded.")
                return None
            signature = b"\x89PNG\r\n\x1a\n" if mime == "image/png" else b"\xff\xd8\xff"
            if not payload.startswith(signature):
                raise ValueError("Image signature does not match MIME type")
            result = preview_image_bytes(
                payload, "output.png" if mime == "image/png" else "output.jpg", max_pixels=self.pixels
            )
            if "file_image" not in result:
                self.notices.add(f"Saved image omitted: {result['message']}")
                return None
            self.bytes -= len(payload)
            self.pixels -= result["image_pixels"]
            self.count += 1
            return result["file_image"]
        except Exception:
            self.notices.add("Saved image omitted: invalid or unsafe PNG/JPEG data.")
            return None


def _outputs(cell, budget, image_budget):
    outputs = cell.get("outputs", [])
    if not isinstance(outputs, list):
        raise ValueError("Notebook outputs must be a list")
    text, images, skipped = [], [], 0
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
            image = image_budget.take(data)
            if image is not None:
                images.append(image)
            skipped += int(any(mime not in {"text/plain", "image/png", "image/jpeg"} for mime in data))
        else:
            skipped += 1
    return budget.take("\n".join(part for part in text if part)), images, skipped


def preview_notebook(file):
    """Parse capped nbformat-4 JSON; only text and static PNG/JPEG outputs are decoded."""
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
    image_budget = _ImageBudget()
    rendered, skipped = [], 0
    for index, cell in enumerate(cells[:NOTEBOOK_CELLS]):
        if not isinstance(cell, dict):
            raise ValueError("Notebook cell must be an object")
        kind = cell.get("cell_type")
        if kind not in {"code", "markdown", "raw"}:
            skipped += 1
            continue
        source = budget.take(_text(cell.get("source", "")))
        output, images, omitted = _outputs(cell, budget, image_budget) if kind == "code" else ("", [], 0)
        skipped += omitted
        rendered.append(
            {"index": index + 1, "kind": kind, "source": source, "output": output, "images": images}
        )
        if not budget.bytes or not budget.lines:
            budget.truncated |= index + 1 < len(cells)
            break
    notice = []
    if budget.truncated or len(cells) > NOTEBOOK_CELLS:
        notice.append("Preview truncated (100 cells / 64 KiB text / 1,000 lines).")
    if skipped:
        notice.append(
            f"{skipped} unsupported cells/rich outputs omitted (HTML, scripts, SVG and widgets are not rendered)."
        )
    notice.extend(sorted(image_budget.notices))
    return {
        "notebook_cells": rendered,
        "language": language,
        "notice": " ".join(notice),
        "message": f"Notebook · {len(rendered)}/{len(cells)} cells · passive preview · {file_actions(file.name, markdown=True)}",
    }
