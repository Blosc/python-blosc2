"""Bounded, passive previews of untrusted ordinary-file content."""

import codecs
import io
import re
import warnings
from pathlib import PurePosixPath

TEXT_BYTES = 64 << 10
TEXT_LINES = 1000
IMAGE_BYTES = 16 << 20
IMAGE_PIXELS = (64 << 20) // 4
TEXT_SUFFIXES = {".md", ".txt", ".rst", ".csv", ".json", ".yaml", ".yml", ".log", ".py"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
EXTERNAL_SUFFIXES = IMAGE_SUFFIXES | {".pdf", ".md", ".txt"}


def safe_text(text):
    """Remove ANSI/OSC sequences and unsafe terminal controls from decoded text."""
    text = re.sub(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)", "", text)
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)
    return "".join(c for c in text if c in "\n\t" or (ord(c) >= 32 and not 127 <= ord(c) < 160))


def preview_file(file, *, raw=False):
    """Return metadata/render content without executing links or decoding objects."""
    suffix = PurePosixPath(file.name).suffix.lower()
    try:
        if suffix == ".pdf":
            return {"message": "PDF preview unavailable; use D to download or O to open externally."}
        if suffix in IMAGE_SUFFIXES:
            return preview_image(file)
        if suffix not in TEXT_SUFFIXES:
            return {"message": "Binary/unknown file; use D to download original bytes."}
        data = file.read_bytes(0, min(file.nbytes, TEXT_BYTES))
        if b"\0" in data:
            return {"message": "Binary content despite text suffix; use D to download."}
        truncated = file.nbytes > len(data)
        decoder = codecs.getincrementaldecoder("utf-8-sig")("replace")
        text = safe_text(decoder.decode(data, final=not truncated))
        lines = text.splitlines(keepends=True)
        truncated |= len(lines) > TEXT_LINES
        text = "".join(lines[:TEXT_LINES])
        notice = "Preview truncated (64 KiB / 1,000 line limit)." if truncated else ""
        if "\ufffd" in text:
            notice += " Invalid UTF-8 replaced."
        return {
            "file_text": text,
            "markdown": suffix == ".md" and not raw,
            "notice": notice,
            "message": "D: download · O: open externally · T: raw/Markdown",
        }
    except Exception as error:
        return {"message": f"Preview unavailable: {error}. Use D to download (chunk limits still apply)."}


def preview_image(file):
    try:
        from PIL import Image, ImageOps
    except ImportError:
        return {"message": "Image preview needs Pillow; download (D) or open externally (O)."}
    if file.nbytes > IMAGE_BYTES:
        return {"message": "Automatic image preview exceeds 16 MiB; download (D) or open externally (O)."}
    data = file.read_bytes(0, file.nbytes)
    if not (data.startswith(b"\xff\xd8\xff") or data.startswith(b"\x89PNG\r\n\x1a\n")):
        return {"message": "Invalid JPEG/PNG signature; download available (D)."}
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(io.BytesIO(data)) as original:
            if original.format not in {"JPEG", "PNG"} or original.width * original.height > IMAGE_PIXELS:
                raise ValueError("Image exceeds decoded pixel budget (64 MiB RGBA)")
            size, format_name = original.size, original.format
            original.seek(0)
            image = ImageOps.exif_transpose(original)
            image.thumbnail((1600, 1200))
            image = image.convert("RGB")
    return {
        "file_image": image,
        "message": f"{format_name} · {size[0]} × {size[1]} · D: download · O: open externally",
    }


def open_external(path):
    """Launch a user-confirmed document with no shell or credential-bearing URL."""
    import os
    import shutil
    import subprocess
    import sys
    from pathlib import Path

    path = Path(path).absolute()
    if path.suffix.lower() not in EXTERNAL_SUFFIXES:
        raise ValueError("External opening is restricted to PDF, JPEG/PNG, Markdown and text")
    with path.open("rb") as file:
        signature = file.read(1024)
    suffix = path.suffix.lower()
    if (
        (suffix == ".pdf" and not signature.startswith(b"%PDF-"))
        or (suffix in {".jpg", ".jpeg"} and not signature.startswith(b"\xff\xd8\xff"))
        or (suffix == ".png" and not signature.startswith(b"\x89PNG\r\n\x1a\n"))
        or (suffix in {".txt", ".md"} and b"\0" in signature)
    ):
        raise ValueError("File content does not match the external-open document type")
    if sys.platform == "win32":
        os.startfile(str(path))
        return
    launcher = "open" if sys.platform == "darwin" else "xdg-open"
    if shutil.which(launcher) is None:
        raise RuntimeError(f"{launcher} is unavailable; open the downloaded file manually: {path}")
    subprocess.run([launcher, str(path)], check=True, timeout=10, capture_output=True)
