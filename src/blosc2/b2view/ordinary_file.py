"""Viewer-only read-only adapters for local and fsspec ordinary files."""

import mimetypes
import os
import stat
import tempfile
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit
from urllib.request import url2pathname

import blosc2
from blosc2.core import find_url_separator, fsspec_filesystem, is_fsspec_url, parse_container_url

MAX_READ_BYTES = 16 << 20
TRANSFER_BYTES = 1 << 20
NATIVE_SUFFIXES = {".b2", ".b2nd", ".b2frame", ".b2b", ".b2d", ".b2z", ".b2t", ".b2e"}


def _source_member(source):
    separator = find_url_separator(source)
    return source if separator == -1 else source[:separator]


def ordinary_source(urlpath, remote_service="auto"):
    """Identify direct file inputs without stealing container/service URLs."""
    from blosc2.caterva2_url import caterva2_urlpath

    urlpath = os.fspath(urlpath)
    if remote_service == "caterva2" or (remote_service == "auto" and caterva2_urlpath(urlpath) is not None):
        return False
    from blosc2.b2view.compressed_file import compressed_document

    if compressed_document(urlpath):
        return True
    base, dataset, source_format = parse_container_url(urlpath)
    if dataset is not None or source_format is not None:
        return False
    member = _source_member(str(base))
    parsed = urlsplit(member)
    # HTTP authorities are hosts, not filenames (e.g. service.example.com).
    # Archive/member protocols may instead carry the filename in the authority.
    name = parsed.path if parsed.scheme in {"http", "https"} else parsed.path or parsed.netloc
    suffix = PurePosixPath(name).suffix.lower()
    if suffix in NATIVE_SUFFIXES:
        return False
    if is_fsspec_url(urlpath):
        # Bare HTTP service bases must retain automatic Caterva2 discovery.
        return remote_service == "fsspec" or parsed.scheme not in {"http", "https"} or bool(suffix)
    if urlpath.startswith("file://"):
        # Validate the authority in the background opener, not the UI constructor.
        return True
    return True


def local_path(source):
    """Resolve file:// paths without interpreting URL escapes in ordinary paths."""
    if source.startswith("file://"):
        parsed = urlsplit(source)
        if parsed.netloc not in {"", "localhost"}:
            raise ValueError("file:// URLs must refer to the local host")
        # url2pathname handles /C:/ drive prefixes on Windows as well as escapes.
        return Path(url2pathname(parsed.path))
    return Path(source)


def ordinary_filesystem(source, storage_options=None):
    """Resolve direct files/directories with bounded default HTTP timeouts."""
    options = dict(storage_options or {})
    if urlsplit(source).scheme in {"http", "https"} and find_url_separator(source) == -1:
        import aiohttp

        client_options = dict(options.get("client_kwargs", {}))
        client_options.setdefault("timeout", aiohttp.ClientTimeout(total=10))
        options["client_kwargs"] = client_options
    return fsspec_filesystem(source, options)


def _publish_download(temporary, destination, overwrite):
    if overwrite:
        os.replace(temporary, destination)
    elif blosc2.IS_WASM:
        # Emscripten has no hard links. Its single-threaded virtual FS cannot
        # interleave a writer between this check and rename (no callbacks/awaits).
        if os.path.lexists(destination):
            raise FileExistsError(destination)
        os.rename(temporary, destination)
    else:
        os.link(temporary, destination)


class OrdinaryFile:
    """A direct regular file with bounded reads and independent transfer aliases.

    Streams are opened only for individual reads/transfers. This adapter does
    not deserialize containers or participate in the Caterva2 payload cache.
    """

    def __init__(self, source, storage_options=None):
        self._closed = False
        self.source_path = os.fspath(source)
        self.fs = None
        if is_fsspec_url(self.source_path):
            self.fs, self.path = ordinary_filesystem(self.source_path, storage_options)
            info = self.fs.info(self.path)
            if info.get("type") != "file":
                raise ValueError("Select a regular file, not a directory")
            size = info.get("size")
            member = urlsplit(_source_member(self.source_path))
            self.name = PurePosixPath(member.path or self.path).name
        else:
            self.path = local_path(self.source_path).absolute()
            info = self.path.stat()
            if not stat.S_ISREG(info.st_mode):
                raise ValueError("Select a regular file, not a directory or device")
            size = info.st_size
            self.name = self.path.name
        if type(size) is not int or size < 0:
            raise ValueError("Ordinary-file access requires a known nonnegative size")
        self.nbytes = size
        self.media_type = mimetypes.guess_type(self.name)[0]

    def _check_open(self):
        if self._closed:
            raise RuntimeError("OrdinaryFile is closed")

    def alias(self):
        """Return an independent handle without extra metadata/payload requests."""
        import copy

        self._check_open()
        return copy.copy(self)

    def _stream(self, *, streaming=False):
        self._check_open()
        if self.fs is None:
            return self.path.open("rb")
        # Disable speculative block caching; each read is explicitly bounded.
        if streaming:
            return self.fs.open(self.path, "rb", block_size=0)
        return self.fs.open(self.path, "rb", block_size=64 << 10, cache_type="none")

    def read_bytes(self, start=0, stop=None):
        self._check_open()
        stop = self.nbytes if stop is None else stop
        if type(start) is not int or type(stop) is not int or not 0 <= start <= stop <= self.nbytes:
            raise ValueError("Byte interval must satisfy 0 <= start <= stop <= nbytes")
        if stop - start > MAX_READ_BYTES:
            raise ValueError("read_bytes exceeds 16 MiB; use streaming download")
        if start == stop:
            return b""
        # Previews start at zero. Streaming avoids HTTP range responses that
        # lie about their length causing an unbounded response-body allocation.
        with self._stream(streaming=start == 0) as stream:
            if start:
                stream.seek(start)
            data = bytearray()
            while len(data) < stop - start:
                remaining = stop - start - len(data)
                requested = min(TRANSFER_BYTES, remaining)
                part = stream.read(requested)
                if not part:
                    raise OSError("File changed or returned a short read; reopen the source")
                if len(part) > requested:
                    raise OSError("Filesystem returned more bytes than requested")
                data.extend(part)
        return bytes(data)

    def download(self, destination, *, overwrite=False, progress=None, cancel=None):
        """Stream an explicit copy with atomic publication, no overwrite by default."""
        self._check_open()
        if not isinstance(overwrite, bool):
            raise TypeError("overwrite must be a bool")
        if (progress is not None and not callable(progress)) or (
            cancel is not None and not callable(cancel)
        ):
            raise TypeError("progress and cancel must be callable or None")
        destination = Path(destination).absolute()
        if not overwrite and destination.exists():
            raise FileExistsError(destination)
        if self.fs is None and destination.exists() and os.path.samefile(self.path, destination):
            raise ValueError("Destination must differ from the source file")
        fd, temporary = tempfile.mkstemp(prefix=".b2view-download-", dir=destination.parent)
        try:
            with os.fdopen(fd, "wb") as output, self._stream(streaming=True) as source:
                done = 0
                while done < self.nbytes:
                    if cancel is not None and cancel():
                        raise InterruptedError("File operation cancelled")
                    requested = min(TRANSFER_BYTES, self.nbytes - done)
                    data = source.read(requested)
                    if not data:
                        raise OSError("File changed or returned a short read; reopen the source")
                    if len(data) > requested:
                        raise OSError("Filesystem returned more bytes than requested")
                    output.write(data)
                    done += len(data)
                    if progress is not None:
                        progress(done, self.nbytes)
                if source.read(1):
                    raise OSError("File size changed; reopen the source")
                output.flush()
                os.fsync(output.fileno())
            self._check_open()
            if cancel is not None and cancel():
                raise InterruptedError("File operation cancelled")
            _publish_download(temporary, destination, overwrite)
            return str(destination)
        finally:
            Path(temporary).unlink(missing_ok=True)

    def close(self):
        self._closed = True
