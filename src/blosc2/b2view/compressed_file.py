"""Lazy local/fsspec document carriers; never deserialize the original payload."""

import mimetypes
import os
import struct
from pathlib import PurePosixPath
from urllib.parse import unquote, urlsplit

import blosc2
from blosc2 import blosc2_ext
from blosc2.b2view.file_preview import IMAGE_SUFFIXES, TEXT_SUFFIXES
from blosc2.b2view.ordinary_file import MAX_READ_BYTES, OrdinaryFile, local_path, ordinary_filesystem
from blosc2.core import is_fsspec_url, parse_container_url
from blosc2.deserialization import set_deserialize
from blosc2.remote_file import MAX_COMPRESSED_CHUNK, MAX_DECODED_CHUNK


def compressed_document(source):
    """Recognize supported-document suffixes followed by .b2, not bare .b2."""
    source = os.fspath(source)
    _, dataset, source_format = parse_container_url(source)
    if dataset is not None or source_format is not None:
        return False
    name = PurePosixPath(unquote(urlsplit(source).path) if "://" in source else source).name
    return name.lower().endswith(".b2") and PurePosixPath(name[:-3]).suffix.lower() in (
        TEXT_SUFFIXES | IMAGE_SUFFIXES | {".pdf"}
    )


class CompressedFile(OrdinaryFile):
    """Original document bytes in a read-only mapped or range-backed SChunk."""

    def __init__(self, source, storage_options=None):
        self._closed = False
        self.fs = None
        self.source_path = os.fspath(source)
        self.storage_options = storage_options
        if is_fsspec_url(self.source_path):
            from blosc2.proxy_source import ByteRangeSChunkSource

            self.fs, self.path = ordinary_filesystem(self.source_path, storage_options)
            info = self.fs.info(self.path)
            if info.get("type") != "file" or type(info.get("size")) is not int:
                raise ValueError("Document carrier requires a regular file with a known size")
            self._frame_size = info["size"]
            self._schunk = ByteRangeSChunkSource(self._range, self._frame_size)
            self.carrier = PurePosixPath(unquote(urlsplit(self.source_path).path)).name
        else:
            self.path = local_path(self.source_path).absolute()
            self._schunk = self._open_local()
            self.carrier = self.path.name
        self._set_metadata()

    def _open_local(self):
        # Open only a native SChunk: do not invoke public-open dispatch for
        # persistent expressions, object serialization or remote references.
        self._schunk = blosc2_ext.open(str(self.path), "r", 0, mmap_mode="r")
        if not isinstance(self._schunk, blosc2.SChunk):
            self._schunk = None
            raise ValueError("Document carrier must be an SChunk byte stream, not an array or table")
        set_deserialize(self._schunk, "safe")
        return self._schunk

    def _set_metadata(self):
        structural = {
            "b2tree",
            "b2o",
            "listarray",
            "vlarray",
            "batcharray",
            "LazyArray",
            "proxy-source",
            "b2remote_array",
            "b2remote_ctable",
            "b2remote_store",
        }
        if structural.intersection(self._schunk.meta):
            self._schunk = None
            raise ValueError("Document carrier must be a plain SChunk byte stream, not a serialized object")
        self.nbytes = self._schunk.nbytes
        self.cbytes = self._schunk.cbytes
        self.chunksize = self._schunk.chunksize
        if not self.nbytes and self.chunksize < 0:
            self.chunksize = 0
        self.nchunks = self._schunk.nchunks
        expected = (
            (self.nbytes + self.chunksize - 1) // self.chunksize if self.nbytes and self.chunksize > 0 else 0
        )
        if (self.nbytes and self.chunksize <= 0) or self.nchunks != expected:
            self._schunk = None
            raise ValueError("Document carrier requires a fixed-chunk byte stream")
        self.name = self.carrier[:-3]
        self.media_type = mimetypes.guess_type(self.name)[0]

    def alias(self):
        self._check_open()
        # Transfers have independent mappings and native lifetimes.
        return CompressedFile(self.source_path, self.storage_options)

    def _range(self, offset, size):
        self._check_open()
        if not 0 <= offset <= self._frame_size - size or not 0 <= size <= MAX_COMPRESSED_CHUNK:
            raise ValueError("Document byte range exceeds bounds")
        if urlsplit(self.source_path).scheme in {"http", "https"}:
            from fsspec.asyn import sync

            return sync(self.fs.loop, _http_range, self.fs, self.path, offset, size, self._frame_size)
        with self.fs.open(self.path, "rb", cache_type="none") as stream:
            stream.seek(offset)
            data = stream.read(size)
        if len(data) != size:
            raise ValueError("Document changed or returned a short range")
        return data

    def _chunk(self, index):
        self._check_open()
        # Lazy chunk headers expose lengths without copying/decompressing the
        # stored payload. Check budgets before fetching the actual chunk.
        header = self._schunk.get_lazychunk(index)
        if len(header) < 16:
            raise ValueError("Invalid document chunk header")
        nbytes, _, cbytes = struct.unpack_from("<III", header, 4)
        if nbytes > MAX_DECODED_CHUNK:
            raise ValueError(
                f"Document chunk exceeds {MAX_DECODED_CHUNK >> 20} MiB decoded limit; rechunk the carrier"
            )
        if cbytes > MAX_COMPRESSED_CHUNK:
            raise ValueError(
                f"Document chunk exceeds {MAX_COMPRESSED_CHUNK >> 20} MiB compressed limit; rechunk the carrier"
            )
        expected = min(self.chunksize, self.nbytes - index * self.chunksize)
        if nbytes != expected:
            raise ValueError("Document chunk does not match fixed-chunk metadata")
        chunk = self._schunk.get_chunk(index)
        if len(chunk) != cbytes:
            raise ValueError("Document chunk has an invalid compressed length")
        data = blosc2.decompress(chunk)
        if len(data) != nbytes:
            raise ValueError("Decoded document chunk does not match metadata")
        return data

    def _stream(self, *, streaming=False):
        self._check_open()
        return _ChunkReader(self)

    def close(self):
        self._closed = True
        self._schunk = None


async def _http_range(fs, path, offset, size, frame_size):
    """Validate HTTP range headers before reading any response body."""
    kwargs = dict(fs.kwargs)
    headers = dict(kwargs.pop("headers", {}))
    headers.update(Range=f"bytes={offset}-{offset + size - 1}", **{"Accept-Encoding": "identity"})
    session = await fs.set_session()
    async with session.get(fs.encode_url(path), headers=headers, **kwargs) as response:
        response.raise_for_status()
        if (
            response.status != 206
            or response.headers.get("Content-Range") != f"bytes {offset}-{offset + size - 1}/{frame_size}"
            or response.headers.get("Content-Encoding", "identity") != "identity"
        ):
            raise ValueError("Remote document requires valid HTTP byte-range support")
        if int(response.headers.get("Content-Length", size)) != size:
            raise ValueError("HTTP document range has an invalid length")
        data = bytearray()
        async for part in response.content.iter_chunked(64 << 10):
            if len(data) + len(part) > size:
                raise ValueError("HTTP document range exceeds requested bytes")
            data.extend(part)
        if len(data) != size:
            raise ValueError("HTTP document range returned a short read")
        return bytes(data)


class _ChunkReader:
    """Keep at most one decoded chunk between bounded sequential reads."""

    def __init__(self, file):
        self.file = file
        self.position = 0
        self.index = None
        self.data = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.data = None

    def seek(self, offset):
        if not 0 <= offset <= self.file.nbytes:
            raise ValueError("Invalid document byte offset")
        self.position = offset

    def read(self, size):
        self.file._check_open()
        if not 0 <= size <= MAX_READ_BYTES:
            raise ValueError("Document stream reads must be bounded to 16 MiB")
        stop = min(self.position + size, self.file.nbytes)
        output = bytearray()
        while self.position < stop:
            index = self.position // self.file.chunksize
            if index != self.index:
                self.data = None
                self.data = self.file._chunk(index)
                self.index = index
            offset = self.position - index * self.file.chunksize
            length = min(stop - self.position, len(self.data) - offset)
            output.extend(self.data[offset : offset + length])
            self.position += length
        return bytes(output)
