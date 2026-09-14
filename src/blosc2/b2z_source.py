#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""Native range reads of external NDArray members in immutable B2Z archives."""

import io
import operator
import re
import zipfile

from blosc2.core import _import_fsspec
from blosc2.proxy_source import REMOTE_MAX_CONCURRENCY, ByteRangeNDSource, Traffic

# One request can carry a small member whole -- local header, frame header and the
# trailing vlmeta -- sparing a round trip each against the object store.
_WHOLE_MEMBER_PREFETCH_MAX = 64 * 1024
_LOCAL_HEADER_HEADROOM = 4096


async def _http_tail(fs, path):
    """Obtain HTTP object identity and the ZIP tail in one bounded request."""
    kwargs = fs.kwargs.copy()
    headers = kwargs.pop("headers", {}).copy()
    headers.update({"Range": "bytes=-8192", "Accept-Encoding": "identity"})
    session = await fs.set_session()
    async with session.get(fs.encode_url(path), headers=headers, **kwargs) as response:
        # Do not consume a potentially huge body if the server ignores ranges.
        if response.status in {200, 400, 405, 416, 501}:
            return None
        fs._raise_not_found_for_status(response, path)
        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
        if response.status != 206 or match is None:
            return None
        start, end, size = map(int, match.groups())
        if start != max(0, size - 8192) or end != size - 1 or size <= 0:
            return None
        if response.headers.get("Content-Encoding", "identity") not in {"", "identity"}:
            return None
        data = await response.content.readexactly(end - start + 1)
        if await response.content.read(1):
            raise ValueError("B2Z transport did not honor the requested byte range")
        # Match fsspec's HTTP info fields so persisted identities remain compatible.
        info = {"name": path, "size": size, "type": "file", "url": str(response.url)}
        if "Content-Type" in response.headers:
            info["mimetype"] = response.headers["Content-Type"].partition(";")[0]
        if response.headers.get("Accept-Ranges") == "none":
            info["partial"] = False
        for key in ("ETag", "Content-MD5", "Digest", "Last-Modified"):
            if response.headers.get(key):
                info[key] = response.headers[key]
        return info, data


class _ArchiveFile(io.RawIOBase):
    """Seekable view using the source's bounded opening buffers."""

    def __init__(self, source, size):
        self.source, self.size, self.pos = source, size, 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        bases = {0: 0, 1: self.pos, 2: self.size}
        if whence not in bases or bases[whence] + offset < 0:
            raise OSError("invalid archive seek")
        self.pos = bases[whence] + offset
        return self.pos

    def read(self, size=-1):
        size = self.size - self.pos if size < 0 else min(size, self.size - self.pos)
        data = self.source._read_archive(self.pos, max(0, size))
        self.pos += len(data)
        return data


class B2ZArchive:
    """Session directory and bounded range access shared by discovery and leaves."""

    def __init__(self, urlpath, *, storage_options=None, _filesystem=None, _traffic=None, _metadata=None):
        self.storage_options = storage_options or {}
        self.urlpath = urlpath
        fsspec = _import_fsspec(urlpath)
        if _filesystem is None:
            self._fs, self._path = fsspec.url_to_fs(urlpath, **self.storage_options)
        else:
            self._fs, self._path = _filesystem, _filesystem._strip_protocol(urlpath)
        self.traffic = _traffic if _traffic is not None else Traffic()
        bootstrap = None
        if self._fs.protocol in ("http", "https", ("http", "https")) and not _metadata:
            from fsspec.asyn import sync
            from fsspec.implementations.http import HTTPFileSystem

            if isinstance(self._fs, HTTPFileSystem):
                bootstrap = sync(self._fs.loop, _http_tail, self._fs, self._path)
        object_info = self._fs.info(self._path) if bootstrap is None else bootstrap[0]
        self.object_info = object_info
        size = object_info["size"]
        self.size = size
        self.metadata = _metadata if _metadata is not None else {}
        self.persist_metadata = _metadata is not None
        identity = repr(sorted((key, str(value)) for key, value in object_info.items()))
        if self.metadata and self.metadata.get("identity") != identity:
            raise ValueError("B2Z source changed; refresh the store cache")
        self.metadata.setdefault("identity", identity)
        self.metadata.setdefault("ranges", [])
        for offset, data in self.metadata["ranges"]:
            if (
                not isinstance(offset, int)
                or not isinstance(data, bytes)
                or not 0 <= offset <= size - len(data)
            ):
                raise ValueError("Invalid cached B2Z metadata range")
        self.capture_metadata = True
        self._opening_ranges = []
        # ponytail: small directories fit in 8 KiB; larger ones use exact reads.
        tail_start = max(0, size - 8192)
        if bootstrap is None:
            tail = self._read_archive(tail_start, size - tail_start)
        else:
            tail = bootstrap[1]
            self.traffic.charge(len(tail))
            if self.persist_metadata:
                self.metadata["ranges"].append((tail_start, tail))
        self._opening_ranges.append((tail_start, tail))
        self.file = _ArchiveFile(self, size)
        self.archive = zipfile.ZipFile(self.file)
        self.members = self.archive.infolist()
        self.capture_metadata = False

    def member_window(self, info, *, prefetch=False):
        size = self.size
        file, archive = self.file, self.archive
        if info.flag_bits & 1 or info.compress_type != zipfile.ZIP_STORED:
            raise NotImplementedError("B2Z array members must be unencrypted ZIP_STORED entries")
        if info.compress_size != info.file_size or not 0 <= info.header_offset <= size - 30:
            raise ValueError("invalid B2Z member size or offset")
        # Cover the local header and the native reader's 8 KiB frame prefix
        # together. Unusually long ZIP headers fall back to exact reads.
        if prefetch:
            if info.file_size <= _WHOLE_MEMBER_PREFETCH_MAX:
                # Take the whole member so the frame's trailing vlmeta is here too.
                want = info.file_size + _LOCAL_HEADER_HEADROOM
            else:
                # Cover the local header and the native reader's 8 KiB frame prefix.
                want = 16384
            prefix = self._read_archive(info.header_offset, min(want, size - info.header_offset))
            self._opening_ranges.append((info.header_offset, prefix))
        # zipfile validates the local signature, filename, and member overlap.
        # Opening does not read/decode member payloads.
        with archive.open(info):
            pass
        file.seek(info.header_offset)
        header = file.read(30)
        if len(header) != 30 or header[:4] != b"PK\x03\x04":
            raise ValueError("invalid B2Z local header")
        if (
            int.from_bytes(header[6:8], "little") != info.flag_bits
            or int.from_bytes(header[8:10], "little") != info.compress_type
        ):
            raise ValueError("inconsistent B2Z local header")
        member_offset = (
            info.header_offset
            + 30
            + int.from_bytes(header[26:28], "little")
            + int.from_bytes(header[28:30], "little")
        )
        member_length = info.file_size
        if member_offset + member_length > size:
            raise ValueError("B2Z member exceeds archive bounds")
        return member_offset, member_length

    def _read_archive(self, offset, size):
        if not size:
            return b""
        for start, data in self.metadata["ranges"] if self.capture_metadata else ():
            if start <= offset and offset + size <= start + len(data):
                return data[offset - start : offset - start + size]
        for start, data in self._opening_ranges:
            if start <= offset and offset + size <= start + len(data):
                return data[offset - start : offset - start + size]
        data = self._fs.cat_file(self._path, start=offset, end=offset + size)
        if len(data) > size:
            raise ValueError("B2Z transport did not honor the requested byte range")
        self.traffic.charge(len(data))
        if self.capture_metadata and self.persist_metadata:
            self.metadata["ranges"].append((offset, data))
        return data

    def close(self):
        self.archive.close()
        self.file.close()
        self._opening_ranges.clear()


class _SeededArchive:
    """Replay a cached B2Z bootstrap; initialize range access only on a read miss.

    A warm DISK carrier already knows where the member is and holds its opening
    frame prefix, so those bytes are served from memory. The filesystem is
    initialized lazily for reads outside the prefix, without ZIP discovery.
    """

    def __init__(self, urlpath, *, storage_options, filesystem, traffic, prefix_start, prefix):
        self.urlpath = urlpath
        self.storage_options = storage_options or {}
        self._filesystem = filesystem
        self.traffic = traffic
        self.prefix_start = prefix_start
        self.prefix = prefix
        self._opening_ranges = []
        self.capture_metadata = False
        self._fs = self._path = None

    def _read_archive(self, offset, size):
        if not size:
            return b""
        if self.prefix_start <= offset and offset + size <= self.prefix_start + len(self.prefix):
            start = offset - self.prefix_start
            return self.prefix[start : start + size]
        if self._fs is None:
            if self._filesystem is None:
                fs, path = _import_fsspec(self.urlpath).url_to_fs(self.urlpath, **self.storage_options)
            else:
                fs = self._filesystem
                path = fs._strip_protocol(self.urlpath)
            self._path, self._fs = path, fs
        data = self._fs.cat_file(self._path, start=offset, end=offset + size)
        if len(data) > size:
            raise ValueError("B2Z transport did not honor the requested byte range")
        self.traffic.charge(len(data))
        return data


class B2ZNDSource(ByteRangeNDSource):
    """Read a stored external NDArray from an immutable B2Z archive via fsspec.

    ``dataset`` is a logical tree key, e.g. ``d0/a3``, without the member's
    ``.b2nd`` suffix. Embedded leaves and ZIP-compressed members are unsupported.
    Opening uses bounded metadata prefetch; native chunks and blocks are fetched
    on demand. Replacing the archive requires replacing its cache.
    """

    def __init__(
        self,
        urlpath,
        dataset,
        max_concurrency=REMOTE_MAX_CONCURRENCY,
        *,
        storage_options=None,
        _filesystem=None,
        _traffic=None,
        _archive=None,
        _seed=None,
    ):
        if not isinstance(dataset, str) or not dataset.strip("/"):
            raise ValueError("B2Z sources require a dataset path (e.g. dataset='d0/a3')")
        dataset = dataset.strip("/")
        if any(part in {"", ".", ".."} for part in dataset.split("/")) or any(
            char in dataset for char in "\\\0\n\r\t"
        ):
            raise ValueError("invalid B2Z dataset path")
        self.dataset = dataset
        if _seed is not None:
            if _archive is not None:
                raise ValueError("a cached B2Z seed cannot be combined with an archive")
            self._init_seeded(urlpath, max_concurrency, storage_options, _filesystem, _traffic, _seed)
            return
        if _archive is not None and _archive.urlpath != urlpath:
            raise ValueError("B2Z source URL does not match its archive")
        archive = _archive or B2ZArchive(
            urlpath, storage_options=storage_options, _filesystem=_filesystem, _traffic=_traffic
        )
        try:
            archive.capture_metadata = True
            self._archive = archive
            self.storage_options = archive.storage_options
            self._fs, self._path = archive._fs, archive._path
            self.traffic = archive.traffic
            self._opening_ranges = archive._opening_ranges
            object_info = archive.object_info
            matches = [info for info in archive.members if info.filename == dataset + ".b2nd"]
            if not matches:
                raise ValueError(
                    f"No supported external NDArray at {dataset!r}; specify an external array leaf"
                )
            if len(matches) != 1:
                raise ValueError("duplicate B2Z array member")
            self.member_offset, self.member_length = archive.member_window(matches[0], prefetch=True)
            prefix_start, prefix = archive._opening_ranges[-1]
            from fsspec.utils import tokenize

            self.stamp = tokenize(urlpath, object_info, dataset, self.member_offset, self.member_length)
            super().__init__(urlpath, max_concurrency, traffic=self.traffic)
            if b"b2o" in self._header[13][1]:
                raise NotImplementedError("B2Z object carriers are not supported; select a plain NDArray")
            if not self._header_len <= self._header[2] <= self.member_length:
                raise ValueError("Blosc2 frame exceeds B2Z member bounds")
            # Keep the whole-member prefetch for the ordinary chunk cache, while
            # the persistent bootstrap itself only needs the frame header.
            self._prepare_prefetch(prefix_start, prefix)
            self._opening_ranges.clear()
        finally:
            archive.capture_metadata = False
            archive._opening_ranges.clear()
            if _archive is None:
                archive.close()

    def _init_seeded(self, urlpath, max_concurrency, storage_options, filesystem, traffic, seed):
        """Rebuild this source from a carrier's cached bootstrap, without network."""
        self.member_offset = int(seed["member_offset"])
        self.member_length = int(seed["member_length"])
        self.stamp = str(seed["stamp"])
        self.storage_options = storage_options or {}
        self.traffic = traffic if traffic is not None else Traffic()
        self._fs = self._path = None
        self._archive = _SeededArchive(
            urlpath,
            storage_options=self.storage_options,
            filesystem=filesystem,
            traffic=self.traffic,
            prefix_start=int(seed["prefix_start"]),
            prefix=bytes(seed["prefix"]),
        )
        self._opening_ranges = self._archive._opening_ranges
        # The frame header comes out of the cached prefix, so this reads nothing.
        start = self.member_offset - int(seed["prefix_start"])
        head = bytes(seed["prefix"])[start:] if start >= 0 else None
        super().__init__(urlpath, max_concurrency, traffic=self.traffic, _head=head)
        self._opening_ranges.clear()
        if b"b2o" in self._header[13][1]:
            raise NotImplementedError("B2Z object carriers are not supported; select a plain NDArray")
        if not self._header_len <= self._header[2] <= self.member_length:
            raise ValueError("Blosc2 frame exceeds B2Z member bounds")
        self._prepare_prefetch(int(seed["prefix_start"]), bytes(seed["prefix"]))
        self._archive.prefix_start = self.member_offset
        self._archive.prefix = self._raw_header

    def _prepare_prefetch(self, prefix_start, prefix):
        """Transfer whole-frame payload to the chunk cache, outside bootstrap metadata."""
        start = self.member_offset - prefix_start
        self._prefetched_frame = None
        if start >= 0 and start + self.member_length <= len(prefix):
            self._prefetched_frame = prefix[start : start + self.member_length]
            # Preserve the offsets and vlmeta already fetched with the payload.
            self._head = self._prefetched_frame
            self._frame_index()
            if self.has_vlmetalayers:
                self._vlmeta = self._read_frame_vlmeta()
        self._head = self._raw_header
        self._seed = {
            "version": 1,
            "member_offset": self.member_offset,
            "member_length": self.member_length,
            "prefix_start": self.member_offset,
            "prefix": self._raw_header,
            "stamp": self.stamp,
        }

    def _take_prefetched_array(self):
        """Consume the opening payload once; Proxy owns its accounting and eviction."""
        import blosc2

        frame, self._prefetched_frame = self._prefetched_frame, None
        return None if frame is None else blosc2.ndarray_from_cframe(frame, copy=True)

    def _read_archive(self, offset, size):
        return self._archive._read_archive(offset, size)

    def read_range(self, offset, size):
        offset, size = operator.index(offset), operator.index(size)
        if offset < 0 or size < 0:
            raise ValueError("invalid B2Z frame range")
        size = max(0, min(size, self.member_length - offset))
        return self._read_archive(self.member_offset + offset, size)


def member_vlmeta(archive, info):
    """Read a member's frame trailer without loading its embedded payload."""
    from blosc2.proxy_source import _parse_trailer_vlmeta

    offset, length = archive.member_window(info)
    # The trailer length occupies a fixed position in the 23-byte frame footer.
    if length < 23:
        raise ValueError("truncated B2Z frame")
    footer = archive._read_archive(offset + length - 23, 23)
    if len(footer) != 23 or footer[0] != 0xCE:
        raise ValueError("invalid B2Z frame footer")
    size = int.from_bytes(footer[1:5], "big")
    if not 23 <= size <= length:
        raise ValueError("invalid B2Z frame trailer length")
    return _parse_trailer_vlmeta(archive._read_archive(offset + length - size, size))


class B2ZEmbeddedMetadata:
    """Read indexed EmbedStore metadata entries through bounded native chunks.

    This is deliberately not an embedded leaf reader. Only TreeStore attribute
    frames are decoded, using the same attribute decoding as EmbedStore.
    """

    def __init__(self, archive, info):
        from blosc2.proxy_source import _read_frame_header, _read_frame_offsets

        self.archive = archive
        self.offset, self.length = archive.member_window(info)
        raw, self.header, head = _read_frame_header(self.read_range)
        if not len(raw) <= self.header[2] <= self.length:
            raise ValueError("Embedded frame exceeds B2Z member bounds")
        self.offsets = _read_frame_offsets(self.read_range, self.header, head, len(raw))

    def read_range(self, offset, size):
        if offset < 0 or size < 0 or offset > self.length:
            raise ValueError("Invalid embedded metadata range")
        return self.archive._read_archive(self.offset + offset, min(size, self.length - offset))

    def attrs(self, entry):
        import blosc2

        start, length = entry["offset"], entry["length"]
        stop = start + length
        chunksize = self.header[8]
        if start < 0 or length <= 0 or stop > self.header[4] or chunksize <= 0:
            raise ValueError("Invalid EmbedStore metadata entry bounds")
        # ponytail: metadata in >1 MiB native chunks stays unavailable; add
        # SChunk block reads before supporting such embedded storage layouts.
        if chunksize > 1024 * 1024:
            raise NotImplementedError("embedded metadata requires reading a native chunk larger than 1 MiB")
        pieces = []
        for index in range(start // chunksize, (stop - 1) // chunksize + 1):
            offset = int(self.offsets[index])
            if offset < self.header[1] or offset + 16 > self.header[1] + self.header[5]:
                raise ValueError("Invalid embedded metadata chunk offset")
            header = self.read_range(offset, 16)
            size = int.from_bytes(header[12:16], "little")
            if (
                not 16 <= size <= chunksize + blosc2.MAX_OVERHEAD
                or offset + size > self.header[1] + self.header[5]
            ):
                raise ValueError("Invalid embedded metadata chunk size")
            chunk = blosc2.decompress2(self.read_range(offset, size))
            lo, hi = max(0, start - index * chunksize), min(len(chunk), stop - index * chunksize)
            pieces.append(chunk[lo:hi])
        frame = b"".join(pieces)
        if len(frame) != length:
            raise ValueError("Truncated embedded attribute frame")
        schunk = blosc2.schunk_from_cframe(frame, copy=True)
        return schunk.vlmeta[:]
