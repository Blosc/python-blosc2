#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""Native range reads of external NDArray members in immutable B2Z archives."""

import io
import operator
import zipfile

from blosc2.core import _import_fsspec
from blosc2.proxy_source import REMOTE_MAX_CONCURRENCY, ByteRangeNDSource, Traffic


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
    ):
        if not isinstance(dataset, str) or not dataset.strip("/"):
            raise ValueError("B2Z sources require a dataset path (e.g. dataset='d0/a3')")
        dataset = dataset.strip("/")
        if any(part in {"", ".", ".."} for part in dataset.split("/")) or any(
            char in dataset for char in "\\\0\n\r\t"
        ):
            raise ValueError("invalid B2Z dataset path")
        self.dataset = dataset
        self.storage_options = storage_options or {}
        fsspec = _import_fsspec(urlpath)
        if _filesystem is None:
            self._fs, self._path = fsspec.url_to_fs(urlpath, **self.storage_options)
        else:
            self._fs, self._path = _filesystem, _filesystem._strip_protocol(urlpath)
        self.traffic = _traffic if _traffic is not None else Traffic()
        object_info = self._fs.info(self._path)
        size = object_info["size"]
        self._opening_ranges = []
        # ponytail: small directories fit in 8 KiB; larger ones use exact reads.
        tail_start = max(0, size - 8192)
        self._opening_ranges.append((tail_start, self._read_archive(tail_start, size - tail_start)))
        with _ArchiveFile(self, size) as file, zipfile.ZipFile(file) as archive:
            matches = [info for info in archive.infolist() if info.filename == dataset + ".b2nd"]
            if not matches:
                raise ValueError(
                    f"No supported external NDArray at {dataset!r}; specify an external array leaf"
                )
            if len(matches) != 1:
                raise ValueError("duplicate B2Z array member")
            info = matches[0]
            if info.flag_bits & 1 or info.compress_type != zipfile.ZIP_STORED:
                raise NotImplementedError("B2Z array members must be unencrypted ZIP_STORED entries")
            if info.compress_size != info.file_size or not 0 <= info.header_offset <= size - 30:
                raise ValueError("invalid B2Z member size or offset")
            # Cover the local header and the native reader's 8 KiB frame prefix
            # together. Unusually long ZIP headers fall back to exact reads.
            prefix = self._read_archive(info.header_offset, min(16384, size - info.header_offset))
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
            self.member_offset = (
                info.header_offset
                + 30
                + int.from_bytes(header[26:28], "little")
                + int.from_bytes(header[28:30], "little")
            )
            self.member_length = info.file_size
            if self.member_offset + self.member_length > size:
                raise ValueError("B2Z member exceeds archive bounds")
        from fsspec.utils import tokenize

        self.stamp = tokenize(urlpath, object_info, dataset, self.member_offset, self.member_length)
        super().__init__(urlpath, max_concurrency, traffic=self.traffic)
        self._opening_ranges.clear()
        if b"b2o" in self._header[13][1]:
            raise NotImplementedError("B2Z object carriers are not supported; select a plain NDArray")
        if not self._header_len <= self._header[2] <= self.member_length:
            raise ValueError("Blosc2 frame exceeds B2Z member bounds")

    def _read_archive(self, offset, size):
        if not size:
            return b""
        for start, data in self._opening_ranges:
            if start <= offset and offset + size <= start + len(data):
                return data[offset - start : offset - start + size]
        data = self._fs.cat_file(self._path, start=offset, end=offset + size)
        if len(data) > size:
            raise ValueError("B2Z transport did not honor the requested byte range")
        self.traffic.charge(len(data))
        return data

    def read_range(self, offset, size):
        offset, size = operator.index(offset), operator.index(size)
        if offset < 0 or size < 0:
            raise ValueError("invalid B2Z frame range")
        size = max(0, min(size, self.member_length - offset))
        return self._read_archive(self.member_offset + offset, size)
