"""Read-only Caterva2 SChunk byte streams, not array or object deserialization."""

import hashlib
import os
import struct
import tempfile
import time
import weakref
from pathlib import Path

import blosc2
from blosc2.remote_array import RemoteMetadataMapping
from blosc2.remote_object import RemoteObject

MAX_COMPRESSED_CHUNK = 32 << 20
MAX_DECODED_CHUNK = 256 << 20
MAX_READ_BYTES = 16 << 20


def validate_file_metadata(info):
    """Accept regular fixed-size SChunks; reject ambiguous/variable-length layouts."""
    for name in ("nbytes", "cbytes", "nchunks", "chunksize"):
        value = info.get(name)
        if type(value) is not int or value < 0:
            raise ValueError(f"Invalid Caterva2 file {name}")
    size, chunk = info["nbytes"], info["chunksize"]
    if (chunk == 0 and size) or info["nchunks"] != ((size + chunk - 1) // chunk if chunk else 0):
        raise ValueError("Caterva2 file requires a fixed-chunk byte stream")
    params = info.get("cparams")
    if not isinstance(params, dict) or type(params.get("typesize")) is not int or params["typesize"] <= 0:
        raise ValueError("Invalid Caterva2 file compression metadata")
    return {key: info[key] for key in ("nbytes", "cbytes", "nchunks", "chunksize", "cparams")}


class RemoteFile(RemoteObject):
    """A lazy, read-only Caterva2 byte stream.

    ``read_bytes(start, stop)`` returns original bytes (at most 16 MiB per call).
    ``download(destination)`` streams original bytes with atomic publication and
    no overwrite by default. Chunk work is limited to 32 MiB compressed / 256 MiB
    decoded; oversized or irregular streams require server-side rechunking.
    Caches and lifetime are shared with the containing RemoteStore owner.
    """

    @classmethod
    def _from_owner(cls, owner, path):
        obj = object.__new__(cls)
        obj._owner, obj.path = owner, path
        obj._generation = owner.generation
        obj._meta = validate_file_metadata(owner.metadata["caterva2_files"][path])
        owner.acquire()
        obj._finalizer = weakref.finalize(obj, owner.release)
        return obj

    def _check_open(self):
        if not self._finalizer.alive:
            raise RuntimeError("RemoteFile is closed")
        if self._generation != self._owner.generation:
            raise RuntimeError("RemoteFile is stale; reopen after refresh")

    @property
    def source(self):
        self._check_open()
        from blosc2.remote_store import caterva2_source_descriptor

        return caterva2_source_descriptor(blosc2.URLPath(self.path, urlbase=self._owner.caterva2.urlbase))

    @property
    def name(self):
        self._check_open()
        return self.path.rsplit("/", 1)[-1]

    @property
    def media_type(self):
        """A filename-based MIME hint, not a guarantee about the payload."""
        import mimetypes

        return mimetypes.guess_type(self.name)[0]

    @property
    def nbytes(self):
        self._check_open()
        return self._meta["nbytes"]

    @property
    def cbytes(self):
        self._check_open()
        return self._meta["cbytes"]

    @property
    def chunksize(self):
        self._check_open()
        return self._meta["chunksize"]

    @property
    def nchunks(self):
        self._check_open()
        return self._meta["nchunks"]

    @property
    def attrs(self):
        self._check_open()
        return RemoteMetadataMapping(self._owner.attrs.get(self.path, {}))

    @property
    def traffic(self):
        self._check_open()
        return self._owner.traffic

    @property
    def cache_policy(self):
        self._check_open()
        return self._owner.cache_policy

    @property
    def max_cache_bytes(self):
        self._check_open()
        return self._owner.max_cache_bytes

    @property
    def cache_bytes(self):
        """Retained bytes for the shared owner, including other leaves."""
        self._check_open()
        return self._owner.cache_coordinator.cache_bytes

    @property
    def metadata_bytes(self):
        self._check_open()
        return self._owner.metadata_bytes

    @property
    def info(self):
        from blosc2.info import InfoReporter

        return InfoReporter(self)

    @property
    def info_items(self):
        self._check_open()
        return [
            ("type", "RemoteFile"),
            ("name", self.name),
            ("source", self.source),
            ("nbytes", self.nbytes),
            ("cbytes", self.cbytes),
            ("chunksize", self._meta["chunksize"]),
            ("nchunks", self._meta["nchunks"]),
            ("cache policy", self.cache_policy),
            ("cache bytes (owner)", self.cache_bytes),
        ]

    def _fetch_chunk(self, index, cancel):
        from blosc2.c2array import _auth_headers, _server_url, _sync_client

        owner = self._owner
        url = _server_url(owner.caterva2.urlbase, f"api/chunk/{self.path}")
        client = owner.transport or _sync_client()
        data = bytearray()
        deadline = time.monotonic() + 20
        headers = dict(_auth_headers(owner.caterva2.auth_token) or {})
        headers["Accept-Encoding"] = "identity"
        with client.stream(
            "GET", url, params={"nchunk": index}, headers=headers, timeout=10, follow_redirects=False
        ) as response:
            response.raise_for_status()
            if response.headers.get("content-encoding", "identity") != "identity":
                raise ValueError("Encoded file chunk responses are unsupported")
            if int(response.headers.get("content-length", 0)) > MAX_COMPRESSED_CHUNK:
                raise ValueError(f"File chunk exceeds {MAX_COMPRESSED_CHUNK >> 20} MiB compressed limit")
            received = 0
            try:
                for part in response.iter_bytes():
                    received += len(part)
                    if time.monotonic() > deadline:
                        raise TimeoutError("File chunk transfer exceeded 20 seconds")
                    if cancel is not None and cancel():
                        raise InterruptedError("File operation cancelled")
                    if len(data) + len(part) > MAX_COMPRESSED_CHUNK:
                        raise ValueError(f"File chunk exceeds {MAX_COMPRESSED_CHUNK >> 20} MiB compressed limit")
                    data.extend(part)
            finally:
                # One HTTP response is one request, regardless of fragmentation.
                # Also retain bytes already received on failure/cancellation.
                owner.traffic.charge(received)
        return bytes(data)

    def _chunk(self, index, cancel=None):
        from blosc2.remote_store import _Caterva2FrameCache

        self._check_open()
        if cancel is not None and cancel():
            raise InterruptedError("File operation cancelled")
        size = min(self._meta["chunksize"], self.nbytes - index * self._meta["chunksize"])
        if size > MAX_DECODED_CHUNK:
            raise ValueError(
                f"File chunk exceeds {MAX_DECODED_CHUNK >> 20} MiB decoded limit; rechunk on the server"
            )
        owner = self._owner
        key = hashlib.sha256(f"file-chunk:{self.path}:{index}".encode()).hexdigest()
        cache = None
        if self.cache_policy is not blosc2.CachePolicy.NONE:
            if owner.table_frame_cache is None:
                owner.table_frame_cache = _Caterva2FrameCache(owner)
            cache = owner.table_frame_cache
        payload = None if cache is None else cache.get(key, max_bytes=MAX_COMPRESSED_CHUNK)
        missing = payload is None
        if payload is None:
            payload = self._fetch_chunk(index, cancel)
        if len(payload) < 16 or len(payload) > MAX_COMPRESSED_CHUNK:
            raise ValueError("Invalid compressed file chunk length")
        nbytes, _, cbytes = struct.unpack_from("<III", payload, 4)
        if nbytes != size or cbytes != len(payload):
            raise ValueError("Compressed file chunk does not match metadata")
        result = blosc2.decompress(payload)
        if len(result) != size:
            raise ValueError("Decoded file chunk does not match metadata")
        if cache is not None and missing:
            cache.put(key, payload)
        return result

    def read_bytes(self, start=0, stop=None):
        """Read an explicitly bounded byte interval; omitted stop means end of file."""
        with self._owner.lock:
            self._check_open()
            stop = self.nbytes if stop is None else stop
            if type(start) is not int or type(stop) is not int or not 0 <= start <= stop <= self.nbytes:
                raise ValueError("Byte interval must satisfy 0 <= start <= stop <= nbytes")
            if stop - start > MAX_READ_BYTES:
                raise ValueError("read_bytes exceeds 16 MiB; use streaming download")
            if start == stop:
                return b""
            chunk = self._meta["chunksize"]
            result = bytearray()
            for index in range(start // chunk, (stop - 1) // chunk + 1):
                data = self._chunk(index)
                result.extend(data[max(0, start - index * chunk) : min(len(data), stop - index * chunk)])
            return bytes(result)

    def download(self, destination, *, overwrite=False, progress=None, cancel=None):
        """Stream original bytes, publishing only on success.

        ``progress(done, total)`` and ``cancel()`` run on the caller's thread.
        A false overwrite uses an atomic hard-link publish to avoid races,
        or a checked rename on Emscripten's single-threaded virtual filesystem.
        """
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
        fd, temporary = tempfile.mkstemp(prefix=".b2view-download-", dir=destination.parent)
        try:
            with os.fdopen(fd, "wb") as file:
                done = 0
                for index in range(self._meta["nchunks"]):
                    with self._owner.lock:
                        data = self._chunk(index, cancel)
                    file.write(data)
                    done += len(data)
                    if progress is not None:
                        progress(done, self.nbytes)
                file.flush()
                os.fsync(file.fileno())
            self._check_open()
            if cancel is not None and cancel():
                raise InterruptedError("File operation cancelled")
            if overwrite:
                os.replace(temporary, destination)
            elif blosc2.IS_WASM:
                # No hard links on Emscripten. No callbacks/awaits can interleave
                # a virtual-FS writer between this final check and publication.
                if os.path.lexists(destination):
                    raise FileExistsError(destination)
                os.rename(temporary, destination)
            else:
                os.link(temporary, destination)
            return str(destination)
        finally:
            Path(temporary).unlink(missing_ok=True)

    def save(self, *args, **kwargs):
        raise NotImplementedError(
            "RemoteFile reference persistence is unsupported; use download for original bytes"
        )

    def close(self):
        self._finalizer()
