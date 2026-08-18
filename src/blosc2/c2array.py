#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

from __future__ import annotations

import atexit
import math
import os
import threading
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

import numpy as np

import blosc2
from blosc2.b2objects import encode_b2object_payload, make_b2object_carrier, write_b2object_payload
from blosc2.info import InfoReporter, format_nbytes_info

# blosc2/__init__ imports this module before blosc2.proxy, so this pulls proxy in
# early; it is safe because proxy only reaches into the package at call time
from blosc2.proxy import REMOTE_MAX_CONCURRENCY, ByteRangeNDSource

_subscriber_data = {
    "urlbase": os.environ.get("BLOSC_C2URLBASE"),
    "auth_token": "",
}
"""Caterva2 subscriber data saved by context manager."""

TIMEOUT = 15
"""Default timeout for HTTP requests."""


def _httpx():
    # Lazy import: c2array.py is imported unconditionally by blosc2/__init__.py,
    # so this keeps httpx's import cost off users who never touch C2Array/Proxy.
    import httpx

    return httpx


_client = None
_client_lock = threading.Lock()


def _forgetful_cookies(httpx):
    """A cookie jar that never keeps anything, for the shared client.

    A client of its own per request could not carry a cookie from one request to
    the next; a shared one can, and must not: the token belongs to the C2Array
    being read, and arrays with different tokens (or none) share this client.  A
    `Set-Cookie` from any response would otherwise start authorizing requests
    that asked for none.
    """

    class _NoCookies(httpx.Cookies):
        def extract_cookies(self, response):
            pass

    return _NoCookies()


def _sync_client():
    """The process-wide HTTP client every synchronous request goes through.

    `httpx.get()` builds a client, opens a connection and negotiates TLS for
    each call and throws all of it away afterwards, which over a WAN link
    measured ~0.37 s per request -- most of what a small read costs, and paid
    once per chunk.  A pooled client keeps the connection alive between
    requests; it is thread-safe, which is what lets `Proxy` fan its fetches out.

    Auth stays per call rather than on the client: the cookie belongs to the
    C2Array being read, and several of them (with different tokens, or none)
    share this one client.
    """
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                httpx = _httpx()
                # More connections than `Proxy`'s default concurrency, so that a
                # caller raising it does not queue on the pool; keepalive covers
                # the fan-out of one fetch, which is what there is to reuse
                _client = httpx.Client(
                    timeout=TIMEOUT,
                    limits=httpx.Limits(max_connections=64, max_keepalive_connections=32),
                    cookies=_forgetful_cookies(httpx),
                )
    return _client


@atexit.register
def _close_sync_client():
    global _client
    if _client is not None:
        _client.close()
        _client = None


@contextmanager
def c2context(
    *,
    urlbase: (str | None) = None,
    username: (str | None) = None,
    password: (str | None) = None,
    auth_token: (str | None) = None,
) -> None:
    """
    Context manager that sets parameters in Caterva2 subscriber requests.

    A parameter not specified or set to ``None`` will inherit the value from the
    previous context manager, defaulting to an environment variable (see
    below) if supported by that parameter.  Parameters set to an empty string
    will not be used in requests (without a default either).

    If the subscriber requires authorization for requests, you can either
    provide an `auth_token` (which you should have obtained previously from the
    subscriber), or both `username` and `password` to obtain the token by
    logging in to the subscriber.  The token will be reused until it is explicitly
    reset or requested again in a later context manager invocation.

    Please note that this manager is reentrant but not safe for concurrent use.

    Parameters
    ----------
    urlbase : str | None
        The base URL to be used when a C2Array instance does not have a subscriber
        URL base set. If not specified, it defaults to the value of the
        ``BLOSC_C2URLBASE`` environment variable.
    username : str | None
        The username for logging in to the subscriber to obtain an authorization token.
        If not specified, it defaults to the value of the ``BLOSC_C2USERNAME`` environment variable.
    password : str | None
        The password for logging in to the subscriber to obtain an authorization token.
        If not specified, it defaults to the value of the ``BLOSC_C2PASSWORD`` environment variable.
    auth_token : str | None
        The authorization token to be used when a C2Array instance does not have an
        authorization token set.

    Yields
    ------
    out: None

    """
    global _subscriber_data
    print("_subscriber_data", _subscriber_data)

    # Perform login to get an authorization token.
    if not auth_token:
        username = username or os.environ.get("BLOSC_C2USERNAME")
        password = password or os.environ.get("BLOSC_C2PASSWORD")
    if username or password:
        if auth_token:
            raise ValueError("Either provide a username/password or an authorization token")
        auth_token = login(username, password, urlbase)

    try:
        old_sub_data = _subscriber_data
        new_sub_data = old_sub_data.copy()  # inherit old values
        if urlbase is not None:
            new_sub_data["urlbase"] = urlbase
        elif old_sub_data["urlbase"] is None:
            # The variable may have gotten a value after program start.
            new_sub_data["urlbase"] = os.environ.get("BLOSC_C2URLBASE")
        if auth_token is not None:
            new_sub_data["auth_token"] = auth_token
        _subscriber_data = new_sub_data
        yield
    finally:
        _subscriber_data = old_sub_data


def _auth_headers(auth_token, headers=None):
    auth_token = auth_token or _subscriber_data["auth_token"]
    if auth_token:
        headers = headers.copy() if headers else {}
        headers["Cookie"] = auth_token
    return headers


def _xget(url, params=None, headers=None, auth_token=None, timeout=TIMEOUT):
    headers = _auth_headers(auth_token, headers)
    response = _sync_client().get(url, params=params, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response


def _xpost(url, json=None, auth_token=None, timeout=TIMEOUT):
    auth_token = auth_token or _subscriber_data["auth_token"]
    headers = {"Cookie": auth_token} if auth_token else None
    response = _sync_client().post(url, json=json, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _sub_url(urlbase, path):
    urlbase = urlbase or _subscriber_data["urlbase"]
    if not urlbase:
        raise RuntimeError("No default Caterva2 subscriber set")
    return f"{urlbase}{path}" if urlbase.endswith("/") else f"{urlbase}/{path}"


def login(username, password, urlbase):
    url = _sub_url(urlbase, "auth/jwt/login")
    creds = {"username": username, "password": password}
    # Not the pooled client: this is the one request whose Set-Cookie matters,
    # and it belongs to the caller rather than to every later request
    resp = _httpx().post(url, data=creds, timeout=TIMEOUT)
    resp.raise_for_status()
    return "=".join(list(resp.cookies.items())[0])


def info(path, urlbase, params=None, headers=None, model=None, auth_token=None):
    url = _sub_url(urlbase, f"api/info/{path}")
    response = _xget(url, params, headers, auth_token)
    json = response.json()
    return json if model is None else model(**json)


def fetch_data(path, urlbase, params, auth_token=None, as_blosc2=False):
    url = _sub_url(urlbase, f"api/fetch/{path}")
    response = _xget(url, params=params, auth_token=auth_token)
    data = response.content
    # Try different deserialization methods
    try:
        data = blosc2.ndarray_from_cframe(data)
    except RuntimeError:
        data = blosc2.schunk_from_cframe(data)
    if as_blosc2:
        return data
    if hasattr(data, "ndim"):  # if b2nd or b2frame
        # catch 0d case where [:] fails
        return data[()] if data.ndim == 0 else data[:]
    else:
        return data[:]


def slice_to_string(slice_):
    if slice_ is None or slice_ == () or slice_ == slice(None):
        return ""
    slice_parts = []
    if not isinstance(slice_, tuple):
        slice_ = (slice_,)
    for index in slice_:
        if isinstance(index, int):
            slice_parts.append(str(index))
        elif isinstance(index, slice):
            start = index.start or ""
            stop = index.stop or ""
            if index.step not in (1, None):
                raise IndexError("Only step=1 is supported")
            # step = index.step or ''
            slice_parts.append(f"{start}:{stop}")
    return ", ".join(slice_parts)


_UNTRIED = object()
"""A block source that has not been asked for yet, as against one that failed."""

MAX_RANGES_PER_REQUEST = 64
"""How many byte ranges one request to a subscriber may ask for.

There is no limit in the protocol, and the saving grows with the count -- but a
`Range` header is a header, which servers and proxies cap the length of (8 KB is
the usual figure, and 64 spans of a large frame are about a kilobyte), and a
failed request costs a round trip and every span in it.
"""


class _NotRanged(Exception):
    """The subscriber answered a range request with something other than a 206."""


class _PartsMissing(Exception):
    """A multi-range answer did not carry all the bytes that were asked for."""


def _content_range(value: str) -> int:
    """Where a `Content-Range: bytes start-end/total` header says its part starts."""
    return int(value.split()[1].split("-")[0])


def _byteranges(response) -> list[tuple[int, bytes]]:
    """The parts of a 206, as (where each starts in the frame, its bytes).

    One part for an ordinary 206, several for a `multipart/byteranges` body: a
    boundary line, the part's own headers, a blank line and its bytes, over and
    over, ending in the boundary followed by two dashes.  `email` would parse it,
    at the price of decoding a megabyte of compressed data as text.
    """
    content_type = response.headers.get("content-type", "")
    if "multipart/byteranges" not in content_type:
        return [(_content_range(response.headers["content-range"]), response.content)]
    boundary = content_type.split("boundary=")[1].strip().strip('"').encode()
    parts = []
    for chunk in response.content.split(b"--" + boundary):
        head, sep, body = chunk.partition(b"\r\n\r\n")
        if not sep:  # the preamble before the first boundary, and the closing --
            continue
        for line in head.split(b"\r\n"):
            name, _, value = line.partition(b":")
            if name.strip().lower() == b"content-range":
                # The body ends with the CRLF that belongs to the next boundary
                parts.append((_content_range(value.decode()), body[: body.rfind(b"\r\n")]))
                break
    return parts


def _span_of(parts: list[tuple[int, bytes]], offset: int, size: int, url: str) -> bytes:
    """The bytes of one requested span, out of whichever part covers it."""
    for start, data in parts:
        if start <= offset < start + len(data):
            return data[offset - start : offset - start + size]
    raise _PartsMissing(f"{url} answered without the bytes at {offset}, which were asked for")


class C2NDSource(ByteRangeNDSource):
    """The frame behind a :ref:`C2Array`, read over HTTP byte ranges.

    Caterva2 serves a *stored* dataset with a Starlette ``FileResponse``, which
    implements RFC 7233 by itself: a ranged request comes back 206 with only the
    bytes asked for, seeked to in the file rather than materialized, and the auth
    cookie composes with it.  That is everything :ref:`ByteRangeNDSource` needs,
    so a slice costs the blocks it touches instead of the chunks they live in.

    A dataset the subscriber *builds* -- a lazy expression, an HDF5 leaf, a
    ``.b2z`` member -- is streamed instead, and a streamed response ignores the
    ``Range`` header and answers with the whole body.  :meth:`read_range` refuses
    such an answer without reading it off the socket, and :ref:`C2Array` then
    keeps to whole chunks for good.  Which is why this is built through
    :meth:`C2Array.block_source` rather than directly: the fallback belongs with
    the array, whose ``api/chunk`` path works for every dataset there is.
    """

    max_ranges = MAX_RANGES_PER_REQUEST

    def __init__(self, array: C2Array, max_concurrency: int = REMOTE_MAX_CONCURRENCY):
        self._url = _sub_url(array.urlbase, f"api/fetch/{array.path}")
        self._auth_token = array.auth_token
        super().__init__(self._url, max_concurrency)

    def read_range(self, offset: int, size: int) -> bytes:
        return self._get([(offset, size)])[0]

    def read_ranges(self, spans: Sequence[tuple[int, int]]) -> list[bytes]:
        """Every span in one request, which HTTP has a shape for and S3 has not.

        RFC 7233 lets a `Range` header name several spans, and the answer is a
        `multipart/byteranges` body carrying each with its own `Content-Range`.
        Starlette builds that, so a whole wave of block reads -- across chunks,
        since they are all the same file -- costs one round trip instead of one
        each.  Measured against cat2.cloud: 32 spans in 0.136 s against 0.208 s
        for 32 requests eight at a time, and 1.530 s for them one at a time.

        The server may serve fewer parts than were asked for: Starlette sorts the
        spans and merges the ones that touch, and answers a single 206 when they
        all merge into one.  So the answer is taken apart by what each part says
        it holds, and each span read out of the part that covers it, rather than
        by trusting the order.  A server that answers a multi-range request with
        bytes that do not cover the whole of it is not one to ask again: this
        keeps to a range per request from there on.
        """
        spans = list(spans)
        if len(spans) > 1 and self.max_ranges > 1:
            try:
                return self._get(spans)
            except _PartsMissing:
                self.max_ranges = 1
        return [self.read_range(*span) for span in spans]

    def _get(self, spans: list[tuple[int, int]]) -> list[bytes]:
        wanted = ", ".join(f"{offset}-{offset + size - 1}" for offset, size in spans)
        headers = _auth_headers(self._auth_token, {"Range": f"bytes={wanted}"})
        with _sync_client().stream("GET", self._url, headers=headers) as response:
            if response.status_code != 206:
                # Whatever this is, it is not the bytes that were asked for: a 200
                # carries the whole dataset, which is the download this exists to
                # avoid, so leave the body unread on the socket
                raise _NotRanged(f"{self._url} answered {response.status_code} to a Range request")
            response.read()
            parts = _byteranges(response)
        return [_span_of(parts, offset, size, self._url) for offset, size in spans]


class C2Array(blosc2.Operand):
    """Remote compressed NDArray accessed from a Caterva2 server."""

    max_concurrency = REMOTE_MAX_CONCURRENCY
    """How many fetches a :ref:`Proxy` over this array may run at once.

    Every chunk or block is a request whose cost is mostly the round trip, so
    overlapping them is what a remote source has to gain; :meth:`Proxy.afetch`
    already used this figure for a `C2Array`, and `fetch` was serial only for
    want of somewhere to read it from.  `get_chunk` and the range reads are
    thread-safe: they share one pooled HTTP client and hold no state of their own.
    """

    def __init__(self, path: str, /, urlbase: str | None = None, auth_token: str | None = None):
        """Create an instance of a remote NDArray.

        Remote NDArrays can be accessed via HTTP from a Caterva2 server
        (e.g., https://cat2.cloud). More information about Caterva2 at:
        https://ironarray.io/caterva2.

        Parameters
        ----------
        path: str
            The path to the remote NDArray file (root + file path) as
            a posix path.
        urlbase: str
            The base URL (slash-terminated) of the subscriber to query.
        auth_token: str
            An optional token to authorize requests via HTTP.  Currently, it
            will be sent as an HTTP cookie.

        Returns
        -------
        out: C2Array

        Examples
        --------
        >>> import blosc2
        >>> urlbase = "https://cat2.cloud/demo"
        >>> path = "@public/examples/dir1/ds-3d.b2nd"
        >>> remote_array = blosc2.C2Array(path, urlbase=urlbase)
        >>> remote_array.shape
        (3, 4, 5)
        >>> remote_array.chunks
        (2, 3, 4)
        >>> remote_array.blocks
        (2, 2, 2)
        >>> remote_array.dtype
        dtype('float32')
        """
        if path.startswith("/"):
            raise ValueError("The path should start with a root name, not a slash")
        self.path = path

        if urlbase and not urlbase.endswith("/"):
            urlbase += "/"
        self.urlbase = urlbase

        self.auth_token = auth_token
        self._aclient = None  # lazy async client, shared across aget_chunk calls
        # The block-reading source, built on first use: _UNTRIED, None (this
        # dataset cannot be read in ranges) or a C2NDSource
        self._block_source = _UNTRIED
        self._block_lock = threading.Lock()

        # Try to 'open' the remote path
        try:
            self.meta = info(self.path, self.urlbase, auth_token=self.auth_token)
        except _httpx().HTTPStatusError as err:
            # HTTPStatusError only (not the broader HTTPError, which also covers
            # connection-level failures): a 404 means "not found", a connection
            # failure should propagate as-is rather than be reported as missing.
            raise FileNotFoundError(f"Remote path not found: {path}.\nError was: {err}") from err
        cparams = self.meta["schunk"]["cparams"]
        # Remove "filters, meta" from cparams; this is an artifact from the server
        cparams.pop("filters, meta", None)
        self._cparams = blosc2.CParams(**cparams)

    def __enter__(self) -> C2Array:
        """Enter a context manager and return this remote array."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Exit a context manager.

        ``C2Array`` does not currently hold explicit closeable resources, so this
        is a logical no-op kept for API consistency with :func:`blosc2.open`.
        """
        return False

    def _to_b2object_payload(self) -> dict:
        payload = encode_b2object_payload(self)
        if payload is None:
            raise TypeError("Unsupported persisted Blosc2 object")
        return payload

    def _to_b2object_carrier(self, **kwargs):
        array = make_b2object_carrier(
            "c2array",
            self.shape,
            self.dtype,
            chunks=self.chunks,
            blocks=self.blocks,
            cparams=self.cparams,
            **kwargs,
        )
        write_b2object_payload(array, self._to_b2object_payload())
        return array

    def to_cframe(self) -> bytes:
        """Serialize the remote array reference as a CFrame-backed Blosc2 object."""
        return self._to_b2object_carrier().to_cframe()

    def save(self, urlpath: str, contiguous: bool = True, **kwargs) -> None:
        """Persist the remote array reference using a CFrame-backed carrier."""
        blosc2.blosc2_ext.check_access_mode(urlpath, "w")
        kwargs["urlpath"] = urlpath
        kwargs["contiguous"] = contiguous
        kwargs["mode"] = "w"
        self._to_b2object_carrier(**kwargs)

    def __getitem__(self, slice_: int | slice | Sequence[slice]) -> np.ndarray:
        """
        Get a slice of the array (returning NumPy array).

        Parameters
        ----------
        slice_ : int, slice, tuple of ints and slices, or None
            The slice to fetch.

        Returns
        -------
        out: numpy.ndarray
            A numpy.ndarray containing the data slice.

        Examples
        --------
        >>> import blosc2
        >>> urlbase = "https://cat2.cloud/demo"
        >>> path = "@public/examples/dir1/ds-2d.b2nd"
        >>> remote_array = blosc2.C2Array(path, urlbase=urlbase)
        >>> data_slice = remote_array[3:5, 1:4]
        >>> data_slice.shape
        (2, 3)
        >>> data_slice[:]
        array([[61, 62, 63],
               [81, 82, 83]], dtype=uint16)
        """
        slice_ = slice_to_string(slice_)
        return fetch_data(
            self.path, self.urlbase, {"slice_": slice_}, auth_token=self.auth_token, as_blosc2=False
        )

    def slice(self, slice_: int | slice | Sequence[slice]) -> blosc2.NDArray:
        """
        Get a slice of the array (returning blosc2 NDArray array).

        Parameters
        ----------
        slice_ : int, slice, tuple of ints and slices, or None
            The slice to fetch.

        Returns
        -------
        out: blosc2.NDArray
            A blosc2.NDArray containing the data slice.

        Examples
        --------
        >>> import blosc2
        >>> urlbase = "https://cat2.cloud/demo"
        >>> path = "@public/examples/dir1/ds-2d.b2nd"
        >>> remote_array = blosc2.C2Array(path, urlbase=urlbase)
        >>> data_slice = remote_array.slice((slice(3,5), slice(1,4)))
        >>> data_slice.shape
        (2, 3)
        >>> type(data_slice)
        blosc2.ndarray.NDArray
        """
        slice_ = slice_to_string(slice_)
        return fetch_data(
            self.path, self.urlbase, {"slice_": slice_}, auth_token=self.auth_token, as_blosc2=True
        )

    def __len__(self) -> int:
        """Returns the length of the first dimension of the array.
        This is equivalent to ``self.shape[0]``.
        """
        return self.shape[0]

    def get_chunk(self, nchunk: int) -> bytes:
        """
        Get the compressed unidimensional chunk of a :ref:`C2Array`.

        Parameters
        ----------
        nchunk: int
            The index of the unidimensional chunk to retrieve.

        Returns
        -------
        out: bytes
            The requested compressed chunk.

        Examples
        --------
        >>> import numpy as np
        >>> import blosc2
        >>> urlbase = "https://cat2.cloud/demo"
        >>> path = "@public/examples/dir1/ds-3d.b2nd"
        >>> a = blosc2.C2Array(path, urlbase)
        >>>  # Get the compressed chunk from array 'a' for index 0
        >>> compressed_chunk = a.get_chunk(0)
        >>> f"Size of chunk {0} from a: {len(compressed_chunk)} bytes"
        Size of chunk 0 from a: 160 bytes
        >>> # Decompress the chunk and convert it to a NumPy array
        >>> decompressed_chunk = blosc2.decompress(compressed_chunk)
        >>> np.frombuffer(decompressed_chunk, dtype=a.dtype)
        array([ 0.,  1.,  5.,  6., 20., 21., 25., 26.,  2.,  3.,  7.,  8., 22.,
               23., 27., 28., 10., 11.,  0.,  0., 30., 31.,  0.,  0., 12., 13.,
                0.,  0., 32., 33.,  0.,  0.], dtype=float32)
        """
        url = _sub_url(self.urlbase, f"api/chunk/{self.path}")
        params = {"nchunk": nchunk}
        response = _xget(url, params=params, auth_token=self.auth_token)
        return response.content

    async def aget_chunk(self, nchunk: int) -> bytes:
        """
        Get the compressed unidimensional chunk of a :ref:`C2Array` asynchronously.

        Same as :meth:`get_chunk`, but performs the HTTP GET with an
        ``httpx.AsyncClient`` instead of blocking the event loop. Used by
        :meth:`Proxy.afetch` to fetch multiple chunks concurrently. The
        underlying client is created lazily and reused across calls; close it
        explicitly with :meth:`aclose` when done, e.g. when the event loop is
        about to be torn down.

        Parameters
        ----------
        nchunk: int
            The index of the unidimensional chunk to retrieve.

        Returns
        -------
        out: bytes
            The requested compressed chunk.
        """
        url = _sub_url(self.urlbase, f"api/chunk/{self.path}")
        params = {"nchunk": nchunk}
        headers = _auth_headers(self.auth_token)
        if self._aclient is None:
            self._aclient = _httpx().AsyncClient(timeout=TIMEOUT)
        response = await self._aclient.get(url, params=params, headers=headers)
        response.raise_for_status()
        return response.content

    async def aclose(self) -> None:
        """Close the underlying async HTTP client opened by :meth:`aget_chunk`, if any."""
        if self._aclient is not None:
            await self._aclient.aclose()
            self._aclient = None

    # -- Block-granular reads.  A :ref:`Proxy` uses these to fetch the blocks a
    # slice touches instead of whole chunks, wherever that is the cheaper way
    # round; every one of them falls back to `get_chunk` when it is not.

    @property
    def stamp(self) -> str | None:
        """What names the exact remote bytes, for a :ref:`Proxy` to check a cache by.

        Geometry cannot tell a dataset that was replaced from the one a cache was
        filled from: a shape and a partitioning survive a rewrite, while every
        cached chunk -- and, in block mode, every offset they were fetched by --
        goes stale.  The subscriber's own mtime does tell, and `api/info` carries
        it, so this costs no request; the compressed size goes in with it, since
        a rewrite within the same clock tick is what an mtime cannot see.

        None when the subscriber reports no mtime, which leaves the cache checked
        on its geometry alone, as every source without a stamp is.
        """
        mtime = self.meta.get("mtime")
        if mtime is None:
            return None
        return f"{mtime}:{self.meta['schunk'].get('cbytes', '')}"

    @property
    def blocks_per_chunk(self) -> int:
        """How many blocks a chunk of the remote array holds.

        Geometry, and `api/info` already carries it, so this costs no request:
        chunks are padded to whole blocks, so every chunk holds the same number
        of them, edge chunks included.
        """
        blocks = self.blocks
        if not all(blocks):  # an empty array partitions into nothing
            return 1
        return math.prod(math.ceil(c / b) for c, b in zip(self.chunks, blocks, strict=True))

    def block_source(self) -> C2NDSource | None:
        """The frame reader behind the block methods, or None if there is none.

        Built on the first request for it and never rebuilt.  The fallback has to
        be permanent: a subscriber that streams this dataset answers a range
        request with the whole body, so retrying would pay a full download to
        rediscover the same answer.
        """
        if self._block_source is _UNTRIED:
            with self._block_lock:
                if self._block_source is _UNTRIED:
                    self._block_source = self._open_block_source()
        return self._block_source

    def _open_block_source(self) -> C2NDSource | None:
        """Decide, at whatever cost it takes, whether this dataset serves ranges."""
        # `api/info` rules out a dataset the subscriber computes for nothing: a
        # stored one reports its geometry where a lazy expression reports
        # `expression` and `operands`
        if not all(key in self.meta for key in ("chunks", "blocks", "schunk")):
            return None
        # Nor is a frame of small chunks worth an index read: blosc2 declines to
        # take a chunk below BLOCK_MIN_CBYTES apart, so nothing here would ever
        # use a block, and the dataset keeps exactly the behaviour it had before
        nchunks = math.prod(math.ceil(s / c) for s, c in zip(self.shape, self.chunks, strict=True))
        if not nchunks or self.cbytes / nchunks < blosc2.proxy.BLOCK_MIN_CBYTES:
            return None
        # Whether a dataset that reports a geometry is *served* from a file is
        # something only the answer to a range request can say: an HDF5 leaf or a
        # `.b2z` member reports one and is streamed all the same
        try:
            return C2NDSource(self, max_concurrency=REMOTE_MAX_CONCURRENCY)
        except (_NotRanged, ValueError, NotImplementedError, RuntimeError, _httpx().HTTPError):
            # Not ranged, not a contiguous frame, not an NDArray, or not
            # reachable: whole chunks work for all of those
            return None

    def wants_blocks(self, nchunk: int, nwanted: int) -> bool:
        """Whether fetching *nwanted* blocks of a chunk beats fetching all of it."""
        source = self.block_source()
        return source is not None and source.wants_blocks(nchunk, nwanted)

    @property
    def max_ranges(self) -> int:
        """How many ranges one request to this subscriber may carry."""
        source = self.block_source()
        return 1 if source is None else source.max_ranges

    def chunk_layout(self, nchunk: int):
        """Where the blocks of a chunk are; see :meth:`ByteRangeNDSource.chunk_layout`."""
        return self.block_source().chunk_layout(nchunk)

    def chunk_layouts(self, nchunks: Sequence[int]) -> list:
        """The same for several chunks; see :meth:`ByteRangeNDSource.chunk_layouts`."""
        return self.block_source().chunk_layouts(nchunks)

    def block_plan(self, nchunk: int, nblocks: Sequence[int]) -> list[tuple[int, int, tuple]]:
        """The range reads covering *nblocks*; see :meth:`ByteRangeNDSource.block_plan`."""
        return self.block_source().block_plan(nchunk, nblocks)

    def read_range(self, offset: int, size: int) -> bytes:
        """The bytes at [*offset*, *offset* + *size*) of the remote frame."""
        return self._ranged().read_range(offset, size)

    def read_ranges(self, spans: Sequence[tuple[int, int]]) -> list[bytes]:
        """The bytes of every span, in one request where the subscriber allows it."""
        return self._ranged().read_ranges(spans)

    def _ranged(self) -> C2NDSource:
        source = self.block_source()
        if source is None:
            raise ValueError(f"{self.path} is not served in byte ranges by {self.urlbase}")
        return source

    @property
    def shape(self) -> tuple[int]:
        """The shape of the remote array"""
        return tuple(self.meta["shape"])

    @property
    def chunks(self) -> tuple[int]:
        """The chunks of the remote array"""
        return tuple(self.meta["chunks"])

    @property
    def blocks(self) -> tuple[int]:
        """The blocks of the remote array"""
        return tuple(self.meta["blocks"])

    @property
    def dtype(self) -> np.dtype:
        """The dtype of the remote array"""
        return np.dtype(self.meta["dtype"])

    @property
    def cparams(self) -> blosc2.CParams:
        """The compression parameters of the remote array"""
        return self._cparams

    @property
    def nbytes(self) -> int:
        """The number of bytes of the remote array"""
        return self.meta["schunk"]["nbytes"]

    @property
    def cbytes(self) -> int:
        """The number of compressed bytes of the remote array"""
        return self.meta["schunk"]["cbytes"]

    @property
    def cratio(self) -> float:
        """The compression ratio of the remote array"""
        return self.meta["schunk"]["cratio"]

    # TODO: Add these to SChunk model in srv_utils and then access them here
    # @property
    # def dparams(self) -> float:
    #     """The dparams of the remote array"""
    #     return
    #
    # @property
    # def meta(self) -> float:
    #     """The meta of the remote array"""
    #     return

    # TODO: This seems to cause problems for proxy sources (see tests/ndarray/test_proxy_c2array.py::test_open)
    # @property
    # def urlpath(self) -> str:
    #     """The URL path of the remote array"""
    #     return self.meta["schunk"]["urlpath"]

    @property
    def vlmeta(self) -> dict:
        """The variable-length metadata f the remote array"""
        return self.meta["schunk"]["vlmeta"]

    @property
    def info(self) -> InfoReporter:
        """
        Print information about this remote array.
        """
        return InfoReporter(self)

    @property
    def info_items(self) -> list:
        """A list of tuples with the information about the remote array.
        Each tuple contains the name of the attribute and its value.
        """
        items = []
        items += [("type", f"{self.__class__.__name__}")]
        items += [("shape", self.shape)]
        items += [("chunks", self.chunks)]
        items += [("blocks", self.blocks)]
        items += [("dtype", self.dtype)]
        items += [("nbytes", format_nbytes_info(self.nbytes))]
        items += [("cbytes", format_nbytes_info(self.cbytes))]
        items += [("cratio", f"{self.cratio:.2f}x")]
        items += [("cparams", self.cparams)]
        # items += [("dparams", self.dparams)]
        return items

    # TODO: Access chunksize, size, ext_chunks, etc.
    # @property
    # def size(self) -> int:
    #     """The size (in bytes) for this container."""
    #     return self.cbytes
    # @property
    # def chunksize(self) -> int:
    #     """NOT the same as `SChunk.chunksize <blosc2.schunk.SChunk.chunksize>`
    #     in case :attr:`chunks` is not multiple in
    #     each dimension of :attr:`blocks` (or equivalently, if :attr:`chunks` is
    #     not the same as :attr:`ext_chunks`).
    #     """
    #     return

    @property
    def blocksize(self) -> int:
        """The block size (in bytes) for the remote container."""
        return self.meta["schunk"]["blocksize"]


class URLPath:
    def __init__(self, path: str, /, urlbase: str | None = None, auth_token: str | None = None):
        """
        Create an instance of a remote data file (aka :ref:`C2Array <C2Array>`) urlpath.
        This is meant to be used in the :func:`blosc2.open` function.

        The parameters are the same as for the :meth:`C2Array.__init__`.

        """
        self.path = path
        self.urlbase = urlbase
        self.auth_token = auth_token
