#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

from __future__ import annotations

import asyncio
import atexit
import math
import os
import struct
import threading
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

import numpy as np

import blosc2
from blosc2.b2objects import encode_b2object_payload, make_b2object_carrier, write_b2object_payload
from blosc2.info import InfoReporter, format_nbytes_info
from blosc2.proxy_source import (
    REMOTE_MAX_CONCURRENCY,
    ByteRangeNDSource,
    NotRanged,
    PartsMissing,
    _is_transient,
)

_server_data = {
    "urlbase": os.environ.get("BLOSC_C2URLBASE"),
    "auth_token": "",
}
"""Caterva2 server data saved by context manager."""

TIMEOUT = 15
"""Default timeout for HTTP requests."""


def _httpx():
    # Lazy import: c2array.py is imported unconditionally by blosc2/__init__.py,
    # so this keeps httpx's import cost off users who never touch C2Array/Proxy.
    import httpx

    return httpx


_client = None
_client_lock = threading.Lock()


def _forgetful_cookies():
    """A cookie jar that never keeps anything, for the shared client.

    A client of its own per request could not carry a cookie from one request to
    the next; a shared one can, and must not: the token belongs to the C2Array
    being read, and arrays with different tokens (or none) share this client.  A
    `Set-Cookie` from any response would otherwise start authorizing requests
    that asked for none.

    A jar rather than an `httpx.Cookies` subclass: the client re-wraps whatever
    it is handed in a plain `httpx.Cookies`, which copies the cookies over and
    drops the subclass, but hands a bare `CookieJar` straight through.
    """
    import http.cookiejar

    class _NoCookies(http.cookiejar.CookieJar):
        def extract_cookies(self, response, request):
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
                    cookies=_forgetful_cookies(),
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
    Context manager that sets parameters in Caterva2 server requests.

    A parameter not specified or set to ``None`` will inherit the value from the
    previous context manager, defaulting to an environment variable (see
    below) if supported by that parameter.  Parameters set to an empty string
    will not be used in requests (without a default either).

    If the server requires authorization for requests, you can either
    provide an `auth_token` (which you should have obtained previously from the
    server), or both `username` and `password` to obtain the token by
    logging in to the server.  The token will be reused until it is explicitly
    reset or requested again in a later context manager invocation.

    Please note that this manager is reentrant but not safe for concurrent use.

    Parameters
    ----------
    urlbase : str | None
        The base URL to be used when a C2Array instance does not have a server
        URL base set. If not specified, it defaults to the value of the
        ``BLOSC_C2URLBASE`` environment variable.
    username : str | None
        The username for logging in to the server to obtain an authorization token.
        If not specified, it defaults to the value of the ``BLOSC_C2USERNAME`` environment variable.
    password : str | None
        The password for logging in to the server to obtain an authorization token.
        If not specified, it defaults to the value of the ``BLOSC_C2PASSWORD`` environment variable.
    auth_token : str | None
        The authorization token to be used when a C2Array instance does not have an
        authorization token set.

    Yields
    ------
    out: None

    """
    global _server_data
    print("_server_data", _server_data)

    # Perform login to get an authorization token.
    if not auth_token:
        username = username or os.environ.get("BLOSC_C2USERNAME")
        password = password or os.environ.get("BLOSC_C2PASSWORD")
    if username or password:
        if auth_token:
            raise ValueError("Either provide a username/password or an authorization token")
        auth_token = login(username, password, urlbase)

    try:
        old_server_data = _server_data
        new_server_data = old_server_data.copy()  # inherit old values
        if urlbase is not None:
            new_server_data["urlbase"] = urlbase
        elif old_server_data["urlbase"] is None:
            # The variable may have gotten a value after program start.
            new_server_data["urlbase"] = os.environ.get("BLOSC_C2URLBASE")
        if auth_token is not None:
            new_server_data["auth_token"] = auth_token
        _server_data = new_server_data
        yield
    finally:
        _server_data = old_server_data


def _auth_headers(auth_token, headers=None):
    auth_token = auth_token or _server_data["auth_token"]
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
    auth_token = auth_token or _server_data["auth_token"]
    headers = {"Cookie": auth_token} if auth_token else None
    response = _sync_client().post(url, json=json, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _chunk_headers(auth_token):
    """What a chunk write is sent with: bytes, not the JSON `_xpost` sends."""
    return _auth_headers(auth_token, {"Content-Type": "application/octet-stream"})


def _chunk_written(response, url, nchunk):
    """Read a chunk write's answer, in one place for both ways of sending it.

    The write contract lives here rather than at each call site: a slot that was
    already claimed is the one refusal a writer is meant to act on, and it must
    read the same whether the request went out on the pooled client or on the
    async one.
    """
    if response.status_code == 409:
        raise ChunkAlreadyWritten(f"{url} already holds a chunk at {nchunk}")
    response.raise_for_status()
    return response.json()


def _xpost_bytes(url, content, params=None, auth_token=None, timeout=TIMEOUT):
    """POST a body of bytes through the pooled client, and read what came back.

    `_xpost` sends JSON, which a compressed chunk is not: it goes as it is, and
    the server reads it as the chunk it will store.
    """
    response = _sync_client().post(
        url, params=params, content=content, headers=_chunk_headers(auth_token), timeout=timeout
    )
    return _chunk_written(response, url, params and params.get("nchunk"))


async def _axpost_bytes(client, url, content, params=None, auth_token=None):
    """The same request off the event loop; see :func:`_xpost_bytes`."""
    response = await client.post(url, params=params, content=content, headers=_chunk_headers(auth_token))
    return _chunk_written(response, url, params and params.get("nchunk"))


def _server_url(urlbase, path):
    urlbase = urlbase or _server_data["urlbase"]
    if not urlbase:
        raise RuntimeError("No default Caterva2 server set")
    return f"{urlbase}{path}" if urlbase.endswith("/") else f"{urlbase}/{path}"


def login(username, password, urlbase):
    url = _server_url(urlbase, "auth/jwt/login")
    creds = {"username": username, "password": password}
    # Not the pooled client: this is the one request whose Set-Cookie matters,
    # and it belongs to the caller rather than to every later request
    resp = _httpx().post(url, data=creds, timeout=TIMEOUT)
    resp.raise_for_status()
    return "=".join(list(resp.cookies.items())[0])


def info(path, urlbase, params=None, headers=None, model=None, auth_token=None):
    url = _server_url(urlbase, f"api/info/{path}")
    response = _xget(url, params, headers, auth_token)
    json = response.json()
    return json if model is None else model(**json)


def fetch_data(path, urlbase, params, auth_token=None, as_blosc2=False):
    url = _server_url(urlbase, f"api/fetch/{path}")
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
        else:
            # Anything else has no spelling here, and dropping it would widen the
            # request rather than narrow it: a fancy index skipped this way asks
            # `api/fetch` for the whole dataset and hands back all of it, which
            # is neither what was asked for nor a smaller answer
            raise IndexError(
                f"Cannot ask a Caterva2 server for {index!r}: only integers and "
                "step-1 slices can be expressed in a fetch request"
            )
    return ", ".join(slice_parts)


_UNTRIED = object()
"""A block source that has not been asked for yet, as against one that failed."""

MAX_RANGES_PER_REQUEST = 64
"""How many byte ranges one request to a server may ask for.

There is no limit in the protocol, and the saving grows with the count -- but a
`Range` header is a header, which servers and proxies cap the length of (8 KB is
the usual figure, and 64 spans of a large frame are about a kilobyte), and a
failed request costs a round trip and every span in it.
"""


MULTIPART_STRIKES = 2
"""How many answers that did not carry their parts end multi-range requests.

Batching is worth an order of magnitude (see :meth:`C2NDSource.read_ranges`), so
one truncated or unreadable answer is worth retrying a range at a time and
asking again; a server that cannot do it says so twice.
"""


def _content_range(value: str) -> tuple[int, int, int | None]:
    """A `Content-Range: bytes start-end/total` header, as (start, nbytes, total).

    The length comes from the header rather than from the bytes that follow it,
    which is what lets a multipart body be cut where its parts really end rather
    than at whatever looks like a boundary inside a compressed payload.

    Malformed answers are refused rather than guessed at: `bytes */1234` for a
    range that could not be satisfied has no start to read, and a part written to
    no shape at all cannot be placed in the frame.
    """
    try:
        span, _, total = value.split()[1].partition("/")
        start, _, end = span.partition("-")
        nbytes = int(end) - int(start) + 1
    except (IndexError, ValueError) as exc:
        raise NotRanged(f"a 206 carried an unreadable Content-Range: {value!r}") from exc
    if nbytes < 0:
        raise NotRanged(f"a 206 carried a Content-Range that ends before it starts: {value!r}")
    # `*` for a server that will not say how long the whole is, which is allowed
    # and which only costs the tolerance for a read clipped by the end of the frame
    return int(start), nbytes, int(total) if total.isdigit() else None


def _byteranges(response) -> list[tuple[int, bytes, int | None]]:
    """The parts of a 206, as (where each starts, its bytes, how long the frame is).

    One part for an ordinary 206, several for a `multipart/byteranges` body: a
    boundary line, the part's own headers, a blank line and its bytes, over and
    over, ending in the boundary followed by two dashes.  `email` would parse it,
    at the price of decoding a megabyte of compressed data as text.

    Each part is cut to the length its own `Content-Range` gives, and the next
    boundary looked for after it: compressed payloads hold arbitrary bytes, the
    boundary and a trailing CRLF among them, so a body split on either would slice
    parts short wherever the data happened to spell one.
    """
    content_type = response.headers.get("content-type", "")
    if "multipart/byteranges" not in content_type:
        # Where the part sits is the one thing the body cannot say, so an answer
        # without it is refused: a caching proxy that strips the header would
        # otherwise have its bytes placed wherever they were asked for
        single = response.headers.get("content-range")
        if single is None:
            raise NotRanged("a 206 arrived without a Content-Range to place it by")
        start, _, total = _content_range(single)
        return [(start, response.content, total)]
    _, sep, boundary = content_type.partition("boundary=")
    if not sep:
        raise NotRanged(f"a multipart answer named no boundary: {content_type!r}")
    marker = b"--" + boundary.strip().strip('"').encode()
    body = response.content
    parts = []
    pos = body.find(marker)
    while pos != -1:
        pos += len(marker)
        if body[pos : pos + 2] == b"--":
            break  # the closing boundary, and nothing after it belongs to a part
        head_end = body.find(b"\r\n\r\n", pos)
        if head_end == -1:
            raise NotRanged("a multipart part arrived without a blank line to end its headers")
        placed = None
        for line in body[pos:head_end].split(b"\r\n"):
            name, sep, value = line.partition(b":")
            if sep and name.strip().lower() == b"content-range":
                placed = _content_range(value.decode())
                break
        if placed is None:
            raise NotRanged("a multipart part arrived without a Content-Range to place it by")
        start, nbytes, total = placed
        data = body[head_end + 4 : head_end + 4 + nbytes]
        parts.append((start, data, total))
        pos = body.find(marker, head_end + 4 + nbytes)
    return parts


def _span_of(parts: list[tuple[int, bytes, int | None]], offset: int, size: int, url: str) -> bytes:
    """The bytes of one requested span, out of whichever part covers it."""
    for start, data, total in parts:
        if start > offset:
            continue
        # The whole span, not just its first byte: a part that begins inside it
        # and ends early would otherwise slice short, and a short block payload
        # is spliced against a `bstarts` that promises the full length.  Unless
        # it ends where the frame does, which `read_range` allows and an open of
        # a frame shorter than the prefetch relies on.
        end = start + len(data)
        if offset + size <= end or (end == total and offset < end):
            return data[offset - start : offset - start + size]
    raise PartsMissing(f"{url} answered without the bytes at {offset}, which were asked for")


class ChunkAlreadyWritten(ValueError):
    """A chunk was written to a slot of a remote array that already held content.

    A server that accepts chunk writes accepts each slot exactly once: the
    frame's own offsets say whether a slot was ever written, and a second write
    would move every chunk that came after it.  So a writer that finds this has
    lost a race, or is repeating work another writer already did; either way the
    array is intact and the chunk it carried is the one to drop.
    """


class C2NDSource(ByteRangeNDSource):
    """The frame behind a :ref:`C2Array`, read over HTTP byte ranges.

    Caterva2 serves a *stored* dataset with a Starlette ``FileResponse``, which
    implements RFC 7233 by itself: a ranged request comes back 206 with only the
    bytes asked for, seeked to in the file rather than materialized, and the auth
    cookie composes with it.  That is everything :ref:`ByteRangeNDSource` needs,
    so a slice costs the blocks it touches instead of the chunks they live in.

    A dataset the server *builds* -- a lazy expression, an HDF5 leaf, a
    ``.b2z`` member -- is streamed instead, and a streamed response ignores the
    ``Range`` header and answers with the whole body.  :meth:`read_range` refuses
    such an answer without reading it off the socket, and :ref:`C2Array` then
    keeps to whole chunks for good.  Which is why this is built through
    :meth:`C2Array.block_source` rather than directly: the fallback belongs with
    the array, whose ``api/chunk`` path works for every dataset there is.
    """

    max_ranges = MAX_RANGES_PER_REQUEST

    def __init__(self, array: C2Array, max_concurrency: int = REMOTE_MAX_CONCURRENCY):
        self._url = _server_url(array.urlbase, f"api/fetch/{array.path}")
        self._auth_token = array.auth_token
        # Answers that did not carry their parts, in a row; see `read_ranges`
        self._misses = 0
        # The array's own tally, so that what it reads through `api/chunk` and
        # what this reads through `api/fetch` add up to what the dataset cost
        super().__init__(self._url, max_concurrency, traffic=array.traffic)
        # A `Proxy` mixes the two: the block grid and the fetched bitmap come from
        # the array's `api/info`, while the header sections and `bstarts` come from
        # this frame.  They have to be the same dataset for that to mean anything,
        # and the magic bytes alone do not say so -- a window off by a member of a
        # `.b2z`, or a path serving a file other than the one described, reads a
        # frame that parses and splices blocks into chunks of the wrong shape.
        # Geometry alone, since that is what the block arithmetic on both sides is
        # built out of, and a dtype `api/info` reports as a repr would fail to
        # parse here for a dataset that reads perfectly well
        described = (tuple(array.shape), tuple(array.chunks), tuple(array.blocks))
        found = (tuple(self._shape), tuple(self._chunks), tuple(self._blocks))
        if described != found:
            raise ValueError(f"{self._url} serves {found}, where its dataset is {described}")

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
        by trusting the order.  An answer that does not carry the whole of what
        was asked for is retried a range at a time, and only a server that does
        that `MULTIPART_STRIKES` times is written off as unable to batch: an
        order of magnitude for the rest of the process is too much to pay for one
        truncated answer.
        """
        spans = list(spans)
        if len(spans) > 1 and self.max_ranges > 1:
            try:
                answers = self._get(spans)
            except PartsMissing:
                self._misses += 1
                if self._misses >= MULTIPART_STRIKES:
                    self.max_ranges = 1
            else:
                self._misses = 0  # what counts is a server that cannot do this
                return answers
        return [self.read_range(*span) for span in spans]

    def _get(self, spans: list[tuple[int, int]]) -> list[bytes]:
        wanted = ", ".join(f"{offset}-{offset + size - 1}" for offset, size in spans)
        headers = _auth_headers(self._auth_token, {"Range": f"bytes={wanted}"})
        with _sync_client().stream("GET", self._url, headers=headers) as response:
            if response.status_code != 206:
                # Whatever this is, it is not the bytes that were asked for: a 200
                # carries the whole dataset, which is the download this exists to
                # avoid, so leave the body unread on the socket
                raise NotRanged(
                    f"{self._url} answered {response.status_code} to a Range request",
                    response.status_code,
                )
            response.read()
            self.traffic.charge(len(response.content))
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
            The base URL (slash-terminated) of the server to query.
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
        # Set when this handle writes: `meta` describes the array as it was read,
        # and a write of its own moves everything `api/info` reports about it.
        # The epoch counts those writes, so a read of `api/info` that was in
        # flight when one landed can tell that its answer predates it
        self._meta_stale = False
        self._meta_epoch = 0
        self._meta_lock = threading.Lock()
        # An index a `Proxy` handed over before the source existed; see _adopt_index
        self._pending_index = None
        # What this handle has read off the server, whichever endpoint it used;
        # the block source built later is handed this same tally
        self.traffic = blosc2.proxy_source.Traffic()

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
        url = self._chunk_url()
        params = {"nchunk": nchunk}
        response = _xget(url, params=params, auth_token=self.auth_token)
        self.traffic.charge(len(response.content))
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
        url = self._chunk_url()
        params = {"nchunk": nchunk}
        headers = _auth_headers(self.auth_token)
        if self._aclient is None:
            self._aclient = _httpx().AsyncClient(timeout=TIMEOUT)
        response = await self._aclient.get(url, params=params, headers=headers)
        response.raise_for_status()
        self.traffic.charge(len(response.content))
        return response.content

    async def aclose(self) -> None:
        """Close the underlying async HTTP client opened by :meth:`aget_chunk`, if any."""
        if self._aclient is not None:
            await self._aclient.aclose()
            self._aclient = None

    # -- Writing chunks.  A pre-sized array is filled a chunk at a time, by as
    # many writers as there are chunks to fill; the server serializes them
    # and refuses a slot that was already written.

    def update_chunk(self, nchunk: int, chunk: bytes) -> dict:
        """Write one compressed chunk into a slot of the remote array.

        The array has to exist and to be laid out already -- `blosc2.uninit` and
        an upload is what makes one -- and the slot has to be one nothing was
        ever written to.  That is not a restriction the transport invents: a
        chunk written into an empty slot is appended to the frame and moves
        nothing, while one written over a chunk that is already there moves every
        byte after it, so a fill made of writes-once is the cheap one and the one
        whose offsets a concurrent reader can keep.

        The chunk must match the array's geometry -- its chunkshape, its typesize
        and its blocksize -- which is what compressing against
        :attr:`cparams` and :attr:`blocks` gives; the server checks it and
        refuses anything else rather than storing a chunk the array cannot read.

        Parameters
        ----------
        nchunk: int
            Which chunk of the array to write, numbered as
            :meth:`NDArray.get_chunk` numbers them.
        chunk: bytes
            The compressed chunk, as :meth:`SChunk.get_chunk` or
            :func:`blosc2.compress2` produce it.

        Returns
        -------
        out: dict
            What the server reports of the array's state now.  Carries
            ``written`` and ``nchunks`` where it counts them, so a writer can see
            a fill finish without asking again.

        Raises
        ------
        ChunkAlreadyWritten
            The slot already holds a chunk.  The array is untouched.

        Examples
        --------
        >>> import math, blosc2, numpy as np  # doctest: +SKIP
        >>> a = blosc2.C2Array("@personal/run.b2nd", urlbase)  # doctest: +SKIP
        >>> data = np.arange(math.prod(a.chunks), dtype=a.dtype).reshape(a.chunks)  # doctest: +SKIP
        >>> itemsize = a.dtype.itemsize  # doctest: +SKIP
        >>> chunk = blosc2.compress2(  # doctest: +SKIP
        ...     data, typesize=itemsize, blocksize=math.prod(a.blocks) * itemsize
        ... )
        >>> a.update_chunk(0, chunk)  # doctest: +SKIP
        {'written': 1, 'nchunks': 320}

        The blocksize is spelled out because :func:`blosc2.compress2` picks its
        own when it is not: left to choose it takes the whole chunk, and a chunk
        blocked differently from the array is one the server refuses.
        """
        url = self._chunk_url()
        try:
            return _xpost_bytes(url, chunk, params={"nchunk": nchunk}, auth_token=self.auth_token)
        finally:
            # However it went.  A refusal is the answer of a server that has
            # already stored someone else's chunk in that slot, and a request that
            # failed on the way home may have stored this one; either way what
            # this handle read of the array is no longer what the array is
            self._forget_index()

    async def aupdate_chunk(self, nchunk: int, chunk: bytes) -> dict:
        """Write one compressed chunk asynchronously; see :meth:`update_chunk`.

        The same request, off the event loop, so a writer with many chunks to
        send can have several in flight.  The server serializes them at the
        far end regardless -- what overlaps is the round trip, which for a
        chunk-sized body is most of the cost.
        """
        url = self._chunk_url()
        if self._aclient is None:
            self._aclient = _httpx().AsyncClient(timeout=TIMEOUT)
        try:
            return await _axpost_bytes(
                self._aclient, url, chunk, params={"nchunk": nchunk}, auth_token=self.auth_token
            )
        finally:
            # Off the loop as well: `_forget_index` waits on the lock a source
            # being opened holds, and that open is a request of its own -- parking
            # the loop on it would stall every write still in flight, which is the
            # whole of what this method has over the blocking one
            await asyncio.to_thread(self._forget_index)

    def written_chunks(self) -> np.ndarray:
        """Which chunks of the remote array hold content; see
        :meth:`ByteRangeNDSource.written_chunks`.

        Read out of the frame's own offsets, which is where a fill records
        itself: no endpoint of its own, and nothing for the server to keep in
        step with the array.  Read afresh every time, since the point of asking
        is to see what other writers have done since -- which is a couple of
        range reads, the header first (a write moves the frame's length, and the
        offsets are found through it) and then the offsets it locates.

        Nothing else about the handle is disturbed: this asks what the *array*
        holds, not what this handle has done, so `meta` is left as it was and no
        `api/info` is spent on it.
        """
        with self._ranged(index_only=True) as source:
            # Through the source rather than around it: `_ranged` is what builds
            # one, and a source built here takes up any index a `Proxy` left in
            # `_pending_index` -- which is as old as the cache it came from.
            # Invalidating what has just been built is what makes this a read of
            # the frame rather than of whatever was already believed about it
            source.invalidate_index()
            return source.written_chunks()

    def _chunk_url(self) -> str:
        """Where a chunk of this array is read from, and written to."""
        return _server_url(self.urlbase, f"api/chunk/{self.path}")

    def _forget_index(self) -> None:
        """Drop what this handle read of a frame it has since written to."""
        # The metadata as well as the index: `meta` is read once when the array
        # is opened, so a handle that goes on to write would otherwise answer for
        # the array as it was before its own writes.  Read again when something
        # asks, rather than here, so a writer that never asks pays no request
        with self._meta_lock:
            self._meta_stale = True
            self._meta_epoch += 1
        with self._block_lock:
            # Under the lock a source being built right now is invalidated after
            # it is built, rather than missed entirely for holding a header this
            # write has already moved
            source = self._block_source
            if source is not _UNTRIED and source is not None:
                source.invalidate_index()
            # And an index that never reached a source: it came out of a `Proxy`
            # cache filled before this write, so a source built later must not
            # start from it
            self._pending_index = None

    def _reread_meta(self) -> None:
        """Read `api/info` again, and keep the answer if it is still an answer.

        The request is made outside the lock -- it is a round trip, and holding a
        lock across one would serialize every reader of this handle behind it --
        so a write of this handle's can land while it is in flight.  Such an
        answer describes the array as it was before that write: it is dropped,
        and the handle left marked stale, rather than stored as current and the
        write it predates forgotten along with it.
        """
        with self._meta_lock:
            seen = self._meta_epoch
        meta = info(self.path, self.urlbase, auth_token=self.auth_token)
        with self._meta_lock:
            if self._meta_epoch != seen:
                return
            self.meta = meta
            self._meta_stale = False

    def _refresh_meta(self) -> None:
        """Read `api/info` again, if this handle has written since it last did.

        Every property built on `meta` goes through this, so that what they say
        does not depend on which of them was read first.  It costs nothing to a
        handle that has not written -- which is every reader -- and one request
        to one that has.
        """
        if self._meta_stale:
            self._reread_meta()

    @property
    def _meta_complete(self) -> bool:
        """Whether `meta` describes an array that can no longer change.

        Every slot of a filled array is claimed, so every write to it is refused:
        what `api/info` says of one is what it will go on saying.  Anything else
        -- an array still being filled, or one that was never filled a chunk at a
        time and so says nothing either way -- can move under this handle at any
        moment, and asking again is the only way to find out.
        """
        vlmeta = self.meta.get("schunk", {}).get("vlmeta") or {}
        return vlmeta.get("fill_nonce") is not None and vlmeta.get("fill_state", "filling") != "filling"

    def refresh_stamp(self) -> None:
        """Look at the array again, so that :attr:`stamp` speaks for it now.

        `meta` is read when the handle is opened and, of itself, never again: a
        `stamp` off it names the array as this handle last saw it, which for a
        handle that has outlived someone else's writes is not the array.  A
        `Proxy` calls this before it reads the stamp it will judge its cache by,
        which is the one moment that difference decides anything.

        One `api/info`, and none at all for an array already known to be complete
        -- nothing can write to one of those, so nothing it reports can move.
        """
        if self._meta_stale or not self._meta_complete:
            self._reread_meta()

    # -- Block-granular reads.  A :ref:`Proxy` uses these to fetch the blocks a
    # slice touches instead of whole chunks, wherever that is the cheaper way
    # round; every one of them falls back to `get_chunk` when it is not.

    @property
    def stamp(self) -> str | None:
        """What names the exact remote bytes, for a :ref:`Proxy` to check a cache by.

        Geometry cannot tell a dataset that was replaced from the one a cache was
        filled from: a shape and a partitioning survive a rewrite, while every
        cached chunk -- and, in block mode, every offset they were fetched by --
        goes stale.  The server's own mtime does tell, and `api/info` carries
        it, so this costs no request of its own; the compressed size goes in with
        it, since a rewrite within the same clock tick is what an mtime cannot
        see.  What it names is the array as this handle last looked at it --
        :meth:`refresh_stamp` is how a caller that needs it to be the array *now*
        says so, and what a `Proxy` calls before judging a cache by it.

        Two questions, and they want different answers.  *Which array is this* is
        answered by the nonce a server writes into an array's vlmeta the first
        time a chunk is written to it: a size and an mtime can both be repeated
        by a different array that came to sit at the same path, and a cache
        served against one of those is stale without ever saying so.  *Has it
        changed since* is answered by the mtime and the compressed size, as
        before.

        The second question stops being worth asking once the array is complete.
        Every slot of a filled array is claimed, so every write to it is refused,
        and the bytes a cache holds cannot move again -- so a complete array is
        stamped by its nonce and its size, and a cache of it survives an mtime
        that churned for reasons of its own.

        An array still being filled is stamped freshly on every write, and has to
        be.  A cache built while a chunk was unwritten holds that chunk as the
        zeros an unwritten chunk reads as, and holds its offset as the run-length
        one it had; when a writer fills that slot, both are wrong, and nothing in
        the cache marks them apart from the chunks that are still good.

        None when the server reports no mtime and the array carries no nonce,
        which leaves the cache checked on its geometry alone, as every source
        without a stamp is.
        """
        self._refresh_meta()
        vlmeta = self.meta.get("schunk", {}).get("vlmeta") or {}
        nonce = vlmeta.get("fill_nonce")
        cbytes = self.meta.get("schunk", {}).get("cbytes", "")
        mtime = self.meta.get("mtime")
        if nonce is None:
            return None if mtime is None else f"{mtime}:{cbytes}"
        # `c` and `f` keep the two apart whatever the rest holds: a complete array
        # and a filling one must never stamp the same, or a cache of the second
        # is adopted against the first and serves the zeros it holds for the
        # chunks nobody had written yet
        if vlmeta.get("fill_state", "filling") != "filling":
            # Complete: nothing can write to it again, so nothing here need move
            return f"n{nonce}:c:{cbytes}"
        return f"n{nonce}:f:{cbytes}" if mtime is None else f"n{nonce}:f:{mtime}:{cbytes}"

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

    @property
    def serves_blocks(self) -> bool:
        """Whether blocks are worth asking this dataset for, as far as info can say.

        What `api/info` already carries, and no request of its own: a dataset the
        server *computes* reports an expression where a stored one reports a
        geometry, and only a stored one has a frame to read ranges of.  That is
        the whole question here.  Whether taking a particular chunk apart pays is
        a different one, and it is asked per fetch by
        :meth:`ByteRangeNDSource.wants_blocks`, which knows what the slice
        touches; this cannot, since it is read before any slice exists.

        It used to answer no as well for a frame whose chunks averaged under
        ``BLOCK_MIN_CBYTES``, which decided from one number, once, that no future
        slice of that dataset would ever be worth taking apart.  That forfeited
        the bytes the block path exists to save: measured against a Caterva2
        server, a dataset of 193 KB chunks reads a slab of 81 of them in 6.4 MB
        against 13.3 MB whole, and one of 650 KB chunks a slab of 36 in 6.4 MB
        against 23.3 MB -- 2.1x and 3.6x the traffic, on every such read, for the
        life of the dataset.  Where the link is what is scarce, and a server's
        uplink is shared by everyone reading through it, those are the bytes that
        decide how many readers it can hold.  The judgement was never wrong, only
        made too early and too widely: a point read of a small chunk really does
        cost more than it saves, and `wants_blocks` still refuses it.

        Read off `api/info` again where this handle has written since it last
        looked, which is the one case where the answer moves under it: a dataset
        may be laid out before it is stored.  That is one request to a handle that
        has just written, and none at all to a reader -- which is what the promise
        below needs.

        False is the whole answer; True is only that it is worth one request to
        find out, which :meth:`block_source` spends.  A :ref:`Proxy` reads this
        when it is built, to decide whether its cache records blocks or chunks,
        so it must cost nothing and must not depend on what has been fetched.

        A server that reports ``accept_ranges`` spares even that request where
        the answer is no: a dataset this server mounts from a peer reports the
        peer's geometry, being stored there, but is fetched from its owner and
        re-serialized here, so a range read of it is refused.  Nothing else in
        what `api/info` says can tell the two apart.  A server that reports
        nothing is an older one, and then this asks as it always did.
        """
        self._refresh_meta()  # `meta` is what carries it, so read it current
        if self.meta.get("accept_ranges") == "none":
            return False
        return self._reports_geometry

    @property
    def _reports_geometry(self) -> bool:
        """Whether `api/info` describes a stored dataset rather than a computed one.

        Necessary for reading the frame at all, where :attr:`serves_blocks` is
        that plus a judgement about whether taking its chunks apart would pay.
        The frame's own index is worth reading either way: it is a range read or
        two, and it is what says where the chunks are and which were written.
        """
        self._refresh_meta()
        return all(key in self.meta for key in ("chunks", "blocks", "schunk"))

    def block_source(self) -> C2NDSource | None:
        """The frame reader behind the block methods, or None if there is none.

        Built on the first request for it and never rebuilt.  The fallback has to
        be permanent: a server that streams this dataset answers a range
        request with the whole body, so retrying would pay a full download to
        rediscover the same answer.

        Every stored frame says yes: whether a given chunk of it is worth taking
        apart is decided per fetch, by `wants_blocks`, and not here.  So this and
        :meth:`_index_source` now come to the same answer, and both remain because
        they ask for different reasons -- one for the blocks of a chunk, one for
        the offsets that say where the chunks are.  Neither remembers a no, since
        a dataset laid out empty becomes a stored one as it is filled.
        """
        return self._source() if self.serves_blocks else None

    def _index_source(self) -> C2NDSource | None:
        """The same reader, built for any stored frame however small its chunks.

        Reading the frame's index is not the same question as reading blocks of
        its chunks, though a stored frame now answers yes to both: the offsets
        say which chunks hold anything, which is worth knowing whatever is done
        with them.  Whatever is built here is the source the block path uses too
        -- there is only ever one.
        """
        return self._source() if self._reports_geometry else None

    def _source(self) -> C2NDSource | None:
        """The one source, built once, whichever question asked for it first."""
        if self._block_source is _UNTRIED:
            with self._block_lock:
                if self._block_source is _UNTRIED:
                    self._block_source = self._open_block_source()
        # A failure that says nothing about the dataset leaves it _UNTRIED, so the
        # next fetch asks again; this one keeps to whole chunks either way
        return None if self._block_source is _UNTRIED else self._block_source

    def _open_block_source(self):
        """Decide, at whatever cost it takes, whether this dataset serves ranges.

        None for a dataset that does not serve ranges, which is an answer for
        good; `_UNTRIED` for a server that could not say, which is not.
        """
        httpx = _httpx()
        # A dataset the server computes has no frame to read at all, and
        # `api/info` says so for free.  Whether its chunks are worth taking apart
        # is a separate judgement, made by whoever asks -- see `block_source`
        if not self._reports_geometry:
            return None
        # Whether a dataset that reports a geometry is *served* from a file is
        # something only the answer to a range request can say: an HDF5 leaf or a
        # `.b2z` member reports one and is streamed all the same
        try:
            source = C2NDSource(self, max_concurrency=REMOTE_MAX_CONCURRENCY)
            source._adopt_index(self._pending_index)
            return source
        except NotRanged as exc:
            # PartsMissing among them, which carries no status and so is not
            # transient: an answer that cannot be taken apart is one to stop
            # asking, and whole chunks read the dataset either way
            return _UNTRIED if exc.transient else None
        except httpx.HTTPStatusError as exc:
            # A busy or broken server said nothing about how this is served
            return _UNTRIED if _is_transient(exc.response.status_code) else None
        except httpx.TransportError:
            # Nothing was downloaded to find this out, so asking again is cheap
            return _UNTRIED
        except (
            ValueError,
            NotImplementedError,
            RuntimeError,
            KeyError,
            IndexError,
            struct.error,
            httpx.HTTPError,
        ):
            # Not ranged, not a contiguous frame, not an NDArray, answered with
            # something unreadable, or described by an `api/info` without the
            # fields these read: whole chunks work for all of those
            return None

    def _adopt_index(self, state) -> None:
        """Keep an index a `Proxy` read out of its cache until there is a source.

        Handing it straight to :meth:`block_source` would build the source to
        receive it, and building it reads the frame's header -- a request, at the
        very moment of a run that may go on to fetch nothing at all.  So it waits
        here, and `_open_block_source` passes it on to the source it builds.
        """
        self._pending_index = state

    def _index_state(self, keep=()) -> dict | None:
        """What a `Proxy` should keep of what was read; see :ref:`ByteRangeNDSource`."""
        source = self._block_source
        if source is _UNTRIED or source is None:
            # No source was ever built, so nothing was read through one: hand back
            # whatever came out of the cache, rather than dropping it
            return self._pending_index
        return source._index_state(keep)

    def wants_blocks(self, nchunk: int, nwanted: int, wave=None) -> bool:
        """Whether fetching *nwanted* blocks of a chunk beats fetching all of it."""
        source = self.block_source()
        return source is not None and source.wants_blocks(nchunk, nwanted, wave)

    @property
    def max_ranges(self) -> int:
        """How many ranges one request to this server may carry."""
        source = self.block_source()
        return 1 if source is None else source.max_ranges

    def chunk_layout(self, nchunk: int):
        """Where the blocks of a chunk are; see :meth:`ByteRangeNDSource.chunk_layout`."""
        with self._ranged() as source:
            return source.chunk_layout(nchunk)

    def chunk_layouts(self, nchunks: Sequence[int]) -> list:
        """The same for several chunks; see :meth:`ByteRangeNDSource.chunk_layouts`."""
        with self._ranged() as source:
            return source.chunk_layouts(nchunks)

    def block_plan(self, nchunk: int, nblocks: Sequence[int]) -> list[tuple[int, int, tuple]]:
        """The range reads covering *nblocks*; see :meth:`ByteRangeNDSource.block_plan`."""
        with self._ranged() as source:
            return source.block_plan(nchunk, nblocks)

    def read_range(self, offset: int, size: int) -> bytes:
        """The bytes at [*offset*, *offset* + *size*) of the remote frame."""
        with self._ranged() as source:
            return source.read_range(offset, size)

    def read_ranges(self, spans: Sequence[tuple[int, int]]) -> list[bytes]:
        """The bytes of every span, in one request where the server allows it."""
        with self._ranged() as source:
            return source.read_ranges(spans)

    @contextmanager
    def _ranged(self, index_only: bool = False):
        """The block source, retired if it turns out to serve ranges no longer.

        The server can stop serving a dataset from a file between one fetch
        and the next -- replaced by a lazy expression, moved into a container it
        streams out of -- and the answer to a range request is where that shows.
        A refusal that says so for good puts the array back where it was before
        any of this: `get_chunk`, which works for every dataset there is.  One
        that says the server was busy leaves the source alone, since the next
        request may well be answered.

        The exception is raised either way: a `Proxy` catches it and fetches the
        chunks it was after whole, and a caller reading ranges directly is
        entitled to hear that the ranges are gone.
        """
        source = self._index_source() if index_only else self.block_source()
        if source is None:
            # A `NotRanged`, which is a `ValueError`: a fetch that finds the
            # source retired under it -- by another thread of the same wave --
            # falls back to whole chunks rather than failing, exactly as the
            # thread that retired it does
            raise NotRanged(f"{self.path} is not served in byte ranges by {self.urlbase}")
        try:
            yield source
        except PartsMissing:
            # One answer that did not carry its bytes, which the source itself
            # answers by asking for a range at a time; nothing about the dataset
            raise
        except NotRanged as exc:
            if not exc.transient:
                self._block_source = None
            raise

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
        self._refresh_meta()
        return self.meta["schunk"]["nbytes"]

    @property
    def cbytes(self) -> int:
        """The number of compressed bytes of the remote array"""
        self._refresh_meta()
        return self.meta["schunk"]["cbytes"]

    @property
    def cratio(self) -> float:
        """The compression ratio of the remote array"""
        self._refresh_meta()
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
        """The variable-length metadata of the remote array.

        Read again where this handle has written since it last looked: a fill
        records itself here, so a writer asking what it just did would otherwise
        be told what was true before it started.
        """
        self._refresh_meta()
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
        self._refresh_meta()
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
