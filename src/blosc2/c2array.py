#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

import httpx
import numpy as np

import blosc2

_subscriber_data = {
    "urlbase": os.environ.get("BLOSC_C2URLBASE"),
    "auth_token": "",
}
"""Caterva2 subscriber data saved by context manager."""


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

    A parameter not specified or set to ``None`` inherits the value set by the
    previous context manager, defaulting to an environment variable (see
    below) if supported by that parameter.  Parameters set to the empty string
    are not to be used in requests (with no default either).

    If the subscriber requires authorization for requests, you may either
    provide `auth_token` (which you should have obtained previously from the
    subscriber), or both `username` and `password` to get that token by first
    logging in to the subscriber.  The token will be reused until explicitly
    reset or requested again in a latter context manager invocation.

    Please note that this manager is reentrant but not concurrency-safe.

    Parameters
    ----------
    urlbase : str | None
        A URL base that will be used when an individual C2Array instance has
        no subscriber URL base set.  Use the ``BLOSC_C2URLBASE`` environment
        variable if set as a last resort default.
    username : str | None
        A name to be used in credentials to login to the subscriber and get an
        authorization token from it.  Use the ``BLOSC_C2USERNAME`` environment
        variable if set as a last resort default.
    password : str | None
        A secret to be used in credentials to login to the subscriber and get
        an authorization token from it.  Use the ``BLOSC_C2PASSWORD``
        environment variable if set as a last resort default.
    auth_token : str | None
        A token that will be used when an individual C2Array instance has no
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
            raise ValueError("Either provide a username/password or an authorizaton token")
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


def _xget(url, params=None, headers=None, auth_token=None, timeout=15):
    auth_token = auth_token or _subscriber_data["auth_token"]
    if auth_token:
        headers = headers.copy() if headers else {}
        headers["Cookie"] = auth_token
    response = httpx.get(url, params=params, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response


def _xpost(url, json=None, auth_token=None, timeout=15):
    auth_token = auth_token or _subscriber_data["auth_token"]
    headers = {"Cookie": auth_token} if auth_token else None
    response = httpx.post(url, json=json, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _sub_url(urlbase, path):
    urlbase = urlbase or _subscriber_data["urlbase"]
    if not urlbase:
        raise RuntimeError("No default Caterva2 subscriber set")
    return f"{urlbase}{path}" if urlbase.endswith("/") else f"{urlbase}/{path}"


def login(username, password, urlbase):
    url = _sub_url(urlbase, "auth/jwt/login")
    creds = dict(username=username, password=password)
    resp = httpx.post(url, data=creds, timeout=15)
    resp.raise_for_status()
    return "=".join(list(resp.cookies.items())[0])


def info(path, urlbase, params=None, headers=None, model=None, auth_token=None):
    url = _sub_url(urlbase, f"api/info/{path}")
    response = _xget(url, params, headers, auth_token)
    json = response.json()
    return json if model is None else model(**json)


def subscribe(root, urlbase, auth_token):
    url = _sub_url(urlbase, f"api/subscribe/{root}")
    return _xpost(url, auth_token=auth_token)


def fetch_data(path, urlbase, params, auth_token=None):
    url = _sub_url(urlbase, f"api/fetch/{path}")
    response = _xget(url, params=params, auth_token=auth_token)
    data = response.content
    try:
        data = blosc2.ndarray_from_cframe(data)
        data = data[:] if data.ndim == 1 else data[()]
    except RuntimeError:
        data = blosc2.schunk_from_cframe(data)
        data = data[:]
    return data


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


class C2Array(blosc2.Operand):
    def __init__(self, path: str, /, urlbase: str = None, auth_token: str = None):
        """Create an instance of a remote NDArray.

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
        >>> urlbase = "https://demo.caterva2.net/"
        >>> path = "example/dir1/ds-3d.b2nd"
        >>> remote_array = blosc2.C2Array(path, urlbase=urlbase)
        >>> remote_array.shape
        (3, 4, 5)
        >>> remote_array.chunks
        (2, 3, 4)
        >>> remote_array.blocks
        (2, 2, 2)
        >>> remote_array.dtype
        float32
        """
        if path.startswith("/"):
            raise ValueError("The path should start with a root name, not a slash")
        self.path = path

        if urlbase and not urlbase.endswith("/"):
            urlbase += "/"
        self.urlbase = urlbase

        self.auth_token = auth_token

        # Try to 'open' the remote path
        try:
            self.meta = info(self.path, self.urlbase, auth_token=self.auth_token)
        except httpx.HTTPStatusError:
            # Subscribe to root and try again. It is less latency to subscribe directly
            # than to check for the subscription.
            root, _ = self.path.split("/", 1)
            subscribe(root, self.urlbase, self.auth_token)
            try:
                self.meta = info(self.path, self.urlbase, auth_token=self.auth_token)
            except httpx.HTTPStatusError as err:
                raise FileNotFoundError(f"Remote path not found: {path}.\nError was: {err}") from err

    def __getitem__(self, slice_: int | slice | Sequence[slice]) -> np.ndarray:
        """
        Get a slice of the array.

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
        >>> urlbase = "https://demo.caterva2.net/"
        >>> path = "example/dir1/ds-2d.b2nd"
        >>> remote_array = blosc2.C2Array(path, urlbase=urlbase)
        >>> data_slice = remote_array[3:5, 1:4]
        >>> data_slice.shape
        (2, 3)
        >>> data_slice[:]
        [[61 62 63]
        [81 82 83]]
        """
        slice_ = slice_to_string(slice_)
        return fetch_data(self.path, self.urlbase, {"slice_": slice_}, auth_token=self.auth_token)

    def get_chunk(self, nchunk: int) -> bytes:
        """
        Get the compressed unidimensional chunk of a :ref:`C2Array`.

        Parameters
        ----------
        nchunk: int
            The unidimensional chunk index.

        Returns
        -------
        out: bytes
            The requested compressed chunk.

        Examples
        --------
        >>> import numpy as np
        >>> import blosc2
        >>> urlbase = "https://demo.caterva2.net/"
        >>> path = "example/dir1/ds-3d.b2nd"
        >>> a = blosc2.C2Array(path, urlbase)
        >>>  # Get the compressed chunk from array 'a' for index 0
        >>> compressed_chunk = a.get_chunk(0)
        >>> f"Size of chunk {0} from a: {len(compressed_chunk)} bytes"
        Size of chunk 0 from a: 160 bytes
        >>> # Decompress the chunk and convert it to a NumPy array
        >>> decompressed_chunk = blosc2.decompress(compressed_chunk)
        >>> np.frombuffer(decompressed_chunk, dtype=a.dtype)
        [ 0.  1.  5.  6. 20. 21. 25. 26.  2.  3.  7.  8. 22. 23. 27. 28. 10. 11.
          0.  0. 30. 31.  0.  0. 12. 13.  0.  0. 32. 33.  0.  0.]
        """
        url = _sub_url(self.urlbase, f"api/chunk/{self.path}")
        params = {"nchunk": nchunk}
        response = _xget(url, params=params, auth_token=self.auth_token)
        return response.content

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


class URLPath:
    def __init__(self, path: str, /, urlbase: str = None, auth_token: str = None):
        """
        Create an instance of a remote data file (aka :ref:`C2Array <C2Array>`) urlpath.
        This is meant to be used in the :func:`blosc2.open` function.

        The parameters are the same as for the :meth:`C2Array.__init__`.

        """
        self.path = path
        self.urlbase = urlbase
        self.auth_token = auth_token
