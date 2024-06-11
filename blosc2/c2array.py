#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
import os

import httpx
import numpy as np

import blosc2


C2SUB_DEFBASE_ENVVAR = 'CATERVA2_SUBSCRIBER_URL'
"""Environment variable with a default Caterva2 subscriber URL base."""

_subscriber_data = {'auth_cookie': None}
"""Caterva2 subscriber data saved by context manager."""


@contextmanager
def c2subscriber_auth_cookie(auth_cookie: str | None):
    """
    Context manager that adds `auth_cookie` to Caterva2 subscriber requests.

    Please note that this manager is reentrant but not concurrency-safe.

    Parameters
    ----------
    auth_cookie: str | None
        A cookie that will be used when an individual C2Array instance has no
        authorization cookie set.  Use ``None`` to disable the cookie set by a
        previous context manager.

    Yields
    ------
    out: None

    """
    global _subscriber_data
    try:
        old_sub_data = _subscriber_data
        new_sub_data = old_sub_data.copy()  # inherit old values
        new_sub_data['auth_cookie'] = auth_cookie
        _subscriber_data = new_sub_data
        yield
    finally:
        _subscriber_data = old_sub_data


def _xget(url, params=None, headers=None, auth_cookie=None, timeout=15):
    auth_cookie = auth_cookie or _subscriber_data['auth_cookie']
    if auth_cookie:
        headers = headers.copy() if headers else {}
        headers["Cookie"] = auth_cookie
    response = httpx.get(url, params=params, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response


def _xpost(url, json=None, auth_cookie=None, timeout=15):
    auth_cookie = auth_cookie or _subscriber_data['auth_cookie']
    headers = {'Cookie': auth_cookie} if auth_cookie else None
    response = httpx.post(url, json=json, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _sub_api_url(urlbase, apipath):
    try:
        urlbase = urlbase or os.environ[C2SUB_DEFBASE_ENVVAR]
    except KeyError as ke:
        raise RuntimeError("No default Caterva2 subscriber set") from ke
    return (f"{urlbase}api/{apipath}" if urlbase.endswith("/")
            else f"{urlbase}/api/{apipath}")


def info(path, urlbase, params=None, headers=None, model=None, auth_cookie=None):
    url = _sub_api_url(urlbase, f"info/{path}")
    response = _xget(url, params, headers, auth_cookie)
    json = response.json()
    return json if model is None else model(**json)


def subscribe(root, urlbase, auth_cookie):
    url = _sub_api_url(urlbase, f"subscribe/{root}")
    return _xpost(url, auth_cookie=auth_cookie)


def fetch_data(path, urlbase, params, auth_cookie=None):
    url = _sub_api_url(urlbase, f"fetch/{path}")
    response = _xget(url, params=params, auth_cookie=auth_cookie)
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
    def __init__(self, path, /, urlbase=None, auth_cookie=None):
        """Create an instance of a remote NDArray.

        Parameters
        ----------
        path: str
            The path to the remote NDArray file (root + file path) as
            a posix path.
        urlbase: str
            The base URL (slash-terminated) of the subscriber to query.
        auth_cookie: str
            An optional cookie to authorize requests via HTTP.

        Returns
        -------
        out: C2Array

        """
        if path.startswith('/'):
            raise ValueError("The path should start with a root name, not a slash")
        self.path = path

        if urlbase and not urlbase.endswith('/'):
            urlbase += '/'
        self.urlbase = urlbase

        self.auth_cookie = auth_cookie

        # Try to 'open' the remote path
        try:
            self.meta = info(self.path, self.urlbase,
                             auth_cookie=self.auth_cookie)
        except httpx.HTTPStatusError:
            # Subscribe to root and try again. It is less latency to subscribe directly
            # than to check for the subscription.
            root, _ = self.path.split('/', 1)
            subscribe(root, self.urlbase, self.auth_cookie)
            try:
                self.meta = info(self.path, self.urlbase,
                                 auth_cookie=self.auth_cookie)
            except httpx.HTTPStatusError as err:
                raise FileNotFoundError(f"Remote path not found: {path}.\n"
                                        f"Error was: {err}") from err

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
        """
        slice_ = slice_to_string(slice_)
        data = fetch_data(self.path, self.urlbase, {"slice_": slice_}, auth_cookie=self.auth_cookie)
        return data

    @property
    def shape(self):
        """The shape of the remote array"""
        return tuple(self.meta["shape"])

    @property
    def chunks(self):
        """The chunks of the remote array"""
        return tuple(self.meta["chunks"])

    @property
    def blocks(self):
        """The blocks of the remote array"""
        return tuple(self.meta["blocks"])

    @property
    def dtype(self):
        """The dtype of the remote array"""
        return np.dtype(self.meta["dtype"])

    @property
    def ext_shape(self):
        """The ext_shape of the remote array"""
        return tuple(self.meta["ext_shape"])


class URLPath:
    def __init__(self, path, /, urlbase=None, auth_cookie=None):
        """
        Create an instance of a remote data file (aka :ref:`C2Array <C2Array>`) urlpath.
        This is meant to be used in the :func:`blosc2.open` function.

        The parameters are the same as for the :meth:`C2Array.__init__`.

        """
        self.path = path
        self.urlbase = urlbase
        self.auth_cookie = auth_cookie
