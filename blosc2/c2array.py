#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

from __future__ import annotations

from collections.abc import Sequence
from collections import namedtuple

import numpy as np
import blosc2
import pathlib

import httpx


def _xget(url, params=None, headers=None, timeout=5, auth_cookie=None):
    if auth_cookie:
        headers = headers.copy() if headers else {}
        headers['Cookie'] = auth_cookie
    response = httpx.get(url, params=params, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response


def get(url, params=None, headers=None, timeout=5, model=None,
        auth_cookie=None):
    response = _xget(url, params, headers, timeout, auth_cookie)
    json = response.json()
    return json if model is None else model(**json)


def fetch_data(path, host, params, auth_cookie=None):
    params['prefer_schunk'] = True
    response = _xget(f'http://{host}/api/fetch/{path}', params=params,
                     auth_cookie=auth_cookie)
    data = response.content
    try:
        data = blosc2.decompress2(data)
    except (ValueError, RuntimeError):
        data = blosc2.ndarray_from_cframe(data)
        data = data[:] if data.ndim == 1 else data[()]
    return data


def slice_to_string(slice_):
    if slice_ is None or slice_ == () or slice_ == slice(None):
        return ''
    slice_parts = []
    if not isinstance(slice_, tuple):
        slice_ = (slice_,)
    for index in slice_:
        if isinstance(index, int):
            slice_parts.append(str(index))
        elif isinstance(index, slice):
            start = index.start or ''
            stop = index.stop or ''
            if index.step not in (1, None):
                raise IndexError('Only step=1 is supported')
            # step = index.step or ''
            slice_parts.append(f"{start}:{stop}")
    return ", ".join(slice_parts)


class C2Array:
    def __init__(self, name, root, host, auth_cookie=None):
        self.root = root
        self.name = name
        self.host = host
        self.path = pathlib.Path(f'{self.root}/{self.name}')
        self.auth_cookie = auth_cookie
        self.meta = get(f'http://{host}/api/info/{self.path}',
                                  auth_cookie=self.auth_cookie)

    def __getitem__(self, slice_: int | slice | Sequence[slice]) -> np.ndarray:
        data = self.fetch(slice_=slice_)
        return data

    def fetch(self, slice_=None, prefer_schunk=True):
        """
        Fetch a slice of a dataset.  Can specify transport serialization.

        Similar to `__getitem__()` but this one lets specify whether to prefer using Blosc2
        schunk serialization or pickle during data transport between the subscriber and the
        client. See below.

        Parameters
        ----------
        slice_ : int, slice, tuple of ints and slices, or None
            The slice to fetch.
        prefer_schunk : bool
            Whether to prefer using Blosc2 schunk serialization during data transport.
            If False, pickle will always be used instead. Default is True, so Blosc2
            serialization will be used if Blosc2 is installed (and data payload is large
            enough).

        Returns
        -------
        numpy.ndarray
            The slice of the dataset.
        """
        slice_ = slice_to_string(slice_)
        data = fetch_data(self.path, self.host,
                                    {'slice_': slice_, 'prefer_schunk': prefer_schunk},
                                    auth_cookie=self.auth_cookie)
        return data

    @property
    def shape(self):
        return self.meta['shape']

    @property
    def chunks(self):
        return self.meta['chunks']

    @property
    def blocks(self):
        return self.meta['blocks']

    @property
    def dtype(self):
        return self.meta['dtype']

    @property
    def nchunks(self):
        return self.meta['schunk']['nchunks']

    def __neg__(self):
        return blosc2.LazyExpr(new_op=(0, "-", self))

    def __and__(self, value: int | float | blosc2.NDArray | C2Array, /):
        blosc2.ndarray._check_allowed_dtypes(value, "numeric", "__and__")
        return blosc2.LazyExpr(new_op=(self, "&", value))

    def __add__(self, value: int | float | blosc2.NDArray | C2Array, /):
        blosc2.ndarray._check_allowed_dtypes(value, "numeric", "__add__")
        return blosc2.LazyExpr(new_op=(self, "+", value))

    def __iadd__(self, value: int | float | blosc2.NDArray | C2Array, /):
        blosc2.ndarray._check_allowed_dtypes(value, "numeric", "__iadd__")
        return blosc2.LazyExpr(new_op=(self, "+", value))

    def __radd__(self, value: int | float | blosc2.NDArray | C2Array, /):
        blosc2.ndarray._check_allowed_dtypes(value, "numeric", "__radd__")
        return blosc2.LazyExpr(new_op=(value, "+", self))

    def __sub__(self, value: int | float | blosc2.NDArray | C2Array, /):
        blosc2.ndarray._check_allowed_dtypes(value, "numeric", "__sub__")
        return blosc2.LazyExpr(new_op=(self, "-", value))

    def __isub__(self, value: int | float | blosc2.NDArray | C2Array, /):
        blosc2.ndarray._check_allowed_dtypes(value, "numeric", "__isub__")
        return blosc2.LazyExpr(new_op=(self, "-", value))

    def __rsub__(self, value: int | float | blosc2.NDArray | C2Array, /):
        blosc2.ndarray._check_allowed_dtypes(value, "numeric", "__rsub__")
        return blosc2.LazyExpr(new_op=(value, "-", self))

    def __mul__(self, value: int | float | blosc2.NDArray | C2Array, /):
        blosc2.ndarray._check_allowed_dtypes(value, "numeric", "__mul__")
        return blosc2.LazyExpr(new_op=(self, "*", value))

    def __imul__(self, value: int | float | blosc2.NDArray | C2Array, /):
        blosc2.ndarray._check_allowed_dtypes(value, "numeric", "__imul__")
        return blosc2.LazyExpr(new_op=(self, "*", value))

    def __rmul__(self, value: int | float | blosc2.NDArray | C2Array, /):
        blosc2.ndarray._check_allowed_dtypes(value, "numeric", "__rmul__")
        return blosc2.LazyExpr(new_op=(value, "*", self))

    def __truediv__(self, value: int | float | blosc2.NDArray | C2Array, /):
        blosc2.ndarray._check_allowed_dtypes(value, "numeric", "__truediv__")
        return blosc2.LazyExpr(new_op=(self, "/", value))

    def __itruediv__(self, value: int | float | blosc2.NDArray | C2Array, /):
        blosc2.ndarray._check_allowed_dtypes(value, "numeric", "__itruediv__")
        return blosc2.LazyExpr(new_op=(self, "/", value))

    def __rtruediv__(self, value: int | float | blosc2.NDArray | C2Array, /):
        blosc2.ndarray._check_allowed_dtypes(value, "numeric", "__rtruediv__")
        return blosc2.LazyExpr(new_op=(value, "/", self))

    def __lt__(self, value: int | float | blosc2.NDArray | C2Array, /):
        blosc2.ndarray._check_allowed_dtypes(value, "numeric", "__lt__")
        return blosc2.LazyExpr(new_op=(self, "<", value))

    def __le__(self, value: int | float | blosc2.NDArray | C2Array, /):
        blosc2.ndarray._check_allowed_dtypes(value, "numeric", "__le__")
        return blosc2.LazyExpr(new_op=(self, "<=", value))

    def __gt__(self, value: int | float | blosc2.NDArray | C2Array, /):
        blosc2.ndarray._check_allowed_dtypes(value, "numeric", "__gt__")
        return blosc2.LazyExpr(new_op=(self, ">", value))

    def __ge__(self, value: int | float | blosc2.NDArray | C2Array, /):
        blosc2.ndarray._check_allowed_dtypes(value, "numeric", "__ge__")
        return blosc2.LazyExpr(new_op=(self, ">=", value))

    def __eq__(self, value: int | float | blosc2.NDArray | C2Array, /):
        blosc2.ndarray._check_allowed_dtypes(value, "all", "__eq__")
        if blosc2._disable_overloaded_equal:  # Check if this works properly
            return self is value
        return blosc2.LazyExpr(new_op=(self, "==", value))

    def __ne__(self, value: int | float | blosc2.NDArray | C2Array, /):
        blosc2.ndarray._check_allowed_dtypes(value, "all", "__ne__")
        return blosc2.LazyExpr(new_op=(self, "!=", value))

    def __pow__(self, value: int | float | blosc2.NDArray | C2Array, /):
        blosc2.ndarray._check_allowed_dtypes(value, "numeric", "__pow__")
        return blosc2.LazyExpr(new_op=(self, "**", value))

    def __ipow__(self, value: int | float | blosc2.NDArray | C2Array, /):
        blosc2.ndarray._check_allowed_dtypes(value, "numeric", "__ipow__")
        return blosc2.LazyExpr(new_op=(self, "**", value))

    def __rpow__(self, value: int | float | blosc2.NDArray | C2Array, /):
        blosc2.ndarray._check_allowed_dtypes(value, "numeric", "__rpow__")
        return blosc2.LazyExpr(new_op=(value, "**", self))
