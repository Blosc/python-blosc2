#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################
import pathlib

import numpy as np
import pytest

import blosc2
from blosc2.lazyexpr import ne_evaluate

pytestmark = pytest.mark.network

ROOT = "@public"
DIR = "expr/"


def get_arrays(shape, chunks_blocks):
    dtype = np.float64
    nelems = np.prod(shape)
    na1 = np.linspace(0, 10, nelems, dtype=dtype).reshape(shape)
    cleanup_paths = []
    urlpath = f"ds-0-10-linspace-{dtype.__name__}-{chunks_blocks}-a1-{shape}d.b2nd"
    path = pathlib.Path(f"{ROOT}/{DIR + urlpath}").as_posix()
    cleanup_paths.append(path)
    a1 = blosc2.C2Array(path)
    urlpath = f"ds-0-10-linspace-{dtype.__name__}-{chunks_blocks}-a2-{shape}d.b2nd"
    cleanup_paths.append(urlpath)
    path = pathlib.Path(f"{ROOT}/{DIR + urlpath}").as_posix()
    a2 = blosc2.C2Array(path)
    # Let other operands be local, on-disk NDArray copies
    urlpath = f"ds-0-10-linspace-{dtype.__name__}-{chunks_blocks}-a3-{shape}d.b2nd"
    cleanup_paths.append(urlpath)
    a3 = blosc2.asarray(a2, urlpath=urlpath, mode="w")
    urlpath = f"ds-0-10-linspace-{dtype.__name__}-{chunks_blocks}-a4-{shape}d.b2nd"
    cleanup_paths.append(urlpath)
    a4 = a3.copy(urlpath=urlpath, mode="w")
    assert isinstance(a1, blosc2.C2Array)
    assert isinstance(a2, blosc2.C2Array)
    assert isinstance(a3, blosc2.NDArray)
    assert isinstance(a4, blosc2.NDArray)

    p1 = blosc2.Proxy(a1, urlpath="p1.b2nd", mode="w")
    p3 = blosc2.Proxy(a3, urlpath="p3.b2nd", mode="w")
    cleanup_paths.extend(["p1.b2nd", "p3.b2nd"])

    return p1, a2, p3, a4, na1, np.copy(na1), np.copy(na1), np.copy(na1), cleanup_paths


@pytest.mark.parametrize(
    "chunks_blocks",
    [
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    ],
)
def test_expr_proxy_operands(chunks_blocks, cat2_context):
    shape = (60, 60)
    a1, a2, a3, a4, na1, na2, na3, na4, cleanup_paths = get_arrays(shape, chunks_blocks)

    # Slice
    sl = slice(10)
    expr = a1 + a2 + a3 + a4
    expr += 3
    nres = ne_evaluate("na1 + na2 + na3 + na4 + 3")
    res = expr.compute(item=sl)
    np.testing.assert_allclose(res[:], nres[sl])

    # Save
    urlpath = "expr_proxies.b2nd"
    expr.save(urlpath=urlpath, mode="w")
    del expr
    expr_opened = blosc2.open("expr_proxies.b2nd")
    assert isinstance(expr_opened, blosc2.LazyExpr)

    # All
    res = expr_opened.compute()
    np.testing.assert_allclose(res[:], nres)

    # Cleanup
    blosc2.remove_urlpath(urlpath)
    for path in cleanup_paths:
        blosc2.remove_urlpath(path)
