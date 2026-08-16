#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Writing and reading NDArrays through fsspec URLs.
#
# Needs the fsspec extra: pip install "blosc2[fsspec]"
#
# This uses memory:// so it runs anywhere, with no network and no credentials.
# Every URL below can be an s3://, gs://, abfs://... one instead, once the
# driver for that protocol is installed (s3fs, gcsfs, adlfs...):
#
#     urlpath = "s3://my-bucket/ds-2d.b2nd"

import tempfile

import numpy as np

import blosc2

urlpath = "memory://ds-2d.b2nd"

a = blosc2.arange(0, 10_000, dtype=np.int32, shape=(100, 100), chunks=(10, 100))

# Write.  The whole array goes up as a single object, replacing whatever was
# there: object stores have no partial write, so this is the only shape a
# remote write takes.  Two writers to the same key silently lose one.
a.save(urlpath)

# Read it back whole.  This is one GET plus a rebuild in memory, which is the
# right thing for an array you are going to use all of.
b = blosc2.open(urlpath)
print(f"read whole: {type(b).__name__} {b.shape} {b.dtype}")
np.testing.assert_array_equal(b[:], a[:])

with tempfile.TemporaryDirectory() as cachedir:
    # Read through a local cache.  The container is downloaded once into
    # cachedir and opened as an ordinary local path, so mmap, offsets and the
    # directory formats (.b2d stores, sparse frames) all work, and a later run
    # starts from the copy that is already there.  Cached copies are checked
    # against the remote on every open, so a replaced array is never served
    # from a stale cache.
    c = blosc2.open(urlpath, cache_storage=cachedir, mmap_mode="r")
    print(f"read cached: {c.shape} (mmapped from {cachedir})")
    np.testing.assert_array_equal(c[:], a[:])

    # Read lazily.  Nothing is transferred up front: the array stays where it
    # is and each slice fetches only the chunks it touches, one range request
    # each.  This is what you want for an array too big to download.
    d = blosc2.open(urlpath, lazy=True, cache_storage=cachedir)
    print(f"read lazy: {type(d).__name__} {d.shape} {d.dtype}")

    # Only the two chunks covering rows 15..25 are fetched here
    np.testing.assert_array_equal(d[15:25], a[15:25])

    # ...and they are cached, in cachedir, for the next run as well as this one
    np.testing.assert_array_equal(d[15:25], a[15:25])

    # A lazy handle is an ordinary operand, so expressions work on it, and
    # slicing one still fetches only the chunks that slice needs
    expr = d * 2
    print(f"lazy expression: {type(expr).__name__} -> {expr[15:17, 0]}")
    np.testing.assert_array_equal(expr[15:25], a[15:25] * 2)
