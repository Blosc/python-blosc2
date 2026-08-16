.. _FsspecNDSource:

FsspecNDSource
==============

A :ref:`ProxyNDSource` that serves the chunks of a Blosc2 frame living behind an
fsspec URL, reading each one with a range request instead of transferring the
whole container.  This is what ``blosc2.open(url, lazy=True)`` builds, with an
in-memory cache; use the class directly to give the fetched chunks a cache that
outlives the process, as ``mode="a"`` picks an existing one back up::

    src = blosc2.FsspecNDSource("s3://bucket/huge.b2nd")
    a = blosc2.Proxy(src, urlpath="huge-cache.b2nd", mode="a")
    a[1000:1010]   # fetched once, then served from huge-cache.b2nd for good

.. currentmodule:: blosc2

.. autoclass:: FsspecNDSource
    :members:
    :exclude-members: all, any, max, mean, min, prod, std, sum, var
    :member-order: groupwise
