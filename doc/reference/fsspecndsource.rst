.. _FsspecNDSource:

FsspecNDSource
==============

A :ref:`ByteRangeNDSource` that serves the chunks of a Blosc2 frame living
behind an fsspec URL, reading each one with a range request instead of
transferring the whole container.  Everything about the frame format, block
granularity included, lives in the base class; this adds the fsspec transport.
For other sources, see :ref:`ProxyNDSource` and :ref:`ProxySource`.

``examples/ndarray/rw-fsspec.py`` is a runnable walkthrough of this and the
other two ways to read an fsspec URL, and of writing one back.
``examples/ndarray/concurrent-fsspec.py`` measures ``max_concurrency`` against a
filesystem with a simulated round trip, since no protocol that runs offline has
latency for the thread pool to hide.

.. currentmodule:: blosc2

.. autoclass:: FsspecNDSource
    :members:
    :exclude-members: all, any, max, mean, min, prod, std, sum, var
    :member-order: groupwise
