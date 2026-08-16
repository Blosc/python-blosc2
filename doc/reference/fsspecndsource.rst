.. _FsspecNDSource:

FsspecNDSource
==============

A :ref:`ProxyNDSource` that serves the chunks of a Blosc2 frame living behind an
fsspec URL, reading each one with a range request instead of transferring the
whole container.  For other sources, see :ref:`ProxyNDSource` and
:ref:`ProxySource`.

``examples/ndarray/rw-fsspec.py`` is a runnable walkthrough of this and the
other two ways to read an fsspec URL, and of writing one back.

.. currentmodule:: blosc2

.. autoclass:: FsspecNDSource
    :members:
    :exclude-members: all, any, max, mean, min, prod, std, sum, var
    :member-order: groupwise
