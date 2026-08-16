.. _FsspecNDSource:

FsspecNDSource
==============

A :ref:`ProxyNDSource` that serves the chunks of a Blosc2 frame living behind an
fsspec URL, reading each one with a range request instead of transferring the
whole container.  For other sources, see :ref:`ProxyNDSource` and
:ref:`ProxySource`.

.. currentmodule:: blosc2

.. autoclass:: FsspecNDSource
    :members:
    :exclude-members: all, any, max, mean, min, prod, std, sum, var
    :member-order: groupwise
