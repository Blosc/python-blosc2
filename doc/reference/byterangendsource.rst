.. _ByteRangeNDSource:

ByteRangeNDSource
=================

A :ref:`ProxyNDSource` that serves the chunks -- and the single blocks -- of a
Blosc2 frame it can read byte ranges of, instead of transferring the whole
container.  It knows the frame format and nothing about where the frame lives:
subclasses supply ``read_range(offset, size)`` and nothing else.
:ref:`FsspecNDSource` reads through fsspec, and :ref:`C2Array` reads over HTTP
ranges from a Caterva2 subscriber.  For other sources, see :ref:`ProxyNDSource`
and :ref:`ProxySource`.

.. currentmodule:: blosc2

.. autoclass:: ByteRangeNDSource
    :members:
    :exclude-members: all, any, max, mean, min, prod, std, sum, var
    :member-order: groupwise
