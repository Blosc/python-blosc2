.. _ByteRangeNDSource:

ByteRangeNDSource
=================

A :ref:`ProxyNDSource` that serves the chunks -- and the single blocks -- of a
Blosc2 frame it can read byte ranges of, instead of transferring the whole
container.  It knows the frame format and nothing about where the frame lives:
subclasses supply ``read_range(offset, size)`` and nothing else.
:ref:`FsspecNDSource` reads through fsspec, and :ref:`C2Array` reads over HTTP
ranges from a Caterva2 server.  For other sources, see :ref:`ProxyNDSource`
and :ref:`ProxySource`.

.. currentmodule:: blosc2

.. autoclass:: ByteRangeNDSource
    :members:
    :exclude-members: all, any, max, mean, min, prod, std, sum, var
    :member-order: groupwise

When a range read is refused
----------------------------

A transport that reads byte ranges may be answered with something other than the
bytes asked for: a server that now streams the dataset, a server that is too
busy to serve it, a body that cannot be taken apart. Those raise
``blosc2.proxy_source.NotRanged``, which a :ref:`Proxy` catches for itself --
whatever the fetch is still missing comes as whole chunks -- and which a caller
reading ranges directly can catch by name.

.. autoclass:: blosc2.proxy_source.NotRanged
    :members:

.. autoclass:: blosc2.proxy_source.PartsMissing
    :members:
