.. _ProxySource:

ProxySource
===========

Base interface for all supported sources in :ref:`ProxySChunk`.

In case the source is multidimensional, the attributes `shape`, `chunks`,
`blocks` and `dtype` are also required when creating the :ref:`ProxySChunk`.
In case the source is unidimensional, the attributes `chunksize`, `typesize`
and `nbytes` are required as well when creating the :ref:`ProxySChunk`.
These attributes do not need to be available when opening an already
existing :ref:`ProxySChunk`.

.. currentmodule:: blosc2.ProxySource

Methods
-------

.. autosummary::
    :toctree: autofiles/proxysource
    :nosignatures:

    get_chunk
