.. _ProxySource:

ProxySource
===========

Base interface for all supported sources in :ref:`Proxy` and are not NDim objects.
For example, a file, a memory buffer, a network resource, etc.  For n-dimemsional
ones, see :ref:`ProxyNDSource`.

.. currentmodule:: blosc2.ProxySource

Methods
-------

.. autosummary::
    :toctree: autofiles/proxysource
    :nosignatures:

    get_chunk
    aget_chunk

Attributes
----------

.. autosummary::
    :toctree: autofiles/proxysource
    :nosignatures:

    nbytes
    chunksize
    typesize
