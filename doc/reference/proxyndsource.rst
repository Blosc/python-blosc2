.. _ProxyNDSource:

ProxyNDSource
=============

Interface for NDim sources in :ref:`Proxy`.  For example, a NDArray, a HDF5 dataset, etc.
For a simpler source, see :ref:`ProxySource`.

.. currentmodule:: blosc2.ProxyNDSource

Methods
-------

.. autosummary::
    :toctree: autofiles/proxyndsource
    :nosignatures:

    get_chunk
    aget_chunk

Attributes
----------

.. autosummary::
    :toctree: autofiles/proxyndsource
    :nosignatures:

    shape
    dtype
    chunks
    blocks
