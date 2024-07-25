.. _ProxySChunk:

ProxySChunk
===========

Class that implements a proxy (with cache support) of a Python-Blosc2 container.

This can be used to cache chunks of
a regular data container which follows the :ref:`ProxySource` interface in an urlpath.

.. currentmodule:: blosc2.ProxySChunk

Methods
-------

.. autosummary::
    :toctree: autofiles/proxyschunk
    :nosignatures:

    __init__
    __getitem__
    fetch
    afetch

Attributes
----------

.. autosummary::
    :toctree: autofiles/proxyschunk

    shape
    dtype
