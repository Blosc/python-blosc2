.. _ProxySChunk:

ProxySChunk
===========

Class that implements a proxy (with cache support) of a Python-Blosc2 container.

This can be used to cache chunks of
a regular SChunk, NDArray or C2Array in an urlpath.

.. currentmodule:: blosc2.ProxySChunk

Methods
-------

.. autosummary::
    :toctree: autofiles/proxyschunk
    :nosignatures:

    __init__
    __getitem__
    eval

Attributes
----------

.. autosummary::
    :toctree: autofiles/proxyschunk

    shape
    dtype
