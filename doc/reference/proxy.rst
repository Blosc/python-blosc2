.. _Proxy:

Proxy
=====

Class that implements a proxy (with cache support) of a Python-Blosc2 container.

This can be used to cache chunks of regular data container which follows the
:ref:`ProxySource` or :ref:`ProxyNDSource` interfaces.

.. currentmodule:: blosc2.Proxy

Methods
-------

.. autosummary::
    :toctree: autofiles/proxy
    :nosignatures:

    __init__
    __getitem__
    fetch
    afetch

Attributes
----------

.. autosummary::
    :toctree: autofiles/proxy

    shape
    dtype
    cparams
    info
    fields
    vlmeta
