.. _SimpleProxy:

SimpleProxy
===========

Simple proxy for a NumPy array (or similar) that can be used with the Blosc2 compute engine.

This only supports the __getitem__ method. No caching is performed.

.. currentmodule:: blosc2.SimpleProxy

Methods
-------

.. autosummary::
    :toctree: autofiles/simpleproxy
    :nosignatures:

    __init__
    __getitem__

Attributes
----------

.. autosummary::
    :toctree: autofiles/simpleproxy

    shape
    dtype
    src
