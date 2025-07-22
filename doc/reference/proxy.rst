.. _Proxy:

Proxy
=====

Class that implements a proxy (with cache support) of a Python-Blosc2 container.

This can be used to cache chunks of regular data container which follows the
:ref:`ProxySource` or :ref:`ProxyNDSource` interfaces.

.. currentmodule:: blosc2
.. autoclass:: Proxy
    :members:
    :exclude-members: all, any, max, mean, min, prod, std, sum, var
    :member-order: groupwise

    :Special Methods:

    .. autosummary::
        __init__
        __getitem__

    Constructor
    -----------
    .. automethod:: __init__

    Utility Methods
    ---------------
    .. automethod:: __getitem__
