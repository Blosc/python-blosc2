.. _SimpleProxy:

SimpleProxy
===========

Simple proxy for a NumPy array (or similar) that can be used with the Blosc2 compute engine.

This only supports the __getitem__ method. No caching is performed.

.. currentmodule:: blosc2

.. autoclass:: SimpleProxy
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
