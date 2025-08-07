.. _Tree:

EmbedStore
====

A dictionary-like container for storing NumPy, Blosc2 (NDArray, either in memory or in external files) and remote (C2Array) arrays as nodes.

For nodes that are stored externally or remotely, only references to the arrays are stored, not the arrays themselves. This allows for efficient storage and retrieval of large datasets.

.. currentmodule:: blosc2

.. autoclass:: EmbedStore
    :members:
    :member-order: groupwise

    :Special Methods:

    .. autosummary::
        __init__
        __getitem__
        __setitem__
        __delitem__
        __contains__
        __len__
        __iter__

    Constructors
    ------------
    .. automethod:: __init__
    .. autofunction:: tree_from_cframe

    Dictionary Interface
    -------------------
    .. automethod:: __getitem__
    .. automethod:: __setitem__
    .. automethod:: __delitem__
    .. automethod:: __contains__
    .. automethod:: __len__
    .. automethod:: __iter__
    .. automethod:: keys
    .. automethod:: values
    .. automethod:: items

    Utility Methods
    ---------------
    .. automethod:: get
    .. automethod:: to_cframe
