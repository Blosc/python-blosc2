.. _DictStore:

DictStore
=========

A directory-based storage container for compressed data using Blosc2.

Manages a directory-based (``.b2d``) structure of NDArrays and SChunks, with an embed store for in-memory data. It also supports creating and reading ``.b2z`` files, which are zip archives that mirror the directory structure.

.. currentmodule:: blosc2

.. autoclass:: DictStore
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
        __enter__
        __exit__

    Constructors
    ------------
    .. automethod:: __init__

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

    Context Manager
    ---------------
    .. automethod:: __enter__
    .. automethod:: __exit__

    Utility Methods
    ---------------
    .. automethod:: get
    .. automethod:: to_b2z
    .. automethod:: close

    Properties
    ----------
    .. autoattribute:: estore
