.. _DictStore:

DictStore
=========

A high‑level, dictionary‑like container to organize compressed arrays with Blosc2.

Overview
--------
DictStore lets you store and retrieve arrays by string keys (paths like ``"/dir/node"``), similar to a Python dict, while transparently handling efficient Blosc2 compression and persistence. It supports two on‑disk representations:

- ``.b2d``: a directory layout (B2DIR) where each external array is a separate file: ``.b2nd`` for NDArray and ``.b2f`` for SChunk; an embedded store file (``embed.b2e``) keeps small/in‑memory arrays.
- ``.b2z``: a single zip file (B2ZIP) that mirrors the directory structure above. You can zip up a ``.b2d`` layout or write directly and later reopen it for reading.

Supported values include ``blosc2.NDArray``, ``blosc2.SChunk`` and ``blosc2.C2Array`` (as well as ``numpy.ndarray``, which is converted to NDArray). Small arrays (below a configurable compression‑size threshold) and in‑memory objects are kept inside the embedded store; larger or explicitly external arrays live as regular ``.b2nd`` (NDArray) or ``.b2f`` (SChunk) files. ``C2Array`` objects are always stored in the embedded store. You can mix all types seamlessly and use the usual mapping methods (``__getitem__``, ``__setitem__``, ``keys()``, ``items()``...).

Quick example
-------------

.. code-block:: python

   import numpy as np
   import blosc2

   # Create a store backed by a zip file
   with blosc2.DictStore("my_dstore.b2z", mode="w") as dstore:
       dstore["/node1"] = np.array([1, 2, 3])  # small -> embedded store
       dstore["/node2"] = blosc2.ones(2)  # small -> embedded store
       arr_ext = blosc2.arange(3, urlpath="n3.b2nd", mode="w")
       dstore["/dir1/node3"] = arr_ext  # external file referenced

   # Reopen and read
   with blosc2.DictStore("my_dstore.b2z", mode="r") as dstore:
       print(sorted(dstore.keys()))  # ['/dir1/node3', '/node1', '/node2']
       print(dstore["/node1"][:])  # [1 2 3]

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

    Public Members
    --------------
