.. _ListArray:

ListArray
=========

Overview
--------
ListArray is a row-oriented container for variable-length list cells.
It is the natural public container for list-valued :class:`blosc2.CTable`
columns, but it is also useful on its own whenever you want typed,
row-addressable list data.

Internally, ListArray uses one of two lower-level backends:

- :class:`blosc2.BatchArray` for append/scan-oriented workloads
- :class:`blosc2.VLArray` for simpler row-level replacement semantics

Quick example
-------------

.. code-block:: python

    import blosc2

    arr = blosc2.ListArray(
        item_spec=blosc2.string(max_length=16),
        nullable=True,
        storage="batch",
        urlpath="ingredients.b2b",
        mode="w",
    )
    arr.append(["salt", "sugar"])
    arr.append([])
    arr.append(None)

    print(arr[0])
    print(arr[1:])

    reopened = blosc2.open("ingredients.b2b", mode="r")
    print(type(reopened).__name__)

.. note::
   Returned Python lists are detached values. Mutating them locally does not
   write back to the container; reassign the whole cell instead.

.. currentmodule:: blosc2

.. autoclass:: ListArray

    Constructors
    ------------
    .. automethod:: __init__
    .. automethod:: from_arrow

    Row Interface
    -------------
    .. automethod:: __getitem__
    .. automethod:: __setitem__
    .. automethod:: __len__
    .. automethod:: __iter__

    Mutation
    --------
    .. automethod:: append
    .. automethod:: extend
    .. automethod:: flush
    .. automethod:: copy
    .. automethod:: close

    Context Manager
    ---------------
    .. automethod:: __enter__
    .. automethod:: __exit__

    Public Members
    --------------
    .. automethod:: to_arrow
    .. automethod:: to_cframe
