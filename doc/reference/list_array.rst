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
- :class:`blosc2.ObjectArray` for simpler row-level replacement semantics

Quick example
-------------

.. code-block:: python

    import blosc2

    with blosc2.ListArray(
        item_spec=blosc2.string(max_length=16, nullable=True),
        nullable=True,
        storage="batch",
        urlpath="ingredients.b2b",
        mode="w",
    ) as arr:
        arr.append(["salt", None, "sugar"])
        arr.append([])
        arr.append(None)

        print(arr[0])
        print(arr[1:])

    reopened = blosc2.open("ingredients.b2b", mode="r")
    print(type(reopened).__name__)

.. note::
   Returned Python lists are detached values. Mutating them locally does not
   write back to the container; reassign the whole cell instead.

Batching
--------
Batch storage writes up to 2048 list cells per compressed chunk by default.
``batch_rows`` counts outer list cells, not child elements or bytes. A partial
final batch is persisted by :meth:`ListArray.flush`, :meth:`ListArray.close`, or
context-manager exit. Pass a smaller positive value to reduce remote overfetch,
usually at the cost of more requests and weaker compression.

``batch_rows=None`` opts into caller-managed boundaries: appends remain pending
until a flush, which writes every pending cell as one batch. This can consume
substantial memory and make a remote scalar read download a large batch. The
setting has no effect with ``storage="vl"``. Existing arrays retain their stored
batch boundaries when reopened.

Nested values and predicates
----------------------------
Nullability is independent at each level. The following schema accepts a null
outer list, null inner lists, and null integers inside an inner list::

    nested = blosc2.list(
        blosc2.list(blosc2.int32(nullable=True), nullable=True),
        nullable=True,
    )

Membership predicates inspect immediate children and do not flatten nesting::

    arr.contains([1, 2])
    arr.overlaps([[1, 2], [3]])

On a CTable column these methods return Boolean row predicates that compose with
other column expressions. An optional membership index accelerates flat lists of
scalar Boolean, numeric, string, or bytes values::

    table.create_index("tags", kind="membership")
    selected = table[table["tags"].overlaps(["python", "numpy"])]

Null outer lists and empty lists do not match. A nullable child matches an
explicit ``None``. Nested-list and struct membership uses the scan path; creating
a membership index for those child types is not supported.

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
    .. automethod:: contains
    .. automethod:: overlaps
