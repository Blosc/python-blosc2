.. _BatchStore:

BatchStore
==========

Overview
--------
BatchStore is a batch-oriented container for variable-length Python items
backed by a single Blosc2 ``SChunk``.

Each batch is stored in one compressed chunk:

- batches contain one or more Python items
- each chunk may contain one or more internal variable-length blocks
- the store itself is indexed by batch
- item-wise traversal is available via :meth:`BatchStore.iter_items`

BatchStore is a good fit when data arrives naturally in batches and you want:

- efficient batch append/update operations
- persistent ``.b2b`` stores
- item-level reads inside a batch
- compact summary information about batches and internal blocks via ``.info``

Serializer support
------------------

BatchStore currently supports two serializers:

- ``"msgpack"``: the default and general-purpose choice for Python items
- ``"arrow"``: optional and requires ``pyarrow``; mainly useful when data is
  already Arrow-shaped before ingestion

For ``"msgpack"``, python-blosc2 supports both ordinary msgpack-safe Python
values and selected Blosc2 objects.

Blosc2 objects serialized by value via :meth:`to_cframe` /
:func:`blosc2.from_cframe`:

- ``NDArray``
- ``SChunk``
- ``VLArray``
- ``BatchStore``
- ``EmbedStore``

Structured reference-style Blosc2 objects currently supported:

- ``C2Array``
- ``LazyExpr``
- ``LazyUDF`` backed by ``@blosc2.dsl_kernel``

``LazyExpr`` values and supported ``LazyUDF`` values preserve reference
semantics and are serialized as a recipe plus durable operand references.
Supported operands are:

- persistent local Blosc2 operands reopenable from ``urlpath``
- remote ``C2Array`` operands
- ``DictStore`` members reopenable from ``(.b2d|.b2z, key)``

Purely in-memory operands are intentionally not supported. Plain Python
``LazyUDF`` callables are not serialized by msgpack.

Quick example
-------------

.. code-block:: python

    import blosc2

    store = blosc2.BatchStore(urlpath="example_batch_store.b2b", mode="w", contiguous=True)
    store.append([{"red": 1, "green": 2, "blue": 3}, {"red": 4, "green": 5, "blue": 6}])
    store.append([{"red": 7, "green": 8, "blue": 9}])

    print(store[0])  # first batch
    print(store[0][1])  # second item in first batch
    print(list(store.iter_items()))

    reopened = blosc2.open("example_batch_store.b2b", mode="r")
    print(type(reopened).__name__)
    print(reopened.info)

.. note::
   BatchStore is batch-oriented by design. ``store[i]`` returns a batch, not a
   single item. Use :meth:`BatchStore.iter_items` for flat item-wise traversal.

.. currentmodule:: blosc2

.. autoclass:: BatchStore

    Constructors
    ------------
    .. automethod:: __init__

    Batch Interface
    ---------------
    .. automethod:: __getitem__
    .. automethod:: __setitem__
    .. automethod:: __delitem__
    .. automethod:: __len__
    .. automethod:: __iter__
    .. automethod:: iter_items

    Mutation
    --------
    .. automethod:: append
    .. automethod:: extend
    .. automethod:: insert
    .. automethod:: pop
    .. automethod:: delete
    .. automethod:: clear
    .. automethod:: copy

    Context Manager
    ---------------
    .. automethod:: __enter__
    .. automethod:: __exit__

    Public Members
    --------------
    .. automethod:: to_cframe

.. autoclass:: Batch
