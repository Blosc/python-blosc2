.. _VLArray:

VLArray
=======

Overview
--------
VLArray is a variable-length array container backed by a single Blosc2 ``SChunk``.

Each entry is stored as one compressed chunk:

- entries can be any serializable Python object
- items are serialized with msgpack before compression
- Blosc2 containers (:class:`NDArray`, :class:`SChunk`, :class:`VLArray`,
  :class:`BatchArray`, :class:`EmbedStore`) are serialized transparently
  via :meth:`to_cframe` / :func:`blosc2.from_cframe`
- structured Blosc2 reference objects (:class:`C2Array`, :class:`LazyExpr`,
  and :class:`LazyUDF` backed by :func:`blosc2.dsl_kernel`) are also supported

VLArray is a good fit when you need:

- a persistent, compressed list of arbitrary Python objects
- per-item random access and mutation
- compact summary information via ``.info``

Quick example
-------------

.. code-block:: python

    import blosc2

    vl = blosc2.VLArray(urlpath="example.b2z", mode="w", contiguous=True)
    vl.append({"x": 1, "y": 2})
    vl.append([3, 4, 5])
    vl.append("hello")

    print(vl[0])  # {'x': 1, 'y': 2}
    print(vl[1])  # [3, 4, 5]
    print(len(vl))  # 3

    reopened = blosc2.open("example.b2z", mode="r")
    print(type(reopened).__name__)
    print(reopened.info)

.. currentmodule:: blosc2

.. autoclass:: VLArray

    Constructors
    ------------
    .. automethod:: __init__

    Item Interface
    --------------
    .. automethod:: __getitem__
    .. automethod:: __setitem__
    .. automethod:: __delitem__
    .. automethod:: __len__
    .. automethod:: __iter__

    Mutation
    --------
    .. automethod:: append
    .. automethod:: extend
    .. automethod:: insert
    .. automethod:: delete
    .. automethod:: pop
    .. automethod:: clear
    .. automethod:: copy

    Context Manager
    ---------------
    .. automethod:: __enter__
    .. automethod:: __exit__

    Public Members
    --------------
    .. automethod:: to_cframe

Constructors
------------
.. autofunction:: blosc2.vlarray_from_cframe
