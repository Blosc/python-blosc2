.. _MsgpackSerialization:

Msgpack Serialization
=====================

python-blosc2 uses msgpack as the default serializer for :class:`ObjectArray` and
for the default ``"msgpack"`` mode of :class:`BatchArray`.

Two MessagePack extension codes are reserved by python-blosc2:

- ``42``: Blosc2 objects serialized by value as CFrames
- ``43``: structured Blosc2 reference or recipe objects

CFrame-backed objects
---------------------

The following objects are serialized by value using
:meth:`to_cframe` / :func:`blosc2.from_cframe`:

- ``NDArray``
- ``SChunk``
- ``ObjectArray``
- ``BatchArray``
- ``EmbedStore``

Structured objects
------------------

Extension code ``43`` stores a msgpack-encoded mapping with a stable envelope:

.. code-block:: text

    {
        "kind": "...",
        "version": 1,
        ...
    }

Currently implemented structured kinds are:

- ``"ref"``
- ``"c2array"``
- ``"urlpath"``
- ``"dictstore_key"``
- ``"lazyexpr"``
- ``"lazyudf"``

The ``"urlpath"``, ``"dictstore_key"``, and ``"c2array"`` reference forms map
directly onto the public :class:`blosc2.Ref` type.

``C2Array``
-----------

Remote arrays are serialized as lightweight references with:

- ``path``
- ``urlbase``

Authentication data is intentionally not serialized.

Persistent local operands
-------------------------

Persistent local Blosc2 operands can be serialized as:

.. code-block:: python

    {"kind": "urlpath", "version": 1, "urlpath": "..."}

and are reopened with :func:`blosc2.open`.

DictStore members
-----------------

Operands coming from :class:`DictStore` external leaves preserve store/member
identity via:

.. code-block:: python

    {"kind": "dictstore_key", "version": 1, "urlpath": "...", "key": "/a"}

This is used for members in both ``.b2d`` and ``.b2z`` stores, where the store
path alone is not enough to identify a specific member.

``LazyExpr``
------------

Expression-based lazy arrays are serialized as a recipe plus operand
references, following the same reference-preserving model as
:meth:`blosc2.LazyExpr.save`.

Only durable reference-style operands are supported:

- persistent local Blosc2 operands reopenable from ``urlpath``
- remote ``C2Array`` operands
- ``DictStore`` members reopenable from ``(.b2d|.b2z, key)``

Purely in-memory operands are intentionally rejected. This keeps msgpack
serialization of ``LazyExpr`` reference-preserving rather than value-copying.

``LazyUDF``
-----------

Currently, msgpack support for ``LazyUDF`` is limited to instances backed by a
``@blosc2.dsl_kernel`` function.

These are serialized as:

- ``function_kind="dsl"``
- ``dsl_version`` for the DSL grammar/semantics
- the UDF source code
- the preserved DSL source
- output ``dtype`` and ``shape``
- durable operand references
- execution kwargs needed to reconstruct the lazy object

Supported operands are the same durable reference-style operands used for
``LazyExpr``:

- persistent local Blosc2 operands reopenable from ``urlpath``
- remote ``C2Array`` operands
- ``DictStore`` members reopenable from ``(.b2d|.b2z, key)``

Plain Python ``LazyUDF`` callables are intentionally not serialized by
msgpack yet.
