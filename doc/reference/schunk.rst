.. _SChunk:

SChunk
======

The basic compressed data container (aka super-chunk). This class consists of a set of useful parameters and methods that allow not only to create compressed data, and decompress it, but also to manage the data in a more sophisticated way. For example, it is possible to append new data, update existing data, delete data, etc.

.. _MsgpackSerialization:

Metadata support
----------------

``SChunk.vlmeta`` uses the general Blosc2 msgpack extensions. This means
variable-length metadata can store not only ordinary msgpack-safe Python
values, but also the currently supported Blosc2 objects and references,
including:

- ``NDArray``, ``SChunk``, ``VLArray``, ``BatchArray``, ``EmbedStore``
- ``Ref``
- ``C2Array``
- ``LazyExpr``
- ``LazyUDF`` backed by ``@blosc2.dsl_kernel``

Both single-key access (``schunk.vlmeta["name"]``) and bulk access
(``schunk.vlmeta[:]``) use this serializer.

Lazy expressions and supported lazy UDFs still require durable operand
references only; purely in-memory operands are intentionally rejected.

.. currentmodule:: blosc2

.. _SChunkAttributes:

.. autoclass:: SChunk
    :members:
    :exclude-members: get_cparams, get_dparams, get_lazychunk, set_slice, update_cparams, update_dparams, c_schunk
    :member-order: groupwise

    :Special Methods:

    .. autosummary::

        __init__
        __len__
        __getitem__
        __setitem__

    Constructor
    -----------
    .. automethod:: __init__

    Utility Methods
    ---------------
    .. automethod:: __len__
    .. automethod:: __getitem__
    .. automethod:: __setitem__

Constructors
------------
.. autofunction:: schunk_from_cframe
