.. _SChunk:

SChunk
======

The basic compressed data container (aka super-chunk). This class consists of a set of useful parameters and methods that allow not only to create compressed data, and decompress it, but also to manage the data in a more sophisticated way. For example, it is possible to append new data, update existing data, delete data, etc.

.. _MsgpackSerialization:

Metadata support
----------------

Use ``obj.attrs`` as the recommended interface for user-defined metadata::

    array = blosc2.zeros(10)
    array.attrs["units"] = "kelvin"
    print(array.attrs["units"])
    print(array.attrs[:])

``attrs`` delegates to ``vlmeta`` on ``NDArray``, ``SChunk``, ``ObjectArray``,
``BatchArray``, ``ListArray``, ``CTable``, ``TreeStore``, ``LazyArray``, ``Proxy``
and ``RemoteProxy``, and on ``ProxyNDSource`` implementations. It preserves
existing persistence, serialization and access rules; it does not introduce
another metadata store or filter operational entries from ordinary containers.
Some objects return a fresh mapping wrapper on each access.

``vlmeta`` remains supported with its existing behavior and is not deprecated.
Prefer ``attrs`` in new user-facing code. Constructor arguments named ``vlmeta``
retain their existing names.

``C2Array`` is the exception: ``attrs`` selects user attributes from the server,
whereas ``vlmeta`` retains raw protocol metadata. Older servers without an
``attrs`` field fall back to raw variable metadata. Neither property writes
changes to the server. ``RemoteProxy.attrs`` is read-only. See the
:doc:`remote array guide <../guides/remote_arrays>` for details.

``SChunk.attrs`` uses the general Blosc2 msgpack extensions. This means
variable-length metadata can store not only ordinary msgpack-safe Python
values, but also the currently supported Blosc2 objects and references,
including:

- ``NDArray``, ``SChunk``, ``ObjectArray``, ``BatchArray``, ``EmbedStore``
- ``Ref``
- ``C2Array``
- ``LazyExpr``
- ``LazyUDF`` backed by ``@blosc2.dsl_kernel``

Both single-key access (``schunk.attrs["name"]``) and bulk access
(``schunk.attrs[:]``) use this serializer.

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
