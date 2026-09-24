.. _RemoteObject:

RemoteObject
============

``RemoteObject`` is the public base class for :class:`blosc2.RemoteArray`,
:class:`blosc2.RemoteStore`, and :class:`blosc2.RemoteCTable`.  It is useful for
type checks; construct a concrete type or use :func:`blosc2.open` instead of
constructing ``RemoteObject`` directly.

.. code-block:: python

    import blosc2

    obj = blosc2.open("https://host/data.b2z")
    assert isinstance(obj, blosc2.RemoteObject)

Remote objects expose a credential-free ``source``, read-only ``attrs``, source
``traffic``, cache policy and accounting properties, an export-default
``mutable`` flag, and ``close()`` with context-manager support.  Cache accounting
has concrete-type scope: arrays report their own retained payload, while stores
and tables report their shared owner's retained payload.

``RemoteObject`` is not a factory, storage backend, serialization format, or
remote-write API.  Data-specific operations remain on the concrete classes.
Arrays and tables provide ``materialize()``; stores provide recursive
``materialize()`` to expand their contents and mounted stores into a local tree.

See :doc:`Working with Remote Data <../guides/remote_objects>` for the shared
cache, traffic, reference-saving, and lifetime behavior.

.. autoclass:: blosc2.RemoteObject
    :members:
