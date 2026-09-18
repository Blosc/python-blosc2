.. _RemoteCTable:

RemoteCTable
============

``RemoteCTable`` is a read-only :class:`blosc2.CTable` backed by a remote B2Z
archive. Fixed-width, shaped, nullable, and UTF-8 columns are fetched on demand.
Standalone tables can be opened directly; tables inside a hierarchy can be
selected with ``dataset=`` or through :class:`blosc2.RemoteStore`.

Saving and materializing have different meanings:

.. code-block:: python

    with blosc2.RemoteCTable(url) as table:
        table["temperature"][:100]  # warm part of one column
        table.save("reference.b2z")
        table.save("cold-reference.b2z", include_cache=False)
        local = table.materialize(urlpath="complete.b2z")

``save()`` writes the source descriptor, table metadata, and by default only
payload already retained in the cache. Missing data is still read from the
original source after reopening. ``materialize()`` returns an independent local
table and reads all data needed for it. ``copy()``, ``to_b2z()``, and ``to_b2d()``
remain local materialization operations inherited from :class:`blosc2.CTable`.

Table cache bytes, limits, and traffic are scoped to the shared remote owner and
may include sibling leaves. A table selected from a store owns an independent
handle, but its columns and views remain borrowed from that table. Refresh a
nested table through its root store; a standalone table can call ``refresh()``.

.. autoclass:: blosc2.RemoteCTable
    :members:
