.. _RemoteCTable:

RemoteCTable
============

``RemoteCTable`` is a read-only :class:`blosc2.CTable` backed by a remote B2Z
archive. Fixed-width, shaped, nullable, UTF-8, batch-backed variable-length,
batch-backed list, struct/object, and dictionary columns are fetched on demand.
Standalone tables can be opened directly; tables inside a hierarchy can be
selected with ``dataset=`` or through :class:`blosc2.RemoteStore`.

Batch-backed reads transfer and decode whole compressed batches. Dictionary
codes remain selective, while the full vocabulary is loaded on first use.
ListArray batches contain 2048 list cells by default. Smaller batches reduce
overfetch for sparse reads; larger batches usually improve scans and compression.
The row limit is not a byte limit, so one unusually large list can still require
a large transfer. See :ref:`ListArray` for batching controls.

Nullable list elements, nested lists, and ``contains``/``overlaps`` predicates
have the same semantics as local tables. Without an index a predicate scans the
remote list batches. A persisted ``kind="membership"`` index on a flat scalar
list fetches only the compressed posting batches for requested values. If the
result projects only other columns, the list payload remains unopened. Indexes
on nested lists and structs are not supported. ListArray ``storage="vl"`` also
remains unavailable through RemoteCTable.

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

See :doc:`Working with Remote Tables <../guides/remote_tables>` for column
access, filtering, buffering, reference saving, and materialization examples.

.. autoclass:: blosc2.RemoteCTable
    :members:
