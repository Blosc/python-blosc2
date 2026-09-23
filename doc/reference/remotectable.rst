.. _RemoteCTable:

RemoteCTable
============

``RemoteCTable`` is a read-only :class:`blosc2.CTable` backed by a remote B2Z
archive. Fixed-width, shaped, nullable, UTF-8, batch-backed variable-length,
batch-backed list, struct/object, and dictionary columns are fetched on demand.
Standalone tables can be opened directly; tables inside a hierarchy can be
selected with ``dataset=`` or through :class:`blosc2.RemoteStore`.
PyTables/HDF5 sources may supply ``hdf5_index=`` as a native index dictionary,
local JSON path, or remote fsspec URL. This skips source discovery and does not
modify the HDF5 file; see :doc:`../guides/remote_arrays`.

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

Scalar queries automatically use persisted ``SUMMARY``, ``FULL``, ``PARTIAL``,
``OPSI``, and ``BUCKET`` indexes. Their sidecars are opened lazily and participate
in the outer table's cache budget and traffic accounting. SUMMARY reads compact
min/max records before fetching candidate data blocks; positional indexes use
their navigation data to fetch selected value and row-position ranges. Queries
retain a correct scan fallback when an index layout or expression is unsupported.

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
may include sibling leaves. This also applies when a table column is itself a
``RemoteArray`` reference to another fsspec or Caterva2 URL: the outer table
policy overrides the carrier policy for that handle, and all such columns share
one budget. Standalone instances of those arrays keep their original policies.
A table selected from a store owns an independent handle, but its columns and
views remain borrowed from that table. Refresh a nested table through its root
store; a standalone table can call ``refresh()``.

Server processes can share one bounded sparse disk cache with
``blosc2.open(url, cache_dir="shared-cache", shared_cache=True)``. All processes
using that directory must enable sharing. Operations serialize per store,
and the aggregate compressed-payload budget defaults to 256 MiB; explicitly
pass ``max_cache_bytes=None`` for unlimited retention. Use a separate directory
from ordinary exclusive caches. The budget does not bound total disk usage or
peak RAM.

``RemoteCTable.with_sparse_cache(url, runtime_cache_path)`` remains available
for advanced attachment with manifests or seed carriers, with the same default
budget and shared-cache implementation.

See :doc:`Working with Remote Tables <../guides/remote_tables>` for column
access, filtering, buffering, reference saving, and materialization examples.

.. autoclass:: blosc2.RemoteCTable
    :members:
