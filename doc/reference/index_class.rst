.. _Index:

Index
=====

Handle for an index attached to a :class:`~blosc2.NDArray` or :class:`~blosc2.CTable`.

``Index`` objects are returned by NDArray indexing APIs such as
:meth:`blosc2.NDArray.create_index`, :meth:`blosc2.NDArray.index`, and
:attr:`blosc2.NDArray.indexes`, and by the equivalent :class:`~blosc2.CTable`
indexing APIs.  Use this handle to inspect index metadata and storage usage, or
to drop, rebuild, and compact the index.  Users normally do not instantiate
``Index`` directly.

.. currentmodule:: blosc2

.. autoclass:: Index

.. autoattribute:: Index.descriptor
.. autoattribute:: Index.kind
.. autoattribute:: Index.col_name
.. autoattribute:: Index.field
.. autoattribute:: Index.name
.. autoattribute:: Index.target
.. autoattribute:: Index.persistent
.. autoattribute:: Index.stale
.. autoattribute:: Index.nbytes
.. autoattribute:: Index.cbytes
.. autoattribute:: Index.cratio
.. automethod:: Index.storage_stats
.. automethod:: Index.__getitem__
.. automethod:: Index.__iter__
.. automethod:: Index.__len__
.. automethod:: Index.drop
.. automethod:: Index.rebuild
.. automethod:: Index.compact
