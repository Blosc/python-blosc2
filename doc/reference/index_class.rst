.. _Index:

Index
=====

Handle for an index attached to a :class:`~blosc2.NDArray`.

``Index`` objects are returned by NDArray indexing APIs such as
:meth:`blosc2.NDArray.create_index`, :meth:`blosc2.NDArray.index`, and
:attr:`blosc2.NDArray.indexes`.  Use this handle to inspect index metadata and
storage usage, or to drop, rebuild, and compact the index.  Users normally do
not instantiate ``Index`` directly.

For table indexes, see :class:`blosc2.ctable.CTableIndex`, documented in the
:ref:`CTable` reference.

.. currentmodule:: blosc2

.. autoclass:: Index

.. autoattribute:: Index.descriptor
.. autoattribute:: Index.kind
.. autoattribute:: Index.field
.. autoattribute:: Index.name
.. autoattribute:: Index.target
.. autoattribute:: Index.persistent
.. autoattribute:: Index.stale
.. autoattribute:: Index.nbytes
.. autoattribute:: Index.cbytes
.. autoattribute:: Index.cratio
.. automethod:: Index.drop
.. automethod:: Index.rebuild
.. automethod:: Index.compact
