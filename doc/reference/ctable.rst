.. _CTable:

CTable
======

A columnar compressed table backed by one :class:`~blosc2.NDArray` per column.
Each column is stored, compressed, and queried independently; rows are never
materialised in their entirety unless you explicitly call :meth:`~blosc2.CTable.to_arrow`
or iterate with :meth:`~blosc2.CTable.__iter__`.

.. currentmodule:: blosc2

.. autoclass:: CTable
    :members:
    :member-order: groupwise

    .. rubric:: Special methods

    .. autosummary::

        CTable.__len__
        CTable.__iter__
        CTable.__getitem__
        CTable.__repr__
        CTable.__str__

    .. automethod:: __len__
    .. automethod:: __iter__
    .. automethod:: __getitem__
    .. automethod:: __repr__
    .. automethod:: __str__


Construction
------------

.. autosummary::

    CTable.__init__
    CTable.open
    CTable.load
    CTable.from_arrow
    CTable.from_csv

.. automethod:: CTable.__init__
.. automethod:: CTable.open
.. automethod:: CTable.load
.. automethod:: CTable.from_arrow
.. automethod:: CTable.from_csv


Attributes
----------

.. autosummary::

    CTable.col_names
    CTable.computed_columns
    CTable.nrows
    CTable.ncols
    CTable.cbytes
    CTable.nbytes
    CTable.schema
    CTable.base

.. autoproperty:: CTable.computed_columns
.. autoproperty:: CTable.nrows
.. autoproperty:: CTable.ncols
.. autoproperty:: CTable.cbytes
.. autoproperty:: CTable.nbytes
.. autoproperty:: CTable.schema


Inserting data
--------------

.. autosummary::

    CTable.append
    CTable.extend

.. automethod:: CTable.append
.. automethod:: CTable.extend


Querying
--------

.. autosummary::

    CTable.where
    CTable.select
    CTable.head
    CTable.tail
    CTable.sample
    CTable.sort_by

.. automethod:: CTable.where
.. automethod:: CTable.select
.. automethod:: CTable.head
.. automethod:: CTable.tail
.. automethod:: CTable.sample
.. automethod:: CTable.sort_by


Aggregates & statistics
-----------------------

.. autosummary::

    CTable.describe
    CTable.cov

.. automethod:: CTable.describe
.. automethod:: CTable.cov


Mutations
---------

In addition to physical schema changes such as :meth:`CTable.add_column`,
CTables can host **computed columns** backed by a lazy expression over stored
columns.  Computed columns are read-only, use no extra storage, participate in
display, filtering, sorting, and aggregates, and are persisted across
:meth:`CTable.save`, :meth:`CTable.load`, and :meth:`CTable.open`.

When a computed result should become a normal stored column, use
:meth:`CTable.materialize_computed_column`.  The materialized column is a stored
snapshot that can be indexed like any other stored column.  New rows inserted
later via :meth:`CTable.append` or :meth:`CTable.extend` auto-fill omitted
materialized-column values from the recorded expression metadata.

CTable indexes can also target **direct expressions** over stored columns via
``create_index(expression=...)``.  This lets queries reuse indexes for derived
predicates without adding either a computed column or a materialized stored one.
A matching ``FULL`` direct-expression index can also be reused by ordering paths
such as :meth:`CTable.sort_by` when sorting by a computed column backed by the
same expression.  ``OPSI`` indexes are a separate exact-filtering tier with a
tunable number of iterative ordering cycles; they are not intended to converge
to a completely sorted ``FULL``/CSI index, so use ``FULL`` when globally sorted
ordered reuse is required.

.. autosummary::

    CTable.delete
    CTable.compact
    CTable.add_column
    CTable.add_computed_column
    CTable.materialize_computed_column
    CTable.drop_computed_column
    CTable.drop_column
    CTable.rename_column

.. automethod:: CTable.delete
.. automethod:: CTable.compact
.. automethod:: CTable.add_column
.. automethod:: CTable.add_computed_column
.. automethod:: CTable.materialize_computed_column
.. automethod:: CTable.drop_computed_column
.. automethod:: CTable.drop_column
.. automethod:: CTable.rename_column


Persistence
-----------

.. autosummary::

    CTable.save
    CTable.to_csv
    CTable.to_arrow

.. automethod:: CTable.save
.. automethod:: CTable.to_csv
.. automethod:: CTable.to_arrow


Inspection
----------

.. autosummary::

    CTable.info
    CTable.schema_dict
    CTable.column_schema

.. automethod:: CTable.info
.. automethod:: CTable.schema_dict
.. automethod:: CTable.column_schema


----

.. _Column:

Column
======

A lazy column accessor returned by ``table["col_name"]`` or ``table.col_name``.
All index operations and aggregates apply the table's tombstone mask
(``_valid_rows``) so deleted rows are silently excluded.

.. autoclass:: Column
    :members:
    :member-order: groupwise

    .. rubric:: Special methods

    .. autosummary::

        Column.__len__
        Column.__iter__
        Column.__getitem__
        Column.__setitem__

    .. automethod:: __len__
    .. automethod:: __iter__
    .. automethod:: __getitem__
    .. automethod:: __setitem__


Attributes
----------

.. autosummary::

    Column.dtype
    Column.null_value

.. autoproperty:: Column.dtype
.. autoproperty:: Column.null_value


Data access
-----------

.. autosummary::

    Column.view
    Column.iter_chunks
    Column.assign

.. autoproperty:: Column.view
.. automethod:: Column.iter_chunks
.. automethod:: Column.assign


Nullable helpers
----------------

.. autosummary::

    Column.is_null
    Column.notnull
    Column.null_count

.. automethod:: Column.is_null
.. automethod:: Column.notnull
.. automethod:: Column.null_count


Unique values
-------------

.. autosummary::

    Column.unique
    Column.value_counts

.. automethod:: Column.unique
.. automethod:: Column.value_counts


Aggregates
----------

Null sentinel values are automatically excluded from all aggregates.

.. autosummary::

    Column.sum
    Column.min
    Column.max
    Column.mean
    Column.std
    Column.any
    Column.all

.. automethod:: Column.sum
.. automethod:: Column.min
.. automethod:: Column.max
.. automethod:: Column.mean
.. automethod:: Column.std
.. automethod:: Column.any
.. automethod:: Column.all


----

.. _SchemaSpecs:

Schema Specs
============

Schema specs are passed to :func:`field` to declare a column's type,
storage constraints, and optional null sentinel.  They are also
available directly in the ``blosc2`` namespace (e.g. ``blosc2.int64``).

.. currentmodule:: blosc2

.. autofunction:: field

Numeric
-------

.. autosummary::

    int8
    int16
    int32
    int64
    uint8
    uint16
    uint32
    uint64
    float32
    float64

.. autoclass:: int8
.. autoclass:: int16
.. autoclass:: int32
.. autoclass:: int64
.. autoclass:: uint8
.. autoclass:: uint16
.. autoclass:: uint32
.. autoclass:: uint64
.. autoclass:: float32
.. autoclass:: float64

Complex
-------

.. autosummary::

    complex64
    complex128

.. autoclass:: complex64
.. autoclass:: complex128

Boolean
-------

.. autosummary::

    bool

.. autoclass:: bool

Text & binary
-------------

.. autosummary::

    string
    bytes

.. autoclass:: string
.. autoclass:: bytes
