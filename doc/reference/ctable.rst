.. _CTable:

CTable
======

A columnar compressed table backed by one physical container per column.
Scalar columns use :class:`~blosc2.NDArray`; list-valued columns use
:class:`~blosc2.ListArray`. Each column is stored, compressed, and queried
independently; rows are never materialised in their entirety unless you
explicitly call :meth:`~blosc2.CTable.to_arrow` or iterate with
:meth:`~blosc2.CTable.__iter__`.

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

       Return the number of live (non-deleted) rows.

    .. automethod:: __iter__

       Iterate over live rows in insertion order, yielding namedtuple-like
       row objects with one attribute per column.

    .. automethod:: __getitem__

       Type-driven indexing:

       * ``str`` — column name returns a :class:`Column`; any other string
         is interpreted as a boolean expression and behaves like
         :meth:`where`.
       * boolean :class:`~blosc2.LazyExpr` / :class:`~blosc2.NDArray` —
         filtered row view, same as :meth:`where`, e.g.
         ``t[t.temperature_f > 70]``.
       * ``int`` — single row as a namedtuple-like object.
       * ``slice`` — row-range view.
       * ``list[int]`` / ``ndarray[int]`` — gathered-row view.
       * ``ndarray[bool]`` — boolean-mask filtered view.
       * ``list[str]`` — column-projection view (same as :meth:`select`).

    .. automethod:: __repr__
    .. automethod:: __str__


Construction
------------

.. autosummary::

    CTable.__init__
    CTable.open
    CTable.load
    CTable.from_arrow
    CTable.from_parquet
    CTable.from_csv

.. automethod:: CTable.__init__
.. automethod:: CTable.open
.. automethod:: CTable.load
.. automethod:: CTable.from_arrow
.. automethod:: CTable.from_parquet
.. automethod:: CTable.from_csv


Parquet interoperability
------------------------

Parquet import/export is intended as logical data interchange between Parquet
and Blosc2 CTable, not as exact preservation of Parquet's physical layout. For
example, Parquet files whose top-level schema is an unnamed ``list<struct<...>>``
may be imported as a regular CTable whose rows are the list elements and whose
nested scalar fields are exposed as ordinary dotted columns. Exporting such a
table writes a valid logical Parquet table, but does not attempt to reconstruct
the original unnamed root-list grouping, row groups, encoding choices, or file
metadata exactly.


Null policy
-----------

Nullable scalar CTable columns are represented with per-column sentinel values,
not native validity bitmaps.  When CTable has to infer those sentinels, the
selection can be customized with :class:`NullPolicy` and scoped with
:func:`null_policy`::

    policy = blosc2.NullPolicy(
        signed_int_strategy="max",
        string_value="<NULL>",
        column_null_values={"user_id": -1, "country": "NA"},
    )

    with blosc2.null_policy(policy):
        table = blosc2.CTable.from_parquet("data.parquet")

The same policy is used by explicit nullable schema specs when no
``null_value`` is supplied::

    from dataclasses import dataclass

    @dataclass
    class Row:
        user_id: int = blosc2.field(blosc2.int64(nullable=True))
        country: str = blosc2.field(blosc2.string(nullable=True))

    with blosc2.null_policy(policy):
        table = blosc2.CTable(Row)

Sentinels are resolved in this order: explicit ``null_value`` in the schema,
``NullPolicy.column_null_values`` for a matching column, then the type-wide
``NullPolicy`` default.  Columns without ``nullable=True`` or an explicit
``null_value`` are not nullable.

.. autosummary::

    NullPolicy
    null_policy
    get_null_policy

.. autoclass:: NullPolicy
.. autofunction:: null_policy
.. autofunction:: get_null_policy


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

.. autoattribute:: CTable.col_names
.. autoproperty:: CTable.computed_columns
.. autoproperty:: CTable.nrows
.. autoproperty:: CTable.ncols
.. autoproperty:: CTable.cbytes
.. autoproperty:: CTable.nbytes
.. autoproperty:: CTable.schema
.. autoattribute:: CTable.base


Inserting data
--------------

.. autosummary::

    CTable.append
    CTable.extend

.. automethod:: CTable.append
.. automethod:: CTable.extend


Querying
--------

Boolean expressions
~~~~~~~~~~~~~~~~~~~

Use bitwise operators (``&``, ``|``, ``~``) or string expressions for
row-wise boolean logic.  Python's logical operators ``and``, ``or`` and
``not`` cannot be overloaded and therefore do not build lazy column
expressions.

Use column expressions with explicit parentheses around comparisons::

    t.where((t.amount > 100) & (t.region == "North"))
    t.where(~t.returned)

or use string expressions when that reads better::

    t.where("amount > 100 and region == 'North'")
    t.where("not returned")
    t["not returned"]

The last three forms for negating a boolean column are equivalent:
``t.where(~t.returned)``, ``t.where("not returned")``, and
``t["not returned"]``.

Indexing & projection
~~~~~~~~~~~~~~~~~~~~~

CTable indexing is type-driven::

    t["amount"]                 # column access
    t[3]                        # one row as a namedtuple-like object
    t[3:8]                      # row view
    t[[1, 4, 7]]                # gathered-row view
    t[mask]                     # filtered row view
    t[t.amount > 100]           # LazyExpr filtered row view, like where()
    t[["region", "amount"]]   # projected column view

String keys first try exact column-name lookup.  If the string is not a
column name, it is interpreted as a boolean expression and behaves like
:meth:`CTable.where`.  Boolean :class:`~blosc2.LazyExpr` and boolean
:class:`~blosc2.NDArray` keys also behave like :meth:`CTable.where`, so computed
column predicates such as ``t[t.temperature_f > 70]`` are supported.

For explicit filtered projection, use::

    t.where("amount > 100", columns=["region", "amount"])

When a NumPy structured array is needed, materialize explicitly::

    np.asarray(t[:10])

.. autosummary::

    CTable.where
    CTable.view
    CTable.select
    CTable.head
    CTable.tail
    CTable.sample
    CTable.sort_by
    CTable.iter_sorted

.. automethod:: CTable.where
.. automethod:: CTable.view
.. automethod:: CTable.select
.. automethod:: CTable.head
.. automethod:: CTable.tail
.. automethod:: CTable.sample
.. automethod:: CTable.sort_by
.. automethod:: CTable.iter_sorted


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


Indexes
-------

CTable indexes are created with :meth:`CTable.create_index` and returned as
:class:`blosc2.Index` handles.  For tables, ``Index`` refers to an entry stored
in the table index catalog and delegates maintenance operations such as
``drop()``, ``rebuild()``, and ``compact()`` back to the owning table.  Users
normally only receive these handles from the CTable API; they do not instantiate
them directly.

Indexes can target stored columns or **direct expressions** over stored columns
via ``create_index(expression=...)``.  This lets queries reuse indexes for
derived predicates without adding either a computed column or a materialized
stored one.  A matching ``FULL`` direct-expression index can also be reused by
ordering paths such as :meth:`CTable.sort_by` when sorting by a computed column
backed by the same expression.  ``OPSI`` indexes are a separate exact-filtering
tier with a tunable number of iterative ordering cycles; they are not intended
to converge to a completely sorted ``FULL``/CSI index, so use ``FULL`` when
globally sorted ordered reuse is required.

.. autosummary::

    CTable.create_index
    CTable.index
    CTable.indexes
    CTable.drop_index
    CTable.rebuild_index
    CTable.compact_index

.. automethod:: CTable.create_index
.. automethod:: CTable.index
.. autoattribute:: CTable.indexes
.. automethod:: CTable.drop_index
.. automethod:: CTable.rebuild_index
.. automethod:: CTable.compact_index

See :class:`blosc2.Index` for the returned handle attributes and methods.


Persistence
-----------

Persist CTables to disk or interchange formats, and restore them later without
losing schema information. These methods cover native Blosc2 persistence as
well as import/export paths for CSV, Arrow, and Parquet data.

.. autosummary::

    CTable.load
    CTable.open
    CTable.save
    CTable.to_b2z
    CTable.to_b2d
    CTable.to_csv
    CTable.to_arrow
    CTable.to_parquet
    CTable.from_arrow
    CTable.from_parquet
    CTable.from_csv

.. automethod:: CTable.load
.. automethod:: CTable.open
.. automethod:: CTable.save
.. automethod:: CTable.to_b2z
.. automethod:: CTable.to_b2d
.. automethod:: CTable.to_csv
.. automethod:: CTable.to_arrow
.. automethod:: CTable.to_parquet
.. automethod:: CTable.from_arrow
.. automethod:: CTable.from_parquet
.. automethod:: CTable.from_csv


Inspection & statistics
-----------------------

Compute common descriptive statistics directly on ``CTable`` data without
materializing rows first. These methods operate column-wise on the compressed
representation, making it easy to summarize distributions or measure
relationships between numeric columns.

.. autosummary::

    CTable.column_schema
    CTable.info
    CTable.schema_dict
    CTable.describe
    CTable.cov

.. automethod:: CTable.column_schema
.. automethod:: CTable.info
.. automethod:: CTable.schema_dict
.. automethod:: CTable.describe
.. automethod:: CTable.cov


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

       Return the number of live (non-deleted) values in this column.

    .. automethod:: __iter__

       Iterate over live values in insertion order, skipping deleted rows.

    .. automethod:: __getitem__
    .. automethod:: __setitem__

       Set one or more live column values.  Accepts the same index forms as
       :meth:`__getitem__`.


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
    timestamp

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
.. autoclass:: timestamp

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
    vlstring
    vlbytes
    struct
    object
    list

.. autoclass:: string
.. autoclass:: bytes
.. autofunction:: vlstring
.. autofunction:: vlbytes
.. autofunction:: struct
.. autofunction:: object
.. autofunction:: list

Object columns
--------------

Timestamp columns
-----------------

Timestamp columns are declared with :class:`blosc2.timestamp` and store signed
64-bit epoch offsets with timestamp metadata.  Column reads return
``numpy.datetime64`` values, comparisons accept ``numpy.datetime64`` values,
ISO-like strings, or Python ``datetime`` objects, and Arrow/Parquet import/export
roundtrips timestamp units and time zones::

    from dataclasses import dataclass
    import numpy as np
    import blosc2 as b2

    @dataclass
    class Event:
        when: np.datetime64 = b2.field(b2.timestamp(unit="us", nullable=True))
        value: int = b2.field(b2.int64())

    table = b2.CTable(Event)
    table.append(["2025-01-01T12:00:00", 42])
    recent = table[table.when >= np.datetime64("2025-01-01", "us")]

Object columns
--------------

Schema-less object columns are declared with :func:`blosc2.object` and store one
msgpack-serializable Python object (or ``None`` when nullable) per row in
batched variable-length storage. Prefer typed specs such as :func:`blosc2.struct`
or :func:`blosc2.list` when the payload has a stable schema; use object columns
for heterogeneous per-row payloads::

    from dataclasses import dataclass
    import blosc2 as b2

    @dataclass
    class Event:
        id: int = b2.field(b2.int64())
        payload: object = b2.field(b2.object(nullable=True))

    table.append([1, {"kind": "click", "xy": [10, 20]}])
    table.append([2, ("custom", {"nested": True})])
    table.append([3, None])

Object columns have no fixed Arrow type, so :meth:`CTable.to_arrow` and
:meth:`CTable.to_parquet` raise for them unless users first convert the payloads
to a typed representation.  They are not used as an implicit fallback during
Parquet import; unsupported Arrow/Parquet types still raise unless explicitly
imported through :meth:`CTable.from_arrow` with ``object_fallback=True``.

Nested fields
-------------

CTable supports first-class **nested struct schemas** by physically flattening
struct leaves into independent compressed columns.  This keeps analytics fast
(each leaf is an ordinary :class:`~blosc2.NDArray`), while preserving the
logical nested row shape on read.

**Automatic flattening from Arrow / Parquet**

When :meth:`CTable.from_arrow` or :meth:`CTable.from_parquet` encounters a
top-level ``struct<…>`` field, it recursively flattens every scalar leaf into a
dotted column name and stores each leaf as its own physical column::

    import pyarrow as pa
    import blosc2

    trip_type = pa.struct([
        ("begin", pa.struct([("lon", pa.float64()), ("lat", pa.float64())])),
        ("end",   pa.struct([("lon", pa.float64()), ("lat", pa.float64())])),
    ])
    schema = pa.schema([pa.field("trip", trip_type),
                        pa.field("fare", pa.float64())])
    batch = pa.record_batch(
        [pa.array([{"begin": {"lon": -87.6, "lat": 41.8},
                    "end":   {"lon": -87.7, "lat": 41.9}}],
                  type=trip_type),
         pa.array([12.5])],
        schema=schema,
    )

    t = blosc2.CTable.from_arrow(schema, [batch])
    # t.col_names → ['trip.begin.lon', 'trip.begin.lat',
    #                 'trip.end.lon',   'trip.end.lat', 'fare']

**Column access**

Nested leaves are accessed with their dotted logical name or via chained
attribute proxies::

    t["trip.begin.lon"].mean()      # Column object (fast path)
    t.trip.begin.lon.max()          # attribute proxy, same column

A literal ``.``, ``/``, or ``\\`` inside an Arrow field name is escaped with a
backslash in the logical column name.  For example, path segments
``("trip.info", "begin/point", "lon.deg")`` become::

    t[r"trip\.info.begin\/point.lon\.deg"]

Such leaves are stored with percent-encoded path segments under ``_cols``; the
example above is stored at ``_cols/trip%2Einfo/begin%2Fpoint/lon%2Edeg``.

**Filtering and expressions**

Dotted names work everywhere a flat column name would::

    t.where("trip.begin.lon > -87.7 and fare > 10")
    t.where(t.trip.begin.lon > -87.7)

**Select / projection**

A struct prefix expands to all descendant leaves::

    t.select(["trip.begin"])        # → columns trip.begin.lon, trip.begin.lat
    t.select(["trip"])              # → all four trip.* leaves

**Indexes and aggregates**

Scalar leaf columns support all the same operations as flat columns::

    t.create_index(col_name="trip.begin.lon")
    t.where("trip.begin.lon > -87.7").nrows   # uses the index

**Row reconstruction**

Single-row access reconstructs the original nested dict shape::

    row = t[0]
    row.trip       # → {"begin": {"lon": ..., "lat": ...}, "end": {...}}
    row.fare       # → 12.5

**Inserting nested rows**

:meth:`CTable.append` and :meth:`CTable.extend` accept either the flat dotted
form or the original nested dict / list-of-dicts shape::

    # flat dotted keys
    t.append({"trip.begin.lon": -87.6, "trip.begin.lat": 41.8,
              "trip.end.lon": -87.7,   "trip.end.lat": 41.9, "fare": 12.5})

    # original nested dict (auto-flattened)
    t.append({"trip": {"begin": {"lon": -87.6, "lat": 41.8},
                        "end":   {"lon": -87.7, "lat": 41.9}},
              "fare": 12.5})

    # extend with a list of nested dicts
    t.extend([
        {"trip": {"begin": {"lon": -87.6, "lat": 41.8},
                  "end":   {"lon": -87.7, "lat": 41.9}}, "fare": 12.5},
        {"trip": {"begin": {"lon": -87.5, "lat": 41.7},
                  "end":   {"lon": -87.8, "lat": 41.6}}, "fare": 8.0},
    ])

**Physical storage layout**

Leaf columns are stored under a hierarchical path in the backing container:
``/_cols/trip/begin/lon``, ``/_cols/trip/begin/lat``, etc.  Intermediate nodes
are namespaces only; no data is stored at non-leaf levels.

**Arrow / Parquet round-trip**

:meth:`CTable.to_parquet` and :meth:`CTable.to_arrow` reconstruct the original
nested Arrow schema from the stored metadata, so round-trips are lossless::

    t.to_parquet("out.parquet")    # Arrow schema has top-level "trip" struct

Struct columns
--------------

Struct columns are declared with :func:`blosc2.struct` and store one dictionary
(or ``None`` when nullable) per row in batched variable-length storage.  They are
also used when importing top-level Arrow/Parquet ``struct<...>`` columns when
**not** using the nested-leaf flattening path described above::

    from dataclasses import dataclass
    import blosc2 as b2

    @dataclass
    class Product:
        properties: dict = b2.field(
            b2.struct({"code": b2.int32(), "label": b2.vlstring()}, nullable=True)
        )

    table.append([{"code": 1, "label": "fresh"}])
    table.append([None])

List columns
------------

List columns are declared with :func:`blosc2.list`, for example::

    from dataclasses import dataclass
    import blosc2 as b2

    @dataclass
    class Product:
        code: str = b2.field(b2.string(max_length=8))
        tags: list[str] = b2.field(b2.list(b2.string(), nullable=True))

Whole-cell replacement is supported, so users should reassign modified lists::

    row_tags = table.tags[0]
    row_tags.append("extra")      # local Python list only
    table.tags[0] = row_tags      # explicit write-back
