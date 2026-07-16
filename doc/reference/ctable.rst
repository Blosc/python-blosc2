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

.. autoclass:: blosc2.CTable
    :members:
    :member-order: groupwise
    :special-members: __getitem__, __getattr__

    .. rubric:: Special methods

    .. autosummary::

        CTable.__len__
        CTable.__iter__
        CTable.__getitem__
        CTable.__getattr__
        CTable.__repr__
        CTable.__str__

    .. automethod:: __len__

       Return the number of live (non-deleted) rows.

    .. automethod:: __iter__

       Iterate over live rows in insertion order, yielding namedtuple-like
       row objects with one attribute per column.

    ``__getitem__`` supports type-driven indexing:

    * ``str`` — column name returns a :class:`Column`; any other string
      is interpreted as a boolean expression and behaves like :meth:`where`.
    * boolean :class:`~blosc2.LazyExpr` / :class:`~blosc2.NDArray` —
      filtered row view, same as :meth:`where`, e.g.
      ``t[t.temperature_f > 70]``.
    * ``int`` — single row as a namedtuple-like object.
    * ``slice`` — row-range view.
    * ``list[int]`` / ``ndarray[int]`` — gathered-row view.
    * ``ndarray[bool]`` — boolean-mask filtered view.
    * ``list[str]`` — column-projection view (same as :meth:`select`).

    ``__getattr__`` provides convenience attribute-style column access only
    after normal Python attribute lookup fails; use ``t["name"]`` for columns
    that conflict with table attributes or methods.

    .. automethod:: __repr__
    .. automethod:: __str__


Display
-------

``CTable`` objects have a compact tabular string representation.  Use
:meth:`CTable.to_string` for one-off formatting choices, or
:func:`set_printoptions` to configure the default display used by ``str(table)``
and ``print(table)``::

    print(table.to_string(display_index=True))

    blosc2.set_printoptions(display_index=True)
    print(table)

The displayed index is a logical, pandas-like row number in the rendered table;
it is not the physical storage position.

.. autosummary::

    CTable.to_string
    set_printoptions
    get_printoptions

.. automethod:: CTable.to_string
.. autofunction:: set_printoptions
.. autofunction:: get_printoptions


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


Nulls in expressions
---------------------

Arithmetic and comparisons built from :class:`Column` objects (``t.x + 1``,
``t.x > 0``, ...) propagate nulls on nullable int/timestamp/bool columns
without touching storage or the on-disk format — a sentinel is a validity
bitmap encoded in-band, so propagation is purely a rewrite of the lazy
expression:

- **Arithmetic** (``+ - * / // % **``): the result is null (NaN) wherever
  any nullable operand is null, promoting integer/timestamp results to
  ``float64`` — the same promotion pandas' legacy (pre-nullable-dtype)
  integer-null arithmetic performs. Non-nullable columns are unaffected and
  pay no overhead::

      (t.price + 1)          # NaN where price is null; float64 either way
      (t.price + t.tax)      # NaN where either operand is null

- **Comparisons** (``< <= > >= == !=``): SQL ``WHERE`` semantics — a null
  never satisfies any comparison, including ``==``/``!=`` against the raw
  sentinel value itself. Only :meth:`Column.is_null` /
  :meth:`Column.notnull` test for nullness::

      t[t.price < 0]         # excludes rows where price is null
      t[t.price == -1]       # False for a null row even if -1 is the sentinel
      t[t.price.is_null()]   # the only way to select null rows

- **Boolean combinators** (``& | ~``) need no special handling: their
  inputs are already-resolved comparison results with null-ness folded to
  ``False``. Mind the consequence for negation: since a null compares
  ``False``, ``~(t.price > 0)`` *selects* null rows (they are "not > 0").
  To exclude them, write the complementary comparison instead
  (``t.price <= 0``), which never matches nulls.

Kleene three-valued logic (where ``null > 0`` evaluates to null rather than
``False``) is intentionally out of scope — it needs a validity channel on
boolean intermediates, i.e. masks, which CTable does not use.

Reductions on derived expressions skip nulls too: arithmetic involving a
nullable column returns a ``NullableExpr`` — a thin wrapper that remembers
which rows are null — so ``sum``/``mean``/``min``/``max``/``std`` on it skip
nulls (and deleted rows) exactly like the corresponding :meth:`Column.sum`
etc. do on a real column::

      t.price.sum()          # skips nulls (real Column)
      (t.price + 1).sum()    # skips nulls too (NullableExpr)
      ((t.price + 1) * 2).mean()  # chaining keeps the null channel

Only *nulls* are skipped: a NaN produced by the arithmetic itself (e.g.
``0/0`` on non-null values) is a value, not a null, and poisons the
reduction as usual. ``min()``/``max()`` raise ``ValueError`` when every
value is null; ``sum()`` returns ``0.0`` and ``mean()``/``std()`` return
NaN — the same conventions as the Column reductions.


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
    t[[1, 4, 7]]                # gathered-row view (mask-based)
    t.take([1, 4, 1])           # materialized row gather preserving order/duplicates
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

Chained pipelines
~~~~~~~~~~~~~~~~~

:meth:`CTable.assign` returns a view with additional computed columns —
never mutating the table, never copying column data — and :func:`blosc2.col`
builds an unbound column expression that resolves against a table only when
bound (in ``assign()``, ``t[...]``, or :meth:`CTable.where`).  Together they
enable pandas-3 style method chains::

    from blosc2 import col

    result = (
        t.assign(profit=col("revenue") - col("cost"))[col("profit") > 0]
        .sort_by("profit", ascending=False)
        .head(10)
    )

.. autosummary::

    CTable.where
    CTable.dropna
    CTable.view
    CTable.take
    CTable.select
    CTable.assign
    CTable.head
    CTable.tail
    CTable.sample
    CTable.sort_by
    CTable.iter_sorted
    CTable.group_by
    col

.. automethod:: CTable.where
.. automethod:: CTable.dropna
.. automethod:: CTable.view
.. automethod:: CTable.take
.. automethod:: CTable.select
.. automethod:: CTable.assign
.. automethod:: CTable.head
.. automethod:: CTable.tail
.. automethod:: CTable.sample
.. automethod:: CTable.sort_by
.. automethod:: CTable.iter_sorted
.. automethod:: CTable.group_by
.. autofunction:: col


Group-by reductions
-------------------

:meth:`CTable.group_by` returns a lightweight deferred group-by object.  It is
not a table view; methods such as :meth:`~blosc2.CTableGroupBy.size`,
:meth:`~blosc2.CTableGroupBy.count`, :meth:`~blosc2.CTableGroupBy.sum`,
:meth:`~blosc2.CTableGroupBy.argmax`, and :meth:`~blosc2.CTableGroupBy.agg`
materialize a new :class:`CTable` with
one row per group::

    by_city = t.group_by("city", sort=True)
    counts = by_city.size()                  # row count per city / COUNT(*)
    non_null = by_city.count("sales")        # non-null sales count / COUNT(sales)
    totals = by_city.sum("sales")            # equivalent to agg({"sales": "sum"})
    means = by_city.mean("sales")
    mins = by_city.min("sales")
    maxs = by_city.max("sales")
    min_rows = by_city.argmin("sales")       # logical row position of min sales
    max_rows = by_city.argmax("sales")       # logical row position of max sales

:meth:`~blosc2.CTableGroupBy.agg` applies several aggregations at once and offers
three ways to name the result columns, which can be combined::

    # Auto-named mapping: result columns are "<column>_<op>", e.g. sales_sum.
    by_city.agg({"sales": ["sum", "mean"]})

    # Auto-named list of pairs: same naming, but accepts Column objects
    # (t.sales), which cannot be dict keys because Column is unhashable.  Ops
    # may also be blosc2 reduction functions (blosc2.sum), matched by identity.
    by_city.agg([(t.sales, [blosc2.sum, "mean"])])

    # Explicitly named (pandas-style): output_name=(column, op).
    by_city.agg(revenue=("sales", "sum"), avg_sale=("sales", "mean"))

    # Forms combine, e.g. a list of pairs with a named row count via ("*", "size").
    by_city.agg([(t.sales, "sum")], n=("*", "size"))

The auto-suffix naming is compact and collision-safe when a column is aggregated
several ways; the list-of-pairs form additionally lets you pass ``Column``
objects (``t.sales``) instead of name strings; use explicit names when you want a
specific (and easily addressable) output column name.

**Custom UDF aggregations** are accepted only via the named form --
``output_name=(column, callable)``, optionally with an explicit output dtype as
a third element::

    by_city.agg(sales_range=(t.sales, lambda a: a.max() - a.min()))
    # explicit dtype instead of inferring it from the callable's results:
    by_city.agg(sales_range=(t.sales, lambda a: a.max() - a.min(), blosc2.float32()))

The callable receives a 1-D NumPy array of the group's live, non-null values
(same null semantics as the built-in aggregations) and returns a scalar; it is
called once per group with a plain Python loop -- there is no JIT
acceleration for arbitrary UDFs yet.  A group with no non-null values for that
column never calls the callable, producing a null result instead (the same
convention ``sum``/``min``/``max`` already use).  The mapping and
list-of-pairs auto-named forms cannot derive an output column name for an
arbitrary callable, so they only accept the string/blosc2-reduction-function
ops described above.

**Execution engine.**  :meth:`CTable.group_by` takes an ``engine=`` parameter
for the *built-in* aggregations (``size``/``count``/``sum``/``mean``/``min``/
``max``/``argmin``/``argmax``): ``"auto"`` (default) and ``"numpy"`` both use
the NumPy/Cython chunked implementation; ``"jit"`` is reserved for a future
miniexpr-JIT path and currently raises :class:`NotImplementedError`. UDF
aggregations always run the plain Python per-group loop regardless of
``engine``.

**Output ordering.**  ``sort`` controls how groups are ordered, and is *always
by the group key(s), never by the aggregated value*:

* ``sort=True`` -- always sorted by key.
* ``sort=False`` -- unsorted (deterministic but unspecified order).
* ``sort=None`` (default) -- *auto*: integer and dictionary/string keys are
  sorted (free or vectorized); float and multi-key results are left unsorted, as
  their only ordering is an O(*G* log *G*) Python sort that can rival the grouping
  cost on high-cardinality data.  This differs from pandas, which defaults to
  ``sort=True``.

To order a result **by an aggregated value** (e.g. top spenders), sort the output
table afterwards with :meth:`CTable.sort_by`, referencing the result column name::

    top = by_city.agg(revenue=("sales", "sum")).sort_by("revenue", ascending=False)
    # or, with the auto-suffix name:
    top = by_city.sum("sales").sort_by("sales_sum", ascending=False)

Grouped results are in-memory by default.  Pass ``urlpath=`` to a terminal
method to write the result as a persistent :class:`CTable`::

    totals = by_city.sum("sales", urlpath="sales_by_city.b2d")

For array-oriented grouped reductions without a :class:`CTable`, see
:func:`blosc2.group_reduce`.

.. autoclass:: CTableGroupBy
    :members: size, count, sum, mean, min, max, argmin, argmax, agg


Mutations
---------

In addition to physical schema changes such as :meth:`CTable.add_column`,
CTables support two kinds of derived columns:

**Computed columns** (:meth:`CTable.add_computed_column`) are purely virtual —
they use no extra storage, are evaluated on demand, and are read-only.  They
participate in display, filtering, sorting, and aggregates, and are persisted
across save/open round-trips.  Because they have no physical storage, they
**cannot be indexed**.

**Generated columns** (:meth:`CTable.add_generated_column`) are physically
stored.  Their values are computed once and written to disk; new rows appended
later are auto-filled automatically.  Because the data is real, generated
columns **can be indexed** with :meth:`CTable.create_index`, which makes
``where()`` queries on them fast.

**Practical rule**: use a computed column when you just need a derived value
available for display, export, or occasional reads.  Use a generated column
(optionally with ``create_index=True``) when you need to filter or sort by a
derived value frequently — the index pays for itself after the first few
queries.

Both forms accept plain expression strings, :func:`blosc2.dsl_kernel`-decorated
functions, and :class:`blosc2.LazyUDF` objects.  DSL kernels support full Python
control flow (``if``/``else``, ``where()``, loops) and have their source
persisted and recompiled on open.

.. warning::

    Because DSL kernel source is persisted in the table metadata and re-executed
    during :func:`blosc2.open`, **do not open** ``.b2d`` files from untrusted
    sources if they may contain DSL computed or generated columns.  The kernel
    source runs with restricted builtins (no ``__import__``), but arbitrary
    Python code execution still carries risk.

When passing a :class:`blosc2.LazyUDF` built with an explicit ``jit_backend=``
(e.g. ``jit_backend="cc"`` to use the system C compiler instead of the default
TCC), that choice is persisted in the column metadata and automatically restored
on :func:`blosc2.open`.  This matters for kernels where one backend produces
measurably faster code — the optimised backend stays active for the lifetime of
the table without any extra configuration::

    t.add_generated_column(
        "score",
        values=blosc2.lazyudf(my_kernel, (t.col_a, t.col_b), jit_backend="cc"),
    )

When a computed result should become a stored snapshot rather than a live
virtual column, use :meth:`CTable.materialize_computed_column` to convert it
in place.

.. autosummary::

    CTable.delete
    CTable.compact
    CTable.add_column
    CTable.add_computed_column
    CTable.materialize_computed_column
    CTable.drop_computed_column
    CTable.drop_column
    CTable.rename_column
    CTable.apply

.. automethod:: CTable.delete
.. automethod:: CTable.compact
.. automethod:: CTable.add_column
.. automethod:: CTable.add_computed_column
.. automethod:: CTable.materialize_computed_column
.. automethod:: CTable.drop_computed_column
.. automethod:: CTable.drop_column
.. automethod:: CTable.rename_column
.. automethod:: CTable.apply


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

Choosing an index kind
    ``BUCKET`` is the default — cheapest to build and store, good for
    single‑column ``where`` and ``sort_by`` reuse.  ``SUMMARY`` stores only
    per‑segment min/max and is the lightest kind.

    ``FULL`` builds a globally sorted index; it enables **cross‑column
    refinement** for multi‑column conjunctions and carries a complete sort
    order for ``sort_by``.  ``PARTIAL`` is cheaper (roughly half the raw
    storage of ``FULL``) while still providing exact positions for
    cross‑column refinement — best for equality or narrow range queries.

    ``OPSI`` is a specialised tier for approximate ordering with iterative
    cycles; it can produce exact positions for cross‑column refinement but
    is not intended to converge to a globally sorted order — prefer
    ``FULL`` when ``sort_by`` acceleration is required.

    For highly selective multi‑column conjunctions, prefer ``FULL``,
    ``PARTIAL``, or ``OPSI`` on the most selective column so the planner
    can refine the other predicates on the compact exact positions instead
    of scanning the whole table.

    When a segment‑level index (``SUMMARY``, ``BUCKET``) would prune fewer
    than 50 % of candidate segments, the planner skips the index and falls
    back to a full scan to avoid per‑segment evaluation overhead.

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
    CTable.to_cframe
    CTable.to_csv
    CTable.to_arrow
    CTable.__arrow_c_stream__
    CTable.to_parquet
    CTable.from_arrow
    CTable.from_parquet
    CTable.from_csv
    ctable_from_cframe

CTable also implements the `Arrow PyCapsule interchange protocol
<https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html>`_
via :meth:`~blosc2.CTable.__arrow_c_stream__`, so pyarrow, DuckDB, Polars, and
pandas >= 2.2 can consume a CTable directly as a stream of record batches
(``pa.table(ct)``, ``duckdb.sql("SELECT ... FROM ct")``, ``pl.DataFrame(ct)``),
and :meth:`~blosc2.CTable.from_arrow` accepts any object implementing that
protocol on ingest. Strict zero-copy is not possible — the underlying data is
compressed, so decompression is unavoidably a copy — but there is no
intermediate materialization: batches are decompressed and handed to the
consumer one at a time, so memory use stays bounded regardless of table size.

.. automethod:: CTable.load
.. automethod:: CTable.open
.. automethod:: CTable.save
.. automethod:: CTable.to_b2z
.. automethod:: CTable.to_b2d
.. automethod:: CTable.to_cframe
.. automethod:: CTable.to_csv
.. automethod:: CTable.to_arrow
.. automethod:: CTable.__arrow_c_stream__
.. automethod:: CTable.to_parquet
.. automethod:: CTable.from_arrow
.. automethod:: CTable.from_parquet
.. automethod:: CTable.from_csv
.. autofunction:: ctable_from_cframe


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
    Column.row_transformer

.. autoproperty:: Column.dtype
.. autoproperty:: Column.null_value
.. autoproperty:: Column.row_transformer


Data access
-----------

.. autosummary::

    Column.view
    Column.raw
    Column.take
    Column.iter_chunks
    Column.assign

.. autoproperty:: Column.view
.. autoproperty:: Column.raw
.. automethod:: Column.take
.. automethod:: Column.iter_chunks
.. automethod:: Column.assign


Row transformers
----------------

``Column.row_transformer`` builds row-wise projections and reductions for
fixed-shape ndarray columns.  Use these transformers with
:meth:`CTable.add_generated_column` when the generated value should be computed
from each row's ndarray payload rather than from scalar columns::

    t.add_generated_column(
        "embedding_norm",
        values=t.embedding.row_transformer.norm(axis=0),
        dtype=blosc2.float64(),
    )
    t.add_generated_column(
        "image_mean_rgb",
        values=t.image.row_transformer.mean(axis=(0, 1)),
        dtype=blosc2.ndarray((3,), dtype=blosc2.float32()),
    )

.. autoclass:: RowTransformer
    :members: sum, mean, min, max, argmin, argmax, norm
    :special-members: __getitem__


Nullable helpers
----------------

.. autosummary::

    Column.is_null
    Column.notnull
    Column.null_count
    Column.fillna

.. automethod:: Column.is_null
.. automethod:: Column.notnull
.. automethod:: Column.null_count
.. automethod:: Column.fillna


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
    Column.argmin
    Column.argmax
    Column.mean
    Column.std
    Column.any
    Column.all

.. automethod:: Column.sum
.. automethod:: Column.min
.. automethod:: Column.max
.. automethod:: Column.argmin
.. automethod:: Column.argmax
.. automethod:: Column.mean
.. automethod:: Column.std
.. automethod:: Column.any
.. automethod:: Column.all


----

.. _NestedColumn:

NestedColumn
============

A read-only accessor for a nested (dotted) group of CTable columns, returned by
attribute access on a :class:`CTable` (or on another ``NestedColumn``) when the
name refers to an internal node of the dotted column tree rather than a leaf.

For a table flattened from a ``struct`` / ``list<struct>`` schema (see
:ref:`Nested fields <NestedFields>`), ``t.trip`` is a ``NestedColumn`` grouping
every leaf under the ``trip.`` prefix, while a leaf such as ``t.trip.sec`` or
``t.trip.begin.lon`` is a :class:`Column`.  Drilling into an intermediate node
yields another ``NestedColumn``::

    t.trip                     # <NestedColumn 'trip'>
    t.trip.col_names           # ['sec', 'km', 'begin.lon', 'begin.lat', ...]
    t.trip.begin               # <NestedColumn 'trip.begin'>
    t.trip.begin.lon           # Column
    print(t.trip.info)         # aggregate metadata over the group

Users do not instantiate ``NestedColumn`` directly.

.. autoclass:: NestedColumn

.. rubric:: Attributes

.. autosummary::

    NestedColumn.col_names
    NestedColumn.nrows
    NestedColumn.ncols
    NestedColumn.nbytes
    NestedColumn.cbytes
    NestedColumn.cratio
    NestedColumn.info

.. autoproperty:: NestedColumn.col_names
.. autoproperty:: NestedColumn.nrows
.. autoproperty:: NestedColumn.ncols
.. autoproperty:: NestedColumn.nbytes
.. autoproperty:: NestedColumn.cbytes
.. autoproperty:: NestedColumn.cratio
.. autoproperty:: NestedColumn.info


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

.. autoclass:: string
.. autoclass:: bytes
.. autofunction:: vlstring
.. autofunction:: vlbytes

Array, encoded, and compound specs
----------------------------------

.. autosummary::

    ndarray
    dictionary
    struct
    list
    object

.. autofunction:: ndarray
.. autofunction:: dictionary
.. autofunction:: struct
.. autofunction:: list
.. autofunction:: object

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

.. _NestedFields:

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

Accessing an intermediate prefix such as ``t.trip`` or ``t.trip.begin`` returns
a :class:`~blosc2.NestedColumn` that groups all descendant leaves and reports
aggregate metadata via :attr:`~blosc2.NestedColumn.info`; a leaf such as
``t.trip.begin.lon`` returns a :class:`Column`.

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
