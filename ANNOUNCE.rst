Announcing Python-Blosc2 4.3.0
===============================

We are happy to announce Python-Blosc2 4.3.0.  This release deepens the
``CTable`` container with three major additions: **N-dimensional columns**,
**group-by aggregation**, and **dictionary/categorical column types**.

N-dimensional columns let each cell in a CTable hold a full compressed
multidimensional array — ideal for embedding vectors, image patches,
time-series windows or any per-row tensor payload.  ndarray columns support
CSV/DataFrame round-trips, nullable semantics, and are automatically detected
when importing from pandas.

Group-by brings SQL-style ``GROUP BY`` directly to CTables::

    by_city = t.group_by("city", sort=True)
    by_city.agg({"sales": ["sum", "mean"], "qty": "sum"})

Multi-key groupings, filtered aggregates (``where=`` pushdown), and persistent
output (``urlpath=``) are all supported.  Behind the scenes, Cython-accelerated
kernels deliver dramatic speedups — ~25× for float keys, ~8× for integer keys —
backed by dense-indexing and general-purpose hash-table paths.

Also, ``DictionarySpec`` introduces dictionary-encoded (categorical) columns
that store compact integer codes mapped to a shared string dictionary, giving
both compact storage and accelerated equality/membership queries.  Dictionary
columns work transparently in ``where`` clauses and nested dotted-name
expressions.

Other highlights in 4.3.0 include:

- **Nested columns and field-name escaping**: Columns from Arrow/Parquet struct
  hierarchies are flattened into physical leaf columns under hierarchical
  ``_cols`` storage paths, with logical dotted-name access preserved.
  Round-trip fidelity is maintained for nested schemas, and literal ``.`` / ``/``
  in field names are automatically escaped.

- **Parquet import improvements**: Arrow serializer is now the default;
  nested columns are always separated; new ``--progress``/``--max-rows``/
  ``--timestamp-unit``/``--float-trunc-prec`` options for
  ``parquet-to-blosc2`` CLI; and a ``list_serializer`` parameter for fine-tuning
  list-type column storage.

- **Inline CTable support in TreeStore**: CTables can now be stored as items
  inside a ``TreeStore``, enabling hierarchical containers that mix arrays
  and tables.

- **Performance wins**: ``CTable.open()`` is faster thanks to lazy ``nrows``
  and deferred column metainfo loading.  Scalar and small-slice access paths
  have been overhauled.  ``import blosc2`` is leaner via late-import
  optimizations for heavy optional dependencies.

- **New tutorials and examples**: Group-by, nested fields, dictionary columns,
  TreeStore–CTable integration, and dedicated benchmarks for group-by,
  nested-filter, and Parquet round-trips.

- **Fixes**: Null/NaN sentinel normalization, empty aggregate results,
  generated-column safety, miniexpr bundling, and more.

- **Updated C-Blosc2** to version 3.0.3 (latest).

You can think of Python-Blosc2 4.x as an extension of NumPy/numexpr that:

- Can deal with NDArray compressed objects using first-class codecs & filters.
- Performs many kinds of math expressions, including reductions, indexing...
- Supports multi-threading and SIMD acceleration (via numexpr/miniexpr).
- Can operate with data from other libraries (like PyTables, h5py, Zarr, Dask, etc).
- Supports NumPy ufunc mechanism: mix and match NumPy and Blosc2 computations.
- Integrates with Numba and Cython via UDFs (User Defined Functions).
- Adheres to modern array API standard conventions (https://data-apis.org/array-api/).
- Can perform linear algebra operations (like ``blosc2.tensordot()``).
- Can store and query compressed columnar tables via ``blosc2.CTable``.

Install it with::

    pip install blosc2 --upgrade   # if you prefer wheels
    conda install -c conda-forge python-blosc2 mkl  # if you prefer conda and MKL

For more info, you can have a look at the release notes in:

https://github.com/Blosc/python-blosc2/releases

Small CTable group-by example::

    import blosc2
    from dataclasses import dataclass

    @dataclass
    class Order:
        city: str = blosc2.field(blosc2.string(max_length=16))
        product: str = blosc2.field(blosc2.string(max_length=16))
        qty: int = blosc2.field(blosc2.int32())
        price: float = blosc2.field(blosc2.float64(nullable=True), default=0.0)

    # Create a table with 200 random orders
    t = blosc2.CTable(Order, new_data=orders)

    # Group by city: total and average price per city in one call
    print(t.group_by("city", sort=True).agg({"price": ["sum", "mean"], "qty": "sum"}))

    # Multi-key: city + product breakdown
    print(t.group_by(["city", "product"], sort=True).agg({"qty": "sum", "price": "mean"}))

    # Filtered: only Widget orders, grouped by city
    print(t.where(t.product == "Widget").group_by("city", sort=True).agg({"qty": "sum"}))

Sources repository
------------------

The sources and documentation are managed through GitHub services at:

https://github.com/Blosc/python-blosc2

Python-Blosc2 is distributed using the BSD license, see
https://github.com/Blosc/python-blosc2/blob/main/LICENSE.txt
for details.

Mastodon feed
-------------

Follow https://fosstodon.org/@Blosc2 to get informed about the latest
developments.

Enjoy!

- Blosc Development Team
  Compress Better, Compute Bigger
