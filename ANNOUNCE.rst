Announcing Python-Blosc2 4.9.0
==============================

Last year's big push was making ``NDArray`` a good ecosystem citizen — array
API compliance, protocols, interop. This release does the same for
``CTable``: instead of asking the tabular ecosystem to learn Blosc2's ways,
``CTable`` now speaks its protocols directly. Arrow tools can read and write
a ``CTable`` with zero glue code, a new string column type stores text in
Arrow's own layout, and ``engine=blosc2.jit`` runs correctly *inside*
pandas 3 itself. Compression doesn't have to mean an island — it can just be
a fast, compact layer underneath the tools you already use.

The main highlights are:

- **Arrow PyCapsule interchange**: ``CTable.__arrow_c_stream__`` lets
  pyarrow, DuckDB, Polars, and pandas >= 2.2 consume a ``CTable`` directly as
  a stream of record batches, with bounded memory — no ``to_arrow()`` copy
  step needed::

      import duckdb, blosc2
      t = blosc2.CTable.open("trips.b2z")
      duckdb.sql("SELECT company, avg(fare) FROM t GROUP BY company").show()

  ``CTable.from_arrow()`` now accepts any object implementing the same
  protocol on ingest too — a pyarrow Table, a Polars DataFrame, a Parquet
  reader — in addition to the existing ``(schema, batches)`` form.

- **``blosc2.utf8()``: string columns that speak Arrow's layout**. Instead of
  a blosc2-specific encoding, ``utf8()`` stores text exactly as Arrow does —
  int64 row offsets plus a UTF-8 byte blob — and reads back as NumPy
  ``StringDType``. It's the new recommended default for free text: 7-13x
  smaller uncompressed than fixed-width ``string()`` on high-cardinality
  text, with a full query surface (comparisons, ``where()``, ``sort_by``,
  ``group_by`` keys, ``fillna``, Arrow round-trip).

- **pandas engine, pandas 3 ready**: ``engine=blosc2.jit`` for
  ``DataFrame.apply`` now returns a properly indexed ``DataFrame``/``Series``
  under pandas 3's default ``raw=False``, and ``Series.map(func,
  engine=blosc2.jit)`` is implemented for the first time.

- **``CTable`` grows a pandas-style API**: ``CTable.assign()`` plus an
  unbound ``blosc2.col()`` for chaining —
  ``t.assign(profit=col("revenue") - col("cost"))[col("profit") > 0].sort_by("profit", ascending=False).head(10)``
  — and a real missing-data story: ``Column.fillna()``, ``CTable.dropna()``,
  and null-propagating arithmetic/comparisons on nullable columns (no more
  ``t[t.x < 0]`` silently matching null rows). ``group_by().agg()`` accepts
  UDF aggregations, and ``CTable.apply()`` runs a UDF across live rows.

- **Faster indexed reads**: ``NDArray.iter_sorted()``/``argsort()`` on a
  ``FULL``-indexed array reads the sidecar range directly instead of
  building the full permutation — ~52x faster and ~193x less memory for
  tail-style queries on a 20M-element array.

Install it with::

    pip install blosc2 --upgrade   # if you prefer wheels
    conda install -c conda-forge python-blosc2 mkl  # if you prefer conda and MKL

For more info, see the release notes at:

https://github.com/Blosc/python-blosc2/releases

What is Python-Blosc2?
----------------------

Python-Blosc2 is a high-performance compressor, compute engine, and format
for binary data containers that are portable and open-source. It comes with
a lazy expression engine allowing for complex calculations on compressed data,
whether stored in memory, on disk, or over the network (e.g., via
`Caterva2 <https://github.com/ironArray/Caterva2>`_).  It is especially
optimized for storing and retrieving data from N-dimensional arrays (`NDArray`)
and columnar tables (`CTable`), bringing a query/indexing layer too.  The main
use case is fast, compressed, out-of-core numerical data — especially when data
is too large to fit comfortably in RAM.

More info: https://www.blosc.org/python-blosc2/getting_started/overview.html


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
