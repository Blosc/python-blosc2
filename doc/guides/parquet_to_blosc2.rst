Parquet to Blosc2 Walkthrough
=============================

The ``parquet-to-blosc2`` CLI converts Parquet files to Blosc2 columnar
table stores (``.b2z`` compact or ``.b2d`` sparse) and can export them
back to Parquet.

Prerequisites
-------------

``pyarrow`` is required for all Parquet operations.  Install it alongside
the optional ``parquet`` extras:

.. code-block:: console

    pip install "blosc2[parquet]"

Step 1 — Create a sample Parquet file
--------------------------------------

Run the snippet below once to produce ``sample.parquet`` with three columns
(``id``, ``name``, ``score``):

.. code-block:: python

    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table(
        {
            "id": pa.array([1, 2, 3, 4], type=pa.int64()),
            "name": pa.array(["Alice", "Bob", "Charlie", "David"], type=pa.string()),
            "score": pa.array([85.5, 90.0, 78.2, 95.4], type=pa.float64()),
        }
    )

    pq.write_table(table, "sample.parquet")

Step 2 — Import to a compact ``.b2z`` store
---------------------------------------------

The default output format is ``.b2z`` — a single-file zip-backed store:

.. code-block:: console

    parquet-to-blosc2 sample.parquet sample.b2z --overwrite

Step 3 — Import to a sparse ``.b2d`` store
--------------------------------------------

Use the ``.b2d`` extension to produce a directory-backed (sparse) store:

.. code-block:: console

    parquet-to-blosc2 sample.parquet sample.b2d --overwrite

Step 4 — Fixed-width string import
------------------------------------

By default, string columns are stored as variable-length ``utf8`` columns
(Arrow-style offsets + bytes; falls back to ``vlstring`` on NumPy < 2.0).
Pass ``--fixed-str-maxlen`` to pre-scan strings and store columns whose
maximum character length fits within the given limit as fixed-width,
indexable strings:

.. code-block:: console

    parquet-to-blosc2 sample.parquet sample_fixed.b2z --fixed-str-maxlen 16 --overwrite

Step 5 — Custom chunk and block layout
----------------------------------------

Override the automatic chunk and block sizes (in rows) chosen by
``blosc2.compute_chunks_blocks()``.  Smaller blocks improve cache locality;
larger chunks reduce per-chunk overhead:

.. code-block:: console

    parquet-to-blosc2 sample.parquet sample_layout.b2z --chunks 1000 --blocks 100 --overwrite

Step 6 — Disable the summary index
-------------------------------------

By default the tool builds a SUMMARY index for eligible scalar columns on
close.  The index costs less than 0.1 % of column size and accelerates
``WHERE`` queries.  Disable it with ``--no-summary-index`` when you do not
need indexed queries:

.. code-block:: console

    parquet-to-blosc2 sample.parquet sample_no_index.b2z --no-summary-index --overwrite

Step 7 — Export back to Parquet
---------------------------------

Use ``--export`` to convert a Blosc2 store back to a Parquet file:

.. code-block:: console

    parquet-to-blosc2 --export sample.b2z exported.parquet --overwrite

Step 8 — Spot-check the exported file
---------------------------------------

Verify the round-trip with a quick Python comparison:

.. code-block:: python

    import pyarrow.parquet as pq

    original = pq.read_table("sample.parquet")
    exported = pq.read_table("exported.parquet")

    # Compare row counts and column names
    assert original.num_rows == exported.num_rows, "row count mismatch"
    assert original.column_names == exported.column_names, "column name mismatch"

    # Compare values column by column
    for col in original.column_names:
        assert original[col].equals(exported[col]), f"value mismatch in column '{col}'"

    print("Round-trip check passed — all columns match.")
