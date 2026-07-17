Announcing Python-Blosc2 4.9.1
==============================

This is a hot-fix release for the Arrow interop work introduced in 4.9.0 —
one real performance regression and one clearer error message, both in
``CTable``.

- **Faster dictionary-column Arrow export**: ``CTable.iter_arrow_batches()``
  (and therefore ``to_arrow()`` and the Arrow PyCapsule interchange,
  ``__arrow_c_stream__``) was recomputing the full live-row-position array
  from scratch on every batch, for every dictionary-encoded string column —
  an ``O(n_rows)`` scan repeated ``O(n_rows / batch_size)`` times. It's now
  computed once per export call instead. 6-14x faster export for
  dictionary columns on a 1M-row benchmark.

- **Worth knowing regardless of this fix**: the Arrow PyCapsule protocol
  (``__arrow_c_stream__``) has no column-projection pushdown — a consumer
  that only needs two columns (DuckDB, pyarrow, Polars, pandas) still
  triggers export of *every* column in the table, since the raw Arrow C
  Stream interface has no way to say "I only need these." Use
  ``CTable.select([...])`` to project down to the columns you actually
  need before handing the table off, especially if any column holds an
  expensive nested/list type::

      sub = t.select(["company", "fare"])
      duckdb.sql("SELECT company, avg(sub.fare) FROM sub GROUP BY company").show()

- **Clearer error on ``mode="a"``**: opening a ``CTable`` with
  ``mode="a"`` at a path that doesn't exist yet now raises a
  ``FileNotFoundError`` explaining that ``mode="a"`` opens an existing
  table (use ``mode="w"`` to create one), instead of silently creating a
  new, empty table.

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
