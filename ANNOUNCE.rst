Announcing Python-Blosc2 4.4.1
==============================

We are happy to announce Python-Blosc2 4.4.1, a feature release that brings an
interactive TUI data viewer, automatic SUMMARY indexes for fast WHERE queries,
chunk-aligned Arrow/Parquet imports, expanded ``where()`` acceleration, and a
range of CTable ergonomics and performance improvements.

The main highlights are:

- **New ``b2view`` interactive viewer**: a terminal-based viewer for all blosc2
  containers (``NDArray``, ``CTable``, ``SChunk``, ``BatchArray``, …), launched
  with ``b2view <file>`` or ``blosc2.b2view()``.  Supports full 1-D/2-D/N-D
  browsing, ``CTable`` row navigation, a vlmeta pane, and keyboard shortcuts.

- **Automatic SUMMARY indexes**: when a ``CTable`` is closed after writing,
  SUMMARY indexes (per-block min/max) are built by default for all eligible
  scalar columns.  They are accumulated *incrementally* during writes so the
  close step adds almost no extra cost.  At query time, a block-skip prefilter
  uses these bitmaps to skip blocks that cannot satisfy the WHERE predicate,
  reducing decompression work for selective queries.

- **Chunk-aligned Arrow/Parquet imports**: fixed-size columns are now written on
  a shared chunk/block grid and incoming batches are buffered to exact chunk
  boundaries, so every chunk is compressed exactly once.  Dictionary columns are
  imported in bulk.  A new ``--reduce-mem`` CLI flag caps Arrow read-batch size
  for memory-constrained nested imports.

- **``where()`` and miniexpr acceleration**: single- and two-argument ``where``
  calls are now dispatched directly to miniexpr, avoiding numexpr overhead.
  Sparse boolean masks trigger a fast gather path, and a new pre-check skips
  per-chunk numexpr setup when the condition is trivially true or false.

- **``CTable.copy()`` enhancements**: a new C-level bulk copy path
  (``chunk_copy()``) transfers pre-compressed chunks without
  serialization/recompression.  ``copy()`` now accepts ``chunks``, ``blocks``,
  and ``cparams`` overrides; the ``parquet-to-blosc2`` CLI gains ``--chunks``
  and ``--blocks`` flags.

- **``sort_by()`` on views is now lazy**: sorting a filtered view returns a
  position-reordered view whose columns are read in sorted order without a full
  materialization pass.

- **``context manager`` support for ``blosc2.open()``**: all objects returned by
  ``blosc2.open()`` now support the ``with`` statement for clean flush-and-close
  semantics.

- **``NestedColumn`` public class**: the dotted-column accessor is now a proper
  public class with aggregate metadata (``nbytes``, ``cbytes``, ``cratio``) and
  a structured ``.info`` report.

- **Python 3.10 dropped**: Python 3.11 is now the minimum supported version.

A small example showing the new SUMMARY index benefit::

    import blosc2

    # Create a table and let SUMMARY indexes be built automatically on close
    t = blosc2.CTable(Row, urlpath="my_table.b2d", mode="w")
    t.extend(data)
    t.close()   # SUMMARY indexes built here

    # Re-open and run a selective WHERE query — block skipping kicks in
    t = blosc2.open("my_table.b2d")
    result = t.where(t.value > 0.99)
    print(result[:])

Install it with::

    pip install blosc2 --upgrade   # if you prefer wheels
    conda install -c conda-forge python-blosc2 mkl  # if you prefer conda and MKL

For more info, see the release notes at:

https://github.com/Blosc/python-blosc2/releases

What is Python-Blosc2?
----------------------

Python-Blosc2 is a high-performance compressor, compute engine, and format
for binary data containers that are portable, and open-source. It comes with
a lazy expression engine allowing for complex calculations on compressed data,
whether stored in memory, on disk, or over the network (e.g., via
`Caterva2 <https://github.com/ironArray/Caterva2>`_).  It is especially
optimized for storing and retrieving data from N-dimensional arrays (`NDArray`),
columnar tables (`CTable`), and a query/indexing layer.  The main use case is
fast, compressed, out-of-core numerical data — especially when data is too
large to fit comfortably in RAM.

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
