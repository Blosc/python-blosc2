Announcing Python-Blosc2 4.6.0
==============================

We are happy to announce this release, which sharpens the **columnar / query
side** of blosc2: zero-copy sorted views, queries over string (dictionary)
columns, a more flexible and faster ``group_by``, and an ``b2view`` terminal
browser that can now group, sort, and plot interactively.

The main highlights are:

- **Zero-copy sorted views**: ``CTable.sort_by(..., view=True)`` returns a
  lightweight sorted view that shares the parent's column data and gathers rows
  on demand — no whole-table copy.  Sorting on a fully indexed column streams
  straight from the index, so reading a sorted slice of a huge (on-disk) table
  is as easy as ``t.sort_by("col", view=True)[:10]``.

- **Queries over string columns**: ``where`` expressions now work on
  dictionary-encoded (string) columns, including membership tests such as
  ``'"Acme" in company'``, filtering categorical text without decoding the whole
  column.

- **Smarter, faster group_by**: ``group_by(...).agg()`` accepts flexible
  aggregation specs and explicit output names (pandas-style), a new tri-state
  ``sort=`` (``None``/``True``/``False``) sorts only when cheap, the last
  grouping is memoized, and ``min``/``max``/``argmin``/``argmax`` are
  accelerated from per-block index summaries instead of decompressing data.

- **b2view grows up**: interactive **group-by** (``G``, including float keys),
  **sort-by-column** (``S``, zero-copy via the index), and much **better plots**
  — bars for categorical keys, lines/stems for numeric keys, hi-res variants for
  every plot type, and ``--max`` to maximize a panel.  ``b2view`` is now an
  **opt-in extra** (``pip install "blosc2[tui]"``), so a plain
  ``pip install blosc2`` stays lean.

- **Fixes & maintenance**: the bundled **C-Blosc2 is upgraded to 3.1.5**, the
  open-file cache now validates cached handles against the file's fingerprint (so
  a file changed underneath an open handle is never served stale), plus
  compatibility with **NumPy 2.5**.

A quick taste — grab a table and explore it::

    $ pip install "blosc2[tui]" --upgrade
    $ b2view --download --panel data

Press ``G`` to group, ``S`` to sort, and ``p`` to plot a column — all without
decompressing anything you do not look at.

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
