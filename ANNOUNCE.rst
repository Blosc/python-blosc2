Announcing Python-Blosc2 4.10.0
==============================

This is the string-support release: string expressions and DSL kernels now
run on miniexpr, ``utf8()`` and ``dictionary()`` columns gain full indexing
and comparisons, and NumPy's ``StringDType`` is understood by the array
constructors. Alongside, slicing with plain keys is up to 1.7x faster and a
new ``blosc2.random`` module brings chunk-parallel, NumPy-quality random
constructors.

- **String expressions and DSL kernels over strings.** Concatenation,
  ``lower``/``upper``/``strip``/``replace``/``substr``/``split_part`` and
  friends now run on miniexpr over fixed-width ``<Un`` and bytes ``S``
  arrays, producing string results sized by miniexpr itself. ``utf8()``
  columns can be queried in expression form
  (``t.where("name == 'x'")``), with scalar comparisons 5-6x faster via a
  raw-byte scan, and new ``blosc2.utf8_array()`` builds variable-length
  arrays directly.

- **Full indexing for string columns.** ``create_index()`` now works on
  ``utf8()`` and ``dictionary()`` columns via alphabetical ranks —
  ``sort_by`` drops from 424 ms to 7 ms at 1M rows — and scalar comparisons
  are served from the index.

- **New ``blosc2.random`` module**: 42 of NumPy's 43 ``Generator`` methods,
  each chunk generated in parallel with its own seeded ``PCG64`` stream
  (~3x faster than the NumPy path on 100M elements).

- **Slicing up to 1.7x faster.** Plain slice/int keys skip ndindex's general
  machinery (it was 43% of a scattered-read loop); strided steps, ellipsis
  and fancy indexing still use it.

- **String plumbing**: ``from_utf8()``/``to_utf8()`` conversions,
  ``CTable.add_column(values=)``, ``Column.assign()`` on variable-length
  columns, ``StringDType`` dispatch in the array constructors, and DSL
  operands can be native NumPy arrays or pandas ``Series``.

- **Important fixes**: string column indexes returning zero rows at the
  default column width, ``SChunk`` slices for typesizes above 255 bytes
  (upstream, via C-Blosc2 3.3.1), scalar bools in tuple keys now matching
  NumPy, and a batch of miniexpr correctness fixes.

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
