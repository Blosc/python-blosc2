Announcing Python-Blosc2 4.4.3
==============================

We are happy to announce this maintenance release that makes CTable
cold-start, printing, querying and groupby noticeably faster, trims the
memory footprint of ``import blosc2``, adds raw-storage access for columns,
and exposes the new JPEG 2000 codec plugins.

The main highlights are:

- **Faster CTable cold-start**: views created by ``select()`` now open
  columns lazily — only the columns actually read are loaded from storage —
  and queries only open the SUMMARY indexes referenced by the predicate,
  instead of every indexed column on a wide persistent table.

- **Faster printing and groupby**: table rendering now performs a single
  combined sparse read per column (instead of ~6), and groupby takes the
  dense fast path for float key columns whose values are integral and fit a
  compact non-negative range.

- **Lighter imports**: the on-disk chunk prefetcher no longer uses asyncio,
  so ``import blosc2`` skips ~30 asyncio modules and saves ~3 MB of memory.
  A latent prefetcher deadlock on early iterator close was fixed as well.

- **New ``Column.raw`` accessor**: the underlying storage container
  (``NDArray``, ``ListArray``, …) as a blosc2-native compressed object —
  unlike ``col[...]`` reads, which always materialize NumPy arrays.  Useful
  as a lazy-expression operand and for storage introspection.

- **J2K and HTJ2K codecs**: ``blosc2.Codec.J2K`` and ``blosc2.Codec.HTJ2K``
  expose the IDs for the new JPEG 2000 plugins (``pip install blosc2-j2k``
  / ``pip install blosc2-htj2k``).  Also, C-Blosc2 has been updated to 3.1.3.

- **Fixes**: ``--float-trunc-prec`` in the ``parquet_to_blosc2`` CLI now
  propagates to nested columns; unsupported computed-column expressions are
  rejected early with an actionable error; and opening a ``.b2z``/``.b2d``
  store in read mode no longer creates a temporary directory.

A quick example of the new ``Column.raw`` accessor::

    import blosc2
    import dataclasses

    @dataclasses.dataclass
    class Row:
        price: float = blosc2.field(blosc2.float64())
        qty:   int   = blosc2.field(blosc2.int64())

    t = blosc2.CTable(Row, new_data=[(1.5, 10), (2.0, 5), (3.0, 3)])

    # The raw storage container, as a blosc2-native compressed object.
    # It is over-allocated to chunk capacity, so slice to the live row count.
    raw = t["price"].raw      # a blosc2.NDArray
    print(raw[:len(t)])         # [1.5 2.  3. ]

    # Usable directly as a lazy-expression operand, without decompressing
    expr = raw * 2.0
    print(expr[:len(t)])        # [3. 4. 6.]

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
