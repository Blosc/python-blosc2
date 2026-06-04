Announcing Python-Blosc2 4.4.2
==============================

We are happy to announce this feature and maintenance release that promotes
DSL kernels to first-class CTable computed columns, adds a convenient new
column-assignment API, speeds up bulk NDArray writes, and fixes several
correctness issues.

The main highlights are:

- **DSL kernels as first-class CTable columns**: ``@blosc2.jit``-decorated
  functions can now back virtual computed columns, stored generated columns,
  and ``where()`` filter predicates directly — in addition to the existing
  string-expression form.  Columns survive save/open round-trips via
  persisted source.

- **New ``t["col"] = arr`` assignment**: a clean shorthand for overwriting all
  live rows of a column.  Accepts any array-like, including ``blosc2.NDArray``.

- **Chunked NDArray writes**: passing a ``blosc2.NDArray`` to ``extend()`` or
  ``col[:] = arr`` now decompresses one chunk at a time instead of loading
  the entire array into memory, keeping peak RSS bounded for large columns.

- **``BLOSC_ME_JIT`` full override**: the environment variable now takes
  unconditional priority over both ``jit=`` and ``jit_backend=`` keyword
  arguments, making backend switching from the command line effortless.

- **Correctness fixes**: a ``None == None`` guard bug that could corrupt rows
  when writing to a view-backed column via the NDArray fast path; a missing
  view guard in the new ``__setitem__``; and the fast path being silently
  disabled for disk-opened tables have all been fixed.

A quick example of the new DSL computed column API::

    import blosc2
    import dataclasses
    import numpy as np

    @dataclasses.dataclass
    class Row:
        price: float = blosc2.field(blosc2.float64())
        qty:   int   = blosc2.field(blosc2.int64())

    @blosc2.dsl_kernel
    def revenue(price, qty):
        return price * qty

    t = blosc2.CTable(Row, new_data=[(1.5, 10), (2.0, 5), (3.0, 3)])
    t.add_computed_column("revenue", revenue, inputs=["price", "qty"])
    print(t["revenue"][:])   # array([15., 10.,  9.])

    # Column assignment with a blosc2.NDArray — written chunk-by-chunk
    new_prices = blosc2.array([1.0, 2.5, 4.0])
    t["price"] = new_prices

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
