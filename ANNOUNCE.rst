Announcing Python-Blosc2 4.17.0
===============================

Python-Blosc2 4.17.0 introduces a new storage format for saved Menudet
expressions, giving users a way to adopt the format ahead of Python-Blosc2 5
without changing execution engines.

- **Menudet 0.2 for new kernels.** New source supports optional dtype
  annotations. Saved Menudet expressions and new Menudet columns in a
  ``CTable`` use structured MessagePack recipes (v2).

- **Existing data stays readable.** Opening a legacy v1 JSON record doesn't
  rewrite it. Explicitly resaving a legacy Menudet expression, or using
  ``CTable.save()`` to save a table copy, writes v2 while preserving the
  legacy kernel's language version and binding contract. Explicit Menudet
  JSON interchange remains supported.

**For shared stores, upgrade all readers before writers start saving Menudet
entries with 4.17.** Older readers may not be able to open v2 entries. Plain
arrays are unaffected by this storage change.

Install it with::

    pip install blosc2 --upgrade   # if you prefer wheels
    conda install -c conda-forge python-blosc2 mkl  # if you prefer conda and MKL

For more info, see the release notes at:

https://github.com/Blosc/python-blosc2/releases/tag/v4.17.0

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
