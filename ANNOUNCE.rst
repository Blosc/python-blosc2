Announcing Python-Blosc2 4.14.1
===============================

Python-Blosc2 4.14.1 is a security and feature release introducing safe
persisted-object deserialization by default, Caterva2 remote stores and tables,
compressed slicing for ``RemoteArray``, and performance optimizations for
expression gather operations.

- **Safe persisted-object deserialization by default.** Persisted-input APIs
  (``blosc2.open()``, ``blosc2.load()``, ``blosc2.from_cframe()``, CFrame
  constructors, store traversal, and CTable variable-length/batch columns) now
  default to ``deserialize="safe"``. Safe mode accepts passive MessagePack
  values but rejects embedded objects, references, remote/source descriptors,
  proxies, and lazy recipes before reconstruction. Trusted callers wishing to
  deserialize rich values must pass ``deserialize="full"`` explicitly; blocked
  values raise the new public ``blosc2.UnsafeDeserializationError``.

- **Caterva2 remote stores and tables.** Caterva2 groups and dataset tables can
  now be opened via ``blosc2.open()`` as ``RemoteStore`` and ``RemoteCTable``
  respectively. Caterva2 tables support on-demand reading, bounded row slices,
  column projection (``select()``), iteration, and materialization. Large row
  slices and materialization fetch data in bounded batches to limit peak memory
  usage, with ``materialize(urlpath=...)`` streaming batches directly to disk.
  Authentication tokens are bound at open time and disk caches are partitioned
  per token for multi-tenant isolation.

- **Compressed slicing for ``RemoteArray``.** Added ``RemoteArray.slice()`` to
  extract a slice selection directly as an independent compressed ``NDArray``
  without materializing an uncompressed NumPy array intermediate. Custom
  compression parameters (``cparams``) can be passed directly, and fetching,
  construction, and cache eviction are serialized into a single atomic operation
  allowing slices larger than the cache quota.

- **Expression gather performance optimization.** Avoided zero-initializing
  full NumPy input blocks in ``miniexpr`` gather operations, as full blocks are
  overwritten byte-for-byte by raw NumPy gather (#728). Only partial edge blocks
  retain zero-padding. Contributed by Johnny Kao (@Johnny-Kao).

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
