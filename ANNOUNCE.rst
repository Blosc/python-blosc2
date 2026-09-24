Announcing Python-Blosc2 4.14.0
===============================

Python-Blosc2 4.14.0 is a major feature release introducing lazy remote
columnar tables (``RemoteCTable``), a unified ``RemoteObject`` architecture,
native remote HDF5 indexing with direct byte-range reads (dropping Kerchunk and
Zarr dependencies for HDF5), native PyTables table interoperability, remote
column indexes, ListArray V2 with recursive nesting and membership predicates,
nested remote stores in ``TreeStore``, the bundled C-Blosc2 3.3.5 upgrade, and
critical bug fixes including partition contiguity verification for large arrays.

- **Remote columnar tables (``RemoteCTable``) and unified ``RemoteObject``.**
  Columnar tables can now be opened lazily over HTTP, S3, or any fsspec-supported
  filesystem without full downloads. The unified ``RemoteObject`` base provides
  consistent cache policies (memory, disk, none), shared process caching via
  ``blosc2.open(url, cache_dir=..., shared_cache=True)``, attributes, traffic
  accounting, and portable reference exports (``table.save("ref.b2z")``).
  Multi-threaded chunk reads bounded by ``max_concurrency`` deliver high-throughput
  streaming.

- **Native remote HDF5 indexing and direct range reads.** Remote HDF5 access has
  been redesigned with a native metadata index and direct fsspec range reads,
  completely removing runtime dependencies on ``kerchunk``, ``zarr``, and
  ``numcodecs`` for HDF5. Uncompressed, deflate/gzip, shuffle, and Blosc2 chunks
  are decoded directly, while unsupported filters fall back cleanly to ``h5py``.
  Index snapshots are cached in ``.hdf5-index.b2`` sidecars and carrier metadata
  to eliminate repeated remote scans on warm opens. The ``refs`` parameter is
  replaced by ``hdf5_index``, with legacy reference maps detected and cleanly
  rejected.

- **PyTables interoperability.** Local and remote PyTables tables can be opened
  directly as ``RemoteCTable`` via ``blosc2.open("file.h5::/path/to/table")``
  without installing PyTables. Existing PyTables column indexes (including FULL
  indexes) are discovered and imported as Blosc2 OPSI indexes, enabling accelerated
  queries directly over existing datasets.

- **Remote column indexing and query acceleration.** Remote queries on
  ``RemoteCTable`` leverage pre-computed column indexes (OPSI, FULL, SUMMARY)
  without fetching full columns. Added ``kind="membership"`` index support to
  accelerate scalar-list membership queries (``contains()``, ``overlaps()``).

- **ListArray V2.** List elements can be nullable, and ``ListSpec`` values can be
  nested recursively to arbitrary depths. Added ``contains()`` and ``overlaps()``
  row predicates for ListArray and CTable list columns. New schemas default to
  ``batch_rows=2048`` for balanced chunking and compression.

- **Nested remote stores in ``TreeStore``.** ``TreeStore`` can embed and persist
  references to ``RemoteStore`` instances, enabling composite hierarchical stores
  spanning local arrays and remote endpoints with shared cache ownership.

- **C-Blosc2 3.3.5 and critical bug fixes.** Bundled C-Blosc2 is updated to 3.3.5.
  Fixed a critical bug (#723) in ``are_partitions_behaved()`` where non-contiguous
  block partitions silently corrupted ``asarray()`` copies for arrays larger than
  16 MB. Also resolved double-closing of fsspec sessions, hardened NumPy and
  object attributes in msgpack vlmeta payloads, and improved Windows path and
  refresh handling.

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
