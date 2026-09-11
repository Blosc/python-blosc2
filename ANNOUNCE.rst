Announcing Python-Blosc2 4.13.0
===============================

Python-Blosc2 4.13.0 turns remote data access into a comprehensive, portable
remote data layer. It introduces bounded in-memory and persistent disk caching
for remote arrays, multi-dataset hierarchy discovery across B2Z, Zarr, and HDF5
containers over fsspec, portable reference snapshots, the ``.attrs`` metadata
interface, interactive remote browsing in ``b2view``, and the ``b2nd-to-zarr``
CLI converter.

Note: remote cache protocol should be considered somewhat experimental until it
has seen more real S3/HTTP usage; feedback is very welcome!

- **Unified RemoteArray and bounded caching.** ``blosc2.open(..., lazy=True)``
  now returns a ``RemoteArray`` with in-memory caching by default (256 MiB quota,
  automatic LRU eviction). Persistent caching on disk is supported via
  ``cache_dir`` or ``cache_path``, and stateless streaming via ``CachePolicy.NONE``.
  ``cache_storage`` is deprecated in favor of ``cache_dir``.

- **Hierarchy discovery and shared caching with RemoteStore.** Discover, navigate,
  and slice multi-dataset hierarchies in ``.b2z``, ``.zarr`` (v2/v3), and HDF5
  (``.h5``) containers over fsspec (HTTP/HTTPS, S3, GCS). All leaves share a
  single cache budget with cross-dataset LRU eviction.

- **Portable reference exports and snapshots.** Export portable references and
  snapshots (``.b2nd`` carriers and ``.b2z`` store archives) with optional warm
  cached data, and reopen them seamlessly with ``blosc2.open()``.

- **On-demand adapters for B2Z, Zarr, and HDF5.** Native chunk/block range reads
  for B2Z archives (``B2ZNDSource``), Zarr v2/v3 datasets (``ZarrNDSource``), and
  remote HDF5 datasets via kerchunk (``HDF5NDSource``). Includes a new
  ``b2nd-to-zarr`` CLI converter.

- **Recommended ``.attrs`` metadata interface.** User-defined metadata across
  arrays, containers, and proxy sources is now accessible via ``.attrs``
  (aliased to ``.vlmeta``, which remains fully supported).

- **Interactive remote browsing in ``b2view``.** Explore remote containers and
  array slices interactively in the terminal with the new ``--cache-dir`` option
  for persistent caching across sessions.

- **Enhanced AST shape inference.** Extended shape inferencer for subscripts,
  slices, builtins, and common array methods in ``LazyExpr``.

- **Bundled C-Blosc2 3.3.4**, alongside CI stability improvements and test
  deadlock diagnostics.

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
