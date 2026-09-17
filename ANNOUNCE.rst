Announcing Python-Blosc2 4.13.1
===============================

Python-Blosc2 4.13.1 is a maintenance and performance follow-up to 4.13.0,
refining the remote data access layer and expanding platform support. Lazy
access is now the default for remote arrays, local HDF5 files are read
directly via ``h5py`` without auxiliary dependencies, warm opens replay cached
bootstrap metadata to eliminate redundant network roundtrips, disk caches use
human-readable folder hierarchies, and official Windows ARM64 wheels are now
available.

- **Lazy access is the default for remote arrays.** ``blosc2.open()`` on remote
  ``.b2nd`` files or container datasets (HDF5, Zarr, B2Z) now returns a lazy
  ``RemoteArray`` without requiring an explicit ``lazy=True``. Dataset paths
  require lazy access and reject ``lazy=False`` with ``NotImplementedError``.
  For standalone array files (such as ``.b2nd``), ``cache_dir=`` keeps the
  default lazy access and persists fetched chunks. Pass ``lazy=False`` to
  download the complete container there instead. ``mmap_mode=`` or a nonzero
  ``offset=`` forces eager reading.

- **Direct local HDF5 reads via ``h5py``.** Local ``.h5``/``.hdf5`` datasets
  are now accessed directly through ``h5py``, eliminating the need for
  ``kerchunk``, ``zarr``, or ``fsspec`` when reading local files. Chunked
  datasets preserve their native HDF5 chunk layout, while contiguous datasets
  automatically receive optimal Blosc2 cache chunks. Explicit ``refs=``
  arguments continue to select the kerchunk reference reader.

- **Faster HTTP discovery and instant warm opens.**

  * *Cold HTTP discovery*: Initial HTTP opens for B2Z archives retrieve the ZIP
    directory tail and object identity within a single bounded range request,
    halving network round-trips.
  * *Warm reopens*: Reopening cached B2Z, Zarr, or HDF5 sources replays
    persisted bootstrap metadata directly from the carrier, completely
    bypassing remote discovery. Sibling HDF5 datasets under a shared
    ``cache_dir=`` reuse a single on-disk reference snapshot.
  * *Small member prefetching*: B2Z members up to 64 KiB are fetched in full on
    open (header, chunks, and trailing metadata), populating the standard chunk
    cache with full quota tracking and LRU eviction.

- **Human-readable disk cache paths.** Persistent caches under ``cache_dir=``
  now mirror the source filename and dataset hierarchy (e.g.,
  ``hierarchy.b2z--03cc6a2f9314/d0/a1.b2nd``) with a 12-character identity
  fingerprint derived from the source URL and storage options. Old hash-only
  cache directories are ignored and can be safely deleted to reclaim space.

- **Windows ARM64 wheels and test stability.** Added official build recipes and
  CI pipelines producing native Windows on ARM64 (``win_arm64``) wheels. Test
  suite execution is accelerated with logical core utilization in pytest-xdist,
  per-test doctest workspace isolation, and test deadlock diagnostics.

- **Bug fixes and robustness:**

  * Embedded frames with nonzero byte offsets open directly even when the file
    name looks like a container.
  * ``file://`` HDF5 URLs with ``::`` dataset separators survive path conversion
    on Windows.
  * ``RemoteArray.info`` and ``str()`` mask sensitive credentials in signed URLs,
    and ``info`` works reliably on local B2Z sources.
  * ``load_tensor()`` explicitly requests eager access, avoiding unexpected lazy
    intermediates for remote paths.
  * Fixed decoding of HDF5 datasets compressed with the Blosc2 filter (such as
    via ``hdf5plugin``) when read through kerchunk, properly handling multi-chunk
    super-chunk frames without an ``AttributeError``.
  * HDF5 reference snapshot publishing is safely guarded, fixing a crash on
    Windows drive-letter paths when opening local h5py sources with a disk cache.
  * Attaching a sparse runtime cache to a read-only legacy B2Z carrier safely
    rebuilds bootstrap metadata in memory without attempting disk writes.

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
