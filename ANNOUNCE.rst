Announcing Python-Blosc2 4.13.1
===============================

Python-Blosc2 4.13.1 is a remote-access follow-up to 4.13.0.  Lazy access is
now the default for known remote arrays, local HDF5 files are read directly
with ``h5py``, and warm opens of B2Z, Zarr, and HDF5 sources replay a cached
bootstrap instead of re-discovering the remote object.

- **Lazy access is the default for known remote arrays.** ``blosc2.open()`` on
  a remote ``.b2nd`` file or a dataset path (HDF5, Zarr, B2Z) now returns a
  lazy ``RemoteArray`` without an explicit ``lazy=True``.  ``lazy=False``
  still requests eager access, and dataset paths reject it explicitly.  Pass
  ``cache_dir=`` (or ``mmap_mode=``, ``offset=``) without ``lazy`` to keep the
  historical eager localization for single-file containers.

- **Local HDF5 files without kerchunk, Zarr, or fsspec.** A local
  ``.h5``/``.hdf5`` dataset is read through ``h5py``; chunked datasets keep
  their HDF5 chunk layout and contiguous ones get automatically chosen Blosc2
  cache chunks.  Explicit ``refs=`` still selects the kerchunk reader.

- **Warm opens reuse a cached bootstrap.**  B2Z persists the ZIP tail and
  frame header (one HTTP range request also carries the object identity),
  Zarr persists the metadata read, and HDF5 persists the kerchunk reference
  map; sibling HDF5 datasets under one ``cache_dir=`` share the reference
  snapshot.  Small B2Z members (up to 64 KiB) are fetched whole on open, with
  their chunks counted and evictable like any other cached payload.

- **Readable disk caches.**  ``cache_dir=`` caches now keep the source
  basename and dataset hierarchy in the path (``data.zarr--a39d11ca41f0/d0/a``)
  with a short identity fingerprint.  Old hash-only entries are neither reused
  nor deleted; remove them to reclaim space.

- **Bug fixes.**  Embedded frames with an offset open directly even when the
  file name looks like a container; ``file://`` HDF5 URLs with ``::`` dataset
  paths survive on Windows; ``info``/``str()`` no longer expose signed-URL
  credentials; ``load_tensor()`` forces eager access; and the
  ``numcodecs.Blosc2`` HDF5 filter decodes whole super-chunk frames.

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
