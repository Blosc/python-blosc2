Announcing Python-Blosc2 4.2.0
===============================

We are happy to announce Python-Blosc2 4.2.0.  This is a large feature
release, with the new compressed columnar table container, ``CTable``, as the
main development.

``CTable`` brings typed, compressed, column-oriented tables to Python-Blosc2.
It supports persistent ``.b2d`` and ``.b2z`` storage, schema-driven columns,
nullable data, variable-length strings/bytes and object columns, computed
columns, table views, mutations, sorting, filtering and persistent indexes.  It
also includes Arrow, Parquet and CSV interoperability, plus a new
``parquet-to-blosc2`` command-line utility.

For a deeper introduction to CTable and its motivation, see our recent blog
post:

https://blosc.org/posts/ctable-blosc2-columnar-table/

Other highlights in 4.2.0 include:

- A new indexing subsystem for NDArrays and CTables, including persistent
  sidecar indexes, expression indexes, sorted iteration and query caching.
- New structured serialization facilities for persisted ``C2Array``,
  ``LazyExpr`` and DSL ``LazyUDF`` objects, plus ``blosc2.Ref`` and
  ``blosc2.load()``.
- New schema helpers such as ``blosc2.struct()`` and ``blosc2.object()``.
- Object/ListArray improvements for variable-length and general object data.
- Faster and lower-memory ``fromiter()`` construction, improved ``BatchArray``
  defaults and continued linalg/matmul optimizations.
- Many documentation, tutorial, example and benchmark additions.
- Numerous fixes for Windows mmap/file-locking behavior, Python 3.14 GC/thread
  interactions, ``.b2z`` persistence, indexed queries and NumPy compatibility.

You can think of Python-Blosc2 4.x as an extension of NumPy/numexpr that:

- Can deal with NDArray compressed objects using first-class codecs & filters.
- Performs many kinds of math expressions, including reductions, indexing...
- Supports multi-threading and SIMD acceleration (via numexpr/miniexpr).
- Can operate with data from other libraries (like PyTables, h5py, Zarr, Dask, etc).
- Supports NumPy ufunc mechanism: mix and match NumPy and Blosc2 computations.
- Integrates with Numba and Cython via UDFs (User Defined Functions).
- Adheres to modern array API standard conventions (https://data-apis.org/array-api/).
- Can perform linear algebra operations (like ``blosc2.tensordot()``).
- Can store and query compressed columnar tables via ``blosc2.CTable``.

Install it with::

    pip install blosc2 --upgrade   # if you prefer wheels
    conda install -c conda-forge python-blosc2 mkl  # if you prefer conda and MKL

For more info, you can have a look at the release notes in:

https://github.com/Blosc/python-blosc2/releases

Small CTable example::

    import blosc2

    table = blosc2.CTable.from_parquet("measurements.parquet", urlpath="measurements.b2z")
    table.create_index("station_id")

    hot = table.where("temperature > 30")
    print(hot.head())

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
