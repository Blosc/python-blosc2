Announcing Python-Blosc2 4.12.0
===============================

This release makes remote Blosc2 arrays faster and easier to use, from object
stores and plain HTTP servers to Caterva2.

- **Read and write through fsspec URLs.** ``blosc2.open()``, ``save_array()``
  and ``save_tensor()`` support URLs such as ``s3://``, ``gs://``, ``https://``
  and chained filesystems via the new ``blosc2[fsspec]`` extra.

- **Fetch only the bytes a slice needs.** Lazy remote proxies read individual
  compressed blocks, overlap requests and can reuse a validated on-disk cache.
  This cuts traffic, latency and peak memory for small remote slices.

- **Improved Caterva2 access.** Stored arrays and leaves inside ``.b2z``
  containers use HTTP byte ranges. Pre-sized remote arrays can also be filled
  concurrently with ``C2Array.update_chunk()`` and ``aupdate_chunk()``.

- **Lean UTF-8 index lookups.** FULL-index queries bisect the vocabulary on
  disk instead of materializing it, greatly reducing memory use for
  high-cardinality string columns.

- **Bundled C-Blosc2 3.3.3**, together with expanded remote-array documentation
  and benchmarks.

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
