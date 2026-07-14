Announcing Python-Blosc2 4.8.1
==============================

This is a maintenance release with a solid batch of bug fixes — including
use-after-free hazards around zero-copy cframes, wrong-chunk deletion with
negative-step slices, and inconsistent ``DictStore`` overwrite semantics —
plus read-only memory mapping for ``CTable`` stores and a documentation
restructuring with a new Optimization Tips section.

The main highlights are:

- **Read-only mmap for ``CTable`` stores**: ``CTable.open()`` gains an
  ``mmap_mode="r"`` parameter, mirroring ``blosc2.open()``. All members of a
  read-only store — scalar, list, varlen and dictionary columns alike — are
  read from mapped pages; for ``.b2z`` archives, in place inside the single
  mapped container file. With several concurrent readers on one file this
  pays off quickly: 2.5x/4.4x/4.5x faster wall time for 1/4/8 readers in
  our benchmark.

- **Zero-copy cframe fix**: ``schunk_from_cframe()`` /
  ``ndarray_from_cframe()`` with ``copy=False`` (the default) returned
  objects pointing into the caller's bytes buffer without keeping it alive,
  so a temporary cframe could be reclaimed under the live object, corrupting
  reads. The buffer is now pinned on the returned object.

- **More correctness fixes**: negative-step slice deletion in
  ``BatchArray``/``ObjectArray`` removed the wrong chunks; ``DictStore``
  overwrite semantics depended on value size (now uniformly dict-like);
  ``stack()``/``vecdot()`` shape inference was off for negative axes;
  chunked ``matmul()`` mishandled broadcast batch dims; and
  ``ListArray.extend_arrow()`` could reorder unflushed rows.

- **Faster ``.b2z``/``.b2d`` opens**: a builtin-shadowing bug made store
  detection in ``blosc2.open()`` silently recurse ~250 times on every open.

- **Docs restructuring**, with a new `Optimization tips
  <https://www.blosc.org/python-blosc2/guides/optimization_tips.html>`_
  section, including tips on grouping related data into a single
  memory-mapped ``.b2z`` file and on using ``mmap_mode="r"`` with many
  concurrent readers.

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
