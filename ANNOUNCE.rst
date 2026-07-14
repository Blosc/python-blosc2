Announcing Python-Blosc2 4.8.0
==============================

We are happy to announce this release, which brings support for
**sharing containers safely across processes** — opt-in file locking for
``SChunk``/``NDArray``/``EmbedStore``/``DictStore``, atomic archive
replacement, and readers that follow another process's writes — plus a
couple of data-loss and data-correctness bug fixes worth upgrading for even
if you don't touch any of the new locking API.

The main highlights are:

- **Cross-process locking**: a new ``locking`` storage parameter (and the
  ``BLOSC_LOCKING`` env var to enable it fleet-wide) serializes accesses to an
  on-disk ``SChunk``/``NDArray``/``EmbedStore``/``DictStore`` against other
  handles and processes via a small sidecar lock file. ``holding_lock()``
  holds the exclusive lock across several operations and now auto-refreshes
  the handle right after acquiring it, so a decision made inside the block
  never acts on a stale in-memory read.

- **Cross-process writes for ``EmbedStore``/``DictStore``** (``.b2d``): under
  locking, one process can add or remove keys while another has the store
  open, and the other side's next lookup sees the change — no need to
  reopen the store.

- **Atomic ``.b2z`` archives**: writing a ``DictStore.to_b2z()`` file (which
  ``TreeStore`` also uses) now swaps the new file in atomically. A process
  reading the archive concurrently always gets either the complete old
  version or the complete new one — never a partially-written file from a
  save that's still in progress. This needs no locking on the reader's side.

- **Growth-SWMR**: a reader ``NDArray`` handle opened before a ``resize()``
  made through another handle follows the new shape on its next data access,
  or via the new explicit ``NDArray.refresh()`` / ``SChunk.refresh()``.

- **Two bug fixes worth knowing about**: ``NDArray.append()`` could silently
  delete another writer's just-appended data under concurrent growth (fixed
  by refreshing the cached shape before computing the resize target); and
  ``detect_aligned_chunks()`` could silently return the wrong chunk's data
  for an aligned slice on arrays whose shape isn't a multiple of the chunk
  shape (a floor-division bug that undercounted the chunk grid).

- New user guide page,
  `Sharing containers across processes
  <https://www.blosc.org/python-blosc2/guides/sharing_across_processes.html>`_,
  covering all of the above plus the caveats (NFS, ``mmap_mode``, Windows
  in-use-file rename).

A quick taste — hold the lock across a read-modify-write from two processes::

    with ndarr.holding_lock():
        ndarr[:] = ndarr[:] + 1   # atomic w.r.t. other locked handles

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
