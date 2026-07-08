Sharing Containers Across Processes
====================================

On-disk Blosc2 containers (``SChunk``, ``NDArray``, ``EmbedStore``,
``DictStore``) can be shared by several processes, or several handles in the
same process, under two complementary mechanisms:

- **SWMR** (single writer, multiple readers) — always on, no configuration
  needed. One process writes, others read and follow along.
- **Locking** — opt-in, via ``locking=True`` or the ``BLOSC_LOCKING``
  environment variable. Serializes accesses with a sidecar lock file so
  several processes can safely write, or a reader can safely observe a
  writer mid-mutation.

Both are advisory: they coordinate cooperating Blosc2 handles, not arbitrary
processes touching the file. Neither works over a network filesystem (NFS).

SWMR without locking
---------------------

A reader handle opened before a writer mutates a container does **not** see
the change through its cached view automatically — it re-syncs the next time
it *touches* the container, whether that's reading data, checking whether a
vlmeta key exists (``"name" in schunk.vlmeta``), or — for ``NDArray`` only —
an explicit :meth:`NDArray.refresh() <blosc2.NDArray.refresh>` call that
polls without reading any data:

.. code-block:: python

    import blosc2
    import numpy as np

    # Writer process
    a = blosc2.zeros((10, 10), urlpath="growing.b2nd", mode="w")

    # Reader process, opened before the writer grows the array
    reader = blosc2.open("growing.b2nd", mode="r")
    reader.shape  # (10, 10)

    # Writer grows and fills the array
    a.resize((20, 10))
    a[10:20, :] = np.arange(100).reshape(10, 10)

    # Reader: any data access re-syncs the cached shape first
    reader[15, :]  # reads the new row, no reopen needed
    reader.shape  # (20, 10)

    # ... or poll without touching data
    reader.refresh()  # True if it re-synced, False if already current

This is the classic HDF5-SWMR use case: a writer grows an array (or appends
schunk chunks) while readers keep up with the new extent. Consistency is
per-operation, not a whole-container snapshot — the same weak ordering
HDF5-SWMR offers.

Contract and limits:

- **Single writer.** Two writers mutating the same container without locking
  is not supported and can corrupt it.
- Staleness is detected from the on-disk container length, so growth (which
  appends chunks) is virtually always noticed. A mutation that leaves the
  length unchanged (e.g. updating a chunk in place, or a resize that shrinks
  within the last chunk) may go undetected until the next mutation that does
  change the length.
- Only shape changes are followed for ``NDArray``; a handle that changes
  ``ndim``, ``chunks`` or ``blocks`` through another handle makes readers
  raise instead of silently reading garbage.
- A reader racing a writer without locking can occasionally get a read
  error on a container mid-rewrite; retrying is the documented workaround
  (see :ref:`locking <SharingLocking>` if that is unacceptable).

Store classes (:class:`blosc2.EmbedStore`, :class:`blosc2.DictStore`) are
stricter without locking: they are single-process, single-writer, since
their key maps are cached in Python and are not re-synced without a
sidecar lock. Reopen the store to see mutations made elsewhere.

.. _SharingLocking:

Locking
-------

Enable cross-process locking on a container by passing ``locking=True`` (in
:class:`blosc2.Storage`, or directly to :class:`blosc2.SChunk`,
:func:`blosc2.open`, or the array constructors), or by setting the
``BLOSC_LOCKING`` environment variable, which enables it globally for every
on-disk container subsequently opened or created, without touching sources:

.. code-block:: python

    import os
    import blosc2

    # Per-handle
    schunk = blosc2.SChunk(chunksize=1_000_000, urlpath="shared.b2frame", locking=True)

    # Or fleet-wide, for a whole deployment
    os.environ["BLOSC_LOCKING"] = "1"

With locking, readers take a shared lock and writers an exclusive one,
against a small sidecar lock file next to the container (``.b2lock``).
Mutating operations become atomic to other locked handles, and a handle
whose view went stale re-syncs before its next operation completes — closing
the read-error race SWMR-without-locking has.

**The locking is advisory**: it only protects a container if *every* handle
that touches it enables it. A plain handle opened on a locked container
bypasses the coordination entirely.

Caveats:

- Not supported together with ``mmap_mode``: explicit ``locking=True`` with
  ``mmap_mode`` raises ``ValueError``. The ``BLOSC_LOCKING`` environment
  variable, being a global switch, is silently ignored for memory-mapped
  containers instead of raising.
- Not supported for in-memory containers (no ``urlpath``): ``locking=True``
  raises ``ValueError`` there too.
- Not supported on network filesystems (NFS) — the underlying ``flock``/
  ``LockFileEx`` primitives are unreliable there.
- Crash safety: locks are held via the OS (released automatically if a
  process dies while holding one), but a crash mid-mutation can still leave
  partial state in the *data* — see each store's notes below.

Atomic multi-operation blocks with ``holding_lock()``
-------------------------------------------------------

Each locked operation locks and unlocks the container individually by
default, so a sequence of operations is not atomic as a whole — another
process could interleave a read or write between them. Use
:meth:`SChunk.holding_lock() <blosc2.SChunk.holding_lock>` to hold the
exclusive lock across several operations, making the whole block atomic to
other handles:

.. code-block:: python

    with schunk.holding_lock():
        schunk.update_data(0, data0, copy=True)
        schunk.update_data(1, data1, copy=True)

Everything inside the block is serialized exclusively — including plain
reads through other locked handles — so keep it short. On a handle without
locking enabled, ``holding_lock()`` is a no-op.

The stores: cross-process guarantees
--------------------------------------

:class:`blosc2.EmbedStore` and :class:`blosc2.DictStore` build their
cross-process story on top of container locking:

- **Without locking**: single-process, single-writer. Key maps are cached in
  Python and not re-synced, so mutations from another handle are invisible
  until reopen, and concurrent writers can corrupt each other's entries.
- **With locking enabled on every handle**: an on-disk store can be shared
  across processes. Each mutation (the data write plus the key-map update)
  runs under one exclusive lock, and every access re-syncs the key maps, so
  readers follow keys added or removed elsewhere.

Accepted races, even under locking:

- **EmbedStore**: a crash between the data write and the map flush can leave
  unreachable bytes in the container. Harmless — reclaimed the next time the
  store is rewritten.
- **DictStore** (directory-backed, ``.b2d``): a reader holding a value whose
  key another process just deleted may get errors reading that value
  afterwards. A crash mid-mutation can leave a partial external file behind.

``.b2z`` archives need no locking at all: they are safe to share read-only
across any number of processes, and :meth:`DictStore.to_b2z()
<blosc2.DictStore.to_b2z>` (which also covers :class:`blosc2.TreeStore`,
built on ``DictStore``) replaces the target atomically — a temporary sibling
file is written and then moved onto the final path, so concurrent readers
always see either the old archive or the complete new one, never a torn
write. On Windows, the final replace fails if another process holds the
target file open.

Detecting mutation without re-reading data
---------------------------------------------

:attr:`SChunk.change_tick <blosc2.SChunk.change_tick>` is a counter bumped
whenever a handle re-syncs from a stale on-disk state (whether via locking's
generation counter or SWMR's length poll). Compare it before and after an
operation to know cheaply whether another handle mutated the container in
between, without needing to re-read or diff the data — this is how the store
classes above detect that their cached key maps need a re-sync.

Summary
-------

.. list-table::
    :widths: 25 35 40
    :header-rows: 1

    * - Mechanism
      - Enable
      - Guarantees
    * - SWMR (default)
      - always on
      - single writer, readers follow shape/length growth on next access
    * - Locking
      - ``locking=True`` or ``BLOSC_LOCKING``
      - multiple writers, atomic ops, no torn reads
    * - ``holding_lock()``
      - context manager on a locked handle
      - atomic multi-operation blocks
    * - ``.b2z`` snapshot
      - :meth:`DictStore.to_b2z() <blosc2.DictStore.to_b2z>`
      - always safe to share read-only, atomic replace on write

Not supported in either mechanism: network filesystems (NFS). Locking
additionally excludes ``mmap_mode`` and in-memory containers.
