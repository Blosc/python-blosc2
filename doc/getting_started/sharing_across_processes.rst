Sharing Containers Across Processes
====================================

On-disk Blosc2 containers (``SChunk``, ``NDArray``, ``EmbedStore``,
``DictStore``) can be shared by several processes, or several handles in the
same process, under two complementary mechanisms:

- **SWMR** (single writer, multiple readers) — always on, no configuration
  needed. One process writes, others read and follow along.
- **Locking** (this is what supports **multiple concurrent writers**, not
  just one) — opt-in, via ``locking=True`` or the ``BLOSC_LOCKING``
  environment variable. Serializes accesses with a sidecar lock file so
  several processes can safely write concurrently, or a reader can safely
  observe a writer mid-mutation.

Both are advisory: they coordinate cooperating Blosc2 handles, not arbitrary
processes touching the file. Neither works over a network filesystem (NFS).
Neither is a substitute for a transaction log, either: there is no
multi-step commit protocol in the container format itself, so a process
crashing mid-mutation can leave partial state in the *data* regardless of
whether locking is in use — see :ref:`Crash safety <SharingLocking>` below
for what locking does and does not give you there.

Two runnable, tested examples cover the common multiple-writers patterns end
to end: ``examples/ndarray/mwmr-mode.py`` (several processes writing disjoint
regions, and a read-modify-write counter that shows why ``holding_lock()``
is required for that case) and ``examples/ndarray/mwmr-enlarge.py``
(several processes concurrently growing the same array with
``append()``). Start there if you just want working code; the rest of this
page is the contract behind it.

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
HDF5-SWMR offers. See ``examples/ndarray/swmr-enlarge.py`` for this pattern
run for real across several reader processes, including how to tell a
"settled" (safely readable) region from a just-grown-but-not-yet-filled one,
and how to retry the occasional transient read error a reader can hit
racing the writer. ``examples/ndarray/swmr-enlarge-bars.py`` is the same
scenario rendered live with ``rich`` progress bars -- one per writer/reader,
each reporting its own throughput -- so the writer-leads-readers-follow
effect is visible instead of just asserted; run it in a real terminal.

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

**SWMR vs. locking, side by side.** ``examples/ndarray/swmr-enlarge-bars.py``
and ``examples/ndarray/locking-enlarge-bars.py`` run the identical
one-writer/three-readers growth scenario, rendered with live throughput
bars, one unlocked and one locked, so the trade-off is visible instead of
just asserted:

- The SWMR version's readers need a "settled vs. just-resized" trailing
  trick to avoid reading uninitialized rows, and a retry loop around every
  read to absorb the occasional transient error from racing the writer's
  in-progress metadata update (see "SWMR without locking" above for why
  even a "settled" region isn't immune — resolving *where* a chunk lives on
  disk can itself race, independently of whether that chunk's data is
  final). The locked version needs neither:
  wrapping each batch's resize-plus-fill in ``holding_lock()`` makes it
  atomic, so a locked reader's ``refresh()`` only ever shows a batch fully
  there or not there at all — trust it immediately, no lag, no retries.
- The two versions also disagree on how a reader knows the writer is
  *done*. The SWMR version's readers can't infer that from shape alone:
  reaching the final expected shape only proves the last ``resize()``
  landed, not that its fill did — the same gap the "settled" trick guards
  every other batch against, just with no following batch left to trail
  behind. So it needs an explicit signal, and a variable-length metalayer
  (``schunk.vlmeta``) is the one piece of SWMR-visible state built for
  exactly this: unlike fixed metalayers (``schunk.meta``, outside the
  re-sync contract entirely), a vlmeta lookup re-syncs on every access, the
  same as a data read or ``refresh()``. The locked version needs no such
  flag: ``holding_lock()`` already made resize-plus-fill atomic, so
  ``shape == expected_rows`` *is* proof the last batch is safe — see the
  ``verify_up_to`` / completion-check difference between the two example
  readers.
- That safety isn't free: the writer's exclusive lock and the readers'
  shared locks now genuinely serialize against each other, where the SWMR
  version has no coordination to wait on at all. On one dev machine, growing
  the same ~8 GB array with the same pacing measured ~2.1 GB/s combined
  throughput over 4.25s unlocked vs. ~1.9 GB/s over 4.63s locked — roughly a
  10% cost here for eliminating the read-error race entirely. The exact
  number depends heavily on read/write pacing and how many readers are
  contending for the lock; measure your own workload rather than trusting
  this one.

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

:meth:`NDArray.holding_lock() <blosc2.NDArray.holding_lock>` delegates to the
same method on the underlying schunk for the locking itself, and additionally
refreshes this handle's cached :attr:`shape <blosc2.NDArray.shape>` right
after the lock is acquired — so code that reads ``shape`` inside the block
(e.g. to decide a :meth:`resize` target) always sees the current on-disk
state, not a stale cache from before the lock was taken.

``holding_lock()`` is required for any read-modify-write across writers,
since a single indexed assignment reads the old value and writes the new one
as two separate locked operations:

.. code-block:: python

    # Two processes both doing `arr[i] += 1` need the increment itself
    # locked, not just each half of it:
    with arr.holding_lock():
        arr[i] = arr[i] + 1

See ``examples/ndarray/mwmr-mode.py`` for this pattern run for real across
several processes, including a direct comparison of the wrong (unlocked)
and right (``holding_lock()``-wrapped) versions so you can see the lost
updates happen.

The same applies to :meth:`NDArray.append() <blosc2.NDArray.append>`: it is
internally a refresh of the current length, a resize, and a slice write —
three steps, not one atomic operation. ``append()`` always refreshes its
cached length first, so when the *whole call* runs inside
``holding_lock()``, concurrent appends from other writers are picked up
correctly instead of being overwritten (each writer's batch lands, in full,
somewhere in the final array — never lost, torn, or duplicated). Without
``holding_lock()``, the refresh does not help: another writer can still grow
the array between the refresh and the resize, and the same race applies as
any read-modify-write above.

.. code-block:: python

    # Every writer must wrap the whole append in holding_lock(), or a
    # concurrent grower's data can be silently discarded:
    with arr.holding_lock():
        arr.append(new_rows)

See ``examples/ndarray/mwmr-enlarge.py`` for several processes appending
concurrently to the same array, then verifying every batch landed exactly
once with nothing lost, torn, or duplicated.

Per-operation atomicity: what counts as "one operation"
------------------------------------------------------------

Without ``holding_lock()``, two writers racing on overlapping regions
resolve last-writer-wins, at a granularity that depends on the API used:

- ``SChunk`` chunk updates (:meth:`SChunk.update_data
  <blosc2.SChunk.update_data>` and friends) are atomic per chunk — an
  overlapping multi-chunk write from two writers can interleave chunk by
  chunk.
- ``NDArray`` slice writes (``arr[...] = value``) are atomic for the *whole
  slice*, even when it spans several chunks — a locked reader never observes
  a half-applied slice write.

Fixed metalayers (``schunk.meta``) are outside the locking contract:
``schunk.meta[name] = value`` from one handle is not visible to another
handle's ``schunk.meta`` reads until that handle re-syncs some other way
(e.g. a data access). Use variable-length metalayers (``schunk.vlmeta``) if
cross-handle visibility of metadata updates matters — those poll for
staleness on every access.

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
      - multiple concurrent writers, atomic ops, no torn reads
    * - ``holding_lock()``
      - context manager on a locked handle
      - atomic multi-operation blocks
    * - ``.b2z`` snapshot
      - :meth:`DictStore.to_b2z() <blosc2.DictStore.to_b2z>`
      - always safe to share read-only, atomic replace on write

Not supported in either mechanism: network filesystems (NFS). Locking
additionally excludes ``mmap_mode`` and in-memory containers.
