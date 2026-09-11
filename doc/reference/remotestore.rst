.. _RemoteStore:

RemoteStore
===========

``RemoteStore`` discovers a read-only B2Z, Zarr or HDF5 hierarchy and returns
:ref:`RemoteArray` leaves. Groups and arrays share one source session: a B2Z
archive, an HDF5 reference map, or a Zarr store. Zarr listing remains lazy.

The default ``CachePolicy.MEMORY`` shares a 256 MiB allowance across all leaves.
Set ``max_cache_bytes`` to a positive integer to change it. ``CachePolicy.NONE``
retains no payload and rejects a limit. Passing ``cache_dir`` selects DISK when
the policy is omitted; an explicit policy must agree with the cache location.
DISK accepts ``max_cache_bytes=None`` for unbounded retention.
Sources must be immutable. Generic ``blosc2.open(..., lazy=True, dataset=...)``
continues to open a single array.

.. code-block:: python

    with blosc2.RemoteStore(
        "https://host/data.h5", cache_policy=blosc2.CachePolicy.NONE
    ) as store:
        print(store.keys())  # immediate children
        info = store.get_info("experiment")  # metadata only
        group = store["experiment"]
        array = group["temperature"]
        values = array[:100]
        result = (array + 273.15).compute()

    # Returned handles own their source lifetime independently.
    values = array[:100]
    group.close()
    array.close()

Paths are relative to the selected group. ``store["a/b"]`` and
``store["a"]["b"]`` use the same source reader. Each lookup returns an independent
handle. With NONE, repeated reads fetch again; no payload cache is retained.
``dataset="a"`` or a subgroup suffix in the URL selects a group at construction.
An array root must be opened with ``RemoteArray`` instead.

``keys()`` and ``get_info()`` do not construct leaf readers or payload caches.
Discovery can read archive prefixes, attributes and small HDF5 inline values.
``get_info()`` returns a ``RemoteNode`` with a relative path, a kind (``group``,
``ndarray`` or ``unsupported``), known attributes and a diagnostic. Unknown array
attributes are ``None``; open the array to retrieve them. Unsupported nodes stay
discoverable and raise ``NotImplementedError`` when selected. Missing paths raise
``KeyError``.

Group ``attrs`` mappings are read-only. ``source`` returns the credential-free
container descriptor and full group path. ``traffic`` is one shared source
counter across all views, including discovery; do not add counts from aliases.
It counts source reads, not connections or every HTTP request: HEAD requests and
failed Zarr probes are excluded. ``cache_bytes`` counts retained compressed chunks
and partial-block duplicates across the store, without double-counting aliases.
The least recently used native chunk is evicted across all leaves when necessary.
Oversized reads return their result before eviction; returned arrays and temporary
buffers are outside the allowance. Closing a leaf preserves its warm cache while
other session handles remain open. With NONE, ``cache_bytes`` is zero and
``max_cache_bytes`` is ``None``.

Closing a handle, or exiting its context, releases its ownership. Existing child
handles remain usable until closed or garbage-collected. The last handle closes
the owned archive/store wrappers and private HTTP/S3 transport sessions. Operations on an explicitly closed handle raise ``RuntimeError``.
Standalone ``RemoteArray`` exports remain self-contained references, including
the HDF5 reference map when applicable.

``b2view`` uses ``RemoteStore`` for remote hierarchies with one 64 MiB MEMORY
allowance, and ``RemoteArray`` for selected or directly opened leaves. Switching
selection releases the UI handle while retaining the store's warm chunks.

For persistent shared caching:

.. code-block:: python

    with blosc2.RemoteStore("https://host/data.h5", cache_dir="remote-cache") as store:
        with store["experiment/temperature"] as array:
            values = array[:100]
        print(store.cache_bytes, store.metadata_bytes)
        store.refresh()  # rebuild discovery; old child handles become stale

Each source and selected root has its own directory under ``cache_dir``. One
owner holds an exclusive operating-system lock until its last dependent handle
closes. Conflicting opens raise ``RuntimeError``, including in other processes;
the operating system releases the lock after a process exits or crashes.

Reopening restores all previously created leaf caches and trims them against the
new aggregate allowance before returning. The manifest preserves B2Z directory
and bounded metadata reads, one HDF5 reference map, and lazily discovered Zarr
metadata. Metadata reads can contain small inline values or incidental bytes in
bounded prefixes; they are separate from evictable payload. ``metadata_bytes``
is the encoded manifest size, and is zero without a disk manifest. Credentials
and storage options must be supplied again at runtime.

The allowance measures compressed retained payload, not filesystem allocation.
Container overhead and old generations are outside it; obsolete generation
files are removed on the next exclusive reopen. Manifests survive payload
eviction. Store backing files are private implementation details; use array
``save()`` or ``to_cframe()`` for standalone exports, optionally with
``include_cache=True``.

Sources must remain immutable until explicit root ``refresh()``. Refresh builds
replacement discovery before publishing a new generation; failed discovery or
publication leaves existing handles valid. Successful refresh makes existing
child groups and arrays stale, requiring fresh lookups. Corrupt manifests and
B2Z validator mismatches raise an error; use a fresh cache directory if the store
cannot be opened. Fully offline reopening is not promised. The lifetime lock
uses POSIX flock or Windows byte locking; Windows execution remains a CI check.

Shared sparse runtime caches
----------------------------

Services and multiple local processes can use ``RemoteStore.with_sparse_cache``
to keep simultaneous handles to the same private runtime cache:

.. code-block:: python

    with blosc2.RemoteStore.with_sparse_cache(
        "https://host/data.b2z", "shared-runtime", max_cache_bytes=64 << 20
    ) as store:
        with store["experiment/temperature"] as array:
            values = array[:100]
        hit, values = store.read_cached("experiment/temperature", slice(0, 100))

This mode stores leaf payload in sparse RemoteArray caches. Each operation
acquires a store-wide OS lock, reloads discovery and leaf accounting, and applies
one aggregate payload allowance. Handles may coexist across processes, while
operations within a store serialize. All users of that directory must use the
shared constructor. A process-local memory cache or the ordinary exclusive
``cache_dir`` constructor must not write to it.

Manifests and generation pointers are published atomically. An interrupted
operation causes the next owner to discard the disposable payload generation;
remote sources are not contacted by offline trimming or manifest recovery.
Refresh publishes a new generation and makes child handles in other processes
stale. Reopen a store handle after another process refreshes it.

``read_cached`` returns ``(False, None)`` on a miss without fetching missing
payload. ``trim_sparse_cache`` trims leaves offline with a bounded chunk count.
Its first implementation evicts in leaf order; the live aggregate coordinator
handles eviction during ordinary reads. Allocated storage and old generations
are separate from the compressed-payload allowance and belong to the service's
storage accounting and lifecycle management.

The shared constructor accepts an authorized ``_filesystem`` and validation
callbacks for server use; these runtime objects are never persisted. A portable
``carrier`` archive can seed a new cache once, with source stamp and geometry
checks. ``save`` exports ordinary portable warm/cold archives. Private sparse
directories are not portable store artifacts. This protocol targets processes
sharing a local filesystem, not distributed or network-filesystem ownership.

.. autoclass:: blosc2.RemoteStore
    :members:
    :special-members: __getitem__, __iter__

.. autoclass:: blosc2.RemoteNode
    :members:
