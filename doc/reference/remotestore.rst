.. _RemoteStore:

RemoteStore
===========

``RemoteStore`` discovers a read-only B2Z, Zarr or HDF5 hierarchy and returns
:ref:`RemoteArray` leaves. Groups and arrays share one source session: a B2Z
archive, an HDF5 reference map, or a Zarr store. Zarr listing remains lazy.

The default ``CachePolicy.MEMORY`` shares a 256 MiB allowance across all leaves.
Set ``max_cache_bytes`` to a positive integer to change it. ``CachePolicy.NONE``
retains no payload and rejects a limit. Store DISK caching is not implemented yet.
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
the owned archive/store wrappers. Transport connection pools remain managed by
fsspec. Operations on an explicitly closed handle raise ``RuntimeError``.
Standalone ``RemoteArray`` exports remain self-contained references, including
the HDF5 reference map when applicable.

.. autoclass:: blosc2.RemoteStore
    :members:
    :special-members: __getitem__, __iter__

.. autoclass:: blosc2.RemoteNode
    :members:
