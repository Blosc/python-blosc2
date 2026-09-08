.. _RemoteProxy:

RemoteProxy
===========

``RemoteProxy`` is a persistable proxy for one remote B2ND, B2Z, Zarr, or HDF5 array. It
accepts an fsspec URL or a Caterva2 :ref:`URLPath`. With disk caching enabled,
its B2ND carrier is both the portable descriptor and the bounded compressed-data
cache.

The default policy is :attr:`blosc2.CachePolicy.NONE`: each operation reads the
remote data it needs and no fetched data is retained afterwards. Saving such an
object writes only its source descriptor and array geometry.

.. code-block:: python

    remote = blosc2.RemoteProxy(
        "s3://public-bucket/dataset.b2nd",
        cache_policy=blosc2.CachePolicy.NONE,
    )
    remote.save("dataset-reference.b2nd")

A Caterva2 dataset is named with :class:`blosc2.URLPath` rather than an fsspec
URL:

.. code-block:: python

    remote = blosc2.RemoteProxy(
        blosc2.URLPath(
            "@public/dataset.b2nd",
            urlbase="https://example.org/caterva2",
        )
    )

By default, ``RemoteProxy`` assumes its source is immutable and skips remote
identity checks before reads. For a replaceable single-file or Caterva2 source,
pass ``assume_immutable=False`` to refresh its identity and invalidate stale
cached data before each operation.

Zarr URLs use a different contract: a ``.zarr`` path component selects
:ref:`ZarrNDSource`, or pass ``source_format="zarr"`` for a suffix-free path.
The URL names one array, including its path inside a hierarchy. Zarr sources are
assumed immutable for the lifetime of every cache; replacing data beneath the
same URL may mix stale and new chunks. Use a new URL or replace the cache when
publishing a new dataset. Mutable Zarr stores are not supported.

.. code-block:: python

    remote = blosc2.open(
        "s3://public-bucket/hierarchy.zarr/d0/a1",
        lazy=True,
        storage_options={"anon": True},
    )

HDF5 URLs (``.h5``, ``.hdf5``, or ``source_format="hdf5"``) select :ref:`HDF5NDSource`.
Datasets within an HDF5 container can be specified via standard slash syntax (``.../file.h5/dataset``),
the double-colon separator (``.../file.h5::dataset``), or the ``dataset="dataset"`` argument.
Zarr containers similarly accept all three forms (``.../file.zarr/dataset``, ``.../file.zarr::dataset``,
or ``dataset="dataset"``).
HDF5 datasets are read through ``kerchunk`` metadata pre-indexing. Like Zarr, HDF5 sources
are assumed immutable (``assume_immutable=True``); mutable HDF5 sources are not supported.
Pre-computed kerchunk references can be supplied via ``refs`` to avoid remote scanning.

.. code-block:: python

    remote = blosc2.open(
        "s3://public-bucket/hierarchy.h5/d0/d1/a2",
        lazy=True,
        storage_options={"profile": "blosc2"},
    )
    # Equivalent to "s3://public-bucket/hierarchy.h5::d0/d1/a2"
    # or blosc2.open("s3://public-bucket/hierarchy.h5", lazy=True, dataset="d0/d1/a2", ...)

B2Z archives
------------

An external NDArray inside an immutable ``.b2z`` archive can be selected using
the same three addressing forms:

.. code-block:: python

    remote = blosc2.open(
        "s3://public-bucket/hierarchy.b2z::/d0/a3",
        lazy=True,
        storage_options={"anon": True},
    )
    values = remote[:10, 0, :5]
    # Also accepts hierarchy.b2z/d0/a3 or dataset="d0/a3".

Use ``source_format="b2z"`` for suffix-free archive URLs. The dataset is a logical
tree key without the member's ``.b2nd`` suffix. The native Blosc2 reader preserves
source chunks, blocks, dtype, and compression parameters; no kerchunk, Zarr, or
HDF5 dependencies are needed. Install the fsspec extra and the protocol backend.

Opening reads the ZIP directory and selected member's headers. Directory cost
scales with archive member count. Subsequent reads fetch native chunks or blocks
by byte range; repeated cache hits perform no remote reads. Reopening a saved
carrier rereads archive/frame metadata and resolves the member offset afresh.

Only unencrypted, ``ZIP_STORED`` external NDArray members are supported. Groups,
embedded leaves inside ``embed.b2e``, other leaf types, and compressed ZIP members
are unsupported. Archives must remain immutable; replacing an archive requires
replacing its cache. Authorized B2Z sparse attachment and Caterva2 federation are
not supported in this version.

.. autoclass:: blosc2.B2ZNDSource

Caching and persistence
-----------------------

Ephemeral in-memory caching is available through :attr:`blosc2.CachePolicy.MEMORY`.
Fetched chunks are kept in RAM, bounded by a finite 256 MiB compressed-payload limit by default
(customizable via ``max_cache_bytes``) with automatic LRU eviction.

Persistent caching is available through :attr:`blosc2.CachePolicy.DISK`.
Disk caches have a finite 256 MiB compressed-payload bound by default and can
take an explicit ``max_cache_bytes`` bound, or ``max_cache_bytes=None`` for an
unbounded cache that never evicts chunks. When bounded, the limit is enforced after an
operation completes and therefore does not limit its temporary working set or
returned NumPy array.

.. code-block:: python

    remote = blosc2.RemoteProxy(
        "s3://public-bucket/dataset.b2nd",
        cache_policy=blosc2.CachePolicy.DISK,
        cache_path="dataset-cache.b2nd",
        max_cache_bytes=2 * 2**30,
    )

When opening a remote array via :func:`blosc2.open` with ``lazy=True``, a :class:`RemoteProxy`
is always returned: specifying ``cache_dir`` or ``cache_path`` configures it with
:attr:`blosc2.CachePolicy.DISK`, while omitting them configures it with
:attr:`blosc2.CachePolicy.MEMORY`.

By default, :meth:`RemoteProxy.save <blosc2.RemoteProxy.save>` and
:meth:`RemoteProxy.to_cframe <blosc2.RemoteProxy.to_cframe>` include valid warm
chunks for DISK proxies; MEMORY proxies always export cold carriers.
Pass ``include_cache=False`` for a cold carrier without changing the
warm original. The cache policy and limit remain in both forms; local paths and
authentication data are not serialized.

Pass ``cache_policy=blosc2.CachePolicy.NONE`` (or another policy) to either
export method to produce a cold carrier with an explicit policy, leaving the
live proxy unchanged. Caterva2 servers accept persisted MEMORY carriers under
opt-in policy but execute them without retained caching (identical to NONE);
older Caterva2 servers reject MEMORY resolution. Use DISK for retained carrier
caching on Caterva2. Cold exports must not overwrite the live disk carrier.

``fetch()`` and ``afetch()`` prefetch and return the proxy. Eviction may discard
requested chunks; ``materialize(item)`` returns an independent complete NDArray.
The raw ``cache`` is incomplete storage for inspection, not a materialized array.

Reads and exports on one handle are serialized. Async methods use worker threads;
cancelling an await does not stop a running operation. Separate handles and
processes sharing a carrier need external locking. Unreadable cache files are
preserved and their opening errors are propagated.

Authentication supplied to a live Caterva2 source is deliberately omitted from
the carrier. Caterva2's first server implementation resolves public HTTPS
sources only; client credentials never travel with the proxy.

Open a disk-caching carrier in append mode to let misses populate that same
file. Read-only mode can use warm chunks but does not retain misses:

.. code-block:: python

    cached = blosc2.open("dataset-cache.b2nd", mode="a")
    cached[100:200]

.. warning::

    Resolving an uploaded remote reference makes the receiving server perform
    an outbound request. Caterva2 installations must reject these references by
    default unless administrators configure allowed protocols, destinations,
    credentials, redirects, and resource limits. Client-side URL checks are not
    a server security boundary.

.. autoclass:: blosc2.RemoteProxy

    .. automethod:: __init__
    .. automethod:: __getitem__
    .. automethod:: fetch
    .. automethod:: afetch
    .. automethod:: get_chunk
    .. automethod:: aget_chunk
    .. automethod:: save
    .. automethod:: materialize
    .. automethod:: to_cframe
    .. autoattribute:: shape
    .. autoattribute:: dtype
    .. autoattribute:: ndim
    .. autoattribute:: chunks
    .. autoattribute:: blocks
    .. autoattribute:: cparams
    .. autoattribute:: nbytes
    .. autoattribute:: info
    .. autoattribute:: cache
    .. autoattribute:: cache_bytes
    .. autoattribute:: cache_policy
    .. autoattribute:: max_cache_bytes
    .. autoattribute:: cache_path
    .. autoattribute:: cache_status
    .. autoattribute:: schunk
    .. autoattribute:: source
    .. autoattribute:: traffic
    .. autoattribute:: urlpath
    .. autoattribute:: dataset

CachePolicy
-----------

.. autoclass:: blosc2.CachePolicy
    :members:
