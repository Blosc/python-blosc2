.. _RemoteProxy:

RemoteProxy
===========

``RemoteProxy`` is a persistable proxy for one remote B2ND array. It accepts an
fsspec URL or a Caterva2 :ref:`URLPath`. With disk caching enabled, its B2ND
carrier is both the portable descriptor and the bounded compressed-data cache.

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

References are floating: before each data operation, ``RemoteProxy`` checks the
source identity and verifies that shape, dtype, chunks, and blocks still match
the captured geometry. A replacement with different geometry is rejected;
cached disk data is invalidated when the source identity moves.

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

CachePolicy
-----------

.. autoclass:: blosc2.CachePolicy
    :members:
