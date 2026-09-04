.. _RemoteProxy:

RemoteProxy
===========

``RemoteProxy`` is a persistable reference to one remote B2ND array. It accepts
an fsspec URL or a Caterva2 :ref:`URLPath` and separates the portable reference
from any runtime data cache.

The default policy is :attr:`blosc2.CachePolicy.NONE`: each operation reads the
remote data it needs and no fetched data is retained afterwards. Saving the
object writes only its source descriptor and array geometry, never fetched data
or credentials.

.. code-block:: python

    remote = blosc2.RemoteProxy(
        "s3://public-bucket/dataset.b2nd",
        cache_policy=blosc2.CachePolicy.NONE,
    )
    remote.save("dataset-reference.b2nd")

Runtime caching is available through :attr:`blosc2.CachePolicy.MEMORY` and
:attr:`blosc2.CachePolicy.DISK`. Memory caches retain at most 256 MiB of
compressed payload by default. Disk caches are unlimited by default, but both
can take an explicit ``max_cache_bytes`` bound. The bound is enforced after an
operation completes and therefore does not limit its temporary working set or
returned NumPy array.

.. code-block:: python

    remote = blosc2.RemoteProxy(
        "s3://public-bucket/dataset.b2nd",
        cache_policy=blosc2.CachePolicy.DISK,
        cache_path="dataset-cache.b2nd",
        max_cache_bytes=2 * 2**30,
    )

Regardless of its runtime policy, :meth:`RemoteProxy.save
<blosc2.RemoteProxy.save>` and :meth:`RemoteProxy.to_cframe
<blosc2.RemoteProxy.to_cframe>` produce a reference-only object that reopens
with :attr:`blosc2.CachePolicy.NONE`. Local cache paths and authentication data are not
serialized.

To cache again after reopening a reference, opt into a runtime policy when
constructing a new proxy from its source:

.. code-block:: python

    reference = blosc2.open("dataset-reference.b2nd")
    cached = blosc2.RemoteProxy(
        reference.urlpath,
        cache_policy=blosc2.CachePolicy.MEMORY,
    )

.. warning::

    Resolving an uploaded remote reference makes the receiving server perform
    an outbound request. Caterva2 installations must reject these references by
    default unless administrators configure allowed protocols, destinations,
    credentials, redirects, and resource limits. Client-side URL checks are not
    a server security boundary.

.. autoclass:: blosc2.RemoteProxy

    .. automethod:: __init__
    .. automethod:: __getitem__
    .. automethod:: get_chunk
    .. automethod:: aget_chunk
    .. automethod:: save
    .. automethod:: to_cframe
    .. autoattribute:: shape
    .. autoattribute:: dtype
    .. autoattribute:: ndim
    .. autoattribute:: chunks
    .. autoattribute:: blocks
    .. autoattribute:: cparams
    .. autoattribute:: nbytes
    .. autoattribute:: info
    .. autoattribute:: cache_bytes
    .. autoattribute:: cache_policy
    .. autoattribute:: max_cache_bytes
    .. autoattribute:: cache_path
    .. autoattribute:: cache_status
    .. autoattribute:: source
    .. autoattribute:: traffic
    .. autoattribute:: urlpath

CachePolicy
-----------

.. autoclass:: blosc2.CachePolicy
    :members:
