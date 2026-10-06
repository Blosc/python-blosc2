.. _RemoteRepository:

RemoteRepository
================

``RemoteRepository`` is a read-only browsing facade returned by
``blosc2.open("https://host/base")`` when a Caterva2-compatible service has zero
or multiple accessible roots. A single root instead opens directly.

The facade subclasses :ref:`RemoteStore` for hierarchy consumers such as b2view.
``keys()`` lists roots without opening them; ``repository["@public"]`` returns
an independent group handle. Direct descendant lookup is supported. Closing
the repository leaves previously returned child handles usable. Alias handles
from ``repository[""]`` share root owners and lifetime accounting.

Cache policy and allowance are per root. ``cache_bytes`` and ``metadata_bytes``
sum opened roots; ``traffic`` counts their reads and excludes the initial roots
probe. Source descriptors contain no authentication token. Root owners bind the
authentication context at repository creation and use existing isolated cache
identities.

Repository persistence, materialization, refresh, and root-level shared sparse
cache opening are intentionally unsupported. Select a specific root for those
operations. ``get_info`` at a root name reports the roots registry metadata;
select/open that root for its actual source attributes. No synthetic empty path
is sent to the server's info endpoint.

.. autoclass:: blosc2.RemoteRepository
    :members: keys, get_info, close, source, attrs, cache_policy, max_cache_bytes, cache_bytes, metadata_bytes, traffic
