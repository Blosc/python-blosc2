.. _ZarrNDSource:

ZarrNDSource
============

``ZarrNDSource`` exposes a Zarr v2 or v3 array through :ref:`ProxyNDSource`.
Zarr decodes each logical chunk and Blosc2 stores the converted compressed
chunk in the surrounding :ref:`Proxy` or :ref:`RemoteProxy` cache.

The source is assumed immutable for the cache lifetime. It supports fixed-size
boolean, integer, floating-point, and complex arrays. Scalar arrays, empty
dimensions, strings, objects, structured dtypes, and ZIP stores are not
supported. Concurrent reads temporarily hold decoded chunks and conversion
buffers in addition to the compressed cache.

Install local support with ``pip install "blosc2[zarr]"``. Remote stores also
need ``blosc2[fsspec]`` and the protocol driver, such as ``s3fs``.

.. autoclass:: blosc2.ZarrNDSource

    .. automethod:: __init__
    .. automethod:: get_chunk
