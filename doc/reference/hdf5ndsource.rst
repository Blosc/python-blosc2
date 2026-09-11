.. _HDF5NDSource:

HDF5NDSource
============

``HDF5NDSource`` exposes an HDF5 dataset through :ref:`ProxyNDSource` using
``kerchunk`` metadata pre-indexing. Individual chunks are fetched on demand
and converted to Blosc2-compressed chunks stored in the surrounding
:ref:`Proxy` or :ref:`RemoteArray` cache.

The source is assumed immutable (``assume_immutable=True``). It supports fixed-size
boolean, integer, floating-point, complex, and fixed-length string arrays.
HDF5 filters such as Blosc2 (via ``hdf5plugin``), gzip, and uncompressed datasets
are supported.

Install local support with ``pip install "blosc2[hdf5]"``. Remote datasets also
need ``blosc2[fsspec]`` and the protocol driver, such as ``s3fs`` for S3.

.. autofunction:: blosc2.available_datasets

.. autoclass:: blosc2.HDF5NDSource

    .. automethod:: __init__
    .. automethod:: get_chunk
