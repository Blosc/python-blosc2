.. _HDF5NDSource:

HDF5NDSource
============

``HDF5NDSource`` exposes an HDF5 dataset through :ref:`ProxyNDSource`.
Local files use ``h5py`` directly. Remote files are scanned once with ``h5py``
to build a native byte-range index, which can also be supplied through ``hdf5_index=``.
Individual chunks are fetched on demand
and converted to Blosc2-compressed chunks stored in the surrounding
:ref:`Proxy` or :ref:`RemoteArray` cache.

The source is assumed immutable (``assume_immutable=True``). It supports fixed-size
boolean, integer, floating-point, complex, and fixed-length string arrays.
Uncompressed, gzip/deflate, shuffle, and Blosc2 chunks use direct range reads.
Other filter pipelines fall back to retained ``h5py`` dataset reads;
``hdf5plugin`` enables its additional registered filters.

Local reads require ``h5py``; ``hdf5plugin`` enables additional HDF5 filters.
Install the full HDF5 support with ``pip install "blosc2[hdf5]"``. Remote datasets also
need ``blosc2[fsspec]`` and the protocol driver, such as ``s3fs`` for S3.

For example, ``blosc2.open("hierarchy.h5::/d0/a2")`` uses the local reader.
Chunked datasets retain their HDF5 chunk shape; contiguous datasets use
automatically chosen Blosc2 cache chunks. The source remains read-only and
assumes the file is immutable. Its file handle is closed when the source is
garbage-collected.

.. autofunction:: blosc2.available_datasets

.. autoclass:: blosc2.HDF5NDSource

    .. automethod:: __init__
    .. automethod:: get_chunk
