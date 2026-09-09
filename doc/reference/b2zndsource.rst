.. _B2ZNDSource:

B2ZNDSource
===========

``B2ZNDSource`` exposes an external NDArray member inside an immutable ``.b2z``
archive through :ref:`ByteRangeNDSource`. Chunks and blocks are read directly
from the ZIP member using native Blosc2 range reads without decompressing or
downloading the archive.

The source is assumed immutable (``assume_immutable=True``). It requires an
explicit dataset path pointing to an uncompressed (``ZIP_STORED``) external
NDArray member. Embedded leaves and CTable columns are not supported.

Install support with ``pip install "blosc2[fsspec]"`` plus the protocol
driver, such as ``s3fs`` for S3.

.. autoclass:: blosc2.B2ZNDSource

    .. automethod:: __init__
