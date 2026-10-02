Glossary
========

Data containers
---------------

.. glossary::

    NDArray (:ref:`API <NDArray>`)
        A compressed, chunked multidimensional data array. Supports NumPy-like
        slicing and broadcasting, and out-of-core computation. Backed by an SChunk.
        Use for array workloads, especially when too large to fit in memory
        uncompressed.

    CTable (:ref:`API <CTable>`)
        A columnar table for structured data. Columns are stored, compressed, and
        queried independently, with SUMMARY indexes available when eligible. Use with
        structured data that benefits from compression, speed, and persistence.

    LazyArray (:ref:`API <LazyArray>`)
        API to store an expression or function and delay computation until the
        value is explicitly requested. Executes and stores results chunk-by-chunk.

    LazyExpr (:ref:`API <LazyExpr>`)
        Object that stores an expression consisting of at least one NDArray
        object. Follows the LazyArray API for storage and deferred computation.

Compression
-----------

.. glossary::

    codec (:class:`API <blosc2.Codec>`)
        A compressor and matching decompressor that implements a particular
        compression method. Codec choice affects compression/decompression speed and size of the compressed data.

    clevel
        An integer from 0 (no compression) to 9 (the highest compression level) that controls compression effort. Higher levels generally take longer to compress and may produce smaller results, depending on the codec and data.

    filters (:class:`API <blosc2.Filter>`)
        Transformations applied to data before the codec compresses it in order to expose patterns that can improve compression. Some filters are reversible (such as SHUFFLE), while others deliberately discard precision (such as TRUNC_PREC).

Low-level data structures
-------------------------

.. glossary::

    SChunk (:ref:`API <SChunk>`)
        The foundational container for managing a sequence of individual,
        compressed chunks. NDArrays and CTable columns are built on top of SChunk.
        Use when you want to directly manipulate raw compressed data and metadata.

    frame
        A serialized format for storing chunks along with a header and trailer for
        metadata. Frames may be contiguous (CFrame) or sparse (SFrame).

    chunk
        The unit of storage and compression, stored within a SChunk. Chunks are
        sized to fit disk/network I/O, typically 1-64MB.

    block
        The unit of decompression, stored within a chunk. Blocks are sized to
        fit CPU caches, typically 32-512KB.

    subblock
        An indexing segment within a block. One eighth the length of a block.

Indexes
-------

.. glossary::

    index (:ref:`API <Index>`)
        Auxiliary data attached to an NDArray or CTable to speed up queries.
        Allows queries to skip chunks, blocks, or rows of data that do not match.

    SUMMARY
        Lightweight index that stores per-segment minimum and maximum values to skip segments that cannot match a query.

    BUCKET
        Stores values sorted within each chunk and groups their positions into buckets to locate possible query matches.

    PARTIAL
        Stores values sorted separately within each chunk, along with their exact positions, to find matches.

    FULL
        Stores values sorted together across all chunks, along with their exact positions, to find matches and speed up sorting.

    OPSI
        Uses repeated ordering cycles to improve filtering and provide exact matching positions for checking conditions on other columns, but is not intended to converge to a globally sorted index.
