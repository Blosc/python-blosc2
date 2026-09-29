Glossary
========

.. glossary::

    block
        The unit of decompression, stored within a chunk. Blocks are sized to
        fit CPU caches, typically 32-512KB.

    BUCKET
        Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod
        tempor incididunt ut labore et dolore magna aliqua.

    chunk
        The unit of storage and compression, stored within a SChunk. Chunks are
        sized to fit disk/network I/O, typically 1-64MB.

    clevel
        Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod
        tempor incididunt ut labore et dolore magna aliqua.

    codec (:class:`API <blosc2.Codec>`)
        Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod
        tempor incididunt ut labore et dolore magna aliqua.

    compression parameters (:class:`API <blosc2.CParams>`)
        Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod
        tempor incididunt ut labore et dolore magna aliqua.

    CTable (:ref:`API <CTable>`)
        A columnar table for structured data. Columns are stored, compressed, and
        queried independently, with SUMMARY indexes available by default. Use with
        structured data that benefits from compression, speed, and persistence.

    filters (:class:`API <blosc2.Filter>`)
        Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod
        tempor incididunt ut labore et dolore magna aliqua.

    frame
        Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod
        tempor incididunt ut labore et dolore magna aliqua.

    FULL
        Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod
        tempor incididunt ut labore et dolore magna aliqua.

    NDArray (:ref:`API <NDArray>`)
        A compressed, chunked multidimensional data array. Supports NumPy-like
        slicing and broadcasting, and out-of-core computation. Backed by an SChunk.
        Use for array workloads, especially when too large to fit in memory
        uncompressed.

    OPSI
        Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod
        tempor incididunt ut labore et dolore magna aliqua.

    PARTIAL
        Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod
        tempor incididunt ut labore et dolore magna aliqua.

    SChunk (:ref:`API <SChunk>`)
        The foundational container for managing a sequence of individual,
        compressed chunks. NDArrays and CTable columns are built on top of SChunk.
        Use when you want to directly manipulate raw compresesd data and metadata.

    subblock
        An indexing segment within a block. One eighth the length of a block.

    SUMMARY
        Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod
        tempor incididunt ut labore et dolore magna aliqua.
