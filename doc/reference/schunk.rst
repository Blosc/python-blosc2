.. _SChunk:

SChunk
======

The basic compressed data container (aka super-chunk). This class consists of a set of useful parameters and methods that allow not only to create compressed data, and decompress it, but also to manage the data in a more sophisticated way. For example, it is possible to append new data, update existing data, delete data, etc.

.. currentmodule:: blosc2.schunk

Methods
-------

.. autosummary::
   :toctree: autofiles/schunk/
   :nosignatures:

    SChunk.__init__
    SChunk.append_data
    SChunk.decompress_chunk
    SChunk.delete_chunk
    SChunk.get_chunk
    SChunk.insert_chunk
    SChunk.insert_data
    SChunk.iterchunks
    SChunk.iterchunks_info
    SChunk.fill_special
    SChunk.update_chunk
    SChunk.update_data
    SChunk.get_slice
    SChunk.__getitem__
    SChunk.__setitem__
    SChunk.__len__
    SChunk.to_cframe
    SChunk.postfilter
    SChunk.remove_postfilter
    SChunk.filler
    SChunk.prefilter
    SChunk.remove_prefilter

.. _SChunkAttributes:

Attributes
----------

.. autosummary::
   :toctree: autofiles/schunk/
   :nosignatures:

    SChunk.blocksize
    SChunk.cbytes
    SChunk.chunkshape
    SChunk.chunksize
    SChunk.contiguous
    SChunk.cparams
    SChunk.cratio
    SChunk.dparams
    SChunk.meta
    SChunk.nbytes
    SChunk.typesize
    SChunk.urlpath
    SChunk.vlmeta

Functions
---------

.. currentmodule:: blosc2

.. autosummary::
   :toctree: autofiles/schunk/

    schunk_from_cframe
