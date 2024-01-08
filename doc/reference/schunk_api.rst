.. _SChunk:

SChunk API
==========

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

.. toctree::
   :titlesonly:
   :maxdepth: 1

   autofiles/schunk/attributes/blosc2.schunk.SChunk.blocksize
   autofiles/schunk/attributes/blosc2.schunk.SChunk.cbytes
   autofiles/schunk/attributes/blosc2.schunk.SChunk.chunkshape
   autofiles/schunk/attributes/blosc2.schunk.SChunk.chunksize
   autofiles/schunk/attributes/blosc2.schunk.SChunk.contiguous
   autofiles/schunk/attributes/blosc2.schunk.SChunk.cparams
   autofiles/schunk/attributes/blosc2.schunk.SChunk.cratio
   autofiles/schunk/attributes/blosc2.schunk.SChunk.dparams
   autofiles/schunk/attributes/meta
   autofiles/schunk/attributes/blosc2.schunk.SChunk.nbytes
   autofiles/schunk/attributes/blosc2.schunk.SChunk.typesize
   autofiles/schunk/attributes/blosc2.schunk.SChunk.urlpath
   autofiles/schunk/attributes/vlmeta

Functions
---------

.. currentmodule:: blosc2

.. autosummary::
   :toctree: autofiles/schunk/

    schunk_from_cframe
