.. _SChunk:

Super-chunk API
===============

.. currentmodule:: blosc2.SChunk

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
    SChunk.update_chunk
    SChunk.update_data
    SChunk.get_slice
    SChunk.__getitem__
    SChunk.__setitem__
    SChunk.to_cframe
    SChunk.postfilter
    SChunk.remove_postfilter
    SChunk.filler
    SChunk.prefilter
    SChunk.remove_prefilter

Attributes
----------

.. toctree::
   :titlesonly:

   autofiles/schunk/attributes/blosc2.SChunk.SChunk.cbytes
   autofiles/schunk/attributes/blosc2.SChunk.SChunk.chunkshape
   autofiles/schunk/attributes/blosc2.SChunk.SChunk.chunksize
   autofiles/schunk/attributes/blosc2.SChunk.SChunk.cparams
   autofiles/schunk/attributes/blosc2.SChunk.SChunk.cratio
   autofiles/schunk/attributes/blosc2.SChunk.SChunk.dparams
   autofiles/schunk/attributes/blosc2.SChunk.SChunk.nbytes
   autofiles/schunk/attributes/blosc2.SChunk.SChunk.typesize
   autofiles/schunk/attributes/vlmeta

Functions
---------

.. currentmodule:: blosc2

.. autosummary::
   :toctree: autofiles/schunk/

    open
    schunk_from_cframe
