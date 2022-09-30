Top level API
=============

This API is meant to be compatible with the existing python-blosc API. There could be some parameters that are called differently, but other than that, they are largely compatible.

.. currentmodule:: blosc2

Compress and decompress
-----------------------

.. autosummary::
   :toctree: autofiles/top_level/
   :nosignatures:

   compress
   compress2
   decompress
   decompress2
   pack
   pack_array
   pack_array2
   unpack
   unpack_array
   unpack_array2
   save_array
   load_array

Set / Get compression params
----------------------------

.. autosummary::
   :toctree: autofiles/top_level/
   :nosignatures:

    clib_info
    compressor_list
    detect_number_of_cores
    free_resources
    get_clib
    print_versions
    set_blocksize
    set_nthreads
    nthreads
    set_releasegil
    set_compressor
    get_compressor
    get_blocksize

Enumerated classes
------------------

.. autosummary::
   :toctree: autofiles/top_level/
   :nosignatures:

   Codec
   Filter
   SplitMode
