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
   pack_tensor
   unpack
   unpack_array
   unpack_array2
   unpack_tensor
   save_array
   load_array
   save_tensor
   load_tensor

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
    nthreads
    print_versions
    register_codec
    register_filter
    set_blocksize
    set_nthreads
    set_releasegil
    set_compressor
    get_compressor
    get_blocksize
    get_cbuffer_sizes
    cparams_dflts
    dparams_dflts
    storage_dflts

Enumerated classes
------------------

.. autosummary::
   :toctree: autofiles/top_level/
   :nosignatures:

   Codec
   Filter
   SpecialValue
   SplitMode
   Tuner

Utils
-----

.. currentmodule:: blosc2

.. autosummary::
   :toctree: autofiles/top_level/

    compute_chunks_blocks
    get_slice_nchunks
    open
    remove_urlpath

Utility variables
-----------------

.. currentmodule:: blosc2

.. autosummary::
   :toctree: autofiles/top_level/

    blosclib_version
    DEFINED_CODECS_STOP
    EXTENDED_HEADER_LENGTH
    MAX_BUFFERSIZE
    MAX_OVERHEAD
    MAX_TYPESIZE
    MIN_HEADER_LENGTH
    prefilter_funcs
    postfilter_funcs
    ucodecs_registry
    ufilters_registry
    VERSION_DATE
    VERSION_STRING
    __version__
