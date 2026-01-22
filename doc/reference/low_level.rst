Compression Utilities
=====================

Although using NDArray/SChunk objects is the recommended way to work with Blosc2 data, there are some utilities that allow you to work with Blosc2 data in a more low-level way.  This is useful when you need to work with data that is not stored in NDArray/SChunk objects, or when you need to work with data that is stored in a different format.

This API is meant to be compatible with the existing python-blosc API. There could be some parameters that are called differently, but other than that, they are largely compatible.  In addition, there are some new functions that are not present in the original python-blosc API that are mainly meant to overcome the 2 GB limit that the original API had.

.. currentmodule:: blosc2

Compress and decompress
-----------------------

.. autosummary::
    :toctree: autofiles/low_level/

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

Set / get compression params
----------------------------

.. autosummary::
    :toctree: autofiles/low_level/

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
    :toctree: autofiles/low_level/

    Codec
    Filter
    SpecialValue
    SplitMode
    Tuner

Utils
-----
.. autosummary::
    :toctree: autofiles/low_level/

    compute_chunks_blocks
    get_slice_nchunks
    remove_urlpath

Utility variables
-----------------
.. autosummary::
    :toctree: autofiles/low_level/

    blosclib_version
    DEFINED_CODECS_STOP
    GLOBAL_REGISTERED_CODECS_STOP
    USER_REGISTERED_CODECS_STOP
    EXTENDED_HEADER_LENGTH
    MAX_BUFFERSIZE
    MAX_BLOCKSIZE
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
