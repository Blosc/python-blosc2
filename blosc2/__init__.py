########################################################################
#
#       Author:  The Blosc development team - blosc@blosc.org
#
########################################################################

from enum import Enum


class Codec(Enum):
    """
    Available codecs.
    """
    BLOSCLZ = 0
    LZ4 = 1
    LZ4HC = 2
    ZLIB = 4
    ZSTD = 5


class Filter(Enum):
    """
    Available filters.
    """
    NOFILTER = 0
    SHUFFLE = 1
    BITSHUFFLE = 2
    DELTA = 3
    TRUNC_PREC = 4


class SplitMode(Enum):
    """
    Available split modes.
    """

    ALWAYS_SPLIT = 1
    NEVER_SPLIT = 2
    AUTO_SPLIT = 3
    FORWARD_COMPAT_SPLIT = 4


# Public API for container module
from .core import (
    clib_info,
    compress,
    compress2,
    compressor_list,
    decompress,
    decompress2,
    detect_number_of_cores,
    free_resources,
    get_blocksize,
    get_clib,
    get_compressor,
    pack,
    pack_array,
    pack_array2,
    save_array,
    print_versions,
    remove_urlpath,
    schunk_from_cframe,
    set_blocksize,
    set_compressor,
    set_nthreads,
    set_releasegil,
    unpack,
    unpack_array,
    unpack_array2,
    load_array,
    pack_tensor,
)
from .SChunk import (SChunk, open)
from .version import __version__

from .blosc2_ext import (
    EXTENDED_HEADER_LENGTH,
    MAX_BUFFERSIZE,
    MAX_TYPESIZE,
    MIN_HEADER_LENGTH,
    VERSION_DATE,
    VERSION_STRING,
)


blosclib_version = "%s (%s)" % (VERSION_STRING, VERSION_DATE)

# Internal Blosc threading
nthreads = ncores = detect_number_of_cores() / 2
"""Number of threads to be used in compression/decompression.
"""
# Protection against too many threads
if nthreads > 8:
    nthreads = 8
set_nthreads(nthreads)

# Defaults for compression params
cparams_dflts = {
    'codec': Codec.BLOSCLZ,
    'codec_meta': 0,
    'clevel': 5,
    'use_dict': False,
    'typesize': 8,
    'nthreads': nthreads,
    'blocksize': 0,
    'splitmode': SplitMode.FORWARD_COMPAT_SPLIT,
    'schunk': None,
    'filters': [0, 0, 0, 0, 0, Filter.SHUFFLE],
    'filters_meta': [0, 0, 0, 0, 0, 0],
    'prefilter': None,
    'preparams': None,
    'udbtune': None,
    'instr_codec': False
}

# Defaults for decompression params
dparams_dflts = {
    'nthreads': nthreads,
    'schunk': None,
    'postfilter': None,
    'postparams': None
}

# Default for storage
storage_dflts = {
    'contiguous': False,
    'urlpath': None,
    'cparams': None,
    'dparams': None,
    'io': None
}


__all__ = [
    "__version__",
    "compress",
    "decompress",
    "cparams_dflts",
    "dparams_dflts",
    "storage_dflts",
    "set_compressor",
    "free_resources",
    "set_nthreads",
    "clib_info",
    "get_clib",
    "compressor_list",
    "set_blocksize",
    "pack",
    "unpack",
    "pack_array",
    "pack_array2",
    "save_array",
    "unpack_array",
    "unpack_array2",
    "load_array",
    "get_compressor",
    "set_releasegil",
    "detect_number_of_cores",
    "print_versions",
    "get_blocksize",
    "MAX_TYPESIZE",
    "MAX_BUFFERSIZE",
    "VERSION_STRING",
    "VERSION_DATE",
    "MIN_HEADER_LENGTH",
    "EXTENDED_HEADER_LENGTH",
    "compress2",
    "decompress2",
    "SChunk",
    "open",
    "remove_urlpath",
    "nthreads"
]
