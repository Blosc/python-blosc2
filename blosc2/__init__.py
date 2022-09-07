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


from .blosc2_ext import (
    EXTENDED_HEADER_LENGTH,
    MAX_BUFFERSIZE,
    MAX_TYPESIZE,
    MIN_HEADER_LENGTH,
    VERSION_DATE,
    VERSION_STRING,
    cparams_dflts,
    dparams_dflts,
    storage_dflts,
)

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
    print_versions,
    remove_urlpath,
    set_blocksize,
    set_compressor,
    set_nthreads,
    set_releasegil,
    unpack,
    unpack_array,
)
from .SChunk import SChunk, open
from .version import __version__

blosclib_version = "%s (%s)" % (VERSION_STRING, VERSION_DATE)

# Internal Blosc threading
nthreads = ncores = detect_number_of_cores()
# Protection against too many cores
if nthreads > 8:
    nthreads = 8
set_nthreads(nthreads)

__all__ = [
    "__version__",
    "compress",
    "decompress",
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
    "unpack_array",
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
    "cparams_dflts",
    "decompress2",
    "dparams_dflts",
    "storage_dflts",
    "SChunk",
    "open",
    "remove_urlpath",
]
