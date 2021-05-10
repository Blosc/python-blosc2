########################################################################
#
#       Created: April 30, 2021
#       Author:  The Blosc development team - blosc@blosc.org
#
########################################################################


# Filters
# Codecs
from .blosc2_ext import (
    BITSHUFFLE,
    BLOSCLZ,
    DELTA,
    EXTENDED_HEADER_LENGTH,
    LZ4,
    LZ4HC,
    MAX_BUFFERSIZE,
    MAX_TYPESIZE,
    MIN_HEADER_LENGTH,
    NOFILTER,
    NOSHUFFLE,
    SHUFFLE,
    TRUNC_PREC,
    VERSION_DATE,
    VERSION_STRING,
    ZLIB,
    ZSTD,
)

# Public API for container module
from .utils import (
    clib_info,
    compress,
    compressor_list,
    decompress,
    detect_number_of_cores,
    free_resources,
    get_blocksize,
    get_clib,
    get_compressor,
    pack,
    pack_array,
    print_versions,
    set_blocksize,
    set_compressor,
    set_nthreads,
    set_releasegil,
    unpack,
    unpack_array,
)
from .version import __version__

blosclib_version = "%s (%s)" % (VERSION_STRING, VERSION_DATE)

# Filter names
filter_names = {
    NOFILTER: "nofilter",
    NOSHUFFLE: "noshuffle",
    SHUFFLE: "shuffle",
    BITSHUFFLE: "bitshuffle",
    DELTA: "delta",
    TRUNC_PREC: "trun_prec",
}

# Internal Blosc threading
nthreads = ncores = detect_number_of_cores()
# Protection against too many cores
if nthreads > 8:
    nthreads = 8
set_nthreads(nthreads)

__all__ = [
    __version__,
    BLOSCLZ,
    LZ4,
    LZ4HC,
    ZLIB,
    ZSTD,
    NOFILTER,
    SHUFFLE,
    BITSHUFFLE,
    DELTA,
    TRUNC_PREC,
    compress,
    decompress,
    set_compressor,
    free_resources,
    set_nthreads,
    clib_info,
    get_clib,
    compressor_list,
    set_blocksize,
    pack,
    unpack,
    pack_array,
    unpack_array,
    get_compressor,
    set_releasegil,
    detect_number_of_cores,
    print_versions,
    get_blocksize,
    MAX_TYPESIZE,
    MAX_BUFFERSIZE,
    VERSION_STRING,
    VERSION_DATE,
    MIN_HEADER_LENGTH,
    EXTENDED_HEADER_LENGTH,
    filter_names,
]
