########################################################################
#
#       Created: April 30, 2021
#       Author:  The Blosc development team - blosc@blosc.org
#
########################################################################


from .version import __version__

# Codecs
from .blosc2_ext import BLOSCLZ, LZ4, LZ4HC, ZLIB, ZSTD

# Filters
from .blosc2_ext import NOFILTER, SHUFFLE, BITSHUFFLE, DELTA, TRUNC_PREC


# Public API for container module
from .utils import (compress, decompress, set_compressor, free_resources, set_nthreads,
                    clib_info, get_clib, compressor_list, set_blocksize, pack, unpack,
                    pack_array, unpack_array, get_compressor, set_releasegil, detect_number_of_cores,
                    print_versions, get_blocksize)

from .blosc2_ext import MAX_TYPESIZE, MAX_BUFFERSIZE, VERSION_STRING, VERSION_DATE
blosclib_version = "%s (%s)" % (VERSION_STRING, VERSION_DATE)

# Filter names
filter_names = {
    NOFILTER: "nofilter",
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
