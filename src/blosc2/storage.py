#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

from dataclasses import dataclass, field, asdict

import blosc2


# Internal Blosc threading
# Get CPU info
cpu_info = blosc2.get_cpu_info()
nthreads = ncores = cpu_info.get("count", 1)
"""Number of threads to be used in compression/decompression.
"""
# Protection against too many threads
nthreads = min(nthreads, 32)
# Experiments say that, when using a large number of threads, it is better to not use them all
nthreads -= nthreads // 8

# Defaults for compression params
cparams_dflts = {
    "codec": blosc2.Codec.ZSTD,
    "codec_meta": 0,
    "clevel": 1,
    "use_dict": False,
    "typesize": 8,
    "nthreads": nthreads,
    "blocksize": 0,
    "splitmode": blosc2.SplitMode.ALWAYS_SPLIT,
    "filters": [
        blosc2.Filter.NOFILTER,
        blosc2.Filter.NOFILTER,
        blosc2.Filter.NOFILTER,
        blosc2.Filter.NOFILTER,
        blosc2.Filter.NOFILTER,
        blosc2.Filter.SHUFFLE,
    ],
    "filters_meta": [0, 0, 0, 0, 0, 0],
    "tuner": blosc2.Tuner.STUNE,
}
"""
Compression params defaults.
"""

# Defaults for decompression params
dparams_dflts = {"nthreads": nthreads}
"""
Decompression params defaults.
"""
# Default for storage
storage_dflts = {"contiguous": False, "urlpath": None, "cparams": None, "dparams": None, "io": None}
"""
Storage params defaults. This is meant only for :ref:`SChunk <SChunk>` or :ref:`NDArray <NDArray>`.
"""


def default_nthreads():
    return nthreads

def default_filters():
    return [blosc2.Filter.NOFILTER,
            blosc2.Filter.NOFILTER,
            blosc2.Filter.NOFILTER,
            blosc2.Filter.NOFILTER,
            blosc2.Filter.NOFILTER,
            blosc2.Filter.SHUFFLE]


def default_filters_meta():
    return [0] * 6

@dataclass
class CParams:
    """Dataclass for hosting the different compression parameters.

    Parameters
    ----------
    codec: :class:`Codec`
        The compressor code. Default is :py:obj:`Codec.ZSTD <Codec>`.
    codec_meta: int
        The metadata for the compressor code, 0 by default.
    clevel: int
        The compression level from 0 (no compression) to 9
        (maximum compression). Default: 1.
    use_dict: bool
        Use dicts or not when compressing
        (only for :py:obj:`blosc2.Codec.ZSTD <Codec>`). Default: `False`.
    typesize: int from 1 to 255
        The data type size. Default: 8.
    nthreads: int
        The number of threads to use internally. By default, blosc2 computes
        a good guess.
    blocksize: int
        The requested size of the compressed blocks. If 0 (the default)
        blosc2 chooses it automatically.
    splitmode: :class:`SplitMode`
        The split mode for the blocks.
        The default value is :py:obj:`SplitMode.ALWAYS_SPLIT <SplitMode>`.
    filters: :class:`Filter` list
        The sequence of filters. Default: [:py:obj:`Filter.NOFILTER <Filter>`,
        :py:obj:`Filter.NOFILTER <Filter>`, :py:obj:`Filter.NOFILTER <Filter>`, :py:obj:`Filter.NOFILTER <Filter>`,
        :py:obj:`Filter.NOFILTER <Filter>`, :py:obj:`Filter.SHUFFLE <Filter>`].
    filters_meta: list
        The metadata for filters. Default: `[0, 0, 0, 0, 0, 0]`.
    tuner: :class:`Tuner`
        The tuner to use. Default: :py:obj:`Tuner.STUNE <Tuner>`.
    """
    codec: blosc2.Codec = blosc2.Codec.ZSTD
    codec_meta: int = 0
    clevel: int = 1
    use_dict: bool = False
    typesize: int = 8
    nthreads: int = field(default_factory=default_nthreads)
    blocksize: int = 0
    splitmode: blosc2.SplitMode = blosc2.SplitMode.ALWAYS_SPLIT
    filters: list[blosc2.Filter] = field(default_factory=default_filters)
    filters_meta: list[int] = field(default_factory=default_filters_meta)
    tuner: blosc2.Tuner = blosc2.Tuner.STUNE

    # def __post_init__(self):
    #     if len(self.filters) > 6: