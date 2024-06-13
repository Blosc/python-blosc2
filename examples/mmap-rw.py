#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################


# Example for writing and reading memory-mapped files

import numpy as np

import blosc2

urlpath = "array.b2nd"
blosc2.remove_urlpath(urlpath)
a = np.arange(1_000_000, dtype=np.int64)

# Optional: the size of the array is generous enough for the mapping size since we expect the compressed data to be
# smaller than the original size
initial_mapping_size = a.size * a.itemsize

# mmap_mode and initial_mapping_size can be used for all functions which create arrays on disk
# (SChunk, asarray, empty, etc.)
blosc2.asarray(a, urlpath=urlpath, mmap_mode="w+", initial_mapping_size=initial_mapping_size)

# Read the ndarray back via the general open function
a_read = blosc2.open(urlpath, mmap_mode="r")

assert np.all(a == a_read)
blosc2.remove_urlpath(urlpath)
