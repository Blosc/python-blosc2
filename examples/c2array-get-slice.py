#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

# Example for opening and reading a C2Array (remote array)

from time import time

import numpy as np

import blosc2

urlbase = "https://cat2.cloud/demo"
root = "@public"

# Access the server
# urlpath = blosc2.URLPath(f'{root}/examples/ds-1d.b2nd', urlbase)
# urlpath = blosc2.URLPath(f'{root}/examples/sa-1M.b2nd', urlbase)
urlpath = blosc2.URLPath(f"{root}/examples/lung-jpeg2000_10x.b2nd", urlbase)
# urlpath = blosc2.URLPath(f'{root}/examples/uncompressed_lung-jpeg2000_10x.b2nd', urlbase)

# Open the remote array
t0 = time()
remote_array = blosc2.open(urlpath, mode="r")
size = np.prod(remote_array.shape) * remote_array.cparams.typesize
print(f"Time for opening data (HTTP): {time() - t0:.3f}s - file size: {size / 2**10:.2f} KB")

# Fetch a slice of the remote array as a numpy array
t0 = time()
a = remote_array[5:9]
print(f"Time for reading data (HTTP): {time() - t0:.3f}s - {a.nbytes / 2**10:.2f} KB")

# TODO: Fetch a slice of the remote array as a blosc2.NDArray
