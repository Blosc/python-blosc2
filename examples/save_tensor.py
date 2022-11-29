#######################################################################
# Copyright (C) 2019-present, Blosc Development team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################


# A simple example using the save_tensor and load_tensor functions

import blosc2
import numpy as np

a = np.arange(1_000_000)

file_size = blosc2.save_tensor(a, "save_tensor.bl2", mode="w")
print("Length of saved tensor in file (bytes):", file_size)

a2 = blosc2.load_tensor("save_tensor.bl2")
assert np.alltrue(a == a2)
