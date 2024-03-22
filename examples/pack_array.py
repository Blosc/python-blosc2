#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################


import numpy as np

# A simple example using the pack and unpack functions
import blosc2

a = np.array(["å", "ç", "ø"])
parray = blosc2.pack(a, 9)
a2 = blosc2.unpack(parray)
assert np.all(a == a2)
