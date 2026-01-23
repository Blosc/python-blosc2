#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# A simple example using the pack and unpack functions

import numpy as np

import blosc2

a = np.array(["å", "ç", "ø"])
parray = blosc2.pack(a, 9)
a2 = blosc2.unpack(parray)
assert np.all(a == a2)
