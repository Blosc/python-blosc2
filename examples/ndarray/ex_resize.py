#######################################################################
# Copyright (C) 2019-present, Blosc Development team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

import blosc2
import numpy as np

np.random.seed(123)

shape = (8, 8)
cparams = {"nthreads": 2}
dparams = {"nthreads": 2}


fill_value = b"1"
a = blosc2.full(shape, fill_value=fill_value, cparams=cparams, dparams=dparams)
print(a.schunk.cparams)
print(a.schunk.dparams)
a.resize((10, 10))

print(a[:])
