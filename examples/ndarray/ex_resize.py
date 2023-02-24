#######################################################################
# Copyright (C) 2019-present, Blosc Development team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

# Resizing an array is simple (and efficient too)

import blosc2

a = blosc2.full((8, 8), fill_value=9)
a.resize((10, 10))
print(a[:])
