########################################################################
#
#       Author:  The Blosc development team - blosc@blosc.org
#
########################################################################


# A simple example using the save_tensor and load_tensor functions

import numpy as np
import blosc2

a = np.arange(1_000_000)

file_size = blosc2.save_tensor(a, "save_tensor.bl2", mode="w")
print("Length of saved tensor in file (bytes):", file_size)

a2 = blosc2.load_tensor("save_tensor.bl2")
assert np.alltrue(a == a2)
