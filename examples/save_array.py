########################################################################
#
#       Author:  The Blosc development team - blosc@blosc.org
#
########################################################################


# A simple example using the save_array and load_array functions

import numpy as np
import blosc2

a = np.arange(1_000_000)

file_size = blosc2.save_array(a, "save_array.bl2", mode="w")
print("Length of saved array in file (bytes):", file_size)

a2 = blosc2.load_array("save_array.bl2")
assert np.alltrue(a == a2)
