########################################################################
#
#       Author:  The Blosc development team - blosc@blosc.org
#
########################################################################


# A simple example using the pack_array2 and unpack_array2 functions

import numpy as np
import blosc2

a = np.arange(1000_000)

cparams = {"codec": blosc2.Codec.BLOSCLZ}
cframe = blosc2.pack_array2(a, cparams=cparams)
print("Length of packed array in bytes:", len(cframe))

a2 = blosc2.unpack_array2(cframe)
assert np.alltrue(a == a2)
