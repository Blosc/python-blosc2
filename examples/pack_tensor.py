########################################################################
#
#       Author:  The Blosc development team - blosc@blosc.org
#
########################################################################


# A simple example using the pack_tensor and unpack_tensor functions

import numpy as np
import blosc2

a = np.arange(1_000_000)

cparams = {"codec": blosc2.Codec.BLOSCLZ}
cframe = blosc2.pack_tensor(a, cparams=cparams)
print("Length of packed array in bytes:", len(cframe))

a2 = blosc2.unpack_tensor(cframe)
assert np.alltrue(a == a2)
