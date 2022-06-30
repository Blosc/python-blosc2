########################################################################
#
#       Author:  The Blosc development team - blosc@blosc.org
#
########################################################################


import numpy as np

# A simple example using the pack and unpack functions
import blosc2

a = np.array(["å", "ç", "ø"])
parray = blosc2.pack(a, 9)
a2 = blosc2.unpack(parray)
assert np.alltrue(a == a2)
