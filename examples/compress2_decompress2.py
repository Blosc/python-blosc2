########################################################################
#
#       Created: April 30, 2021
#       Author:  The Blosc development team - blosc@blosc.org
#
########################################################################

import blosc2
import numpy as np

a = np.linspace(0, 1, 1_000_000, dtype=np.float64)
a_bytesobj = a.tobytes()
c_bytesobj = blosc2.compress2(a_bytesobj, typesize=8,
                              filters=[blosc2.TRUNC_PREC, blosc2.SHUFFLE],
                              filters_meta=[40, 0],
                              )
assert len(c_bytesobj) < len(a_bytesobj)
cratio = len(a_bytesobj) / len(c_bytesobj)
print("cratio: %.3f" % cratio)

# The next check does not work when using truncation (obviously)
#a_bytesobj2 = blosc2.decompress2(c_bytesobj)
#assert a_bytesobj == a_bytesobj2
