import pathlib

import numpy as np

import blosc2

host = "https://cat2.cloud/demo"
root = "@public"
dir = "examples/"

# For a Caterva2 server running locally, use:
# host = 'http://localhost:8002'

name1 = "ds-1d.b2nd"
name2 = "dir1/ds-2d.b2nd"
path1 = pathlib.Path(f"{root}/{dir + name1}").as_posix()
path2 = pathlib.Path(f"{root}/{dir + name2}").as_posix()

a = blosc2.C2Array(path1, host)
b = blosc2.C2Array(path2, host)

# Evaluate only a slice of the expression
c = a[:20] + b
print(type(c))
print(c[10:20])

np.testing.assert_allclose(c[:], a[:20] + b[:])

# Get an NDArray instance instead of a NumPy array
ndarr = c.compute()
np.testing.assert_allclose(ndarr[:], a[:20] + b[:])
