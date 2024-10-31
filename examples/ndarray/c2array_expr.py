import pathlib

import numpy as np

import blosc2

host = "https://demo.caterva2.net/"
root = "b2tests"
dir = "expr/"

# For a Caterva2 server running locally, use:
# host = 'localhost:8002'

# The root of the datasets
root = "b2tests"
# The directory inside root where the datasets are stored
dir = "expr/"

name1 = "ds-0-10-linspace-float64-(True, True)-a1-(60, 60)d.b2nd"
name2 = "ds-0-10-linspace-float64-(True, True)-a2-(60, 60)d.b2nd"
path1 = pathlib.Path(f"{root}/{dir + name1}").as_posix()
path2 = pathlib.Path(f"{root}/{dir + name2}").as_posix()

a = blosc2.C2Array(path1, host)
b = blosc2.C2Array(path2, host)

# Evaluate only a slice of the expression
c = a + b
print(type(c))
print(c[10:20, 10:20])

np.testing.assert_allclose(c[:], a[:] + b[:])

# Get an NDArray instance instead of a NumPy array
ndarr = c.compute()
np.testing.assert_allclose(ndarr[:], a[:] + b[:])
