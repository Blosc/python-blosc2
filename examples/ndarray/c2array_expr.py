import pathlib

import numpy as np

import blosc2

host = "https://demo.caterva2.net/"
# host = 'localhost:8002'
# root = 'example'
# name = 'dir1/ds-2d.b2nd'

root = "b2tests"
dir = "expr/"
name1 = "ds-0-10-linspace-float64-(True, True)-a1-(60, 60)d.b2nd"
name2 = "ds-0-10-linspace-float64-(True, True)-a2-(60, 60)d.b2nd"
path1 = pathlib.Path(f"{root}/{dir + name1}").as_posix()
path2 = pathlib.Path(f"{root}/{dir + name2}").as_posix()

a = blosc2.C2Array(path1, host)
b = blosc2.C2Array(path2, host)
print(a[10:20, 10:20])
c = a + b
print(type(c))
_ = c[:]
print(c)
d = c.eval()
np.testing.assert_allclose(c[:], a[:] + b[:])
np.testing.assert_allclose(d[:], a[:] + b[:])
