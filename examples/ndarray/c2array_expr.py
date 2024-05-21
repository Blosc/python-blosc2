import blosc2
import numpy as np

# host = 'demo-api.caterva2.net:8202'
host = 'localhost:8002'
# root = 'example'
root = 'foo'
# name = 'dir1/ds-2d.b2nd'
name1 = 'operands/ds-0-10-linspace-float32-(True, True)-a1-(1000,)d.b2nd'
name2 = 'operands/ds-0-10-linspace-float32-(True, True)-a1-(1000,)d.b2nd'
a = blosc2.C2Array(name1, root, host)
b = blosc2.C2Array(name2, root, host)
print(a[990:1000])
c = a + b
print(type(c))
_ = c[:]
print(c)
d = c.eval()
np.testing.assert_allclose(c[:], a[:] + b[:])

np.testing.assert_allclose(d[:], a[:] + b[:])
