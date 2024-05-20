import blosc2
import numpy as np

# host = 'demo-api.caterva2.net:8202'
host = 'localhost:8002'
# root = 'example'
root = 'foo'
name = 'dir1/ds-2d.b2nd'

a = blosc2.C2Array(name, root, host)
print(a.meta.keys())

print(a.meta['schunk'])
c = a + a
print(c)

np.testing.assert_allclose(c[...], a[:]*2)
