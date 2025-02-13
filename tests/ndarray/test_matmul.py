import pytest
import numpy as np
import blosc2
from blosc2.ndarray import matmul


@pytest.mark.parametrize(
    ("ashape", "achunks", "ablocks"),
    [
        ((12, 10), (6, 5), (3, 3)),
        ((1, 10), (1, 5), (1, 2)),
    ],
)
@pytest.mark.parametrize(
    ("bshape", "bchunks", "bblocks"),
    [
        ((10, 1), (4, 1), (2, 1)),
        ((10, 5), (2, 4), (1, 3)),
        ((10, 12), (2, 4), (1, 3)),
    ],
)
@pytest.mark.parametrize(
    ("dtype"), [np.float32, np.float64, np.complex64, np.complex128],
)
def test_matmul(ashape, achunks, ablocks, bshape, bchunks, bblocks, dtype):
    a = blosc2.linspace(0, 10, dtype=dtype, shape=ashape, chunks=achunks, blocks=ablocks)
    b = blosc2.linspace(0, 10, dtype=dtype, shape=bshape, chunks=bchunks, blocks=bblocks)

    na = a[:]
    nb = b[:]
    blosc2_res = matmul(a, b)
    np_res = np.matmul(na, nb)
    np.testing.assert_allclose(blosc2_res, np_res, rtol=1e-6)


@pytest.mark.parametrize(
    ("ashape", "achunks", "ablocks"),
    [
        ((12, 11), (6, 5), (3, 1)),
        ((6, 2), (3, 2), (3, 1)),
        ((0, 0), (0, 0), (0, 0)),

    ],
)
@pytest.mark.parametrize(
    ("bshape", "bchunks", "bblocks"),
    [
        ((1, 5), (1, 4), (1, 3)),
        ((10, 12), (2, 4), (1, 3)),
    ],
)
def test_matmul_raises(ashape, achunks, ablocks, bshape, bchunks, bblocks):
    a = blosc2.linspace(0, 10, shape=ashape, chunks=achunks, blocks=ablocks)
    b = blosc2.linspace(0, 10, shape=bshape, chunks=bchunks, blocks=bblocks)
    if a.shape[1] != b.shape[0]:
        with pytest.raises(ValueError):
            matmul(a, b)
