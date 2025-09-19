import builtins

import numpy as np

import blosc2
from blosc2.ndarray import matmul, matrix_transpose, tensordot, vecdot  # noqa : F401


def diagonal(x: blosc2.NDArray, offset: int = 0) -> blosc2.NDArray:
    """
    Returns the specified diagonals of a matrix (or a stack of matrices) x.

    Parameters
    ----------
    x: NDArray
        Input array having shape (..., M, N) and whose innermost two dimensions form MxN matrices.

    offset: int
        Offset specifying the off-diagonal relative to the main diagonal.

        * offset = 0: the main diagonal.
        * offset > 0: off-diagonal above the main diagonal.
        * offset < 0: off-diagonal below the main diagonal.

        Default: 0.

    Returns
    -------
    out: NDArray
        An array containing the diagonals and whose shape is determined by
        removing the last two dimensions and appending a dimension equal to the size of the
        resulting diagonals.

    Reference: https://data-apis.org/array-api/latest/extensions/generated/array_api.linalg.diag.html#diag
    """
    n_rows, n_cols = x.shape[-2:]
    if offset < 0:
        rows = np.arange(-offset, n_rows)
        cols = np.arange(len(rows))
    elif offset > 0:
        cols = np.arange(offset, n_cols)
        rows = np.arange(len(cols))
    else:
        rows = cols = np.arange(builtins.min(n_rows, n_cols))
    key = tuple(slice(None, None, 1) for i in range(x.ndim - 2)) + (rows, cols)
    # TODO: change to use slice to give optimised compressing
    return blosc2.asarray(x[key])


def outer(x1: blosc2.NDArray, x2: blosc2.NDArray) -> blosc2.NDArray:
    """
    Returns the outer product of two vectors x1 and x2.

    Parameters
    ----------
    x1: NDArray
        First one-dimensional input array of size N. Must have a numeric data type.

    x2: NDArray
        Second one-dimensional input array of size M. Must have a numeric data type.

    Returns
    -------
    out: NDArray
        A two-dimensional array containing the outer product and whose shape is (N, M).
    """
    if (x1.ndim != 1) or (x2.ndim != 1):
        raise ValueError("outer only valid for 1D inputs.")
    return tensordot(x1, x2, ((), ()))


def cholesky(x: blosc2.NDArray, upper: bool = False) -> blosc2.NDArray:
    """
    Not Implemented
    Reference: https://data-apis.org/array-api/latest/extensions/generated/array_api.linalg.cholesky.html#cholesky
    """
    raise NotImplementedError


def cross(x1: blosc2.NDArray, x2: blosc2.NDArray, axis: int = -1) -> blosc2.NDArray:
    """
    Not Implemented
    Reference: https://data-apis.org/array-api/latest/extensions/generated/array_api.linalg.cross.html#cross
    """
    raise NotImplementedError


def det(x: blosc2.NDArray) -> blosc2.NDArray:
    """
    Not Implemented
    Reference: https://data-apis.org/array-api/latest/extensions/generated/array_api.linalg.det.html#det
    """
    raise NotImplementedError


def eigh(x: blosc2.NDArray) -> tuple[blosc2.NDArray, blosc2.NDArray]:
    """
    Not Implemented
    Reference: https://data-apis.org/array-api/latest/extensions/generated/array_api.linalg.eigh.html#eigh
    """
    raise NotImplementedError


def eigvalsh(x: blosc2.NDArray) -> blosc2.NDArray:
    """
    Not Implemented
    Reference: https://data-apis.org/array-api/latest/extensions/generated/array_api.linalg.eigvalsh.html#eigvalsh
    """
    raise NotImplementedError


def inv(x: blosc2.NDArray) -> blosc2.NDArray:
    """
    Not Implemented
    Reference: https://data-apis.org/array-api/latest/extensions/generated/array_api.linalg.inv.html#inv
    """
    raise NotImplementedError


def matrix_norm(
    x: blosc2.NDArray, keepdims: bool = False, ord: int | float | str | None = "fro"
) -> blosc2.NDArray:
    """
    Not Implemented but could be doable. ord may take values:
        * 'fro' - Frobenius norm
        * 'nuc' - nuclear norm
        * 1 - max(sum(abs(x), axis=-2))
        * 2 - largest singular value (sum(x**2, axis=[-1,-2]))
        * inf - max(sum(abs(x), axis=-1))
        * -1 - min(sum(abs(x), axis=-2))
        * -2 - smallest singular value
        * -inf - min(sum(abs(x), axis=-1))
    Reference: https://data-apis.org/array-api/latest/extensions/generated/array_api.linalg.matrix_norm.html#matrix_norm
    """
    raise NotImplementedError


def matrix_power(x: blosc2.NDArray, n: int) -> blosc2.NDArray:
    """
    Not Implemented
    Reference: https://data-apis.org/array-api/latest/extensions/generated/array_api.linalg.matrix_power.html#matrix_power
    """
    raise NotImplementedError


def matrix_rank(x: blosc2.NDArray, rtol: float | blosc2.NDArray | None = None) -> blosc2.NDArray:
    """
    Not Implemented
    Reference: https://data-apis.org/array-api/latest/extensions/generated/array_api.linalg.matrix_rank.html#matrix_rank
    """
    raise NotImplementedError


def pinv(x: blosc2.NDArray, rtol: float | blosc2.NDArray | None = None) -> blosc2.NDArray:
    """
    Not Implemented
    Reference: https://data-apis.org/array-api/latest/extensions/generated/array_api.linalg.pinv.html#pinv
    """
    raise NotImplementedError


def qr(x: blosc2.NDArray, mode: str = "reduced") -> tuple[blosc2.NDArray, blosc2.NDArray]:
    """
    Not Implemented
    Reference: https://data-apis.org/array-api/latest/extensions/generated/array_api.linalg.qr.html#qr
    """
    raise NotImplementedError


def slogdet(x: blosc2.NDArray) -> tuple[blosc2.NDArray, blosc2.NDArray]:
    """
    Not Implemented
    Reference: https://data-apis.org/array-api/latest/extensions/generated/array_api.linalg.slogdet.html#slogdet
    """
    raise NotImplementedError


def solve(x1: blosc2.NDArray, x2: blosc2.NDArray) -> blosc2.NDArray:
    """
    Not Implemented
    Reference: https://data-apis.org/array-api/latest/extensions/generated/array_api.linalg.solve.html#solve
    """
    raise NotImplementedError


def svd(
    x: blosc2.NDArray, full_matrices: bool = True
) -> tuple[blosc2.NDArray, blosc2.NDArray, blosc2.NDArray]:
    """
    Not Implemented
    Reference: https://data-apis.org/array-api/latest/extensions/generated/array_api.linalg.svd.html#svd
    """
    raise NotImplementedError


def svdvals(x: blosc2.NDArray) -> blosc2.NDArray:
    """
    Not Implemented
    Reference: https://data-apis.org/array-api/latest/extensions/generated/array_api.linalg.svdvals.html#svdvals
    """
    raise NotImplementedError


def trace(x: blosc2.NDArray, offset: int = 0, dtype: np.dtype | None = None) -> blosc2.NDArray:
    """
    Not Implemented
    Reference: https://data-apis.org/array-api/latest/extensions/generated/array_api.linalg.trace.html#trace
    """
    raise NotImplementedError


def vector_norm(
    x: blosc2.NDArray, axis: int | tuple[int] | None = None, keepdims: bool = False, ord: int | float = 2
) -> blosc2.NDArray:
    """
    Not Implemented but could be doable. ord may take values:
        * p: int - p-norm
        * inf - max(x)
        * -inf - min(abs(x))

    Reference: https://data-apis.org/array-api/latest/extensions/generated/array_api.linalg.vector_norm.html#vector_norm
    """
    raise NotImplementedError
