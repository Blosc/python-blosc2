"""Fresh-process integration probe, invoked by test_safer_jit.py."""

import sys

import numpy as np

import blosc2


@blosc2.dsl_kernel
def kernel(x):
    acc = x
    for i in range(3):
        acc = acc + i
    return acc * 2


def main():
    backend, mode = sys.argv[1:]
    jit = mode != "off"
    kwargs = {"jit": jit, "jit_backend": backend}
    a = blosc2.arange(0, 128, **kwargs)
    np.testing.assert_array_equal(a[:], np.arange(128))
    a = blosc2.linspace(0, 10, 128, **kwargs)
    np.testing.assert_allclose(a[:], np.linspace(0, 10, 128))
    x = np.arange(128, dtype=np.float64)
    arr = blosc2.asarray(x)
    result = (arr * 2 + 1).compute(**kwargs)
    np.testing.assert_array_equal(result[:], x * 2 + 1)
    result = blosc2.lazyudf(kernel, (arr,), dtype=np.float64, **kwargs).compute()
    np.testing.assert_array_equal(result[:], (x + 3) * 2)
    np.testing.assert_allclose((arr * 2 + 1).sum(**kwargs), (x * 2 + 1).sum())


if __name__ == "__main__":
    main()
