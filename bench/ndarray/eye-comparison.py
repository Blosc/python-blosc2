"""Compare eye's former chunkwise NumPy callback with portable execution."""

from statistics import median
from time import perf_counter

import numpy as np

import blosc2


def legacy_eye(shape):
    def fill(inputs, output, offset):
        output[:] = np.eye(*output.shape, offset[0] - offset[1], dtype=output.dtype)

    return blosc2.lazyudf(fill, (0,), dtype=np.float64, shape=shape).compute()


for shape in ((100, 100), (1000, 1000), (4000, 4000), (10_000, 10_000)):
    expected = np.eye(*shape)
    for name, run in (
        ("numpy", lambda shape=shape: np.eye(*shape)),
        ("previous", lambda shape=shape: legacy_eye(shape)),
        ("menudet-interpreter", lambda shape=shape: blosc2.eye(*shape, jit=False)),
        ("menudet-tcc", lambda shape=shape: blosc2.eye(*shape, jit=True, jit_backend="tcc")),
        ("menudet-cc", lambda shape=shape: blosc2.eye(*shape, jit=True, jit_backend="cc")),
    ):
        result = run()
        np.testing.assert_array_equal(result if name == "numpy" else result[:], expected)
        del result
        samples = []
        for _ in range(5):
            begin = perf_counter()
            result = run()
            samples.append(1000 * (perf_counter() - begin))
            del result
        print(shape, name, f"{median(samples):.3f} ms", flush=True)
