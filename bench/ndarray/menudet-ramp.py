"""Separate portable ramp setup from execution: conda run -n blosc2 python <this file>."""

import argparse
import importlib
from statistics import median
from time import perf_counter

import numpy as np

import blosc2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    ndarray = importlib.import_module("blosc2.ndarray")
    print("Extension:", blosc2.blosc2_ext.__file__)
    print("Threads:", blosc2.nthreads)
    print("Setup includes export and load; evaluation excludes output allocation.")
    for shape in ((8_000_000,), (2000, 4000), (200, 200, 200)):
        for dtype in (np.dtype("int64"), np.dtype("float64")):
            size = int(np.prod(shape))
            start = 0 if dtype.kind == "i" else 0.0
            step = 1 if dtype.kind == "i" else 1.0 / (size - 1)
            expected = (np.arange(size) if dtype.kind == "i" else np.linspace(0, 1, size)).reshape(shape)
            for backend in ("interpreter", "tcc", "cc"):
                setup, evaluation = [], []
                for repetition in range(args.repeats + 1):
                    begin = perf_counter()
                    lazy = ndarray._menudet_ramp(
                        start, step, dtype, shape,
                        execution={"jit": backend != "interpreter",
                                   "jit_backend": None if backend == "interpreter" else backend},
                    )
                    setup_time = perf_counter() - begin
                    output = blosc2.empty(shape, dtype=dtype)
                    begin = perf_counter()
                    try:
                        lazy.kernel._handle.evaluate_prefilter({}, out=output)
                    except blosc2.PortableArtifactError as error:
                        print(shape, dtype, backend, "FAILED:", error, flush=True)
                        break
                    evaluation_time = perf_counter() - begin
                    if repetition == 0:
                        np.testing.assert_allclose(output[:], expected)
                    else:
                        setup.append(setup_time * 1000)
                        evaluation.append(evaluation_time * 1000)
                else:
                    print(shape, dtype, backend, "has_jit=", lazy.kernel.has_jit,
                          f"setup={median(setup):.3f} ms eval={median(evaluation):.3f} ms", flush=True)


if __name__ == "__main__":
    main()
