"""Compare eager NumPy, legacy DSL and portable ramp constructors."""

import argparse
import importlib
from statistics import median
from time import perf_counter

import numpy as np

import blosc2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--size", type=int, choices=(100_000, 8_000_000, 100_000_000), default=8_000_000)
    args = parser.parse_args()
    nd = importlib.import_module("blosc2.ndarray")
    original_gate = nd._menudet_ramp_supported
    print("Extension:", blosc2.blosc2_ext.__file__, flush=True)
    print("Threads:", blosc2.nthreads, flush=True)
    print("Times include kernel setup, allocation and storage; one warmup per case.", flush=True)
    try:
        shapes = {
            100_000: ((100_000,), (250, 400), (40, 50, 50)),
            8_000_000: ((8_000_000,), (2000, 4000), (200, 200, 200)),
            100_000_000: ((100_000_000,), (10_000, 10_000), (400, 500, 500)),
        }[args.size]
        for shape in shapes:
            size = int(np.prod(shape))
            for name in ("arange", "linspace"):
                expected = (np.arange(size) if name == "arange" else np.linspace(0, 1, size)).reshape(shape)
                for route in ("numpy", "legacy_dsl", "menudet"):
                    modes = ("native",) if route == "numpy" else ("interpreter", "tcc", "cc")
                    for mode in modes:
                        execution = {"jit": mode != "interpreter"}
                        if mode in ("tcc", "cc"):
                            execution["jit_backend"] = mode
                        nd._menudet_ramp_supported = (
                            (lambda *unused: False) if route == "legacy_dsl" else original_gate
                        )

                        def run(route=route, size=size, name=name, shape=shape, execution=execution):
                            if route == "numpy":
                                return (np.arange(size) if name == "arange" else np.linspace(0, 1, size)).reshape(shape)
                            if route == "menudet":
                                dtype = np.dtype("int64" if name == "arange" else "float64")
                                start = 0 if name == "arange" else 0.0
                                step = 1 if name == "arange" else 1.0 / (size - 1)
                                return nd._compute_menudet_ramp(start, step, dtype, shape, execution, {})
                            if name == "arange":
                                return blosc2.arange(size, shape=shape, **execution)
                            return blosc2.linspace(0, 1, size, shape=shape, **execution)

                        output = run()
                        values = output if route == "numpy" else output[:]
                        if name == "arange":
                            np.testing.assert_array_equal(values, expected)
                        else:
                            np.testing.assert_allclose(values, expected)
                        del output, values
                        samples = []
                        for _ in range(args.repeats):
                            begin = perf_counter()
                            output = run()
                            samples.append(1000 * (perf_counter() - begin))
                            del output
                        print(name, shape, route, mode, f"{median(samples):.3f} ms", flush=True)
    finally:
        nd._menudet_ramp_supported = original_gate


if __name__ == "__main__":
    main()
