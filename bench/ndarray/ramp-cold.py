"""First-call ramp latency in fresh processes with private empty JIT caches."""

import argparse
import importlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from statistics import median
from time import perf_counter


def child(name, rank, route, mode, size):
    import numpy as np

    import blosc2

    nd = importlib.import_module("blosc2.ndarray")
    shapes = {
        100_000: ((100_000,), (250, 400), (40, 50, 50)),
        100_000_000: ((100_000_000,), (10_000, 10_000), (400, 500, 500)),
    }
    shape = shapes[size][rank - 1]
    execution = {"jit": mode != "interpreter"}
    if mode in ("tcc", "cc"):
        execution["jit_backend"] = mode
    if route == "legacy_dsl":
        nd._menudet_ramp_supported = lambda *unused: False
    begin = perf_counter()
    if route == "numpy":
        output = (np.arange(size) if name == "arange" else np.linspace(0, 1, size)).reshape(shape)
    elif route == "menudet":
        dtype = np.dtype("int64" if name == "arange" else "float64")
        start = 0 if name == "arange" else 0.0
        step = 1 if name == "arange" else 1.0 / (size - 1)
        output = nd._compute_menudet_ramp(start, step, dtype, shape, execution, {})
    else:
        output = (blosc2.arange(size, shape=shape, **execution) if name == "arange"
                  else blosc2.linspace(0, 1, size, shape=shape, **execution))
    elapsed = 1000 * (perf_counter() - begin)
    expected = (np.arange(size) if name == "arange" else np.linspace(0, 1, size)).reshape(shape)
    np.testing.assert_allclose(output if route == "numpy" else output[:], expected)
    print(elapsed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--size", type=int, choices=(100_000, 100_000_000), default=100_000)
    parser.add_argument("--child", nargs=4)
    args = parser.parse_args()
    if args.child:
        name, rank, route, mode = args.child
        child(name, int(rank), route, mode, args.size)
        return
    for name in ("arange", "linspace"):
        for route in ("numpy", "legacy_dsl", "menudet"):
            for mode in (("native",) if route == "numpy" else ("interpreter", "tcc", "cc")):
                times = []
                for rank in (1, 2, 3):
                    samples = []
                    for _ in range(args.repeats):
                        with tempfile.TemporaryDirectory(prefix="ramp-cold-") as cache:
                            env = dict(os.environ, ME_DSL_JIT_CACHE_DIR=cache)
                            result = subprocess.run(
                                [sys.executable, str(Path(__file__).resolve()), "--size", str(args.size), "--child",
                                 name, str(rank), route, mode],
                                env=env, capture_output=True, text=True, check=True,
                            )
                            samples.append(float(result.stdout.strip().splitlines()[-1]))
                    times.append(round(median(samples), 3))
                print(name, route, mode, "1D/2D/3D ms:", times, flush=True)


if __name__ == "__main__":
    main()
