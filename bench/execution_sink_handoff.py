"""Installed-wheel compressed sink timing; compare identical isolated configurations."""

import argparse
import json
import statistics
import sys
import sysconfig
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--items", type=int, default=16_000_019)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--repeat", type=int, default=7)
    args = parser.parse_args()
    package = args.package.resolve()
    sys.path[:0] = [str(package), sysconfig.get_path("purelib"), sysconfig.get_path("platlib")]
    import numpy as np

    import blosc2

    assert Path(blosc2.__file__).is_relative_to(package)
    assert Path(blosc2.blosc2_ext.__file__).is_relative_to(package)
    blosc2.set_nthreads(args.workers)
    values = np.arange(args.items, dtype=np.float64) / 8
    source = blosc2.asarray(values, chunks=(262144,), blocks=(4096,))
    expression = blosc2.lazyexpr("x * 2.5 + 1", {"x": source})
    report = {}
    schedule = None
    if blosc2.blosc2_ext._b2nd_exec_available():
        from blosc2.graph_plan import NativeGraph

        plan = NativeGraph.from_expression(
            "x * 2.5 + 1", {"x": "float64"}, jit=True, _executor="c-blosc2"
        )
        schedule = plan.specialize({"x": (values.dtype, values.shape)}, 4096)

    def execute():
        if schedule is not None:
            output = blosc2.uninit(values.shape, dtype=values.dtype, chunks=source.chunks, blocks=source.blocks)
            actual, observed = schedule.execute_arrays({"x": source}, out=output, workers=args.workers)
        else:
            actual = expression.compute(jit=True, chunks=source.chunks, blocks=source.blocks)
            observed = expression._native_execution_report
        report.clear()
        report.update(observed)
        return actual

    output = execute()
    assert isinstance(output, blosc2.NDArray)
    np.testing.assert_array_equal(output[:], values * 2.5 + 1)
    samples = []
    for _ in range(args.repeat):
        start = time.perf_counter_ns()
        execute()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    print(
        json.dumps(
            {
                "storage": "compressed-sink",
                "items": args.items,
                "requested_workers": args.workers,
                "warmed_ms": statistics.median(samples),
                "report": report,
            },
            default=str,
        )
    )


if __name__ == "__main__":
    main()
