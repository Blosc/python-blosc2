"""Profile normal LazyExpr dispatch against the former backend-decline control.

The control restores the old rejection of an explicit compiler backend in the
compression-prefilter adapter. It does not change numerical code or force NumPy.
Both runs use the ordinary benchmark's validation and matched output storage.
"""

import argparse
import cProfile
import json
import pstats
import runpy
import sys
from pathlib import Path

import blosc2
import blosc2.native_graph as native_graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fallback-control", action="store_true")
    parser.add_argument("--jit", choices=("tcc", "cc"), default="cc")
    parser.add_argument("--size", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=blosc2.nthreads)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    args = parser.parse_args()
    original = native_graph.compute_native_prefilter_if_eligible

    def old_gate(expr, item, **kwargs):
        if kwargs.get("jit_backend") is not None:
            return None
        return original(expr, item, **kwargs)

    argv = sys.argv
    sys.argv = [
        str(Path(__file__).with_name("dsl-kernel-bench.py")),
        "--jit",
        args.jit,
        "--size",
        str(args.size),
        "--threads",
        str(args.threads),
        "--samples",
        str(args.samples),
        "--warmups",
        str(args.warmups),
        "--output",
        str(args.output),
    ]
    profile = cProfile.Profile()
    try:
        if args.fallback_control:
            native_graph.compute_native_prefilter_if_eligible = old_gate
        profile.runcall(runpy.run_path, sys.argv[0], run_name="__main__")
    finally:
        native_graph.compute_native_prefilter_if_eligible = original
        sys.argv = argv
    data = json.loads(args.output.read_text())
    data["metadata"]["control"] = (
        "Restore original configured-backend prefilter rejection"
        if args.fallback_control
        else "New configured map prefilter dispatch"
    )
    args.output.write_text(json.dumps(data, indent=2) + "\n")
    with args.profile.open("w") as stream:
        pstats.Stats(profile, stream=stream).strip_dirs().sort_stats("cumulative").print_stats(25)


if __name__ == "__main__":
    main()
