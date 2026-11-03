#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""Compare the public string kernel with its serial/general and internal legacy arms.

This is a same-data diagnostic control, not a cross-engine benchmark. The legacy
arm explicitly bypasses the Menudet compatibility adapter and is not supported
authoring. Use --skip-serial to keep a full-table run bounded to two arms.
"""

import argparse
import hashlib
import json
import platform
import random
import runpy
import statistics
import time
from pathlib import Path

import blosc2
from blosc2.dsl_kernel import SourceKernel

ROOT = Path(__file__).resolve().parents[2]
BENCH = Path(__file__).with_name("string-ops.py")


@SourceKernel
def legacy_label(company, ptype):
    """Internal old-execution control; not a public authoring example."""
    pay = ptype.lower()
    c = company.lower()
    if " cab" in c:
        return "cab|" + c.removesuffix(" cab") + "|" + pay
    return "other|" + c + "|" + pay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=100_000, help="0 loads the entire table")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument(
        "--parquet", type=Path, default=ROOT / "bench/chicago-taxi/chicago-taxi-flat.parquet"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-serial", action="store_true", help="Omit the same-artifact serial arm")
    args = parser.parse_args()
    if args.rows < 0 or args.samples < 1:
        parser.error("rows must be nonnegative and samples must be positive")
    bench = runpy.run_path(str(BENCH))
    co, pt = bench["load"](args.parquet, args.rows)
    a, b, cp = bench["blosc2_setup"](co, pt)
    kernel = bench["make_taxi_label"](a.dtype, b.dtype).specialize(a, b)
    storage = {"chunks": a.chunks, "blocks": a.blocks, **cp}
    parallel = kernel.lazy({"company": a, "ptype": b}, partitions=a.blocks)
    serial = kernel.lazy({"company": a, "ptype": b}, partitions=a.chunks)
    legacy = blosc2.lazyudf(legacy_label, (a, b))
    legacy._legacy_source_recipe = True
    calls = {
        "prefilter": lambda: parallel.compute(**storage),
        "legacy": lambda: legacy.compute(**storage),
    }
    if not args.skip_serial:
        calls["serial"] = lambda: serial.compute(**storage)

    identity = None
    for name, call in calls.items():
        result = call()
        digest = bench["digest"](result, a.size)
        if identity is None:
            identity = digest
        assert digest == identity, name
        del result

    samples = {name: [] for name in calls}
    order = list(calls)
    rng = random.Random(42)
    for _ in range(args.samples):
        rng.shuffle(order)
        for name in order:
            wall, cpu = time.perf_counter(), time.process_time()
            result = calls[name]()
            samples[name].append({"wall_s": time.perf_counter() - wall, "cpu_s": time.process_time() - cpu})
            del result

    for name, call in calls.items():
        result = call()
        assert bench["digest"](result, a.size) == identity
        del result

    data = {
        "rows": a.size,
        "company_dtype": str(a.dtype),
        "payment_dtype": str(b.dtype),
        "control_note": (
            "Legacy explicitly sets the internal _legacy_source_recipe flag; serial uses the public artifact "
            "with chunk-sized logical groups."
        ),
        "public_source_sha256": hashlib.sha256(kernel.source.encode()).hexdigest(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "package": blosc2.__file__,
        "blosc2_version": blosc2.__version__,
        "c_blosc2_version": blosc2.blosclib_version,
        "extension": blosc2.blosc2_ext.__file__,
        "extension_sha256": hashlib.sha256(Path(blosc2.blosc2_ext.__file__).read_bytes()).hexdigest(),
        "threads": blosc2.nthreads,
        "fingerprint": identity,
        "native_report": parallel._source_execution_report,
        "arms": {
            name: {
                "median_wall_s": statistics.median(v["wall_s"] for v in values),
                "median_cpu_s": statistics.median(v["cpu_s"] for v in values),
                "samples": values,
            }
            for name, values in samples.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
