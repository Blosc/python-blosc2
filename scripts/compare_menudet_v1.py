"""Compare v1 import contracts in independent candidate and reference wheels."""

import argparse
import copy
import json
import subprocess
import sys
import sysconfig
from pathlib import Path


def corpus():
    base = {
        "schema_version": "0.1",
        "language": {"name": "menudet", "version": "0.1"},
        "requires": ["numeric"],
        "source": "def k(x):\n    return x + 1\n",
        "entry_point": "k",
        "inputs": [{"name": "x", "dtype": "float64"}],
        "constants": [],
        "output": {"dtype": "float64", "contract": "elementwise"},
        "semantics": {"numeric": "numpy-2.5", "casting": "unsafe"},
        "context": {"ndim": 0},
    }
    cases = [("map", base)]
    for name, source, output, requires in (
        ("narrow", "def k(x):\n    return x + 0.25\n", "float32", ["numeric"]),
        ("block", "def k(x):\n    return block_sum(x)\n", "float64", ["numeric", "block-reductions"]),
        ("missing-feature", "def k(x):\n    return block_sum(x)\n", "float64", ["numeric"]),
        ("invalid-source", "def k(x):\n    return (\n", "float64", ["numeric"]),
    ):
        document = copy.deepcopy(base)
        document.update(source=source, requires=requires)
        document["output"].update(
            dtype=output, contract="block_scalar" if name in {"block", "missing-feature"} else "elementwise"
        )
        cases.append((name, document))
    for name, key, value in (
        ("unknown-version", "schema_version", "9000"),
        ("bad-entry", "entry_point", "other"),
        ("unknown-requirement", "requires", ["numeric", "unknown"]),
    ):
        document = copy.deepcopy(base)
        document[key] = value
        cases.append((name, document))
    return [(name, json.dumps(document)) for name, document in cases] + [
        ("duplicate-key", '{"schema_version":"0.1","schema_version":"0.1"}'),
        ("invalid-json", "{"),
    ]


def snapshot(package, core):
    paths = [str(package.resolve()), sysconfig.get_path("purelib"), sysconfig.get_path("platlib")]
    code = f"""
import sys, json
sys.path[:0] = {paths!r}
from pathlib import Path
import numpy as np
import blosc2
assert Path(blosc2.__file__).is_relative_to({str(package.resolve())!r})
assert blosc2.blosc2_ext._b2nd_exec_available() is {core!r}
records = {{}}
for name, document in {corpus()!r}:
    try:
        kernel = blosc2.MenudetKernel.from_json(document, jit=False)
        assert kernel._handle._has_core_kernel() is {core!r}
        values, status = kernel.evaluate(
            {{"x": np.array([1., 2., 3.])}}, return_status=True)
        records[name] = {{
            "dtype": values.dtype.str, "shape": list(values.shape),
            "bytes": values.tobytes().hex(), "fp": status,
            "inferred": kernel.inferred_dtype.str,
            "cardinality": kernel.result_cardinality,
        }}
    except blosc2.MenudetError as error:
        records[name] = {{"error": error.status, "line": error.line, "column": error.column}}
print(json.dumps(records, sort_keys=True))
"""
    return json.loads(subprocess.check_output([sys.executable, "-S", "-c", code], text=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    candidate, reference = snapshot(args.candidate, True), snapshot(args.reference, False)
    differences = {
        name: {"candidate": result, "reference": reference[name]}
        for name, result in candidate.items()
        if result != reference[name]
    }
    if differences:
        print(json.dumps(differences, indent=2, sort_keys=True))
        raise SystemExit(1)
    print(f"v1 differential: {len(candidate)} cases match (values, metadata, FP and error locations)")


if __name__ == "__main__":
    main()
