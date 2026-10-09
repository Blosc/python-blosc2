"""Pinned NumPy vector generator and installed Python integration runner.

The corpus is owned by miniexpr; pass its path explicitly. No sibling checkout or
native shared library is discovered. This tool never evaluates source with eval.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REFERENCE = "2.5.3"
REVISION = "checkpoint-1"


def encode(array):
    array = np.asarray(array)
    return array.astype(array.dtype.newbyteorder(">"), copy=False).tobytes().hex()


def decode(buffer):
    dtype = np.dtype(buffer["dtype"])
    return (
        np.frombuffer(bytes.fromhex(buffer["hex"]), dtype=dtype.newbyteorder(">"))
        .astype(dtype)
        .reshape(buffer["shape"])
    )


def buffer(array, name=None):
    array = np.asarray(array)
    result = {
        "dtype": array.dtype.name,
        "shape": list(array.shape),
        "encoding": "raw-be-hex",
        "hex": encode(array),
    }
    if name is not None:
        result.update(name=name, category="array", layout="C")
    return result


def reference(operation, arrays, literal):
    x = arrays[0]
    y = arrays[1] if len(arrays) > 1 else literal
    binary = {
        "add": np.add,
        "subtract": np.subtract,
        "multiply": np.multiply,
        "divide": np.true_divide,
        "floor_divide": np.floor_divide,
        "remainder": np.remainder,
        "less": np.less,
        "left_shift": np.left_shift,
    }
    if operation in binary:
        return binary[operation](x, y)
    if operation == "cancel":
        return (x + literal) - x
    if operation == "negative":
        return np.negative(x)
    if operation == "absolute":
        return np.absolute(x)
    if operation.startswith("cast_"):
        return x.astype(operation[5:])
    if operation == "sin":
        return np.sin(x)
    raise ValueError(operation)


def generate():
    if np.__version__ != REFERENCE:
        raise RuntimeError(f"Generation requires NumPy {REFERENCE}, not {np.__version__}")
    cases = []

    def add(
        identifier, operation, expression, dtype, values, literal=None, other=None, output=None, scalar=None
    ):
        arrays = [np.array(values, dtype=dtype)]
        if other is not None:
            arrays.append(np.array(other[1], dtype=other[0]))
        with np.errstate(all="ignore"):
            expected = np.asarray(
                reference(
                    operation, arrays, np.asarray(scalar[1], dtype=scalar[0])[()] if scalar else literal
                )
            )
        names = ["x", "y"][: len(arrays)]
        artifact = {
            "schema_version": "1.0",
            "language": {"name": "miniexpr", "version": "1.0"},
            "requires": ["numeric"],
            "source": f"def k({', '.join(names)}):\n    return {expression}\n",
            "entry_point": "k",
            "inputs": [{"name": n, "dtype": a.dtype.name} for n, a in zip(names, arrays, strict=True)],
            "constants": [],
            "output": {"dtype": output or expected.dtype.name, "contract": "elementwise"},
            "semantics": {"fp": "strict"},
            "context": {"ndim": 0},
            "metadata": {},
        }
        if scalar:
            artifact["source"] = artifact["source"].replace("def k(x):", "def k(x, y):")
            value = np.asarray(scalar[1], dtype=scalar[0])
            artifact["constants"] = [
                {
                    "name": "y",
                    "dtype": scalar[0],
                    "encoding": "ieee754-hex" if value.dtype.kind == "f" else "decimal",
                    "value": encode(value) if value.dtype.kind == "f" else str(int(value)),
                }
            ]
        cases.append(
            {
                "id": identifier,
                "operation": operation,
                "semantic_revision": "menudet-draft-1.0-checked",
                "requires": ["numeric"],
                "artifact": json.dumps(artifact, sort_keys=True),
                "inputs": [buffer(a, n) for n, a in zip(names, arrays, strict=True)],
                "literal": None
                if literal is None
                else {"category": "weak", "type": type(literal).__name__, "decimal": str(literal)},
                "expected": buffer(expected),
                "comparison": {"kind": "bitwise", "nan": "bits", "signed_zero": "exact"},
            }
        )
        if scalar:
            cases[-1]["scalar_operands"] = [{"name": "y", "category": "typed_scalar", **buffer(value)}]
        for item in cases[-1]["inputs"]:
            if item["shape"] == []:
                item["category"] = "zero_dimensional_array"

    for dtype in ("int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"):
        info = np.iinfo(dtype)
        add(f"{dtype}-add-boundary", "add", "x + 1", dtype, [info.max - 1, info.max], 1)
    add(
        "int64-exact-subtract",
        "subtract",
        "x - 9007199254740993",
        "int64",
        [9007199254740993, 9007199254740994],
        9007199254740993,
    )
    add("int8-multiply-boundary", "multiply", "x * 2", "int8", [63, 64], 2)
    add("int8-negate-min", "negative", "-x", "int8", [-128, -1, 0])
    add("int64-absolute-min", "absolute", "abs(x)", "int64", [-(2**63), -1])
    add("float32-weak-literal", "cancel", "(x + 1.0) - x", "float32", [16777216, 2], 1.0)
    add("float32-typed-float64-capture", "add", "x + y", "float32", [16777216, 2], scalar=("float64", 1.0))
    add("int8-typed-int64-capture", "add", "x + y", "int8", [127], scalar=("int64", 1))
    add("float32-zero-dimensional", "add", "x + 1.0", "float32", 16777216, 1.0)
    add("int8-float-literal", "add", "x + 1.0", "int8", [127, 1], 1.0)
    add("int8-uint8-promote", "add", "x + y", "int8", [-1, 127], other=("uint8", [255, 1]))
    add("int64-uint64-promote", "add", "x + y", "int64", [-1, 2**63 - 1], other=("uint64", [0, 2**64 - 1]))
    add("int64-uint64-compare", "less", "x < y", "int64", [-1, 2**63 - 1], other=("uint64", [0, 2**64 - 1]))
    add("int64-uint64-compare-rounding", "less", "x < y", "int64", [2**53], other=("uint64", [2**53 + 1]))
    add("bool-add", "add", "x + y", "bool", [True, False], other=("bool", [True, True]), output="int64")
    add("int64-true-divide", "divide", "x / 3", "int64", [-7, 7], 3)
    add("int64-floor-divide", "floor_divide", "x // -3", "int64", [-7, 7], -3)
    add("int64-remainder", "remainder", "x % -3", "int64", [-7, 7], -3)
    add("int64-zero-divide", "floor_divide", "x // 0", "int64", [-7, 0, 7], 0)
    add("int64-min-divide", "floor_divide", "x // -1", "int64", [-(2**63)], -1)
    add("int8-large-shift", "left_shift", "x << 8", "int8", [1, -1], 8)
    add("float64-cast-int", "cast_int64", "int(x)", "float64", [1.9, -1.9, -0.0])
    add("float64-cast-infinity", "cast_int64", "int(x)", "float64", [np.inf])
    add("int64-narrow-output", "cast_int8", "x", "int64", [127, 128, -129], output="int8")
    add("float32-cast-float", "cast_float64", "float(x)", "float32", [1.25, -0.0], output="float64")
    add("float32-sin", "sin", "sin(x)", "float32", [0.0, -0.0])
    for case in cases:
        if case["id"] == "float64-cast-infinity":
            case["platform_qualification"] = (
                "NumPy nonfinite-to-integer sentinel is host-specific; not a universal contract."
            )
    return {
        "schema_version": "menudet-numpy-vectors-1",
        "generator_revision": REVISION,
        "reference": {
            "numpy": REFERENCE,
            "intp_bits": np.dtype(np.intp).itemsize * 8,
            "byteorder": sys.byteorder,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "floating_errors": "ignore",
            "python": platform.python_version(),
        },
        "cases": cases,
    }


def integrate(corpus, repeats=1):
    import blosc2

    results = []
    for case in corpus["cases"]:
        start = time.perf_counter_ns()
        try:
            kernel = blosc2.PortableKernel.from_json(case["artifact"], jit=True)
            compile_ns = time.perf_counter_ns() - start
            inputs = {item["name"]: decode(item) for item in case["inputs"]}
            start = time.perf_counter_ns()
            for _ in range(repeats):
                actual = kernel.evaluate(inputs)
            elapsed = time.perf_counter_ns() - start
            result = {
                "id": case["id"],
                "status": 0,
                "backend": "jit" if kernel.has_jit else "interpreter",
                "output": buffer(actual),
                "compile_ns": compile_ns,
                "evaluation_ns": elapsed // repeats,
            }
        except blosc2.PortableArtifactError as error:
            statuses = {"invalid_source": -3, "evaluation_error": -5}
            result = {
                "id": case["id"],
                "status": statuses.get(error.status, error.native_status),
                "category": error.status,
                "native_status": error.native_status,
                "message": str(error),
                "backend": "interpreter",
            }
        results.append(result)
    extension = Path(blosc2.blosc2_ext.__file__)
    return {
        "numpy": np.__version__,
        "package": blosc2.__file__,
        "extension": str(extension),
        "extension_sha256": hashlib.sha256(extension.read_bytes()).hexdigest(),
        "results": results,
    }


def check_baseline(corpus, report):
    for case, actual in zip(corpus["cases"], report["results"], strict=True):
        baseline = case["baseline"]
        if actual["id"] != case["id"] or actual["status"] != baseline["status"]:
            raise AssertionError(f"Changed diagnostic baseline: {case['id']}: {actual}")
        if actual["status"] == 0 and actual["output"]["hex"] != baseline["hex"]:
            raise AssertionError(f"Changed value baseline: {case['id']}: {actual}")
        if actual["status"] == 0:
            signature = json.loads(case["artifact"])["output"]["dtype"]
            if (
                actual["output"]["dtype"] != signature
                or actual["output"]["shape"] != case["inputs"][0]["shape"]
            ):
                raise AssertionError(f"Changed dtype/shape baseline: {case['id']}")
        if actual["backend"] != baseline["backend"]:
            raise AssertionError(f"Changed backend: {case['id']}")


def reference_drift(corpus):
    differences = []
    for case in corpus["cases"]:
        arrays = [decode(item) for item in case["inputs"]]
        literal = case["literal"]
        value = (
            None if literal is None else {"int": int, "float": float}[literal["type"]](literal["decimal"])
        )
        if case.get("scalar_operands"):
            value = decode(case["scalar_operands"][0])[()]
        with np.errstate(all="ignore"):
            actual = buffer(reference(case["operation"], arrays, value))
        if actual != case["expected"]:
            differences.append({"id": case["id"], "actual": actual, "expected": case["expected"]})
    return {"version_changed": np.__version__ != corpus["reference"]["numpy"], "differences": differences}


def benchmark():
    import numexpr

    import blosc2

    rows = []
    for dtype in ("float32", "float64", "int64"):
        artifact = blosc2.DSLKernel.from_source("def k(x):\n    return x * 2 + 1\n").export(
            {"x": dtype}, dtype
        )
        start = time.perf_counter_ns()
        kernel = blosc2.PortableKernel.from_json(artifact, jit=True)
        compile_ns = time.perf_counter_ns() - start
        for count in (16, 262144):
            x = np.arange(count, dtype=dtype)
            engines = {
                "portable": lambda x=x, kernel=kernel: kernel.evaluate({"x": x}),
                "numpy": lambda x=x: x * 2 + 1,
                "numexpr": lambda x=x: numexpr.evaluate("x * 2 + 1", local_dict={"x": x}),
            }
            for name, engine in engines.items():
                start = time.perf_counter_ns()
                actual = engine()
                first_ns = time.perf_counter_ns() - start
                np.testing.assert_array_equal(actual, x * 2 + 1)
                samples = []
                for _ in range(7):
                    start = time.perf_counter_ns()
                    engine()
                    samples.append(time.perf_counter_ns() - start)
                rows.append(
                    {
                        "dtype": dtype,
                        "nitems": count,
                        "engine": name,
                        "compile_ns": compile_ns if name == "portable" else None,
                        "first_ns": first_ns,
                        "median_ns": int(np.median(samples)),
                        "backend": "interpreter" if name == "portable" else name,
                    }
                )
    return {
        "platform": platform.platform(),
        "numpy": np.__version__,
        "numexpr": numexpr.__version__,
        "extension_sha256": hashlib.sha256(Path(blosc2.blosc2_ext.__file__).read_bytes()).hexdigest(),
        "numexpr_threads": numexpr.get_num_threads(),
        "samples": 7,
        "results": rows,
        "limitations": "Single process, fixed order, allocated output; no RSS or direct-native large-array timing.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["generate", "run", "record", "benchmark"])
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--native-runner", type=Path)
    parser.add_argument("--native-revision", help="Explicit source SHA used to build the standalone runner")
    parser.add_argument(
        "--installed-native-revision", help="Explicit dependency SHA used to build the installed extension"
    )
    parser.add_argument("--report", type=Path)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.action == "benchmark":
        report = benchmark()
        report["installed_native_revision"] = args.installed_native_revision
        args.corpus.write_text(json.dumps(report, indent=2) + "\n")
        return
    if args.action == "generate":
        args.corpus.write_text(json.dumps(generate(), indent=2) + "\n")
        return
    corpus = json.loads(args.corpus.read_text())
    if corpus["schema_version"] != "menudet-numpy-vectors-1":
        raise ValueError("Unsupported vector schema")
    report = integrate(corpus, args.repeats)
    report["provenance"] = {
        "python_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True
        ).strip(),
        "native_revision": args.native_revision,
        "installed_native_revision": args.installed_native_revision,
        "local_changes": [
            "tools/menudet_numpy_compat.py",
            "tests/test_menudet_numpy_compat.py",
            "checkpoint reports",
        ],
    }
    if args.action == "run":
        check_baseline(corpus, report)
    report["reference_drift"] = reference_drift(corpus)
    if args.native_runner:
        proc = subprocess.run(
            [str(args.native_runner.resolve()), str(args.corpus.resolve())],
            capture_output=True,
            text=True,
            check=True,
        )
        report["native"] = [json.loads(line) for line in proc.stdout.splitlines()]
    if args.action == "record":
        if not args.native_runner:
            parser.error("record requires --native-runner")
        for case, native, python in zip(corpus["cases"], report["native"], report["results"], strict=True):
            assert case["id"] == native["id"] == python["id"]
            assert native["status"] == python["status"], (native, python)
            if native["status"] == 0:
                assert native["hex"] == python["output"]["hex"], (native, python)
            case["baseline"] = {
                "status": native["status"],
                "hex": native.get("hex", ""),
                "backend": "interpreter",
            }
            case["classification"] = (
                "matching" if native["status"] == 0 and python["output"] == case["expected"] else "divergent"
            )
        args.corpus.write_text(json.dumps(corpus, indent=2) + "\n")
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
