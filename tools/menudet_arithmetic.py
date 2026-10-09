"""Pinned NumPy M3 reference vectors, retaining checked 1.0 corpus separately.

Generate: python -m tools.menudet_arithmetic CORPUS_PATH
Run with tools.menudet_conformance run and an explicit native runner.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np

from tools import menudet_conformance as compat

DTYPES = (
    "bool",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float32",
    "float64",
)
OPERATORS = {
    "add": ("+", np.add),
    "subtract": ("-", np.subtract),
    "multiply": ("*", np.multiply),
    "divide": ("/", np.true_divide),
    "floor-divide": ("//", np.floor_divide),
    "remainder": ("%", np.remainder),
    "power": ("**", np.power),
    "bit-and": ("&", np.bitwise_and),
    "bit-or": ("|", np.bitwise_or),
    "bit-xor": ("^", np.bitwise_xor),
    "shift-left": ("<<", np.left_shift),
    "shift-right": (">>", np.right_shift),
    "equal": ("==", np.equal),
    "less": ("<", np.less),
}
OPERATORS.update(
    {
        "not-equal": ("!=", np.not_equal),
        "less-equal": ("<=", np.less_equal),
        "greater": (">", np.greater),
        "greater-equal": (">=", np.greater_equal),
    }
)


def recipe(expression, output):
    if expression == "(x + 1) * 2":
        return {"operation": "intermediate", "cast_to": output}
    if expression == "x * y - 1":
        return {"operation": "noncontracting", "cast_to": output}
    for name, (spelling, _function) in OPERATORS.items():
        if expression == f"x {spelling} y":
            return {"operation": name, "cast_to": output}
    for expression_key, name in {
        "+x": "positive",
        "-x": "negative",
        "~x": "invert",
        "abs(x)": "absolute",
    }.items():
        if expression == expression_key:
            return {"operation": name, "cast_to": output}
    for name, (spelling, _function) in OPERATORS.items():
        prefix = f"x {spelling} "
        if expression.startswith(prefix):
            literal = expression[len(prefix) :]
            return {
                "operation": name,
                "literal": {"type": "float" if "." in literal else "int", "decimal": literal},
                "cast_to": output,
            }
    for dtype in DTYPES:
        if expression == f"{dtype}(x)":
            return {"operation": "identity", "cast_to": dtype}
    if expression == "x":
        return {"operation": "identity", "cast_to": output}
    raise ValueError("No independent NumPy recipe for expression")


def reference_recipe(case):
    arrays = [compat.checkpoint.decode(item) for item in case["inputs"]]
    spec = case["numpy_recipe"]
    with np.errstate(all="ignore"):
        op = spec["operation"]
        if op == "identity":
            actual = arrays[0]
        elif op == "intermediate":
            actual = (arrays[0] + 1) * 2
        elif op == "noncontracting":
            actual = arrays[0] * arrays[1] - 1
        elif op in {"positive", "negative", "absolute", "invert"}:
            actual = getattr(np, op)(arrays[0])
        else:
            literal = spec.get("literal")
            if spec.get("captured_category"):
                scalar = compat.checkpoint.decode(case["scalar_operands"][0])[()]
                right = scalar.item() if spec["captured_category"] == "weak" else scalar
            else:
                right = (
                    arrays[1]
                    if literal is None
                    else {"int": int, "float": float}[literal["type"]](literal["decimal"])
                )
            actual = OPERATORS[op][1](arrays[0], right)
        if spec["cast_to"]:
            actual = actual.astype(spec["cast_to"])
    return compat.describe(actual)


def make_case(
    identifier,
    arrays,
    expression,
    expected,
    operation="identity",
    output=None,
    casting="unsafe",
    diagnostic=None,
):
    case = compat.make_case(identifier, "identity", "x", arrays, output=output)
    artifact = json.loads(case["artifact"])
    names = [item["name"] for item in case["inputs"]]
    artifact.update(
        schema_version="1.1",
        language={"name": "miniexpr", "version": "1.1"},
        source=f"def k({', '.join(names)}):\n    return {expression}\n",
        semantics={"fp": "strict", "numeric": "numpy-2.5", "casting": casting},
    )
    if expected is not None:
        artifact["output"]["dtype"] = output or expected.dtype.name
    case.update(
        artifact=json.dumps(artifact, sort_keys=True),
        semantic_revision="menudet-numpy-1.1",
        operation=operation,
        reference_kind="native-contract",
        expected={"diagnostic": {"category": diagnostic}} if diagnostic else compat.describe(expected),
    )
    if diagnostic is None:
        case["inferred_dtype"] = arrays[0].dtype.name if output is not None else expected.dtype.name
        case["numpy_recipe"] = recipe(expression, output)
        case["reference_kind"] = "numpy"
        if expected.dtype.kind == "f" and expression != "x":
            case["comparison"]["nan"] = "equal"
    return case


def generate():  # noqa: C901 -- explicit finite operator/promotion/cast matrix, no source evaluation
    if np.__version__ != compat.checkpoint.REFERENCE:
        raise RuntimeError("Generation requires pinned NumPy 2.5.3")
    cases = []
    promotions = []
    cast_matrix = []
    with np.errstate(all="ignore"):
        for left in DTYPES:
            for right in DTYPES:
                a = np.array([0, 1, 1], dtype=left)
                b = np.array([1, 0, 1], dtype=right)
                expected = np.add(a, b)
                promotions.append({"left": left, "right": right, "dtype": np.result_type(a, b).name})
                cases.append(make_case(f"promotion-{left}-{right}", [a, b], "x + y", expected, "add"))
                cases.append(make_case(f"compare-{left}-{right}", [a, b], "x == y", np.equal(a, b), "equal"))
                for name, (spelling, function) in OPERATORS.items():
                    if name in {"add", "equal"}:
                        continue
                    try:
                        expected = function(a, b)
                        diagnostic = None
                    except TypeError:
                        expected, diagnostic = None, "invalid_source"
                    cases.append(
                        make_case(
                            f"mixed-{left}-{right}-{name}",
                            [a, b],
                            f"x {spelling} y",
                            expected,
                            name,
                            diagnostic=diagnostic,
                        )
                    )
        for dtype in DTYPES:
            kind = np.dtype(dtype).kind
            if kind == "b":
                a, b = np.array([True, False, True]), np.array([True, True, False])
            elif kind in "iu":
                info = np.iinfo(dtype)
                a = np.array([info.min, info.max, 0, 1, 2, 3], dtype=dtype)
                b = np.array([1, 2, 0, 0, 2, 3], dtype=dtype)
                if kind == "i":
                    a[-2:] = [-7, -3]
                    b[-2:] = [3, -1]
            else:
                a = np.array([-7, -3, 0.0, -0.0, 0.1, 1], dtype=dtype)
                b = np.array([3, -1, 0.0, 2, 0.1, 0.1], dtype=dtype)
            for name, (spelling, function) in OPERATORS.items():
                try:
                    expected = function(a, b)
                    diagnostic = None
                except (TypeError, ValueError):
                    expected, diagnostic = (
                        None,
                        "invalid_source" if kind == "b" or kind == "f" else "evaluation_error",
                    )
                cases.append(
                    make_case(
                        f"operator-{dtype}-{name}",
                        [a, b],
                        f"x {spelling} y",
                        expected,
                        name,
                        diagnostic=diagnostic,
                    )
                )
            for name, spelling, function in [
                ("positive", "+x", np.positive),
                ("negative", "-x", np.negative),
                ("absolute", "abs(x)", np.absolute),
                ("invert", "~x", np.invert),
            ]:
                try:
                    expected = function(a)
                    diagnostic = None
                except TypeError:
                    expected, diagnostic = None, "invalid_source"
                cases.append(
                    make_case(f"unary-{dtype}-{name}", [a], spelling, expected, name, diagnostic=diagnostic)
                )
            if kind in "iu":
                boundary_a = np.array([np.iinfo(dtype).min, 1, 0], dtype=dtype)
                boundary_b = np.array([-1 if kind == "i" else 1, 0, 0], dtype=dtype)
                for name, spelling, function in [
                    ("floor", "//", np.floor_divide),
                    ("mod", "%", np.remainder),
                ]:
                    cases.append(
                        make_case(
                            f"integer-divisor-boundary-{dtype}-{name}",
                            [boundary_a, boundary_b],
                            f"x {spelling} y",
                            function(boundary_a, boundary_b),
                        )
                    )
                counts = np.array(
                    [0, 1, np.iinfo(dtype).bits - 1, np.iinfo(dtype).bits, np.iinfo(dtype).bits + 1, 127],
                    dtype=dtype,
                )
                if kind == "i":
                    counts[-1] = -1
                for name, function, spelling in [
                    ("left", np.left_shift, "<<"),
                    ("right", np.right_shift, ">>"),
                ]:
                    cases.append(
                        make_case(
                            f"shift-boundary-{dtype}-{name}",
                            [a, counts],
                            f"x {spelling} y",
                            function(a, counts),
                        )
                    )
            for literal in (1, 128, -1, 0.5):
                try:
                    expected = np.add(a, literal)
                    diagnostic = None
                except OverflowError:
                    expected, diagnostic = None, "invalid_source"
                label = str(literal).replace("-", "negative-").replace(".", "point-")
                cases.append(
                    make_case(
                        f"weak-{dtype}-{label}", [a], f"x + {literal}", expected, diagnostic=diagnostic
                    )
                )
                for name, spelling, function in [("equal", "==", np.equal), ("less", "<", np.less)]:
                    cases.append(
                        make_case(
                            f"weak-compare-{dtype}-{label}-{name}",
                            [a],
                            f"x {spelling} {literal}",
                            function(a, literal),
                        )
                    )
        for source in DTYPES:
            values = np.array([0, 1], dtype=source)
            for target in DTYPES:
                for policy in ("safe", "same_kind", "unsafe"):
                    allowed = bool(np.can_cast(source, target, casting=policy))
                    cast_matrix.append({"from": source, "to": target, "policy": policy, "allowed": allowed})
                    cases.append(
                        make_case(
                            f"cast-{source}-{target}-{policy.replace('_', '-')}",
                            [values],
                            "x",
                            values.astype(target) if allowed else None,
                            output=target,
                            casting=policy,
                            diagnostic=None if allowed else "binding_error",
                        )
                    )
        for source in DTYPES[1:9]:
            info = np.iinfo(source)
            values = np.array([info.min, info.max, 0, 1], dtype=source)
            for target in DTYPES:
                cases.append(
                    make_case(
                        f"cast-boundary-{source}-{target}",
                        [values],
                        "x",
                        values.astype(target),
                        output=target,
                    )
                )
                cases.append(
                    make_case(f"explicit-{source}-{target}", [values], f"{target}(x)", values.astype(target))
                )
        for source in ("float32", "float64"):
            values = np.array([-1.9, -0.0, 0.0, 1.9, 127.9], dtype=source)
            for target in DTYPES:
                # Negative -> unsigned, NaN/inf, and finite out-of-range conversion
                # are deliberately defined diagnostics, not host sentinel goldens.
                diagnostic = "evaluation_error" if target.startswith("uint") else None
                cases.append(
                    make_case(
                        f"float-cast-{source}-{target}",
                        [values],
                        "x",
                        None if diagnostic else values.astype(target),
                        output=target,
                        diagnostic=diagnostic,
                    )
                )
            for target in DTYPES[1:9]:
                case = make_case(
                    f"nonfinite-cast-{source}-{target}",
                    [np.array([np.nan, np.inf, -np.inf], dtype=source)],
                    "x",
                    None,
                    output=target,
                    diagnostic="evaluation_error",
                )
                case["divergence"] = {
                    "id": "defined-float-cast",
                    "reason": "Reject unstable nonfinite/out-of-range floating-to-integer conversions instead of preserving a host-specific NumPy sentinel",
                }
                cases.append(case)
        # Computation must wrap at int8 before conversion to a wider output.
        a = np.array([127, -128], dtype="int8")
        cases.append(
            make_case(
                "intermediate-not-output-driven",
                [a],
                "(x + 1) * 2",
                ((a + 1) * 2).astype("int64"),
                output="int64",
            )
        )
        for dtype in DTYPES[1:9]:
            maximum = np.iinfo(dtype).max
            a, b = np.array([maximum, 2, 0], dtype=dtype), np.array([3, 10, 0], dtype=dtype)
            cases.append(make_case(f"power-wrapping-{dtype}", [a, b], "x ** y", np.power(a, b)))
        for left, right in [("int64", "uint64"), ("uint64", "int64")]:
            a = np.array([2**63 - 1], dtype=left)
            b = np.array([2**63 - 2], dtype=right)
            cases.append(make_case(f"comparison-rounded-{left}-{right}", [a, b], "x == y", np.equal(a, b)))
        from blosc2.portable_kernel import portable_scalar_descriptor

        for dtype in ("int8", "float32", "float64"):
            a = np.array([0, 1, 2], dtype=dtype)
            for label, scalar in [
                ("weak-int", 1),
                ("weak-float", 0.5),
                ("typed-int", np.int64(1)),
                ("typed-float", np.float64(0.5)),
                ("typed-float32", np.float32(0.5)),
            ]:
                expected = np.add(a, scalar)
                case = make_case(f"capture-{dtype}-{label}", [a], "x + 1", expected)
                artifact = json.loads(case["artifact"])
                category = "weak" if label.startswith("weak") else "typed_scalar"
                artifact["source"] = "def k(x, c):\n    return x + c\n"
                artifact["constants"] = [{**portable_scalar_descriptor("c", scalar), "category": category}]
                case["artifact"] = json.dumps(artifact, sort_keys=True)
                case["scalar_operands"] = [compat.describe(np.asarray(scalar))]
                case["numpy_recipe"] = {"operation": "add", "cast_to": None, "captured_category": category}
                cases.append(case)
        # Standard float divmod has correction beyond floor(x/y).
        for dtype in ("float32", "float64"):
            epsilon = np.finfo(dtype).eps
            x, y = np.array([1 + epsilon], dtype=dtype), np.array([1 - epsilon], dtype=dtype)
            cases.append(make_case(f"noncontracting-{dtype}", [x, y], "x * y - 1", x * y - 1))
            a = np.array([1, -1, 6, -6, np.inf, -np.inf, 0.0, -0.0, 3, -3], dtype=dtype)
            b = np.array([0.1, 0.1, 0.1, 0.1, 3, 3, -3, 3, np.inf, np.inf], dtype=dtype)
            for name, spelling, function in [("floor", "//", np.floor_divide), ("mod", "%", np.remainder)]:
                case = make_case(f"float-divmod-{dtype}-{name}", [a, b], f"x {spelling} y", function(a, b))
                case["comparison"]["nan"] = "equal"
                cases.append(case)
    return {
        "schema_version": compat.VERSION,
        "generator_revision": "m3-1",
        "reference": {
            "numpy": np.__version__,
            "intp_bits": np.dtype(np.intp).itemsize * 8,
            "byteorder": sys.byteorder,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "floating_errors": "ignore",
            "python": platform.python_version(),
        },
        "provenance": {"checkpoint_schema": compat.VERSION, "rng": "NumPy PCG64", "seeds": []},
        "promotions": promotions,
        "cast_matrix": cast_matrix,
        "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    args = parser.parse_args()
    args.corpus.write_text(json.dumps(generate(), indent=2) + "\n")


def property_cases(seed, count=128):
    """Full-width random bit patterns, not only small overflow-free operands."""
    rng = np.random.Generator(np.random.PCG64(seed))
    result = []
    operations = (
        "add",
        "subtract",
        "multiply",
        "floor-divide",
        "remainder",
        "shift-left",
        "shift-right",
        "less",
    )
    with np.errstate(all="ignore"):
        for index in range(count):
            dtype = np.dtype(DTYPES[1:9][index % 8])
            name = operations[(index // 8) % len(operations)]
            spelling, function = OPERATORS[name]
            arrays = [np.frombuffer(rng.bytes(dtype.itemsize * 16), dtype=dtype).copy() for _ in range(2)]
            case = make_case(
                f"arithmetic-property-{seed}-{index}", arrays, f"x {spelling} y", function(*arrays)
            )
            case["seed"] = seed
            result.append(case)
    return result


if __name__ == "__main__":
    main()
