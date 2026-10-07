"""Bounded, independent Menudet math certification (not a universal libm bound).

Normal checks need only installed blosc2 and NumPy. Regeneration additionally
needs mpmath; references must agree at 100 and 200 decimal digits. Checked-in
vectors store exact input/output encodings, with per-function ULP budgets.
Run from any directory: python scripts/certify_menudet_math.py
"""

import argparse
import json
import platform
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().with_name("menudet_math_vectors.json")


def reference_cases(mp):
    """Finite representative domains, including endpoints and cancellation."""
    unary = {
        "acos": (-1, -0.75, 0.125, 0.875, 1),
        "acosh": (1, 1.0009765625, 1.5, 8),
        "asin": (-1, -0.75, 0.125, 0.875, 1),
        "asinh": (-8, -0.125, 0.0009765625, 2),
        "atan": (-8, -0.125, 0.0009765625, 2),
        "atanh": (-0.875, -0.125, 0.0009765625, 0.875),
        "cbrt": (-8, -0.125, 0.25, 2),
        "cos": (-8, -0.125, 0.0009765625, 2),
        "cosh": (-8, -0.125, 0.0009765625, 2),
        "cospi": (-2, -0.75, 0.125, 0.5),
        "erf": (-4, -0.125, 0.0009765625, 2),
        "erfc": (-4, -0.125, 0.0009765625, 2),
        "exp": (-8, -0.125, 0.0009765625, 2),
        "exp10": (-8, -0.125, 0.0009765625, 2),
        "exp2": (-8, -0.125, 0.0009765625, 2),
        "expm1": (-8, -0.125, 0.0009765625, 2),
        "lgamma": (0.125, 0.75, 1.5, 8),
        "log": (0.0009765625, 0.75, 1.0009765625, 8),
        "log10": (0.0009765625, 0.75, 1.0009765625, 8),
        "log1p": (-0.875, -0.125, 0.0009765625, 8),
        "log2": (0.0009765625, 0.75, 1.0009765625, 8),
        "sin": (-8, -0.125, 0.0009765625, 2),
        "sinh": (-8, -0.125, 0.0009765625, 2),
        "sinpi": (-2, -0.75, 0.125, 0.5),
        "sqrt": (0, 0.125, 1.0009765625, 8),
        "tan": (-8, -0.125, 0.0009765625, 2),
        "tanh": (-8, -0.125, 0.0009765625, 2),
        "tgamma": (0.125, 0.75, 1.5, 8),
    }
    special = {
        "cbrt": lambda x: mp.sign(x) * mp.root(abs(x), 3),
        "cospi": mp.cospi,
        "exp10": lambda x: mp.power(10, x),
        "exp2": lambda x: mp.power(2, x),
        "lgamma": lambda x: mp.log(abs(mp.gamma(x))),
        "log10": lambda x: mp.log(x, 10),
        "log2": lambda x: mp.log(x, 2),
        "sinpi": mp.sinpi,
        "tgamma": mp.gamma,
    }
    budgets = {"cospi": 8, "sinpi": 8, "exp10": 8, "lgamma": 8, "tgamma": 8}
    for name, inputs in unary.items():
        function = special[name] if name in special else getattr(mp, name)
        yield name, function, [(x,) for x in inputs], budgets.get(name, 4)
    binary = {
        "atan2": (mp.atan2, [(-2, -3), (0.125, -8), (2, 3)], 4),
        "hypot": (lambda x, y: mp.sqrt(x * x + y * y), [(3, 4), (0.125, 8), (2, 3)], 4),
        "logaddexp": (lambda x, y: mp.log(mp.exp(x) + mp.exp(y)), [(-8, -8), (-8, 8), (0.125, 0.25)], 8),
        "pow": (mp.power, [(0.125, 1.5), (2, -0.75), (8, 0.125)], 4),
        "fdim": (lambda x, y: max(x - y, mp.mpf(0)), [(8, 0.125), (0.125, 8)], 0),
        "fmin": (min, [(-8, 0.125), (0.125, 8)], 0),
        "fmax": (max, [(-8, 0.125), (0.125, 8)], 0),
        "copysign": (lambda x, y: abs(x) * mp.sign(y), [(0.125, -8), (-8, 0.125)], 0),
        "fmod": (lambda x, y: x - int(x / y) * y, [(-8.125, 2), (8.125, 2)], 0),
        "remainder": (lambda x, y: x - mp.nint(x / y) * y, [(-7, 2), (7, 2)], 0),
        "ldexp": (lambda x, y: x * mp.power(2, int(y)), [(0.125, -8), (1.5, 8)], 0),
        "fma": (lambda x, y, z: x * y + z, [(1.0009765625, 0.9990234375, -1), (0.125, 8, -0.75)], 0),
    }
    for name, (function, inputs, budget) in binary.items():
        yield name, function, inputs, budget


def generate():
    import mpmath as mp

    cases = []
    for name, function, inputs, budget in reference_cases(mp):
        for dtype in ("float32", "float64"):
            for args in inputs:
                typed = [float(np.asarray(x, dtype=dtype)) for x in args]
                results = []
                for precision in (100, 200):
                    with mp.workdps(precision):
                        result = function(*(mp.mpf(x) for x in typed))
                        results.append(float(np.asarray(float(result), dtype=dtype)).hex())
                if results[0] != results[1]:
                    raise ValueError(f"Unstable reference: {name} {dtype} {typed}")
                cases.append(
                    {
                        "function": name,
                        "dtype": dtype,
                        "inputs": [x.hex() for x in typed],
                        "expected": results[0],
                        "max_ulp": budget,
                    }
                )
    return {
        "generator": f"mpmath {mp.__version__}, 100/200 decimal digits",
        "scope": "Finite inputs only; not a global accuracy or bitwise portability guarantee",
        "cases": cases,
    }


def check(record):
    import blosc2

    failures = []
    maxima = {}
    for case in record["cases"]:
        name, dtype = case["function"], case["dtype"]
        names = list("xyz"[: len(case["inputs"])])
        signature = dict.fromkeys(names, dtype)
        if name == "ldexp":
            signature["y"] = "int64"
        source = f"def k({', '.join(names)}):\n    return {name}({', '.join(names)})\n"
        artifact = blosc2.DSLKernel.from_source(source).export(signature, dtype)
        kernel = blosc2.PortableKernel.from_json(artifact, jit=False)
        inputs = {
            key: np.array([float.fromhex(value)], dtype=signature[key])
            for key, value in zip(names, case["inputs"], strict=True)
        }
        actual = float(kernel.evaluate(inputs)[0])
        expected = float.fromhex(case["expected"])
        # Use the smaller adjacent-format gap, including at powers of two. Python
        # doubles represent all tested f32/f64 gaps/differences exactly enough here.
        ref = np.asarray(expected, dtype=dtype)
        upper = float(np.nextafter(ref, np.asarray(np.inf, dtype=dtype)))
        lower = float(np.nextafter(ref, np.asarray(-np.inf, dtype=dtype)))
        gap = min(upper - expected, expected - lower)
        error = abs(actual - expected) / gap
        key = f"{name}/{dtype}"
        maxima[key] = max(maxima.get(key, 0), error)
        if not np.isfinite(actual) or error > case["max_ulp"]:
            failures.append({**case, "actual": actual.hex(), "error_ulp": error})
    return {
        "package": blosc2.__file__,
        "version": blosc2.__version__,
        "platform": platform.platform(),
        "reference": record["generator"],
        "samples": len(record["cases"]),
        "max_observed_ulp": maxima,
        "failures": failures,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", action="store_true", help="regenerate independent reference vectors")
    parser.add_argument("--report", type=Path, help="write machine-readable certification report")
    args = parser.parse_args()
    if args.generate:
        DATA.write_text(json.dumps(generate(), indent=2) + "\n")
        return
    report = check(json.loads(DATA.read_text()))
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if report["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
