"""Generate M4 native-owned signatures, accuracy and exceptional-value vectors."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from tools import menudet_arithmetic as arithmetic
from tools import menudet_conformance as compat

UNARY = {
    name: name
    for name in (
        "sin",
        "cos",
        "tan",
        "sinh",
        "cosh",
        "tanh",
        "sqrt",
        "cbrt",
        "exp",
        "exp2",
        "expm1",
        "log",
        "log2",
        "log10",
        "log1p",
        "ceil",
        "floor",
        "trunc",
        "rint",
        "round",
        "sign",
        "square",
        "conj",
        "real",
        "imag",
        "isfinite",
        "isinf",
        "isnan",
        "signbit",
        "fabs",
        "absolute",
    )
}
UNARY.update(
    {
        "abs": "absolute",
        "ln": "log",
        "acos": "arccos",
        "asin": "arcsin",
        "atan": "arctan",
        "acosh": "arccosh",
        "asinh": "arcsinh",
        "atanh": "arctanh",
    }
)
UNARY.update({name: name for name in ("arccos", "arcsin", "arctan", "arccosh", "arcsinh", "arctanh")})
BINARY = {
    name: name
    for name in (
        "minimum",
        "maximum",
        "fmin",
        "fmax",
        "hypot",
        "copysign",
        "nextafter",
        "logaddexp",
        "fmod",
        "remainder",
        "power",
        "ldexp",
    )
}
BINARY.update({"atan2": "arctan2", "arctan2": "arctan2", "pow": "power"})
EXTRA = (
    "erf",
    "erfc",
    "lgamma",
    "tgamma",
    "exp10",
    "sinpi",
    "cospi",
    "fdim",
    "fma",
    "fac",
    "ncr",
    "npr",
    "e",
    "pi",
)
EXACT = {
    "where",
    "ceil",
    "floor",
    "trunc",
    "rint",
    "round",
    "sign",
    "square",
    "conj",
    "real",
    "imag",
    "isfinite",
    "isinf",
    "isnan",
    "signbit",
    "fabs",
    "absolute",
    "abs",
    "minimum",
    "maximum",
    "fmin",
    "fmax",
    "copysign",
    "nextafter",
    "ldexp",
    "fac",
    "ncr",
    "npr",
    "e",
    "pi",
    "fma",
    "fmod",
    "remainder",
}


def reference_values(name, arrays):  # noqa: C901 -- explicit extension contracts, no source execution
    with np.errstate(all="ignore"):
        if name in UNARY:
            return np.asarray(getattr(np, UNARY[name])(arrays[0]))
        if name in BINARY:
            actual = np.asarray(getattr(np, BINARY[name])(*arrays))
            if name in {"minimum", "maximum", "fmin", "fmax"} and actual.dtype.kind == "f":
                ties = (arrays[0] == 0) & (arrays[1] == 0)
                negative = (
                    (np.signbit(arrays[0]) | np.signbit(arrays[1]))
                    if name in {"minimum", "fmin"}
                    else (np.signbit(arrays[0]) & np.signbit(arrays[1]))
                )
                actual[ties] = np.copysign(0, np.where(negative[ties], -1, 1))
            return actual
        if name == "where":
            return np.where(*arrays)
        # Extensions have explicit native contracts rather than disguised NumPy goldens.
        dtype = arrays[0].dtype
        if name in {"fac", "ncr", "npr"} and dtype.kind == "b":
            dtype = np.dtype("int64")
        elif name == "fma":
            dtype = np.dtype("float32" if dtype.name == "float32" else "float64")
        elif name not in {"fac", "ncr", "npr", "e", "pi"} and dtype.kind in "iub":
            dtype = np.dtype("float32" if dtype.itemsize == 2 else "float64")
        if name in {"e", "pi"}:
            return np.full(arrays[0].shape, getattr(math, name), dtype="float64")
        if name in {"fac", "ncr", "npr"}:
            function = {"fac": math.factorial, "ncr": math.comb, "npr": math.perm}[name]
            return np.array(
                [function(*(int(a[i]) for a in arrays)) for i in range(arrays[0].size)], dtype=dtype
            )
        if name == "fma":
            values = []
            for i in range(arrays[0].size):
                try:
                    values.append(math.fma(*(float(a[i]) for a in arrays)))
                except ValueError:
                    values.append(np.nan)
                except OverflowError:
                    values.append(np.copysign(np.inf, float(arrays[0][i]) * float(arrays[1][i])))
            return np.array(values, dtype=dtype)
        if name == "fdim":
            a, b = [x.astype(dtype) for x in arrays]
            result = np.zeros_like(a)
            selected = a > b
            result[selected] = a[selected] - b[selected]
            result[np.isnan(a) | np.isnan(b)] = np.nan
            return result
        if name == "exp10":
            return np.power(np.array(10, dtype=dtype), arrays[0])
        if name in {"sinpi", "cospi"}:
            import mpmath as mp

            with mp.workdps(100):
                function = mp.sinpi if name == "sinpi" else mp.cospi
                result = np.array([float(function(float(x))) for x in arrays[0]], dtype=dtype)
                if name == "sinpi":
                    zeros = result == 0
                    result[zeros] = np.copysign(0, arrays[0][zeros])
                return result
        function = {"tgamma": math.gamma}.get(name, getattr(math, name, None))
        values = []
        for x in arrays[0]:
            try:
                values.append(function(float(x)))
            except ValueError:
                values.append(np.inf if name == "lgamma" else np.copysign(np.inf, x) if x == 0 else np.nan)
            except OverflowError:
                values.append(np.inf)
        return np.array(values, dtype=dtype)


def reference_recipe(case):
    return compat.describe(
        reference_values(case["function_recipe"], [compat.checkpoint.decode(x) for x in case["inputs"]])
    )


def make_case(identifier, name, arrays, expected=None, diagnostic=None):
    identifier = identifier.replace("_", "-")
    count = 0 if name in {"pi", "e"} else len(arrays)
    expression = f"{name}({', '.join(('x', 'y', 'z')[:count])})"
    case = compat.make_case(
        identifier,
        "identity",
        "x",
        arrays,
        output=(expected.dtype.name if expected is not None else "float64"),
    )
    artifact = json.loads(case["artifact"])
    artifact.update(
        schema_version="1.1",
        language={"name": "miniexpr", "version": "1.1"},
        source=f"def k({', '.join(x['name'] for x in case['inputs'])}):\n    return {expression}\n",
        semantics={"fp": "strict", "numeric": "numpy-2.5", "casting": "unsafe"},
    )
    case.update(
        artifact=json.dumps(artifact, sort_keys=True),
        operation=name,
        semantic_revision="menudet-numpy-1.1",
        reference_kind="numpy" if diagnostic is None and name not in EXTRA else "native-contract",
    )
    if diagnostic:
        case["expected"] = {"diagnostic": {"category": diagnostic}}
    else:
        case["function_recipe"] = name
        case["expected"] = compat.describe(expected)
        case["inferred_dtype"] = expected.dtype.name
    case["comparison"]["nan"] = "equal"
    if (
        name in {"minimum", "maximum", "fmin", "fmax"}
        and expected is not None
        and expected.dtype.kind == "f"
    ):
        case["reference_kind"] = "native-contract"
        case["divergence"] = {
            "id": "ieee-extrema-zero",
            "reason": "Signed-zero ties use deterministic IEEE minimum/maximum; NumPy tie results vary by loop/platform",
        }
    if name not in EXACT:
        case["comparison"].update(kind="ulp", max_ulp=8)
    return case


def generate():  # noqa: C901 -- enumerated signatures, domains, exceptions and status vectors
    if np.__version__ != compat.checkpoint.REFERENCE:
        raise RuntimeError("Function generation requires pinned NumPy 2.5.3")
    cases, signatures = [], []
    for name in (*UNARY, *BINARY):
        for dtype in arithmetic.DTYPES:
            a = np.array([0, 1, 2], dtype=dtype)
            arrays = (
                [a]
                if name in UNARY
                else [a, np.array([1, 1, 2], dtype="int32" if name == "ldexp" else dtype)]
            )
            try:
                expected = reference_values(name, arrays)
                diagnostic = "invalid_source" if expected.dtype.name == "float16" else None
            except (TypeError, ValueError):
                expected, diagnostic = None, "invalid_source"
            signatures.append(
                {
                    "function": name,
                    "inputs": [x.dtype.name for x in arrays],
                    "result": expected.dtype.name if expected is not None else None,
                    "supported": diagnostic is None,
                    "reason": diagnostic,
                }
            )
            cases.append(
                make_case(
                    f"signature-{name}-{dtype}",
                    name,
                    arrays,
                    expected if not diagnostic else None,
                    diagnostic,
                )
            )
    for name in EXTRA:
        for dtype in arithmetic.DTYPES:
            a = np.ones(2, dtype=dtype)
            arrays = [a]
            if name in {"fdim", "fma", "ncr", "npr"}:
                arrays.append(a.copy())
            if name == "fma":
                arrays.append(a.copy())
            diagnostic = None
            if name in {"fac", "ncr", "npr"} and a.dtype.kind == "f":
                diagnostic = "invalid_source"
            if name not in {"fac", "ncr", "npr", "e", "pi", "fma"} and dtype in {"bool", "int8", "uint8"}:
                diagnostic = "invalid_source"
            expected = None if diagnostic else reference_values(name, arrays)
            signatures.append(
                {
                    "function": name,
                    "inputs": [x.dtype.name for x in arrays],
                    "result": None if expected is None else expected.dtype.name,
                    "supported": diagnostic is None,
                    "reason": diagnostic,
                    "reference_kind": "native-contract",
                }
            )
            cases.append(
                make_case(f"signature-extension-{name}-{dtype}", name, arrays, expected, diagnostic)
            )
    for dtype in arithmetic.DTYPES:
        arrays = [
            np.array([True, False, True]),
            np.array([0, 1, 2], dtype=dtype),
            np.array([2, 1, 0], dtype=dtype),
        ]
        expected = reference_values("where", arrays)
        signatures.append(
            {
                "function": "where",
                "inputs": [a.dtype.name for a in arrays],
                "result": dtype,
                "supported": True,
                "reason": None,
            }
        )
        cases.append(make_case(f"signature-where-{dtype}", "where", arrays, expected))
    for dtype in ("float32", "float64"):
        info = np.finfo(dtype)
        # Signed zeros, normal/subnormal boundaries, NaNs, infinities, domain edges.
        a = np.array(
            [
                -np.inf,
                -info.max,
                -2,
                -1,
                -0.5,
                -0.0,
                0.0,
                info.smallest_subnormal,
                info.tiny,
                0.5,
                1,
                2,
                info.max,
                np.inf,
                np.nan,
            ],
            dtype=dtype,
        )
        b = np.array(
            [
                np.nan,
                0,
                0,
                2,
                -0.0,
                0.0,
                -0.0,
                info.tiny,
                info.smallest_subnormal,
                2,
                -1,
                np.inf,
                -info.max,
                np.inf,
                np.nan,
            ],
            dtype=dtype,
        )
        for name in (*UNARY, *BINARY):
            arrays = [a] if name in UNARY else [a, b]
            if name == "ldexp":
                arrays = [
                    a,
                    np.array([-2000, 2000, -1, 0, 1, 0, 0, -1, -1, 1, 2000, 2000, 1, 0, 0], dtype="int32"),
                ]
            cases.append(
                make_case(f"exceptional-{name}-{dtype}", name, arrays, reference_values(name, arrays))
            )
        for name in ("rint", "round"):
            values = np.array([-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5], dtype=dtype)
            cases.append(make_case(f"ties-{name}-{dtype}", name, [values], reference_values(name, [values])))
        for name in EXTRA:
            arrays = [np.array([0.25, 0.5, 1, 2], dtype=dtype)]
            if name in {"fac", "ncr", "npr"}:
                arrays = [np.array([1, 2, 3, 4], dtype="int64")]
            if name in {"fdim", "fma", "ncr", "npr"}:
                arrays.append(np.ones(4, dtype=arrays[0].dtype))
            if name == "fma":
                arrays.append(-np.ones(4, dtype=dtype))
            cases.append(
                make_case(f"extension-{name}-{dtype}", name, arrays, reference_values(name, arrays))
            )
            if name not in {"fac", "ncr", "npr", "e", "pi"}:
                inputs = [a]
                if name in {"fdim", "fma"}:
                    inputs.append(b)
                if name == "fma":
                    inputs.append(np.zeros_like(a))
                cases.append(
                    make_case(
                        f"extension-exceptional-{name}-{dtype}", name, inputs, reference_values(name, inputs)
                    )
                )
    for name in ("fac", "ncr", "npr"):
        arrays = [np.array([-1, 0, 30], dtype="int64")]
        if name != "fac":
            arrays.append(np.array([1, 1, 15], dtype="int64"))
        cases.append(make_case(f"extension-domain-{name}", name, arrays, diagnostic="evaluation_error"))
        cases.append(
            make_case(
                f"extension-float-reject-{name}",
                name,
                [x.astype("float64") for x in arrays],
                diagnostic="invalid_source",
            )
        )
    for dtype in ("float32", "float64"):
        cond = np.array([True, False, True])
        arrays = [cond, np.array([1, np.nan, -0.0], dtype=dtype), np.array([np.nan, 2, 0.0], dtype=dtype)]
        cases.append(make_case(f"where-{dtype}", "where", arrays, reference_values("where", arrays)))
    # Values and reporting must be independent of the caller's current IEEE flags.
    for label, expression, values, expected, flags in [
        ("invalid", "sqrt(x)", [-1.0], [np.nan], 1),
        ("divide", "log(x)", [0.0], [-np.inf], 2),
        ("overflow", "exp(x)", [1000.0], [np.inf], 4),
        ("underflow", "exp(x)", [-1000.0], [0.0], 8),
        ("clear", "sqrt(x)", [4.0], [2.0], 0),
        ("lazy-where", "where(x > 0, sqrt(x), 0)", [-1.0, 4.0], [0.0, 2.0], 0),
    ]:
        case = make_case(f"status-{label}", "sqrt", [np.array(values)], np.array(expected))
        artifact = json.loads(case["artifact"])
        artifact["source"] = f"def k(x):\n    return {expression}\n"
        case.update(
            artifact=json.dumps(artifact, sort_keys=True),
            fp_expected=flags,
            reference_kind="native-contract",
        )
        case.pop("function_recipe", None)
        if label == "lazy-where":
            case["divergence"] = {
                "id": "lazy-where",
                "reason": "Only selected lanes are evaluated, unlike eager NumPy argument evaluation",
            }
        cases.append(case)
    for name in (*UNARY, *BINARY, *EXTRA, "where"):
        arity = (
            0
            if name in {"e", "pi"}
            else 3
            if name in {"fma", "where"}
            else 2
            if name in BINARY or name in {"fdim", "ncr", "npr"}
            else 1
        )
        case = make_case(f"arity-reject-{name}", name, [np.array([1.0])], diagnostic="invalid_source")
        artifact = json.loads(case["artifact"])
        artifact["source"] = f"def k(x):\n    return {name}({', '.join(['x'] * (arity + 1))})\n"
        case["artifact"] = json.dumps(artifact, sort_keys=True)
        cases.append(case)
    for name in ("float_power", "logaddexp2", "heaviside", "spacing", "frexp", "modf"):
        cases.append(
            make_case(f"spelling-reject-{name}", name, [np.array([1.0])], diagnostic="invalid_source")
        )
    certified = certified_cases()
    cases.extend(certified)
    metadata = arithmetic.generate()
    return {
        **{k: v for k, v in metadata.items() if k not in {"cases", "promotions", "cast_matrix"}},
        "generator_revision": "m4-1",
        "signatures": signatures,
        "cases": cases,
        "certification": {
            "reference": "mpmath",
            "decimal_precision": 100,
            "seed": 20261009,
            "cases": len(certified),
            "samples_per_case": 32,
            "max_ulp": 8,
        },
        "unsupported_spellings": [
            "fmin.reduce",
            "maximum.reduce",
            "float_power",
            "logaddexp2",
            "heaviside",
            "spacing",
            "frexp",
            "modf",
        ],
        "divergences": {
            "where": "lazy selected-lane evaluation",
            "round": "one argument; decimals not implemented",
            "float16": "NumPy loops producing float16 reject",
            "extensions": "libm/native names without NumPy equivalents use explicit contracts",
        },
    }


def certified_cases():
    """Independent finite-reference samples; bounded accuracy, not a proof over R."""
    import mpmath as mp

    functions = {
        name: getattr(mp, name)
        for name in (
            "sqrt",
            "exp",
            "expm1",
            "log",
            "log10",
            "log1p",
            "sin",
            "cos",
            "tan",
            "asin",
            "acos",
            "atan",
            "sinh",
            "cosh",
            "tanh",
            "asinh",
            "acosh",
            "atanh",
            "erf",
            "erfc",
            "sinpi",
            "cospi",
        )
    }
    functions.update(
        exp2=lambda x: mp.power(2, x),
        exp10=lambda x: mp.power(10, x),
        log2=lambda x: mp.log(x, 2),
        cbrt=lambda x: mp.sign(x) * mp.root(abs(x), 3),
        lgamma=lambda x: mp.log(abs(mp.gamma(x))),
        tgamma=mp.gamma,
        hypot=lambda x, y: mp.sqrt(x * x + y * y),
        atan2=mp.atan2,
        logaddexp=lambda x, y: mp.log(mp.exp(x) + mp.exp(y)),
        fma=lambda x, y, z: x * y + z,
    )
    rng = np.random.Generator(np.random.PCG64(20261009))
    cases = []
    with mp.workdps(100):
        for dtype in ("float32", "float64"):
            for name, function in functions.items():
                values = rng.uniform(-0.8, 0.8, 32)
                if name in {"sqrt", "log", "log2", "log10", "lgamma", "tgamma", "logaddexp"}:
                    values = rng.uniform(0.1, 8, 32)
                if name == "acosh":
                    values = rng.uniform(1, 8, 32)
                arrays = [values.astype(dtype)]
                if name in {"hypot", "atan2", "logaddexp", "fma"}:
                    arrays.append(rng.uniform(0.1, 8, 32).astype(dtype))
                if name == "fma":
                    arrays.append(rng.uniform(-0.8, 0.8, 32).astype(dtype))
                expected = np.array(
                    [float(function(*(mp.mpf(float(a[i])) for a in arrays))) for i in range(32)], dtype=dtype
                )
                case = make_case(f"certified-{name}-{dtype}", name, arrays, expected)
                case.update(
                    reference_kind="native-contract",
                    seed=20261009,
                    certification={"reference": "mpmath-100-dps", "max_ulp": 8},
                )
                case.pop("function_recipe", None)
                if name != "fma":
                    case["comparison"].update(kind="ulp", max_ulp=8)
                cases.append(case)
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    args = parser.parse_args()
    args.corpus.write_text(json.dumps(generate(), indent=2) + "\n")


if __name__ == "__main__":
    main()
