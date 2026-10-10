"""M2 shared, versioned conformance tooling (NumPy is generation tooling only).

Use ``python -m tools.menudet_conformance``. Native corpus locations and executable
paths must be explicit. Generation/recording is separate from regression checking.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import platform
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from tools import menudet_numpy_compat as checkpoint

VERSION = "menudet-numpy-vectors-2"
CAPABILITIES = {
    "numeric",
    "control-flow",
    "block-reductions",
    "nd-context",
    "fixed-strings",
    "layout-copying",
}
STATUS = {
    0: "success",
    -1: "invalid_artifact",
    -2: "unsupported_requirement",
    -3: "invalid_source",
    -4: "binding_error",
    -5: "evaluation_error",
    -6: "out_of_memory",
}
BITWISE = {"kind": "bitwise", "atol": 0.0, "rtol": 0.0, "max_ulp": 0, "nan": "bits", "signed_zero": "exact"}


def describe(array, name=None, layout=None):
    result = checkpoint.buffer(array, name)
    if name is not None:
        result["category"] = "zero_dimensional_array" if result["shape"] == [] else "array"
        result["layout"] = layout or {"order": "C", "byteorder": "native", "step": 1, "reverse_axis": None}
    if np.asarray(array).dtype.kind == "f":
        result["classes"] = [
            "nan"
            if np.isnan(x)
            else "negative_inf"
            if np.isneginf(x)
            else "positive_inf"
            if np.isposinf(x)
            else "negative_zero"
            if x == 0 and np.signbit(x)
            else "positive_zero"
            if x == 0
            else "subnormal"
            if abs(x) < np.finfo(np.asarray(array).dtype).tiny
            else "finite"
            for x in np.asarray(array).ravel()
        ]
    return result


def decode(item):
    """Reconstruct logical values in the declared physical layout, without eval."""
    array = checkpoint.decode(item)
    layout = item.get("layout")
    if not isinstance(layout, dict):
        return array
    if layout["order"] not in {"C", "F"} or layout["byteorder"] not in {"native", "little", "big"}:
        raise ValueError("Invalid layout")
    step = layout["step"]
    if type(step) is not int or not 1 <= step <= 4:
        raise ValueError("Invalid layout step")
    axis = layout["reverse_axis"]
    if axis is not None and (type(axis) is not int or not 0 <= axis < array.ndim):
        raise ValueError("Invalid reverse axis")
    dtype = array.dtype.newbyteorder({"native": "=", "little": "<", "big": ">"}[layout["byteorder"]])
    shape = list(array.shape)
    if shape:
        shape[-1] *= step
    backing = np.zeros(shape, dtype=dtype, order=layout["order"])
    view = backing[..., ::step] if shape else backing
    if axis is not None:
        slices = [slice(None)] * view.ndim
        slices[axis] = slice(None, None, -1)
        view = view[tuple(slices)]
    view[...] = array
    return view


def ulp_distance(a, b, dtype):
    unsigned = np.dtype("uint32" if np.dtype(dtype).itemsize == 4 else "uint64")
    bits = unsigned.itemsize * 8
    sign = 1 << (bits - 1)
    mask = (1 << bits) - 1

    def ordered(value):
        raw = int(np.asarray(value, dtype=dtype).view(unsigned))
        return (~raw & mask) if raw & sign else raw | sign

    return abs(ordered(a) - ordered(b))


def compare(actual, expected, policy):
    """Same comparison contract as the native runner, including special values."""
    if actual["dtype"] != expected["dtype"]:
        return ["dtype"]
    if actual["shape"] != expected["shape"]:
        return ["shape"]
    a, e = checkpoint.decode(actual).ravel(), checkpoint.decode(expected).ravel()
    dtype = a.dtype
    for index, (left, right) in enumerate(zip(a, e, strict=True)):
        if dtype.kind != "f":
            equal = left == right
        elif np.isnan(left) or np.isnan(right):
            equal = (
                np.isnan(left)
                and np.isnan(right)
                and (
                    policy["nan"] == "equal"
                    or checkpoint.encode(np.asarray(left)) == checkpoint.encode(np.asarray(right))
                )
            )
        elif left == 0 and right == 0:
            equal = policy["signed_zero"] == "ignore" or np.signbit(left) == np.signbit(right)
        elif np.isinf(left) or np.isinf(right):
            equal = left == right
        elif policy["kind"] == "bitwise":
            equal = checkpoint.encode(np.asarray(left)) == checkpoint.encode(np.asarray(right))
        elif policy["kind"] == "exact":
            equal = left == right
        elif policy["kind"] == "ulp":
            equal = ulp_distance(left, right, dtype) <= policy["max_ulp"]
        else:
            equal = (
                abs(float(left) - float(right)) <= policy["atol"] + policy["rtol"] * abs(float(right))
                or ulp_distance(left, right, dtype) <= policy["max_ulp"]
            )
        if not equal:
            return [f"value:{index}"]
    return []


def classify(case, result):
    if result.get("outcome") == "skipped":
        return result
    expected = case["expected"]
    errors = (
        (["diagnostic"] if result["category"] != expected["diagnostic"]["category"] else [])
        if "diagnostic" in expected
        else (
            ["diagnostic"] if result["status"] else compare(result["output"], expected, case["comparison"])
        )
    )
    baseline = case.get("baseline")
    regression = False
    if baseline:
        regression = result["status"] != baseline["status"]
        if result["status"] == 0 and baseline["status"] == 0:
            regression |= bool(compare(result["output"], baseline["output"], case["comparison"]))
    result.update(reference_match=not errors, mismatches=errors, baseline_regression=regression)
    result["outcome"] = (
        "regression"
        if regression
        else "known_divergence"
        if errors and baseline and case.get("divergence")
        else "mismatch"
        if errors
        else "matching"
    )
    return result


def reference(case):
    if case["reference_kind"] == "native-contract":
        return case["expected"]
    if "numpy_recipe" in case:
        from tools.menudet_arithmetic import reference_recipe

        return reference_recipe(case)
    if "function_recipe" in case:
        from tools.menudet_functions import reference_recipe

        return reference_recipe(case)
    arrays = [checkpoint.decode(item) for item in case["inputs"]]
    literal = case.get("literal")
    value = None if not literal else {"int": int, "float": float}[literal["type"]](literal["decimal"])
    if case.get("scalar_operands"):
        value = checkpoint.decode(case["scalar_operands"][0])[()]
    with np.errstate(all="ignore"):
        try:
            if case["operation"] == "identity":
                actual = arrays[0]
            elif case["operation"] in {"cos", "log", "sqrt", "exp"}:
                actual = getattr(np, case["operation"])(arrays[0])
            elif case["operation"] == "sum":
                actual = np.sum(arrays[0], dtype=np.dtype(json.loads(case["artifact"])["output"]["dtype"]))
            else:
                actual = checkpoint.reference(case["operation"], arrays, value)
        except OverflowError:
            return {"diagnostic": {"category": "invalid_source", "numpy_category": "OverflowError"}}
    return describe(actual)


def reference_expected(case):
    """Select a reviewed, exact host observation, never the native baseline.

    NumPy's nonfinite-to-integer sentinel is explicitly platform-qualified in
    the corpus. Native execution still must reject it with the recorded status;
    only the independent NumPy drift check uses this host-specific observation.
    Unlisted machines retain the original reference and therefore fail on drift.
    """
    variants = case.get("reference_expected_by_machine", {})
    if variants and not case.get("platform_qualification"):
        raise ValueError("Host reference variants require a platform qualification")
    machine = platform.machine().lower()
    machine = {"amd64": "x86_64", "aarch64": "arm64"}.get(machine, machine)
    return variants.get(machine, case["expected"])


def make_case(
    identifier, operation, expression, arrays, output=None, literal=None, policy=None, layout=None
):
    names = ["x", "y", "z"][: len(arrays)]
    artifact = {
        "schema_version": "1.0",
        "language": {"name": "miniexpr", "version": "1.0"},
        "requires": ["numeric"],
        "source": f"def k({', '.join(names)}):\n    return {expression}\n",
        "entry_point": "k",
        "inputs": [{"name": n, "dtype": a.dtype.name} for n, a in zip(names, arrays, strict=True)],
        "constants": [],
        "output": {"dtype": output or arrays[0].dtype.name, "contract": "elementwise"},
        "semantics": {"fp": "strict"},
        "context": {"ndim": 0},
        "metadata": {},
    }
    case = {
        "id": identifier,
        "operation": operation,
        "semantic_revision": "menudet-draft-1.0-checked",
        "requires": ["numeric"],
        "artifact": json.dumps(artifact, sort_keys=True),
        "inputs": [describe(a, n, layout) for n, a in zip(names, arrays, strict=True)],
        "scalar_operands": [],
        "literal": None
        if literal is None
        else {"category": "weak", "type": type(literal).__name__, "decimal": str(literal)},
        "comparison": policy or copy.deepcopy(BITWISE),
        "reference_kind": "numpy",
        "seed": None,
    }
    case["expected"] = reference(case)
    return case


def generate(checkpoint_path):
    if np.__version__ != checkpoint.REFERENCE:
        raise RuntimeError("Generation requires NumPy 2.5.3")
    previous = json.loads(checkpoint_path.read_text())
    if checkpoint.reference_drift(previous)["differences"]:
        raise ValueError("Pinned reference has drifted; do not silently regenerate")
    cases = []
    for original in previous["cases"]:
        case = copy.deepcopy(original)
        case.update(reference_kind="numpy", seed=None, comparison=copy.deepcopy(BITWISE))
        case["inputs"] = [describe(checkpoint.decode(item), item["name"]) for item in original["inputs"]]
        case["expected"] = describe(checkpoint.decode(original["expected"]))
        if case["id"] == "float64-cast-infinity":
            # Exact observations from the pinned NumPy ARM64 and x64 builds;
            # keep the native diagnostic/divergence baseline unchanged.
            case["reference_expected_by_machine"] = {
                machine: {**case["expected"], "hex": bits}
                for machine, bits in {"x86_64": "8000000000000000", "arm64": "7fffffffffffffff"}.items()
            }
        case["scalar_operands"] = copy.deepcopy(original.get("scalar_operands", []))
        old = case["baseline"]
        case["baseline"] = {"status": old["status"], "category": STATUS[old["status"]]}
        if old["status"] == 0:
            case["baseline"]["output"] = {
                "dtype": json.loads(case["artifact"])["output"]["dtype"],
                "shape": original["inputs"][0]["shape"],
                "encoding": "raw-be-hex",
                "hex": old["hex"],
            }
        if original["classification"] == "divergent":
            case["divergence"] = {
                "id": "checkpoint-" + original["id"],
                "reason": "Reviewed current draft divergence; see native checkpoint divergence register",
            }
        cases.append(case)
    for dtype in ("float32", "float64"):
        for operation in ("sin", "cos", "log", "sqrt", "exp"):
            values = np.array([0.125, 0.5, 1, 2, 8], dtype=dtype)
            cases.append(
                make_case(
                    f"{dtype}-{operation}-finite-ulp",
                    operation,
                    f"{operation}(x)",
                    [values],
                    policy={**BITWISE, "kind": "ulp", "max_ulp": 2, "nan": "equal"},
                )
            )
        special = np.array(
            [
                0.0,
                -0.0,
                np.inf,
                -np.inf,
                np.nan,
                np.nextafter(np.array(0, dtype=dtype), np.array(1, dtype=dtype)),
            ],
            dtype=dtype,
        )
        cases.append(make_case(f"{dtype}-identity-special", "identity", "x", [special]))
        cases.append(
            make_case(
                f"{dtype}-sqrt-domain",
                "sqrt",
                "sqrt(x)",
                [np.array([-1, -0.0, 4, np.inf, np.nan], dtype=dtype)],
                policy={**BITWISE, "kind": "exact", "nan": "equal"},
            )
        )
        payload = "7fc01234ffc05678" if dtype == "float32" else "7ff8000000001234fff8000000005678"
        values = checkpoint.decode({"dtype": dtype, "shape": [2], "hex": payload})
        cases.append(make_case(f"{dtype}-nan-payload-identity", "identity", "x", [values]))
    cases.append(
        make_case(
            "float64-exp-tolerance",
            "exp",
            "exp(x)",
            [np.array([-1, 0, 1], dtype="float64")],
            policy={**BITWISE, "kind": "tolerance", "atol": 1e-15, "rtol": 1e-15, "nan": "equal"},
        )
    )
    for label, layout in [
        ("fortran", {"order": "F", "byteorder": "native", "step": 1, "reverse_axis": None}),
        ("negative-stride", {"order": "C", "byteorder": "native", "step": 1, "reverse_axis": 0}),
        ("stepped-big-endian", {"order": "F", "byteorder": "big", "step": 2, "reverse_axis": 1}),
        ("little-endian", {"order": "C", "byteorder": "little", "step": 1, "reverse_axis": None}),
    ]:
        cases.append(
            make_case(
                f"layout-{label}",
                "identity",
                "x",
                [np.arange(6, dtype="float32").reshape(2, 3)],
                layout=layout,
            )
        )
    cases.append(make_case("empty-elementwise", "identity", "x", [np.empty((2, 0), dtype="int64")]))
    cases.append(
        make_case("weak-scalar-out-of-range", "add", "x + 128", [np.array([1], dtype="int8")], literal=128)
    )
    # Native validation/binding contracts are not represented as NumPy arithmetic.
    for label, change, diagnostic in [
        ("future-artifact", "schema_version", "unsupported_requirement"),
        ("invalid-source", "source", "invalid_source"),
        ("wrong-binding", "binding", "binding_error"),
    ]:
        case = make_case(f"diagnostic-{label}", "identity", "x", [np.array([1], dtype="int64")])
        artifact = json.loads(case["artifact"])
        if change == "schema_version":
            artifact[change] = "1.1"
        elif change == "source":
            artifact[change] = "def k(x):\n    return missing_name\n"
        else:
            artifact["inputs"][0]["dtype"] = "float64"
        case.update(
            artifact=json.dumps(artifact, sort_keys=True),
            reference_kind="native-contract",
            expected={"diagnostic": {"category": diagnostic}},
        )
        cases.append(case)
    recovery = make_case(
        "failure-then-success-same-handle", "add", "x + 1", [np.array([127], dtype="int8")], literal=1
    )
    recovery["divergence"] = {"id": "D01", "reason": "Checked integer overflow instead of NumPy wrapping"}
    recovery["recovery"] = {
        "inputs": [describe(np.array([1], dtype="int8"), "x")],
        "expected": describe(np.array([2], dtype="int8")),
    }
    cases.append(recovery)
    reduced = make_case(
        "block-sum-explicit-int64", "sum", "block_sum(x)", [np.arange(6, dtype="int64").reshape(2, 3)]
    )
    artifact = json.loads(reduced["artifact"])
    artifact["requires"].append("block-reductions")
    artifact["output"]["contract"] = "block_scalar"
    reduced.update(artifact=json.dumps(artifact, sort_keys=True), requires=["numeric", "block-reductions"])
    cases.append(reduced)
    complex_case = make_case(
        "capability-complex-skipped", "identity", "x", [np.array([1 + 2j], dtype="complex64")]
    )
    complex_case["requires"] = ["numpy-complex"]
    complex_case["skip_reason"] = (
        "Complex numerical semantics are explicitly outside the current native capability set"
    )
    cases.append(complex_case)
    broadcast = make_case(
        "capability-broadcast-skipped",
        "add",
        "x + y",
        [np.array([[1], [2]], dtype="float32"), np.array([[2, 3, 4]], dtype="float32")],
    )
    broadcast.update(
        requires=["numeric", "array-broadcasting"],
        skip_reason="Native array broadcasting is milestone 5, not an implemented kernel capability",
    )
    cases.append(broadcast)
    # Stable minimized property regression, independently replayed by the fuzz command.
    minimized = make_case(
        "seed-1729-minimized-int8-overflow",
        "add",
        "x + y",
        [np.array([127], dtype="int8"), np.array([1], dtype="int8")],
    )
    minimized.update(
        seed=1729,
        minimized_from="seeded boundary property",
        divergence={"id": "D01", "reason": "Checked integer overflow instead of NumPy wrapping"},
    )
    cases.append(minimized)
    return {
        "schema_version": VERSION,
        "generator_revision": "m2-1",
        "reference": previous["reference"],
        "provenance": {
            "checkpoint_schema": previous["schema_version"],
            "rng": "NumPy PCG64",
            "seeds": [1729],
        },
        "cases": cases,
    }


def validate(corpus):
    """Bounded execution schema checks; JSON Schema is also checked in natively."""
    if corpus.get("schema_version") != VERSION or not corpus.get("cases"):
        raise ValueError("Unsupported or empty vector schema")
    ids = set()
    for case in corpus["cases"]:
        identifier = case["id"]
        if (
            not identifier
            or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in identifier)
            or identifier in ids
        ):
            raise ValueError("Invalid or duplicate case ID")
        ids.add(identifier)
        if case["semantic_revision"] not in {"menudet-draft-1.0-checked", "menudet-numpy-1.1"}:
            raise ValueError("Unsupported semantic revision")
        policy = case["comparison"]
        if (
            policy["kind"] not in {"bitwise", "exact", "ulp", "tolerance"}
            or policy["nan"] not in {"bits", "equal"}
            or policy["signed_zero"] not in {"exact", "ignore"}
        ):
            raise ValueError("Invalid comparison policy")
        if (
            any(not math.isfinite(policy[key]) or policy[key] < 0 for key in ("atol", "rtol"))
            or type(policy["max_ulp"]) is not int
            or not 0 <= policy["max_ulp"] <= 2**64 - 1
        ):
            raise ValueError("Invalid comparison threshold")
        if not isinstance(case["requires"], list) or any(
            not isinstance(cap, str)
            or not cap
            or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-:._" for c in cap)
            for cap in case["requires"]
        ):
            raise ValueError("Invalid capabilities")
        for item in case["inputs"]:
            if (
                len(item["shape"]) > 8
                or any(type(dim) is not int or not 0 <= dim <= 4096 for dim in item["shape"])
                or math.prod(item["shape"]) > 4096
            ):
                raise ValueError("Invalid input extent")
            # Unknown capabilities skip execution but never excuse corrupt transport.
            if (
                item["encoding"] != "raw-be-hex"
                or len(bytes.fromhex(item["hex"]))
                != math.prod(item["shape"]) * np.dtype(item["dtype"]).itemsize
            ):
                raise ValueError("Invalid input encoding")
            decode(item)


def integrate(corpus, jit=False):  # noqa: C901 -- paired execution, diagnostics, status and recovery checks
    import blosc2

    validate(corpus)
    results = []
    for case in corpus["cases"]:
        unsupported = sorted(set(case["requires"]) - CAPABILITIES)
        if unsupported:
            results.append(
                {
                    "id": case["id"],
                    "outcome": "skipped",
                    "backend": "none",
                    "skipped_capabilities": unsupported,
                    "skip_reason": case.get("skip_reason", "unsupported required capability"),
                }
            )
            continue
        kernel = None
        actual = None
        recovery_ok = True
        try:
            kernel = blosc2.PortableKernel.from_json(case["artifact"], jit=jit)
            for _ in range(3):
                current = kernel.evaluate(
                    {item["name"]: decode(item) for item in case["inputs"]},
                    return_status="fp_expected" in case,
                )
                if "fp_expected" in case:
                    current, fp_status = current
                    assert not fp_status["supported"] or fp_status["flags"] == case["fp_expected"], case[
                        "id"
                    ]
                if actual is not None:
                    assert checkpoint.encode(current) == checkpoint.encode(actual), case["id"]
                actual = current
            result = {"id": case["id"], "status": 0, "category": "success", "output": describe(actual)}
            if case.get("inferred_dtype"):
                assert kernel.inferred_dtype == np.dtype(case["inferred_dtype"]), case["id"]
        except blosc2.PortableArtifactError as error:
            status = next(code for code, category in STATUS.items() if category == error.status)
            # Replay the failure through the same owned handle when one was loaded.
            if kernel is not None:
                for _ in range(2):
                    try:
                        kernel.evaluate({item["name"]: decode(item) for item in case["inputs"]})
                    except blosc2.PortableArtifactError as again:
                        if again.status != error.status or again.native_status != error.native_status:
                            raise AssertionError("Repeated diagnostic changed") from again
                    else:
                        raise AssertionError("A repeated failure changed its diagnostic") from error
            result = {
                "id": case["id"],
                "status": status,
                "category": error.status,
                "native_status": error.native_status,
            }
        if case.get("recovery"):
            if kernel is None:
                raise AssertionError("Recovery requires a compiled handle")
            recovery = case["recovery"]
            recovered = kernel.evaluate({item["name"]: decode(item) for item in recovery["inputs"]})
            recovery_ok = not compare(describe(recovered), recovery["expected"], case["comparison"])
            assert recovery_ok, case["id"]
        result.update(
            backend="jit" if kernel and kernel.has_jit else "interpreter" if kernel else "none",
            requested_backend="jit" if jit else "interpreter",
            jit_eligible=bool(kernel and kernel.has_jit),
            jit_skip_reason=""
            if kernel and kernel.has_jit
            else "no compiled route for this artifact/request",
            recovery_ok=recovery_ok,
        )
        results.append(classify(case, result))
    return results


def native(corpus, runner, work_dir, jit=False, observe=False):
    validate(corpus)
    # Close the input before launching the native reader. Windows cannot reopen
    # a live NamedTemporaryFile with its default delete/share mode.
    with tempfile.TemporaryDirectory(dir=work_dir) as directory:
        path = Path(directory) / "corpus.json"
        path.write_text(json.dumps(corpus))
        command = [str(runner.resolve()), str(path), "on" if jit else "off"] + (
            ["observe"] if observe else []
        )
        proc = subprocess.run(command, capture_output=True, text=True, check=False)
    rows = [json.loads(line) for line in proc.stdout.splitlines()]
    if len(rows) != len(corpus["cases"]):
        raise AssertionError(f"Native runner did not report every case: {proc.stderr}")
    if proc.returncode and not any(
        row.get("outcome") in {"mismatch", "regression"}
        or not row.get("environment_restored", True)
        or not row.get("recovery_ok", True)
        for row in rows
    ):
        raise AssertionError(f"Invalid native vector or runner failure: {proc.stderr}")
    return rows


def paired(corpus, runner, work_dir, observe=False):
    validate(corpus)
    reports = {}
    for jit in (False, True):
        c_results = native(corpus, runner, work_dir, jit, observe)
        p_results = integrate(corpus, jit)
        for case, c, p in zip(corpus["cases"], c_results, p_results, strict=True):
            assert case["id"] == c["id"] == p["id"]
            if p["outcome"] == "skipped":
                assert c["outcome"] == "skipped", (c, p)
                assert c["skipped_capabilities"] == p["skipped_capabilities"]
                continue
            assert c["status"] == p["status"], (c, p)
            assert c["backend"] == p["backend"], (c, p)
            assert c["reference_match"] == p["reference_match"], (c, p)
            assert c["outcome"] == p["outcome"], (c, p)
            assert c["environment_restored"]
            assert c["recovery_ok"]
            assert p["recovery_ok"]
            if not c["status"]:
                assert not compare(c["output"], p["output"], case["comparison"]), (c, p)
        reports["on" if jit else "off"] = {"native": c_results, "python": p_results}
    drift = []
    for case in corpus["cases"]:
        actual = reference(case)
        expected = reference_expected(case)
        mismatch = (
            actual != expected
            if "diagnostic" in expected
            else bool(compare(actual, expected, case["comparison"]))
        )
        if mismatch:
            drift.append({"id": case["id"], "actual": actual})
    import blosc2

    try:
        git = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=False,
        )
        python_revision = git.stdout.strip() if git.returncode == 0 else None
    except OSError:
        python_revision = None
    return {
        "schema_version": "menudet-conformance-report-2",
        "provenance": {
            "python_revision": python_revision,
            "runner_sha256": hashlib.sha256(runner.read_bytes()).hexdigest(),
            "extension_sha256": hashlib.sha256(Path(blosc2.blosc2_ext.__file__).read_bytes()).hexdigest(),
            "package": blosc2.__file__,
            "generator_revision": corpus["generator_revision"],
            "corpus_sha256": hashlib.sha256(json.dumps(corpus, sort_keys=True).encode()).hexdigest(),
            "semantic_revisions": sorted({case["semantic_revision"] for case in corpus["cases"]}),
            "local_changes": (
                "M4 native 1.1 function signatures, accuracy and scoped floating status; checked 1.0 retained"
                if corpus["generator_revision"].startswith("m4")
                else "M3 opt-in native 1.1 arithmetic, inference, casts, Python authoring and tests; checked 1.0 retained"
                if any(case["semantic_revision"] == "menudet-numpy-1.1" for case in corpus["cases"])
                else "Checked 1.0 corpus; M1/M2 tooling plus shared infrastructure used by opt-in M3 1.1"
            ),
        },
        "reference_numpy": corpus["reference"]["numpy"],
        "installed_numpy": np.__version__,
        "installed_machine": platform.machine(),
        "reference_drift": drift,
        "eligible_jit_cases": sum(bool(row.get("jit_eligible")) for row in reports["on"]["native"]),
        "requests": reports,
    }


def check(report):
    bad = [
        (request, engine, row["id"], row["outcome"])
        for request, engines in report["requests"].items()
        for engine, rows in engines.items()
        for row in rows
        if row["outcome"] in {"mismatch", "regression"}
    ]
    if bad or report["reference_drift"]:
        raise AssertionError(
            f"Unexpected conformance failures: {bad}; reference drift: {report['reference_drift']}"
        )


def record(corpus, report):
    for case, c, _p in zip(
        corpus["cases"],
        report["requests"]["off"]["native"],
        report["requests"]["off"]["python"],
        strict=True,
    ):
        if c["outcome"] == "skipped":
            continue
        if c.get("baseline_regression"):
            raise AssertionError(f"Existing baseline changed: {case['id']}; review it separately")
        if not c["reference_match"] and not case.get("divergence"):
            raise AssertionError(f"Unreviewed divergence: {case['id']}; do not turn it into golden data")
        case["baseline"] = {"status": c["status"], "category": c["category"]}
        if not c["status"]:
            case["baseline"]["output"] = c["output"]
        case["classification"] = "matching" if c["reference_match"] else "divergent"


def property_cases(seed, count=48):
    rng = np.random.Generator(np.random.PCG64(seed))
    result = []
    for index in range(count):
        dtype = ("int8", "int16", "float32", "float64")[index % 4]
        arrays = [rng.integers(-30, 31, size=8).astype(dtype) for _ in range(2)]
        case = make_case(f"property-{seed}-{index}", "add", "x + y", arrays)
        case["seed"] = seed
        result.append(case)
    boundary = make_case(
        f"property-{seed}-boundary",
        "add",
        "x + y",
        [np.array([0, 127, 2], dtype="int8"), np.array([0, 1, 2], dtype="int8")],
    )
    boundary.update(
        seed=seed,
        divergence={"id": "D01", "reason": "Known checked-overflow boundary used to test minimization"},
    )
    result.append(boundary)
    return result


def minimized_failure(case, fails):  # noqa: C901 -- bounded delta-debugging plus scalar shrinking
    """Deterministic lane delta-debugging and scalar shrinking; no source eval."""
    result = copy.deepcopy(case)

    def candidate(arrays):
        current = copy.deepcopy(result)
        current.pop("baseline", None)
        current["inputs"] = [
            describe(array, item["name"]) for array, item in zip(arrays, result["inputs"], strict=True)
        ]
        current["expected"] = reference(current)
        return current

    arrays = [checkpoint.decode(item).ravel() for item in result["inputs"]]
    if not fails(result):
        raise ValueError("Cannot minimize a passing case")
    size = len(arrays[0])
    granularity = 2
    while size > 1:
        changed = False
        for indices in np.array_split(np.arange(size), min(size, granularity)):
            trial_arrays = [array[indices] for array in arrays]
            trial = candidate(trial_arrays)
            if fails(trial):
                result, arrays, size = trial, trial_arrays, len(indices)
                changed = True
                granularity = 2
                break
        if not changed:
            if granularity >= size:
                break
            granularity = min(size, granularity * 2)
    for array_index, array in enumerate(arrays):
        for index in range(len(array)):
            integral = array.dtype.kind in "iub"
            value = int(array[index]) if integral else float(array[index])
            options = [0, 1, -1]
            if math.isfinite(value):
                options.append((value // 2 if value >= 0 else -((-value) // 2)) if integral else value / 2)
                if value > 0:
                    options.append(value - 1)
                elif value < 0:
                    options.append(value + 1)
            for option in sorted(set(options), key=abs):
                if abs(option) >= abs(value):
                    continue
                if integral:
                    lower, upper = (
                        (0, 1)
                        if array.dtype.kind == "b"
                        else (np.iinfo(array.dtype).min, np.iinfo(array.dtype).max)
                    )
                    if not lower <= option <= upper:
                        continue
                trial_arrays = [a.copy() for a in arrays]
                trial_arrays[array_index][index] = option
                trial = candidate(trial_arrays)
                if fails(trial):
                    result, arrays = trial, trial_arrays
                    value = option
    result["minimized_from"] = case["id"]
    return result


def properties(corpus, runner, work_dir, seeds):
    summaries = []
    for seed in seeds:
        generated = {**corpus, "cases": property_cases(seed)}
        observed = paired(generated, runner, work_dir, observe=True)
        rows = observed["requests"]["off"]["native"]
        failures = [
            case for case, row in zip(generated["cases"], rows, strict=True) if not row["reference_match"]
        ]
        if len(failures) != 1 or failures[0]["id"] != f"property-{seed}-boundary":
            raise AssertionError(f"New seeded failure: {seed}: {failures}")

        def fails(case):
            trial = {**corpus, "cases": [case]}
            # The property predicate preserves both diagnostic and cross-runner agreement.
            report = paired(trial, runner, work_dir, observe=True)
            return report["requests"]["off"]["native"][0]["status"] == -5

        reduced = minimized_failure(failures[0], fails)
        assert [item["hex"] for item in reduced["inputs"]] == ["7f", "01"]
        summaries.append(
            {
                "seed": seed,
                "cases": len(rows),
                "matching": len(rows) - len(failures),
                "known_divergences": len(failures),
                "minimized": reduced,
                "promoted_case_id": "seed-1729-minimized-int8-overflow" if seed == 1729 else None,
            }
        )
    return {"rng": "PCG64", "seeds": seeds, "results": summaries}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["generate", "record", "run", "properties"])
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--native-runner", type=Path)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--seed", type=int, action="append")
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    if args.action == "generate":
        if args.checkpoint is None:
            parser.error("--checkpoint is required for explicit checkpoint migration")
        args.corpus.write_text(json.dumps(generate(args.checkpoint), indent=2) + "\n")
        return
    if args.native_runner is None:
        parser.error("--native-runner is required")
    corpus = json.loads(args.corpus.read_text())
    if args.action == "properties":
        report = properties(corpus, args.native_runner, args.work_dir, args.seed or [1729, 20261009])
    else:
        report = paired(corpus, args.native_runner, args.work_dir, observe=args.action == "record")
        if args.action == "record":
            record(corpus, report)
            args.corpus.write_text(json.dumps(corpus, indent=2) + "\n")
            report = paired(corpus, args.native_runner, args.work_dir)
        check(report)
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
