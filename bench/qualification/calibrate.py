"""Bounded geometric size sweeps; never silently approve or overwrite profiles."""

import argparse
import json
import math
import os
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qualification.menudet import load_manifest, run_pair
from qualification.report import SCHEMA_VERSION
from qualification.workloads import CASES


def wilson(successes, total):
    if total == 0:
        return None
    z = 1.96
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return [max(0, center - radius), min(1, center + radius)]


def eligible(result):
    if result["coverage"] != "complete":
        return False, "Required correctness/route coverage is incomplete"
    for row in result["blocks"]:
        execution, workload = row["execution"], row["workload"]
        if (
            result["profile"]["threads"] > 1
            and workload["route"] == "resident"
            and (execution.get("tasks") or 0) < 2
        ):
            return False, "Size is serial; cannot represent parallel throughput"
        if workload["storage"] == "slabs":
            shape, chunks = workload["shape"], workload["storage_grid"][0]
            if math.prod(math.ceil(n / c) for n, c in zip(shape, chunks, strict=True)) < 2:
                return False, "Insufficient physical chunks for streaming throughput"
    return True, None


def validate_sweep(args):
    if args.control == "historical" and not args.historical_label:
        raise ValueError("Historical controls require a reproduction label")
    if args.control == "aa" and not args.local_control:
        baseline, candidate = load_manifest(args.baseline_manifest), load_manifest(args.candidate_manifest)
        for key in ("python_revision", "miniexpr_revision", "c_blosc2_revision"):
            if baseline[key] != candidate[key]:
                raise ValueError(
                    "A/A requires independently built installations of the same integrated revisions"
                )
    cases = args.cases or ["evaluate-polynomial-cc-new", "authored-slabs-compressed"]
    if any(name not in CASES or CASES[name].fixed_size is not None for name in cases):
        raise ValueError("Calibration cases must be known, scalable workloads (not fixed dispatch controls)")
    if args.control == "extra-copy" and any(CASES[name].compressed_output for name in cases):
        raise ValueError("Select fresh NumPy-output cases for the extra-copy control")
    return cases


def sweep(args):
    cases = validate_sweep(args)
    started = time.monotonic()
    result = {
        "schema_version": SCHEMA_VERSION,
        "kind": "calibration",
        "allocation_id": os.environ.get("GITHUB_RUN_ID", str(uuid.uuid4())),
        "profile": args.profile,
        "control": args.control,
        "historical_label": args.historical_label,
        "independent_builds": not args.local_control,
        "selection_status": "unreviewed; requires fresh and held-out allocations",
        "cache_capacity_bytes": None,
        "limitations": [
            "Cache capacity not reliably determined; no streaming/cache-size claim.",
            "Single allocation does not establish detection probabilities.",
        ],
        "sizes": [],
        "coverage": "incomplete",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    directory = args.output.with_suffix("")
    directory.mkdir(exist_ok=True)
    try:
        for size in sorted(set(args.sizes)):
            remaining = args.budget_seconds - (time.monotonic() - started)
            if remaining <= 0:
                break
            options = argparse.Namespace(**vars(args))
            options.size, options.cases = size, cases
            options.sensitivity_copy = args.control == "extra-copy"
            options.output = directory / f"size-{size}.json"
            options.budget_seconds = remaining
            pair = run_pair(options)
            admitted, reason = eligible(pair)
            case_eligibility = {}
            for case_id in cases:
                subset = {**pair, "blocks": [b for b in pair["blocks"] if b["case_id"] == case_id]}
                subset["coverage"] = (
                    "complete"
                    if all(b["status"] == "ok" for b in subset["blocks"]) and subset["blocks"]
                    else "incomplete"
                )
                case_eligibility[case_id] = dict(zip(("eligible", "reason"), eligible(subset), strict=True))
            result["sizes"].append(
                {
                    "size": size,
                    "eligible": admitted,
                    "reason": reason,
                    "artifact": str(options.output),
                    "coverage": pair["coverage"],
                    "comparisons": pair["comparisons"],
                    "wall_seconds": pair["wall_seconds"],
                    "case_eligibility": case_eligibility,
                    "execution_failures": [
                        b.get("error")
                        for b in pair["blocks"]
                        if b["status"] == "failed" and b.get("phase") != "execution-contract"
                    ],
                    "historical_contract_signal": all(
                        any(
                            b["case_id"] == case_id and b["arm"] == "baseline" and b["status"] == "ok"
                            for b in pair["blocks"]
                        )
                        and any(
                            b["case_id"] == case_id
                            and b["arm"] == "candidate"
                            and b["status"] == "failed"
                            and b.get("phase") == "execution-contract"
                            for b in pair["blocks"]
                        )
                        for case_id in cases
                    ),
                }
            )
        if len(result["sizes"]) == len(set(args.sizes)):
            result["coverage"] = "complete"
    finally:
        result["wall_seconds"] = time.monotonic() - started
        result["mechanism_only_minimum_sizes"] = {
            case_id: min(
                (r["size"] for r in result["sizes"] if r["case_eligibility"][case_id]["eligible"]),
                default=None,
            )
            for case_id in cases
        }
        result["sweep_complete"] = len(result["sizes"]) == len(set(args.sizes))
        if any(r["execution_failures"] for r in result["sizes"]):
            result["coverage"] = "failed"
        elif any(size is None for size in result["mechanism_only_minimum_sizes"].values()):
            result["coverage"] = "incomplete"
        if args.control == "historical":
            result["control_reproduced"] = any(r["historical_contract_signal"] for r in result["sizes"])
            if (
                result["control_reproduced"]
                and result["sweep_complete"]
                and not any(r["execution_failures"] for r in result["sizes"])
            ):
                result["coverage"] = "complete"
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        rows = [
            "# Menudet size calibration",
            "",
            f"Coverage: **{result['coverage']}**. Control: {args.control}.",
            "",
            "| Elements | Mechanism eligible | Pair coverage | Cost (s) |",
            "|---:|---|---|---:|",
        ]
        rows += [
            f"| {r['size']} | {r['eligible']} | {r['coverage']} | {r['wall_seconds']:.2f} |"
            for r in result["sizes"]
        ]
        rows += [
            "",
            "No profile was approved or changed. Preserve evidence across independent and held-out runner allocations.",
            "",
        ]
        args.output.with_suffix(".md").write_text("\n".join(rows))
    return result


def evidence_summary(training, validation):
    groups = {"training": training, "held_out": validation}
    identities = [d["allocation_id"] for docs in groups.values() for d in docs]
    if len(identities) != len(set(identities)):
        raise ValueError("Repeated allocations are not independent or held-out evidence")
    result = {"schema_version": SCHEMA_VERSION, "profile_approved": False, "groups": {}}
    for group, documents in groups.items():
        aa, detection = [], []
        for document in documents:
            if document["schema_version"] != SCHEMA_VERSION or document["kind"] != "calibration":
                raise ValueError("Unsupported calibration evidence schema")
            if not document["independent_builds"] or document["coverage"] != "complete":
                continue
            eligible_rows = [r for r in document["sizes"] if r["eligible"]]
            if document["control"] == "historical" and document.get("control_reproduced"):
                detection.append(True)
                continue
            if not eligible_rows:
                continue
            # Suite-wide warning event per allocation, not per inner sample.
            warning = any(c["verdict"] == "warning" for r in eligible_rows for c in r["comparisons"])
            (aa if document["control"] == "aa" else detection).append(warning)
        result["groups"][group] = {
            "aa_allocations": len(aa),
            "aa_warning_allocations": sum(aa),
            "aa_warning_rate_interval": wilson(sum(aa), len(aa)),
            "control_allocations": len(detection),
            "control_warning_allocations": sum(detection),
            "control_detection_interval": wilson(sum(detection), len(detection)),
        }
    result["limitations"] = [
        "Descriptive evidence only; controls/sizes/fingerprints must be reviewed separately.",
        "Extra-copy overhead is measured, not assumed to equal a 20% regression.",
        "No timing gate or profile promotion is automatic.",
    ]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["observation", "macos-arm"], default="observation")
    parser.add_argument("--baseline-python", default=sys.executable)
    parser.add_argument("--candidate-python", default=sys.executable)
    parser.add_argument("--baseline-manifest", type=Path)
    parser.add_argument("--candidate-manifest", type=Path)
    parser.add_argument("--local-control", action="store_true")
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--sizes", type=int, nargs="+", default=[4096, 16384, 65536, 262144, 1048576])
    parser.add_argument("--control", choices=["aa", "extra-copy", "historical"], default="aa")
    parser.add_argument("--historical-label")
    parser.add_argument("--threads", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--target-ms", type=float, default=50)
    parser.add_argument("--budget-seconds", type=float, default=300)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training", type=Path, nargs="+")
    parser.add_argument("--held-out", type=Path, nargs="+")
    args = parser.parse_args()
    if args.training or args.held_out:
        result = evidence_summary(
            [json.loads(p.read_text()) for p in args.training or []],
            [json.loads(p.read_text()) for p in args.held_out or []],
        )
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        return 0
    result = sweep(args)
    print(f"Calibration coverage: {result['coverage']}; {args.output}")
    return 0 if result["coverage"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
