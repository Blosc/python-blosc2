"""Versioned reports and conservative observation-only process-block comparisons."""

import hashlib
import json
import math
import random
import statistics
from pathlib import Path

SCHEMA_VERSION = 1


def validate_document(result):
    if result.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported result schema")
    for row in result.get("blocks", []):
        if row.get("status") not in ("ok", "failed", "incomplete"):
            raise ValueError("Unknown process verdict")
        if row["status"] == "ok":
            if not row.get("execution") or not row.get("warm_samples"):
                raise ValueError("Successful process lacks execution/timing evidence")
            for sample in row["warm_samples"]:
                if (
                    not math.isfinite(sample["per_call_ns"])
                    or sample["per_call_ns"] <= 0
                    or sample["iterations"] < 1
                ):
                    raise ValueError("Invalid raw measurement sample")
    return result


def ratio_interval(ratios, seed=42):
    """Descriptive paired-block bootstrap, not a calibrated decision guarantee."""
    if len(ratios) < 4:
        return None
    rng = random.Random(seed)
    logs = [math.log(r) for r in ratios]
    estimates = sorted(math.exp(statistics.mean(rng.choices(logs, k=len(logs)))) for _ in range(1000))
    return [estimates[24], estimates[974]]


def read_profile(path):
    from .workloads import CASES

    profile = json.loads(Path(path).read_text())
    if profile.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported profile schema")
    cases = profile.get("cases", [])
    if not cases or len(cases) != len(set(cases)) or set(cases) - CASES.keys():
        raise ValueError("Empty, duplicate or unknown case IDs")
    for name in ("size", "threads", "max_memory_bytes"):
        if type(profile.get(name)) is not int or profile[name] < 1:
            raise ValueError(f"Invalid profile {name}")
    # Inputs, reference, output, validation temporaries and conservative scratch.
    if profile["size"] * 128 > profile["max_memory_bytes"]:
        raise ValueError("Profile exceeds declared memory budget")
    return profile


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare(blocks, cases, expected_blocks, threshold=1.2):
    """Use process medians, never inner-loop iterations as independent evidence.

    Without cross-allocation calibration, even consistent warnings are not gates
    or confidence intervals. Overlapping block ranges remain inconclusive.
    """
    rows = []
    for case_id in cases:
        count = expected_blocks[case_id] if isinstance(expected_blocks, dict) else expected_blocks
        selected = [b for b in blocks if b["case_id"] == case_id]
        shape = next(
            (b.get("workload", {}).get("shape") for b in selected if b.get("workload", {}).get("shape")),
            None,
        )
        row = {"case_id": case_id, "shape": shape, "ratio": None, "verdict": "incomplete"}
        if any(b["status"] == "failed" for b in selected):
            row["verdict"] = "failed"
        elif len(selected) == 2 * count and all(b["status"] == "ok" for b in selected):
            values = {}
            for arm in ("baseline", "candidate"):
                arm_blocks = [b for b in selected if b["arm"] == arm]
                if len(arm_blocks) != count or len({b["block_id"] for b in arm_blocks}) != count:
                    raise ValueError("Duplicate or missing process-block identities")
                values[arm] = [
                    statistics.median(s["per_call_ns"] for s in b["warm_samples"]) for b in arm_blocks
                ]
                if not all(math.isfinite(v) and v > 0 for v in values[arm]):
                    raise ValueError("Invalid timing samples")
                row[f"{arm}_ns"] = statistics.median(values[arm])
                row[f"{arm}_block_range_ns"] = [min(values[arm]), max(values[arm])]
            paired = {}
            for block in selected:
                paired.setdefault(block["block_id"], {})[block["arm"]] = statistics.median(
                    s["per_call_ns"] for s in block["warm_samples"]
                )
            row["paired_block_ratios"] = [p["candidate"] / p["baseline"] for _, p in sorted(paired.items())]
            row["ratio"] = statistics.median(row["paired_block_ratios"])
            row["descriptive_ratio_interval"] = ratio_interval(row["paired_block_ratios"])
            row["executions"] = {
                arm: sorted(
                    {
                        f"{b.get('execution', {}).get('engine', 'unavailable')}/"
                        f"{b.get('execution', {}).get('route', 'unavailable')}"
                        for b in selected
                        if b["arm"] == arm
                    }
                )
                for arm in ("baseline", "candidate")
            }
            row["verdict"] = (
                "warning"
                if min(values["candidate"]) > threshold * max(values["baseline"])
                else "inconclusive"
                if row["ratio"] >= threshold
                else "observed"
            )
        rows.append(row)
    return rows


def markdown(result):
    lines = [
        "# Menudet performance qualification",
        "",
        f"Coverage: **{result['coverage']}**. Timing policy: observation only.",
        "",
        "| Case | Elements | Baseline ms | Candidate ms | Candidate/baseline | Actual engine/route | Verdict |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for row in result["comparisons"]:

        def number(key, scale=1, row=row):
            return f"{row[key] / scale:.3f}" if row.get(key) is not None else "—"

        execution = ", ".join(row.get("executions", {}).get("candidate", [])) or "—"
        elements = math.prod(row["shape"]) if row.get("shape") else "—"
        lines.append(
            f"| {row['case_id']} | {elements} | {number('baseline_ns', 1e6)} | "
            f"{number('candidate_ns', 1e6)} | {number('ratio')} | {execution} | {row['verdict']} |"
        )
    lines += [
        "",
        f"Wall time: {result['wall_seconds']:.2f} s. "
        "Process blocks, raw samples, execution reports and failures are in the JSON artifact.",
        "",
        "Uncalibrated profiles; no statistical confidence or timing pass/fail claim. "
        "Effective compiler is unavailable in the current native report.",
        "",
    ]
    return "\n".join(lines)
