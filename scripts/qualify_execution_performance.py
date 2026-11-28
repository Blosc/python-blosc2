"""Interleave isolated Release wheels; shared-runner results are diagnostic only."""

import argparse
import json
import platform
import statistics
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--items", type=int, default=16_000_019)
    args = parser.parse_args()
    if args.rounds < 2 or args.items < 1:
        parser.error("at least two interleaved rounds and a positive item count are required")
    args.results.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[1]
    samples = {}
    for round_id in range(args.rounds):
        order = ("candidate", "reference") if round_id % 2 == 0 else ("reference", "candidate")
        for kind in order:
            package = getattr(args, kind).resolve()
            for script in ("execution_handoff", "execution_sink_handoff"):
                command = [
                    sys.executable,
                    "-S",
                    str(repo / "bench" / (script + ".py")),
                    "--package",
                    str(package),
                    "--items",
                    str(args.items),
                    "--repeat",
                    "7",
                    "--workers",
                    "4",
                ]
                result = subprocess.run(command, text=True, capture_output=True, timeout=600)
                (args.results / f"{script}-{kind}-{round_id}.log").write_text(result.stdout + result.stderr)
                result.check_returncode()
                rows = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
                assert rows, "No benchmark records"
                for row in rows:
                    if kind == "candidate":
                        route = row.get("route", row.get("report", {}).get("route", ""))
                        assert route.startswith("c-blosc2-typed"), route
                    key = (script, row["storage"], row.get("expression", "sink"))
                    samples.setdefault(key, {}).setdefault(kind, []).append(row["warmed_ms"])
    summary = []
    with (args.results / "compressed-reduction.log").open("w") as log:
        subprocess.run(
            [
                sys.executable,
                "-S",
                str(repo / "bench/compressed_reduction_plan.py"),
                "--package",
                str(args.candidate.resolve()),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=600,
        )
    for key, timings in samples.items():
        candidate, reference = (statistics.median(timings[k]) for k in ("candidate", "reference"))
        summary.append(
            {
                "case": key,
                "candidate_ms": candidate,
                "reference_ms": reference,
                "ratio": candidate / reference,
                "investigate": candidate > reference * 1.05,
            }
        )
    report = {
        "platform": platform.platform(),
        "scope": "diagnostic; not a stable-runner gate",
        "cases": summary,
    }
    (args.results / "performance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
