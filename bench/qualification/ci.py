"""Small CI orchestration layer; untrusted workflow inputs never become shell code."""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qualification.build import build, exact_sha, sha256


def resolve(output):
    anchor = json.loads(Path(".github/menudet-performance-baseline.json").read_text())
    if anchor.get("timing_gates_enabled"):
        raise ValueError(
            "Timing gates require calibrated decision evidence and are not enabled by this implementation"
        )
    candidate = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    mode = os.environ.get("REQUESTED_MODE") or (
        "calibration" if os.environ.get("SCHEDULE") == "41 5 * * 0" else "observation"
    )
    if (
        mode == "calibration"
        and not os.environ.get("REQUESTED_MODE")
        and date.today().isocalendar().week % 2
    ):
        mode = "larger"
    baseline = (
        candidate
        if mode == "calibration"
        else os.environ.get("REQUESTED_BASELINE") or anchor["approved_baseline"]
    )
    if baseline is None:
        raise ValueError("No approved anchor. Supply an exact baseline SHA for manual qualification.")
    exact_sha(baseline)
    subprocess.run(["git", "cat-file", "-e", f"{baseline}^{{commit}}"], check=True)
    if platform.machine() != os.environ["EXPECTED_ARCH"]:
        raise ValueError("Allocated runner architecture differs from the required profile")
    if mode not in ("smoke", "observation", "calibration", "larger"):
        raise ValueError("Unknown measurement mode")
    data = {
        "baseline": baseline,
        "candidate": exact_sha(candidate),
        "mode": mode,
        "profile": os.environ["PROFILE"],
        "machine": platform.machine(),
        "image_version": os.environ.get("ImageVersion"),  # noqa: SIM112 -- GitHub runner spelling
        "image_os": os.environ.get("ImageOS"),  # noqa: SIM112 -- GitHub runner spelling
        "harness_revision": candidate,
        "baseline_source": "independent-build-aa"
        if mode == "calibration"
        else "manual-explicit"
        if os.environ.get("REQUESTED_BASELINE")
        else "approved-anchor",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2) + "\n")


def environments(output):
    data = json.loads(subprocess.check_output(["conda", "env", "list", "--json"], text=True))
    paths = {}
    for arm in ("baseline", "candidate"):
        artifact = Path("artifacts/builds") / arm
        manifest = json.loads((artifact / "manifest.json").read_text())
        if (
            sha256((artifact / "wheels" / manifest["wheel_filename"]).read_bytes())
            != manifest["wheel_sha256"]
        ):
            raise ValueError("Cached wheel differs from exact build manifest")
        prefix = next(Path(p) for p in data["envs"] if Path(p).name == f"menudet-{arm}")
        paths[arm] = str(prefix / "bin" / "python")
    output.write_text(json.dumps(paths, indent=2) + "\n")


def measure(identity, envs):
    mode, profile = identity["mode"], identity["profile"]
    common = [
        "--baseline-python",
        envs["baseline"],
        "--candidate-python",
        envs["candidate"],
        "--baseline-manifest",
        "artifacts/builds/baseline/manifest.json",
        "--candidate-manifest",
        "artifacts/builds/candidate/manifest.json",
    ]
    script = "menudet.py"
    options = [
        "--profile",
        "smoke" if mode == "smoke" else profile,
        "--output",
        "artifacts/qualification.json",
        "--budget-seconds",
        "180",
    ]
    if mode == "calibration":
        script = "calibrate.py"
        options = [
            "--profile",
            profile,
            "--output",
            "artifacts/calibration.json",
            "--budget-seconds",
            "300",
            "--cases",
            "evaluate-polynomial-cc-new",
            "authored-slabs-compressed",
        ]
    elif mode == "larger":
        options += [
            "--size",
            "4194304",
            "--cases",
            "evaluate-polynomial-cc-new",
            "authored-slabs-compressed",
        ]
    subprocess.run([sys.executable, str(Path(__file__).parent / script), *common, *options], check=True)


def summary(artifacts):
    summaries = sorted(artifacts.glob("*.md"))
    text = (
        "\n".join(p.read_text() for p in summaries)
        if summaries
        else "Qualification incomplete: no measurement summary. Inspect build logs.\n"
    )
    destination = os.environ.get("GITHUB_STEP_SUMMARY")
    if destination:
        with Path(destination).open("a") as stream:
            stream.write(text)
    print(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=["resolve", "cache-key", "build", "environments", "measure", "summary"]
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--identity", type=Path)
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--environments", type=Path)
    parser.add_argument("--artifacts", type=Path)
    args = parser.parse_args()
    if args.action == "resolve":
        resolve(args.output)
    elif args.action == "environments":
        environments(args.output)
    elif args.action == "summary":
        summary(args.artifacts)
    elif args.action == "cache-key":
        compiler = subprocess.check_output([os.environ.get("CC", "cc"), "--version"])
        key = hashlib.sha256(
            args.identity.read_bytes()
            + args.lock.read_bytes()
            + compiler
            + Path(__file__).with_name("build.py").read_bytes()
        ).hexdigest()
        with Path(os.environ["GITHUB_ENV"]).open("a") as stream:
            stream.write(f"MENUDET_PERF_CACHE_KEY={key}\n")
    else:
        identity = json.loads(args.identity.read_text())
        if args.action == "build":
            for arm in ("baseline", "candidate"):
                build(
                    argparse.Namespace(
                        repository=Path.cwd(),
                        revision=identity[arm],
                        lock=args.lock,
                        output=Path("artifacts/builds") / arm,
                        jobs=2,
                    )
                )
        elif args.action == "measure":
            measure(identity, json.loads(args.environments.read_text()))


if __name__ == "__main__":
    main()
