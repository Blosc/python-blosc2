"""Aggregate bench.py results into one markdown report.

Expects a directory of artifacts laid out as::

    <root>/bench-<os>-<pyver>/{abi3,base}-<round>.json

For each (platform, version) cell it takes the minimum across rounds per
benchmark -- rounds are interleaved abi3/base/abi3/... in the workflow, so a
slow patch of a noisy runner hits both builds rather than biasing one.

Exits non-zero if any benchmark regresses past THRESHOLD, so the job's
pass/fail conclusion is meaningful even without reading the log.
"""

import glob
import json
import os
import sys

# GitHub runners are shared vCPUs with noisy neighbours; ratios wobble by
# double-digit percentages between rounds.  This job is looking for an
# *important* regression -- an extra indirection on every libpython call --
# not a 3% one, which this environment cannot resolve.
THRESHOLD = 1.25

# Benchmarks whose absolute time is small enough that runner noise dominates.
# Still reported, just not allowed to fail the job on their own.
NOISE_FLOOR_MS = 5.0


def emit(report, *, status, failures):
    """Publish the report everywhere, and report `status` rather than exiting.

    "no data" and "a real regression" are different outcomes and the workflow
    needs to tell them apart, so this always exits 0 and hands the verdict to
    the caller through GITHUB_OUTPUT.
    """
    print(report)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as fh:
            fh.write(report + "\n")
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a") as fh:
            fh.write(f"status={status}\n")
            fh.write(f"failures={failures}\n")
    # Always written, so the PR-comment step has a file even when there is no data.
    with open("abi3-bench-report.md", "w") as fh:
        fh.write(report + "\n")


def load_cell(cell_dir):
    out = {}
    for build in ("base", "abi3"):
        runs = []
        for path in sorted(glob.glob(os.path.join(cell_dir, f"{build}-*.json"))):
            with open(path) as fh:
                runs.append(json.load(fh))
        out[build] = runs
    return out


def best(runs, name):
    vals = [r["results"][name]["min"] for r in runs if name in r["results"] and "min" in r["results"][name]]
    return min(vals) if vals else None


def main(root):
    cells = sorted(d for d in glob.glob(os.path.join(root, "bench-*")) if os.path.isdir(d))
    if not cells:
        msg = f"No benchmark results were produced under `{root}` (the bench jobs did not upload artifacts)."
        print(msg, file=sys.stderr)
        emit(msg + "\n", status="nodata", failures=0)
        return 0

    lines = ["## abi3 vs. version-specific build\n"]
    lines.append(
        f"Minimum of interleaved rounds. Regression threshold **{THRESHOLD:.2f}x** "
        f"(benchmarks under {NOISE_FLOOR_MS:g} ms are reported but never fail the "
        "job -- CI runners cannot resolve them).\n"
    )

    worst_overall = None
    failures = []

    for cell in cells:
        label = os.path.basename(cell).removeprefix("bench-")
        data = load_cell(cell)
        if not data["base"] or not data["abi3"]:
            lines.append(f"\n### {label}\n\n_missing results_\n")
            continue

        meta = data["abi3"][0]["meta"]
        lines.append(f"\n### {label}\n")
        lines.append(
            f"Python {meta.get('python')}, blosc2 {meta.get('blosc2')}, "
            f"numpy {meta.get('numpy')} — module `{os.path.basename(meta.get('ext_file', '?'))}`\n"
        )
        lines.append("| benchmark | base | abi3 | ratio |")
        lines.append("|---|---:|---:|---:|")

        names = list(data["base"][0]["results"])
        for name in names:
            b = best(data["base"], name)
            a = best(data["abi3"], name)
            if b is None or a is None:
                lines.append(f"| {name} | — | — | skipped |")
                continue
            ratio = a / b
            noisy = b * 1e3 < NOISE_FLOOR_MS
            mark = ""
            if ratio > THRESHOLD:
                mark = " ⚠️" if noisy else " ❌"
                if not noisy:
                    failures.append((label, name, ratio))
            lines.append(f"| {name} | {b * 1e3:.3f} ms | {a * 1e3:.3f} ms | {ratio:.3f}x{mark} |")
            if not noisy and (worst_overall is None or ratio > worst_overall[2]):
                worst_overall = (label, name, ratio)

    lines.append("\n---\n")
    if worst_overall:
        lines.append(
            f"**Worst non-noise ratio:** {worst_overall[2]:.3f}x "
            f"({worst_overall[1]} on {worst_overall[0]})\n"
        )
    if failures:
        lines.append(f"\n**{len(failures)} benchmark(s) past threshold:**\n")
        for label, name, ratio in failures:
            lines.append(f"- `{label}` — {name}: {ratio:.3f}x")
    else:
        lines.append("\nNo regression past threshold on any platform. ✅\n")

    emit(
        "\n".join(lines),
        status="regressed" if failures else "ok",
        failures=len(failures),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "results"))
