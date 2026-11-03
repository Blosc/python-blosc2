"""Sequential, same-machine paired Menudet observation driver.

Run by absolute script path with conda Python; workers use isolated mode so a
source checkout cannot shadow the selected installation. See README.md.
"""

import argparse
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
import traceback
import zipfile
from dataclasses import asdict
from pathlib import Path

# Only the benchmark directory, not the checkout/package source, enters sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qualification.measurement import BudgetExhausted, Settings, compiler_cache, timed, warm_samples
from qualification.report import SCHEMA_VERSION, compare, digest, markdown, read_profile, validate_document
from qualification.resources import cpu_count, cpu_model, memory_limit

HERE = Path(__file__).resolve().parent


def available_cpus():
    return cpu_count()


def verify_imports(paths, prefix, allow_editable=False):
    if not allow_editable and any(
        not Path(path).resolve().is_relative_to(Path(prefix).resolve()) for path in paths
    ):
        raise ValueError("Package/extension import is outside the selected environment")


def verify_installed_wheel(manifest, package_file):
    files = manifest.get("installed_files")
    if not files:
        raise ValueError("Build manifest lacks installed wheel file hashes")
    site_packages = Path(package_file).parent.parent
    for name, expected in files.items():
        path = (site_packages / name).resolve()
        if (
            not path.is_relative_to(site_packages.resolve())
            or not path.is_file()
            or digest(path) != expected
        ):
            raise ValueError(f"Installed package differs from attested wheel: {name}")


def verify_stack(row, previous):
    """Runtime dependencies are frozen across arms; installed code cannot drift."""
    if row["status"] != "ok":
        return
    current = row["fingerprint"]
    dependencies = [d for d in current["dependencies"] if not d.lower().startswith("blosc2==")]
    for block in previous:
        if block["status"] != "ok":
            continue
        other = block["fingerprint"]
        if dependencies != [d for d in other["dependencies"] if not d.lower().startswith("blosc2==")]:
            raise ValueError("Runtime dependency versions differ across process blocks/arms")
        if row["arm"] == block["arm"] and current["extension_sha256"] != other["extension_sha256"]:
            raise ValueError("Installed extension changed during qualification")


def fingerprint():
    import numpy as np

    import blosc2

    paths = [blosc2.__file__, blosc2.blosc2_ext.__file__]
    return {
        "python": sys.version,
        "executable": sys.executable,
        "prefix": sys.prefix,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_model": cpu_model(),
        "memory_limit_bytes": memory_limit(),
        "available_cpus": available_cpus(),
        "numpy": np.__version__,
        "blosc2": blosc2.__version__,
        "import_paths": paths,
        "extension_sha256": digest(paths[1]),
        "dependencies": sorted(
            f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions()
        ),
        "environment": {
            k: v
            for k, v in os.environ.items()
            if k.startswith(("ME_", "BLOSC_")) or k in ("CC", "CFLAGS", "OMP_NUM_THREADS")
        },
        "runner_image": {k: os.environ.get(k) for k in ("ImageOS", "ImageVersion", "RUNNER_ARCH")},
    }


def worker(request):
    import blosc2
    from qualification.diagnostics import owner_report, resident_reports, validate_contract
    from qualification.workloads import CASES, validate_result

    started, cpu_started = time.monotonic(), time.process_time()
    row = {k: request[k] for k in ("case_id", "arm", "block_id", "order_index")}
    row.update(status="failed", warm_samples=[], phase="preflight")
    previous_threads = None
    try:
        row["fingerprint"] = fingerprint()
        verify_imports(row["fingerprint"]["import_paths"], sys.prefix, request["local_control"])
        if not request["local_control"] or request.get("manifest", {}).get("installed_files"):
            verify_installed_wheel(request["manifest"], row["fingerprint"]["import_paths"][0])
        if request["threads"] > available_cpus():
            raise ValueError("Requested workers exceed available CPUs")
        previous_threads = blosc2.set_nthreads(request["threads"])
        case = CASES[request["case_id"]]
        size = case.fixed_size or request["size"]
        limit = row["fingerprint"]["memory_limit_bytes"]
        if limit is not None and size * 128 > limit:
            raise ValueError("Workload exceeds detected physical/cgroup memory limit")
        row["workload"] = case.metadata(size, request["seed"])
        inputs, expected, destination = case.bind(size, request["seed"])
        settings = Settings(**request["settings"])
        # Fresh process + fresh cache for every block. This is first public-API
        # execution, NOT an isolated compiler-only preparation measurement.
        with compiler_cache(request["cache_dir"]), blosc2.jit_options(**row["workload"]["requested_policy"]):
            row["phase"] = "preparation"
            (call, owner), preparation = timed(lambda: case.prepare(inputs, destination))
            if request.get("sensitivity_copy"):
                original_call = call

                def call():
                    result = original_call()
                    return result.copy() if not isinstance(result, blosc2.NDArray) else result[:].copy()

            row["preparation_ns"] = None if case.frontend == "evaluate" else preparation
            row["preparation_scope"] = (
                "Public evaluate preparation/compilation included in first_execution_ns"
                if case.frontend == "evaluate"
                else "Artifact export/import or lazy graph preparation; excludes input generation"
            )
            row["phase"] = "first-execution"
            first, row["first_execution_ns"] = timed(call)
            row["phase"] = "correctness"
            validate_result(first, expected, destination, case.compressed_output)
            del first
            # Diagnostics are explicitly outside every recorded timer.
            with resident_reports() as reports:
                result = call()
            validate_result(result, expected, destination, case.compressed_output)
            row["phase"] = "execution-contract"
            observed = reports[0] if len(reports) == 1 else owner_report(owner)
            row["execution"] = validate_contract(
                case, [observed] if observed else reports, size, request["threads"]
            )
            del result
            result, row["batch_pilot_ns"] = timed(call)
            del result
            row["phase"] = "warm-execution"
            row["warm_samples"] = warm_samples(
                call, row["batch_pilot_ns"], settings, started + request["budget_seconds"]
            )
            with resident_reports() as reports:
                result = call()
            row["phase"] = "correctness"
            validate_result(result, expected, destination, case.compressed_output)
            observed = reports[0] if len(reports) == 1 else owner_report(owner)
            row["phase"] = "execution-contract"
            after = validate_contract(case, [observed] if observed else reports, size, request["threads"])
            if (after["route"], after["engine"]) != (row["execution"]["route"], row["execution"]["engine"]):
                raise AssertionError("Execution contract changed during measurement")
            del result
            # Current buffers must be rebound, not cached numerical results.
            inputs["x"][:] = 64
            result = call()
            row["phase"] = "correctness"
            current = {
                name: value[:] if isinstance(value, blosc2.NDArray) else value
                for name, value in inputs.items()
            }
            validate_result(result, case.reference(current), destination, case.compressed_output)
            del result
            row["status"] = "ok"
            row["phase"] = "complete"
    except BudgetExhausted as error:
        row.update(status="incomplete", error=str(error))
    except Exception:
        # Record errors, never replay failed numerical work through another API.
        row["error"] = traceback.format_exc()
    finally:
        if previous_threads is not None:
            blosc2.set_nthreads(previous_threads)
    row["wall_seconds"] = time.monotonic() - started
    row["cpu_seconds"] = time.process_time() - cpu_started
    return row


def execution_order(blocks, seed):
    for block in range(blocks):
        order = ["baseline", "candidate"] if block % 2 == 0 else ["candidate", "baseline"]
        if seed % 2:
            order.reverse()
        # Seed controls orientation, deterministic alternating AB/BA blocks.
        yield block, order


def load_manifest(path):
    data = json.loads(Path(path).read_text())
    for key in ("python_revision", "miniexpr_revision", "c_blosc2_revision"):
        value = data.get(key, "")
        if len(value) != 40 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"Manifest requires exact {key}")
    for key in ("wheel_sha256", "dependency_lock_sha256"):
        value = data.get(key, "")
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"Manifest requires {key}")
    if not data.get("build_configuration"):
        raise ValueError("Manifest requires build configuration")
    return data


def run_pair(args):  # noqa: C901 -- sequential orchestration and failure artifact handling
    profile_path = HERE / "profiles" / f"{args.profile}.json"
    profile = read_profile(profile_path)
    if args.profile == "macos-arm" and (
        sys.platform != "darwin" or platform.machine() not in ("arm64", "aarch64")
    ):
        raise ValueError("macos-arm requires a verified macOS ARM host")
    if getattr(args, "size", None) is not None:
        if args.size < 1 or args.size * 128 > profile["max_memory_bytes"]:
            raise ValueError("Size exceeds profile memory bound")
        profile["size"] = args.size
    if getattr(args, "sensitivity_copy", False):
        from qualification.workloads import CASES

        selected = args.cases or profile["cases"]
        if any(CASES[name].compressed_output or CASES[name].explicit_output for name in selected):
            raise ValueError("Extra-copy control only supports fresh NumPy-output contracts")
    if args.cases:
        if set(args.cases) - set(profile["cases"]):
            raise ValueError("Selected case is not in profile")
        profile["cases"] = list(dict.fromkeys(args.cases))
    if args.threads is not None:
        profile["threads"] = args.threads
    if profile["threads"] < 1 or profile["threads"] > available_cpus():
        raise ValueError("Invalid thread count for this machine")
    if not math.isfinite(args.budget_seconds) or args.budget_seconds <= 0 or args.blocks < 2:
        raise ValueError("Require positive finite budget and at least two process blocks per arm")
    settings = Settings(args.warmups, args.samples, args.target_ms)
    interpreters = {
        "baseline": str(Path(args.baseline_python).resolve()),
        "candidate": str(Path(args.candidate_python).resolve()),
    }
    manifests = {}
    for arm in interpreters:
        path = getattr(args, f"{arm}_manifest")
        if not args.local_control and path is None:
            raise ValueError(
                "Wheel comparisons require both build manifests; use --local-control for editable A/A"
            )
        manifests[arm] = load_manifest(path) if path else {"provenance": "unverified local control"}
    if not args.local_control and interpreters["baseline"] == interpreters["candidate"]:
        raise ValueError("Wheel comparisons require separate environments")
    if not args.local_control:
        for key in ("build_configuration", "dependency_lock_sha256"):
            if manifests["baseline"][key] != manifests["candidate"][key]:
                raise ValueError(f"Incompatible arm manifests: {key}")
    started = time.monotonic()
    result = {
        "schema_version": SCHEMA_VERSION,
        "measurement_mode": "local-control" if args.local_control else "paired-wheels",
        "profile": profile,
        "profile_sha256": digest(profile_path),
        "settings": asdict(settings),
        "seed": args.seed,
        "control": "extra-copy" if getattr(args, "sensitivity_copy", False) else "normal",
        "manifests": manifests,
        "harness_files": {p.name: digest(p) for p in sorted(HERE.glob("*.py"))},
        "workflow": {k: os.environ.get(k) for k in ("GITHUB_SHA", "GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT")},
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "blocks": [],
        "expected_blocks": dict.fromkeys(profile["cases"], args.blocks),
        "confirmation_blocks": getattr(args, "confirmation_blocks", 0),
        "escalation": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    snapshot = args.output.with_suffix(".harness.zip")
    with zipfile.ZipFile(snapshot, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(HERE.rglob("*")):
            if path.is_file() and path.suffix in (".py", ".json", ".txt", ".md"):
                archive.write(path, path.relative_to(HERE))
    result["harness_archive"] = {"filename": snapshot.name, "sha256": digest(snapshot)}

    def save():
        result["wall_seconds"] = time.monotonic() - started
        result["measured_process_cpu_seconds"] = sum(row.get("cpu_seconds", 0) for row in result["blocks"])
        result["measured_warm_calls"] = sum(
            sample["iterations"] for row in result["blocks"] for sample in row.get("warm_samples", [])
        )
        result["warm_bytes_processed"] = sum(
            row.get("workload", {}).get("bytes_processed", 0) * sample["iterations"]
            for row in result["blocks"]
            for sample in row.get("warm_samples", [])
        )
        result["comparisons"] = compare(result["blocks"], profile["cases"], result["expected_blocks"])
        result["coverage"] = (
            "failed"
            if any(r["verdict"] == "failed" for r in result["comparisons"])
            else "incomplete"
            if any(r["verdict"] == "incomplete" for r in result["comparisons"])
            else "complete"
        )
        validate_document(result)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        temporary.replace(args.output)
        args.output.with_suffix(".md").write_text(markdown(result))

    def schedule(case_id):
        yield from execution_order(args.blocks, args.seed)
        verdict = compare(result["blocks"], [case_id], args.blocks)[0]["verdict"]
        confirmation = getattr(args, "confirmation_blocks", 0)
        confirm = verdict in ("warning", "inconclusive") and confirmation > 0
        result["escalation"][case_id] = {"initial_verdict": verdict, "confirmation": confirm}
        if confirm:
            result["expected_blocks"][case_id] += confirmation
            for block_id, order in execution_order(args.blocks + confirmation, args.seed):
                if block_id >= args.blocks:
                    yield block_id, order

    try:
        with tempfile.TemporaryDirectory(prefix="menudet-qualification-") as directory:
            root = Path(directory)
            for case_id in profile["cases"]:
                for block_id, order in schedule(case_id):
                    for arm in order:
                        remaining = args.budget_seconds - (time.monotonic() - started)
                        if remaining <= 0:
                            return result
                        index = len(result["blocks"])
                        cache = root / f"cache-{index}"
                        cache.mkdir()
                        request = {
                            "case_id": case_id,
                            "arm": arm,
                            "block_id": block_id,
                            "order_index": index,
                            "size": profile["size"],
                            "seed": args.seed,
                            "threads": profile["threads"],
                            "settings": asdict(settings),
                            "budget_seconds": remaining,
                            "local_control": args.local_control,
                            "cache_dir": str(cache),
                            "manifest": manifests[arm],
                            "sensitivity_copy": getattr(args, "sensitivity_copy", False)
                            and arm == "candidate",
                        }
                        request_path, response_path = root / "request.json", root / "response.json"
                        response_path.unlink(missing_ok=True)
                        request_path.write_text(json.dumps(request))
                        env = os.environ.copy()
                        env.pop("PYTHONPATH", None)
                        env.update(
                            ME_DSL_JIT_CACHE_DIR=str(cache),
                            ME_DSL_JIT_POS_CACHE="0",
                            OMP_NUM_THREADS="1",
                            OPENBLAS_NUM_THREADS="1",
                            MKL_NUM_THREADS="1",
                        )
                        row = {k: request[k] for k in ("case_id", "arm", "block_id", "order_index")}
                        try:
                            process = subprocess.run(
                                [
                                    interpreters[arm],
                                    "-I",
                                    str(Path(__file__).resolve()),
                                    "--worker",
                                    str(request_path),
                                    str(response_path),
                                ],
                                cwd=root,
                                env=env,
                                capture_output=True,
                                text=True,
                                timeout=remaining,
                            )
                            if process.returncode != 0 or not response_path.exists():
                                row.update(
                                    status="failed",
                                    error=process.stderr,
                                    stdout=process.stdout,
                                    returncode=process.returncode,
                                )
                            else:
                                row = json.loads(response_path.read_text())
                                response_path.unlink()
                                verify_stack(row, result["blocks"])
                                if process.stderr or process.stdout:
                                    row.update(process_stderr=process.stderr, process_stdout=process.stdout)
                        except subprocess.TimeoutExpired:
                            row.update(status="incomplete", error="Process exceeded remaining budget")
                        except (OSError, ValueError) as error:
                            row.update(status="failed", error=str(error))
                        result["blocks"].append(row)
                        save()
    finally:
        save()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", nargs=2, metavar=("REQUEST", "RESPONSE"), help=argparse.SUPPRESS)
    parser.add_argument("--profile", choices=["smoke", "observation", "macos-arm"], default="smoke")
    parser.add_argument("--size", type=int, help="Explicit uncalibrated size override (calibration only)")
    parser.add_argument(
        "--sensitivity-copy", action="store_true", help="Benchmark-only extra-copy candidate control"
    )
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--baseline-python", default=sys.executable)
    parser.add_argument("--candidate-python", default=sys.executable)
    parser.add_argument("--baseline-manifest", type=Path)
    parser.add_argument("--candidate-manifest", type=Path)
    parser.add_argument("--local-control", action="store_true")
    parser.add_argument("--threads", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--confirmation-blocks", type=int, choices=[0, 2], default=2)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--target-ms", type=float, default=50)
    parser.add_argument("--budget-seconds", type=float, default=180)
    parser.add_argument("--output", type=Path, default=Path("menudet-qualification.json"))
    args = parser.parse_args()
    if args.worker:
        request, response = map(Path, args.worker)
        response.write_text(json.dumps(worker(json.loads(request.read_text()))))
        return 0
    result = run_pair(args)
    print(f"Coverage: {result['coverage']}; artifacts: {args.output}, {args.output.with_suffix('.md')}")
    return 0 if result["coverage"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
