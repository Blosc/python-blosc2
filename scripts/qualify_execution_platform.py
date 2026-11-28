"""Qualify local Linux execution-migration sources inside the blosc2 conda environment.

Pass sibling source directories named miniexpr, c-blosc2 and python-blosc2.
Builds, installations and logs stay under --work/--results, outside the sources.
This is a correctness runner; emulated results are not performance qualification.
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import sysconfig
from pathlib import Path


def run_phases(phases, selected, results):
    status_path = results / "status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    failed = False
    for name in selected:
        print(f"{name}: starting", flush=True)
        try:
            with (results / (name + ".log")).open("w") as log:
                phases[name](log)
        except (OSError, subprocess.CalledProcessError, StopIteration) as error:
            status[name] = str(error)
            failed = True
        else:
            status[name] = "passed"
        status_path.write_text(json.dumps(status, indent=2) + "\n")
        print(f"{name}: {status[name]}", flush=True)
    return int(failed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--phase", action="append", default=[])
    args = parser.parse_args()
    sources, work, results = (p.resolve() for p in (args.sources, args.work, args.results))
    work.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, CMAKE_BUILD_PARALLEL_LEVEL=str(args.jobs))
    (results / "environment.json").write_text(
        json.dumps(
            {
                "platform": platform.platform(),
                "machine": platform.machine(),
                "python": sys.version,
                "executable": sys.executable,
                "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            },
            indent=2,
        )
        + "\n"
    )

    def run(command, log, cwd=None):
        print("+", *map(str, command), file=log, flush=True)
        subprocess.run(
            list(map(str, command)),
            cwd=cwd,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )

    def build(source, name, options, log, test=True):
        directory = work / name
        run(["cmake", "-S", source, "-B", directory, "-G", "Ninja", *options], log)
        run(["cmake", "--build", directory, "--parallel", args.jobs], log)
        if test:
            run(["ctest", "--test-dir", directory, "--output-on-failure", "--parallel", args.jobs], log)
        return directory

    core_options = [
        "-DCMAKE_BUILD_TYPE=Release",
        "-DMINIEXPR_BUILD_TESTS=ON",
        "-DMINIEXPR_BUILD_EXAMPLES=OFF",
        "-DMINIEXPR_BUILD_BENCH=OFF",
    ]

    def core(log, artifact):
        build(
            sources / "miniexpr",
            "core-" + artifact,
            core_options
            + [
                "-DMINIEXPR_BUILD_ARTIFACT=" + artifact.upper(),
                "-DCMAKE_C_COMPILER=" + ("gcc" if artifact == "on" else "clang"),
                "-DMINIEXPR_USE_SLEEF=" + artifact.upper(),
                "-DMINIEXPR_ENABLE_TCC_JIT=" + artifact.upper(),
            ],
            log,
        )

    def wheel(log, kind):
        directory = work / ("wheel-" + kind)
        wheels = work / ("wheels-" + kind)
        package = work / ("package-" + kind)
        run(
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                ".",
                "--no-deps",
                "--no-build-isolation",
                "-w",
                wheels,
                "-Cbuild-dir=" + str(directory),
                "-Ccmake.define.B2_EXPERIMENTAL_SHARED_MINIEXPR_CORE="
                + ("OFF" if kind == "standard" else "ON"),
                "-Ccmake.define.MINIEXPR_BUILD_ARTIFACT=" + ("OFF" if kind == "off" else "ON"),
                "-Ccmake.define.FETCHCONTENT_SOURCE_DIR_MINIEXPR=" + str(sources / "miniexpr"),
                "-Ccmake.define.FETCHCONTENT_SOURCE_DIR_BLOSC2=" + str(sources / "c-blosc2"),
            ],
            log,
            sources / "python-blosc2",
        )
        run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--upgrade",
                "--target",
                package,
                next(wheels.glob("*.whl")),
            ],
            log,
        )
        if kind != "standard":
            run(
                [
                    sys.executable,
                    sources / "python-blosc2/scripts/qualify_execution_migration.py",
                    "--package",
                    package,
                    "--artifact",
                    kind,
                ],
                log,
                sources / "python-blosc2",
            )
        else:
            roots = [str(package), sysconfig.get_path("purelib"), sysconfig.get_path("platlib")]
            code = f"""
import sys
sys.path[:0] = {roots!r}
from pathlib import Path
import blosc2, pytest
assert Path(blosc2.__file__).is_relative_to({str(package)!r})
assert Path(blosc2.blosc2_ext.__file__).is_relative_to({str(package)!r})
assert not blosc2.blosc2_ext._b2nd_exec_available()
assert blosc2.blosc2_ext.menudet_legacy_adapter_available()
raise SystemExit(pytest.main(["-n", "0", "--override-ini=addopts=", "--tb=short", "-q",
    "tests/test_b2nd_exec_m8.py", "tests/test_menudet_decorator.py",
    "tests/test_menudet_artifact.py", "tests/test_menudet_native_graph.py",
    "tests/test_native_expression_ingestion.py", "tests/test_native_graph_output.py"]))
"""
            run([sys.executable, "-S", "-c", code], log, sources / "python-blosc2")
        if kind == "off":
            run(["cmake", "--install", directory, "--prefix", work / "prefix"], log)

    def native(log, name):
        source = "installed-miniexpr" if name == "consumers" else name.removeprefix("san-")
        options = ["-DCMAKE_PREFIX_PATH=" + str(work / "prefix/blosc2")]
        if name.startswith("san-"):
            options += [
                "-DCMAKE_BUILD_TYPE=Debug",
                "-DCMAKE_C_FLAGS=-fsanitize=address,undefined -fno-omit-frame-pointer",
                "-DCMAKE_EXE_LINKER_FLAGS=-fsanitize=address,undefined",
            ]
        if name == "san-exec-plan":
            options += ["-DEXEC_PLAN_TEST_JIT=ON"]
        build(sources / "c-blosc2/tests" / source, name, options, log)

    def compression(log):
        build(
            sources / "c-blosc2",
            "compression",
            [
                "-DCMAKE_BUILD_TYPE=Release",
                "-DBUILD_TESTS=ON",
                "-DBUILD_EXAMPLES=OFF",
                "-DBUILD_BENCHMARKS=OFF",
                "-DBUILD_FUZZERS=OFF",
            ],
            log,
        )

    phases = {
        "core-on": lambda log: core(log, "on"),
        "core-off": lambda log: core(log, "off"),
        "wheel-off": lambda log: wheel(log, "off"),
        "wheel-on": lambda log: wheel(log, "on"),
        "wheel-standard": lambda log: wheel(log, "standard"),
        **{
            name: lambda log, name=name: native(log, name)
            for name in ("consumers", "san-exec-plan", "san-exec-storage", "san-exec-stages")
        },
        "compression": compression,
    }
    for name in args.phase or phases:
        if name not in phases:
            parser.error(f"Unknown phase: {name}")
    return run_phases(phases, args.phase or phases, results)


if __name__ == "__main__":
    raise SystemExit(main())
