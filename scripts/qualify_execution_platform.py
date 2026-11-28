"""Qualify execution-migration sources with the selected environment/toolchain.

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


def toolchain_options(compiler, sanitizers):
    options = ["-DCMAKE_C_COMPILER=" + compiler] if compiler else []
    if sanitizers:
        options += [
            "-DCMAKE_C_FLAGS=-fsanitize=address,undefined -fno-omit-frame-pointer",
            "-DCMAKE_EXE_LINKER_FLAGS=-fsanitize=address,undefined",
            "-DCMAKE_SHARED_LINKER_FLAGS=-fsanitize=address,undefined",
        ]
    return options


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--phase", action="append", default=[])
    parser.add_argument("--compiler", help="C compiler; default is CMake's selected compiler")
    parser.add_argument("--sanitizers", action="store_true", help="Instrument native C4 builds")
    args = parser.parse_args()
    sources, work, results = (p.resolve() for p in (args.sources, args.work, args.results))
    work.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, CMAKE_BUILD_PARALLEL_LEVEL=str(args.jobs))
    # CTest executables and external consumers must find the candidate DLLs.
    environment["PATH"] = (
        os.pathsep.join(
            str(path)
            for path in (
                work / "native-prefix/bin",
                work / "native-prefix/lib",
                work / "native-blosc2/blosc",
                work / "compression/blosc",
            )
        )
        + os.pathsep
        + environment.get("PATH", "")
    )
    (results / "environment.json").write_text(
        json.dumps(
            {
                "platform": platform.platform(),
                "machine": platform.machine(),
                "python": sys.version,
                "executable": sys.executable,
                "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
                "compiler": args.compiler,
                "sanitizers": args.sanitizers,
                "revisions": {
                    name: subprocess.check_output(
                        ["git", "-C", str(sources / name), "rev-parse", "HEAD"], text=True
                    ).strip()
                    for name in ("miniexpr", "c-blosc2", "python-blosc2")
                },
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
        options = [*options, *toolchain_options(args.compiler, args.sanitizers)]
        run(["cmake", "-S", source, "-B", directory, "-G", "Ninja", *options], log)
        run(["cmake", "--build", directory, "--parallel", args.jobs], log)
        if test:
            run(
                [
                    "ctest",
                    "--test-dir",
                    directory,
                    "--output-on-failure",
                    "--timeout",
                    "180",
                    "--parallel",
                    args.jobs,
                    "--output-junit",
                    results / (name + ".xml"),
                ],
                log,
            )
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
                    "--junit",
                    results / ("wheel-" + kind + ".xml"),
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
        options = [
            "-DCMAKE_PREFIX_PATH=" + str(work / "native-prefix"),
            "-DCMAKE_BUILD_TYPE=Release",
        ]
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
        directory = build(
            sources / "c-blosc2",
            "compression",
            [
                "-DCMAKE_BUILD_TYPE=Release",
                "-DBUILD_TESTS=ON",
                "-DBUILD_EXAMPLES=OFF",
                "-DBUILD_BENCHMARKS=OFF",
                "-DBUILD_FUZZERS=OFF",
                "-DCMAKE_INSTALL_PREFIX=" + str(work / "compression-prefix"),
            ],
            log,
        )
        run(["cmake", "--install", directory], log)
        run(
            [
                sys.executable,
                sources / "python-blosc2/scripts/audit_execution_install.py",
                "--prefix",
                work / "compression-prefix",
                "--results",
                results / "compression-audit",
                "--numerical",
                "off",
            ],
            log,
        )
        build(
            sources / "python-blosc2/scripts/execution-consumer",
            "compression-consumer",
            ["-DCMAKE_PREFIX_PATH=" + str(work / "compression-prefix"), "-DEXPECT_EXECUTION=OFF"],
            log,
        )

    def public_api(log):
        prefix = work / "native-prefix"
        core_directory = build(
            sources / "miniexpr",
            "native-core",
            core_options
            + [
                "-DMINIEXPR_BUILD_ARTIFACT=OFF",
                "-DMINIEXPR_INSTALL_CORE=ON",
                "-DMINIEXPR_BUILD_SHARED=OFF",
                "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
                "-DCMAKE_INSTALL_PREFIX=" + str(prefix),
            ],
            log,
        )
        run(["cmake", "--install", core_directory], log)
        directory = build(
            sources / "c-blosc2",
            "native-blosc2",
            [
                "-DCMAKE_BUILD_TYPE=Release",
                "-DBLOSC2_MINIEXPR_ROOT=" + str(prefix),
                "-DCMAKE_INSTALL_PREFIX=" + str(prefix),
                "-DBUILD_TESTS=ON",
                "-DBUILD_EXAMPLES=ON",
                "-DBUILD_BENCHMARKS=OFF",
                "-DBUILD_FUZZERS=OFF",
            ],
            log,
            test=False,
        )
        run(
            [
                "ctest",
                "--test-dir",
                directory,
                "--output-on-failure",
                "--timeout",
                "180",
                "-R",
                "test_b2nd_exec_api|b2nd_example_exec_eval",
                "--no-tests=error",
                "--output-junit",
                results / "public-api.xml",
            ],
            log,
        )
        run(["cmake", "--install", directory], log)
        run(
            [
                sys.executable,
                sources / "python-blosc2/scripts/audit_execution_install.py",
                "--prefix",
                prefix,
                "--results",
                results / "public-audit",
                "--numerical",
                "on",
            ],
            log,
        )
        # A shared consumer tests DLL import/export independently of static tests.
        build(
            sources / "python-blosc2/scripts/execution-consumer",
            "public-consumer",
            ["-DCMAKE_PREFIX_PATH=" + str(prefix), "-DEXPECT_EXECUTION=ON"],
            log,
        )

    phases = {
        "public-api": public_api,
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
    unknown = set(args.phase) - phases.keys()
    if unknown:
        parser.error(f"Unknown phases: {sorted(unknown)}")
    return run_phases(phases, args.phase or phases, results)


if __name__ == "__main__":
    raise SystemExit(main())
