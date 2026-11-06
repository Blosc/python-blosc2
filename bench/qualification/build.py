"""Build an exact integrated revision without modifying a working checkout.

The caller provisions/fixes dependencies first. Every build uses git archive,
its revision's own native pins, and an independent persistent build directory.
"""

import argparse
import hashlib
import io
import json
import os
import platform
import re
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def exact_sha(value):
    if not re.fullmatch("[0-9a-f]{40}", value):
        raise ValueError("An exact lowercase 40-character SHA is required")
    return value


def native_pins(cmake):
    miniexpr = re.search(r"FetchContent_Declare\(miniexpr\s+GIT_REPOSITORY\s+\S+\s+GIT_TAG\s+(\w+)", cmake)
    blosc = re.search(r"^set\(BLOSC2_BUNDLED_VERSION\s+(\w+)\)", cmake, re.MULTILINE)
    if not miniexpr or not blosc:
        raise ValueError("Cannot identify integrated native dependency pins")
    return {"miniexpr_revision": exact_sha(miniexpr[1]), "c_blosc2_revision": exact_sha(blosc[1])}


def wheel_files(path):
    with zipfile.ZipFile(path) as archive:
        names = [n for n in archive.namelist() if n.startswith("blosc2/") and not n.endswith("/")]
        return {name: sha256(archive.read(name)) for name in names}


def build(args):
    started = time.monotonic()
    exact_sha(args.revision)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source, build_dir, wheels = output / "source", output / "build", output / "wheels"
    if source.exists() or build_dir.exists() or wheels.exists():
        raise ValueError("Build output must not contain a previous source/build/wheel tree")
    source.mkdir()
    archive = subprocess.check_output(["git", "-C", str(args.repository), "archive", args.revision])
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(source, filter="data")
    pins = native_pins((source / "CMakeLists.txt").read_text())
    compiler = os.environ.get("CC", "cc")
    compiler_version = subprocess.check_output([compiler, "--version"], text=True).splitlines()[0]
    configuration = {
        "build_type": "Release",
        "compiler": compiler_version,
        "cflags": os.environ.get("CFLAGS"),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
    }
    command = [
        sys.executable,
        "-m",
        "pip",
        "wheel",
        str(source),
        "--no-deps",
        "--no-build-isolation",
        "--wheel-dir",
        str(wheels),
        "--config-settings",
        f"build-dir={build_dir}",
        "--config-settings",
        "cmake.build-type=Release",
    ]
    env = os.environ.copy()
    env.pop("USE_SYSTEM_BLOSC2", None)
    env["CMAKE_BUILD_PARALLEL_LEVEL"] = str(args.jobs)
    with (output / "build.log").open("w") as log:
        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    paths = list(wheels.glob("*.whl"))
    if len(paths) != 1:
        raise ValueError("Expected exactly one distribution wheel")
    for name, key in (("miniexpr", "miniexpr_revision"), ("blosc2", "c_blosc2_revision")):
        actual = subprocess.check_output(
            ["git", "-C", str(build_dir / "_deps" / f"{name}-src"), "rev-parse", "HEAD"], text=True
        ).strip()
        if actual != pins[key]:
            raise ValueError(f"Built native checkout differs from declared {key}")
    manifest = {
        "python_revision": args.revision,
        **pins,
        "wheel_sha256": sha256(paths[0].read_bytes()),
        "wheel_filename": paths[0].name,
        "installed_files": wheel_files(paths[0]),
        "dependency_lock_sha256": sha256(args.lock.read_bytes()),
        "build_configuration": configuration,
        "build_wall_seconds": time.monotonic() - started,
        "source_archive_sha256": sha256(archive),
        "native_build_options": [
            line
            for line in (build_dir / "CMakeCache.txt").read_text().splitlines()
            if re.match(r"(?:MINIEXPR|CMAKE_C_(?:FLAGS|COMPILER)|BLOSC2).*:", line)
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Built {args.revision}: {paths[0].name} ({manifest['build_wall_seconds']:.1f} s)")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    build(args)


if __name__ == "__main__":
    main()
