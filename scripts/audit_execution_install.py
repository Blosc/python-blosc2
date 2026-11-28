"""Audit an installed execution prefix; save raw platform-specific ABI evidence."""

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
from pathlib import Path


def audit(prefix, results, numerical, wasm=False):
    results.mkdir(parents=True, exist_ok=True)
    header = next(prefix.rglob("b2nd_exec.h"))
    companion = list(prefix.rglob("b2nd_exec_kernel.h"))
    assert bool(companion) == numerical, companion
    public = set(re.findall(r"BLOSC_EXPORT\s+\w+\s+(b2nd_\w+)\s*\(", header.read_text()))
    assert len(public) == 14, sorted(public)
    libraries = [
        p
        for p in prefix.rglob("*")
        if p.is_file()
        and "blosc2" in p.name
        and p.suffix in ((".a",) if wasm else (".dll", ".dylib", ".so"))
    ]
    library = next(iter(libraries))
    if wasm:
        command = ["emnm", "--defined-only", str(library)]
    elif os.name == "nt":
        command = ["dumpbin", "/exports", str(library)]
    elif platform.system() == "Darwin":
        command = ["nm", "-gU", str(library)]
    else:
        command = ["nm", "-D", "--defined-only", str(library)]
    symbols = subprocess.check_output(command, text=True)
    (results / "exports.txt").write_text(symbols)
    exported = set(re.findall(r"\b_?(b2nd_\w+)\b", symbols))
    assert public <= exported, sorted(public - exported)
    assert ("b2nd_plan_from_kernel" in exported) == numerical
    assert not re.search(r"\b_?(?:me_artifact_|me_graph_|yyjson_)\w+", symbols)
    dependencies = (
        ["dumpbin", "/dependents", str(library)]
        if os.name == "nt"
        else ["otool", "-L", str(library)]
        if platform.system() == "Darwin"
        else ["ldd", str(library)]
    )
    (results / "dependencies.txt").write_text(
        "WASM static archive; dependencies verified by external link/runtime consumer\n"
        if wasm
        else subprocess.check_output(dependencies, text=True)
    )
    pc_files = list(prefix.rglob("blosc2.pc"))
    assert pc_files, "Missing installed blosc2.pc"
    if shutil.which("pkg-config"):
        environment = dict(os.environ, PKG_CONFIG_PATH=str(pc_files[0].parent))
        metadata = subprocess.check_output(
            ["pkg-config", "--modversion", "--cflags", "--libs", "blosc2"],
            env=environment,
            text=True,
        )
        (results / "pkg-config.txt").write_text(metadata)
    else:
        (results / "pkg-config.txt").write_text(
            "pkg-config unavailable; CMake consumer checked separately\n"
        )
    report = {
        "prefix": str(prefix),
        "library": str(library),
        "numerical": numerical,
        "public_exports": sorted(public),
        "kernel_export": "b2nd_plan_from_kernel" in exported,
    }
    (results / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--numerical", choices=("on", "off"), required=True)
    parser.add_argument("--wasm", action="store_true")
    args = parser.parse_args()
    audit(args.prefix.resolve(), args.results.resolve(), args.numerical == "on", args.wasm)


if __name__ == "__main__":
    main()
