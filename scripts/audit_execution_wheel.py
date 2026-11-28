"""Audit installed wheel/link inputs without importing an editable installation."""

import argparse
import json
import os
import platform
import re
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--artifact", choices=("on", "off"), required=True)
    args = parser.parse_args()
    args.results.mkdir(parents=True, exist_ok=True)
    extension = next(
        path for path in (args.package / "blosc2").glob("blosc2_ext*") if path.suffix in (".so", ".pyd")
    )
    headers = args.package / "blosc2/include"
    assert (headers / "b2nd_exec.h").is_file()
    assert (headers / "b2nd_exec_kernel.h").is_file()
    commands = subprocess.check_output(
        ["ninja", "-C", str(args.build), "-t", "commands", "blosc2_ext"], text=True
    )
    (args.results / "link-inputs.txt").write_text(commands)
    links = [
        line
        for line in commands.splitlines()
        if "blosc2_ext" in line
        and (" -o " in line or "/out:" in line.lower())
        and re.search(r"(?:libminiexpr\.a|miniexpr(?:_static)?\.lib)", line)
    ]
    assert links, "Cannot identify extension link command/core provider"
    link = links[-1]
    core_inputs = re.findall(r"[\w./\\:-]*(?:libminiexpr\.a|miniexpr(?:_static)?\.lib)", link)
    assert len(set(core_inputs)) == 1, core_inputs
    forbidden = re.compile(r"(?:me_artifact_|me_graph_|yyjson_)\w+")
    if os.name == "nt":
        # PE exports alone cannot prove absence of statically linked internals.
        # The linker map retains that evidence after the .pyd symbol table is stripped.
        symbols = next(args.build.rglob("blosc2_ext*.map")).read_text()
        dependencies = subprocess.check_output(["dumpbin", "/dependents", str(extension)], text=True)
    else:
        symbols = subprocess.check_output(["nm", "-g", str(extension)], text=True)
        dependencies = subprocess.check_output(
            ["otool", "-L", str(extension)] if platform.system() == "Darwin" else ["ldd", str(extension)],
            text=True,
        )
    (args.results / "symbols.txt").write_text(symbols)
    (args.results / "dependencies.txt").write_text(dependencies)
    providers = re.findall(r"\b_?me_prepare_formula\b", symbols)
    assert providers, "Core preparation provider missing from symbol evidence"
    if args.artifact == "off":
        assert not forbidden.search(symbols), "Unexpected native artifact/JSON symbols"
        assert "miniexpr_artifact" not in link
        assert "yyjson" not in link
        assert not any("yyjson" in path.name.lower() for path in (args.build / "_deps").glob("*"))
    report = {"extension": str(extension), "artifact": args.artifact, "core_inputs": core_inputs}
    (args.results / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
