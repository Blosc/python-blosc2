"""Run pytest with NumExpr unavailable, without modifying a shared installation.

CI uses --require-absent after uninstalling its optional reference package. Local
isolated builds can select a package root and a dependency-only search path that
omits both NumExpr and its distribution metadata. Optional external reference
tests see ModuleNotFoundError; direct Blosc2 import attempts fail qualification
even if an optional-import handler swallows that exception.
"""

import argparse
import importlib.abc
import importlib.metadata
import json
import os
import sys
from pathlib import Path


class BlockNumExpr(importlib.abc.MetaPathFinder):
    def __init__(self):
        self.runtime_imports = []

    def find_spec(self, fullname, path=None, target=None):
        if fullname != "numexpr" and not fullname.startswith("numexpr."):
            return
        frame = sys._getframe(1)
        try:
            while frame:
                owner = frame.f_globals.get("__name__", "")
                if not owner.startswith(("importlib", "_frozen_importlib")):
                    if owner == "blosc2" or owner.startswith("blosc2."):
                        self.runtime_imports.append(owner)
                    break
                frame = frame.f_back
        finally:
            del frame
        raise ModuleNotFoundError("NumExpr is excluded from this qualification run", name="numexpr")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-absent", action="store_true")
    parser.add_argument("--package-root", type=Path)
    parser.add_argument("--dependencies", type=Path)
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.dependencies:
        sys.path[:] = [p for p in sys.path if "site-packages" not in p and "dist-packages" not in p]
        sys.path.insert(0, str(args.dependencies.resolve()))
    if args.package_root:
        sys.path.insert(0, str(args.package_root.resolve()))
    selected = [str(path.resolve()) for path in (args.package_root, args.dependencies) if path is not None]
    if selected:
        os.environ["PYTHONPATH"] = os.pathsep.join(selected)
    # pytest subprocesses inherit the selected package/dependency paths. The
    # script's main guard is essential for multiprocessing's spawn start method.
    if "numexpr" in sys.modules:
        raise RuntimeError("NumExpr was imported before qualification started")
    if args.require_absent:
        try:
            importlib.metadata.distribution("numexpr")
        except importlib.metadata.PackageNotFoundError:
            pass
        else:
            raise RuntimeError("NumExpr distribution is still installed on the qualification path")
        if importlib.util.find_spec("numexpr") is not None:
            raise RuntimeError("NumExpr module is still available on the qualification path")
    blocker = BlockNumExpr()
    sys.meta_path.insert(0, blocker)
    import pytest

    import blosc2

    assert blosc2.numexpr is None
    requirements = importlib.metadata.requires("blosc2") or []
    assert not any(item.lower().startswith("numexpr") for item in requirements)
    options = args.pytest_args
    if options[:1] == ["--"]:
        options = options[1:]
    print(f"NumExpr-free package: {blosc2.__file__}", flush=True)
    print(f"Native extension: {blosc2.blosc2_ext.__file__}", flush=True)
    status = pytest.main(options)
    print(json.dumps({"numexpr_runtime_imports": blocker.runtime_imports, "pytest_status": int(status)}))
    return 1 if blocker.runtime_imports else int(status)


if __name__ == "__main__":
    raise SystemExit(main())
