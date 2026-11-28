"""Run migration tests against an isolated installed wheel, never an editable tree.

Use conda's blosc2 Python. --package is a pip --target installation, not a source
directory. The selected wheel supplies blosc2; the current environment supplies
pytest and dependencies. Startup .pth files are deliberately not processed.
"""

import argparse
import subprocess
import sys
import sysconfig
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--artifact", choices=("on", "off"), required=True)
    parser.add_argument("--test", action="append", default=[])
    args = parser.parse_args()
    package = args.package.resolve()
    repo = Path(__file__).resolve().parents[1]
    tests = args.test or sorted(
        {
            str(path.relative_to(repo))
            for pattern in (
                "test_b2nd_exec*.py",
                "test_native_graph*.py",
                "test_native_expression_ingestion.py",
                "test_menudet*.py",
                "test_native_first.py",
                "test_lazyexpr*.py",
                "test_expression_graph*.py",
                "test_storage_readers.py",
            )
            for path in (repo / "tests").glob(pattern)
        }
    )
    roots = [str(package), sysconfig.get_path("purelib"), sysconfig.get_path("platlib")]
    code = f"""
import sys
sys.path[:0] = {roots!r}
from pathlib import Path
import blosc2
import pytest
root = Path({str(package)!r})
assert Path(blosc2.__file__).is_relative_to(root), blosc2.__file__
assert Path(blosc2.blosc2_ext.__file__).is_relative_to(root), blosc2.blosc2_ext.__file__
assert blosc2.blosc2_ext._b2nd_exec_available()
assert blosc2.blosc2_ext.menudet_legacy_adapter_available() is {args.artifact == "on"!r}
print("Python package:", blosc2.__file__, flush=True)
print("Native module:", blosc2.blosc2_ext.__file__, flush=True)
raise SystemExit(pytest.main(
    ["-n", "0", "--override-ini=addopts=", "--tb=short", "-q"] + {tests!r}
))
"""
    result = subprocess.run([sys.executable, "-S", "-c", code], cwd=repo)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
