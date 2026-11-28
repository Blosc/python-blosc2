"""Harness tests: no native build or installed blosc2 package is required."""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from qualify_execution_platform import run_phases, toolchain_options


class QualificationTests(unittest.TestCase):
    def test_failure_is_recorded_and_other_phases_continue(self):
        visited = []

        def fail(log):
            log.write("intentional failure\n")
            raise subprocess.CalledProcessError(1, ["fixture"])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "status.json").write_text('{"stale": "passed"}')
            code = run_phases(
                {"bad": fail, "good": lambda log: visited.append("good")},
                ["bad", "good"],
                root,
            )
            assert code == 1
            assert visited == ["good"]
            status = json.loads((root / "status.json").read_text())
            assert status["good"] == "passed"
            assert "stale" not in status
            assert "fixture" in status["bad"]
            assert "intentional failure" in (root / "bad.log").read_text()

    def test_toolchain_defaults_do_not_override_environment(self):
        assert toolchain_options(None, False) == []
        assert toolchain_options("cl", False) == ["-DCMAKE_C_COMPILER=cl"]

    def test_sanitizers_cover_compile_and_link(self):
        options = toolchain_options("clang", True)
        assert "-DCMAKE_C_COMPILER=clang" in options
        for kind in ("C_FLAGS", "EXE_LINKER_FLAGS", "SHARED_LINKER_FLAGS"):
            assert any(option.startswith(f"-DCMAKE_{kind}=-fsanitize=") for option in options)


if __name__ == "__main__":
    unittest.main()
