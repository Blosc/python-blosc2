"""Wheel smoke tests can disable pytest's optional cache provider."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import blosc2


@pytest.mark.skipif(blosc2.IS_WASM, reason="Nested pytest requires subprocess support")
def test_without_cache_provider(tmp_path):
    shutil.copyfile(Path(__file__).with_name("conftest.py"), tmp_path / "conftest.py")
    (tmp_path / "test_smoke.py").write_text("def test_smoke():\n    pass\n", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q", "test_smoke.py"],
        cwd=tmp_path,
        env={**os.environ, "PYTEST_ADDOPTS": ""},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
