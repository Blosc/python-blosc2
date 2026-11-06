"""Qualification cannot be borrowed from another SHA or from a lightweight gate."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "native_qualification", Path(__file__).parents[1] / "scripts" / "verify_native_qualification.py"
)
qualification = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qualification)


@pytest.mark.parametrize("sha", ["main", "abc123", "A" * 40])
def test_exact_revision_required(sha):
    with pytest.raises(ValueError, match="exact"):
        qualification.verify(sha, lambda _: pytest.fail("Invalid SHA must not call GitHub"))


@pytest.mark.parametrize("conclusion", ["success", "skipped", "failure"])
def test_full_gate_required(conclusion):
    sha = "a" * 40

    def get(path):
        if "workflows" in path:
            return {
                "workflow_runs": [{"head_sha": sha, "conclusion": "success", "id": 42, "html_url": "url"}]
            }
        return {
            "total_count": 1,
            "jobs": [
                {
                    "name": "Publish full native qualification",
                    "status": "completed",
                    "conclusion": conclusion,
                }
            ],
        }

    if conclusion == "success":
        assert qualification.verify(sha, get)["run_id"] == 42
    else:
        with pytest.raises(RuntimeError, match="full miniexpr"):
            qualification.verify(sha, get)


def test_other_revision_not_accepted():
    with pytest.raises(RuntimeError):
        qualification.verify(
            "a" * 40, lambda _: {"workflow_runs": [{"head_sha": "b" * 40, "conclusion": "success"}]}
        )


@pytest.mark.parametrize(
    ("name", "status"),
    [("Compact native gate", "completed"), ("Publish full native qualification", "in_progress")],
)
def test_lightweight_or_unfinished_job_not_accepted(name, status):
    sha = "a" * 40

    def get(path):
        if "workflows" in path:
            return {"workflow_runs": [{"head_sha": sha, "conclusion": "success", "id": 42}]}
        return {"total_count": 1, "jobs": [{"name": name, "status": status, "conclusion": "success"}]}

    with pytest.raises(RuntimeError, match="full miniexpr"):
        qualification.verify(sha, get)
