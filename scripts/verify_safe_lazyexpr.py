"""Run the bounded safe-graph corpus and record actual platform/GIL/skip evidence.

This verifies the selected tests, not the complete experiment's exit criteria.
It never disables the GIL or interprets free-threaded builds as no-GIL success.
"""

import argparse
import importlib.util
import json
import platform
import sys
import sysconfig
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

import blosc2


class Outcomes:
    def __init__(self):
        self.cases = {}

    def pytest_runtest_logreport(self, report):
        if (report.when == "call" or report.failed or report.skipped) and self.cases.get(
            report.nodeid, {}
        ).get("outcome") != "failed":
            result = {"outcome": report.outcome}
            if report.skipped:
                result["reason"] = str(report.longrepr)
            self.cases[report.nodeid] = result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    selected = [
        "test_expression_graph.py",
        "test_expression_graph_properties.py",
        "test_expression_graph_lifetime.py",
        "test_portable_lazy.py",
        "ctable/test_ctable_computed_cols.py::test_safe_column_graph_observes_update_at_controlled_read_boundary",
        "ctable/test_ctable_computed_cols.py::test_safe_portable_column_metadata_does_not_evaluate_rows",
    ]
    parquet = "ctable/test_remote_parquet.py"
    if all(importlib.util.find_spec(name) is not None for name in ("pyarrow", "fsspec")):
        selected += [
            f"{parquet}::{name}"
            for name in (
                "test_safe_parquet_operand_checks_metadata_without_row_group_reads",
                "test_safe_parquet_operand_rejects_invalid_lifetime_before_reads",
                "test_safe_parquet_operand_rejects_same_metadata_storage_rebinding",
                "test_safe_parquet_ndarray_operand_rejects_unreviewed_row_shape",
                "test_safe_parquet_operand_rejects_hostile_dependencies_before_hooks",
                "test_safe_parquet_graph_refresh_waits_for_read_and_invalidates_old_graph",
            )
        ]
    else:
        # Collect the module itself so importorskip records absent adapters;
        # node-qualified selectors for an uncollectable module can exit with 4.
        selected.append(parquet)
    outcomes = Outcomes()
    code = pytest.main(
        ["-q", "--override-ini", "addopts=", *[str(root / "tests" / name) for name in selected]],
        plugins=[outcomes],
    )
    result = {
        "scope": "bounded experimental safe-graph corpus; not whole-project safety certification",
        "exit_code": int(code),
        "platform": platform.platform(),
        "python": sys.version,
        "numpy": np.__version__,
        "blosc2": blosc2.__version__,
        "c_blosc2": blosc2.blosclib_version,
        "package": blosc2.__file__,
        "free_threaded_build": bool(sysconfig.get_config_var("Py_GIL_DISABLED")),
        "gil_enabled_after_tests": getattr(sys, "_is_gil_enabled", lambda: None)(),
        "counts": dict(Counter(case["outcome"] for case in outcomes.cases.values())),
        "exceptions": {name: case for name, case in outcomes.cases.items() if case["outcome"] != "passed"},
    }
    text = json.dumps(result, indent=2)
    if args.report:
        args.report.write_text(text + "\n")
    print(text)
    return int(code)


if __name__ == "__main__":
    raise SystemExit(main())
