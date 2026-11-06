"""Certification references must follow the public Menudet numerical profile."""

import json
from pathlib import Path

import numpy as np
import pytest

VECTORS = Path(__file__).parents[1] / "scripts" / "menudet_math_vectors.json"
REMAINDERS = [case for case in json.loads(VECTORS.read_text())["cases"] if case["function"] == "remainder"]


@pytest.mark.parametrize("case", REMAINDERS)
def test_remainder_reference_uses_divisor_sign(case):
    x, y = [np.asarray(float.fromhex(value), dtype=case["dtype"]) for value in case["inputs"]]
    assert float(np.remainder(x, y)).hex() == case["expected"]
    assert case["max_ulp"] == 0


def test_remainder_reference_covers_all_operand_signs():
    for dtype in ("float32", "float64"):
        signs = {
            tuple(float.fromhex(value) < 0 for value in case["inputs"])
            for case in REMAINDERS
            if case["dtype"] == dtype
        }
        assert signs == {(False, False), (False, True), (True, False), (True, True)}
