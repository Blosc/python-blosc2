#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""Unit tests for b2view cell formatting (no app session needed)."""

import numpy as np

from blosc2.b2view.render import column_float_decimals, format_cell


def test_column_decimals_follow_max_magnitude():
    assert column_float_decimals(np.array([0.1, 5.0])) == 6
    assert column_float_decimals(np.array([0.1, 44.5])) == 5
    assert column_float_decimals(np.array([0.1, 448.5])) == 4
    assert column_float_decimals(np.array([0.1, 123456.7])) == 1
    assert column_float_decimals(np.array([0.1, 12345678.0])) == 0


def test_column_decimals_special_cases():
    # All-zero columns read best as plain 0.0
    assert column_float_decimals(np.zeros(3)) == 1
    # Scientific-notation territory and non-float columns: per-value fallback
    assert column_float_decimals(np.array([1e10])) is None
    assert column_float_decimals(np.array([1e-9])) is None
    assert column_float_decimals(np.arange(5)) is None
    assert column_float_decimals(np.array(["a", "b"])) is None
    assert column_float_decimals(np.array([])) is None
    assert column_float_decimals(np.array([np.nan])) is None
    # NaN/inf cells are ignored when picking the column format
    assert column_float_decimals(np.array([np.nan, 1.5])) == 6


def test_format_cell_uniform_decimals_align():
    vals = np.array([0.0, 1.5, -3.25, 448.5])
    decimals = column_float_decimals(vals)
    cells = [format_cell(v, float_decimals=decimals) for v in vals]
    assert cells == ["   0.0000", "   1.5000", "  -3.2500", " 448.5000"]
    # Same width and aligned decimal points for the whole column
    assert len({len(cell) for cell in cells}) == 1
    assert len({cell.index(".") for cell in cells}) == 1


def test_format_cell_without_column_context_unchanged():
    # The per-value fallback keeps its historical behavior
    assert format_cell(np.float64(0.0)) == " 0.0"
    assert format_cell(np.float64(1.5)) == " 1.500000"
