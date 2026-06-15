#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""Identity-case fast path for Column strided slicing.

A clean base CTable (no column mask, no sorted/filtered view, no deletions)
has logical row k == physical row k, so a positive-step logical slice is a
physical slice and can be read straight from the underlying NDArray — skipping
the O(nrows) live-position scan and reaching NDArray's strided-gather fast path
(see ``Column._has_identity_positions`` / ``_values_from_key`` in ctable.py).

These tests check that the fast path returns values identical to the
position-gather path, and that it engages only when it is safe to.
"""

from dataclasses import dataclass
from unittest import mock

import numpy as np
import pytest

import blosc2
from blosc2 import CTable
from blosc2.ctable import Column


@dataclass
class Row:
    a: int = blosc2.field(blosc2.int64(), default=0)
    b: float = blosc2.field(blosc2.float64(), default=0.0)
    t: int = blosc2.field(blosc2.timestamp(), default=0)  # exercises timestamp decode


def _make(n=1000, a_values=None):
    arr = np.empty(n, dtype=[("a", "<i8"), ("b", "<f8"), ("t", "<i8")])
    arr["a"] = np.arange(n) if a_values is None else a_values
    arr["b"] = np.arange(n) * 0.5
    arr["t"] = np.arange(n) * 1_000_000
    table = CTable(Row, expected_size=n)
    table.extend(arr, validate=False)
    return table, arr


POSITIVE_SLICES = [
    np.s_[::100],
    np.s_[5:900:7],
    np.s_[:50],
    np.s_[3:],
    np.s_[::1],
    np.s_[10:10],  # empty
    np.s_[100:50],  # empty (start > stop)
]

NEGATIVE_SLICES = [
    np.s_[::-1],
    np.s_[::-3],
    np.s_[900:5:-7],
    np.s_[-1::-1],
    np.s_[5:900:-1],  # empty (start < stop with negative step)
]

ALL_SLICES = POSITIVE_SLICES + NEGATIVE_SLICES


def _force_slow(col, key):
    """Read *key* with the identity fast path disabled (position-gather path)."""
    with mock.patch.object(Column, "_has_identity_positions", return_value=False):
        return np.asarray(col[key])


def _resolve_spy():
    """Patch context + a list recording _resolve_live_positions calls."""
    calls = []
    orig = Column._resolve_live_positions

    def spy(self):
        calls.append(self._col_name)
        return orig(self)

    return mock.patch.object(Column, "_resolve_live_positions", spy), calls


# ---------------------------------------------------------------------------
# Correctness: fast path == position-gather path, and == NumPy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", ALL_SLICES)
@pytest.mark.parametrize("colname", ["a", "b", "t"])
def test_identity_fast_path_matches_slow_path(key, colname):
    table, _ = _make()
    fast = np.asarray(table[colname][key])
    slow = _force_slow(table[colname], key)
    np.testing.assert_array_equal(fast, slow)


@pytest.mark.parametrize("key", ALL_SLICES)
@pytest.mark.parametrize("colname", ["a", "b"])
def test_identity_fast_path_matches_numpy(key, colname):
    table, arr = _make()
    np.testing.assert_array_equal(np.asarray(table[colname][key]), arr[colname][key])


def test_timestamp_column_is_decoded_on_fast_path():
    table, _ = _make()
    fast = np.asarray(table["t"][::100])
    assert np.issubdtype(fast.dtype, np.datetime64)  # decode applied, not raw int64


# ---------------------------------------------------------------------------
# Dispatch: the fast path engages only when positions are the identity
# ---------------------------------------------------------------------------


def test_clean_positive_step_skips_position_scan():
    table, _ = _make()
    patcher, calls = _resolve_spy()
    with patcher:
        _ = table["a"][::100]
    assert calls == []  # identity fast path: no live-position scan


def test_clean_negative_step_uses_fast_path():
    # Negative steps on a clean table are also the identity case now: the slice
    # is read straight from the NDArray, no live-position scan.
    table, arr = _make()
    patcher, calls = _resolve_spy()
    with patcher:
        result = np.asarray(table["a"][::-3])
    assert calls == []
    np.testing.assert_array_equal(result, arr["a"][::-3])


@pytest.mark.parametrize("key", [np.s_[::10], np.s_[::-3], np.s_[15:2:-1]])
def test_deletions_use_position_path_and_stay_correct(key):
    table, arr = _make()
    table.delete([3, 7, 50])
    expected = np.delete(arr, [3, 7, 50])
    patcher, calls = _resolve_spy()
    with patcher:
        result = np.asarray(table["a"][key])
    assert calls  # identity broken by deletions
    np.testing.assert_array_equal(result, expected["a"][key])


@pytest.mark.parametrize("key", [np.s_[::5], np.s_[::-2], np.s_[::-1]])
def test_filtered_view_uses_position_path_and_stays_correct(key):
    table, arr = _make()
    view = table.where("a >= 500")
    expected = arr["b"][arr["a"] >= 500]
    patcher, calls = _resolve_spy()
    with patcher:
        result = np.asarray(view["b"][key])
    assert calls  # a view is not the identity case
    np.testing.assert_allclose(result, expected[key])


def test_materialized_sort_is_identity_and_correct():
    # Non-inplace sort_by returns a physically materialized table (base is None,
    # no cached positions), so it is a legitimate identity case: the fast path
    # is used and must return the rows in sorted order.
    rng = np.random.default_rng(0)
    a_values = rng.permutation(1000)
    table, arr = _make(a_values=a_values)
    sorted_table = table.sort_by("a")
    patcher, calls = _resolve_spy()
    with patcher:
        result = np.asarray(sorted_table["b"][::50])
    assert calls == []  # materialized sorted table is the identity case
    order = np.argsort(arr["a"], kind="stable")
    np.testing.assert_allclose(result, arr["b"][order][::50])


def test_setitem_strided_unaffected():
    """The fast path is read-only; strided assignment is unchanged."""
    table, arr = _make()
    table["a"][::100] = -1
    expected = arr["a"].copy()
    expected[::100] = -1
    np.testing.assert_array_equal(np.asarray(table["a"][:]), expected)
