#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""Tests for b2view's CTable group-by ('G'): model layer + one TUI journey.

Group-by replaces the table with a small materialised result (one row per
group, columns = key + aggregate).  The model tests drive ``StoreBrowser``
directly (no Textual); the TUI test drives the real app through a ``Pilot``.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import blosc2

NROWS = 240
VENDORS = ["acme", "globex", "initech"]


@dataclasses.dataclass
class _Row:
    vendor: str = blosc2.field(blosc2.dictionary())  # dictionary key
    region: int = blosc2.field(blosc2.int64())  # low-cardinality int key
    amount: float = blosc2.field(blosc2.float64())  # numeric value column


def _columns():
    i = np.arange(NROWS)
    return {
        "vendor": [VENDORS[j % len(VENDORS)] for j in range(NROWS)],  # plain str for dictionary
        "region": (i % 4).astype(np.int64),
        "amount": (i * 1.5).astype(np.float64),
    }


@pytest.fixture(scope="module")
def group_store(tmp_path_factory):
    """A TreeStore with one CTable carrying a dictionary + int + float column."""
    path = str(tmp_path_factory.mktemp("group") / "group.b2z")
    cols = _columns()
    tstore = blosc2.TreeStore(path, mode="w")
    try:
        t = blosc2.CTable(_Row, expected_size=NROWS, validate=False)
        t.extend(cols, validate=False)
        tstore["/ctable"] = t
    finally:
        tstore.close()
    return path, cols


# ── Model layer ────────────────────────────────────────────────────────────

from blosc2.b2view.model import StoreBrowser  # noqa: E402


def test_group_key_columns_are_dict_and_int(group_store):
    path, _ = group_store
    with StoreBrowser(path) as browser:
        keys = browser.group_key_columns("/ctable")
    assert set(keys) == {"vendor", "region"}  # float 'amount' excluded


def test_group_value_columns_are_numeric_scalars(group_store):
    path, _ = group_store
    with StoreBrowser(path) as browser:
        values = browser.group_value_columns("/ctable")
    assert "amount" in values
    assert "region" in values
    assert "vendor" not in values  # dictionary excluded


def test_size_counts_rows_per_group(group_store):
    path, cols = group_store
    with StoreBrowser(path) as browser:
        ngroups = browser.set_group("/ctable", "vendor", "size", None)
        assert ngroups == len(VENDORS)
        page = browser.preview("/ctable", max_rows=100)
    assert "size" in page["columns"]
    assert int(np.asarray(page["data"]["size"]).sum()) == NROWS


def test_mean_matches_numpy(group_store):
    path, cols = group_store
    with StoreBrowser(path) as browser:
        browser.set_group("/ctable", "region", "mean", "amount")
        page = browser.preview("/ctable", max_rows=100)
    got = dict(
        zip(np.asarray(page["data"]["region"]), np.asarray(page["data"]["amount_mean"]), strict=False)
    )
    region, amount = cols["region"], cols["amount"]
    for r in np.unique(region):
        assert got[r] == pytest.approx(amount[region == r].mean())


def test_get_and_clear_group_roundtrip(group_store):
    path, _ = group_store
    with StoreBrowser(path) as browser:
        browser.set_group("/ctable", "vendor", "size", None)
        assert browser.get_group("/ctable") == ("vendor", "size", None)
        browser.clear_group("/ctable")
        assert browser.get_group("/ctable") is None
        page = browser.preview("/ctable", max_rows=5)
    assert page["nrows"] == NROWS  # base table restored


def test_group_clears_active_filter(group_store):
    path, _ = group_store
    with StoreBrowser(path) as browser:
        browser.set_filter("/ctable", "region == 0")
        browser.set_group("/ctable", "vendor", "size", None)
        assert browser.get_filter("/ctable") is None  # filter dropped (exclusive)


def test_group_sort_orders_result_by_column(group_store):
    path, _ = group_store
    with StoreBrowser(path) as browser:
        browser.set_group("/ctable", "region", "mean", "amount")
        browser.set_group_sort("/ctable", "amount_mean", reverse=True)
        assert browser.get_group_sort("/ctable") == ("amount_mean", True)
        page = browser.preview("/ctable", max_rows=100)
    vals = np.asarray(page["data"]["amount_mean"])
    assert list(vals) == sorted(vals, reverse=True)  # descending


def test_group_sort_cleared_by_regroup_and_clear(group_store):
    path, _ = group_store
    with StoreBrowser(path) as browser:
        browser.set_group("/ctable", "region", "mean", "amount")
        browser.set_group_sort("/ctable", "amount_mean", reverse=False)
        # Re-grouping drops the stale sort...
        browser.set_group("/ctable", "vendor", "size", None)
        assert browser.get_group_sort("/ctable") is None
        # ...as does clearing the group.
        browser.set_group_sort("/ctable", "size", reverse=True)
        browser.clear_group("/ctable")
        assert browser.get_group_sort("/ctable") is None


def test_group_sort_noop_when_not_grouped(group_store):
    path, _ = group_store
    with StoreBrowser(path) as browser:
        browser.set_group_sort("/ctable", "region", reverse=True)  # no group active
        assert browser.get_group_sort("/ctable") is None


def test_group_bars_sorted_desc_and_capped(group_store):
    path, _ = group_store
    with StoreBrowser(path) as browser:
        browser.set_group("/ctable", "region", "mean", "amount")
        bars = browser.group_bars("/ctable", top_n=2)
    assert bars["total"] == 4
    assert len(bars["values"]) == 2
    assert bars["values"][0] >= bars["values"][1]  # descending
    assert bars["agg"] == "amount_mean"
    assert bars["key"] == "region"


# ── TUI journey ────────────────────────────────────────────────────────────

pytest.importorskip("textual")
pytest.importorskip("pytest_asyncio")

if blosc2.IS_WASM:
    pytest.skip("Textual apps need a terminal driver (termios)", allow_module_level=True)

from textual.widgets import DataTable  # noqa: E402

from blosc2.b2view.app import B2ViewApp, GroupByScreen, SortByScreen  # noqa: E402

TERM_SIZE = (120, 40)


async def _wait_table(pilot):
    for _ in range(50):
        await pilot.pause()
        page = getattr(pilot.app, "table_page", None)
        if page and page.get("source_kind") == "ctable":
            return page
    raise AssertionError("data grid never loaded")


@pytest.mark.asyncio
@pytest.mark.tui
async def test_group_key_applies_and_escape_clears(group_store):
    path, _ = group_store
    app = B2ViewApp(path)
    async with app.run_test(size=TERM_SIZE) as pilot:
        await pilot.press("down", "enter")  # select + show the /ctable node
        await _wait_table(pilot)
        pilot.app.query_one("#data-table", DataTable).focus()
        await pilot.pause()

        await pilot.press("G")
        await pilot.pause()
        assert isinstance(pilot.app.screen, GroupByScreen)
        await pilot.press("enter")  # key list -> aggregation list
        await pilot.press("enter")  # apply first aggregation (count rows / size)
        await _wait_table(pilot)

        assert pilot.app.browser.get_group("/ctable") is not None
        table = pilot.app.query_one("#data-table", DataTable)
        cols = [str(c.label) for c in table.columns.values()]
        assert any("size" in c for c in cols)

        # Sort the grouped result by one of its columns via 'S'.
        await pilot.press("S")
        await pilot.pause()
        assert isinstance(pilot.app.screen, SortByScreen)
        await pilot.press("enter")  # apply the highlighted column
        await _wait_table(pilot)
        assert pilot.app.browser.get_group_sort("/ctable") is not None

        await pilot.press("escape")  # clear the group sort, keep the group
        await _wait_table(pilot)
        assert pilot.app.browser.get_group_sort("/ctable") is None
        assert pilot.app.browser.get_group("/ctable") is not None

        await pilot.press("escape")  # ungroup
        await _wait_table(pilot)
        assert pilot.app.browser.get_group("/ctable") is None
