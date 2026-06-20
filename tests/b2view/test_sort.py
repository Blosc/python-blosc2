#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""b2view sort-by support: the StoreBrowser model layer and the 'S' key flow.

Model tests drive ``StoreBrowser`` directly; the TUI tests drive the real
Textual app through a headless ``Pilot``, pressing the same keys a user would.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

pytest.importorskip("textual")
pytest.importorskip("pytest_asyncio")

import blosc2

if blosc2.IS_WASM:
    pytest.skip("Textual apps need a terminal driver (termios)", allow_module_level=True)

from blosc2.b2view.app import B2ViewApp, SortByScreen
from blosc2.b2view.model import StoreBrowser

N = 200
TERM_SIZE = (120, 40)


@pytest.fixture(scope="module")
def sort_store(tmp_path_factory):
    """Standalone CTable with FULL indexes on a numeric and a dictionary column."""
    path = str(tmp_path_factory.mktemp("sort") / "sort.b2z")

    @dataclasses.dataclass
    class Row:
        b: int = blosc2.field(blosc2.int64())
        label: str = blosc2.field(blosc2.dictionary())

    rng = np.random.default_rng(0)
    bvals = rng.integers(0, 1000, N).astype(np.int64)
    pool = ["delta", "alpha", "charlie", "bravo"]
    labels = [pool[i] for i in rng.integers(0, len(pool), N)]

    t = blosc2.CTable(Row, urlpath=path, mode="w", expected_size=N)
    t.extend(list(zip(bvals.tolist(), labels, strict=True)))
    t.create_index("b", kind=blosc2.IndexKind.FULL)
    t.create_index("label", kind=blosc2.IndexKind.FULL)
    t.close()
    return path, bvals, labels


# ── Model layer (StoreBrowser) ────────────────────────────────────────────


def _head(browser, column, k):
    return [browser.read_cell("/", column, i) for i in range(k)]


def test_full_index_columns(sort_store):
    path, _, _ = sort_store
    with StoreBrowser(path) as browser:
        assert set(browser.full_index_columns("/")) == {"b", "label"}


def test_sort_numeric_ascending_and_reverse(sort_store):
    path, bvals, _ = sort_store
    expected = sorted(bvals.tolist())
    with StoreBrowser(path) as browser:
        browser.set_sort("/", "b", reverse=False)
        assert browser.get_sort("/") == ("b", False)
        assert _head(browser, "b", 5) == expected[:5]

        browser.set_sort("/", "b", reverse=True)
        assert _head(browser, "b", 5) == expected[::-1][:5]


def test_sort_dictionary_by_decoded_string(sort_store):
    path, _, labels = sort_store
    with StoreBrowser(path) as browser:
        browser.set_sort("/", "label", reverse=False)
        assert _head(browser, "label", 5) == sorted(labels)[:5]


def test_clear_sort_restores_original_order(sort_store):
    path, bvals, _ = sort_store
    with StoreBrowser(path) as browser:
        browser.set_sort("/", "b", reverse=False)
        browser.clear_sort("/")
        assert browser.get_sort("/") is None
        assert _head(browser, "b", 3) == bvals[:3].tolist()  # original row order


def test_window_composes_over_sort(sort_store):
    path, bvals, _ = sort_store
    expected = sorted(bvals.tolist())
    with StoreBrowser(path) as browser:
        browser.set_sort("/", "b", reverse=False)
        assert browser.set_row_window("/", 0, 5) == 5  # locked to first 5 sorted rows
        assert _head(browser, "b", 5) == expected[:5]


def test_filter_clears_sort(sort_store):
    path, _, _ = sort_store
    with StoreBrowser(path) as browser:
        browser.set_sort("/", "b", reverse=False)
        browser.set_filter("/", "b > 500")
        assert browser.get_sort("/") is None  # mutually exclusive


# ── End-to-end TUI flow (Pilot) ───────────────────────────────────────────


async def _wait_for_table(pilot) -> None:
    for _ in range(100):
        await pilot.pause()
        if pilot.app.table_page is not None and not pilot.app.loading_table_page:
            return
    raise AssertionError("data table never loaded")


@pytest.mark.asyncio
@pytest.mark.tui
async def test_sort_key_opens_screen_applies_and_escape_clears(sort_store):
    path, _, _ = sort_store
    app = B2ViewApp(path, start_panel="data")
    async with app.run_test(size=TERM_SIZE) as pilot:
        await _wait_for_table(pilot)
        app.query_one("#data-table").focus()
        await pilot.pause()

        # 'S' opens the sort dropdown listing the FULL-indexed columns.
        await pilot.press("S")
        await pilot.pause()
        assert isinstance(app.screen, SortByScreen)

        # Enter applies the highlighted column ascending; grid reorders.
        await pilot.press("enter")
        await pilot.pause()
        assert app.browser.get_sort("/") in {("b", False), ("label", False)}
        col = app.browser.get_sort("/")[0]
        assert app.table_page["data"][col][0] == min(app.table_page["data"][col])

        # Escape clears the sort, restoring original order.
        await pilot.press("escape")
        await pilot.pause()
        assert app.browser.get_sort("/") is None


@pytest.mark.asyncio
@pytest.mark.tui
async def test_reverse_key_flips_active_sort(sort_store):
    path, _, _ = sort_store
    app = B2ViewApp(path, start_panel="data")
    async with app.run_test(size=TERM_SIZE) as pilot:
        await _wait_for_table(pilot)
        app.query_one("#data-table").focus()
        await pilot.pause()

        await pilot.press("S")
        await pilot.pause()
        await pilot.press("enter")  # ascending
        await pilot.pause()
        col, reverse = app.browser.get_sort("/")
        assert reverse is False

        await pilot.press("R")  # flip to descending in place
        await pilot.pause()
        assert app.browser.get_sort("/") == (col, True)
        assert app.table_page["data"][col][0] == max(app.table_page["data"][col])


@pytest.mark.asyncio
@pytest.mark.tui
async def test_sort_reverse_toggle_in_dropdown(sort_store):
    path, _, _ = sort_store
    app = B2ViewApp(path, start_panel="data")
    async with app.run_test(size=TERM_SIZE) as pilot:
        await _wait_for_table(pilot)
        app.query_one("#data-table").focus()
        await pilot.pause()

        await pilot.press("S")
        await pilot.pause()
        await pilot.press("R")  # toggle reverse (descending) before applying
        await pilot.press("enter")
        await pilot.pause()
        col, reverse = app.browser.get_sort("/")
        assert reverse is True
        assert app.table_page["data"][col][0] == max(app.table_page["data"][col])
