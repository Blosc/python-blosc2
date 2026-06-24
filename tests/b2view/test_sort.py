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


@pytest.mark.parametrize("kind", ["SUMMARY", "FULL", "PARTIAL", "BUCKET", "OPSI"])
def test_indexed_column_plots_from_summary(tmp_path, kind):
    """Any index kind exposes block-level (min, max) summaries, so a numeric
    indexed column plots via method 'summary' — no data decompression."""

    @dataclasses.dataclass
    class Row:
        v: float = blosc2.field(blosc2.float64())

    rng = np.random.default_rng(0)
    n = 20000
    vals = rng.standard_normal(n)
    path = str(tmp_path / f"{kind}.b2z")
    t = blosc2.CTable(Row, urlpath=path, mode="w", expected_size=n)
    t.extend([(float(x),) for x in vals])
    t.create_index("v", kind=getattr(blosc2.IndexKind, kind))
    t.close()

    with StoreBrowser(path) as browser:
        env = browser.plot_series("/", column="v", max_points=64)
        assert env["method"] == "summary"
        # The block-summary envelope bounds the true data range exactly.
        assert np.nanmin(env["ymin"]) == pytest.approx(vals.min())
        assert np.nanmax(env["ymax"]) == pytest.approx(vals.max())


def test_sort_non_indexed_column(tmp_path):
    """A column with no FULL index still sorts (materialise key + lexsort)."""

    @dataclasses.dataclass
    class Row:
        v: int = blosc2.field(blosc2.int64())

    path = str(tmp_path / "noidx.b2z")
    rng = np.random.default_rng(0)
    vals = rng.integers(0, 1000, N).astype(np.int64)
    t = blosc2.CTable(Row, urlpath=path, mode="w", expected_size=N)
    t.extend([(int(x),) for x in vals])
    t.close()  # no create_index

    with StoreBrowser(path) as browser:
        assert browser.full_index_columns("/") == []  # nothing indexed
        browser.set_sort("/", "v", reverse=False)
        assert _head(browser, "v", 5) == sorted(vals.tolist())[:5]


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
        assert browser.get_sort("/") is None  # re-filtering drops the old sort


def test_sort_composes_over_active_filter(sort_store):
    """Sort applied after a filter orders only the filtered rows, keeping both."""
    path, bvals, _ = sort_store
    expected = sorted(int(v) for v in bvals if v > 500)
    with StoreBrowser(path) as browser:
        browser.set_filter("/", "b > 500")
        browser.set_sort("/", "b", reverse=False)
        assert browser.get_filter("/") == "b > 500"  # filter persists under the sort
        assert browser.get_sort("/") == ("b", False)
        assert _head(browser, "b", 5) == expected[:5]  # sorted, and all > 500


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
        # Cursor parks on the sorted column, first row.
        table = app.query_one("#data-table")
        cur_row, cur_col = table.cursor_coordinate
        assert cur_row == 0
        assert app.table_page["columns"][cur_col] == col

        # Escape clears the sort, restoring original order.
        await pilot.press("escape")
        await pilot.pause()
        assert app.browser.get_sort("/") is None


@pytest.fixture(scope="module")
def wide_store(tmp_path_factory):
    """A CTable wider than the viewport, FULL index only on the LAST column."""
    path = str(tmp_path_factory.mktemp("wide") / "wide.b2z")
    ncols = 25
    cols = [f"c{i:02d}" for i in range(ncols)]
    Row = dataclasses.make_dataclass(
        "WideRow",
        [(name, int, blosc2.field(blosc2.int64())) for name in cols],
    )
    rng = np.random.default_rng(1)
    data = rng.integers(0, 100000, size=(N, ncols)).astype(np.int64)
    t = blosc2.CTable(Row, urlpath=path, mode="w", expected_size=N)
    t.extend([tuple(int(v) for v in row) for row in data])
    t.create_index(cols[-1], kind=blosc2.IndexKind.FULL)  # only the last column
    t.close()
    return path, cols


@pytest.mark.asyncio
@pytest.mark.tui
async def test_sort_last_column_keeps_full_window(wide_store):
    path, cols = wide_store
    last = cols[-1]
    app = B2ViewApp(path, start_panel="data")
    async with app.run_test(size=TERM_SIZE) as pilot:
        await _wait_for_table(pilot)
        app.query_one("#data-table").focus()
        await pilot.pause()

        await pilot.press("S")
        await pilot.pause()
        # Every column is now offered; jump to the last one (the indexed one).
        app.screen.query_one("#sortby-list").highlighted = len(cols) - 1
        await pilot.press("enter")
        await pilot.pause()
        assert app.browser.get_sort("/") == (last, False)

        page = app.table_page
        # The tail window holds several columns ending at the last one — not a
        # lone column — and they keep their natural left-to-right order.
        assert page["col_stop"] == page["ncols"]
        assert len(page["columns"]) > 1
        assert page["columns"] == cols[page["col_start"] : page["col_stop"]]

        # Cursor sits on the sorted (last) column, first row.
        table = app.query_one("#data-table")
        cur_row, cur_col = table.cursor_coordinate
        assert cur_row == 0
        assert page["columns"][cur_col] == last

        # 'R' reverses in place: the horizontal window (same columns, same start)
        # stays put, the cursor stays on the sorted column, and the order flips.
        cols_before = list(page["columns"])
        col_start_before = page["col_start"]
        await pilot.press("R")
        await pilot.pause()
        page = app.table_page
        assert app.browser.get_sort("/") == (last, True)
        assert page["col_start"] == col_start_before  # window did not re-scroll
        assert list(page["columns"]) == cols_before  # same columns, none dropped
        cur_row, cur_col = app.query_one("#data-table").cursor_coordinate
        assert page["columns"][cur_col] == last
        assert page["data"][last][0] == max(page["data"][last])


@pytest.mark.asyncio
@pytest.mark.tui
async def test_sort_non_indexed_column_async(wide_store):
    """Sorting a non-indexed column runs in the background and still applies."""
    path, cols = wide_store
    app = B2ViewApp(path, start_panel="data")
    async with app.run_test(size=TERM_SIZE) as pilot:
        await _wait_for_table(pilot)
        app.query_one("#data-table").focus()
        await pilot.pause()

        await pilot.press("S")
        await pilot.pause()
        app.screen.query_one("#sortby-list").highlighted = 0  # c00 — not indexed
        await pilot.press("enter")
        # Wait for the background sort worker to finish and repaint.
        for _ in range(100):
            await pilot.pause()
            if app.browser.get_sort("/") == (cols[0], False):
                break
        assert app.browser.get_sort("/") == (cols[0], False)
        col = cols[0]
        assert app.table_page["data"][col][0] == min(app.table_page["data"][col])


@pytest.mark.asyncio
@pytest.mark.tui
async def test_columns_stable_across_vertical_scroll(wide_store):
    """Scrolling down (across buffer reloads) keeps the same visible columns —
    the width re-fit is sticky, so a wider lower row block can't drop a column."""
    path, _ = wide_store
    app = B2ViewApp(path, start_panel="data")
    async with app.run_test(size=TERM_SIZE) as pilot:
        await _wait_for_table(pilot)
        app.query_one("#data-table").focus()
        await pilot.pause()

        key_top = app._col_fit_key()
        cols_top = list(app.table_page["columns"])

        await pilot.press("b")  # jump to the last row → buffer reloads far down
        await pilot.pause()
        assert app._col_fit_key() == key_top  # vertical move does not change the fit key
        assert list(app.table_page["columns"]) == cols_top  # columns unchanged


def test_take_n_columns_pins_count():
    """_take_n_columns keeps exactly the first n columns (clamped to range)."""
    cols = [f"c{i}" for i in range(5)]
    data = {
        "source_kind": "ctable",
        "nrows": 3,
        "ncols": 5,
        "col_start": 0,
        "hidden_columns": 0,
        "columns": cols,
        "data": {name: ["x"] * 3 for name in cols},
    }
    app = B2ViewApp.__new__(B2ViewApp)  # no event loop needed for this pure helper
    three = app._take_n_columns({**data, "data": dict(data["data"])}, 3)
    assert three["columns"] == cols[:3]
    assert three["col_stop"] == 3
    assert three["hidden_columns"] == 2
    allcols = app._take_n_columns({**data, "data": dict(data["data"])}, 99)
    assert allcols["columns"] == cols  # clamped to available


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
