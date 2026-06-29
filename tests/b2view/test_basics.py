#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""Basic keyboard-navigation tests for the b2view TUI.

The store is generated with ``tree_store_gen.py`` (small parameters, a few
hundred KB) so every cell value is predictable: NDArray leaves come from
``blosc2.linspace(0, 1, ...)`` and the per-level CTable columns follow the
formulas documented in ``ctable_values()`` of that module.

All tests drive the real Textual app through a ``Pilot`` (headless terminal
of a fixed size), pressing the same keys a user would, and then assert on
the loaded page (``app.table_page``) and the underlying values.  The focus
is navigation of objects *larger than the data panel viewport*: row paging,
column paging, jump-to-row, and dim-mode for N-D arrays.

NOTE for test authors (humans and LLMs alike): booting an app session
(``app.run_test()``) costs ~0.3 s and every key press ~0.1 s, dwarfing the
assertions themselves.  When adding tests, do NOT create a new session per
scenario: extend an existing test that already starts at the right node, or
group related scenarios that share a start state into one self-contained
keyboard journey (see ``test_ctable_column_paging`` for the pattern).  Only
start a new session when the scenarios genuinely need independent app state.
Deselect the whole TUI suite with ``pytest -m "not tui"``.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

pytest.importorskip("textual")
pytest.importorskip("pytest_asyncio")

import blosc2

if blosc2.IS_WASM:
    # Instantiating a Textual app selects a terminal driver, and the Linux
    # driver needs termios, which Emscripten does not provide.
    pytest.skip("Textual apps need a terminal driver (termios)", allow_module_level=True)

import tree_store_gen as gen
from textual.widgets import DataTable, Input, SelectionList, Tree

from blosc2.b2view.app import (
    B2ViewApp,
    ColumnFilterScreen,
    ColumnSelectScreen,
    FilterScreen,
    GoToColumnScreen,
    GoToRowScreen,
    HelpScreen,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.tui]

# ── Store generation (via tree_store_gen.py, next to this module) ────────

NLEVELS = 2
NLEAVES = 4  # leaf0: scalar, leaf1: 1-D, leaf2: 2-D, leaf3: 3-D
MAX_ELEMS = 10_000
NROWS = 300  # CTable rows; well beyond one viewport page

# Shapes produced by leaf_shape() for the parameters above
LEAF1_LEN = 10_000
LEAF2_SHAPE = (100, 100)
LEAF3_SHAPE = (21, 21, 21)

# Fixed terminal size for deterministic viewports
TERM_SIZE = (120, 40)


@pytest.fixture(scope="session")
def store_path(tmp_path_factory) -> str:
    path = tmp_path_factory.mktemp("b2view") / "tree-store.b2z"
    gen.create_store(NLEVELS, NLEAVES, MAX_ELEMS, NROWS, output=str(path))
    return str(path)


# ── Helpers ──────────────────────────────────────────────────────────────


async def wait_for_table(pilot) -> None:
    """Wait until the data grid has a loaded, settled page."""
    for _ in range(100):
        await pilot.pause()
        app = pilot.app
        if app.table_page is not None and not app.loading_table_page:
            return
    raise AssertionError("data table never finished loading")


async def wait_until(pilot, predicate, *, message="condition not met in time") -> None:
    """Pump the event loop until *predicate* holds.

    Setting ``Input.value`` posts an ``Input.Changed`` that rebuilds dependent widgets
    asynchronously; a single ``pilot.pause()`` is not always enough on slower/loaded CI
    (e.g. Windows), so poll until the resulting state settles.
    """
    for _ in range(100):
        await pilot.pause()
        if predicate():
            return
    raise AssertionError(message)


async def focus_data_table(pilot) -> DataTable:
    table = pilot.app.query_one("#data-table", DataTable)
    table.focus()
    await pilot.pause()
    return table


def leaf1_values() -> np.ndarray:
    return np.linspace(0, 1, num=LEAF1_LEN)


def leaf2_values() -> np.ndarray:
    return np.linspace(0, 1, num=int(np.prod(LEAF2_SHAPE))).reshape(LEAF2_SHAPE)


def leaf3_values() -> np.ndarray:
    return np.linspace(0, 1, num=int(np.prod(LEAF3_SHAPE))).reshape(LEAF3_SHAPE)


def _assert_ctable_window_values(page, expected):
    """Check every visible cell of *page* against the generator columns."""
    for name in page["columns"]:
        got = page["data"][name]
        want = expected[name][page["start"] : page["stop"]]
        if np.issubdtype(np.asarray(want).dtype, np.number):
            np.testing.assert_allclose(np.asarray(got, dtype=float), want)
        else:
            np.testing.assert_array_equal(got, want)


# ── Tree and panel focus navigation ──────────────────────────────────────


async def _wait_focus(pilot, expected_id: str) -> str | None:
    """Pause until the focused widget is *expected_id* (or give up)."""
    for _ in range(30):
        await pilot.pause()
        if getattr(pilot.app.focused, "id", None) == expected_id:
            break
    return getattr(pilot.app.focused, "id", None)


async def test_start_panel_focus_with_path(store_path):
    """``--panel`` focuses the right widget on startup, even with a ``--path``.

    Regression: the data panel was left unfocused when both a starting path
    and ``--panel data`` were given (a timer raced the node selection, which
    pulled focus back to the tree).
    """
    # The bug case: data panel on a leaf must focus the data grid itself.
    app = B2ViewApp(store_path, start_path="/level0/leaf1", start_panel="data")
    async with app.run_test(size=TERM_SIZE) as pilot:
        await wait_for_table(pilot)
        assert await _wait_focus(pilot, "data-table") == "data-table"

    # Other panels still land where asked.
    for panel, expected in [("meta", "meta-scroll"), ("tree", "tree")]:
        app = B2ViewApp(store_path, start_path="/level0/leaf1", start_panel=panel)
        async with app.run_test(size=TERM_SIZE) as pilot:
            await wait_for_table(pilot)
            assert await _wait_focus(pilot, expected) == expected


async def test_tree_and_panel_focus(store_path):
    """Tab cycles the panels; Down/Enter in the tree selects nodes."""
    app = B2ViewApp(store_path)
    async with app.run_test(size=TERM_SIZE) as pilot:
        await pilot.pause()
        assert isinstance(app.focused, Tree)

        # Tab: tree -> meta -> vlmeta -> data and wraps back to the tree
        for expected in ["meta-scroll", "vlmeta-scroll", "data-scroll", "tree"]:
            await pilot.press("tab")
            assert await _wait_focus(pilot, expected) == expected

        await pilot.press("down", "enter")  # root -> level0, select + expand
        await pilot.pause()
        assert app.selected_path == "/level0"

        first_child = app.browser.list_children("/level0")[0]
        await pilot.press("down", "enter")  # -> first child of level0
        await wait_for_table(pilot)
        assert app.selected_path == first_child.path

        # '?' opens the help screen; escape closes it
        await pilot.press("question_mark")
        await pilot.pause()
        assert isinstance(app.screen, HelpScreen)
        await pilot.press("escape")
        await pilot.pause()
        assert not isinstance(app.screen, HelpScreen)


# ── 1-D array: row paging beyond the viewport ────────────────────────────


async def test_1d_row_paging_and_jumps(store_path):
    """Cursor-down at the last row pages forward; 'b'/'t' jump to bottom/top."""
    app = B2ViewApp(store_path, start_path="/level0/leaf1", start_panel="data")
    async with app.run_test(size=TERM_SIZE) as pilot:
        await wait_for_table(pilot)
        table = await focus_data_table(pilot)

        page = app.table_page
        assert page["nrows"] == LEAF1_LEN
        assert page["start"] == 0
        first_stop = page["stop"]
        assert first_stop < LEAF1_LEN  # viewport smaller than the array

        expected = leaf1_values()
        np.testing.assert_allclose(page["data"]["value"], expected[: page["stop"]])

        # Move the cursor to the last row of the page and step once more
        table.move_cursor(row=page["stop"] - page["start"] - 1)
        await pilot.press("down")
        await wait_for_table(pilot)

        page = app.table_page
        assert page["start"] == first_stop  # new page starts where the old ended
        assert table.cursor_row == 0
        np.testing.assert_allclose(page["data"]["value"], expected[page["start"] : page["stop"]])

        # 'b' jumps to the very last row of the array
        await pilot.press("b")
        await wait_for_table(pilot)
        page = app.table_page
        assert page["stop"] == LEAF1_LEN
        assert page["start"] + table.cursor_row == LEAF1_LEN - 1
        np.testing.assert_allclose(page["data"]["value"], expected[page["start"] : page["stop"]])

        # ...and 't' back to the first
        await pilot.press("t")
        await wait_for_table(pilot)
        page = app.table_page
        assert page["start"] == 0
        assert page["start"] + table.cursor_row == 0


# ── 2-D array: row *and* column paging beyond the viewport ───────────────


async def test_2d_paging(store_path):
    """Column paging shows the right values; row paging stops at the bottom."""
    app = B2ViewApp(store_path, start_path="/level0/leaf2", start_panel="data")
    async with app.run_test(size=TERM_SIZE) as pilot:
        await wait_for_table(pilot)
        table = await focus_data_table(pilot)

        page = app.table_page
        assert page["ncols"] == LEAF2_SHAPE[1]
        assert page["col_start"] == 0
        first_col_stop = page["col_stop"]
        assert first_col_stop < LEAF2_SHAPE[1]  # more columns than the viewport

        expected = leaf2_values()
        # Column labels are the global column indices
        assert page["columns"] == [str(c) for c in range(page["col_start"], page["col_stop"])]
        for c in range(page["col_start"], page["col_stop"]):
            np.testing.assert_allclose(page["data"][str(c)], expected[page["start"] : page["stop"], c])

        # Move the cursor to the last visible column and step right once more
        table.move_cursor(column=len(page["columns"]) - 1)
        await pilot.press("right")
        await wait_for_table(pilot)

        page = app.table_page
        assert page["col_start"] == first_col_stop
        assert table.cursor_column == 0
        for c in range(page["col_start"], page["col_stop"]):
            np.testing.assert_allclose(page["data"][str(c)], expected[page["start"] : page["stop"], c])

        # Row paging stops at the bottom: 'b', then one more down is a no-op
        await pilot.press("b")
        await wait_for_table(pilot)
        page = app.table_page
        assert page["stop"] == LEAF2_SHAPE[0]
        last_cursor = table.cursor_row

        await pilot.press("down")  # already at the last row: must not page/move
        await wait_for_table(pilot)
        assert app.table_page["stop"] == LEAF2_SHAPE[0]
        assert table.cursor_row == last_cursor

        # 'end' jumps to the widest whole-column window ending at the last
        # column; paging left from there must not skip any column.
        await pilot.press("end")
        await wait_for_table(pilot)
        page = app.table_page
        assert page["col_stop"] == LEAF2_SHAPE[1]
        end_col_start = page["col_start"]
        assert end_col_start > 0

        table.move_cursor(column=0)
        await pilot.press("left")
        await wait_for_table(pilot)
        page = app.table_page
        assert page["col_start"] < end_col_start
        assert page["col_stop"] >= end_col_start  # no column skipped
        for c in range(page["col_start"], page["col_stop"]):
            np.testing.assert_allclose(page["data"][str(c)], expected[page["start"] : page["stop"], c])

        # 'home' returns to the first column window
        await pilot.press("home")
        await wait_for_table(pilot)
        assert app.table_page["col_start"] == 0

        # 'c' jumps to a column by index (arrays have no column names).  The
        # modal pre-fills the current index and pre-selects it, so typing a new
        # index replaces it (not appends, e.g. "0" + "97" -> "097").
        await pilot.press("c")
        await pilot.pause()
        assert isinstance(app.screen, GoToColumnScreen)
        gotocol_input = app.screen.query_one("#gotocol-input", Input)
        assert gotocol_input.value == "0"  # pre-filled with current column
        for ch in "97":
            await pilot.press(ch)
        assert gotocol_input.value == "97"  # replaced, not "097"
        await pilot.press("enter")
        await wait_for_table(pilot)
        page = app.table_page
        assert page["col_start"] == 97
        assert page["col_stop"] == LEAF2_SHAPE[1]
        np.testing.assert_allclose(page["data"]["97"], expected[page["start"] : page["stop"], 97])

        # Row paging re-aligns after a dim-mode single-row scroll.  Back to the
        # top so the row window starts on a page_size boundary.
        await pilot.press("t")
        await wait_for_table(pilot)
        assert app.table_page["start"] == 0
        page_size = app._table_page_size()
        assert page_size < LEAF2_SHAPE[0]  # several row pages exist

        # In dim mode the active (row) dim scrolls by one row, nudging the
        # window off the page grid.
        await pilot.press("d")
        assert app._dim_mode
        await pilot.press("up")
        await wait_for_table(pilot)
        assert app.table_page["start"] == 1  # off-grid by one row
        await pilot.press("escape")
        assert not app._dim_mode

        # An explicit page down now snaps back onto the page grid instead of
        # carrying the one-row offset (the bug), and page up returns to 0.
        await pilot.press("pagedown")
        await wait_for_table(pilot)
        assert app.table_page["start"] == page_size

        await pilot.press("pageup")
        await wait_for_table(pilot)
        assert app.table_page["start"] == 0


# ── 3-D array: dim mode navigation ───────────────────────────────────────


async def test_3d_dim_mode_fixed_value(store_path):
    """In dim mode, up/down change the fixed index of the active dimension."""
    app = B2ViewApp(store_path, start_path="/level0/leaf3", start_panel="data")
    async with app.run_test(size=TERM_SIZE) as pilot:
        await wait_for_table(pilot)
        await focus_data_table(pilot)

        layout = app._data_layout
        assert layout is not None
        assert layout.fixed_values == {0: 0}
        assert layout.navigable_dims == [1, 2]

        await pilot.press("d")  # enter dim mode (active dim is d0, fixed)
        assert app._dim_mode

        await pilot.press("up")  # d0: 0 -> 1
        await wait_for_table(pilot)
        assert app._data_layout.fixed_values[0] == 1

        page = app.table_page
        expected = leaf3_values()[1]  # the d0=1 slice
        for c in range(page["col_start"], page["col_stop"]):
            np.testing.assert_allclose(page["data"][str(c)], expected[page["start"] : page["stop"], c])

        await pilot.press("escape")
        assert not app._dim_mode


# ── CTable: row paging, goto, and wide tables ────────────────────────────


async def test_ctable_row_paging_and_goto(store_path):
    """Row paging and the 'g'(oto) modal land on the expected CTable rows."""
    app = B2ViewApp(store_path, start_path="/level0/ctable", start_panel="data")
    async with app.run_test(size=TERM_SIZE) as pilot:
        await wait_for_table(pilot)
        table = await focus_data_table(pilot)

        page = app.table_page
        assert page["nrows"] == NROWS
        expected = gen.ctable_values(NROWS)
        np.testing.assert_array_equal(page["data"]["b"], expected["b"][: page["stop"]])

        # Row paging and jumps must keep the cursor on the current column
        cursor_col = page["columns"].index("c")
        table.move_cursor(column=cursor_col)

        await pilot.press("pagedown")
        await wait_for_table(pilot)
        assert app.table_page["start"] > 0
        assert table.cursor_column == cursor_col

        await pilot.press("pageup")
        await wait_for_table(pilot)
        assert app.table_page["start"] == 0
        assert table.cursor_column == cursor_col

        # 'b' jumps to the last row
        await pilot.press("b")
        await wait_for_table(pilot)
        page = app.table_page
        assert page["stop"] == NROWS
        assert page["start"] + table.cursor_row == NROWS - 1
        assert table.cursor_column == cursor_col

        # 'g' opens the goto modal; submit a row in the middle
        await pilot.press("g")
        await pilot.pause()
        assert isinstance(app.screen, GoToRowScreen)
        app.screen.query_one("#goto-input", Input).value = "250"
        await pilot.press("enter")
        await wait_for_table(pilot)

        page = app.table_page
        assert page["start"] <= 250 < page["stop"]
        assert page["start"] + table.cursor_row == 250
        assert table.cursor_column == cursor_col  # goto keeps the column too
        np.testing.assert_array_equal(page["data"]["b"], expected["b"][page["start"] : page["stop"]])


async def test_ctable_column_paging(store_path):
    """A 20-column CTable pages columns left/right without losing the row."""
    app = B2ViewApp(store_path, start_path="/level0/ctable", start_panel="data")
    async with app.run_test(size=TERM_SIZE) as pilot:
        await wait_for_table(pilot)
        table = await focus_data_table(pilot)

        all_names = list(gen.ctable_values(1).keys())
        expected = gen.ctable_values(NROWS)

        # The table does not fit: hidden columns and a horizontal scrollbar
        page = app.table_page
        first_columns = list(page["columns"])
        assert gen.NCOLS == 20
        assert page["col_start"] == 0
        assert page["ncols"] == gen.NCOLS
        assert len(first_columns) < gen.NCOLS
        assert page["hidden_columns"] == gen.NCOLS - len(first_columns)
        # The visible columns are the leading ones, in schema order
        assert first_columns == all_names[: len(first_columns)]
        assert app.query_one("#col-scrollbar").display
        # The two-pass fit must not overflow the table (no inner h-scroll)
        assert table.virtual_size.width <= table.size.width

        # Page right from the last visible column
        table.move_cursor(column=len(first_columns) - 1)
        await pilot.press("right")
        await wait_for_table(pilot)

        page = app.table_page
        assert page["col_start"] == len(first_columns)
        assert page["columns"] == all_names[page["col_start"] : page["col_stop"]]
        assert table.cursor_column == 0
        _assert_ctable_window_values(page, expected)

        # ...and page back left from the first visible column
        right_columns = list(page["columns"])
        table.move_cursor(column=0)
        await pilot.press("left")
        await wait_for_table(pilot)

        page = app.table_page
        assert page["col_start"] == 0
        assert page["columns"] == first_columns
        assert table.cursor_column == len(right_columns) - 1
        _assert_ctable_window_values(page, expected)

        # 'e' jumps to the widest whole-column window ending at the last
        # column, and paging left from there must not skip any column.
        # ('s'/'e' are aliases of Home/End, which the 2-D test covers.)
        await pilot.press("e")
        await wait_for_table(pilot)
        page = app.table_page
        assert page["col_stop"] == gen.NCOLS
        end_col_start = page["col_start"]
        assert end_col_start > 0
        assert table.cursor_column == len(page["columns"]) - 1

        table.move_cursor(column=0)
        await pilot.press("left")
        await wait_for_table(pilot)
        page = app.table_page
        assert page["col_start"] < end_col_start
        assert page["col_stop"] >= end_col_start  # no column skipped
        assert page["columns"] == all_names[page["col_start"] : page["col_stop"]]
        _assert_ctable_window_values(page, expected)

        # 's' returns to the first window
        await pilot.press("s")
        await wait_for_table(pilot)
        assert app.table_page["col_start"] == 0

        # Column paging must not lose the current row: goto 150, page right
        await pilot.press("g")
        await pilot.pause()
        app.screen.query_one("#goto-input", Input).value = "150"
        await pilot.press("enter")
        await wait_for_table(pilot)
        page = app.table_page
        assert page["start"] + table.cursor_row == 150

        table.move_cursor(column=len(page["columns"]) - 1)
        await pilot.press("right")
        await wait_for_table(pilot)

        page = app.table_page
        assert page["col_start"] > 0
        assert page["start"] + table.cursor_row == 150
        _assert_ctable_window_values(page, expected)

        # 'c' opens a searchable column picker (type to filter, ↑/↓, Enter);
        # the row position is kept.  Pick v12 by typing its name.
        await pilot.press("c")
        await pilot.pause()
        assert isinstance(app.screen, ColumnSelectScreen)
        for ch in "v12":
            await pilot.press(ch)
        await pilot.pause()
        await pilot.press("enter")
        await wait_for_table(pilot)
        page = app.table_page
        assert page["col_start"] == all_names.index("v12")
        assert page["columns"][0] == "v12"
        assert table.cursor_column == 0
        assert page["start"] + table.cursor_row == 150
        _assert_ctable_window_values(page, expected)

        # ↑/↓ drive the highlight: filter to the v1x family, then arrow to v12
        await pilot.press("c")
        await pilot.pause()
        assert isinstance(app.screen, ColumnSelectScreen)
        for ch in "v1":  # matches v10, v11, ..., v19 in column order
            await pilot.press(ch)
        await pilot.pause()
        await pilot.press("down")  # v10 -> v11
        await pilot.press("down")  # v11 -> v12
        await pilot.press("enter")
        await wait_for_table(pilot)
        assert app.table_page["col_start"] == all_names.index("v12")
        assert app.table_page["columns"][0] == "v12"

        # escape cancels the picker without moving
        await pilot.press("c")
        await pilot.pause()
        assert isinstance(app.screen, ColumnSelectScreen)
        await pilot.press("escape")
        await wait_for_table(pilot)
        assert app.table_page["col_start"] == all_names.index("v12")

        # Shrinking the terminal re-fits the column window to the new width
        wide_columns = list(app.table_page["columns"])
        await pilot.resize_terminal(80, 40)
        for _ in range(100):
            await pilot.pause()
            if not app.loading_table_page and app.table_page.get("viewport_width") == table.size.width:
                break
        page = app.table_page
        assert page["viewport_width"] == table.size.width
        assert len(page["columns"]) < len(wide_columns)
        assert table.virtual_size.width <= table.size.width

        # 'p' on a non-numeric column must not open a plot (notify only).
        # This also passes when textual-plotext is not installed.
        await pilot.press("s")
        await wait_for_table(pilot)
        table.move_cursor(column=app.table_page["columns"].index("d"))
        await pilot.press("p")
        await pilot.pause()
        assert type(app.screen).__name__ != "PlotScreen"


# ── CTable filtering ─────────────────────────────────────────────────────


async def test_ctable_filtering(store_path):
    """The 'f' modal filters CTable rows; errors and clearing keep state sane."""
    app = B2ViewApp(store_path, start_path="/level0/ctable", start_panel="data")
    async with app.run_test(size=TERM_SIZE) as pilot:
        await wait_for_table(pilot)
        await focus_data_table(pilot)
        expected = gen.ctable_values(NROWS)

        async def submit_filter(expr: str) -> None:
            await pilot.press("f")
            await pilot.pause()
            assert isinstance(app.screen, FilterScreen)
            app.screen.query_one("#filter-input", Input).value = expr
            await pilot.press("enter")
            await wait_for_table(pilot)

        # Apply a filter: rows with b in [100, 110) (column b holds 0..NROWS-1)
        await submit_filter("b >= 100 and b < 110")
        page = app.table_page
        assert page["nrows"] == 10
        np.testing.assert_array_equal(page["data"]["b"], expected["b"][100:110])
        np.testing.assert_allclose(page["data"]["c"], expected["c"][100:110])

        # An invalid expression notifies and keeps the previous filter
        await submit_filter("nosuchcol > 1")
        assert app.browser.get_filter("/level0/ctable") == "b >= 100 and b < 110"
        assert app.table_page["nrows"] == 10

        # Re-opening the modal prefills the active filter; escape cancels
        await pilot.press("f")
        await pilot.pause()
        assert app.screen.query_one("#filter-input", Input).value == "b >= 100 and b < 110"
        await pilot.press("escape")
        await wait_for_table(pilot)
        assert app.table_page["nrows"] == 10

        # A filter matching nothing yields an empty (but live) grid
        await submit_filter("b < 0")
        assert app.table_page["nrows"] == 0

        # An empty input clears the filter and restores the full table
        await submit_filter("")
        page = app.table_page
        assert app.browser.get_filter("/level0/ctable") is None
        assert page["nrows"] == NROWS
        np.testing.assert_array_equal(page["data"]["b"], expected["b"][: page["stop"]])

        # Escape on the data grid also clears an active filter
        await submit_filter("b >= 100 and b < 110")
        assert app.table_page["nrows"] == 10
        await pilot.press("escape")
        await wait_for_table(pilot)
        assert app.browser.get_filter("/level0/ctable") is None
        assert app.table_page["nrows"] == NROWS

        # ── Column picking ('/' searchable multi-select) ─────────────────

        ncols = len(expected)  # a, b, c, d, v04..v19

        # '/' opens the picker with the currently-shown columns pre-checked.
        await pilot.press("slash")
        await pilot.pause()
        assert isinstance(app.screen, ColumnFilterScreen)
        sel = app.screen.query_one("#colfilter-list", SelectionList)
        assert sel.option_count == ncols
        assert len(sel.selected) == ncols  # all checked initially

        # Typing narrows the candidate list (substring, case-insensitive).
        app.screen.query_one("#colfilter-input", Input).value = "v1"
        await wait_until(pilot, lambda: sel.option_count == 10, message="list did not narrow")
        assert sel.option_count == 10  # v10..v19
        # Clear the filter again so the first column ('a') is reachable.
        app.screen.query_one("#colfilter-input", Input).value = ""
        await wait_until(pilot, lambda: sel.option_count == ncols, message="list did not reset")

        # ↓ moves focus into the list; Space unchecks the highlighted ('a');
        # Enter applies the remaining set.
        await pilot.press("down")
        await pilot.pause()
        assert sel.has_focus
        await pilot.press("space")
        await pilot.pause()
        await pilot.press("enter")
        await wait_for_table(pilot)
        page = app.table_page
        assert page["ncols"] == ncols - 1
        assert "a" not in page["columns"]
        assert page["columns"][0] == "b"
        assert app.browser.get_column_filter("/level0/ctable") == f"{ncols - 1} of {ncols}"

        # The goto-column picker lists names within the visible set.
        await pilot.press("c")
        await pilot.pause()
        assert isinstance(app.screen, ColumnSelectScreen)
        for ch in "v15":
            await pilot.press(ch)
        await pilot.pause()
        await pilot.press("enter")
        await wait_for_table(pilot)
        assert app.table_page["columns"][0] == "v15"
        await pilot.press("s")  # back to the first column window
        await wait_for_table(pilot)

        # Row and column filters combine.
        await submit_filter("b >= 100 and b < 110")
        page = app.table_page
        assert page["nrows"] == 10
        assert page["ncols"] == ncols - 1
        np.testing.assert_array_equal(page["data"]["b"], expected["b"][100:110])

        # Re-opening pre-checks the visible set; Escape cancels (no change).
        await pilot.press("slash")
        await pilot.pause()
        assert isinstance(app.screen, ColumnFilterScreen)
        assert len(app.screen.query_one("#colfilter-list", SelectionList).selected) == ncols - 1
        await pilot.press("escape")
        await wait_for_table(pilot)
        assert app.table_page["ncols"] == ncols - 1

        # Escape clears one layer at a time: row filter first, then columns.
        await pilot.press("escape")
        await wait_for_table(pilot)
        assert app.browser.get_filter("/level0/ctable") is None
        assert app.table_page["nrows"] == NROWS
        assert app.table_page["ncols"] == ncols - 1
        await pilot.press("escape")
        await wait_for_table(pilot)
        assert app.browser.get_column_filter("/level0/ctable") is None
        assert app.table_page["ncols"] == ncols


# ── Plotting ('p' key, optional textual-plotext) ─────────────────────────


async def test_plot_column(store_path):
    """'p' plots a min/max envelope of the whole 1-D leaf in a modal."""
    pytest.importorskip("textual_plotext")
    from blosc2.b2view.app import PlotScreen

    app = B2ViewApp(store_path, start_path="/level0/leaf1", start_panel="data")
    async with app.run_test(size=TERM_SIZE) as pilot:
        await wait_for_table(pilot)
        await focus_data_table(pilot)

        await pilot.press("p")
        await pilot.pause()
        assert isinstance(app.screen, PlotScreen)
        screen = app.screen

        # Bucketed envelope covering the whole leaf; bracketed by true extremes
        assert 0 < len(screen.x) <= app._PLOT_MAX_POINTS
        leaf = leaf1_values()
        assert min(screen.ymin) <= leaf.min() + 1e-9
        assert max(screen.ymax) >= leaf.max() - 1e-9
        assert "leaf1" in screen.plot_title
        assert "envelope" in screen.plot_title

        # ── Zoom / pan / reset / exact range from the plot modal ─────────
        from blosc2.b2view.app import PlotRangeScreen

        n = screen.n
        assert (screen.row_start, screen.row_stop) == (0, n)

        # '+' zooms in about the left edge: the window halves, start unchanged.
        await pilot.press("plus")
        await pilot.pause()
        assert screen.row_stop - screen.row_start == n // 2
        assert screen.row_start == 0
        assert "rows" in screen.plot_title

        # '-' zooms back out to the whole series.
        await pilot.press("minus")
        await pilot.pause()
        assert (screen.row_start, screen.row_stop) == (0, n)

        # Pan right shifts a zoomed window without changing its width.
        await pilot.press("plus")
        await pilot.pause()
        width = screen.row_stop - screen.row_start
        start_before = screen.row_start
        await pilot.press("right")
        await pilot.pause()
        assert screen.row_start > start_before
        assert screen.row_stop - screen.row_start == width

        # '0' resets to the whole series.
        await pilot.press("0")
        await pilot.pause()
        assert (screen.row_start, screen.row_stop) == (0, n)

        # 'g' opens a range modal; an exact range zooms there and reads it exactly.
        await pilot.press("g")
        await pilot.pause()
        assert isinstance(app.screen, PlotRangeScreen)
        app.screen.query_one("#range-input", Input).value = "1000:2000"
        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.screen, PlotScreen)
        screen = app.screen
        assert (screen.row_start, screen.row_stop) == (1000, 2000)
        sub = leaf1_values()[1000:2000]
        assert min(screen.ymin) <= sub.min() + 1e-9
        assert max(screen.ymax) >= sub.max() - 1e-9
        assert min(screen.x) >= 1000
        assert max(screen.x) < 2000

        # 'v' locks the 1-D array grid to the [1000:2000) window via the layout:
        # the grid sees 1000 rows, re-indexed from 0, and logical row 0 reads
        # absolute row 1000.  Paging cannot leave the window.
        await pilot.press("v")
        await wait_for_table(pilot)
        assert not isinstance(app.screen, PlotScreen)
        table = app.query_one("#data-table", DataTable)
        assert app.row_window == (1000, 2000)
        page = app.table_page
        assert page["nrows"] == 1000
        assert page["start"] == 0
        assert page["data"]["value"][0] == pytest.approx(leaf1_values()[1000])

        # 'b'(ottom) lands on the window's last row (absolute 1999), still inside.
        await pilot.press("b")
        await wait_for_table(pilot)
        page = app.table_page
        assert page["stop"] == 1000
        assert page["data"]["value"][table.cursor_row] == pytest.approx(leaf1_values()[1999])

        # 'esc' unlocks and restores the full array.
        await pilot.press("escape")
        await wait_for_table(pilot)
        assert app.row_window is None
        assert app._data_layout.row_window is None
        assert app.table_page["nrows"] == LEAF1_LEN

        # 'p' re-opens the plot; 'escape' is the only way to close it
        await pilot.press("p")
        await pilot.pause()
        assert isinstance(app.screen, PlotScreen)
        await pilot.press("escape")
        await pilot.pause()
        assert not isinstance(app.screen, PlotScreen)


async def test_plot_view_locks_ctable_window(store_path):
    """'v' on a CTable plot replaces the grid with a locked [start:stop] window."""
    pytest.importorskip("textual_plotext")
    from blosc2.b2view.app import PlotScreen

    app = B2ViewApp(store_path, start_path="/level0/ctable", start_panel="data")
    async with app.run_test(size=TERM_SIZE) as pilot:
        await wait_for_table(pilot)
        table = await focus_data_table(pilot)
        assert app.table_page["nrows"] == NROWS

        # Maximize the data panel before plotting.  Textual's default
        # escape-to-minimize would otherwise shadow our layered exit; with
        # ESCAPE_TO_MINIMIZE=False the escape below must unlock the locked
        # window (not restore the panel), and restore stays on the 'r' key.
        await pilot.press("m")
        await wait_for_table(pilot)
        assert app.screen.maximized is not None

        # Plot column 'b' (== row index), then zoom to an exact 100:110 range.
        table.move_cursor(column=app.table_page["columns"].index("b"))
        await pilot.press("p")
        await pilot.pause()
        assert isinstance(app.screen, PlotScreen)
        await pilot.press("g")
        await pilot.pause()
        app.screen.query_one("#range-input", Input).value = "100:110"
        await pilot.press("enter")
        await pilot.pause()
        assert (app.screen.row_start, app.screen.row_stop) == (100, 110)

        # 'v' locks the grid to that window: the modal closes, the grid shows
        # exactly those 10 rows (b == 100..109), re-indexed from 0.
        await pilot.press("v")
        await wait_for_table(pilot)
        assert not isinstance(app.screen, PlotScreen)
        assert app.row_window == (100, 110)
        assert app.screen.maximized is not None  # locking keeps the panel maximized
        page = app.table_page
        assert page["nrows"] == 10
        np.testing.assert_array_equal(page["data"]["b"], np.arange(100, 110))

        # Paging cannot leave the window: 'b'(ottom) lands on its last row (109).
        await pilot.press("b")
        await wait_for_table(pilot)
        page = app.table_page
        assert page["stop"] == 10
        assert page["data"]["b"][table.cursor_row] == 109

        # 'esc' unlocks the window even while maximized (the panel stays
        # maximized — escape did not get hijacked into a restore).
        await pilot.press("escape")
        await wait_for_table(pilot)
        assert app.row_window is None
        assert app.browser.get_row_window("/level0/ctable") is False
        assert app.table_page["nrows"] == NROWS
        assert app.screen.maximized is not None

        # 'r' is the way to restore a maximized panel.
        await pilot.press("r")
        await wait_for_table(pilot)
        assert app.screen.maximized is None


async def test_plot_hires_view(store_path):
    """'h' opens a hi-res envelope; 'r' toggles raw values; 'q' returns."""
    pytest.importorskip("textual_plotext")
    pytest.importorskip("textual_image")
    pytest.importorskip("matplotlib")
    from blosc2.b2view.app import HiResPlotScreen, PlotScreen, TextualImage

    app = B2ViewApp(store_path, start_path="/level0/ctable", start_panel="data")
    async with app.run_test(size=TERM_SIZE) as pilot:
        await wait_for_table(pilot)
        table = await focus_data_table(pilot)
        table.move_cursor(column=app.table_page["columns"].index("b"))

        await pilot.press("p")
        await pilot.pause()
        assert isinstance(app.screen, PlotScreen)
        plot = app.screen

        # 'h' opens the hi-res view in envelope mode over the whole range — no
        # zoom gate; it reuses the on-screen envelope, so it always renders.
        await pilot.press("h")
        await pilot.pause()
        assert isinstance(app.screen, HiResPlotScreen)
        hires = app.screen
        assert hires._mode == "envelope"
        assert "min/max envelope" in hires._current_title()
        assert hires.query_one("#hires-image", TextualImage) is not None

        # Force a tiny raw cap so the full 300-row range is strided-sampled,
        # then 'r' toggles to the raw view (no refusal — it samples instead).
        hires._raw_fetch = lambda s, e: app.browser.read_series(
            "/level0/ctable", column="b", row_start=s, row_stop=e, max_points=50
        )
        await pilot.press("r")
        await pilot.pause()
        assert hires._mode == "raw"
        title = hires._current_title()
        assert "raw values" in title
        assert "sampled" in title
        assert hires.query_one("#hires-image", TextualImage) is not None

        # 'r' again toggles back to the envelope.
        await pilot.press("r")
        await pilot.pause()
        assert hires._mode == "envelope"
        assert "min/max envelope" in hires._current_title()

        # 'escape' returns to the braille plot with the zoom intact.
        await pilot.press("escape")
        await pilot.pause()
        assert app.screen is plot
        assert (plot.row_start, plot.row_stop) == (0, plot.n)


async def test_plot_scatter_col_vs_col(store_path):
    """'s' on a CTable plot scatters the X column against a chosen Y column."""
    pytest.importorskip("textual_plotext")
    from blosc2.b2view.app import ColumnSelectScreen, PlotScreen, ScatterPlotScreen

    app = B2ViewApp(store_path, start_path="/level0/ctable", start_panel="data")
    async with app.run_test(size=TERM_SIZE) as pilot:
        await wait_for_table(pilot)
        table = await focus_data_table(pilot)
        # Plot column 'b' (== row index), then zoom to a small range.
        table.move_cursor(column=app.table_page["columns"].index("b"))
        await pilot.press("p")
        await pilot.pause()
        assert isinstance(app.screen, PlotScreen)
        plot = app.screen
        await pilot.press("g")
        await pilot.pause()
        app.screen.query_one("#range-input", Input).value = "100:140"
        await pilot.press("enter")
        await pilot.pause()
        assert (plot.row_start, plot.row_stop) == (100, 140)

        # 's' opens the searchable Y-column picker over all visible columns.
        await pilot.press("s")
        await pilot.pause()
        assert isinstance(app.screen, ColumnSelectScreen)
        from textual.widgets import OptionList

        picker = app.screen
        option_list = picker.query_one("#colselect-list", OptionList)
        # The picker spans the full visible-column universe, not the paged window.
        all_cols = app.browser.column_names("/level0/ctable")
        assert option_list.option_count == len(all_cols)

        # Typing live-filters the list (substring, case-insensitive): only the
        # 'v0X' columns survive, with the first match highlighted.
        picker.query_one("#colselect-input", Input).value = "v0"
        await pilot.pause()
        assert 0 < option_list.option_count < len(all_cols)
        assert option_list.highlighted == 0

        # Clear the filter and pick 'c' (a second numeric column) by name + Enter.
        picker.query_one("#colselect-input", Input).value = "c"
        await pilot.pause()  # let the live filter repopulate the list
        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.screen, ScatterPlotScreen)
        scatter = app.screen
        # Row-aligned over the framed range: b == row index, c == row * 1.5.
        assert scatter.xcol == "b"
        assert scatter.ycol == "c"
        assert len(scatter.x) == len(scatter.y) == 40
        np.testing.assert_allclose(scatter.x, np.arange(100, 140))
        np.testing.assert_allclose(scatter.y, np.arange(100, 140) * 1.5)

        # 'h' opens a high-res matplotlib scatter over the braille scatter, when
        # textual-image + matplotlib are available; 'escape' returns to it.
        if importlib.util.find_spec("textual_image") and importlib.util.find_spec("matplotlib"):
            from blosc2.b2view.app import HiResPlotScreen, TextualImage

            await pilot.press("h")
            await pilot.pause()
            assert isinstance(app.screen, HiResPlotScreen)
            assert app.screen.query_one("#hires-image", TextualImage) is not None
            await pilot.press("escape")
            await pilot.pause()
            assert app.screen is scatter

        # 'escape' returns to the braille plot with the zoom intact.
        await pilot.press("escape")
        await pilot.pause()
        assert app.screen is plot
        assert (plot.row_start, plot.row_stop) == (100, 140)


# ── Expensive (skipped) CTable cell: decode on demand with Enter ─────────


async def test_enter_decodes_skipped_cell(tmp_path):
    """Enter on a ``<...; skipped>`` list cell opens the decoded-cell modal."""
    import dataclasses

    from blosc2.b2view.app import CellDetailScreen

    @dataclasses.dataclass
    class TaggedRow:
        id: int = blosc2.field(blosc2.int32())
        tags: list[int] = blosc2.field(blosc2.list(blosc2.int64(), nullable=True))  # noqa: RUF009

    path = str(tmp_path / "tagged.b2z")
    rows = [(i, list(range(i + 1))) for i in range(6)]
    with blosc2.TreeStore(path, mode="w") as store:
        store["/t"] = blosc2.CTable(TaggedRow, new_data=rows)

    app = B2ViewApp(path, start_path="/t", start_panel="data")
    async with app.run_test(size=TERM_SIZE) as pilot:
        await wait_for_table(pilot)
        table = await focus_data_table(pilot)

        # The list column is a placeholder in the grid.
        assert "tags" in (app.table_buffer.get("skipped_columns") or {})

        # Enter on the cheap 'id' column does nothing special (no modal).
        table.move_cursor(row=2, column=app.table_page["columns"].index("id"))
        await pilot.press("enter")
        await pilot.pause()
        assert not isinstance(app.screen, CellDetailScreen)

        # Enter on the skipped 'tags' cell decodes just that row into a modal.
        table.move_cursor(row=2, column=app.table_page["columns"].index("tags"))
        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.screen, CellDetailScreen)
        assert app.screen._value == [0, 1, 2]  # row 2 of the generator above
        assert app.screen._row == 2

        # esc returns to the table with its position intact.
        await pilot.press("escape")
        await pilot.pause()
        assert not isinstance(app.screen, CellDetailScreen)
        assert table.cursor_row == 2


# ── SChunk: paged hex dump in the data grid ──────────────────────────────


async def test_schunk_hex_dump_paging(tmp_path):
    """An SChunk node renders a paged hex+ascii dump with byte-offset labels."""
    path = str(tmp_path / "raw.b2z")
    # 4 KiB of a repeating 0..255 ramp, so values at any offset are predictable.
    payload = bytes(range(256)) * 16
    with blosc2.TreeStore(path, mode="w") as store:
        store["/raw"] = blosc2.SChunk(chunksize=2**16, data=payload)

    app = B2ViewApp(path, start_path="/raw", start_panel="data")
    async with app.run_test(size=TERM_SIZE) as pilot:
        await wait_for_table(pilot)
        table = await focus_data_table(pilot)

        page = app.table_page
        assert page["source_kind"] == "schunk"
        assert page["columns"] == ["hex", "ascii"]
        assert page["nrows"] == len(payload) // 16  # 16 bytes/row
        assert page["start"] == 0
        first_stop = page["stop"]
        assert first_stop < page["nrows"]  # bigger than one viewport

        # Row 0 is bytes 0x00..0x0f; the gutter shows the hex byte offset.
        assert page["row_labels"][0] == "00000000"
        assert page["data"]["hex"][0] == "00 01 02 03 04 05 06 07 08 09 0a 0b 0c 0d 0e 0f"

        # The header reports the dump in bytes, not "rows".
        from textual.widgets import Static

        header = app.query_one("#data-header", Static).render()
        assert f"{len(payload)} bytes" in str(header)

        # Page forward by stepping off the last visible row; offsets keep going.
        table.move_cursor(row=first_stop - 1)
        await pilot.press("down")
        await wait_for_table(pilot)
        page = app.table_page
        assert page["start"] == first_stop
        assert page["row_labels"][0] == format(first_stop * 16, "08x")

        # 'b' jumps to the last row of the dump.
        await pilot.press("b")
        await wait_for_table(pilot)
        page = app.table_page
        assert page["stop"] == page["nrows"]
        last_offset = (page["nrows"] - 1) * 16
        assert page["row_labels"][-1] == format(last_offset, "08x")


# ── Download-then-browse (--download option) ─────────────────────────────


async def test_download_then_browse(store_path, tmp_path, monkeypatch):
    """--download shows a progress screen, fetches the bundle, then browses it."""
    import shutil
    import threading

    from textual.widgets import ProgressBar

    from blosc2.b2view import app as app_module
    from blosc2.b2view.app import DownloadScreen

    dest = str(tmp_path / "fetched.b2z")
    size = 987_654  # the info endpoint's reported cbytes
    release = threading.Event()  # let the test observe DownloadScreen before the copy
    calls: list[tuple[str, str]] = []

    def fake_download(url, dst, on_progress):
        on_progress(0, None)  # download stream sends no Content-Length
        assert release.wait(timeout=5)
        shutil.copyfile(store_path, dst)  # stand in for the network fetch
        on_progress(size, None)
        calls.append((url, dst))

    monkeypatch.setattr(app_module, "_http_download", fake_download)
    monkeypatch.setattr(app_module, "_fetch_remote_size", lambda info_url: size)

    download_url = "https://cat2.cloud/demo/api/download/@public/large/fetched.b2z"
    app = B2ViewApp(
        dest,
        download_url=download_url,
        info_url="https://cat2.cloud/demo/api/info/@public/large/fetched.b2z",
    )
    async with app.run_test(size=TERM_SIZE) as pilot:
        await pilot.pause()
        # The bundle is not opened yet: the download screen is up, and the info
        # endpoint's size made the progress bar determinate.
        assert isinstance(app.screen, DownloadScreen)
        assert app.browser is None
        assert app.screen.query_one("#download-bar", ProgressBar).total == size

        release.set()  # let the (faked) download complete
        for _ in range(100):
            await pilot.pause()
            if app.browser is not None:
                break
        # Download finished -> screen dismissed and normal browsing resumed.
        assert calls == [(download_url, dest)]
        assert not isinstance(app.screen, DownloadScreen)
        assert app.browser is not None
        assert len(app.query_one("#tree", Tree).root.children) > 0
        # The header shows the @public-relative path to the left of the title.
        from textual.widgets._header import HeaderTitle

        assert "large/fetched.b2z" in app.query_one(HeaderTitle).render().plain
