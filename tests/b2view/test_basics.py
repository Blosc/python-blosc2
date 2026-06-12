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

import numpy as np
import pytest

pytest.importorskip("textual")
pytest.importorskip("pytest_asyncio")

import tree_store_gen as gen
from textual.widgets import DataTable, Input, Tree

from blosc2.b2view.app import B2ViewApp, FilterScreen, GoToColumnScreen, GoToRowScreen, HelpScreen

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


async def test_tree_and_panel_focus(store_path):
    """Tab cycles the panels; Down/Enter in the tree selects nodes."""
    app = B2ViewApp(store_path)
    async with app.run_test(size=TERM_SIZE) as pilot:
        await pilot.pause()
        assert isinstance(app.focused, Tree)

        # Tab: tree -> meta -> vlmeta -> data and wraps back to the tree
        for expected in ["meta-scroll", "vlmeta-scroll", "data-scroll", "tree"]:
            await pilot.press("tab")
            await pilot.pause()
            assert app.focused is not None
            assert app.focused.id == expected

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

        # 'c' jumps to a column by index (arrays have no column names)
        await pilot.press("c")
        await pilot.pause()
        assert isinstance(app.screen, GoToColumnScreen)
        app.screen.query_one("#gotocol-input", Input).value = "97"
        await pilot.press("enter")
        await wait_for_table(pilot)
        page = app.table_page
        assert page["col_start"] == 97
        assert page["col_stop"] == LEAF2_SHAPE[1]
        np.testing.assert_allclose(page["data"]["97"], expected[page["start"] : page["stop"], 97])


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

        # 'c' goes to a column by name; the row position is kept
        await pilot.press("c")
        await pilot.pause()
        assert isinstance(app.screen, GoToColumnScreen)
        app.screen.query_one("#gotocol-input", Input).value = "v12"
        await pilot.press("enter")
        await wait_for_table(pilot)
        page = app.table_page
        assert page["col_start"] == all_names.index("v12")
        assert page["columns"][0] == "v12"
        assert table.cursor_column == 0
        assert page["start"] + table.cursor_row == 150
        _assert_ctable_window_values(page, expected)

        # An ambiguous name prefix keeps the modal open; escape cancels
        await pilot.press("c")
        await pilot.pause()
        app.screen.query_one("#gotocol-input", Input).value = "v1"
        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.screen, GoToColumnScreen)
        await pilot.press("escape")
        await wait_for_table(pilot)
        assert app.table_page["col_start"] == all_names.index("v12")

        # ...and a numeric index works as well
        await pilot.press("c")
        await pilot.pause()
        app.screen.query_one("#gotocol-input", Input).value = "0"
        await pilot.press("enter")
        await wait_for_table(pilot)
        assert app.table_page["col_start"] == 0

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

        # ── Column filtering ('/' modal) ─────────────────────────────────

        async def submit_column_filter(pattern: str) -> None:
            await pilot.press("slash")
            await pilot.pause()
            assert isinstance(app.screen, FilterScreen)
            app.screen.query_one("#filter-input", Input).value = pattern
            await pilot.press("enter")
            await wait_for_table(pilot)

        # 'v1' matches v10..v19; paging universe shrinks to those 10 columns
        await submit_column_filter("v1")
        page = app.table_page
        assert page["ncols"] == 10
        assert page["columns"][0] == "v10"
        assert all(name.startswith("v1") for name in page["columns"])

        # The goto-column modal resolves names within the filtered set
        await pilot.press("c")
        await pilot.pause()
        app.screen.query_one("#gotocol-input", Input).value = "v15"
        await pilot.press("enter")
        await wait_for_table(pilot)
        assert app.table_page["columns"][0] == "v15"

        # Row and column filters combine (back at the first column window)
        await pilot.press("s")
        await wait_for_table(pilot)
        await submit_filter("b >= 100 and b < 110")
        page = app.table_page
        assert page["nrows"] == 10
        assert page["ncols"] == 10
        np.testing.assert_array_equal(page["data"]["v10"], expected["v10"][100:110])

        # A pattern matching nothing notifies and keeps the selection
        await submit_column_filter("nosuchcol")
        assert app.browser.get_column_filter("/level0/ctable") == "v1"
        assert app.table_page["ncols"] == 10

        # Escape clears one layer at a time: row filter first, then columns
        await pilot.press("escape")
        await wait_for_table(pilot)
        assert app.browser.get_filter("/level0/ctable") is None
        assert app.table_page["nrows"] == NROWS
        assert app.table_page["ncols"] == 10
        await pilot.press("escape")
        await wait_for_table(pilot)
        assert app.browser.get_column_filter("/level0/ctable") is None
        assert app.table_page["ncols"] == len(expected)
