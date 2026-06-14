"""Textual application for b2view."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from rich.markup import escape as markup_escape
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import DataTable, Footer, Header, Input, Static, Tree

try:
    from textual_plotext import PlotextPlot
except ImportError:  # plotting is optional
    PlotextPlot = None

from blosc2.b2view.model import DataSliceLayout, StoreBrowser
from blosc2.b2view.render import (
    column_float_decimals,
    format_cell,
    make_metadata_renderable,
    make_preview_renderables,
)

if TYPE_CHECKING:
    from textual import events

_KIND_ICONS = {
    "group": "📁",
    "ndarray": "▦",
    "c2array": "▦",
    "ctable": "▤",
    "schunk": "▣",
    "unknown": "?",
}

# Source kinds whose data grid supports horizontal (column) paging.
_COL_PAGED_KINDS = frozenset({"ndarray2d", "ndarray_slice", "ctable"})


class B2ViewPanel(Vertical):
    """Pane container that can be maximized."""

    ALLOW_MAXIMIZE = True


class BufferedDataTable(DataTable):
    """DataTable with app-controlled page changes at row boundaries."""

    def action_cursor_down(self) -> None:
        app = self.app
        if getattr(app, "_dim_mode", False):
            getattr(app, "_dim_adjust", lambda _: None)(-1)
            return
        if self.cursor_row >= self.row_count - 1 and getattr(app, "page_table", lambda _: False)(1):
            return
        super().action_cursor_down()

    def action_cursor_up(self) -> None:
        app = self.app
        if getattr(app, "_dim_mode", False):
            getattr(app, "_dim_adjust", lambda _: None)(1)
            return
        if self.cursor_row <= 0 and getattr(app, "page_table", lambda _: False)(-1):
            return
        super().action_cursor_up()

    def action_cursor_right(self) -> None:
        app = self.app
        if getattr(app, "_dim_mode", False):
            getattr(app, "_dim_cursor", lambda _: None)(1)
            return
        if self.cursor_column >= len(self.columns) - 1 and getattr(
            app, "page_grid_columns", lambda _: False
        )(1):
            return
        super().action_cursor_right()

    def action_cursor_left(self) -> None:
        app = self.app
        if getattr(app, "_dim_mode", False):
            getattr(app, "_dim_cursor", lambda _: None)(-1)
            return
        if self.cursor_column <= 0 and getattr(app, "page_grid_columns", lambda _: False)(-1):
            return
        super().action_cursor_left()

    def action_page_down(self) -> None:
        if getattr(self.app, "page_table", lambda *a, **k: False)(1, align=True):
            return
        super().action_page_down()

    def action_page_up(self) -> None:
        if getattr(self.app, "page_table", lambda *a, **k: False)(-1, align=True):
            return
        super().action_page_up()

    def action_page_right(self) -> None:
        if getattr(self.app, "page_grid_columns", lambda _: False)(1):
            return
        super().action_page_right()

    def action_page_left(self) -> None:
        if getattr(self.app, "page_grid_columns", lambda _: False)(-1):
            return
        super().action_page_left()

    def action_select_cursor(self) -> None:
        app = self.app
        if getattr(app, "_dim_mode", False):
            getattr(app, "action_dim_toggle_nav", lambda: None)()
            return
        super().action_select_cursor()

    def _wheel_step(self) -> int:
        # Half the visible rows per tick; arrow keys remain the
        # single-step path (also for dim-mode index changes).
        return max(1, self.row_count // 2)

    def on_mouse_scroll_down(self, event: events.MouseScrollDown) -> None:
        # The grid holds exactly one viewport-sized page, so the default
        # scroll handler has nothing to scroll; move the cursor instead,
        # which pages at the edges just like the arrow keys.
        event.stop()
        event.prevent_default()
        for _ in range(self._wheel_step()):
            self.action_cursor_down()

    def on_mouse_scroll_up(self, event: events.MouseScrollUp) -> None:
        event.stop()
        event.prevent_default()
        for _ in range(self._wheel_step()):
            self.action_cursor_up()

    def on_resize(self, event) -> None:
        # The column/row windows are fitted to this table's size; re-check
        # whenever it changes (terminal resize, panel maximize, ...).
        getattr(self.app, "_on_data_table_resized", lambda: None)()

    def action_scroll_home(self) -> None:
        if getattr(self.app, "_grid_col_home", lambda: False)():
            pass
        else:
            super().action_scroll_home()

    def action_scroll_end(self) -> None:
        if getattr(self.app, "_grid_col_end", lambda: False)():
            pass
        else:
            super().action_scroll_end()


class HelpScreen(ModalScreen[None]):
    """Modal listing all key bindings, grouped by area."""

    CSS = """
    HelpScreen {
        align: center middle;
    }
    #help-dialog {
        width: 62;
        height: auto;
        max-height: 90%;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }
    #help-title {
        text-style: bold;
        margin-bottom: 1;
    }
    #help-body {
        height: auto;
    }
    """

    BINDINGS: ClassVar = [
        ("escape", "close", "Close"),
        ("question_mark", "close", "Close"),
        ("q", "close", "Close"),
    ]

    _SECTIONS: ClassVar = [
        (
            "Panels",
            [
                ("tab / shift+tab", "next / previous panel"),
                ("m", "maximize the focused panel"),
                ("r", "restore panel (or refresh the tree)"),
                ("q", "quit"),
            ],
        ),
        (
            "Tree",
            [
                ("up / down", "move between nodes"),
                ("enter", "select node (and expand groups)"),
            ],
        ),
        (
            "Data grid — rows",
            [
                ("up / down", "move cursor; pages at the edges"),
                ("pageup / pagedown", "previous / next page"),
                ("t / b", "first / last row"),
                ("g", "go to row..."),
                ("f", "filter rows (CTable)"),
                ("escape", "unlock a row window / clear the active filter"),
            ],
        ),
        (
            "Data grid — columns",
            [
                ("left / right", "move cursor; pages at the edges"),
                ("s / e  (home / end)", "first / last column window"),
                ("c", "go to column index or name..."),
                ("/", "filter visible columns by substring (CTable)"),
                ("p", "plot a whole-column overview (needs textual-plotext)"),
            ],
        ),
        (
            "Plot modal (after 'p')",
            [
                ("+ / -", "zoom in / out about the centre"),
                ("left / right", "pan the zoomed window"),
                ("0", "reset to the whole series"),
                ("g", "type an exact start:stop row range"),
                ("v", "lock the data grid to the current range (esc unlocks)"),
            ],
        ),
        (
            "Dim mode (N-D arrays)",
            [
                ("d", "toggle dim mode"),
                ("left / right", "select the active dimension"),
                ("up / down", "change fixed index / scroll viewport"),
                ("enter", "toggle fixed <-> navigable"),
                ("escape", "exit dim mode"),
            ],
        ),
    ]

    def compose(self) -> ComposeResult:
        from rich.table import Table

        body = Table(show_header=False, box=None, padding=(0, 1))
        body.add_column("key", style="bold cyan", no_wrap=True)
        body.add_column("action")
        for i, (section, entries) in enumerate(self._SECTIONS):
            if i:
                body.add_row("", "")
            body.add_row(f"[bold]{section}[/bold]", "")
            for key, action in entries:
                body.add_row(key, action)
        with Vertical(id="help-dialog"):
            yield Static("b2view keys  (esc to close)", id="help-title")
            with VerticalScroll(id="help-body"):
                yield Static(body)

    def action_close(self) -> None:
        self.dismiss(None)


class GoToRowScreen(ModalScreen[int | None]):
    """Small modal asking for a global row number."""

    CSS = """
    GoToRowScreen {
        align: center middle;
    }
    #goto-dialog {
        width: 50;
        height: auto;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }
    #goto-title {
        text-style: bold;
        margin-bottom: 1;
    }
    """

    BINDINGS: ClassVar = [("escape", "cancel", "Cancel")]

    def __init__(self, *, nrows: int, current: int):
        super().__init__()
        self.nrows = nrows
        self.current = current

    def compose(self) -> ComposeResult:
        with Vertical(id="goto-dialog"):
            yield Static(f"Go to row 0..{self.nrows - 1} (current: {self.current})", id="goto-title")
            yield Input(placeholder="row number", id="goto-input")

    def on_mount(self) -> None:
        input_widget = self.query_one("#goto-input", Input)
        input_widget.value = str(self.current)
        input_widget.focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        value = event.value.strip().replace("_", "")
        try:
            row = int(value)
        except ValueError:
            self.query_one("#goto-title", Static).update("Please enter an integer row number")
            return
        if not 0 <= row < self.nrows:
            self.query_one("#goto-title", Static).update(f"Row must be in range 0..{self.nrows - 1}")
            return
        self.dismiss(row)

    def action_cancel(self) -> None:
        self.dismiss(None)


class GoToColumnScreen(ModalScreen[int | None]):
    """Small modal asking for a column index or (for CTables) a column name."""

    CSS = """
    GoToColumnScreen {
        align: center middle;
    }
    #gotocol-dialog {
        width: 50;
        height: auto;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }
    #gotocol-title {
        text-style: bold;
        margin-bottom: 1;
    }
    """

    BINDINGS: ClassVar = [("escape", "cancel", "Cancel")]

    def __init__(self, *, ncols: int, current: int, names: list[str] | None = None):
        super().__init__()
        self.ncols = ncols
        self.current = current
        self.names = names

    def compose(self) -> ComposeResult:
        what = f"column 0..{self.ncols - 1}"
        if self.names:
            what += " or name"
        with Vertical(id="gotocol-dialog"):
            yield Static(f"Go to {what} (current: {self.current})", id="gotocol-title")
            yield Input(placeholder="column index or name", id="gotocol-input")

    def on_mount(self) -> None:
        input_widget = self.query_one("#gotocol-input", Input)
        input_widget.value = str(self.current)
        input_widget.focus()

    def _fail(self, message: str) -> None:
        self.query_one("#gotocol-title", Static).update(message)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        value = event.value.strip().replace("_", "")
        try:
            col = int(value)
        except ValueError:
            col = self._match_name(event.value.strip())
            if col is None:
                return
        if not 0 <= col < self.ncols:
            self._fail(f"Column must be in range 0..{self.ncols - 1}")
            return
        self.dismiss(col)

    def _match_name(self, value: str) -> int | None:
        """Resolve a column name (exact, or unique prefix) to its index."""
        if not self.names:
            self._fail("Please enter an integer column index")
            return None
        if value in self.names:
            return self.names.index(value)
        matches = [i for i, name in enumerate(self.names) if name.startswith(value)] if value else []
        if len(matches) == 1:
            return matches[0]
        self._fail(f"{'Ambiguous' if matches else 'Unknown'} column name {value!r}")
        return None

    def action_cancel(self) -> None:
        self.dismiss(None)


class FilterScreen(ModalScreen[str | None]):
    """Small modal asking for a CTable filter (row expression or column pattern)."""

    CSS = """
    FilterScreen {
        align: center middle;
    }
    #filter-dialog {
        width: 70;
        height: auto;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }
    #filter-title {
        text-style: bold;
        margin-bottom: 1;
    }
    """

    BINDINGS: ClassVar = [("escape", "cancel", "Cancel")]

    def __init__(
        self,
        *,
        current: str | None = None,
        title: str = "Filter rows (empty clears)",
        placeholder: str = "e.g. payment.tips > 100 and trip.km > 0",
    ):
        super().__init__()
        self.current = current or ""
        self.title_text = title
        self.placeholder = placeholder

    def compose(self) -> ComposeResult:
        with Vertical(id="filter-dialog"):
            yield Static(self.title_text, id="filter-title")
            yield Input(placeholder=self.placeholder, id="filter-input")

    def on_mount(self) -> None:
        input_widget = self.query_one("#filter-input", Input)
        input_widget.value = self.current
        input_widget.focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value.strip())

    def action_cancel(self) -> None:
        self.dismiss(None)


def _plot_view(series: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Turn a ``plot_series`` result into drawable arrays + a method label.

    Drops all-NaN buckets (no finite extremes) and maps the read method to a
    human description shown in the title.
    """
    x = np.asarray(series["x"])
    ymin = np.asarray(series["ymin"], dtype=np.float64)
    ymax = np.asarray(series["ymax"], dtype=np.float64)
    finite = np.isfinite(ymin) & np.isfinite(ymax)
    x, ymin, ymax = x[finite], ymin[finite], ymax[finite]
    method = series.get("method")
    descr = {"summary": "min/max envelope", "reduce": "min/max envelope"}.get(
        method, "sampled — may miss extremes"
    )
    return x, ymin, ymax, descr


class PlotRangeScreen(ModalScreen["tuple[int, int] | None"]):
    """Small modal asking for an explicit ``start:stop`` row range."""

    CSS = """
    PlotRangeScreen {
        align: center middle;
    }
    #range-dialog {
        width: 50;
        height: auto;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }
    #range-title {
        text-style: bold;
        margin-bottom: 1;
    }
    """

    BINDINGS: ClassVar = [("escape", "cancel", "Cancel")]

    def __init__(self, *, n: int, start: int, stop: int):
        super().__init__()
        self.n = n
        self.start = start
        self.stop = stop

    def compose(self) -> ComposeResult:
        with Vertical(id="range-dialog"):
            yield Static(
                f"Row range start:stop within 0..{self.n} (current {self.start}:{self.stop})",
                id="range-title",
            )
            yield Input(placeholder="start:stop", id="range-input")

    def on_mount(self) -> None:
        widget = self.query_one("#range-input", Input)
        widget.value = f"{self.start}:{self.stop}"
        widget.focus()

    def _parse(self, text: str) -> tuple[int, int] | None:
        if ":" not in text:
            return None
        lo, hi = text.split(":", 1)
        try:
            start = int(lo) if lo.strip() else 0
            stop = int(hi) if hi.strip() else self.n
        except ValueError:
            return None
        start = max(0, min(start, self.n))
        stop = max(0, min(stop, self.n))
        return None if stop <= start else (start, stop)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        parsed = self._parse(event.value.strip().replace("_", ""))
        if parsed is None:
            self.query_one("#range-title", Static).update("Enter a range as start:stop")
            return
        self.dismiss(parsed)

    def action_cancel(self) -> None:
        self.dismiss(None)


class PlotScreen(ModalScreen["tuple[int, int] | None"]):
    """Modal plotting one numeric column; zoomable into a row sub-range.

    Keys: ``+``/``-`` zoom about the view centre, ``←``/``→`` pan, ``0`` reset to
    the whole series, ``g`` type an exact ``start:stop`` range.  Each change
    re-fetches the envelope for the new range (exact for sub-ranges) via the
    *fetch* closure, so zooming reveals detail the whole-series buckets hide.

    ``v`` dismisses with the current ``(row_start, row_stop)`` so the caller can
    jump the data grid to the range you navigated to; closing dismisses ``None``.
    """

    CSS = """
    PlotScreen {
        align: center middle;
    }
    #plot-dialog {
        width: 90%;
        height: 80%;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }
    #plot-title {
        text-style: bold;
        height: 1;
    }
    #plot-widget {
        height: 1fr;
    }
    #plot-keys {
        height: 1;
        color: $text-muted;
    }
    """

    _KEYS_HINT = "+/- zoom · ←/→ pan · 0 reset · g range · v view rows · q close"
    _MIN_WIDTH = 16  # smallest zoom window (rows), so the envelope still reads

    BINDINGS: ClassVar = [
        ("escape", "close", "Close"),
        ("q", "close", "Close"),
        ("p", "close", "Close"),
        ("plus", "zoom_in", "Zoom in"),
        ("equals_sign", "zoom_in", "Zoom in"),
        ("minus", "zoom_out", "Zoom out"),
        ("left", "pan_left", "Pan left"),
        ("right", "pan_right", "Pan right"),
        ("0", "reset_range", "Reset"),
        ("g", "goto_range", "Range"),
        ("v", "view_range", "View rows"),
    ]

    def __init__(self, *, title_prefix: str, fetch, n: int, row_start: int, row_stop: int, series: dict):
        super().__init__()
        self.title_prefix = title_prefix
        self._fetch = fetch
        self.n = n
        self.row_start = row_start
        self.row_stop = row_stop
        self._apply(series)

    def _apply(self, series: dict) -> None:
        x, ymin, ymax, descr = _plot_view(series)
        self.x = list(x)
        self.ymin = list(ymin)
        self.ymax = list(ymax)
        full = self.row_start == 0 and self.row_stop == self.n
        rng = "" if full else f" · rows {self.row_start}:{self.row_stop}"
        note = "" if self.x else " · (no finite values in range)"
        self.plot_title = f"{self.title_prefix} · {self.n} rows{rng} · {descr}{note}"

    def compose(self) -> ComposeResult:
        with Vertical(id="plot-dialog"):
            yield Static(markup_escape(self.plot_title), id="plot-title")
            yield PlotextPlot(id="plot-widget")
            yield Static(self._KEYS_HINT, id="plot-keys")

    def on_mount(self) -> None:
        self._redraw()

    def _redraw(self) -> None:
        widget = self.query_one(PlotextPlot)
        plt = widget.plt
        plt.clear_figure()
        if self.x:
            # Upper (max) and lower (min) envelope; a single line when they
            # coincide (a sampled series).
            plt.plot(self.x, self.ymax, marker="braille")
            if self.ymin != self.ymax:
                plt.plot(self.x, self.ymin, marker="braille")
        plt.xlabel("row")
        widget.refresh()
        self.query_one("#plot-title", Static).update(markup_escape(self.plot_title))

    def _set_range(self, start: int, stop: int) -> None:
        start = max(0, min(int(start), self.n))
        stop = max(0, min(int(stop), self.n))
        if stop <= start or (start, stop) == (self.row_start, self.row_stop):
            return
        self.row_start, self.row_stop = start, stop
        self._apply(self._fetch(start, stop))
        self._redraw()

    def _zoom(self, factor: float) -> None:
        width = self.row_stop - self.row_start
        center = (self.row_start + self.row_stop) // 2
        new_w = width // 2 if factor < 1 else width * 2
        new_w = max(min(self._MIN_WIDTH, self.n), min(self.n, new_w))
        start = max(0, min(center - new_w // 2, self.n - new_w))
        self._set_range(start, start + new_w)

    def _pan(self, direction: int) -> None:
        width = self.row_stop - self.row_start
        delta = max(1, width // 4) * direction
        start = max(0, min(self.row_start + delta, self.n - width))
        self._set_range(start, start + width)

    def action_zoom_in(self) -> None:
        self._zoom(0.5)

    def action_zoom_out(self) -> None:
        self._zoom(2.0)

    def action_pan_left(self) -> None:
        self._pan(-1)

    def action_pan_right(self) -> None:
        self._pan(1)

    def action_reset_range(self) -> None:
        self._set_range(0, self.n)

    def action_goto_range(self) -> None:
        def _on_range(result: tuple[int, int] | None) -> None:
            if result is not None:
                self._set_range(*result)

        self.app.push_screen(PlotRangeScreen(n=self.n, start=self.row_start, stop=self.row_stop), _on_range)

    def action_view_range(self) -> None:
        """v key — close the plot and jump the data grid to the current range."""
        self.dismiss((self.row_start, self.row_stop))

    def action_close(self) -> None:
        self.dismiss(None)


class B2ViewApp(App):
    """Browse TreeStore hierarchy and preview objects."""

    CSS = """
    #main { height: 1fr; }
    #tree-pane { width: 35%; border: solid $primary; }
    #right-pane { width: 65%; }
    #top-row { height: 40%; }
    #meta-pane, #vlmeta-pane { width: 50%; border: solid $secondary; }
    #data-pane { height: 60%; border: solid $secondary; }
    #tree { height: 1fr; }
    #data-header { height: auto; padding: 0 1; }
    #data-table-row { height: 1fr; }
    #data-table { width: 1fr; height: 1fr; }
    #row-scrollbar { width: 1; height: 1fr; color: $accent; }
    #col-scrollbar { height: 1; width: 1fr; color: $accent; }
    #meta-scroll, #vlmeta-scroll, #data-scroll { height: 1fr; padding: 0 1; }
    #tree-pane:focus-within, #meta-pane:focus-within, #vlmeta-pane:focus-within, #data-pane:focus-within { border: heavy $accent; }
    B2ViewPanel.-maximized,
    #tree-pane.-maximized,
    #meta-pane.-maximized,
    #data-pane.-maximized { width: 1fr; height: 1fr; }
    """

    BINDINGS: ClassVar = [
        ("q", "quit", "Quit"),
        ("question_mark", "show_help", "Help"),
        ("tab", "focus_next_panel", "Next panel"),
        ("shift+tab", "focus_previous_panel", "Previous panel"),
        Binding("g", "go_to_row", "Go to row", show=False),
        ("m", "maximize_panel", "Maximize"),
        ("r", "restore_or_refresh", "Restore/Refresh"),
        Binding("t", "grid_row_top", "Top", show=False),
        Binding("b", "grid_row_bottom", "Bottom", show=False),
        Binding("s", "grid_col_start", "Row start", show=False),
        Binding("e", "grid_col_end", "Row end", show=False),
        Binding("c", "go_to_column", "Go to column", show=False),
        Binding("f", "filter_rows", "Filter rows", show=False),
        Binding("slash", "filter_columns", "Filter columns", show=False),
        Binding("p", "plot_column", "Plot column", show=False),
        Binding("d", "dim_cycle", "Dim mode", show=False),
        Binding("enter", "dim_toggle_nav", "Toggle nav", show=False),
        Binding("escape", "dim_exit", "Exit dim mode", show=False),
    ]

    def __init__(
        self,
        urlpath: str,
        *,
        start_path: str = "/",
        start_panel: str = "tree",
        preview_rows: int = 20,
        preview_cols: int = 10,
    ):
        super().__init__()
        self.urlpath = urlpath
        self.start_path = start_path
        self.start_panel = start_panel
        self.preview_rows = preview_rows
        self.preview_cols = preview_cols
        self.browser: StoreBrowser | None = None
        self.loaded_paths: set[str] = set()
        self.selected_path = "/"
        self.table_page: dict | None = None
        self.table_buffer: dict | None = None
        self.grid_col_start = 0
        self._data_layout: DataSliceLayout | None = None
        self._active_dim = 0
        self._dim_mode = False
        self.loading_table_page = False
        # Absolute (start, stop) of a locked row window from the plot's 'v' key.
        self.row_window: tuple[int, int] | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main"):
            with B2ViewPanel(id="tree-pane") as tree_pane:
                tree_pane.border_title = "tree"
                yield Tree("/", id="tree")
            with Vertical(id="right-pane"):
                with Horizontal(id="top-row"):
                    with B2ViewPanel(id="meta-pane") as meta_pane:
                        meta_pane.border_title = "meta"
                        with VerticalScroll(id="meta-scroll", can_focus=True):
                            yield Static("Select a node", id="metadata")
                    with B2ViewPanel(id="vlmeta-pane") as vlmeta_pane:
                        vlmeta_pane.border_title = "vlmeta"
                        with VerticalScroll(id="vlmeta-scroll", can_focus=True):
                            yield Static("", id="vlmetadata")
                with B2ViewPanel(id="data-pane") as data_pane:
                    data_pane.border_title = "data"
                    data_pane.border_subtitle = (
                        "?(help) | d(im mode) | filter: f(rows) /(cols) | "
                        "rows: t/b/g(oto) | cols: s/e/c(goto) | p(lot)"
                    )
                    yield Static("", id="data-header")
                    with Horizontal(id="data-table-row"):
                        yield BufferedDataTable(id="data-table", show_row_labels=True, zebra_stripes=True)
                        yield Static("", id="row-scrollbar")
                    yield Static("", id="col-scrollbar")
                    with VerticalScroll(id="data-scroll", can_focus=True):
                        yield Static("", id="preview")
        yield Footer()

    def on_mount(self) -> None:
        self.browser = StoreBrowser(self.urlpath)
        tree = self.query_one("#tree", Tree)
        tree.root.data = "/"
        self.load_children(tree.root)
        tree.root.expand()
        self.query_one("#data-table-row", Horizontal).display = False
        self.query_one("#col-scrollbar", Static).display = False

        if self.start_path and self.start_path != "/":
            self._navigate_to_path(self.start_path)
        else:
            self.call_after_refresh(self.update_panels, "/")
            tree.focus()

        # Override focus after render settles, when starting panel is not the tree
        if self.start_panel != "tree":
            self.set_timer(0.05, lambda: self._focus_panel_by_name(self.start_panel))

    def _focus_panel_by_name(self, name: str) -> None:
        """Focus a panel by its user-facing name."""
        panel_map = {
            "tree": lambda: self.query_one("#tree", Tree),
            "meta": lambda: self.query_one("#meta-scroll", VerticalScroll),
            "vlmeta": lambda: self.query_one("#vlmeta-scroll", VerticalScroll),
            "data": lambda: (
                self.query_one("#data-table", DataTable)
                if self.query_one("#data-table-row", Horizontal).display
                else self.query_one("#data-scroll", VerticalScroll)
            ),
        }
        getter = panel_map.get(name)
        if getter is not None:
            getter().focus()

    def _navigate_to_path(self, path: str) -> None:
        """Expand the tree and select the node at *path*."""
        tree = self.query_one("#tree", Tree)
        parts = [p for p in path.split("/") if p]
        node = tree.root
        # Walk down the tree expanding each level
        for part in parts:
            self.load_children(node)
            found = None
            for child in node.children:
                if child.label and child.label.plain.endswith(f" {part}"):
                    found = child
                    break
            if found is None:
                # Path not found — fall back to root
                self.call_after_refresh(self.update_panels, "/")
                tree.focus()
                return
            if found.allow_expand:
                self.load_children(found)
            found.expand()
            node = found

        # Selecting the node fires NodeSelected → on_tree_node_selected → update_panels
        def _do_select():
            tree.select_node(node)
            tree.scroll_to_node(node)
            tree.focus()

        self.call_after_refresh(_do_select)

    def on_unmount(self) -> None:
        if self.browser is not None:
            self.browser.close()

    def load_children(self, node) -> None:
        path = node.data or "/"
        if self.browser is None or path in self.loaded_paths:
            return
        for child in self.browser.list_children(path):
            icon = _KIND_ICONS.get(child.kind, "?")
            node.add(f"{icon} {child.name}", data=child.path, allow_expand=child.has_children)
        self.loaded_paths.add(path)

    def on_tree_node_expanded(self, event: Tree.NodeExpanded) -> None:
        self.load_children(event.node)

    def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        path = event.node.data or "/"
        self.selected_path = path
        self.update_panels(path)
        if event.node.allow_expand:
            self.load_children(event.node)

    def update_panels(self, path: str) -> None:
        if self.browser is None:
            return
        metadata = self.query_one("#metadata", Static)
        data_header = self.query_one("#data-header", Static)
        data_table_row = self.query_one("#data-table-row", Horizontal)
        data_scroll = self.query_one("#data-scroll", VerticalScroll)
        preview = self.query_one("#preview", Static)
        vlmeta_pane = self.query_one("#vlmeta-pane", B2ViewPanel)
        vlmeta_widget = self.query_one("#vlmetadata", Static)
        try:
            info = self.browser.get_info(path)
            metadata.update(make_metadata_renderable(info))
            self.table_buffer = None
            self.grid_col_start = 0
            self._data_layout = None
            self._active_dim = 0
            self._dim_mode = False
            # A locked row window does not survive navigating to a node.
            self.row_window = None
            self.browser.clear_row_window(path)
            if info.kind == "group":
                data_header.display = False
                data_table_row.display = False
                data_scroll.display = True
                self.query_one("#col-scrollbar", Static).display = False
                data_header.update("")
                preview.update("Group node; select an array or table to preview.")
                self._update_vlmeta(vlmeta_pane, vlmeta_widget, path)
            else:
                if self._uses_grid_preview(info):
                    data_header.display = True
                    data_table_row.display = True
                    data_scroll.display = False
                    preview.update("")
                    shape = tuple(info.metadata.get("shape", ()) or ())
                    ndim = len(shape)
                    if ndim >= 1 and self._data_layout is None:
                        self._data_layout = DataSliceLayout.from_shape(shape)
                        self._active_dim = 0
                    data = self._load_table_page(path, 0)
                else:
                    data = self.browser.preview(path, max_rows=self.preview_rows, max_cols=self.preview_cols)
                if self._is_table_preview(data):
                    # A freshly selected node starts at the first column
                    self._update_data_table(data, cursor_col=0)
                    self._update_data_header(data)
                    self.call_after_refresh(self._ensure_viewport_consistent)
                else:
                    header, body = make_preview_renderables(data)
                    data_header.display = header is not None
                    data_table_row.display = False
                    data_scroll.display = True
                    self.query_one("#col-scrollbar", Static).display = False
                    data_header.update("" if header is None else header)
                    preview.update(body)
            self._update_vlmeta(vlmeta_pane, vlmeta_widget, path)
            self._reset_panel_scroll()
        except Exception as exc:
            metadata.update(f"Error reading {path}: {exc}")
            data_header.display = False
            data_table_row.display = False
            data_scroll.display = True
            self.query_one("#col-scrollbar", Static).display = False
            data_header.update("")
            preview.update("")
            self._update_vlmeta(vlmeta_pane, vlmeta_widget, None)
            self._reset_panel_scroll()

    @staticmethod
    def _format_vlmeta_value(value: Any) -> str:
        """Format a vlmeta value for display."""
        if isinstance(value, bool):
            return str(value)
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, (list, tuple)):
            return ", ".join(str(v) for v in value)
        if isinstance(value, dict):
            return ", ".join(f"{k}: {v}" for k, v in value.items())
        return str(value)

    def _update_vlmeta(self, pane, widget: Static, path: str | None) -> None:
        """Populate the vlmeta pane with variable-length metadata."""
        pane.display = True
        if path is None or self.browser is None:
            widget.update("<not available>")
            return
        try:
            info = self.browser.get_info(path)
            if info.user_attrs is None:
                widget.update("<not available>")
            elif not info.user_attrs:
                widget.update("")
            else:
                from rich.table import Table

                table = Table(show_header=False, box=None, expand=True)
                table.add_column("key", style="bold cyan", no_wrap=True)
                table.add_column("value")
                for k, v in info.user_attrs.items():
                    table.add_row(str(k), self._format_vlmeta_value(v))
                widget.update(table)
        except Exception:
            widget.update("<not available>")

    @staticmethod
    def _is_table_preview(data) -> bool:
        return isinstance(data, dict) and "data" in data and "columns" in data

    @staticmethod
    def _uses_grid_preview(info) -> bool:
        # 1D, 2D, 3D+ NDArray/C2Array all use grid preview
        return info.kind == "ctable" or (
            info.kind in {"ndarray", "c2array"} and info.metadata.get("ndim", 0) >= 1
        )

    def _col_page_size(self) -> int:
        """Return the number of columns that fit in the current data table width."""
        table = self.query_one("#data-table", DataTable)
        width = table.size.width
        if width <= 1:
            return self.preview_cols
        # Each column uses roughly 9 characters (float format width) + 2 padding.
        # Row labels take about 6 characters.
        col_width = 11
        # Subtract row-label column space
        usable = max(1, width - 6)
        return max(1, usable // col_width)

    # DataTable pads each cell with one space on both sides (cell_padding=1)
    _CELL_PAD = 2

    def _data_table_width(self) -> int:
        return self.query_one("#data-table", DataTable).size.width

    def _col_avail_width(self, nrows: int) -> int:
        """Width available for data columns, or 0 before layout has settled."""
        width = self._data_table_width()
        if width <= 1:
            return 0
        label_width = len(str(max(0, int(nrows) - 1))) + self._CELL_PAD
        return max(1, width - label_width)

    def _candidate_max_cols(self) -> int:
        """Upper bound of columns worth fetching before the width-based trim."""
        width = self._data_table_width()
        if width <= 1:
            return self.preview_cols
        # The narrowest possible column is one character plus padding.
        return max(1, width // (1 + self._CELL_PAD))

    @classmethod
    def _measure_column_widths(cls, data: dict) -> list[int]:
        """Rendered width (content + padding) of every column in *data*."""
        widths = []
        for name in data["columns"]:
            cells = data["data"][name]
            decimals = column_float_decimals(cells)
            content = max(
                len(str(name)),
                max((len(format_cell(value, float_decimals=decimals)) for value in cells), default=1),
            )
            widths.append(content + cls._CELL_PAD)
        return widths

    def _trim_columns_to_fit(self, data: dict) -> dict:
        """Drop trailing columns of *data* that do not fit the table width.

        The preview fetches a generous candidate window of columns; this
        second pass measures the actual rendered cell widths and keeps only
        as many whole columns as truly fit the data table.
        """
        if data.get("source_kind") not in _COL_PAGED_KINDS:
            return data
        avail = self._col_avail_width(data["nrows"])
        if avail <= 0:
            return data  # layout not settled; keep the estimate-based window
        widths = self._measure_column_widths(data)
        keep = 0
        total = 0
        for width in widths:
            if keep >= 1 and total + width > avail:
                break
            total += width
            keep += 1
        if keep >= len(data["columns"]):
            return data
        kept = data["columns"][:keep]
        data = dict(data)
        data["data"] = {name: data["data"][name] for name in kept}
        data["columns"] = kept
        data["col_stop"] = data["col_start"] + keep
        data["hidden_columns"] = max(0, data["ncols"] - keep)
        return data

    def _fetch_columns_for_measure(self, col_start: int, count: int) -> dict:
        """Fetch the current page rows for columns [col_start, col_start+count)."""
        page = self.table_page
        max_rows = max(1, page["stop"] - page["start"])
        layout = self._data_layout
        if layout is not None and len(layout.shape) >= 1:
            probe = layout.copy_with(row_start=page["start"], col_start=col_start)
            return self.browser.preview(self.selected_path, max_rows=max_rows, max_cols=count, layout=probe)
        return self.browser.preview(
            self.selected_path,
            start=page["start"],
            stop=page["stop"],
            max_cols=count,
            col_start=col_start,
        )

    def _fit_col_start_backward(self, end: int) -> int:
        """Start of the widest whole-column window ending just before *end*."""
        page = self.table_page
        avail = self._col_avail_width(page["nrows"])
        if avail <= 0:
            return max(0, end - max(1, self._col_page_size()))
        candidate = min(end, max(1, avail // (1 + self._CELL_PAD)))
        cand_start = end - candidate
        widths = self._measure_column_widths(self._fetch_columns_for_measure(cand_start, candidate))
        start = end - 1  # always keep at least one column
        total = widths[-1]
        for i in range(len(widths) - 2, -1, -1):
            if total + widths[i] > avail:
                break
            total += widths[i]
            start = cand_start + i
        return max(0, start)

    def _table_page_size(self) -> int:
        table = self.query_one("#data-table", DataTable)
        # Keep only rows likely to be visible.  The DataTable header consumes one
        # line; fall back to the CLI limit before layout has assigned sizes.
        height = table.size.height
        if height <= 1:
            height = self.query_one("#data-pane", Vertical).size.height - 2
        return max(1, height - 1) if height > 1 else max(1, self.preview_rows)

    def _load_table_page(self, path: str, start: int) -> dict:
        if self.browser is None:
            raise RuntimeError("Store browser is not open")
        page_size = self._table_page_size()
        start = max(0, start)
        layout = self._data_layout

        if self.table_buffer is not None:
            buffer_start = self.table_buffer["start"]
            buffer_stop = self.table_buffer["stop"]
            buffer_kind = self.table_buffer.get("source_kind")
            if buffer_kind in {"ndarray2d", "ndarray_slice"}:
                same_columns = self.table_buffer.get(
                    "col_start"
                ) == self.grid_col_start and self.table_buffer.get("slice_indices") == (
                    [
                        layout.fixed_values.get(i, 0)
                        for i in range(len(layout.shape))
                        if i in layout.fixed_values
                    ]
                    if layout is not None
                    else []
                )
            elif buffer_kind == "ctable":
                same_columns = self.table_buffer.get("col_start") == self.grid_col_start
            else:
                same_columns = True
            if same_columns and buffer_start <= start and start + page_size <= buffer_stop:
                data = self._slice_table_buffer(start, page_size)
                self.table_page = data
                return data

        buffer_size = page_size * 10
        buffer_start = max(0, start - page_size * 4)

        if layout is not None and len(layout.shape) >= 1:
            # Use the layout-based preview for all array types (1D+)
            # Scalar view (0 navigable dims) always starts at 0
            if not layout.navigable_dims:
                start = 0
            self._sync_layout_scroll(start, layout)
            data = self.browser.preview(
                path,
                max_rows=buffer_size,
                max_cols=self._candidate_max_cols(),
                layout=layout,
            )
        else:
            # CTable or non-array objects — use legacy preview
            data = self.browser.preview(
                path,
                start=buffer_start,
                stop=buffer_start + buffer_size,
                max_rows=buffer_size,
                max_cols=self._candidate_max_cols(),
                col_start=self.grid_col_start,
            )
        data = self._trim_columns_to_fit(data)
        data["viewport_width"] = self._data_table_width()
        self.table_buffer = data
        data = self._slice_table_buffer(start, page_size)
        self.table_page = data
        return data

    def _sync_layout_scroll(self, start: int, layout: DataSliceLayout) -> None:
        """Update the layout's row/col scroll positions to match the page start."""
        if layout is None:
            return
        navigable = layout.navigable_dims
        if len(navigable) >= 1:
            row_dim = navigable[0]
            total = layout.shape[row_dim]
            layout.row_start = max(0, min(start, total))
            layout.row_stop = min(layout.row_start + self._table_page_size() * 10, total)
        if len(navigable) >= 2:
            col_dim = navigable[1]
            total = layout.shape[col_dim]
            layout.col_start = max(0, min(self.grid_col_start, total))
            layout.col_stop = min(layout.col_start + self._col_page_size(), total)

    def _slice_table_buffer(self, start: int, page_size: int) -> dict:
        if self.table_buffer is None:
            raise RuntimeError("No table buffer loaded")
        buffer = self.table_buffer
        offset = start - buffer["start"]
        available = max(0, buffer["stop"] - start)
        count = min(page_size, available)
        stop = start + count
        return {
            "start": start,
            "stop": stop,
            "nrows": buffer["nrows"],
            "columns": buffer["columns"],
            "hidden_columns": buffer["hidden_columns"],
            "data": {name: values[offset : offset + count] for name, values in buffer["data"].items()},
            **{
                key: buffer[key]
                for key in (
                    "source_kind",
                    "shape",
                    "col_start",
                    "col_stop",
                    "ncols",
                    "slice_indices",
                    "n_slices_per_dim",
                    "viewport_width",
                )
                if key in buffer
            },
        }

    def _update_data_table(self, data: dict, *, cursor_row: int = 0, cursor_col: int | None = None) -> None:
        """Refresh the data grid; *cursor_col* None keeps the current column."""
        table = self.query_one("#data-table", DataTable)
        if cursor_col is None:
            cursor_col = table.cursor_column
        self.loading_table_page = True
        try:
            table.clear(columns=True)
            for name in data["columns"]:
                table.add_column(name, key=name)
            # Uniform decimals per float column, taken from the whole buffer
            # when available so the format is stable while paging rows.
            buffer = self.table_buffer
            source = buffer if buffer is not None and buffer["columns"] == data["columns"] else data
            decimals = {name: column_float_decimals(source["data"][name]) for name in data["columns"]}
            nrows = data["stop"] - data["start"]
            for i in range(nrows):
                table.add_row(
                    *[
                        format_cell(data["data"][name][i], float_decimals=decimals[name])
                        for name in data["columns"]
                    ],
                    label=str(data["start"] + i),
                )
            nrows = data["stop"] - data["start"]
            cursor_row = min(max(0, cursor_row), max(0, nrows - 1))
            cursor_col = min(max(0, cursor_col), max(0, len(data["columns"]) - 1))
            table.cursor_coordinate = (cursor_row, cursor_col)
            table.scroll_home(animate=False)
            self._update_global_row_scrollbar(data)
            self._update_global_col_scrollbar(data)
        finally:
            self.call_after_refresh(self._finish_table_page_load)

    def _finish_table_page_load(self) -> None:
        self.loading_table_page = False

    def page_table(self, direction: int, *, align: bool = False) -> bool:
        if self.loading_table_page or self.table_page is None:
            return False
        page = self.table_page
        page_size = self._table_page_size()
        if direction > 0:
            if page["stop"] >= page["nrows"]:
                return False
            # An explicit page down re-aligns to the page grid: dim-mode
            # single-row scrolls (_scroll_navigable_viewport) can leave `start`
            # off a page_size boundary, and contiguous paging from `stop` would
            # carry that offset forever.  Snapping to the next page_size
            # multiple mirrors how column paging re-fits on each page.  For an
            # already-aligned page this equals `stop`, so cursor-edge paging
            # (align=False) is unchanged.
            new_start = (page["start"] // page_size + 1) * page_size if align else page["stop"]
            data = self._load_table_page(self.selected_path, new_start)
            cursor_row = 0
        else:
            if page["start"] <= 0:
                return False
            if align:
                # Previous grid line: floor for an off-grid start, start-page
                # for an aligned one (ceil-div keeps aligned pages contiguous).
                new_start = (-(-page["start"] // page_size) - 1) * page_size
            else:
                new_start = page["start"] - page_size
            new_start = max(0, new_start)
            data = self._load_table_page(self.selected_path, new_start)
            cursor_row = data["stop"] - data["start"] - 1
        self._update_data_table(data, cursor_row=cursor_row)
        self._update_data_header(data)
        return True

    def page_grid_columns(self, direction: int) -> bool:
        if self.loading_table_page or self.table_page is None:
            return False
        page = self.table_page
        if page.get("source_kind") not in _COL_PAGED_KINDS:
            return False
        # Whole-column windows of data-dependent size: paging right starts at
        # the first hidden column; paging left fits as many whole columns as
        # possible ending just before the current first one (no skips, no gaps).
        if direction > 0:
            if page["col_stop"] >= page["ncols"]:
                return False
            self.grid_col_start = page["col_stop"]
        else:
            if page["col_start"] <= 0:
                return False
            self.grid_col_start = self._fit_col_start_backward(page["col_start"])
        self.table_buffer = None
        data = self._load_table_page(self.selected_path, page["start"])
        cursor_row = self.query_one("#data-table", DataTable).cursor_row
        cursor_col = 0 if direction > 0 else len(data["columns"]) - 1
        self._update_data_table(data, cursor_row=cursor_row, cursor_col=cursor_col)
        self._update_data_header(data)
        return True

    def _grid_col_home(self) -> bool:
        if self.table_page is None or self.table_page.get("source_kind") not in _COL_PAGED_KINDS:
            return False
        self.grid_col_start = 0
        self.table_buffer = None
        data = self._load_table_page(self.selected_path, self.table_page["start"])
        cursor_row = self.query_one("#data-table", DataTable).cursor_row
        self._update_data_table(data, cursor_row=cursor_row, cursor_col=0)
        self._update_data_header(data)
        return True

    def _grid_col_end(self) -> bool:
        if self.table_page is None or self.table_page.get("source_kind") not in _COL_PAGED_KINDS:
            return False
        page = self.table_page
        # Jump to the widest whole-column window ending at the last column
        self.grid_col_start = self._fit_col_start_backward(page["ncols"])
        self.table_buffer = None
        data = self._load_table_page(self.selected_path, page["start"])
        cursor_row = self.query_one("#data-table", DataTable).cursor_row
        self._update_data_table(data, cursor_row=cursor_row, cursor_col=len(data["columns"]) - 1)
        self._update_data_header(data)
        return True

    def _update_data_header(self, data: dict) -> None:
        layout = self._data_layout
        header_parts: list[str] = []

        if layout is not None and len(layout.shape) >= 1:
            ndim = len(layout.shape)
            for i in range(ndim):
                is_active = i == self._active_dim

                if i in layout.fixed_values:
                    idx = layout.fixed_values[i]
                    part = f"d{i} [{idx}]"
                elif i in layout.navigable_dims:
                    pos = layout.navigable_dims.index(i)
                    if pos == 0:
                        s, e = data["start"], data["stop"]
                    else:
                        s, e = data.get("col_start", 0), data.get("col_stop", 0)
                    part = f"d{i}[{s}:{e}]"
                else:
                    part = f"d{i} ?"

                if is_active and self._dim_mode:
                    part = f"[bold]{part}[/bold]"
                header_parts.append(part)

            if self._dim_mode:
                header_parts.append("[reverse] DIM MODE [/reverse]")
                header_parts.append("←→dim  ↑↓val  <Enter>fix/nav  <Esc>exit")
        else:
            header_parts.append(f"rows {data['start']}:{data['stop']} of {data['nrows']}")
            if "col_start" in data:
                header_parts.append(f"cols {data['col_start']}:{data['col_stop']} of {data['ncols']}")
            header_parts.extend(self._window_and_filter_chips(data))

        line = ", ".join(header_parts)
        if self._dim_mode and layout is not None:
            line = f"[reverse]{line}[/reverse]"
        self.query_one("#data-header", Static).update(line)

    def _window_and_filter_chips(self, data: dict) -> list[str]:
        """Header chips for a locked row window and any active CTable filters."""
        chips: list[str] = []
        if self.row_window is not None:
            ws, we = self.row_window
            chips.append(f"[reverse] WINDOW {ws}:{we} [/reverse]")
        if data.get("source_kind") == "ctable" and self.browser is not None:
            flt = self.browser.get_filter(self.selected_path)
            col_flt = self.browser.get_column_filter(self.selected_path)
            if flt:
                total = self.browser.base_nrows(self.selected_path)
                chips.append(f"filter: [bold]{markup_escape(flt)}[/bold] ({total} total)")
            if col_flt:
                total_cols = self.browser.base_ncols(self.selected_path)
                chips.append(f"cols: [bold]{markup_escape(col_flt)}[/bold] ({total_cols} total)")
            if flt or col_flt or self.row_window is not None:
                chips.append("<Esc>unlock/clear")
        return chips

    def _make_global_scrollbar(self, *, start: int, stop: int, total: int, size: int, track: str) -> str:
        size = max(1, size)
        total = max(1, total)
        start = min(max(0, start), total)
        stop = min(max(start, stop), total)
        visible = max(1, stop - start)
        thumb_size = max(1, round(size * min(1.0, visible / total)))
        if total <= visible:
            thumb_start = 0
            thumb_size = size
        else:
            thumb_start = round((size - thumb_size) * (start / (total - visible)))
        thumb_stop = min(size, thumb_start + thumb_size)
        return "".join("█" if thumb_start <= i < thumb_stop else track for i in range(size))

    def _update_global_row_scrollbar(self, data: dict) -> None:
        scrollbar = self.query_one("#row-scrollbar", Static)
        height = max(1, self.query_one("#data-table", DataTable).size.height)
        bar = self._make_global_scrollbar(
            start=int(data["start"]),
            stop=int(data["stop"]),
            total=int(data["nrows"]),
            size=height,
            track="│",
        )
        scrollbar.update("\n".join(bar))

    def _update_global_col_scrollbar(self, data: dict) -> None:
        scrollbar = self.query_one("#col-scrollbar", Static)
        if data.get("source_kind") not in _COL_PAGED_KINDS:
            scrollbar.display = False
            scrollbar.update("")
            return
        scrollbar.display = True
        width = max(1, self.query_one("#data-table", DataTable).size.width)
        scrollbar.update(
            self._make_global_scrollbar(
                start=int(data["col_start"]),
                stop=int(data["col_stop"]),
                total=int(data["ncols"]),
                size=width,
                track="─",
            )
        )

    def _reset_panel_scroll(self) -> None:
        for selector in ("#meta-scroll", "#data-scroll"):
            self.query_one(selector, VerticalScroll).scroll_home(animate=False)
        data_table_row = self.query_one("#data-table-row", Horizontal)
        if data_table_row.display:
            self.query_one("#data-table", DataTable).scroll_home(animate=False)
            if self.table_page is not None:
                self._update_global_row_scrollbar(self.table_page)
                self._update_global_col_scrollbar(self.table_page)

    def _focusable_panels(self):
        data_table_row = self.query_one("#data-table-row", Horizontal)
        data_panel = (
            self.query_one("#data-table", DataTable)
            if data_table_row.display
            else self.query_one("#data-scroll", VerticalScroll)
        )
        return [
            self.query_one("#tree", Tree),
            self.query_one("#meta-scroll", VerticalScroll),
            self.query_one("#vlmeta-scroll", VerticalScroll),
            data_panel,
        ]

    def _focus_panel(self, step: int) -> None:
        panels = self._focusable_panels()
        focused = self.focused
        try:
            index = panels.index(focused)
        except ValueError:
            index = 0 if step > 0 else len(panels) - 1
        panels[(index + step) % len(panels)].focus()

    def action_focus_next_panel(self) -> None:
        self._focus_panel(1)

    def action_focus_previous_panel(self) -> None:
        self._focus_panel(-1)

    def _in_data_grid(self) -> bool:
        """Return True if focus is inside the data pane and a grid is active."""
        if self.table_page is None:
            return False
        if not self.query_one("#data-table-row", Horizontal).display:
            return False
        focused = self.focused
        if focused is None:
            return False
        pane = self.query_one("#data-pane", Vertical)
        return focused is pane or pane in focused.ancestors

    def action_go_to_row(self) -> None:
        if not self._in_data_grid():
            return
        current = self.table_page["start"] + self.query_one("#data-table", DataTable).cursor_row
        screen = GoToRowScreen(nrows=self.table_page["nrows"], current=current)
        self.push_screen(screen, self._go_to_row)

    _PLOT_MAX_POINTS = 2000

    def action_plot_column(self) -> None:
        """p key — plot a downsampled overview of the whole cursor column."""
        if not self._in_data_grid():
            return
        if PlotextPlot is None:
            self.notify("Plotting needs the 'textual-plotext' package", severity="warning")
            return
        buffer = self.table_buffer or self.table_page
        columns = buffer["columns"]
        if not columns:
            return
        cursor_col = self.query_one("#data-table", DataTable).cursor_column
        name = columns[min(max(0, cursor_col), len(columns) - 1)]
        # Cheap numeric check on the already-loaded buffer; this also rejects
        # expensive object columns before any whole-column strided read.
        sample = np.asarray(buffer["data"][name])
        if sample.dtype.kind not in "iufb":
            self.notify(f"Column {name!r} is not numeric", severity="warning")
            return

        column: str | int | None
        if buffer.get("source_kind") == "ctable":
            column = name
        elif name.isdigit():  # array grids label columns with global indices
            column = int(name)
        else:  # 1-D arrays (single navigable dim) have one "value" column
            column = None

        layout = self._data_layout

        def fetch(start: int, stop: int | None) -> dict:
            return self.browser.plot_series(
                self.selected_path,
                column=column,
                layout=layout,
                max_points=self._PLOT_MAX_POINTS,
                row_start=start,
                row_stop=stop,
            )

        series = fetch(0, None)  # whole series (uses the fast SUMMARY tier if any)
        x, _ymin, _ymax, _descr = _plot_view(series)
        if x.size == 0:
            self.notify(f"Column {name!r} has no finite values to plot", severity="warning")
            return
        self.push_screen(
            PlotScreen(
                title_prefix=f"{self.selected_path} · {name}",
                fetch=fetch,
                n=series["n"],
                row_start=series["row_start"],
                row_stop=series["row_stop"],
                series=series,
            ),
            self._view_plot_range,
        )

    def _view_plot_range(self, span: tuple[int, int] | None) -> None:
        """Lock the data grid to a row range chosen with 'v' in the plot modal.

        For CTable nodes the grid is replaced in place with a zero-copy
        ``slice`` view of the range, so paging cannot leave it (``esc`` unlocks).
        Other source kinds (e.g. plain NDArrays) fall back to a cursor jump until
        their windowing lands.
        """
        if span is None or self.table_page is None:
            return
        start, stop = span
        if self.table_page.get("source_kind") == "ctable" and self.browser is not None:
            self._enter_row_window(start, stop)
        else:
            self._go_to_row(start)
            self.notify(f"Viewing rows {start}:{stop}")

    def _enter_row_window(self, start: int, stop: int) -> None:
        """Replace the CTable grid with a locked [start:stop] window view."""
        try:
            self.browser.set_row_window(self.selected_path, start, stop)
        except Exception as exc:  # pragma: no cover - defensive
            self.notify(f"Could not lock rows: {exc}", severity="error")
            return
        self.row_window = (start, stop)
        self.table_buffer = None
        data = self._load_table_page(self.selected_path, 0)
        self._update_data_table(data, cursor_row=0, cursor_col=0)
        self._update_data_header(data)
        self.query_one("#data-table", DataTable).focus()
        self.notify(f"Locked to rows {start}:{stop} · esc to unlock")

    def _exit_row_window(self) -> None:
        """Unlock the row window and restore the full CTable grid."""
        if self.row_window is None or self.browser is None:
            return
        self.browser.clear_row_window(self.selected_path)
        self.row_window = None
        self.table_buffer = None
        data = self._load_table_page(self.selected_path, 0)
        self._update_data_table(data, cursor_row=0, cursor_col=0)
        self._update_data_header(data)
        self.query_one("#data-table", DataTable).focus()

    def action_go_to_column(self) -> None:
        if not self._in_data_grid():
            return
        page = self.table_page
        if page.get("source_kind") not in _COL_PAGED_KINDS:
            return
        current = page["col_start"] + self.query_one("#data-table", DataTable).cursor_column
        names = self.browser.column_names(self.selected_path) if page["source_kind"] == "ctable" else None
        screen = GoToColumnScreen(ncols=page["ncols"], current=current, names=names)
        self.push_screen(screen, self._go_to_column)

    def action_filter_rows(self) -> None:
        if not self._in_data_grid():
            return
        if self.table_page.get("source_kind") != "ctable":
            self.notify("Filtering is only supported for CTable nodes", severity="warning")
            return
        screen = FilterScreen(current=self.browser.get_filter(self.selected_path))
        self.push_screen(screen, self._apply_filter)

    def _apply_filter(self, expr: str | None) -> None:
        if expr is None or self.browser is None or self.table_page is None:
            return
        if expr == (self.browser.get_filter(self.selected_path) or ""):
            return
        try:
            self.browser.set_filter(self.selected_path, expr)
        except Exception as exc:
            self.notify(f"Invalid filter: {exc}", severity="error")
            return
        self.table_buffer = None
        data = self._load_table_page(self.selected_path, 0)
        self._update_data_table(data, cursor_row=0, cursor_col=0)
        self._update_data_header(data)
        self.query_one("#data-table", DataTable).focus()

    def action_filter_columns(self) -> None:
        if not self._in_data_grid():
            return
        if self.table_page.get("source_kind") != "ctable":
            self.notify("Column filtering is only supported for CTable nodes", severity="warning")
            return
        screen = FilterScreen(
            current=self.browser.get_column_filter(self.selected_path),
            title="Filter columns by substring (empty clears)",
            placeholder="e.g. payment",
        )
        self.push_screen(screen, self._apply_column_filter)

    def _apply_column_filter(self, pattern: str | None) -> None:
        if pattern is None or self.browser is None or self.table_page is None:
            return
        if pattern == (self.browser.get_column_filter(self.selected_path) or ""):
            return
        try:
            self.browser.set_column_filter(self.selected_path, pattern)
        except Exception as exc:
            self.notify(f"Invalid column filter: {exc}", severity="error")
            return
        self.grid_col_start = 0
        self.table_buffer = None
        data = self._load_table_page(self.selected_path, self.table_page["start"])
        cursor_row = self.query_one("#data-table", DataTable).cursor_row
        self._update_data_table(data, cursor_row=cursor_row, cursor_col=0)
        self._update_data_header(data)
        self.query_one("#data-table", DataTable).focus()

    def _go_to_column(self, col: int | None) -> None:
        if col is None or self.table_page is None:
            return
        self.grid_col_start = col
        self.table_buffer = None
        data = self._load_table_page(self.selected_path, self.table_page["start"])
        cursor_row = self.query_one("#data-table", DataTable).cursor_row
        self._update_data_table(data, cursor_row=cursor_row, cursor_col=0)
        self._update_data_header(data)
        self.query_one("#data-table", DataTable).focus()

    def _focused_pane(self):
        focused = self.focused
        if focused is None:
            return None
        for selector in ("#tree-pane", "#meta-pane", "#vlmeta-pane", "#data-pane"):
            pane = self.query_one(selector, Vertical)
            if focused is pane or pane in focused.ancestors:
                return pane
        return None

    def action_maximize_panel(self) -> None:
        pane = self._focused_pane()
        if pane is None:
            self.notify("Focus a pane before maximizing", severity="warning")
            return
        if self.screen.maximize(pane, container=False):
            self.call_after_refresh(self._reload_table_for_current_viewport)

    def action_restore_or_refresh(self) -> None:
        if self.screen.maximized is not None:
            self.screen.maximized = None
            self.call_after_refresh(self._reload_table_for_current_viewport)
            return
        self.action_refresh()

    def _ensure_viewport_consistent(self) -> None:
        """Reload the page if it was sized before the layout had settled.

        The first page of a node may be loaded while the DataTable still has
        no size, in which case the CLI fallbacks (preview_rows/preview_cols)
        determine the window.  Later paging then uses the settled viewport
        sizes, so the windows would drift unless we reload once here.
        """
        page = self.table_page
        if page is None or not self.query_one("#data-table-row", Horizontal).display:
            return
        rows_loaded = page["stop"] - page["start"]
        rows_want = min(self._table_page_size(), page["nrows"] - page["start"])
        cols_ok = True
        if page.get("source_kind") in _COL_PAGED_KINDS:
            # The column window is fitted to the width current at load time
            cols_ok = page.get("viewport_width") == self._data_table_width()
        if rows_loaded == rows_want and cols_ok:
            return
        self._reload_table_for_current_viewport()

    def _on_data_table_resized(self) -> None:
        self.call_after_refresh(self._ensure_viewport_consistent)

    def _reload_table_for_current_viewport(self) -> None:
        """Reload the table page after layout changes such as maximize/restore."""
        if self.table_page is None or not self.query_one("#data-table-row", Horizontal).display:
            return
        current = self.table_page["start"] + self.query_one("#data-table", DataTable).cursor_row
        page_size = self._table_page_size()
        start = (current // page_size) * page_size
        self.table_buffer = None
        data = self._load_table_page(self.selected_path, start)
        self._update_data_table(data, cursor_row=current - data["start"])
        self._update_data_header(data)

    def _go_to_row(self, row: int | None) -> None:
        if row is None or self.table_page is None:
            return
        page_size = self._table_page_size()
        start = (row // page_size) * page_size
        data = self._load_table_page(self.selected_path, start)
        self._update_data_table(data, cursor_row=row - data["start"])
        self._update_data_header(data)
        self.query_one("#data-table", DataTable).focus()

    def action_refresh(self) -> None:
        tree = self.query_one("#tree", Tree)
        node = tree.cursor_node or tree.root
        self.loaded_paths.discard(node.data or "/")
        node.remove_children()
        self.load_children(node)
        self.update_panels(node.data or "/")

    def _adjust_fixed_value(self, direction: int) -> None:
        """Adjust the fixed value of the active dimension (if it is fixed).

        The value clamps at the boundaries (no wrap-around).
        """
        layout = self._data_layout
        if layout is None or self.table_page is None:
            return
        dim = self._active_dim
        if dim not in layout.fixed_values:
            return
        total = layout.shape[dim]
        if total <= 0:
            return
        current = layout.fixed_values[dim]
        new_val = min(max(current + direction, 0), total - 1)
        if new_val == current:
            return
        new_fixed = dict(layout.fixed_values)
        new_fixed[dim] = new_val
        self._data_layout = layout.copy_with(fixed_values=new_fixed)
        self.table_buffer = None
        data = self._load_table_page(self.selected_path, self.table_page["start"])
        cursor_row = self.query_one("#data-table", DataTable).cursor_row
        self._update_data_table(data, cursor_row=cursor_row)
        self._update_data_header(data)

    def _rebuild_layout(self, navigable: list[int]) -> DataSliceLayout:
        """Return a copy of the current layout with the given *navigable* dims.

        All non-navigable dimensions are fixed at their previous value (or 0).
        """
        layout = self._data_layout
        if layout is None:
            raise RuntimeError("No layout available")
        new_fixed: dict[int, int] = {}
        for d in range(len(layout.shape)):
            if d in navigable:
                continue
            if d in layout.fixed_values:
                new_fixed[d] = layout.fixed_values[d]
            else:
                new_fixed[d] = 0
        return layout.copy_with(fixed_values=new_fixed, navigable_dims=navigable)

    def _dim_toggle(self) -> None:
        """: key — toggle active dim between fixed and navigable."""
        layout = self._data_layout
        if layout is None or self.table_page is None:
            return
        dim = self._active_dim
        if dim not in range(len(layout.shape)):
            return

        if dim in layout.navigable_dims:
            # Navigable → fixed (at index 0)
            new_nav = [d for d in layout.navigable_dims if d != dim]
            self._data_layout = self._rebuild_layout(new_nav)
        elif dim in layout.fixed_values:
            # Fixed → navigable (if room)
            if len(layout.navigable_dims) >= 2:
                self.notify("At most 2 navigable dimensions are allowed")
                return
            new_nav = sorted(layout.navigable_dims + [dim])
            self._data_layout = self._rebuild_layout(new_nav)
        else:
            return

        # Refresh the display (DataTable for 1-2 nav dims, same path for 0)
        self.table_buffer = None
        data = self._load_table_page(self.selected_path, self.table_page["start"])
        cursor_row = self.query_one("#data-table", DataTable).cursor_row
        self._update_data_table(data, cursor_row=cursor_row)
        self._update_data_header(data)

    def _dim_cursor(self, direction: int) -> None:
        """In dim mode, move the active dimension up (+1) or down (-1)."""
        layout = self._data_layout
        if layout is None or len(layout.shape) < 1:
            return
        ndim = len(layout.shape)
        self._active_dim = (self._active_dim + direction) % ndim
        if self.table_page is not None:
            self._update_data_header(self.table_page)

    def _dim_adjust(self, direction: int) -> None:
        """In DIM mode, adjust the active dim: fixed value or navigable viewport."""
        layout = self._data_layout
        if layout is None or self.table_page is None:
            return
        dim = self._active_dim
        if dim in layout.fixed_values:
            self._adjust_fixed_value(direction)
        elif dim in layout.navigable_dims:
            self._scroll_navigable_viewport(direction)

    def _scroll_navigable_viewport(self, direction: int) -> None:
        """Shift the viewport of a navigable dimension by one step (clamps)."""
        layout = self._data_layout
        if layout is None or self.table_page is None:
            return
        dim = self._active_dim
        if dim not in layout.navigable_dims:
            return

        pos = layout.navigable_dims.index(dim)
        page = self.table_page
        total = layout.shape[dim]

        if pos == 0:
            # Row navigable dim — shift start by one row, keeping a full page
            max_start = max(0, total - self._table_page_size())
            new_start = min(max(page["start"] + direction, 0), max_start)
            if new_start == page["start"]:
                return
            self.table_buffer = None
            data = self._load_table_page(self.selected_path, new_start)
        else:
            # Column navigable dim — shift col_start by one whole column
            max_col = self._fit_col_start_backward(total)
            new_col = min(max(page["col_start"] + direction, 0), max_col)
            if new_col == page["col_start"]:
                return
            self.grid_col_start = new_col
            self.table_buffer = None
            data = self._load_table_page(self.selected_path, page["start"])

        self._update_data_table(data)
        self._update_data_header(data)

    def action_dim_cycle(self) -> None:
        """d key — toggle DIM mode on/off."""
        if not self._in_data_grid():
            return
        layout = self._data_layout
        if layout is None or len(layout.shape) < 1:
            self.notify("No dimensions to navigate")
            return

        self._dim_mode = not self._dim_mode
        if self.table_page is not None:
            self._update_data_header(self.table_page)

    def action_dim_toggle_nav(self) -> None:
        """Enter — toggle active dim between fixed and navigable (in dim mode)."""
        if not self._in_data_grid() or not self._dim_mode:
            return
        self._dim_toggle()

    def action_dim_exit(self) -> None:
        """Escape: exit dim mode, unlock a row window, or clear a CTable filter.

        One layer per press: dim mode, then the locked row window, then the
        row filter, then the column filter.
        """
        if self._dim_mode:
            self._dim_mode = False
            if self.table_page is not None:
                self._update_data_header(self.table_page)
            return
        if self.row_window is not None:
            self._exit_row_window()
            return
        if (
            not self._in_data_grid()
            or self.table_page.get("source_kind") != "ctable"
            or self.browser is None
        ):
            return
        if self.browser.get_filter(self.selected_path):
            self._apply_filter("")
        elif self.browser.get_column_filter(self.selected_path):
            self._apply_column_filter("")

    def action_grid_row_top(self) -> None:
        """Jump to the first row of the table."""
        if not self._in_data_grid():
            return
        self._go_to_row(0)

    def action_grid_row_bottom(self) -> None:
        """Jump to the last row of the table."""
        if not self._in_data_grid():
            return
        self._go_to_row(self.table_page["nrows"] - 1)

    def action_show_help(self) -> None:
        self.push_screen(HelpScreen())

    def action_grid_col_start(self) -> None:
        """Jump to the first column window (alias of Home)."""
        if self._in_data_grid():
            self._grid_col_home()

    def action_grid_col_end(self) -> None:
        """Jump to the last column window (alias of End)."""
        if self._in_data_grid():
            self._grid_col_end()
