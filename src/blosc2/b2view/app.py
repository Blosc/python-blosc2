"""Textual application for b2view."""

from __future__ import annotations

from typing import ClassVar

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import DataTable, Footer, Header, Input, Static, Tree

from blosc2.b2view.model import StoreBrowser
from blosc2.b2view.render import format_cell, make_metadata_renderable, make_preview_renderables

_KIND_ICONS = {
    "group": "📁",
    "ndarray": "▦",
    "c2array": "▦",
    "ctable": "▤",
    "schunk": "▣",
    "unknown": "?",
}


class B2ViewPanel(Vertical):
    """Pane container that can be maximized."""

    ALLOW_MAXIMIZE = True


class BufferedDataTable(DataTable):
    """DataTable with app-controlled page changes at row boundaries."""

    def action_cursor_down(self) -> None:
        if self.cursor_row >= self.row_count - 1 and getattr(self.app, "page_table", lambda _: False)(1):
            return
        super().action_cursor_down()

    def action_cursor_up(self) -> None:
        if self.cursor_row <= 0 and getattr(self.app, "page_table", lambda _: False)(-1):
            return
        super().action_cursor_up()

    def action_cursor_right(self) -> None:
        if self.cursor_column >= len(self.columns) - 1 and getattr(
            self.app, "page_grid_columns", lambda _: False
        )(1):
            return
        super().action_cursor_right()

    def action_cursor_left(self) -> None:
        if self.cursor_column <= 0 and getattr(self.app, "page_grid_columns", lambda _: False)(-1):
            return
        super().action_cursor_left()

    def action_page_down(self) -> None:
        if getattr(self.app, "page_table", lambda _: False)(1):
            return
        super().action_page_down()

    def action_page_up(self) -> None:
        if getattr(self.app, "page_table", lambda _: False)(-1):
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


class B2ViewApp(App):
    """Browse TreeStore hierarchy and preview objects."""

    CSS = """
    #main { height: 1fr; }
    #tree-pane { width: 35%; border: solid $primary; }
    #right-pane { width: 65%; }
    #meta-pane { height: 40%; border: solid $secondary; }
    #data-pane { height: 60%; border: solid $secondary; }
    #tree { height: 1fr; }
    #data-header { height: auto; padding: 0 1; }
    #data-table-row { height: 1fr; }
    #data-table { width: 1fr; height: 1fr; }
    #row-scrollbar { width: 1; height: 1fr; color: $accent; }
    #col-scrollbar { height: 1; width: 1fr; color: $accent; }
    #meta-scroll, #data-scroll { height: 1fr; padding: 0 1; }
    #tree-pane:focus-within, #meta-pane:focus-within, #data-pane:focus-within { border: heavy $accent; }
    B2ViewPanel.-maximized,
    #tree-pane.-maximized,
    #meta-pane.-maximized,
    #data-pane.-maximized { width: 1fr; height: 1fr; }
    """

    BINDINGS: ClassVar = [
        ("q", "quit", "Quit"),
        ("tab", "focus_next_panel", "Next panel"),
        ("shift+tab", "focus_previous_panel", "Previous panel"),
        Binding("g", "go_to_row", "Go to row", show=False),
        ("m", "maximize_panel", "Maximize"),
        ("r", "restore_or_refresh", "Restore/Refresh"),
        Binding("t", "grid_row_top", "Top", show=False),
        Binding("b", "grid_row_bottom", "Bottom", show=False),
    ]

    def __init__(self, urlpath: str, *, preview_rows: int = 20, preview_cols: int = 10):
        super().__init__()
        self.urlpath = urlpath
        self.preview_rows = preview_rows
        self.preview_cols = preview_cols
        self.browser: StoreBrowser | None = None
        self.loaded_paths: set[str] = set()
        self.selected_path = "/"
        self.table_page: dict | None = None
        self.table_buffer: dict | None = None
        self.grid_col_start = 0
        self.loading_table_page = False

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main"):
            with B2ViewPanel(id="tree-pane") as tree_pane:
                tree_pane.border_title = "tree"
                yield Tree("/", id="tree")
            with Vertical(id="right-pane"):
                with B2ViewPanel(id="meta-pane") as meta_pane:
                    meta_pane.border_title = "meta"
                    with VerticalScroll(id="meta-scroll", can_focus=True):
                        yield Static("Select a node", id="metadata")
                with B2ViewPanel(id="data-pane") as data_pane:
                    data_pane.border_title = "data"
                    data_pane.border_subtitle = "t(op) - b(ottom) - g(oto)"
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
        self.call_after_refresh(self.update_panels, "/")
        tree.focus()

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
        try:
            info = self.browser.get_info(path)
            metadata.update(make_metadata_renderable(info))
            self.table_buffer = None
            self.grid_col_start = 0
            if info.kind == "group":
                data_header.display = False
                data_table_row.display = False
                data_scroll.display = True
                self.query_one("#col-scrollbar", Static).display = False
                data_header.update("")
                preview.update("Group node; select an array or table to preview.")
            else:
                if self._uses_grid_preview(info):
                    data_header.display = True
                    data_table_row.display = True
                    data_scroll.display = False
                    preview.update("")
                    data = self._load_table_page(path, 0)
                else:
                    data = self.browser.preview(path, max_rows=self.preview_rows, max_cols=self.preview_cols)
                if self._is_table_preview(data):
                    self._update_data_table(data)
                    self._update_data_header(data)
                else:
                    header, body = make_preview_renderables(data)
                    data_header.display = header is not None
                    data_table_row.display = False
                    data_scroll.display = True
                    self.query_one("#col-scrollbar", Static).display = False
                    data_header.update("" if header is None else header)
                    preview.update(body)
            self._reset_panel_scroll()
        except Exception as exc:
            metadata.update(f"Error reading {path}: {exc}")
            data_header.display = False
            data_table_row.display = False
            data_scroll.display = True
            self.query_one("#col-scrollbar", Static).display = False
            data_header.update("")
            preview.update("")
            self._reset_panel_scroll()

    @staticmethod
    def _is_table_preview(data) -> bool:
        return isinstance(data, dict) and "data" in data and "columns" in data

    @staticmethod
    def _uses_grid_preview(info) -> bool:
        return info.kind == "ctable" or (
            info.kind in {"ndarray", "c2array"} and info.metadata.get("ndim") in (1, 2)
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
        if self.table_buffer is not None:
            buffer_start = self.table_buffer["start"]
            buffer_stop = self.table_buffer["stop"]
            same_columns = (
                self.table_buffer.get("source_kind") != "ndarray2d"
                or self.table_buffer.get("col_start") == self.grid_col_start
            )
            if same_columns and buffer_start <= start and start + page_size <= buffer_stop:
                data = self._slice_table_buffer(start, page_size)
                self.table_page = data
                return data

        buffer_size = page_size * 10
        # Keep requested page around the middle of the buffer.  This makes both
        # forward and backward page turns fast after a boundary-crossing fetch.
        buffer_start = max(0, start - page_size * 4)
        data = self.browser.preview(
            path,
            start=buffer_start,
            stop=buffer_start + buffer_size,
            max_rows=buffer_size,
            max_cols=self._col_page_size(),
            col_start=self.grid_col_start,
        )
        self.table_buffer = data
        data = self._slice_table_buffer(start, page_size)
        self.table_page = data
        return data

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
                for key in ("source_kind", "shape", "col_start", "col_stop", "ncols")
                if key in buffer
            },
        }

    def _update_data_table(self, data: dict, *, cursor_row: int = 0, cursor_col: int = 0) -> None:
        table = self.query_one("#data-table", DataTable)
        self.loading_table_page = True
        try:
            table.clear(columns=True)
            for name in data["columns"]:
                table.add_column(name, key=name)
            nrows = data["stop"] - data["start"]
            for i in range(nrows):
                table.add_row(
                    *[format_cell(data["data"][name][i]) for name in data["columns"]],
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

    def page_table(self, direction: int) -> bool:
        if self.loading_table_page or self.table_page is None:
            return False
        page = self.table_page
        page_size = self._table_page_size()
        if direction > 0:
            if page["stop"] >= page["nrows"]:
                return False
            data = self._load_table_page(self.selected_path, page["stop"])
            cursor_row = 0
        else:
            if page["start"] <= 0:
                return False
            start = max(0, page["start"] - page_size)
            data = self._load_table_page(self.selected_path, start)
            cursor_row = data["stop"] - data["start"] - 1
        self._update_data_table(data, cursor_row=cursor_row)
        self._update_data_header(data)
        return True

    def page_grid_columns(self, direction: int) -> bool:
        if self.loading_table_page or self.table_page is None:
            return False
        page = self.table_page
        if page.get("source_kind") != "ndarray2d":
            return False
        page_cols = max(1, len(page["columns"]))
        ncols = page["ncols"]
        col_start = page["col_start"]
        if direction > 0:
            if page["col_stop"] >= ncols:
                return False
            self.grid_col_start = min(ncols - 1, col_start + page_cols)
            cursor_col = 0
        else:
            if col_start <= 0:
                return False
            self.grid_col_start = max(0, col_start - page_cols)
            cursor_col = page_cols - 1
        self.table_buffer = None
        data = self._load_table_page(self.selected_path, page["start"])
        cursor_row = self.query_one("#data-table", DataTable).cursor_row
        self._update_data_table(data, cursor_row=cursor_row, cursor_col=cursor_col)
        self._update_data_header(data)
        return True

    def _grid_col_home(self) -> bool:
        if self.table_page is None or self.table_page.get("source_kind") != "ndarray2d":
            return False
        self.grid_col_start = 0
        self.table_buffer = None
        data = self._load_table_page(self.selected_path, self.table_page["start"])
        cursor_row = self.query_one("#data-table", DataTable).cursor_row
        self._update_data_table(data, cursor_row=cursor_row, cursor_col=0)
        self._update_data_header(data)
        return True

    def _grid_col_end(self) -> bool:
        if self.table_page is None or self.table_page.get("source_kind") != "ndarray2d":
            return False
        page = self.table_page
        page_cols = max(1, len(page["columns"]))
        self.grid_col_start = max(0, page["ncols"] - page_cols)
        self.table_buffer = None
        data = self._load_table_page(self.selected_path, page["start"])
        cursor_row = self.query_one("#data-table", DataTable).cursor_row
        self._update_data_table(data, cursor_row=cursor_row, cursor_col=page_cols - 1)
        self._update_data_header(data)
        return True

    def _update_data_header(self, data: dict) -> None:
        header = f"rows {data['start']}:{data['stop']} of {data['nrows']}"
        if data.get("source_kind") == "ndarray2d":
            header += f", cols {data['col_start']}:{data['col_stop']} of {data['ncols']}"
        self.query_one("#data-header", Static).update(header)

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
        if data.get("source_kind") != "ndarray2d":
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

    def action_go_to_row(self) -> None:
        if self.table_page is None or not self.query_one("#data-table-row", Horizontal).display:
            self.notify("Go to row is only available for table previews", severity="warning")
            return
        current = self.table_page["start"] + self.query_one("#data-table", DataTable).cursor_row
        screen = GoToRowScreen(nrows=self.table_page["nrows"], current=current)
        self.push_screen(screen, self._go_to_row)

    def _focused_pane(self):
        focused = self.focused
        if focused is None:
            return None
        for selector in ("#tree-pane", "#meta-pane", "#data-pane"):
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

    def action_grid_row_top(self) -> None:
        """Jump to the first row of the table."""
        if self.table_page is None:
            return
        self._go_to_row(0)

    def action_grid_row_bottom(self) -> None:
        """Jump to the last row of the table."""
        if self.table_page is None:
            return
        self._go_to_row(self.table_page["nrows"] - 1)
