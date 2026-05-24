"""Textual application for b2view."""

from __future__ import annotations

from typing import ClassVar

from textual.app import App, ComposeResult
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

    def action_page_down(self) -> None:
        if getattr(self.app, "page_table", lambda _: False)(1):
            return
        super().action_page_down()

    def action_page_up(self) -> None:
        if getattr(self.app, "page_table", lambda _: False)(-1):
            return
        super().action_page_up()


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
    #meta-scroll, #data-scroll { height: 1fr; padding: 0 1; }
    #tree-pane:focus-within, #meta-pane:focus-within, #data-pane:focus-within { border: heavy $accent; }
    """

    BINDINGS: ClassVar = [
        ("q", "quit", "Quit"),
        ("tab", "focus_next_panel", "Next panel"),
        ("shift+tab", "focus_previous_panel", "Previous panel"),
        ("g", "go_to_row", "Go to row"),
        ("r", "refresh", "Refresh"),
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
        self.loading_table_page = False

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main"):
            with Vertical(id="tree-pane") as tree_pane:
                tree_pane.border_title = "tree"
                yield Tree("/", id="tree")
            with Vertical(id="right-pane"):
                with Vertical(id="meta-pane") as meta_pane:
                    meta_pane.border_title = "meta"
                    with VerticalScroll(id="meta-scroll", can_focus=True):
                        yield Static("Select a node", id="metadata")
                with Vertical(id="data-pane") as data_pane:
                    data_pane.border_title = "data"
                    data_pane.border_subtitle = "g(oto)"
                    yield Static("", id="data-header")
                    with Horizontal(id="data-table-row"):
                        yield BufferedDataTable(id="data-table", show_row_labels=True, zebra_stripes=True)
                        yield Static("", id="row-scrollbar")
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
            if info.kind == "group":
                data_header.display = False
                data_table_row.display = False
                data_scroll.display = True
                data_header.update("")
                preview.update("Group node; select an array or table to preview.")
            else:
                if info.kind == "ctable":
                    data_header.display = True
                    data_table_row.display = True
                    data_scroll.display = False
                    preview.update("")
                    data = self._load_table_page(path, 0)
                else:
                    data = self.browser.preview(path, max_rows=self.preview_rows, max_cols=self.preview_cols)
                if self._is_table_preview(data):
                    self._update_data_table(data)
                    data_header.update(f"rows {data['start']}:{data['stop']} of {data['nrows']}")
                else:
                    header, body = make_preview_renderables(data)
                    data_header.display = header is not None
                    data_table_row.display = False
                    data_scroll.display = True
                    data_header.update("" if header is None else header)
                    preview.update(body)
            self._reset_panel_scroll()
        except Exception as exc:
            metadata.update(f"Error reading {path}: {exc}")
            data_header.display = False
            data_table_row.display = False
            data_scroll.display = True
            data_header.update("")
            preview.update("")
            self._reset_panel_scroll()

    @staticmethod
    def _is_table_preview(data) -> bool:
        return isinstance(data, dict) and "data" in data and "columns" in data

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
            if buffer_start <= start and start + page_size <= buffer_stop:
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
            max_cols=self.preview_cols,
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
        }

    def _update_data_table(self, data: dict, *, cursor_row: int = 0) -> None:
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
            table.cursor_coordinate = (cursor_row, 0)
            table.scroll_home(animate=False)
            self._update_global_row_scrollbar(data)
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
        self.query_one("#data-header", Static).update(
            f"rows {data['start']}:{data['stop']} of {data['nrows']}"
        )
        return True

    def _update_global_row_scrollbar(self, data: dict) -> None:
        scrollbar = self.query_one("#row-scrollbar", Static)
        height = max(1, self.query_one("#data-table", DataTable).size.height)
        nrows = max(1, int(data["nrows"]))
        start = min(max(0, int(data["start"])), nrows)
        stop = min(max(start, int(data["stop"])), nrows)
        visible = max(1, stop - start)
        thumb_height = max(1, round(height * min(1.0, visible / nrows)))
        if nrows <= visible:
            thumb_top = 0
            thumb_height = height
        else:
            thumb_top = round((height - thumb_height) * (start / (nrows - visible)))
        thumb_bottom = min(height, thumb_top + thumb_height)
        lines = ["█" if thumb_top <= i < thumb_bottom else "│" for i in range(height)]
        scrollbar.update("\n".join(lines))

    def _reset_panel_scroll(self) -> None:
        for selector in ("#meta-scroll", "#data-scroll"):
            self.query_one(selector, VerticalScroll).scroll_home(animate=False)
        data_table_row = self.query_one("#data-table-row", Horizontal)
        if data_table_row.display:
            self.query_one("#data-table", DataTable).scroll_home(animate=False)
            if self.table_page is not None:
                self._update_global_row_scrollbar(self.table_page)

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

    def _go_to_row(self, row: int | None) -> None:
        if row is None or self.table_page is None:
            return
        page_size = self._table_page_size()
        start = (row // page_size) * page_size
        data = self._load_table_page(self.selected_path, start)
        self._update_data_table(data, cursor_row=row - data["start"])
        self.query_one("#data-header", Static).update(
            f"rows {data['start']}:{data['stop']} of {data['nrows']}"
        )
        self.query_one("#data-table", DataTable).focus()

    def action_refresh(self) -> None:
        tree = self.query_one("#tree", Tree)
        node = tree.cursor_node or tree.root
        self.loaded_paths.discard(node.data or "/")
        node.remove_children()
        self.load_children(node)
        self.update_panels(node.data or "/")
