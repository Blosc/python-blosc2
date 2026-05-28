"""Textual application for b2view."""

from __future__ import annotations

from typing import Any, ClassVar

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import DataTable, Footer, Header, Input, Static, Tree

from blosc2.b2view.model import DataSliceLayout, StoreBrowser
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

    def action_select_cursor(self) -> None:
        app = self.app
        if getattr(app, "_dim_mode", False):
            getattr(app, "action_dim_toggle_nav", lambda: None)()
            return
        super().action_select_cursor()

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
        ("tab", "focus_next_panel", "Next panel"),
        ("shift+tab", "focus_previous_panel", "Previous panel"),
        Binding("g", "go_to_row", "Go to row", show=False),
        ("m", "maximize_panel", "Maximize"),
        ("r", "restore_or_refresh", "Restore/Refresh"),
        Binding("t", "grid_row_top", "Top", show=False),
        Binding("b", "grid_row_bottom", "Bottom", show=False),
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
                    data_pane.border_subtitle = "d(im mode) | t(op) - b(ottom) - g(oto)"
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
            same_columns = self.table_buffer.get("source_kind") not in {"ndarray2d", "ndarray_slice"} or (
                self.table_buffer.get("col_start") == self.grid_col_start
                and self.table_buffer.get("slice_indices")
                == (
                    [
                        layout.fixed_values.get(i, 0)
                        for i in range(len(layout.shape))
                        if i in layout.fixed_values
                    ]
                    if layout is not None
                    else []
                )
            )
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
                max_cols=self._col_page_size(),
                layout=layout,
            )
        else:
            # CTable or non-array objects — use legacy preview
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
                )
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
        if page.get("source_kind") not in ("ndarray2d", "ndarray_slice"):
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
        if self.table_page is None or self.table_page.get("source_kind") not in (
            "ndarray2d",
            "ndarray_slice",
        ):
            return False
        self.grid_col_start = 0
        self.table_buffer = None
        data = self._load_table_page(self.selected_path, self.table_page["start"])
        cursor_row = self.query_one("#data-table", DataTable).cursor_row
        self._update_data_table(data, cursor_row=cursor_row, cursor_col=0)
        self._update_data_header(data)
        return True

    def _grid_col_end(self) -> bool:
        if self.table_page is None or self.table_page.get("source_kind") not in (
            "ndarray2d",
            "ndarray_slice",
        ):
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

        line = ", ".join(header_parts)
        if self._dim_mode and layout is not None:
            line = f"[reverse]{line}[/reverse]"
        self.query_one("#data-header", Static).update(line)

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
        if data.get("source_kind") not in ("ndarray2d", "ndarray_slice"):
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

        In DIM mode the value wraps around at boundaries (0 ↔ max).
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
        if self._dim_mode and total > 1:
            # Cycle: up at max → 0, down at 0 → max-1
            new_val = (current + direction) % total
        else:
            # Clamp at boundaries (normal mode)
            if direction > 0:
                if current >= total - 1:
                    return
                new_val = current + 1
            else:
                if current <= 0:
                    return
                new_val = current - 1
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
        """Shift the viewport of a navigable dimension by one step (wraps)."""
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
            # Row navigable dim — shift start by one row (wraps)
            new_start = (page["start"] + direction) % total
            self.table_buffer = None
            data = self._load_table_page(self.selected_path, new_start)
        else:
            # Column navigable dim — shift col_start by one column (wraps)
            new_col = (page["col_start"] + direction) % total
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
        """Escape: exit dim mode."""
        if not self._dim_mode:
            return
        self._dim_mode = False
        if self.table_page is not None:
            self._update_data_header(self.table_page)

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
