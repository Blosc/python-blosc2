"""A non-destructive discovered-node filter and explicit recursive search."""

import threading
from dataclasses import dataclass
from typing import ClassVar

from rich.text import Text
from textual import work
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Static, Tree

from blosc2.b2view.file_preview import safe_text
from blosc2.b2view.tree_search import MAX_DEPTH, MAX_DIRECTORIES, MAX_NODES, matching_nodes, search_tree


@dataclass
class SearchSelection:
    path: str


class TreeSearchScreen(ModalScreen):
    """Leave the main tree's selection/expansion untouched unless a result is chosen."""

    DEFAULT_CSS = """
    TreeSearchScreen { align: center middle; }
    #tree-search-dialog { width: 85%; height: 85%; border: round $accent; padding: 1 2; }
    #tree-search-status { height: auto; }
    #tree-search-results { height: 1fr; }
    #tree-search-buttons { height: 3; }
    """
    BINDINGS: ClassVar = [("escape", "cancel", "Close")]

    def __init__(self, browser, listings, session):
        super().__init__()
        self.browser = browser
        self.listings = dict(listings)
        self.session = session
        self.cancelled = threading.Event()
        self.searching = False
        self.mode = "Discovered nodes only (no network requests)"
        self.matches = set()

    def compose(self):
        with Vertical(id="tree-search-dialog"):
            yield Static("Find by name/path — case-insensitive substring", markup=False)
            yield Input(placeholder="Filter discovered nodes…", id="tree-search-query")
            yield Static(self.mode, id="tree-search-status", markup=False)
            yield Tree("Matches", id="tree-search-results")
            with Horizontal(id="tree-search-buttons"):
                yield Button("Search recursively", id="tree-search-recursive", variant="primary")
                yield Button("Close", id="tree-search-close")

    def on_mount(self):
        self.query_one(Input).focus()
        self._filter()

    def on_input_changed(self, event: Input.Changed):
        self._filter()

    def _filter(self):
        if not self.is_mounted:
            return
        matches = matching_nodes(self.listings, self.query_one(Input).value)
        self.matches = {node.path for node in matches}
        visible = matches[:500]
        tree = self.query_one(Tree)
        tree.clear()
        ancestors = {"/": tree.root}
        for node in visible:
            parts = node.path.strip("/").split("/")
            path = ""
            parent = tree.root
            for part in parts:
                path += "/" + part
                if path not in ancestors:
                    ancestors[path] = parent.add(Text(safe_text(part)), data=path)
                parent = ancestors[path]
        for node in ancestors.values():
            node.allow_expand = bool(node.children)
            node.expand()
        count = f"{len(matches)} matches"
        if len(visible) < len(matches):
            count += "; showing first 500 — refine the filter"
        self.query_one("#tree-search-status", Static).update(f"{self.mode}\n{count}")

    def on_input_submitted(self):
        self.query_one(Tree).focus()

    def on_tree_node_selected(self, event: Tree.NodeSelected):
        event.stop()
        if event.node.data in self.matches and not self.searching:
            self.dismiss(SearchSelection(event.node.data))

    def on_tree_node_expanded(self, event: Tree.NodeExpanded):
        # This is a filtered mirror, not the app's navigation tree. Expanding
        # its ancestors must never trigger background discovery in the app.
        event.stop()

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "tree-search-close":
            self.action_cancel()
        elif self.searching:
            self.cancelled.set()
        else:
            self.cancelled.clear()
            self.searching = True
            event.button.label = "Cancel search"
            self.mode = f"Searching listings/metadata (limits: {MAX_DIRECTORIES} directories, {MAX_NODES:,} nodes, depth {MAX_DEPTH})"
            self._filter()
            self._search()

    def _obsolete(self):
        return (
            self.cancelled.is_set()
            or self.app._closing
            or self.session != self.app._remote_session
            or self.browser is not self.app.browser
        )

    @work(thread=True, exit_on_error=False)
    def _search(self):
        def progress(report):
            # Avoid rebuilding hundreds of result-tree nodes for every fast
            # directory listing. Always deliver the final complete snapshot.
            if report.directories != 1 and report.directories % 10:
                return
            self.app.call_from_thread(
                self._update_search,
                dict(report.listings),
                f"Searching… {report.directories} directories",
                False,
            )

        report = search_tree(self.browser, self.listings, cancel=self._obsolete, progress=progress)
        message = f"{report.status}: {report.directories} directories; {len(report.errors)} listing errors"
        if report.errors:
            message += "\n" + self.app._error_message(report.errors[-1][1])
        self.app.call_from_thread(self._update_search, report.listings, message, True)

    def _update_search(self, listings, message, finished):
        if (
            not self.is_mounted
            or self.session != self.app._remote_session
            or self.browser is not self.app.browser
        ):
            return
        self.listings = listings
        self.mode = message
        if finished:
            self.searching = False
            self.query_one("#tree-search-recursive", Button).label = "Search recursively"
        self._filter()

    def action_cancel(self):
        self.cancelled.set()
        self.dismiss(None)

    def on_unmount(self):
        self.cancelled.set()
