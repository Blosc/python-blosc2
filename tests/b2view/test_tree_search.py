"""Search uses listings only and preserves the navigation tree on dismissal."""

from threading import RLock

import pytest
from test_remote_caterva2 import caterva2_source  # noqa: F401
from tui_wait import wait_until

from blosc2.b2view.model import NodeInfo
from blosc2.b2view.tree_search import matching_nodes, search_tree


class ListingBrowser:
    def __init__(self):
        self.io_lock = RLock()
        self.calls = []
        self.children = {
            "/": [NodeInfo("/folder", "folder", "group", True), NodeInfo("/bad", "bad", "group", True)],
            "/folder": [NodeInfo("/folder/README.md", "README.md", "file", False)],
            "/bad": OSError("Cannot list directory"),
        }

    def list_children(self, path):
        self.calls.append(path)
        result = self.children[path]
        if isinstance(result, Exception):
            raise result
        return result


def test_discovered_filter_and_listing_only_search():
    browser = ListingBrowser()
    known = {"/": browser.children["/"]}
    assert matching_nodes(known, "README") == []
    assert not browser.calls
    report = search_tree(browser, known, cancel=lambda: False)
    assert browser.calls == ["/folder", "/bad"]
    assert matching_nodes(report.listings, "readme")[0].path == "/folder/README.md"
    assert matching_nodes(report.listings, "FOLDER/")[0].name == "README.md"
    assert report.status == "Completed with listing errors"
    assert len(report.errors) == 1
    assert known == {"/": browser.children["/"]}


@pytest.mark.parametrize(
    ("limits", "status"),
    [
        ({"max_directories": 1}, "Directory limit reached"),
        ({"max_nodes": 2}, "Node limit reached"),
        ({"max_depth": 0}, "Depth limit reached"),
    ],
)
def test_recursive_limits(limits, status):
    browser = ListingBrowser()
    report = search_tree(browser, {}, cancel=lambda: False, **limits)
    assert report.status == status
    assert browser.calls == ["/"]


def test_recursive_cancellation():
    browser = ListingBrowser()
    report = search_tree(browser, {}, cancel=lambda: bool(browser.calls))
    assert report.status == "Cancelled"
    assert browser.calls == ["/"]


@pytest.mark.tui
@pytest.mark.asyncio
async def test_discovered_filter_makes_no_remote_requests_and_preserves_tree(caterva2_source):  # noqa: F811
    from textual.widgets import Input, Tree

    from blosc2.b2view.app import B2ViewApp
    from blosc2.b2view.search_screen import TreeSearchScreen

    base, _, _, stats = caterva2_source
    app = B2ViewApp(base)
    async with app.run_test(size=(120, 40)) as pilot:
        await wait_until(pilot, lambda: app._selected_info is not None)
        tree = app.query_one("#tree", Tree)
        selected = app.selected_path
        children = tuple(tree.root.children)
        before = repr(stats)
        assert "f(ind)" in str(app.query_one("#tree-pane").border_subtitle)
        tree.focus()
        await pilot.press("f")
        await wait_until(pilot, lambda: isinstance(app.screen, TreeSearchScreen))
        app.screen.query_one(Input).value = "mount"
        await pilot.pause()
        assert "/mount" in app.screen.matches
        assert repr(stats) == before
        app.screen.query_one(Input).value = "nonexistent"
        await pilot.pause()
        assert not app.screen.matches
        await pilot.press("escape")
        assert app.selected_path == selected
        assert tuple(tree.root.children) == children
    app.wait_for_close()


@pytest.mark.tui
@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["local", "memory"])
async def test_recursive_search_reveals_result(tmp_path, backend):
    from textual.widgets import Button, Input, Static, Tree

    from blosc2.b2view.app import B2ViewApp
    from blosc2.b2view.search_screen import TreeSearchScreen

    if backend == "memory":
        fs = pytest.importorskip("fsspec").filesystem("memory")
        source = "memory://b2view-search/root"
        fs.pipe(source + "/deep/README.md", b"# Found")
    else:
        (tmp_path / "deep").mkdir()
        (tmp_path / "deep/README.md").write_text("# Found")
        source = str(tmp_path)
    app = B2ViewApp(source)
    async with app.run_test(size=(120, 40)) as pilot:
        await wait_until(pilot, lambda: app._selected_info is not None)
        await pilot.press("ctrl+f")
        await wait_until(pilot, lambda: isinstance(app.screen, TreeSearchScreen))
        screen = app.screen
        screen.query_one(Input).value = "readme"
        await pilot.pause()
        assert not screen.matches
        screen.query_one("#tree-search-recursive", Button).press()
        await wait_until(pilot, lambda: not screen.searching and "/deep/README.md" in screen.matches)
        results = screen.query_one(Tree)
        assert results.root.children[0].data == "/deep"
        result = results.root.children[0].children[0]
        results.select_node(result)
        results.focus()
        await pilot.press("enter")
        await wait_until(pilot, lambda: not isinstance(app.screen, TreeSearchScreen))
        await wait_until(pilot, lambda: app.selected_path == "/deep/README.md")
        await wait_until(
            pilot, lambda: "T: raw/Markdown" in str(app.query_one("#data-header", Static).render())
        )
        assert app._selected_info.kind == "file"
        assert app.query_one("#tree", Tree).has_focus
    app.wait_for_close()


@pytest.mark.tui
@pytest.mark.asyncio
async def test_dataset_search_selection_returns_focus_to_tree(tmp_path):
    import numpy as np
    from textual.widgets import Input, Tree

    import blosc2
    from blosc2.b2view.app import B2ViewApp
    from blosc2.b2view.search_screen import TreeSearchScreen

    blosc2.asarray(np.arange(10), urlpath=str(tmp_path / "array.b2nd"))
    app = B2ViewApp(str(tmp_path))
    async with app.run_test(size=(120, 40)) as pilot:
        await wait_until(pilot, lambda: app._selected_info is not None)
        await pilot.press("ctrl+f")
        await wait_until(pilot, lambda: isinstance(app.screen, TreeSearchScreen))
        app.screen.query_one(Input).value = "array"
        await pilot.pause()
        results = app.screen.query_one(Tree)
        results.select_node(results.root.children[0])
        await wait_until(pilot, lambda: not isinstance(app.screen, TreeSearchScreen))
        await wait_until(pilot, lambda: app.table_page is not None)
        assert app.selected_path == "/array.b2nd"
        assert app.query_one("#tree", Tree).has_focus
    app.wait_for_close()


@pytest.mark.tui
@pytest.mark.asyncio
async def test_recursive_caterva2_search_does_not_fetch_array_values(caterva2_source):  # noqa: F811
    from textual.widgets import Button, Input

    from blosc2.b2view.app import B2ViewApp
    from blosc2.b2view.search_screen import TreeSearchScreen

    base, _, _, stats = caterva2_source
    app = B2ViewApp(base)
    async with app.run_test(size=(120, 40)) as pilot:
        await wait_until(pilot, lambda: app._selected_info is not None)
        await pilot.press("ctrl+f")
        await wait_until(pilot, lambda: isinstance(app.screen, TreeSearchScreen))
        screen = app.screen
        screen.query_one(Input).value = "array"
        screen.query_one("#tree-search-recursive", Button).press()
        await wait_until(pilot, lambda: not screen.searching and "/mount/array" in screen.matches)
        # Table classification may fetch an empty frame for schema discovery.
        # No array values or nonempty table rows may be requested.
        assert all(start == stop == 0 for start, stop in stats["ranges"])
        assert app.selected_path == "/"
        await pilot.press("escape")
        before = repr(stats)
        await pilot.press("ctrl+f")
        await wait_until(pilot, lambda: isinstance(app.screen, TreeSearchScreen))
        app.screen.query_one(Input).value = "array"
        await pilot.pause()
        assert "/mount/array" in app.screen.matches
        assert repr(stats) == before
        await pilot.press("escape")
    app.wait_for_close()


@pytest.mark.tui
@pytest.mark.asyncio
async def test_close_cancels_search_without_waiting_for_listing(tmp_path, monkeypatch):
    import threading

    from textual.widgets import Button

    from blosc2.b2view.app import B2ViewApp
    from blosc2.b2view.search_screen import TreeSearchScreen

    (tmp_path / "deep").mkdir()
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    app = B2ViewApp(str(tmp_path))
    try:
        async with app.run_test(size=(120, 40)) as pilot:
            await wait_until(pilot, lambda: app._selected_info is not None)
            original = app.browser.list_children

            def slow(path):
                entered.set()
                try:
                    assert release.wait(5)
                    return original(path)
                finally:
                    finished.set()

            monkeypatch.setattr(app.browser, "list_children", slow)
            await pilot.press("ctrl+f")
            await wait_until(pilot, lambda: isinstance(app.screen, TreeSearchScreen))
            screen = app.screen
            screen.query_one("#tree-search-recursive", Button).press()
            await wait_until(pilot, entered.is_set)
            await pilot.press("escape")
            assert screen.cancelled.is_set()
            assert not isinstance(app.screen, TreeSearchScreen)
            assert app.selected_path == "/"
            assert not finished.is_set()
            release.set()
            await wait_until(pilot, finished.is_set)
    finally:
        release.set()
        app.wait_for_close()
