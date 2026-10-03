"""Repository and mount-boundary browsing through the existing viewer."""

import numpy as np
import pytest
from test_remote_caterva2 import caterva2_source  # noqa: F401
from tui_wait import wait_until

from blosc2.b2view.model import StoreBrowser


def test_service_browser_metadata_and_previews(caterva2_source):  # noqa: F811
    base, array, _, stats = caterva2_source
    with StoreBrowser(base) as browser:
        assert browser.is_tree
        assert [node.name for node in browser.list_children()] == ["mount"]
        assert not stats["fetches"]
        assert [node.kind for node in browser.list_children("/mount")] == ["ndarray", "group", "ctable"]
        assert browser.get_info("/mount").metadata["catalog_attrs"] == {"note": "curated"}
        assert browser.get_info("/mount/table").metadata["nrows"] == 12
        preview = browser.preview("/mount/table", start=2, stop=4, max_cols=1)
        np.testing.assert_array_equal(preview["data"]["ident"], [2, 3])
        assert not browser.supports_table_transforms("/mount/table")
        for operation in (
            lambda: browser.set_filter("/mount/table", "ident > 2"),
            lambda: browser.set_sort("/mount/table", "ident", False),
            lambda: browser.set_group("/mount/table", "ident", "count", None),
            lambda: browser.read_series("/mount/table", column="ident"),
        ):
            with pytest.raises(NotImplementedError):
                operation()
        np.testing.assert_array_equal(
            browser.preview("/mount/array", slices=(slice(2), slice(3))), array[:2, :3]
        )


def test_repository_browser_and_broken_sibling(caterva2_source):  # noqa: F811
    base, _, _, stats = caterva2_source
    stats["roots"]["@broken"] = {"name": "@broken"}
    with StoreBrowser(base) as browser:
        assert [node.name for node in browser.list_children()] == ["@broken", "@public"]
        assert [node.name for node in browser.list_children("/@public")] == ["mount"]
    stats["roots"].pop("@broken")
    stats["groups"]["@public/mount"].append("broken")
    with StoreBrowser(base) as browser:
        children = browser.list_children("/mount")
        assert [(node.name, node.kind) for node in children] == [
            ("array", "ndarray"),
            ("broken", "unavailable"),
            ("empty", "group"),
            ("table", "ctable"),
        ]
        assert next(node for node in children if node.name == "broken").has_children
        stats["groups"]["@public/mount/broken"] = []
        assert browser.list_children("/mount/broken") == []


def test_published_extensions_do_not_select_direct_file_backends(caterva2_source):  # noqa: F811
    base, _, _, stats = caterva2_source
    stats["aliases"]["@public/group.h5"] = "@public/group"
    stats["aliases"]["@public/table.parquet"] = "@public/table"
    with StoreBrowser(base + "@public/group.h5") as browser:
        assert browser.is_tree
        assert [node.name for node in browser.list_children()] == ["array", "table"]
    with StoreBrowser(base + "@public/table.parquet") as browser:
        assert browser.kind("/") == "ctable"
        assert browser.preview("/", stop=2)["nrows"] == 12


@pytest.mark.tui
@pytest.mark.asyncio
@pytest.mark.parametrize("multiple", [False, True])
async def test_repository_tui_expands_mounts_and_previews(caterva2_source, multiple):  # noqa: F811
    from blosc2.b2view.app import B2ViewApp

    base, array, _, stats = caterva2_source
    prefix = "/@public" if multiple else ""
    if multiple:
        stats["roots"]["@broken"] = {"name": "@broken"}
    app = B2ViewApp(base, start_path=prefix + "/mount/array")
    async with app.run_test(size=(120, 40)) as pilot:
        await wait_until(pilot, lambda: app.table_page is not None and bool(app.table_page["columns"]))
        assert app.selected_path == prefix + "/mount/array"
        page = app.table_page
        np.testing.assert_array_equal(page["data"]["0"], array[: page["stop"], 0])
        app.update_panels(prefix + "/mount/table")
        await wait_until(
            pilot, lambda: app.table_page is not None and app.table_page.get("source_kind") == "ctable"
        )
        assert app.table_page["nrows"] == 12
    app.wait_for_close()


def test_cli_service_override():
    from blosc2.b2view.cli import build_parser

    args = build_parser().parse_args(["https://host/@data/a.h5", "--remote-service", "fsspec"])
    assert args.remote_service == "fsspec"


@pytest.mark.parametrize(
    ("target", "relative", "expected"),
    [
        ("", "/", "@public"),
        ("", "/mount/table", "@public/mount/table"),
        ("@public", "/mount/table", "@public/mount/table"),
        ("@public/mount", "/array", "@public/mount/array"),
        ("@public/table", "/", "@public/table"),
    ],
)
def test_service_metadata_displays_full_path(caterva2_source, target, relative, expected):  # noqa: F811
    from rich.console import Console

    from blosc2.b2view.render import make_metadata_renderable

    base, _, _, _ = caterva2_source
    with StoreBrowser(base + target) as browser:
        info = browser.get_info(relative)
        assert info.path == relative
        assert info.display_path == expected
        console = Console(record=True, width=120)
        console.print(make_metadata_renderable(info))
        assert expected in console.export_text()
        assert "/@public" not in console.export_text()


def test_multiroot_display_does_not_duplicate_prefix(caterva2_source):  # noqa: F811
    base, _, _, stats = caterva2_source
    stats["roots"]["@broken"] = {"name": "@broken"}
    with StoreBrowser(base) as browser:
        info = browser.get_info("/@public/mount/table")
        assert info.display_path == "@public/mount/table"


@pytest.mark.tui
@pytest.mark.asyncio
async def test_remote_table_window_is_background_and_stale_results_are_discarded(
    caterva2_source,  # noqa: F811
    monkeypatch,
):
    import threading

    from blosc2.b2view.app import B2ViewApp

    base, _, _, _ = caterva2_source
    entered, release = threading.Event(), threading.Event()
    original = StoreBrowser.prepare_row_window

    def slow(self, path, start, stop):
        entered.set()
        assert release.wait(5)
        return original(self, path, start, stop)

    monkeypatch.setattr(StoreBrowser, "prepare_row_window", slow)
    app = B2ViewApp(base, start_path="/mount/table")
    try:
        async with app.run_test(size=(120, 40)) as pilot:
            await wait_until(
                pilot, lambda: app.table_page is not None and app.table_page.get("source_kind") == "ctable"
            )
            app._enter_row_window(2, 5, backend="ctable")
            await wait_until(pilot, entered.is_set)
            # A slow bounded table fetch must not block event handling.
            await pilot.press("tab")
            app.update_panels("/mount")
            release.set()
            await wait_until(
                pilot, lambda: app._selected_info is not None and app._selected_info.path == "/mount"
            )
            assert not app.browser.get_row_window("/mount/table")
            app.update_panels("/mount/table")
            await wait_until(
                pilot, lambda: app.table_page is not None and app.table_page.get("source_kind") == "ctable"
            )
            app._enter_row_window(2, 5, backend="ctable")
            await wait_until(
                pilot,
                lambda: app.row_window == (2, 5) and app.table_page and bool(app.table_page["columns"]),
            )
            assert app.table_page["nrows"] == 3
            np.testing.assert_array_equal(app.table_page["data"]["ident"], [2, 3, 4])
            plotted = app.browser.plot_series("/mount/table", column="ident")
            assert plotted["n"] == 3
    finally:
        release.set()
        app.wait_for_close()
