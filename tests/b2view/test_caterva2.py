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
