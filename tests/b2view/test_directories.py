"""Lazy ordinary directories, including expandable native dataset mounts."""

import numpy as np
import pytest
from tui_wait import wait_until

import blosc2
from blosc2.b2view.model import StoreBrowser


@pytest.mark.parametrize("backend", ["local", "file", "memory"])
def test_directory_browsing_is_lazy(tmp_path, backend):
    root = tmp_path / "root"
    root.mkdir()
    (root / "sub").mkdir()
    (root / "sub/README.md").write_text("# Hello")
    (root / "broken.b2nd").write_bytes(b"broken")
    if backend == "memory":
        fs = pytest.importorskip("fsspec").filesystem("memory")
        source = "memory://b2view-directory/root"
        fs.pipe(source + "/sub/README.md", b"# Hello")
        fs.pipe(source + "/broken.b2nd", b"broken")
    else:
        source = root.as_uri() if backend == "file" else str(root) + "/"
    with StoreBrowser(source, cache_dir=str(tmp_path / "cache")) as browser:
        assert browser.is_tree
        children = browser.list_children()
        assert [node.name for node in children] == ["broken.b2nd", "sub"]
        assert not browser.store._browsers
        assert browser.get_info("/").kind == "group"
        assert browser.list_children("/sub")[0].kind == "file"
        assert not browser.store._browsers
        assert browser.preview("/sub/README.md")["file_text"] == "# Hello"
        with pytest.raises((RuntimeError, ValueError)):
            browser.get_info("/broken.b2nd")
        assert browser.preview("/sub/README.md")["markdown"]


@pytest.mark.parametrize("backend", ["local", "memory"])
def test_dataset_mounts_and_standalone_arrays(tmp_path, backend):
    data = np.arange(10)
    blosc2.asarray(data, urlpath=str(tmp_path / "array.b2nd"))
    with blosc2.TreeStore(str(tmp_path / "tree.b2z"), mode="w") as tree:
        tree["/group/array"] = blosc2.asarray(data)
    source = str(tmp_path)
    if backend == "memory":
        fs = pytest.importorskip("fsspec").filesystem("memory")
        source = "memory://b2view-directory/datasets"
        for name in ("array.b2nd", "tree.b2z"):
            fs.pipe(source + "/" + name, (tmp_path / name).read_bytes())
    with StoreBrowser(source, cache_dir=str(tmp_path / "cache")) as browser:
        nodes = {node.name: node for node in browser.list_children()}
        assert nodes["tree.b2z"].has_children
        assert not nodes["array.b2nd"].has_children
        assert not browser.store._browsers
        assert browser.get_info("/array.b2nd").kind == "ndarray"
        assert browser.list_children("/tree.b2z")[0].path == "/tree.b2z/group"
        assert browser.list_children("/tree.b2z/group")[0].path == "/tree.b2z/group/array"
        np.testing.assert_array_equal(browser.preview("/tree.b2z/group/array")["data"]["value"], data)
    with StoreBrowser(str(tmp_path / "tree.b2z")) as browser:
        assert browser.is_tree
        assert browser.list_children()[0].name == "group"


def test_hdf5_mount_in_directory(tmp_path):
    h5py = pytest.importorskip("h5py")
    data = np.arange(6)
    with h5py.File(tmp_path / "data.h5", "w") as file:
        file.create_dataset("group/array", data=data)
    with StoreBrowser(str(tmp_path)) as browser:
        assert browser.list_children()[0].has_children
        assert browser.list_children("/data.h5")[0].path == "/data.h5/group"
        np.testing.assert_array_equal(browser.preview("/data.h5/group/array")["data"]["value"], data)


def test_chained_fsspec_directory(tmp_path):
    pytest.importorskip("fsspec")
    import zipfile

    archive = tmp_path / "files.zip"
    with zipfile.ZipFile(archive, "w") as file:
        file.writestr("notes/README.md", "# Hello")
    with StoreBrowser(f"zip://notes::{archive.as_uri()}") as browser:
        assert browser.is_tree
        assert browser.list_children()[0].name == "README.md"
        assert browser.preview("/README.md")["file_text"] == "# Hello"


def test_symlinks_and_parent_paths_do_not_escape_root(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("private")
    try:
        (root / "link.txt").symlink_to(outside)
    except OSError:
        pytest.skip("Symlinks unavailable")
    with StoreBrowser(str(root)) as browser:
        assert browser.list_children()[0].kind == "unsupported"
        with pytest.raises(ValueError, match="symlinks"):
            browser.get_info("/link.txt")
        with pytest.raises(ValueError, match="escape"):
            browser.get_info("/../outside.txt")


@pytest.mark.tui
@pytest.mark.asyncio
async def test_directory_tui_navigation_and_refresh(tmp_path):
    from textual.widgets import Static

    from blosc2.b2view.app import B2ViewApp

    (tmp_path / "README.md").write_text("# Hello")
    app = B2ViewApp(str(tmp_path))
    async with app.run_test(size=(120, 40)) as pilot:
        await wait_until(pilot, lambda: app._selected_info is not None)
        assert app.browser.is_tree
        assert app.query_one("#tree-pane").display
        app.update_panels("/README.md")
        await wait_until(
            pilot, lambda: "T: raw/Markdown" in str(app.query_one("#data-header", Static).render())
        )
        await pilot.press("T")
        await wait_until(pilot, lambda: "# Hello" in str(app.query_one("#preview", Static).render()))
        (tmp_path / "new.txt").write_text("new")
        app.action_refresh()
        await wait_until(pilot, lambda: app.browser is not None and app._selected_info is not None)
        assert "new.txt" in [node.name for node in app.browser.list_children()]
    app.wait_for_close()
