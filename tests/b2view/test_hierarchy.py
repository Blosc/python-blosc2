"""Remote discovery uses metadata and opens only the selected leaf."""

import io
import zipfile

import numpy as np
import pytest
from tui_wait import wait_until

import blosc2
from blosc2.b2view.model import StoreBrowser

fsspec = pytest.importorskip("fsspec")


def assert_warm_revisit(browser, data):
    selection = (slice(2), slice(3))
    np.testing.assert_array_equal(browser.preview("/group/a", slices=selection), data[:2, :3])
    leaf = browser._get_object("/group/a")
    browser.preview("/group/b", slices=selection)
    with pytest.raises(RuntimeError, match="closed"):
        leaf[:1]
    before = browser.store.traffic.nbytes
    np.testing.assert_array_equal(browser.preview("/group/a", slices=selection), data[:2, :3])
    assert browser.store.traffic.nbytes == before
    assert browser.store.cache_bytes <= 64 << 20


def b2z_url(tmp_path, *, threshold=0):
    path = tmp_path / "hierarchy.b2z"
    data = np.random.default_rng(12).integers(0, 256, (200, 1000), dtype="uint8")
    with blosc2.TreeStore(str(path), mode="w", threshold=threshold) as store:
        store["/group/a"] = blosc2.asarray(data, chunks=(40, 250), blocks=(10, 50))
        store["/group/b"] = blosc2.asarray(data[::-1].copy())
        store.attrs["title"] = "root"
        store.get_subtree("/group").attrs["title"] = "child"
    url = "memory://v12/" + path.name
    fsspec.filesystem("memory").pipe_file(url, path.read_bytes())
    return url, data


def test_b2z_discovery_and_dispatch(tmp_path, monkeypatch):
    url, data = b2z_url(tmp_path)
    fs = fsspec.filesystem("memory")
    reads = []
    cat_file = type(fs).cat_file

    def counted(self, path, start=None, end=None, **kwargs):
        reads.append((start, end))
        return cat_file(self, path, start=start, end=end, **kwargs)

    monkeypatch.setattr(type(fs), "cat_file", counted)
    with pytest.raises(NotImplementedError, match="B2Z containers"):
        blosc2.open(url)
    assert reads == []
    with StoreBrowser(url) as browser:
        assert browser.is_tree
        assert [n.name for n in browser.list_children()] == ["group"]
        assert browser.get_info("/").user_attrs == {"title": "root"}
        assert browser.get_info("/group").user_attrs == {"title": "child"}
        assert [n.kind for n in browser.list_children("/group")] == ["ndarray", "ndarray"]
        assert sum(end - start for start, end in reads) < data.nbytes // 4
        reads.clear()
        info = browser.get_info("/group/a")
        assert info.metadata["shape"] == data.shape
        np.testing.assert_array_equal(
            browser.preview("/group/a", slices=(slice(2, 5), slice(4, 9))), data[2:5, 4:9]
        )
        leaf = browser._get_object("/group/a")
        assert isinstance(browser.store, blosc2.RemoteStore)
        assert browser.store.max_cache_bytes == 64 << 20
        assert isinstance(leaf, blosc2.RemoteArray)
        assert leaf is browser._get_object("/group/a")
        reads.clear()
        browser.preview("/group/a", slices=(slice(2, 5), slice(4, 9)))
        assert reads == []
        browser.preview("/group/b", slices=(slice(2, 5), slice(4, 9)))
        browser.get_info("/group")
        reads.clear()
        np.testing.assert_array_equal(
            browser.preview("/group/a", slices=(slice(2, 5), slice(4, 9))), data[2:5, 4:9]
        )
        assert reads == []
    with StoreBrowser(url + "/group/") as browser:
        assert browser.get_info("/").user_attrs == {"title": "child"}
        assert [n.path for n in browser.list_children()] == ["/a", "/b"]
    with StoreBrowser(url + "::group/a") as browser:
        assert not browser.is_tree
        np.testing.assert_array_equal(browser.preview("/", slices=(slice(2), slice(3))), data[:2, :3])


@pytest.mark.parametrize("version", [2, 3])
@pytest.mark.parametrize("consolidated", [False, True])
def test_zarr_hierarchy(version, consolidated):
    zarr = pytest.importorskip("zarr")
    url = f"memory://v12/{version}-{consolidated}.zarr"
    root = zarr.open_group(url, mode="w", zarr_format=version)
    root.attrs["title"] = "root"
    group = root.create_group("group")
    group.attrs["title"] = "child"
    group.create_group("empty").attrs["empty"] = True
    data = np.arange(600).reshape(30, 20)
    group.create_array("a", data=data, chunks=(10, 10))
    group.create_array("b", data=data + 1, chunks=(10, 10))
    if consolidated:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            zarr.consolidate_metadata(url)
    with StoreBrowser(url) as browser:
        assert browser.get_info("/").user_attrs == {"title": "root"}
        assert [n.name for n in browser.list_children()] == ["group"]
        assert browser.get_info("/group").user_attrs == {"title": "child"}
        assert [n.name for n in browser.list_children("/group")] == ["a", "b", "empty"]
        assert browser.kind("/group/empty") == "group"
        assert browser.list_children("/group/empty") == []
        np.testing.assert_array_equal(browser.preview("/group/a", slices=(slice(2), slice(3))), data[:2, :3])
        assert_warm_revisit(browser, data)
    with StoreBrowser(url + "/group") as browser:
        assert [n.path for n in browser.list_children()] == ["/a", "/b", "/empty"]
    with StoreBrowser(url + "/group/a") as browser:
        assert not browser.is_tree
        np.testing.assert_array_equal(browser.preview("/", slices=(slice(2), slice(3))), data[:2, :3])


def test_hdf5_hierarchy(tmp_path, monkeypatch):
    h5py = pytest.importorskip("h5py")
    kerchunk = pytest.importorskip("kerchunk.hdf")
    path = tmp_path / "hierarchy.h5"
    data = np.arange(600).reshape(30, 20)
    with h5py.File(path, "w") as root:
        root.attrs["title"] = "root"
        group = root.create_group("group")
        group.attrs["title"] = "child"
        group.create_group("empty").attrs["empty"] = True
        group.create_dataset("a", data=data, chunks=(10, 10), compression="gzip")
        group.create_dataset("b", data=data + 1)
    url = "memory://v12/hierarchy.h5"
    fsspec.filesystem("memory").pipe_file(url, path.read_bytes())
    calls = []
    translate = kerchunk.SingleHdf5ToZarr.translate

    def counted(self, *args, **kwargs):
        calls.append(1)
        return translate(self, *args, **kwargs)

    monkeypatch.setattr(kerchunk.SingleHdf5ToZarr, "translate", counted)
    with StoreBrowser(url) as browser:
        assert browser.get_info("/").user_attrs == {"title": "root"}
        assert browser.get_info("/group").user_attrs == {"title": "child"}
        assert browser.kind("/group/empty") == "group"
        for leaf in ("a", "b", "a"):
            np.testing.assert_array_equal(
                browser.preview(f"/group/{leaf}", slices=(slice(2), slice(3))), data[:2, :3] + (leaf == "b")
            )
        assert_warm_revisit(browser, data)
        assert calls == [1]


def test_b2z_ambiguous_paths():
    for names in (("a.b2nd", "a.b2f"), ("a.b2nd", "a/b.b2nd"), ("../a.b2nd",)):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            for name in names:
                archive.writestr(name, b"")
        url = "memory://v12/invalid.b2z"
        fsspec.filesystem("memory").pipe_file(url, buffer.getvalue())
        with pytest.raises(ValueError):
            StoreBrowser(url)


@pytest.mark.tui
@pytest.mark.asyncio
async def test_remote_tui_lifecycle(tmp_path, monkeypatch):
    import threading

    from textual.widgets import Tree

    from blosc2.b2view.app import B2ViewApp

    url, data = b2z_url(tmp_path)
    app = B2ViewApp(url, start_path="/group/a")

    async with app.run_test(size=(120, 40)) as pilot:

        async def wait_for(predicate):
            await wait_until(pilot, predicate, message="remote UI did not settle")

        await wait_for(lambda: app.table_page is not None and bool(app.table_page["columns"]))
        assert app.query_one("#tree-pane").display
        assert app.selected_path == "/group/a"
        assert len(app.table_page["columns"]) > app.preview_cols
        np.testing.assert_array_equal(app.table_page["data"]["0"], data[: app.table_page["stop"], 0])
        table = app.query_one("#data-table")
        app._update_data_table(app.table_page)
        first_column = table.ordered_columns[0]
        assert (
            first_column.get_render_width(table)
            >= max(len(str(value)) for value in app.table_page["data"]["0"]) + 2 * table.cell_padding
        )
        assert table._row_label_column_width >= len(str(app.table_page["stop"] - 1)) + 2 * table.cell_padding
        await pilot.pause()
        app._go_to_row(100)
        await wait_for(lambda: app.table_page["start"] > 0 and bool(app.table_page["columns"]))
        page = app.table_page
        np.testing.assert_array_equal(page["data"]["0"], data[page["start"] : page["stop"], 0])

        app._grid_col_end()
        await wait_for(
            lambda: bool(app.table_page["columns"]) and app.table_page.get("col_stop") == data.shape[1]
        )
        page = app.table_page
        last = page["columns"][-1]
        np.testing.assert_array_equal(page["data"][last], data[page["start"] : page["stop"], int(last)])
        previous_start = page["col_start"]
        await wait_for(lambda: not app.loading_table_page)
        app.page_grid_columns(-1)
        await wait_for(
            lambda: bool(app.table_page["columns"]) and app.table_page.get("col_stop") == previous_start
        )
        page = app.table_page
        last = page["columns"][-1]
        np.testing.assert_array_equal(page["data"][last], data[page["start"] : page["stop"], int(last)])

        original = StoreBrowser.preview
        entered, release = threading.Event(), threading.Event()

        def slow(self, path, **kwargs):
            if path == "/group/b":
                entered.set()
                release.wait(5)
            return original(self, path, **kwargs)

        monkeypatch.setattr(StoreBrowser, "preview", slow)
        app.update_panels("/group/b")
        await wait_for(entered.is_set)
        app.update_panels("/group")
        # The event loop remains usable while a blocking transport read runs.
        await pilot.press("tab")
        release.set()
        await wait_for(lambda: app._selected_info is not None and app._selected_info.path == "/group")
        assert app.table_page is None
        previous = app.browser
        app.action_refresh()
        await wait_for(
            lambda: (
                app.browser is not None and app.browser is not previous and app._selected_info is not None
            )
        )
        assert app.selected_path == "/group"
        assert "/" in app.loaded_paths
        assert app.query_one("#tree", Tree).root.children
        await pilot.press("q")


def test_embedded_b2z_index_is_bounded(tmp_path, monkeypatch):
    url, data = b2z_url(tmp_path, threshold=10**9)
    fs = fsspec.filesystem("memory")
    reads = []
    original = type(fs).cat_file

    def counted(self, path, start=None, end=None, **kwargs):
        reads.append((start, end))
        return original(self, path, start=start, end=end, **kwargs)

    monkeypatch.setattr(type(fs), "cat_file", counted)
    with StoreBrowser(url) as browser:
        assert [n.kind for n in browser.list_children("/group")] == ["unsupported", "unsupported"]
        assert browser.get_info("/").user_attrs == {"title": "root"}
        assert browser.get_info("/group").user_attrs == {"title": "child"}
        # Index reads plus the native chunks containing attribute frames are
        # bounded even when embed.b2e contains much larger embedded arrays.
        assert sum(end - start for start, end in reads) < data.nbytes // 4
        assert all(start is not None and end is not None for start, end in reads)


def test_zarr_discovery_reads_no_chunks_and_isolates_codec(monkeypatch):
    import json

    zarr = pytest.importorskip("zarr")
    url = "memory://v12/unsupported.zarr"
    group = zarr.open_group(url, mode="w", zarr_format=2)
    group.create_array("a", data=np.arange(50), chunks=(10,))
    group.create_array("bad", data=np.arange(50), chunks=(10,))
    fs = fsspec.filesystem("memory")
    key = "/v12/unsupported.zarr/bad/.zarray"
    metadata = json.loads(fs.cat_file(key))
    metadata["compressor"] = {"id": "not-an-installed-codec"}
    fs.pipe_file(key, json.dumps(metadata).encode())
    keys = []
    original = zarr.storage.FsspecStore.get

    async def counted(self, key, *args, **kwargs):
        keys.append(key)
        return await original(self, key, *args, **kwargs)

    monkeypatch.setattr(zarr.storage.FsspecStore, "get", counted)
    with StoreBrowser(url) as browser:
        assert [(n.name, n.kind) for n in browser.list_children()] == [
            ("a", "ndarray"),
            ("bad", "unsupported"),
        ]
        assert all(
            key.rsplit("/", 1)[-1] in {".zarray", ".zattrs", ".zgroup", ".zmetadata", "zarr.json"}
            for key in keys
        )
        assert "codec" in browser.get_info("/bad").metadata["preview"]
        np.testing.assert_array_equal(
            browser.preview("/a", start=2, stop=5)["data"]["value"], np.arange(2, 5)
        )


def test_hdf5_unsupported_and_links(tmp_path):
    h5py = pytest.importorskip("h5py")
    pytest.importorskip("kerchunk")
    path = tmp_path / "links.h5"
    with h5py.File(path, "w") as file:
        file.create_dataset("a", data=np.arange(10))
        file.create_dataset("null", shape=None, dtype="i4")
        file["alias"] = file["a"]
        file["soft"] = h5py.SoftLink("/a")
        file["external"] = h5py.ExternalLink("must-not-be-opened.h5", "/a")
        file["cycle"] = file["/"]
        file.create_group("empty")
    url = "memory://v12/links.h5"
    fsspec.filesystem("memory").pipe_file(url, path.read_bytes())
    with StoreBrowser(url) as browser:
        children = {n.name: n.kind for n in browser.list_children()}
        assert children["a"] == "ndarray"
        assert children["null"] == "unsupported"
        assert children["empty"] == "group"
        assert "external" not in children
        assert "cycle" not in children
        assert "Kerchunk" in browser.get_info("/").metadata["notice"]
        np.testing.assert_array_equal(browser.preview("/a", start=0, stop=3)["data"]["value"], np.arange(3))


@pytest.mark.tui
@pytest.mark.asyncio
async def test_remote_listing_retry_and_shutdown(tmp_path, monkeypatch):
    import asyncio
    import threading

    from textual.widgets import Static, Tree

    from blosc2.b2view.app import B2ViewApp

    url, _ = b2z_url(tmp_path)
    original = StoreBrowser.list_children
    attempts = []

    def flaky(self, path="/"):
        if path == "/group":
            attempts.append(path)
            if len(attempts) == 1:
                raise OSError("temporary listing failure")
        return original(self, path)

    monkeypatch.setattr(StoreBrowser, "list_children", flaky)
    app = B2ViewApp(url)
    async with app.run_test(size=(120, 40)) as pilot:
        for _ in range(100):
            if app.browser is not None:
                break
            await asyncio.sleep(0.01)
        node = app.query_one("#tree", Tree).root.children[0]
        app.load_children(node)
        await pilot.pause()
        assert "/group" not in app.loaded_paths
        app.load_children(node)
        for _ in range(100):
            if "/group" in app.loaded_paths:
                break
            await asyncio.sleep(0.01)
        assert len(node.children) == 2
        entered, release, closed = threading.Event(), threading.Event(), threading.Event()
        preview = StoreBrowser.preview
        close = StoreBrowser.close

        def blocked(self, path, **kwargs):
            entered.set()
            release.wait(15)
            return preview(self, path, **kwargs)

        def counted_close(self):
            assert release.is_set()
            close(self)
            closed.set()

        monkeypatch.setattr(StoreBrowser, "preview", blocked)
        monkeypatch.setattr(StoreBrowser, "close", counted_close)
        app.query_one("#tree", Tree).select_node(
            next(child for child in node.children if child.data == "/group/a")
        )
        for _ in range(500):
            if entered.is_set():
                break
            await pilot.pause(0.02)
        assert entered.is_set(), (
            f"preview was never called; metadata={app.query_one('#metadata', Static).render()!r}"
        )
        await pilot.press("q")
        assert not closed.is_set()
        release.set()
    for _ in range(300):
        if closed.is_set():
            break
        await asyncio.sleep(0.02)
    assert closed.is_set()


@pytest.mark.parametrize("format", ["zarr", "h5"])
@pytest.mark.tui
def test_group_decoder_fresh_process(tmp_path, format):
    import subprocess
    import sys

    path = tmp_path / ("compressed." + format)
    if format == "zarr":
        zarr = pytest.importorskip("zarr")
        root = zarr.open_group(path, mode="w", zarr_format=3)
        root.create_array(
            "a", data=np.arange(100, dtype="i4"), chunks=(20,), compressors=[zarr.codecs.BloscCodec()]
        )
    else:
        h5py = pytest.importorskip("h5py")
        plugin = pytest.importorskip("hdf5plugin")
        pytest.importorskip("kerchunk")
        with h5py.File(path, "w") as file:
            file.create_dataset("a", data=np.arange(100, dtype="i4"), chunks=(20,), **plugin.Blosc())
    script = """
import asyncio
from pathlib import Path
import sys
import fsspec
from blosc2.b2view.app import B2ViewApp
path = Path(sys.argv[1])
url = 'memory://fresh/' + path.name
fs = fsspec.filesystem('memory')
if path.is_dir():
    for file in path.rglob('*'):
        if file.is_file():
            fs.pipe_file(url + '/' + file.relative_to(path).as_posix(), file.read_bytes())
else:
    fs.pipe_file(url, path.read_bytes())
app = B2ViewApp(url, start_path='/a')
async def main():
    async with app.run_test(size=(120, 40)) as pilot:
        for _ in range(300):
            if app.table_page and app.table_page['columns']:
                break
            await asyncio.sleep(.02)
        assert app.table_page and app.table_page['columns'], 'preview did not load'
        assert app.query_one('#tree-pane').display
        values = app.table_page['data']['value']
        assert list(values) == list(range(len(values)))
        await pilot.press('q')
asyncio.run(main())
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(path)], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_b2z_optional_dependency_isolation(tmp_path):
    import subprocess
    import sys

    b2z_url(tmp_path)
    script = """
import importlib.abc
import sys
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'zarr', 'h5py', 'kerchunk'}:
            raise ImportError('optional dependency intentionally unavailable')
sys.meta_path.insert(0, BlockOptional())
from pathlib import Path
import fsspec
from blosc2.b2view.model import StoreBrowser
url = 'memory://isolated.b2z'
fsspec.filesystem('memory').pipe_file(url, Path(sys.argv[1]).read_bytes())
with StoreBrowser(url) as browser:
    assert browser.list_children()[0].name == 'group'
    assert browser.preview('/group/a', max_rows=2)['stop'] == 2
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "hierarchy.b2z")],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("threshold", [0, 10**9])
def test_b2z_empty_groups_and_ctable_boundaries(tmp_path, threshold):
    from dataclasses import dataclass

    @dataclass
    class Row:
        x: int = 0

    path = tmp_path / "objects.b2z"
    table = blosc2.CTable(Row)
    table.append(Row(3))
    with blosc2.TreeStore(path, mode="w", threshold=threshold) as store:
        store["/table"] = table
        store.get_subtree("/empty").attrs["empty"] = True
        store["/ordinary/_meta"] = blosc2.SChunk(data=b"ordinary user data")
    url = "memory://v12/objects.b2z"
    fsspec.filesystem("memory").pipe_file(url, path.read_bytes())
    with StoreBrowser(url) as browser:
        assert [(n.name, n.kind) for n in browser.list_children()] == [
            ("empty", "group"),
            ("ordinary", "group"),
            ("table", "unsupported"),
        ]
        assert browser.list_children("/empty") == []
        assert browser.get_info("/empty").user_attrs == {"empty": True}
        assert browser.list_children("/table") == []
        assert "CTable" in browser.get_info("/table").metadata["preview"]


def test_b2z_large_embedded_chunk_notice():
    store = blosc2.EmbedStore(chunksize=2**21)
    attrs = blosc2.SChunk()
    attrs.attrs["title"] = "root"
    store["/__vlmeta__"] = attrs
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("embed.b2e", store.to_cframe())
    url = "memory://v12/large-chunks.b2z"
    fsspec.filesystem("memory").pipe_file(url, buffer.getvalue())
    with StoreBrowser(url) as browser:
        assert "larger than 1 MiB" in browser.get_info("/").metadata["notice"]
        assert browser.get_info("/").user_attrs is None


def test_b2z_explicit_localization(tmp_path):
    url, data = b2z_url(tmp_path)
    with blosc2.open(url, cache_dir=tmp_path / "localized", mode="r") as store:
        assert isinstance(store, blosc2.TreeStore)
        np.testing.assert_array_equal(store["/group/a"][:2, :3], data[:2, :3])


def test_zarr_listing_permission_preserves_direct_array(monkeypatch):
    zarr = pytest.importorskip("zarr")
    url = "memory://v12/no-list.zarr"
    root = zarr.open_group(url, mode="w", zarr_format=2)
    root.create_array("a", data=np.arange(10), chunks=(5,))

    async def denied(self, prefix):
        raise PermissionError("LIST denied")
        yield  # async iterator, matching Store.list_dir

    monkeypatch.setattr(zarr.storage.FsspecStore, "list_dir", denied)
    with StoreBrowser(url) as browser:
        with pytest.raises(OSError, match="LIST permission"):
            browser.list_children()
        assert isinstance(browser.store, blosc2.RemoteStore)
    with StoreBrowser(url + "/a") as browser:
        np.testing.assert_array_equal(browser.preview("/", stop=3)["data"]["value"], np.arange(3))


def test_remote_store_browser_cache_dir_reused(tmp_path):
    url, data = b2z_url(tmp_path)
    cache_dir = tmp_path / "b2view_cache"

    selection = (slice(2), slice(3))
    # First session: cold
    with StoreBrowser(url, cache_dir=str(cache_dir)) as browser:
        assert isinstance(browser.store, blosc2.RemoteStore)
        np.testing.assert_array_equal(browser.preview("/group/a", slices=selection), data[:2, :3])
        traffic1 = browser.store.traffic.nbytes
        assert traffic1 > 0

    # Second session: warm reuse upon restart
    with StoreBrowser(url, cache_dir=str(cache_dir)) as browser:
        assert isinstance(browser.store, blosc2.RemoteStore)
        assert browser.store.traffic.nbytes == 0
        np.testing.assert_array_equal(browser.preview("/group/a", slices=selection), data[:2, :3])
        assert browser.store.traffic.nbytes == 0
