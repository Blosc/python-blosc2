from __future__ import annotations

import dataclasses

import numpy as np

import blosc2
from blosc2.b2view.model import StoreBrowser, preview_array, preview_ctable
from blosc2.b2view.render import make_preview_renderables


@dataclasses.dataclass
class Row:
    x: int = 0
    y: float = 0.0


def make_ctable(n=5):
    table = blosc2.CTable(Row)
    for i in range(n):
        table.append(Row(x=i, y=i * 1.5))
    return table


def make_store(path):
    with blosc2.TreeStore(str(path), mode="w") as store:
        store["/group/arr"] = np.arange(12).reshape(3, 4)
        store["/table"] = make_ctable(6)


def test_store_browser_lists_children_and_kinds(tmp_path):
    path = tmp_path / "bundle.b2z"
    make_store(path)

    with StoreBrowser(str(path)) as browser:
        root = browser.list_children("/")
        assert [(node.path, node.kind, node.has_children) for node in root] == [
            ("/group", "group", True),
            ("/table", "ctable", False),
        ]
        group = browser.list_children("/group")
        assert [(node.path, node.kind) for node in group] == [("/group/arr", "ndarray")]


def test_store_browser_metadata_and_previews(tmp_path):
    path = tmp_path / "bundle.b2d"
    make_store(path)

    with StoreBrowser(str(path)) as browser:
        arr_info = browser.get_info("/group/arr")
        assert arr_info.kind == "ndarray"
        assert arr_info.metadata["shape"] == (3, 4)
        assert arr_info.metadata["dtype"] == "int64"
        np.testing.assert_array_equal(
            browser.preview("/group/arr", max_rows=2, max_cols=3), np.array([[0, 1, 2], [4, 5, 6]])
        )

        table_info = browser.get_info("/table")
        assert table_info.kind == "ctable"
        assert table_info.metadata["rows"] == 6
        preview = browser.preview("/table", max_rows=3, max_cols=1)
        assert preview["columns"] == ["x"]
        assert preview["hidden_columns"] == 1
        np.testing.assert_array_equal(preview["data"]["x"], np.array([0, 1, 2]))


def test_store_browser_supports_standalone_ctable(tmp_path):
    path = tmp_path / "table.b2z"
    table = make_ctable(4)
    persistent = blosc2.CTable(Row, urlpath=str(path), mode="w")
    persistent.extend(table)
    persistent.close()

    with StoreBrowser(str(path)) as browser:
        assert browser.list_children("/") == []
        info = browser.get_info("/")
        assert info.kind == "ctable"
        assert info.metadata["rows"] == 4
        preview = browser.preview("/", max_rows=2)
        np.testing.assert_array_equal(preview["data"]["x"], np.array([0, 1]))


def test_preview_ctable_preserves_ragged_nested_values():
    class Column:
        def __init__(self, values):
            self.values = values

        def __getitem__(self, key):
            return self.values[key]

    class Table:
        def __init__(self):
            self.col_names = ["path"]
            self.columns = {"path": Column([[{"x": 1}], [{"x": 2}, {"x": 3}]])}

        def __len__(self):
            return 2

        def __getitem__(self, name):
            return self.columns[name]

    preview = preview_ctable(Table(), max_cols=1)
    assert preview["data"]["path"].dtype == object
    assert preview["data"]["path"][1] == [{"x": 2}, {"x": 3}]


def test_ctable_preview_buffer_reuses_loaded_rows(tmp_path):
    path = tmp_path / "table.b2z"
    persistent = blosc2.CTable(Row, urlpath=str(path), mode="w")
    for i in range(100):
        persistent.append(Row(x=i, y=float(i)))
    persistent.close()

    from blosc2.b2view.app import B2ViewApp

    app = B2ViewApp(str(path), preview_rows=5)
    with StoreBrowser(str(path)) as browser:
        app.browser = browser
        app.table_buffer = None
        app.query_one = lambda selector, cls=None: type(
            "FakeTable", (), {"size": type("Size", (), {"height": 6})()}
        )()
        page0 = app._load_table_page("/", 0)
        first_buffer = app.table_buffer
        page1 = app._load_table_page("/", 5)
        assert app.table_buffer is first_buffer
        np.testing.assert_array_equal(page0["data"]["x"], np.arange(5))
        np.testing.assert_array_equal(page1["data"]["x"], np.arange(5, 10))


def test_ctable_preview_header_uses_column_names_without_dtype_labels():
    preview = {
        "start": 0,
        "stop": 1,
        "nrows": 1,
        "columns": ["when", "value"],
        "hidden_columns": 0,
        "data": {
            "when": np.array(["2025-01-01"], dtype="datetime64[D]"),
            "value": np.array([1], dtype=np.int64),
        },
    }
    from rich.console import Console

    header, _ = make_preview_renderables(preview)
    console = Console(width=80, record=True)
    console.print(header)
    rendered = console.export_text()
    assert "when" in rendered
    assert "value" in rendered
    assert "datetime64" not in rendered
    assert "int64" not in rendered


def test_preview_array_high_dimensional_slice():
    arr = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    preview = preview_array(arr, max_rows=2, max_cols=3)
    np.testing.assert_array_equal(preview, arr[0, :2, :3])
