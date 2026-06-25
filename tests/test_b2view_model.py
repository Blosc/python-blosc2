from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import blosc2
from blosc2.b2view.model import (
    StoreBrowser,
    preview_array,
    preview_array_1d,
    preview_array_2d,
    preview_ctable,
    preview_schunk,
    schunk_row_geometry,
)
from blosc2.b2view.render import make_preview_renderables


@dataclasses.dataclass
class Row:
    x: int = 0
    y: float = 0.0


def make_ctable(n=5):
    table = blosc2.CTable(Row)
    # Bulk extend instead of n single appends (same data, ~100x faster to build).
    table.extend({"x": np.arange(n), "y": np.arange(n) * 1.5})
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
        assert arr_info.metadata["dtype"] == np.arange(12).dtype.name
        arr_preview = browser.preview("/group/arr", max_rows=2, max_cols=3)
        assert arr_preview["source_kind"] == "ndarray2d"
        np.testing.assert_array_equal(arr_preview["data"]["0"], np.array([0, 4]))
        np.testing.assert_array_equal(arr_preview["data"]["2"], np.array([2, 6]))

        table_info = browser.get_info("/table")
        assert table_info.kind == "ctable"
        assert table_info.metadata["nrows"] == 6
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
        assert info.metadata["nrows"] == 4
        preview = browser.preview("/", max_rows=2)
        np.testing.assert_array_equal(preview["data"]["x"], np.array([0, 1]))


def test_preview_array_1d_returns_grid_preview():
    arr = np.arange(10)
    preview = preview_array_1d(arr, start=3, stop=7)
    assert preview["start"] == 3
    assert preview["stop"] == 7
    assert preview["nrows"] == 10
    assert preview["columns"] == ["value"]
    assert preview["source_kind"] == "ndarray1d"
    np.testing.assert_array_equal(preview["data"]["value"], np.array([3, 4, 5, 6]))


def test_preview_array_2d_returns_grid_preview():
    arr = np.arange(30).reshape(5, 6)
    preview = preview_array_2d(arr, start=1, stop=4, col_start=2, max_cols=3)
    assert preview["start"] == 1
    assert preview["stop"] == 4
    assert preview["nrows"] == 5
    assert preview["columns"] == ["2", "3", "4"]
    assert preview["hidden_columns"] == 3
    assert preview["col_start"] == 2
    assert preview["col_stop"] == 5
    assert preview["ncols"] == 6
    np.testing.assert_array_equal(preview["data"]["2"], np.array([8, 14, 20]))
    np.testing.assert_array_equal(preview["data"]["4"], np.array([10, 16, 22]))


def test_store_browser_uses_grid_preview_for_2d_ndarray(tmp_path):
    path = tmp_path / "bundle.b2z"
    with blosc2.TreeStore(str(path), mode="w") as store:
        store["/arr"] = np.arange(30).reshape(5, 6)

    with StoreBrowser(str(path)) as browser:
        preview = browser.preview("/arr", start=2, stop=5, max_cols=2)
        assert preview["source_kind"] == "ndarray2d"
        assert preview["columns"] == ["0", "1"]
        np.testing.assert_array_equal(preview["data"]["1"], np.array([13, 19, 25]))


def test_ctable_preview_buffer_reuses_loaded_rows(tmp_path):
    pytest.importorskip("textual", reason="b2view TUI requires textual")
    if blosc2.IS_WASM:
        pytest.skip("instantiating a Textual app needs a terminal driver (termios)")
    path = tmp_path / "table.b2z"
    persistent = blosc2.CTable(Row, urlpath=str(path), mode="w")
    persistent.extend({"x": np.arange(100), "y": np.arange(100, dtype=np.float64)})
    persistent.close()

    from blosc2.b2view.app import B2ViewApp

    app = B2ViewApp(str(path), preview_rows=5)
    with StoreBrowser(str(path)) as browser:
        app.browser = browser
        app.table_buffer = None
        app.query_one = lambda selector, cls=None: type(
            "FakeTable", (), {"size": type("Size", (), {"height": 6, "width": 80})()}
        )()
        page0 = app._load_table_page("/", 0)
        first_buffer = app.table_buffer
        page1 = app._load_table_page("/", 5)
        assert app.table_buffer is first_buffer
        np.testing.assert_array_equal(page0["data"]["x"], np.arange(5))
        np.testing.assert_array_equal(page1["data"]["x"], np.arange(5, 10))


def test_preview_ctable_skips_expensive_nested_columns_by_default():
    class Table:
        def __init__(self):
            self.col_names = ["path"]

        def __len__(self):
            return 3

        def __getitem__(self, name):
            raise AssertionError("expensive column should not be read")

        def schema_dict(self):
            return {"columns": [{"name": "path", "kind": "list", "item": {"kind": "struct"}}]}

        @property
        def info_items(self):
            return [("columns", {"path": "list[struct]"})]

    preview = preview_ctable(Table(), max_cols=1)
    assert preview["skipped_columns"] == {"path": "list[struct]"}
    assert preview["data"]["path"].tolist() == ["<list[struct]; skipped>"] * 3


@dataclasses.dataclass
class TaggedRow:
    id: int = blosc2.field(blosc2.int32())
    tags: list[int] = blosc2.field(blosc2.list(blosc2.int64(), nullable=True))  # noqa: RUF009


def test_read_cell_decodes_expensive_column_on_demand(tmp_path):
    path = tmp_path / "tagged.b2z"
    rows = [(0, [0]), (1, [1, 10]), (2, [2, 20, 200]), (3, None)]
    with blosc2.TreeStore(str(path), mode="w") as store:
        store["/t"] = blosc2.CTable(TaggedRow, new_data=rows)

    with StoreBrowser(str(path)) as browser:
        # The expensive list column is a placeholder in the preview, ...
        preview = browser.preview("/t", max_cols=2)
        assert "tags" in preview["skipped_columns"]
        # ... but read_cell decodes the exact cell the grid row points at.
        assert browser.read_cell("/t", "tags", 2) == [2, 20, 200]
        assert browser.read_cell("/t", "tags", 0) == [0]
        assert browser.read_cell("/t", "tags", 3) is None


def test_read_cell_honors_filter_view_row_space(tmp_path):
    path = tmp_path / "tagged_filter.b2z"
    rows = [(0, [0]), (1, [1, 10]), (2, [2, 20]), (3, [3, 30])]
    with blosc2.TreeStore(str(path), mode="w") as store:
        store["/t"] = blosc2.CTable(TaggedRow, new_data=rows)

    with StoreBrowser(str(path)) as browser:
        browser.set_filter("/t", "id >= 2")  # live view is rows [2, 3]
        # read_cell row 0 must resolve to the first *visible* row (id == 2).
        assert browser.read_cell("/t", "tags", 0) == [2, 20]
        assert browser.read_cell("/t", "tags", 1) == [3, 30]


def test_schunk_row_geometry_groups_by_typesize():
    assert schunk_row_geometry(1) == (16, 16)
    assert schunk_row_geometry(2) == (8, 16)
    assert schunk_row_geometry(4) == (4, 16)
    assert schunk_row_geometry(8) == (2, 16)
    assert schunk_row_geometry(3) == (5, 15)  # rows stay a whole multiple of typesize
    assert schunk_row_geometry(32) == (1, 32)  # never below one whole item


def test_preview_schunk_hex_dump_bytes_and_offsets():
    data = bytes(range(256))
    s = blosc2.SChunk(chunksize=2**16, data=data)
    p = preview_schunk(s, start=0, stop=4)

    assert p["source_kind"] == "schunk"
    assert p["columns"] == ["hex", "ascii"]
    assert p["nrows"] == 16  # 256 bytes / 16 per row
    assert p["nbytes"] == 256
    # Row 0 is bytes 0x00..0x0f, byte-offset label in hex.
    assert p["row_labels"][:2] == ["00000000", "00000010"]
    assert p["data"]["hex"][0] == "00 01 02 03 04 05 06 07 08 09 0a 0b 0c 0d 0e 0f"
    # Printable ASCII renders; non-printable bytes become dots.
    assert p["data"]["ascii"][0] == "." * 16
    assert p["data"]["ascii"][3] == "0123456789:;<=>?"  # bytes 0x30..0x3f


def test_preview_schunk_groups_hex_by_typesize():
    s = blosc2.SChunk(chunksize=2**16, cparams={"typesize": 4}, data=bytes(range(32)))
    p = preview_schunk(s, start=0, stop=2)
    assert p["typesize"] == 4
    # 4-byte items, no inter-byte spaces inside an item.
    assert p["data"]["hex"][0] == "00010203 04050607 08090a0b 0c0d0e0f"


def test_preview_schunk_paging_reads_only_the_window(tmp_path):
    path = tmp_path / "raw.b2z"
    with blosc2.TreeStore(str(path), mode="w") as store:
        store["/raw"] = blosc2.SChunk(chunksize=2**16, data=bytes(range(256)))

    with StoreBrowser(str(path)) as browser:
        # A later page resolves the right byte offsets without reading earlier rows.
        page = browser.preview("/raw", start=10, stop=12)
        assert page["row_labels"] == ["000000a0", "000000b0"]  # rows 10, 11 → bytes 160, 176
        assert page["data"]["hex"][0].startswith("a0 a1 a2 a3")


def test_preview_schunk_empty():
    s = blosc2.SChunk(chunksize=2**16)
    p = preview_schunk(s, start=0, stop=20)
    assert p["nrows"] == 0
    assert list(p["data"]["hex"]) == []


def test_ctable_preview_preserves_ragged_nested_values():
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
    pytest.importorskip("rich", reason="b2view rendering requires rich")
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


def test_plot_series_1d_envelope_captures_extremes(tmp_path):
    path = tmp_path / "plot1d.b2z"
    n = 100_000
    data = np.linspace(0, 1, num=n)
    data[12345] = 9.0  # a spike between bucket samples
    data[98765] = -9.0
    with blosc2.TreeStore(str(path), mode="w") as store:
        store["/wave"] = blosc2.asarray(data)

    with StoreBrowser(str(path)) as browser:
        series = browser.plot_series("/wave", max_points=300)
        assert series["n"] == n
        assert series["method"] == "reduce"  # NDArray leaf, fits the read budget
        assert len(series["x"]) <= 300
        # The envelope must contain the true global extremes, including spikes
        assert np.isclose(np.nanmax(series["ymax"]), 9.0)
        assert np.isclose(np.nanmin(series["ymin"]), -9.0)

        # Small arrays: one bucket per element, ymin == ymax == the values
        small = browser.plot_series("/wave", max_points=n)
        assert len(small["x"]) == n
        np.testing.assert_allclose(small["ymin"], data)
        np.testing.assert_allclose(small["ymax"], data)


def test_plot_series_uses_summary_index_when_available(tmp_path):
    # A persisted CTable builds SUMMARY indexes on close; plot_series should
    # read per-block min/max from the index (no data decompression).
    path = str(tmp_path / "plotsum.b2t")
    n = 200_000
    data = np.linspace(-1.0, 1.0, n)
    data[1234] = 7.0  # spikes the per-block summary must capture
    data[199_001] = -7.0

    @dataclasses.dataclass
    class VRow:
        v: float = blosc2.field(blosc2.float64(), default=0.0)

    arr = np.empty(n, dtype=[("v", "<f8")])
    arr["v"] = data
    table = blosc2.CTable(VRow, urlpath=path, mode="w", expected_size=n)
    table.extend(arr, validate=False)
    table.close()

    with StoreBrowser(path) as browser:
        series = browser.plot_series("/", column="v", max_points=2000)
        assert series["method"] == "summary"
        assert series["n"] == n
        assert np.isclose(np.nanmax(series["ymax"]), 7.0)
        assert np.isclose(np.nanmin(series["ymin"]), -7.0)


def test_plot_series_2d_column_with_layout(tmp_path):
    from blosc2.b2view.model import DataSliceLayout

    path = tmp_path / "plot2d.b2z"
    values = np.linspace(0, 1, 200 * 8).reshape(200, 8)
    with blosc2.TreeStore(str(path), mode="w") as store:
        store["/grid"] = values

    with StoreBrowser(str(path)) as browser:
        layout = DataSliceLayout.from_shape((200, 8))
        series = browser.plot_series("/grid", column=5, layout=layout, max_points=50)
        assert series["n"] == 200
        assert series["method"] == "reduce"
        # Column 5 over all rows, bucketed; envelope brackets the true column
        col = values[:, 5]
        assert np.isclose(np.nanmax(series["ymax"]), col.max())
        assert np.isclose(np.nanmin(series["ymin"]), col.min())


def test_plot_series_ctable_column_honors_row_filter(tmp_path):
    path = tmp_path / "plotct.b2z"
    with blosc2.TreeStore(str(path), mode="w") as store:
        store["/table"] = make_ctable(100)

    with StoreBrowser(str(path)) as browser:
        series = browser.plot_series("/table", column="y", max_points=1000)
        assert series["n"] == 100
        # In-memory TreeStore CTable has no summary index -> exact reduce
        assert series["method"] in ("summary", "reduce")
        assert np.isclose(np.nanmax(series["ymax"]), 99 * 1.5)
        assert np.isclose(np.nanmin(series["ymin"]), 0.0)

        # An active row filter restricts the plotted universe (and forces reduce)
        browser.set_filter("/table", "x >= 50")
        filtered = browser.plot_series("/table", column="y", max_points=1000)
        assert filtered["n"] == 50
        assert filtered["method"] == "reduce"
        assert np.isclose(np.nanmin(filtered["ymin"]), 50 * 1.5)
        assert np.isclose(np.nanmax(filtered["ymax"]), 99 * 1.5)
