from __future__ import annotations

from dataclasses import dataclass

import pytest

import blosc2


@dataclass
class Product:
    code: str = blosc2.field(blosc2.string(max_length=8))
    qty: int = blosc2.field(blosc2.int32())
    tags: list[str] = blosc2.field(  # noqa: RUF009
        blosc2.list(blosc2.string(max_length=16), nullable=True, batch_rows=2)
    )


DATA = [
    ("a", 1, ["x", "y"]),
    ("b", 2, []),
    ("c", 3, None),
    ("d", 4, ["z"]),
]


def test_ctable_varlen_append_extend_and_reads():
    t = blosc2.CTable(Product)
    t.append(DATA[0])
    t.extend(DATA[1:])

    assert len(t) == 4
    assert t.tags[0] == ["x", "y"]
    assert t.tags[1:4] == [[], None, ["z"]]
    assert t.row[2].tags[0] is None

    t.tags[2] = ["r", "s"]
    assert t.tags[2] == ["r", "s"]


def test_ctable_varlen_where_select_head_tail_and_compact():
    t = blosc2.CTable(Product, new_data=DATA)
    view = t.where(t.qty >= 2)
    assert view.tags[:] == [[], None, ["z"]]
    sel = t.select(["code", "tags"])
    assert sel.tags[:] == [["x", "y"], [], None, ["z"]]
    assert t.head(2).tags[:] == [["x", "y"], []]
    assert t.tail(2).tags[:] == [None, ["z"]]

    t.delete([1])
    t.compact()
    assert t.tags[:] == [["x", "y"], None, ["z"]]


def test_ctable_varlen_persistence_save_load_open(tmp_path):
    path = tmp_path / "products.b2d"
    t = blosc2.CTable(Product, new_data=DATA, urlpath=str(path), mode="w")
    t.close()

    opened = blosc2.CTable.open(str(path), mode="r")
    assert opened.tags[:] == [["x", "y"], [], None, ["z"]]

    loaded = blosc2.CTable.load(str(path))
    assert loaded.tags[:] == [["x", "y"], [], None, ["z"]]
    loaded.tags[1] = ["changed"]
    assert loaded.tags[1] == ["changed"]

    save_path = tmp_path / "products-save.b2d"
    loaded.save(str(save_path))
    reopened = blosc2.CTable.open(str(save_path), mode="r")
    assert reopened.tags[:] == [["x", "y"], ["changed"], None, ["z"]]


def test_ctable_varlen_arrow_roundtrip():
    pytest.importorskip("pyarrow")

    t = blosc2.CTable(Product, new_data=DATA)
    arrow = t.to_arrow()
    assert arrow.column("tags").to_pylist() == [["x", "y"], [], None, ["z"]]

    roundtrip = blosc2.CTable.from_arrow(arrow)
    assert roundtrip.tags[:] == [["x", "y"], [], None, ["z"]]
