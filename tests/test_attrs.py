"""The user-facing metadata name shares the existing storage and access rules."""

from dataclasses import dataclass

import pytest

import blosc2


@dataclass
class Row:
    value: int = 0


@pytest.mark.parametrize(
    "factory",
    [
        lambda: blosc2.zeros(4),
        blosc2.SChunk,
        blosc2.ObjectArray,
        lambda: blosc2.BatchArray(items_per_block=2),
        lambda: blosc2.ListArray(item_spec=blosc2.int32()),
        lambda: blosc2.CTable(Row),
        lambda: blosc2.zeros(4) + 1,
        lambda: blosc2.Proxy(blosc2.zeros(4)),
    ],
)
def test_attrs_alias(factory):
    obj = factory()
    assert obj.attrs is obj.vlmeta
    obj.attrs["units"] = "kelvin"
    assert obj.vlmeta["units"] == "kelvin"
    obj.vlmeta["units"] = "celsius"
    assert obj.attrs["units"] == "celsius"
    del obj.attrs["units"]
    assert "units" not in obj.vlmeta


def test_attrs_persistence_and_read_only(tmp_path):
    path = tmp_path / "array.b2nd"
    array = blosc2.zeros(4, urlpath=path)
    array.attrs["units"] = "kelvin"
    reopened = blosc2.open(path, mode="r")
    assert reopened.attrs["units"] == "kelvin"
    with pytest.raises(ValueError):
        reopened.attrs["units"] = "celsius"

    path = tmp_path / "tree.b2z"
    with blosc2.TreeStore(path, mode="w") as tree:
        tree.attrs["units"] = "kelvin"
        assert tree.vlmeta["units"] == "kelvin"
        tree.vlmeta["units"] = "celsius"
        assert tree.attrs["units"] == "celsius"
    with blosc2.TreeStore(path, mode="r") as tree:
        assert tree.attrs["units"] == "celsius"
        with pytest.raises(ValueError):
            tree.attrs["units"] = "kelvin"
