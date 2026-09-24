"""Optional interoperability checks using files written by PyTables itself."""

from __future__ import annotations

import numpy as np
import pytest

import blosc2

fsspec = pytest.importorskip("fsspec")
tables = pytest.importorskip("tables", reason="PyTables is optional")


class NativeRow(tables.IsDescription):
    id = tables.Int32Col(pos=0)
    value = tables.Float64Col(pos=1)
    active = tables.BoolCol(pos=2)
    label = tables.StringCol(8, pos=3)


def native_pytables_url(tmp_path, name, *, index_kind=None, csi=False):
    path = tmp_path / name
    values = np.random.default_rng(4).permutation(2049)
    with tables.open_file(path, mode="w") as h5file:
        table = h5file.create_table("/", "table", NativeRow, expectedrows=len(values))
        data = np.empty(len(values), dtype=table.dtype)
        data["id"] = values
        data["value"] = values + 0.5
        data["active"] = values % 2 == 0
        data["label"] = [str(value).encode() for value in values]
        table.append(data)
        table.flush()
        table.attrs.owner = "pytables"
        if csi:
            table.cols.id.create_csindex()
        elif index_kind is not None:
            table.cols.id.create_index(kind=index_kind, optlevel=6)
            table.cols.label.create_index(kind=index_kind, optlevel=6)
        expected = table.read_where("(id >= 15) & (id < 25)")
        is_csi = table.cols.id.index.is_csi if table.cols.id.is_indexed else False

    url = f"memory://{tmp_path.name}/{name}"
    fsspec.filesystem("memory").pipe(url, path.read_bytes())
    return url, data, expected, is_csi


def test_native_pytables_table_and_full_indexes(tmp_path):
    url, data, expected, _ = native_pytables_url(tmp_path, "native-full.h5", index_kind="full")
    with blosc2.RemoteCTable(url, dataset="table") as table:
        catalog = table._get_index_catalog()
        assert catalog["id"]["kind"] == catalog["label"]["kind"] == "opsi"
        assert table.schema_dict()["columns"][2]["kind"] == "bool"
        assert table.attrs["owner"] == b"pytables"
        assert "label" in str(table[:3])
        np.testing.assert_array_equal(table.where("(id >= 15) & (id < 25)").id[:], expected["id"])
        np.testing.assert_array_equal(
            table.where("(id >= 15) & active").id[:], data["id"][(data["id"] >= 15) & data["active"]]
        )
        np.testing.assert_array_equal(
            table.where(table.label == b"5").id[:], data["id"][data["label"] == b"5"]
        )


def test_native_pytables_csi(tmp_path):
    url, _, expected, is_csi = native_pytables_url(tmp_path, "native-csi.h5", csi=True)
    assert is_csi
    with blosc2.RemoteCTable(url, dataset="table") as table:
        descriptor = table._get_index_catalog()["id"]
        assert descriptor["kind"] == "opsi"
        assert descriptor["opsi"]["is_csi"]
        np.testing.assert_array_equal(table.where("(id >= 15) & (id < 25)").id[:], expected["id"])


def test_local_pytables_disk_cache_reuse_and_invalidation(tmp_path):
    native_pytables_url(tmp_path, "local-csi.h5", csi=True)
    path = tmp_path / "local-csi.h5"
    cache_dir = tmp_path / "cache"
    options = {"path": "table", "cache_dir": cache_dir}

    with blosc2.open(path, **options) as table:
        assert sorted(table.where("id < 10").id[:].tolist()) == list(range(10))
        generation = table._storage._owner.generation
        descriptor = table._get_index_catalog()["id"]
        assert descriptor["persistent"]

    with blosc2.open(path, **options) as table:
        assert table._storage._owner.generation == generation
        assert table._get_index_catalog()["id"]["opsi"]["values_path"] == descriptor["opsi"]["values_path"]

    import h5py

    with h5py.File(path, "r+") as h5file:
        row = h5file["table"][0]
        row["value"] = 999
        h5file["table"][0] = row

    with blosc2.open(path, **options) as table:
        assert table._storage._owner.generation != generation
        assert table["value"][0] == 999
        assert sorted(table.where("id < 10").id[:].tolist()) == list(range(10))


def test_native_pytables_light_index_falls_back_to_scan(tmp_path):
    url, _, expected, _ = native_pytables_url(tmp_path, "native-light.h5", index_kind="light")
    with blosc2.RemoteCTable(url, dataset="table") as table:
        assert table._get_index_catalog() == {}
        np.testing.assert_array_equal(table.where("(id >= 15) & (id < 25)").id[:], expected["id"])
