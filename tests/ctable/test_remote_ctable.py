"""Read-only CTable access through fsspec-backed B2Z archives."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import blosc2

fsspec = pytest.importorskip("fsspec")


@dataclasses.dataclass
class Row:
    x: int = blosc2.field(blosc2.int64(null_storage="mask"))
    vec: np.ndarray = blosc2.field(  # noqa: RUF009
        blosc2.ndarray((2,), dtype=blosc2.float32())
    )
    tag: str = blosc2.field(blosc2.string(max_length=8), default="")


def remote_table_url(tmp_path, table, name="table"):
    path = tmp_path / f"{name}.b2z"
    table.to_b2z(path)
    url = f"memory://{tmp_path.name}-{name}.b2z"
    fsspec.filesystem("memory").pipe(url, path.read_bytes())
    return url


def test_remote_ctable_fixed_width_reads_and_queries(tmp_path):
    rows = [
        (1, np.array([1, 2], dtype=np.float32), "one"),
        (None, np.array([3, 4], dtype=np.float32), "missing"),
        (3, np.array([5, 6], dtype=np.float32), "three"),
        (4, np.array([7, 8], dtype=np.float32), "four"),
    ]
    local = blosc2.CTable(Row, rows, create_summary_index=False)
    persistent = str(tmp_path / "persistent.b2d")
    local.save(persistent)
    with blosc2.CTable.open(persistent, mode="a") as table:
        table.attrs["source"] = "test"
    with blosc2.CTable.open(persistent) as table:
        url = remote_table_url(tmp_path, table)

    with blosc2.RemoteCTable(url) as table:
        assert len(table) == 4
        assert table.col_names == ["x", "vec", "tag"]
        assert table.attrs["source"] == "test"
        assert table["x"][0] == 1
        assert table["x"].is_null().tolist() == [False, True, False, False]
        assert table["tag"][1:3].tolist() == ["missing", "three"]
        np.testing.assert_array_equal(table["vec"][2], [5, 6])
        np.testing.assert_array_equal(table.where(table["x"] > 2)["x"][:], [3, 4])
        assert table["x"].sum() == 8
        assert table.source["urlpath"] == url
        assert table.cache_policy is blosc2.CachePolicy.MEMORY
        assert table.cache_bytes > 0
        with pytest.raises(ValueError, match="read-only"):
            table.append((5, [9, 10], "five"))

    with pytest.raises(RuntimeError, match="closed"):
        table["tag"][:]


def test_remote_store_returns_table_with_independent_lifetime(tmp_path):
    source = tmp_path / "tree.b2z"
    table = blosc2.CTable(Row, [(1, [1, 2], "one"), (2, [3, 4], "two")], create_summary_index=False)
    with blosc2.TreeStore(source, mode="w", threshold=0) as root:
        root["/group/table"] = table
        root["/group/array"] = blosc2.arange(3)
    url = f"memory://{tmp_path.name}-tree.b2z"
    fsspec.filesystem("memory").pipe(url, source.read_bytes())

    with blosc2.RemoteStore(url) as store:
        assert store.kind("group/table") == "ctable"
        remote = store["group/table"]
        assert isinstance(remote, blosc2.RemoteCTable)
    np.testing.assert_array_equal(remote["x"][:], [1, 2])
    remote.close()

    with blosc2.RemoteCTable(url, dataset="group/table") as direct:
        np.testing.assert_array_equal(direct["x"][:], [1, 2])


def test_remote_ctable_unsupported_column_is_lazy(tmp_path):
    @dataclasses.dataclass
    class Mixed:
        x: int = 0
        text: str = blosc2.field(blosc2.vlstring(), default="")

    url = remote_table_url(
        tmp_path,
        blosc2.CTable(Mixed, [(1, "a"), (2, "bb")], create_summary_index=False),
        "mixed",
    )
    with blosc2.RemoteCTable(url, cache_policy=blosc2.CachePolicy.NONE) as table:
        np.testing.assert_array_equal(table["x"][:], [1, 2])
        with pytest.raises(NotImplementedError, match="variable-length column 'text'"):
            table["text"][:]


def test_remote_ctable_deleted_rows_and_disk_cache(tmp_path):
    source = str(tmp_path / "deleted.b2d")
    table = blosc2.CTable(Row, urlpath=source, mode="w", expected_size=8, create_summary_index=False)
    table.extend([(i, [i, i + 1], str(i)) for i in range(6)])
    table.delete([1, 4])
    table.close()
    with blosc2.CTable.open(source) as table:
        url = remote_table_url(tmp_path, table, "deleted")

    cache = tmp_path / "cache"
    with blosc2.RemoteCTable(url, cache_dir=cache) as remote:
        np.testing.assert_array_equal(remote["x"][:], [0, 2, 3, 5])
        assert remote["x"][1] == 2
        assert list(remote["x"]) == [0, 2, 3, 5]
        assert remote["x"].sum() == 10
        assert remote.cache_policy is blosc2.CachePolicy.DISK

    with blosc2.RemoteCTable(url, cache_dir=cache) as reopened:
        np.testing.assert_array_equal(reopened["x"][:], [0, 2, 3, 5])


def test_remote_store_rejects_table_root(tmp_path):
    url = remote_table_url(
        tmp_path,
        blosc2.CTable(Row, [(1, [1, 2], "one")], create_summary_index=False),
        "root",
    )
    with pytest.raises(ValueError, match="use RemoteCTable"):
        blosc2.RemoteStore(url)
