"""Read-only CTable access through fsspec-backed B2Z archives."""

from __future__ import annotations

import dataclasses
import itertools
import zipfile

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


@pytest.mark.parametrize("policy", list(blosc2.CachePolicy))
@pytest.mark.parametrize("null_storage", ["mask", "sentinel"])
@pytest.mark.parametrize("deleted", [False, True])
def test_remote_ctable_utf8(tmp_path, policy, null_storage, deleted):
    @dataclasses.dataclass
    class TextRow:
        x: int
        text: str = blosc2.field(blosc2.utf8(null_storage=null_storage))

    values = ["", "café", "東京", "🌦️", None, "long" * 2000, "last"]
    local = blosc2.CTable(TextRow, list(enumerate(values)), create_summary_index=False)
    if deleted:
        local.delete([1, 3])
    url = remote_table_url(tmp_path, local)
    kwargs = {"cache_policy": policy}
    if policy == blosc2.CachePolicy.DISK:
        kwargs["cache_dir"] = tmp_path / "cache"
    with blosc2.CTable.open(tmp_path / "table.b2z") as local, blosc2.RemoteCTable(url, **kwargs) as remote:
        assert remote.nbytes == local.nbytes
        assert remote.cbytes == local.cbytes
        assert remote.cratio == local.cratio
        raw = remote["text"].raw
        assert isinstance(raw.offsets, blosc2.RemoteArray)
        assert isinstance(raw.data, blosc2.RemoteArray)
        for item in (
            0,
            -1,
            slice(None),
            slice(1, 4),
            slice(None, None, 2),
            slice(None, None, -1),
            [2, 0, 2],
        ):
            np.testing.assert_array_equal(remote["text"][item], local["text"][item])
        np.testing.assert_array_equal(remote["text"].is_null(), local["text"].is_null())
        assert remote["text"].null_count() == local["text"].null_count()
        assert list(remote["text"]) == list(local["text"])
        for expression in ('text == "東京"', 'text != ""', '(text >= "café") & (x > 0)'):
            np.testing.assert_array_equal(remote.where(expression)["x"][:], local.where(expression)["x"][:])
        assert str(remote[:3])
        before = (len(raw), raw._pending[:], raw._pending_chars)
        for write in (
            lambda: raw.append("new"),
            lambda: raw.extend(["new"]),
            lambda: raw.set_all(["new"]),
            lambda: raw.__setitem__(0, "new"),
        ):
            with pytest.raises(ValueError, match="read-only"):
                write()
            assert (len(raw), raw._pending, raw._pending_chars) == before
        copied = remote.copy()
        assert type(copied) is blosc2.CTable
        np.testing.assert_array_equal(copied["text"][:], local["text"][:])
        copied["text"][0] = "changed"
        raw[:2]
        requests = remote.traffic.requests
        raw[:2]
        if policy != blosc2.CachePolicy.NONE:
            assert remote.traffic.requests == requests
    for read in (
        lambda: raw[:0],
        lambda: len(raw),
        lambda: raw.dtype,
        lambda: raw.nbytes,
        lambda: raw.flush(),
    ):
        with pytest.raises(RuntimeError, match="closed"):
            read()


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


@pytest.mark.parametrize("empty", [False, True])
def test_remote_utf8_nested_lifetime(tmp_path, empty):
    @dataclasses.dataclass
    class TextRow:
        text: str = blosc2.field(blosc2.utf8())

    local = blosc2.CTable(TextRow, [] if empty else [("café",), ("東京",)], create_summary_index=False)
    local.rename_column("text", "nested.text")
    path = tmp_path / "tree.b2z"
    with blosc2.TreeStore(path, mode="w", threshold=0) as tree:
        tree["group/table"] = local
    url = f"memory://{tmp_path.name}-tree.b2z"
    fsspec.filesystem("memory").pipe(url, path.read_bytes())
    with blosc2.RemoteStore(url) as store:
        table = store["group/table"]
        raw = table["nested.text"].raw
        np.testing.assert_array_equal(raw[:], local["nested.text"][:])
        assert raw.nbytes == local["nested.text"].raw.nbytes
        view = table[:1]
        store.refresh()
        for read in (lambda: raw[:0], lambda: raw.shape, lambda: raw.cbytes, lambda: view["nested.text"][:]):
            with pytest.raises(RuntimeError, match="stale"):
                read()
        table.close()
        table = store["group/table"]
    try:
        np.testing.assert_array_equal(table["nested.text"][:], local["nested.text"][:])
        dest = tmp_path / "copy.b2z"
        table.to_b2z(dest)
        with blosc2.CTable.open(dest) as copied:
            np.testing.assert_array_equal(copied["nested.text"][:], local["nested.text"][:])
    finally:
        table.close()


@pytest.mark.parametrize("bad_data", [None, np.arange(4, dtype="int64")])
def test_remote_utf8_invalid_companion(tmp_path, bad_data):
    @dataclasses.dataclass
    class TextRow:
        x: int
        text: str = blosc2.field(blosc2.utf8())

    local = blosc2.CTable(TextRow, [(1, "abc")], create_summary_index=False)
    path = tmp_path / "bad.b2z"
    local.to_b2z(path)
    # Repack only the companion, retaining all table metadata unchanged.
    with zipfile.ZipFile(path) as archive:
        members = {info.filename: archive.read(info) for info in archive.infolist()}
    member = "_cols/text.utf8.b2nd"
    if bad_data is None:
        del members[member]
    else:
        members[member] = blosc2.asarray(bad_data).to_cframe()
    broken = tmp_path / "broken.b2z"
    with zipfile.ZipFile(broken, "w", zipfile.ZIP_STORED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    url = f"memory://{tmp_path.name}-broken.b2z"
    fsspec.filesystem("memory").pipe(url, broken.read_bytes())
    with blosc2.RemoteCTable(url) as table:
        assert table["x"][0] == 1
        handles = len(table._storage._arrays)
        for _ in range(2):
            with pytest.raises((ValueError, NotImplementedError), match=r"(data array|unavailable)"):
                table["text"][:]
            assert len(table._storage._arrays) == handles


@pytest.mark.parametrize("policy", list(blosc2.CachePolicy))
@pytest.mark.parametrize("blocks", [False, True])
def test_remote_utf8_bounded_transfer(tmp_path, monkeypatch, policy, blocks):
    from blosc2 import proxy_source
    from blosc2._utf8_array import UTF8Array

    # Lower only the cost threshold so a small fixture exercises both native paths.
    monkeypatch.setattr(proxy_source, "BLOCK_MIN_CBYTES", 0 if blocks else 1 << 30)

    @dataclasses.dataclass
    class TextRow:
        text: str = blosc2.field(blosc2.utf8())

    rng = np.random.default_rng(7)
    lengths = rng.integers(80, 120, size=20000)
    offsets = np.concatenate(([0], np.cumsum(lengths)))
    data = rng.integers(32, 127, size=offsets[-1], dtype="uint8")
    values = [data[a:b].tobytes().decode() for a, b in itertools.pairwise(offsets)]
    local = blosc2.CTable(TextRow, [(s,) for s in values], create_summary_index=False)
    local._cols["text"] = UTF8Array(
        blosc2.utf8(),
        blosc2.asarray(offsets, chunks=(8192,), blocks=(256,)),
        blosc2.asarray(data, chunks=(262144,), blocks=(4096,)),
    )
    url = remote_table_url(tmp_path, local)
    # Table export reconstructs UTF-8 arrays with default grids. Install the
    # smaller fixture grids in the archive to exercise multi-chunk reads.
    raw = local["text"].raw
    replacements = {
        "_cols/text.b2nd": raw.offsets.to_cframe(),
        "_cols/text.utf8.b2nd": raw.data.to_cframe(),
    }
    path = tmp_path / "grids.b2z"
    with zipfile.ZipFile(tmp_path / "table.b2z") as source, zipfile.ZipFile(path, "w") as dest:
        for info in source.infolist():
            dest.writestr(info, replacements.get(info.filename, source.read(info)))
    fsspec.filesystem("memory").pipe(url, path.read_bytes())
    kwargs = {"cache_policy": policy}
    if policy != blosc2.CachePolicy.NONE:
        kwargs["max_cache_bytes"] = 1 << 20
    if policy == blosc2.CachePolicy.DISK:
        kwargs["cache_dir"] = tmp_path / "cache"
    plans = []
    original = proxy_source.ByteRangeNDSource.block_plan

    def record_plan(source, *args):
        plans.append(source.dataset)
        return original(source, *args)

    monkeypatch.setattr(proxy_source.ByteRangeNDSource, "block_plan", record_plan)
    with blosc2.RemoteCTable(url, **kwargs) as table:
        assert not table._storage._arrays[1:]  # Only validity is opened eagerly.
        assert table.nbytes > 0
        assert table.cbytes > 0
        assert table.traffic.nbytes < (tmp_path / "table.b2z").stat().st_size // 4
        # Metadata prefetch can warm the entire small offsets member.
        table["text"].raw.offsets.trim_cache(0)
        table["text"].raw.data.trim_cache(0)
        before = table.traffic.nbytes
        np.testing.assert_array_equal(table["text"][10000:10003], values[10000:10003])
        assert table.traffic.nbytes - before < len(data) // 4
        if blocks:
            assert set(plans) >= {"_cols/text", "_cols/text.utf8"}
        else:
            assert not plans
        before = table.traffic.requests
        np.testing.assert_array_equal(table["text"][10000:10003], values[10000:10003])
        if policy != blosc2.CachePolicy.NONE:
            assert table.traffic.requests == before
        assert table.cache_bytes <= kwargs.get("max_cache_bytes", 0)
        boundary = int(np.searchsorted(offsets, 262144))
        for item in (slice(8190, 8195), slice(boundary - 1, boundary + 2), slice(9990, 10010, 3)):
            np.testing.assert_array_equal(table["text"][item], values[item])
    if policy == blosc2.CachePolicy.DISK:
        with blosc2.RemoteCTable(url, **kwargs) as table:
            before = table.traffic.requests
            np.testing.assert_array_equal(table["text"][10000:10003], values[10000:10003])
            assert table.traffic.requests == before


def test_remote_store_rejects_table_root(tmp_path):
    url = remote_table_url(
        tmp_path,
        blosc2.CTable(Row, [(1, [1, 2], "one")], create_summary_index=False),
        "root",
    )
    with pytest.raises(ValueError, match="use RemoteCTable"):
        blosc2.RemoteStore(url)


@pytest.mark.parametrize("policy", list(blosc2.CachePolicy))
def test_open_dispatches_local_and_remote_tables(tmp_path, policy):
    @dataclasses.dataclass
    class TextRow:
        text: str = blosc2.field(blosc2.utf8(null_storage="mask"))

    local = blosc2.CTable(TextRow, [("café",), (None,), ("東京",)], create_summary_index=False)
    url = remote_table_url(tmp_path, local)
    with blosc2.open(tmp_path / "table.b2z") as table:
        assert type(table) is blosc2.CTable
        assert table["text"][0] == "café"
    options = {"cache_policy": policy}
    if policy == blosc2.CachePolicy.DISK:
        options["cache_dir"] = tmp_path / "cache"
    with blosc2.open(url, **options) as table:
        assert isinstance(table, blosc2.RemoteCTable)
        assert table.cache_policy == policy
        assert table["text"][-1] == "東京"
        assert table["text"].null_count() == 1
    with pytest.raises(RuntimeError, match="closed"):
        table["text"][:]
    with blosc2.open(url) as table:
        assert table.cache_policy == blosc2.CachePolicy.MEMORY


@pytest.mark.parametrize("suffix", [".b2z", ""])
def test_open_dispatches_remote_table_hierarchy(tmp_path, suffix, monkeypatch):
    local = blosc2.CTable(Row, [(1, [1, 2], "one")], create_summary_index=False)
    path = tmp_path / "tree.b2z"
    with blosc2.TreeStore(path, mode="w", threshold=0) as tree:
        tree["group/table"] = local
        tree["group/array"] = blosc2.arange(5)
    url = f"memory://{tmp_path.name}-tree{suffix}"
    format_options = {} if suffix else {"source_format": "b2z"}
    fsspec.filesystem("memory").pipe(url, path.read_bytes())
    reads = []
    filesystem_type = type(fsspec.filesystem("memory"))
    original = filesystem_type.cat_file

    def counted(self, path, start=None, end=None, **kwargs):
        result = original(self, path, start=start, end=end, **kwargs)
        reads.append(len(result))
        return result

    monkeypatch.setattr(filesystem_type, "cat_file", counted)
    with blosc2.open(url, **format_options) as store:
        assert isinstance(store, blosc2.RemoteStore)
        table = store["group/table"]
    assert table["x"][0] == 1
    table.close()
    for target, options in (
        (url, {"dataset": "group/table"}),
        (url + "::group/table", {}),
        (url + "/group/table", {"lazy": True}),
    ):
        if not suffix and target != url:
            continue  # URL suffix parsing requires a recognizable archive suffix.
        reads.clear()
        with blosc2.open(target, **options, **format_options) as table:
            assert isinstance(table, blosc2.RemoteCTable)
            assert table["x"][0] == 1
            assert table.traffic.requests == len(reads)
            assert table.traffic.nbytes == sum(reads)
    with blosc2.open(url, dataset="group", **format_options) as group:
        assert isinstance(group, blosc2.RemoteStore)
        assert group.keys() == ["array", "table"]
        group.refresh()
        assert group.keys() == ["array", "table"]
    with blosc2.open(url, dataset="group/array", **format_options) as array:
        assert isinstance(array, blosc2.RemoteArray)
        np.testing.assert_array_equal(array[:], np.arange(5))
    with blosc2.open(url, lazy=False, cache_dir=tmp_path / "localized", **format_options) as store:
        assert isinstance(store, blosc2.TreeStore)
        assert store["group/table"]["x"][0] == 1
