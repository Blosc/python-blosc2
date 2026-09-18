"""Read-only CTable access through fsspec-backed B2Z archives."""

from __future__ import annotations

import dataclasses
import itertools
import os
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


@pytest.mark.parametrize("include_note", [False, True])
@pytest.mark.parametrize("nrows", [0, 2, 20, 21])
def test_remote_example_total_timing(tmp_path, capsys, include_note, nrows):
    import runpy
    from pathlib import Path
    from types import SimpleNamespace

    fields = [("x", int)]
    rows = [(i,) for i in range(nrows)]
    if include_note:
        fields.append(("note", str, blosc2.field(blosc2.utf8())))
        rows = [(i, f"café 東京 #{i}") for i in range(nrows)]
    local = blosc2.CTable(dataclasses.make_dataclass("Sample", fields), rows, create_summary_index=False)
    url = remote_table_url(tmp_path, local)
    example = runpy.run_path(str(Path(__file__).resolve().parents[2] / "examples/ctable/remote_handling.py"))
    ticks = iter(range(10))
    access = example["access_table"]
    access.__globals__["time"] = SimpleNamespace(perf_counter=lambda: next(ticks))
    access(SimpleNamespace(url=url, cache_dir=None))
    output = capsys.readouterr().out
    expected_ms = 5000 if include_note else 3000
    assert f"\n\nTotal network : {expected_ms:7.1f} ms  (" in output
    assert "\nRetained cache:" in output
    assert "  - Total network" not in output
    start = max(0, nrows // 2 - 2)
    stop = min(start + 5, nrows)
    heading = f"Sample rows [{start}:{stop}] around the midpoint (1st fetch):\n"
    sample = output.split(heading)[1].split("\nTiming & Network Traffic:")[0].strip()
    assert sample == str(local[start:stop]).strip()


def test_remote_example_cache_dir(tmp_path, capsys, monkeypatch):
    import runpy
    import sys
    from pathlib import Path

    local = blosc2.CTable(dataclasses.make_dataclass("Sample", [("x", int)]), [(i,) for i in range(20)])
    url = remote_table_url(tmp_path, local)
    cache_dir = tmp_path / "cache"
    script = Path(__file__).resolve().parents[2] / "examples/ctable/remote_handling.py"
    example = runpy.run_path(str(script))
    monkeypatch.setattr(sys, "argv", [str(script), url, "--cache-dir", str(cache_dir)])
    for _ in range(2):
        assert example["main"]() == 0
        output = capsys.readouterr().out
        assert "cache_policy : DISK" in output
    assert any(cache_dir.iterdir())
    assert "(0 requests," in output.split("Total network :")[1]


def test_disk_cache_metadata_key_order(tmp_path, monkeypatch):
    local = blosc2.CTable(dataclasses.make_dataclass("Sample", [("x", int)]), [(i,) for i in range(20)])
    url = remote_table_url(tmp_path, local)
    cache_dir = tmp_path / "cache"
    with blosc2.open(url, cache_dir=cache_dir) as table:
        np.testing.assert_array_equal(table["x"][:], np.arange(20))

    fs = fsspec.filesystem("memory")
    original_info = type(fs).info

    def reordered_info(self, path, **kwargs):
        return dict(reversed(list(original_info(self, path, **kwargs).items())))

    monkeypatch.setattr(type(fs), "info", reordered_info)
    with blosc2.open(url, cache_dir=cache_dir) as table:
        np.testing.assert_array_equal(table["x"][:], np.arange(20))
        assert table.traffic.requests == 0

    def changed_info(self, path, **kwargs):
        return {**original_info(self, path, **kwargs), "ETag": "changed"}

    monkeypatch.setattr(type(fs), "info", changed_info)
    # Immutable reopening trusts the persisted identity, even if remote metadata changes.
    with blosc2.open(url, cache_dir=cache_dir) as table:
        np.testing.assert_array_equal(table["x"][:], np.arange(20))


@pytest.mark.parametrize("max_concurrency", [1, 8])
@pytest.mark.parametrize("legacy", [False, True])
def test_disk_cache_reuses_ctable_bootstrap(tmp_path, monkeypatch, max_concurrency, legacy):
    schema = dataclasses.make_dataclass("Sample", [("x", float), ("note", str, blosc2.field(blosc2.utf8()))])
    values = np.random.default_rng(42).random(20_000)
    local = blosc2.CTable(
        schema,
        [(x, f"東京 café #{i}") for i, x in enumerate(values)],
        create_summary_index=False,
    )
    local._cols["x"] = blosc2.asarray(values, chunks=(2000,), blocks=(200,))
    local.attrs["description"] = "Persistent UTF-8 metadata"
    url = remote_table_url(tmp_path, local)
    # Export rechunks columns; retain a multi-chunk fixture for cold-row checks.
    path = tmp_path / "grids.b2z"
    with zipfile.ZipFile(tmp_path / "table.b2z") as source, zipfile.ZipFile(path, "w") as dest:
        for info in source.infolist():
            data = local._cols["x"].to_cframe() if info.filename == "_cols/x.b2nd" else source.read(info)
            dest.writestr(info, data)
    fsspec.filesystem("memory").pipe(url, path.read_bytes())
    cache_dir = tmp_path / "cache"
    options = {"cache_dir": cache_dir, "max_concurrency": max_concurrency}
    with blosc2.open(url, **options) as table:
        nbytes = table.nbytes
        attrs = dict(table.attrs)
        sample = list(table[9998:10003])
        notes = table["note"][-5:].tolist()
        if legacy:
            from fsspec.utils import tokenize

            owner = table._remote_storage()._owner
            # Before object_info was persisted, stamps used native backend types
            # (including memory:// creation datetimes), not normalized strings.
            info = owner.archive._fs.info(owner.archive._path)
            for key, source in owner.sources.items():
                stamp = tokenize(url, sorted(info.items()), key, source.member_offset, source.member_length)
                owner.caches[key].schunk.vlmeta["proxy-stamp"] = stamp
            owner.archive.metadata.pop("ctable_seeds")
            owner.archive.metadata.pop("object_info")
            owner.archive.metadata.pop("member_stamps")
            owner.attrs.clear()

    if legacy:
        # Old caches acquire bootstraps and attributes on their next access.
        with blosc2.open(url, **options) as table:
            assert table.nbytes == nbytes
            assert dict(table.attrs) == attrs

    def unexpected_read(*args, **kwargs):
        pytest.fail("Warm metadata/rows must not download archive bytes")

    with monkeypatch.context() as patch:
        patch.setattr(type(fsspec.filesystem("memory")), "cat_file", unexpected_read)
        patch.setattr(type(fsspec.filesystem("memory")), "info", unexpected_read)
        with blosc2.open(url, **options) as table:
            assert table.nbytes == nbytes
            assert dict(table.attrs) == attrs
            assert list(table[9998:10003]) == sample
            assert table["note"][-5:].tolist() == notes
            assert table.traffic.requests == 0

    # Restored bootstraps must still support transport for uncached rows.
    with blosc2.open(url, **options) as table:
        np.testing.assert_array_equal(table["x"][:5], values[:5])
        assert table["note"][:5].tolist() == [f"東京 café #{i}" for i in range(5)]
        assert table.traffic.requests > 0


@pytest.mark.parametrize(
    "policy", [blosc2.CachePolicy.NONE, blosc2.CachePolicy.MEMORY, blosc2.CachePolicy.DISK]
)
def test_remote_ctable_refresh(tmp_path, policy):
    schema = dataclasses.make_dataclass("Sample", [("x", int), ("note", str, blosc2.field(blosc2.utf8()))])
    local = blosc2.CTable(schema, [(1, "old"), (2, "café")], create_summary_index=False)
    url = remote_table_url(tmp_path, local)
    options = {"cache_policy": policy, "max_concurrency": 2, "row_buffer_bytes": 1 << 20}
    if policy is blosc2.CachePolicy.DISK:
        options["cache_dir"] = tmp_path / "cache"
    with blosc2.RemoteCTable(url, **options) as table:
        column, raw = table["x"], table["x"].raw
        view, projection = table[:1], table.select("x")
        assert column[:].tolist() == [1, 2]
        changed_schema = dataclasses.make_dataclass(
            "Changed", [("x", int), ("note", str, blosc2.field(blosc2.utf8())), ("active", bool)]
        )
        changed_path = str(tmp_path / "changed.b2d")
        with blosc2.CTable(
            changed_schema, [(9, "東京", True)], create_summary_index=False, urlpath=changed_path, mode="w"
        ) as changed:
            changed.attrs["version"] = 2
        with blosc2.CTable.open(changed_path) as changed:
            remote_table_url(tmp_path, changed, "changed")
        fsspec.filesystem("memory").pipe(url, (tmp_path / "changed.b2z").read_bytes())
        assert table.refresh() is None
        assert table.col_names == ["x", "note", "active"]
        assert table.nrows == 1
        assert table["x"][:].tolist() == [9]
        assert table["note"][:].tolist() == ["東京"]
        assert dict(table.attrs) == {"version": 2}
        assert table.max_concurrency == 2
        assert table.row_buffer_bytes == 1 << 20
        for read in (lambda: column[:], lambda: raw[:], lambda: list(view), lambda: list(projection)):
            with pytest.raises(RuntimeError, match=r"stale|closed"):
                read()
        table.refresh()
        assert table["x"][:].tolist() == [9]
    if policy is blosc2.CachePolicy.DISK:
        with blosc2.RemoteCTable(url, **options) as table:
            assert table["note"][:].tolist() == ["東京"]


@pytest.mark.parametrize("failure", ["discovery", "initialization", "publication"])
def test_remote_ctable_failed_refresh(tmp_path, monkeypatch, failure):
    local = blosc2.CTable(dataclasses.make_dataclass("Sample", [("x", int)]), [(1,), (2,)])
    url = remote_table_url(tmp_path, local)
    cache = tmp_path / "cache"
    with blosc2.RemoteCTable(url, cache_dir=cache) as table:
        column = table["x"]
        assert column[:].tolist() == [1, 2]
        owner = table._remote_storage()._owner
        generation = owner.generation

        def fail(*args, **kwargs):
            raise OSError("refresh failed")

        with monkeypatch.context() as patch:
            if failure == "discovery":
                patch.setattr(type(fsspec.filesystem("memory")), "info", fail)
            elif failure == "initialization":
                from blosc2.ctable_storage import RemoteTableStorage

                patch.setattr(RemoteTableStorage, "open_valid_rows", fail)
            else:
                patch.setattr(owner.disk, "publish", fail)
            with pytest.raises(OSError, match="refresh failed"):
                table.refresh()
        assert owner.generation == generation
        assert column[:].tolist() == [1, 2]
        assert table["x"][:].tolist() == [1, 2]
        table.refresh()
        assert table["x"][:].tolist() == [1, 2]


def test_remote_ctable_is_cache_mutable(tmp_path, monkeypatch):
    local = blosc2.CTable(dataclasses.make_dataclass("Sample", [("x", int)]), [(1,)])
    url = remote_table_url(tmp_path, local)
    with blosc2.RemoteCTable(url) as table:
        assert isinstance(table, blosc2.RemoteObject)
        assert isinstance(table, blosc2.CTable)
        assert table.mutable is False
        table.mutable = True
        assert table.mutable is True
        with pytest.raises(TypeError, match="boolean"):
            table.mutable = "invalid"
        assert table.is_cache_mutable is True
        with monkeypatch.context() as patch:
            patch.setattr(table._remote_storage()._owner, "is_mutable", False)
            assert table.is_cache_mutable is False
        with pytest.raises(AttributeError):
            table.is_cache_mutable = False
    with pytest.raises(RuntimeError, match="closed"):
        _ = table.is_cache_mutable
    with pytest.raises(RuntimeError, match="closed"):
        _ = table.mutable


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


def test_remote_ctable_nullable_vlstring_none_cache(tmp_path, monkeypatch):
    @dataclasses.dataclass
    class TextRow:
        text: str = blosc2.field(blosc2.vlstring(nullable=True, batch_rows=32))

    rng = np.random.default_rng(42)
    values = np.asarray(
        [
            None if i % 19 == 0 else "" if i % 23 == 0 else f"café 東京 {i} " + rng.bytes(1024).hex()
            for i in range(160)
        ],
        dtype=object,
    )
    local = blosc2.CTable(TextRow, [(value,) for value in values], create_summary_index=False)
    url = remote_table_url(tmp_path, local, "vlstring-none")
    fs = fsspec.filesystem("memory")
    reads = []
    original = type(fs).cat_file

    def counted(self, path, start=None, end=None, **kwargs):
        reads.append((start, end))
        return original(self, path, start=start, end=end, **kwargs)

    monkeypatch.setattr(type(fs), "cat_file", counted)
    with blosc2.RemoteCTable(url, cache_policy=blosc2.CachePolicy.NONE) as remote:
        assert remote.col_names == ["text"]
        column = remote["text"]
        reads.clear()
        assert column[-1] == values[-1]
        transferred = sum(end - start for start, end in reads)
        assert transferred < (tmp_path / "vlstring-none.b2z").stat().st_size
        assert column[0] is None
        assert column[23] == ""
        assert column[31:34] == values[31:34].tolist()


@pytest.mark.parametrize("policy", [blosc2.CachePolicy.MEMORY, blosc2.CachePolicy.DISK])
def test_remote_ctable_vlstring_cache_and_reopen(tmp_path, policy):
    @dataclasses.dataclass
    class Mixed:
        x: int
        text: str = blosc2.field(blosc2.vlstring(batch_rows=16))

    rng = np.random.default_rng(7)
    values = [rng.bytes(1024).hex() for _ in range(128)]
    local = blosc2.CTable(Mixed, list(enumerate(values)), create_summary_index=False)
    url = remote_table_url(tmp_path, local, f"vlstring-{policy.value}")
    options = {"cache_policy": policy, "max_cache_bytes": 48 << 10}
    if policy is blosc2.CachePolicy.DISK:
        options["cache_dir"] = tmp_path / "cache"

    with blosc2.RemoteCTable(url, **options) as remote:
        assert remote["text"][80:85] == values[80:85]
        requests = remote.traffic.requests
        assert remote["text"][80:85] == values[80:85]
        assert remote.traffic.requests == requests
        np.testing.assert_array_equal(remote["x"][80:85], np.arange(80, 85))
        assert remote.cache_bytes <= options["max_cache_bytes"]

    if policy is blosc2.CachePolicy.DISK:
        with blosc2.RemoteCTable(url, **options) as remote:
            requests = remote.traffic.requests
            assert remote["text"][80:85] == values[80:85]
            assert remote.traffic.requests == requests


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


def test_remote_ctable_reference_save_roundtrip(tmp_path):
    @dataclasses.dataclass
    class TextRow:
        x: int = blosc2.field(blosc2.int64(null_storage="mask"))
        text: str = blosc2.field(blosc2.utf8())

    local = blosc2.CTable(
        TextRow,
        [(1, "café"), (None, "東京"), (3, "three")],
        create_summary_index=False,
    )
    local.attrs["title"] = "remote table"
    url = remote_table_url(tmp_path, local, "reference-source")
    warm_path = tmp_path / "warm-reference.b2z"
    cold_path = tmp_path / "cold-reference.b2z"
    mutable_path = tmp_path / "mutable-reference.b2z"

    with blosc2.RemoteCTable(url) as remote:
        np.testing.assert_array_equal(remote["x"][:], [1, 0, 3])
        assert remote.attrs["title"] == "remote table"
        before = remote.traffic.requests
        assert remote.save(warm_path) == os.path.abspath(warm_path)
        assert remote.save(urlpath=cold_path, include_cache=False) == os.path.abspath(cold_path)
        remote.save(mutable_path, mutable=True)
        assert remote.traffic.requests == before

    with blosc2.open(warm_path) as warm:
        assert isinstance(warm, blosc2.RemoteCTable)
        assert warm.attrs["title"] == "remote table"
        warm.traffic.reset()
        np.testing.assert_array_equal(warm["x"][:], [1, 0, 3])
        assert warm.traffic.requests == 0
        assert warm["text"][:].tolist() == ["café", "東京", "three"]
        assert warm.traffic.requests > 0

    with blosc2.open(cold_path) as cold:
        assert isinstance(cold, blosc2.RemoteCTable)
        assert cold.cache_bytes == 0
        cold.traffic.reset()
        np.testing.assert_array_equal(cold["x"][:], [1, 0, 3])
        assert cold.traffic.requests > 0

    with blosc2.open(mutable_path) as mutable:
        assert isinstance(mutable, blosc2.RemoteCTable)
        assert mutable.is_cache_mutable
        mutable.refresh()
        assert mutable.attrs["title"] == "remote table"

    with blosc2.RemoteCTable(url) as remote, pytest.raises(FileExistsError):
        remote.save(warm_path)


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("mutable", [False, True])
def test_reference_max_concurrency(tmp_path, nested, mutable):
    local = blosc2.CTable(Row, [(1, [1, 2], "one")])
    source = tmp_path / "source.b2z"
    if nested:
        with blosc2.TreeStore(source, mode="w") as tree:
            tree["table"] = local
    else:
        local.to_b2z(source)
    url = f"memory://{tmp_path.name}-settings.b2z"
    fsspec.filesystem("memory").pipe(url, source.read_bytes())
    artifact = tmp_path / "reference.b2z"
    with blosc2.open(url) as remote:
        remote.save(artifact, mutable=mutable)
    options = {"dataset": "table"} if nested else {}
    with blosc2.open(artifact, max_concurrency=1, **options) as reopened:
        assert reopened.max_concurrency == 1
        np.testing.assert_array_equal(reopened["x"][:], [1])
    with pytest.raises(ValueError, match="max_concurrency must be a positive integer"):
        blosc2.open(artifact, max_concurrency=0, **options)


@pytest.mark.parametrize("mutation", ["metadata", "kind", "schema", "source_kind"])
def test_remote_ctable_reference_rejects_invalid_manifest(tmp_path, mutation):
    local = blosc2.CTable(Row, [(1, [1, 2], "one")], create_summary_index=False)
    url = remote_table_url(tmp_path, local, f"invalid-{mutation}")
    artifact = tmp_path / f"reference-{mutation}.b2z"
    with blosc2.RemoteCTable(url) as remote:
        remote.save(artifact)

    unpacked = tmp_path / f"unpacked-{mutation}"
    with zipfile.ZipFile(artifact) as archive:
        archive.extractall(unpacked)
    embed = blosc2.blosc2_ext.open(str(unpacked / "embed.b2e"), "a", 0)
    manifest = dict(embed.vlmeta["b2remote_manifest"])
    root = manifest["source"]["dataset"]
    nodes = dict(manifest["nodes"])
    metadata = dict(nodes[root][1])
    if mutation == "metadata":
        metadata = None
    elif mutation == "kind":
        metadata["kind"] = "group"
    elif mutation == "schema":
        metadata["schema"] = None
    else:
        manifest["source"] = {**manifest["source"], "kind": "hdf5"}
    nodes[root] = ("ctable", metadata)
    manifest["nodes"] = nodes
    embed.vlmeta["b2remote_manifest"] = manifest
    del embed

    broken = tmp_path / f"broken-{mutation}.b2z"
    with zipfile.ZipFile(broken, "w", zipfile.ZIP_STORED) as archive:
        for member in unpacked.rglob("*"):
            if member.is_file():
                archive.write(member, member.relative_to(unpacked))
    with pytest.raises(ValueError, match="Invalid RemoteStore CTable node"):
        blosc2.open(broken)


def test_nested_remote_ctable_reference_save(tmp_path):
    source = tmp_path / "tree.b2z"
    table = blosc2.CTable(Row, [(1, [1, 2], "one"), (2, [3, 4], "two")])
    with blosc2.TreeStore(source, mode="w", threshold=0) as root:
        root["group/table"] = table
        root["group/sibling"] = blosc2.arange(10)
    url = f"memory://{tmp_path.name}-reference-tree.b2z"
    fsspec.filesystem("memory").pipe(url, source.read_bytes())
    destination = tmp_path / "nested-reference.b2z"
    group_destination = tmp_path / "group-reference.b2z"

    with blosc2.RemoteStore(url) as store, store["group/table"] as remote:
        np.testing.assert_array_equal(remote["x"][:], [1, 2])
        with store["group/sibling"] as sibling:
            np.testing.assert_array_equal(sibling[:], np.arange(10))
        remote.save(destination)
        with store["group"] as group:
            group.save(group_destination)

    with zipfile.ZipFile(destination) as archive:
        names = archive.namelist()
        assert any(name.startswith("group/table/") for name in names)
        assert not any(name.startswith("group/sibling") for name in names)
    with zipfile.ZipFile(group_destination) as archive:
        assert any(name.startswith("group/sibling") for name in archive.namelist())

    with blosc2.open(destination) as reopened:
        assert isinstance(reopened, blosc2.RemoteCTable)
        assert reopened.source["dataset"] == "group/table"
        reopened.traffic.reset()
        np.testing.assert_array_equal(reopened["x"][:], [1, 2])
        assert reopened.traffic.requests == 0

    with blosc2.open(group_destination) as group:
        assert isinstance(group, blosc2.RemoteStore)
        with group["table"] as reopened:
            assert isinstance(reopened, blosc2.RemoteCTable)
            np.testing.assert_array_equal(reopened["x"][:], [1, 2])
        with group["sibling"] as sibling:
            assert isinstance(sibling, blosc2.RemoteArray)


def test_remote_ctable_batch_column_is_lazy(tmp_path):
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
        assert table["text"][:] == ["a", "bb"]


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

        materialized = remote.materialize()
        assert type(materialized) is blosc2.CTable
        np.testing.assert_array_equal(materialized["x"][:], [0, 2, 3, 5])

        destination = tmp_path / "materialized.b2z"
        persisted = remote.materialize(urlpath=destination)
        assert type(persisted) is blosc2.CTable
        np.testing.assert_array_equal(persisted["x"][:], [0, 2, 3, 5])

    with blosc2.RemoteCTable(url, cache_dir=cache) as reopened:
        np.testing.assert_array_equal(reopened["x"][:], [0, 2, 3, 5])

    np.testing.assert_array_equal(materialized["x"][:], [0, 2, 3, 5])
    np.testing.assert_array_equal(persisted["x"][:], [0, 2, 3, 5])
    materialized.close()
    persisted.close()


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
        with pytest.raises(ValueError, match="root RemoteStore"):
            table.refresh()
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


def test_parallel_metadata_benchmark(tmp_path, monkeypatch):
    import runpy
    import time
    from pathlib import Path

    benchmark = runpy.run_path(str(Path(__file__).resolve().parents[2] / "bench/remote_ctable_metadata.py"))

    @dataclasses.dataclass
    class TextRow:
        x: int
        y: float
        text: str = blosc2.field(blosc2.utf8(null_storage="mask"))

    local = blosc2.CTable(TextRow, [(1, 2.0, "café"), (2, 3.0, None)], create_summary_index=False)
    url = remote_table_url(tmp_path, local)
    filesystem_type = type(fsspec.filesystem("memory"))
    original = filesystem_type.cat_file

    def delayed(self, *args, **kwargs):
        time.sleep(0.01)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(filesystem_type, "cat_file", delayed)
    serial, expected = benchmark["run"](url, 1)
    buffered, buffered_values = benchmark["run"](url, 1)
    parallel, actual = benchmark["run"](url, 4)
    assert actual == expected
    assert buffered_values == expected
    assert serial["peak_metadata_reads"] == 1
    assert parallel["peak_metadata_reads"] > 1
    assert parallel["metadata"]["bytes"] <= serial["metadata"]["bytes"]
    assert buffered["metadata"]["bytes"] == parallel["metadata"]["bytes"]
    assert buffered["metadata"]["requests"] == parallel["metadata"]["requests"]
    with blosc2.open(url) as table:
        archive = table._remote_storage()._owner.archive
        original_read = archive._read_archive

        def fail(*args):
            raise OSError("injected range failure")

        with monkeypatch.context() as patch:
            patch.setattr(archive, "_read_archive", fail)
            with pytest.raises(OSError, match="injected range failure"):
                benchmark["metadata"](table)
            assert archive._read_archive is fail
        assert archive._read_archive == original_read
        assert benchmark["metadata"](table) == expected[0]

    # Large, uncompressed columns keep payloads out of the metadata prefixes.
    local = blosc2.CTable(
        TextRow,
        [(i, i / 7, None if i % 3 == 0 else f"東京 café {i}") for i in range(20000)],
        cparams={"clevel": 0},
        create_summary_index=False,
    )
    url = remote_table_url(tmp_path, local, name="row-payloads")
    serial, expected = benchmark["run"](url, 1, row_workers=1)
    buffered, buffered_values = benchmark["run"](url, 1)
    parallel, actual = benchmark["run"](url, 4)
    assert actual == buffered_values == expected
    assert serial["peak_row_reads"] == 1
    assert parallel["peak_row_reads"] > 1
    for key in ("requests", "bytes"):
        assert parallel["cold_rows"][key] == buffered["cold_rows"][key] == serial["cold_rows"][key]
    assert parallel["cache_bytes"] == serial["cache_bytes"]
    with blosc2.open(url) as table:
        benchmark["metadata"](table)
        archive = table._remote_storage()._owner.archive
        original_read = archive.read_transport
        with monkeypatch.context() as patch:
            patch.setattr(archive, "read_transport", fail)
            with pytest.raises(OSError, match="injected range failure"):
                list(table[:5])
            assert archive.read_transport is fail
        assert archive.read_transport == original_read
        assert repr(list(table[:5])) == expected[1]


@pytest.mark.parametrize(
    "policy", [blosc2.CachePolicy.NONE, blosc2.CachePolicy.MEMORY, blosc2.CachePolicy.DISK]
)
@pytest.mark.parametrize("budget", [128, 1 << 20])
def test_parallel_rows_cache_policies(tmp_path, policy, budget):
    @dataclasses.dataclass
    class Mixed:
        x: int = blosc2.field(blosc2.int64(null_storage="mask"))
        y: float = blosc2.field(blosc2.float64())
        text: str = blosc2.field(blosc2.utf8(null_storage="mask"))

    local = blosc2.CTable(
        Mixed,
        [
            (None if i % 7 == 0 else i, i / 3, None if i % 5 == 0 else f"🌦 café 東京 {i}")
            for i in range(20000)
        ],
        cparams={"clevel": 0},
        create_summary_index=False,
    )
    local.delete([2, 9])
    url = remote_table_url(tmp_path, local)
    options = {"cache_policy": policy, "row_buffer_bytes": budget, "metadata_buffer_bytes": budget}
    if policy == blosc2.CachePolicy.DISK:
        options["cache_dir"] = tmp_path / "cache"
    if policy != blosc2.CachePolicy.NONE:
        options["max_cache_bytes"] = 1024
    with blosc2.RemoteCTable(url, **options) as table:
        for selection in (slice(0, 5), slice(3, 12, 2), slice(8, 1, -2), [15, 0, 15, 3]):
            assert repr(list(table[selection])) == repr(list(local[selection]))
        assert repr(table[3]) == repr(local[3])
        assert table[:4].to_string() == local[:4].to_string()
        pd = pytest.importorskip("pandas")
        pd.testing.assert_frame_equal(table[:12].to_pandas(), local[:12].to_pandas())
        pytest.importorskip("pyarrow")
        assert table[:12].to_arrow().equals(local[:12].to_arrow())
        assert table.cache_bytes <= 1024
        if policy == blosc2.CachePolicy.NONE:
            assert table.cache_bytes == 0


def test_parallel_table_settings(tmp_path):
    local = blosc2.CTable(Row, [(1, [1, 2], "one")], create_summary_index=False)
    url = remote_table_url(tmp_path, local)
    with blosc2.open(url, max_concurrency=2) as table:
        assert table.max_concurrency == 2
        assert table.metadata_buffer_bytes == 8 << 20
        assert table.row_buffer_bytes == 64 << 20
        table.row_buffer_bytes = 256 << 20
        assert table[:1]._remote_read_storage().row_buffer_bytes == 256 << 20
        for name in ("max_concurrency", "metadata_buffer_bytes", "row_buffer_bytes"):
            for value, error in ((0, ValueError), (-1, ValueError), (True, TypeError), (1.5, TypeError)):
                with pytest.raises(error):
                    setattr(table, name, value)
                with pytest.raises(error):
                    blosc2.RemoteCTable(url, **{name: value})
        assert repr(list(table[:1])) == repr(list(local[:1]))
    for name in ("metadata_buffer_bytes", "row_buffer_bytes"):
        with pytest.raises((TypeError, NotImplementedError), match=name):
            blosc2.open(url, **{name: 1024})
    with blosc2.RemoteStore(url, _allow_array_root=True) as store:
        first, second = store[""], store[""]
        first.max_concurrency = 1
        assert second.max_concurrency == 8
        first.close()
        second.close()


def test_parallel_scheduler_budget_and_cleanup():
    import threading
    import time

    from blosc2.ctable_remote_read import run_reads

    lock = threading.Lock()
    active = peak = 0
    closed = []

    def fetch(size):
        nonlocal active, peak
        with lock:
            active += size
            peak = max(peak, active)
        try:
            time.sleep(0.01)
            return bytes(size)
        finally:
            with lock:
                active -= size

    def reader(i, size):
        try:
            data = yield fetch, (size,), size
            return len(data)
        finally:
            closed.append(i)

    values, reserved = run_reads(((i, reader(i, 4)) for i in range(12)), 8, 10)
    assert values == dict.fromkeys(range(12), 4)
    assert 4 < peak <= reserved <= 10
    assert sorted(closed) == list(range(12))
    values, reserved = run_reads(((i, reader(i, 20)) for i in range(2)), 8, 10)
    assert reserved == 20  # One oversized unit alone, never two together.
    assert active == 0

    closed.clear()

    def fail():
        raise OSError("injected worker failure")

    def broken():
        try:
            yield fail, (), 4
        finally:
            closed.append("broken")

    with pytest.raises(OSError, match="injected worker failure"):
        run_reads([(0, broken()), (1, reader(1, 4))], 2, 10)
    assert active == 0
    assert set(closed) == {"broken", 1}


@pytest.mark.parametrize(
    "policy", [blosc2.CachePolicy.NONE, blosc2.CachePolicy.MEMORY, blosc2.CachePolicy.DISK]
)
def test_parallel_rows_blocks_and_reopen(tmp_path, monkeypatch, policy):
    import threading

    @dataclasses.dataclass
    class Numbers:
        x: float
        y: float

    rng = np.random.default_rng(42)
    local = blosc2.CTable(
        Numbers,
        {"x": rng.normal(size=400000), "y": rng.normal(size=400000)},
        create_summary_index=False,
    )
    url = remote_table_url(tmp_path, local)
    writes = []
    original = blosc2.Proxy._write_blocks

    def write(self, *args):
        writes.append(threading.current_thread())
        return original(self, *args)

    monkeypatch.setattr(blosc2.Proxy, "_write_blocks", write)
    options = {"cache_policy": policy}
    if policy == blosc2.CachePolicy.DISK:
        options["cache_dir"] = tmp_path / "cache"
    with blosc2.RemoteCTable(url, **options) as table:
        before = table.traffic.nbytes
        assert list(table[:10]) == list(local[:10])
        assert table.traffic.nbytes - before < 2 << 20
        assert writes
        assert all(t is threading.current_thread() for t in writes)
    if policy == blosc2.CachePolicy.DISK:
        with blosc2.RemoteCTable(url, **options) as table:
            before = table.traffic.requests
            assert list(table[:10]) == list(local[:10])
            assert table.traffic.requests == before
            assert list(table[399990:]) == list(local[399990:])


def test_parallel_timestamp_and_projection(tmp_path):
    @dataclasses.dataclass
    class Timed:
        time: object = blosc2.field(blosc2.timestamp(unit="ns", null_storage="mask"))
        vec: object = blosc2.field(blosc2.ndarray((2,), dtype=blosc2.float32(), null_storage="mask"))
        text: str = blosc2.field(blosc2.utf8())

    local = blosc2.CTable(
        Timed,
        [(np.datetime64("2025-01-01", "ns"), [1, 2], "東京"), (None, None, "")],
        create_summary_index=False,
    )
    url = remote_table_url(tmp_path, local)
    with blosc2.open(url) as table:
        assert repr(list(table)) == repr(list(local))
        assert table.to_string() == local.to_string()
        pd = pytest.importorskip("pandas")
        pd.testing.assert_frame_equal(table.to_pandas(), local.to_pandas())
        pytest.importorskip("pyarrow")
        assert table.to_arrow().equals(local.to_arrow())
    with blosc2.open(url) as table:
        view = table[["text"]]
        assert list(view) == list(local[["text"]])
        assert "_cols/time" not in table._remote_storage()._owner.sources
        assert "_cols/vec" not in table._remote_storage()._owner.sources
