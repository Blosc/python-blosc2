from dataclasses import dataclass
from uuid import uuid4

import numpy as np
import pytest

import blosc2

fsspec = pytest.importorskip("fsspec")


@dataclass
class Row:
    local: int = blosc2.field(blosc2.int32())
    remote: float = blosc2.field(blosc2.float32())


def remote_array(values, *, chunks=None, cache_policy=blosc2.CachePolicy.NONE):
    values = np.asarray(values)
    array = blosc2.asarray(values, chunks=chunks)
    name = f"ctable-sources-{uuid4().hex}.b2nd"
    fsspec.filesystem("memory").pipe_file(name, array.to_cframe())
    return blosc2.RemoteArray(f"memory://{name}", cache_policy=cache_policy)


def test_mixed_sources_read_query_and_borrowed_lifetime():
    local = blosc2.asarray(np.arange(8, dtype=np.int32), chunks=(3,))
    remote = remote_array(np.arange(8, dtype=np.float32) / 2, chunks=(5,))

    table = blosc2.CTable(Row, sources={"local": local, "remote": remote})

    np.testing.assert_array_equal(table.local[1:7:2], np.array([1, 3, 5], dtype=np.int32))
    np.testing.assert_array_equal(table[table.local >= 4].remote[:], np.array([2, 2.5, 3, 3.5]))
    np.testing.assert_array_equal((table.local + table.remote)[:], np.arange(8) * 1.5)
    with pytest.raises(ValueError, match="read-only"):
        table.append((8, 4.0))

    table.close()
    np.testing.assert_array_equal(remote[:2], np.array([0, 0.5], dtype=np.float32))


def test_source_validation_and_empty_sources():
    empty = blosc2.asarray(np.array([], dtype=np.int32))
    empty_remote = remote_array(np.array([], dtype=np.float32))
    table = blosc2.CTable(Row, sources={"local": empty, "remote": empty_remote})
    assert len(table) == 0

    with pytest.raises(ValueError, match="missing: remote"):
        blosc2.CTable(Row, sources={"local": empty})
    with pytest.raises(TypeError, match="dtype"):
        blosc2.CTable(
            Row,
            sources={
                "local": blosc2.asarray(np.array([], dtype=np.int64)),
                "remote": empty_remote,
            },
        )
    with pytest.raises(ValueError, match="different row counts"):
        blosc2.CTable(
            Row,
            sources={
                "local": blosc2.asarray(np.arange(2, dtype=np.int32)),
                "remote": empty_remote,
            },
        )


def test_constrained_source_requires_validation_opt_out():
    @dataclass
    class Constrained:
        value: int = blosc2.field(blosc2.int32(ge=0))

    source = blosc2.asarray(np.arange(3, dtype=np.int32))
    with pytest.raises(ValueError, match="validate=False"):
        blosc2.CTable(Constrained, sources={"value": source})
    table = blosc2.CTable(Constrained, sources={"value": source}, validate=False)
    np.testing.assert_array_equal(table.value[:], np.arange(3, dtype=np.int32))


def test_save_materializes_by_default_and_can_preserve_sources(tmp_path):
    values = np.arange(6, dtype=np.float32)
    table = blosc2.CTable(
        Row,
        sources={
            "local": blosc2.asarray(np.arange(6, dtype=np.int32)),
            "remote": remote_array(values),
        },
    )

    materialized_path = tmp_path / "materialized.b2z"
    table.save(materialized_path)
    materialized = blosc2.open(materialized_path)
    assert isinstance(materialized._cols["remote"], blosc2.NDArray)
    assert not isinstance(materialized._cols["remote"], blosc2.RemoteArray)
    np.testing.assert_array_equal(materialized.remote[:], values)

    referenced_path = tmp_path / "referenced.b2z"
    table.save(referenced_path, preserve_sources=True)
    referenced = blosc2.open(referenced_path)
    assert isinstance(referenced._cols["remote"], blosc2.RemoteArray)
    assert referenced._read_only
    np.testing.assert_array_equal(referenced.remote[:], values)
    with pytest.raises(ValueError, match="read-only"):
        referenced.append((6, 6.0))


def test_treestore_and_cframe_preserve_sources(tmp_path):
    values = np.arange(4, dtype=np.float32)
    table = blosc2.CTable(
        Row,
        sources={
            "local": blosc2.asarray(np.arange(4, dtype=np.int32)),
            "remote": remote_array(values),
        },
    )

    path = tmp_path / "table-tree.b2z"
    with blosc2.TreeStore(path, mode="w") as tree:
        tree["table"] = table
    with blosc2.TreeStore(path, mode="r") as tree:
        reopened = tree["table"]
        assert isinstance(reopened._cols["remote"], blosc2.RemoteArray)
        np.testing.assert_array_equal(reopened.remote[:], values)

    restored = blosc2.ctable_from_cframe(table.to_cframe(preserve_sources=True))
    assert isinstance(restored._cols["remote"], blosc2.RemoteArray)
    np.testing.assert_array_equal(restored.remote[:], values)


@pytest.mark.parametrize("policy", list(blosc2.CachePolicy))
def test_remote_ctable_owns_external_column_cache_policy(tmp_path, policy):
    values = np.arange(20, dtype=np.float32)
    source = remote_array(values, chunks=(5,), cache_policy=blosc2.CachePolicy.MEMORY)
    table = blosc2.CTable(
        Row,
        sources={
            "local": blosc2.asarray(np.arange(20, dtype=np.int32)),
            "remote": source,
        },
    )
    archive = tmp_path / "referenced.b2z"
    table.save(archive, preserve_sources=True)
    archive_name = f"ctable-archive-{uuid4().hex}.b2z"
    fsspec.filesystem("memory").pipe_file(archive_name, archive.read_bytes())

    kwargs = {"cache_policy": policy}
    if policy is not blosc2.CachePolicy.NONE:
        kwargs["max_cache_bytes"] = 1 << 20
    if policy is blosc2.CachePolicy.DISK:
        kwargs["cache_dir"] = tmp_path / "cache"
    remote = blosc2.RemoteCTable(f"memory://{archive_name}", **kwargs)
    external = remote._cols["remote"]

    assert external.cache_policy is policy
    assert source.cache_policy is blosc2.CachePolicy.MEMORY
    np.testing.assert_array_equal(remote.remote[3:13:2], values[3:13:2])
    np.testing.assert_array_equal(remote[remote.local >= 17].remote[:], values[17:])
    if policy is not blosc2.CachePolicy.NONE:
        assert remote.cache_bytes <= remote.max_cache_bytes
    if policy is blosc2.CachePolicy.MEMORY:
        artifact = tmp_path / "remote-artifact.b2z"
        remote.save(artifact)
        restored = blosc2.open(artifact)
        assert restored.cache_policy is policy
        np.testing.assert_array_equal(restored.remote[:], values)
