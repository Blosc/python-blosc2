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
