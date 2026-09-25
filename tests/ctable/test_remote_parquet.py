"""Parity and row-group isolation for the Parquet remote-table prototype."""

import http.server
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from decimal import Decimal
from email.utils import formatdate

import numpy as np
import pytest

import blosc2
from blosc2 import remote_parquet
from blosc2.core import cache_directory_name
from blosc2.ctable import CTable
from blosc2.schema_compiler import schema_to_dict

fsspec = pytest.importorskip("fsspec")
pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")


def test_windows_cache_directory_uses_source_basename():
    assert cache_directory_name(r"C:\data\source.parquet", b"cache").startswith("source.parquet--")


def test_source_url_preserves_non_sensitive_query():
    assert (
        remote_parquet._source_url("https://example.com/source.parquet?version=2#section")
        == "https://example.com/source.parquet?version=2"
    )
    with pytest.raises(ValueError, match="credential-like"):
        remote_parquet._disk_cache_path(
            "https://example.com/source.parquet?token=secret",
            None,
            (),
            None,
            {"size": "1", "etag": "revision"},
        )


@pytest.mark.parametrize(
    "source",
    [
        pa.table({"id": [1, 2, None, 4], "text": ["a", None, "é", ""]}),
        pa.table(
            {
                "unsigned": pa.array([0, 1, None, 255], type=pa.uint8()),
                "float": pa.array([1.0, float("nan"), None, 4.5], type=pa.float32()),
                "flag": pa.array([True, None, False, True], type=pa.bool_()),
            }
        ),
        pa.table(
            {
                "large_text": pa.array(["é", "", None, "xyz"], type=pa.large_string()),
                "binary": pa.array([b"\x00x", b"", None, b"hi"], type=pa.large_binary()),
            }
        ),
        pa.table({"time": pa.array([0, None, 1234, 5678], type=pa.timestamp("ms", tz="UTC"))}),
        pa.table({"items": pa.array([[1, None], [], None, [4]], type=pa.list_(pa.int64()))}),
        pa.table(
            {
                "record": pa.array(
                    [{"id": 1}, None, {"id": 3}, {"id": 4}],
                    type=pa.struct([("id", pa.int32())]),
                )
            }
        ),
        pa.table({"category": pa.array(["a", "b", "a", "c"]).dictionary_encode()}),
        pa.table({"cell": pa.array([[1, 2], [2, 3], [3, 4], [5, 6]], type=pa.list_(pa.int32(), 2))}),
        pa.table(
            {
                "items": pa.array(
                    [[{"id": 1, "name": "a"}], None, [], [{"id": None, "name": "b"}]],
                    type=pa.list_(pa.struct([("id", pa.int32()), ("name", pa.string())])),
                )
            }
        ),
        pa.table(
            {
                "trip": pa.array(
                    [
                        {"begin": {"lon": 1.5, "when": 1000}},
                        None,
                        {"begin": {"lon": None, "when": None}},
                        {"begin": {"lon": 4.5, "when": 4000}},
                    ],
                    type=pa.struct(
                        [("begin", pa.struct([("lon", pa.float64()), ("when", pa.timestamp("ms"))]))]
                    ),
                )
            }
        ),
        pa.table({"id": pa.array([], type=pa.int32()), "text": pa.array([], type=pa.string())}),
        pa.Table.from_arrays(
            [pa.array([1, 2, 3, 4]), pa.array([5, 6, 7, 8]), pa.array([9, 10, 11, 12])],
            names=["", "root", "foo.bar"],
        ),
        pa.Table.from_arrays(
            [pa.array([[{"id": 1}], [], None, [{"id": 4}]], type=pa.list_(pa.struct([("id", pa.int32())])))],
            names=[""],
        ),
    ],
    ids=[
        "scalar",
        "numeric-bool",
        "large-string-binary",
        "timestamp",
        "list",
        "struct",
        "dictionary",
        "fixed-cell",
        "list-struct",
        "nested-struct-timestamp",
        "empty",
        "escaped-root",
        "flattened-root",
    ],
)
def test_remote_parquet_matches_importer(source, tmp_path):
    path = tmp_path / "data.parquet"
    pq.write_table(source, path, row_group_size=2)
    fs = fsspec.filesystem("memory")
    fs.pipe("/remote-parity.parquet", path.read_bytes())
    with blosc2.CTable.from_parquet(path) as eager, blosc2.open("memory:///remote-parity.parquet") as remote:
        assert remote.col_names == eager.col_names
        assert len(remote) == len(eager)
        assert schema_to_dict(remote._schema) == schema_to_dict(eager._schema)
        if "float" in source.column_names:
            np.testing.assert_equal(remote.to_arrow().to_pylist(), eager.to_arrow().to_pylist())
        else:
            assert remote.to_arrow().equals(eager.to_arrow())
            if "cell" not in source.column_names:
                assert list(remote) == list(eager)


def test_narrow_read_uses_only_one_later_group(tmp_path):
    path = tmp_path / "wide.parquet"
    pq.write_table(pa.table({"one": list(range(12)), "two": list(range(12))}), path, row_group_size=3)
    fs = fsspec.filesystem("memory")
    fs.pipe("/remote-wide.parquet", path.read_bytes())
    with blosc2.open("memory:///remote-wide.parquet") as remote:
        before = remote.traffic.requests
        assert remote["one"][10] == 10
        assert remote.traffic.requests == before + 1
        assert remote["one"][10] == 10
        assert remote.traffic.requests == before + 1


def test_info_without_loading_parquet_groups(tmp_path):
    path = tmp_path / "info.parquet"
    pq.write_table(pa.table({"id": [1, 2, 3], "text": ["a", "bb", "ccc"]}), path)
    with blosc2.open(path, cache_dir=tmp_path / "cache") as remote:
        before = remote.traffic.requests
        info = dict(remote.info_items)
        assert info["nbytes"] == "27 (27 B)"  # Three int64 values plus the virtual validity mask.
        assert info["cbytes"] == info["cratio"] == "n/a"
        assert "id" in info["columns"]
        assert remote.traffic.requests == before


def test_unnamed_root_without_flattening(tmp_path):
    path = tmp_path / "root.parquet"
    source = pa.Table.from_arrays(
        [
            pa.array(
                [[{"root": 1}], [], None, [{"root": 4}]],
                type=pa.list_(pa.struct([("root", pa.int32())])),
            )
        ],
        names=[""],
    )
    pq.write_table(source, path, row_group_size=2)
    with blosc2.CTable.from_parquet(path, separate_nested_cols=False) as eager:
        with blosc2.open(path, separate_nested_cols=False) as remote:
            assert len(remote) == len(eager) == 4
            assert remote.to_arrow().equals(eager.to_arrow())


def test_empty_flattened_root(tmp_path):
    path = tmp_path / "empty-root.parquet"
    source = pa.Table.from_arrays(
        [pa.array([[], None, []], type=pa.list_(pa.struct([("id", pa.int32())])))], names=[""]
    )
    pq.write_table(source, path, row_group_size=2)
    with blosc2.CTable.from_parquet(path) as eager, blosc2.open(path) as remote:
        assert len(remote) == len(eager) == 0
        assert schema_to_dict(remote._schema) == schema_to_dict(eager._schema)
        assert remote.to_arrow().equals(eager.to_arrow())


def test_flattened_root_counts_from_smallest_leaf(tmp_path):
    path = tmp_path / "wide-root.parquet"
    source = pa.Table.from_arrays(
        [
            pa.array(
                [
                    [{"id": 1, "blob": os.urandom(128_000)}],
                    [],
                    [{"id": 2, "blob": os.urandom(128_000)}],
                    [{"id": 3, "blob": os.urandom(128_000)}],
                ]
            )
        ],
        names=[""],
    )
    pq.write_table(source, path, row_group_size=2, compression="NONE")
    with blosc2.open(path) as remote:
        assert len(remote) == 3
        assert remote.traffic.requests == 3  # Footer and one leaf per group.
        assert remote.traffic.nbytes < path.stat().st_size // 2
        before = remote.traffic.nbytes
        assert remote["id"][-1] == 3
        assert remote.traffic.nbytes - before < path.stat().st_size // 2
        assert remote["blob"][-1] == source.column(0)[-1].as_py()[-1]["blob"]


@pytest.mark.parametrize("limit", [0, 2, 5])
def test_flattened_root_max_rows(tmp_path, limit):
    path = tmp_path / "root.parquet"
    source = pa.Table.from_arrays(
        [
            pa.array(
                [[{"id": 1}, {"id": 2}], [], None, [{"id": 4}], [{"id": 5}, {"id": 6}]],
                type=pa.list_(pa.struct([("id", pa.int32())])),
            )
        ],
        names=[""],
    )
    pq.write_table(source, path, row_group_size=2)
    with blosc2.CTable.from_parquet(path, max_rows=limit) as eager:
        with blosc2.open(path, max_rows=limit) as remote:
            assert len(remote) == len(eager)
            assert remote.to_arrow().equals(eager.to_arrow())


@pytest.mark.parametrize("serializer", ["msgpack", "arrow"])
def test_nested_list_serializers(tmp_path, serializer):
    path = tmp_path / "nested.parquet"
    source = pa.table(
        {"items": pa.array([[[1, None], []], None, [], [[4]]], type=pa.list_(pa.list_(pa.int32())))}
    )
    pq.write_table(source, path, row_group_size=2)
    options = {"list_serializer": serializer, "blosc2_batch_size": 2, "blosc2_items_per_block": 1}
    with blosc2.CTable.from_parquet(path, **options) as eager, blosc2.open(path, **options) as remote:
        assert schema_to_dict(remote._schema) == schema_to_dict(eager._schema)
        assert remote.to_arrow().equals(eager.to_arrow())


def test_null_policy_frozen_at_open(tmp_path):
    path = tmp_path / "nullable.parquet"
    pq.write_table(pa.table({"id": pa.array([1, None, 3], type=pa.int8())}), path, row_group_size=1)
    with blosc2.null_policy(blosc2.NullPolicy(signed_int_strategy="max")):
        eager = blosc2.CTable.from_parquet(path)
        remote = blosc2.open(path)
    try:
        assert schema_to_dict(remote._schema) == schema_to_dict(eager._schema)
        assert remote.to_arrow().equals(eager.to_arrow())
    finally:
        eager.close()
        remote.close()


def test_selection_and_materialization(tmp_path):
    path = tmp_path / "selected.parquet"
    target = tmp_path / "selected.b2z"
    pq.write_table(pa.table({"a": [0, 1, 2, 3], "b": [10, 11, 12, 13]}), path, row_group_size=2)
    with blosc2.open(path, columns=["b", "a"], max_rows=3) as remote:
        assert remote.col_names == ["b", "a"]
        assert [row.a for row in remote] == [0, 1, 2]
        assert remote.where("a > 0").to_arrow().column("a").to_pylist() == [1, 2]
        with remote.copy(urlpath=target) as materialized:
            assert materialized.to_arrow().equals(remote.to_arrow())
    with blosc2.open(target) as reopened:
        assert reopened.to_arrow().column("b").to_pylist() == [10, 11, 12]


def test_direct_constructor_and_settings(tmp_path):
    path = tmp_path / "direct.parquet"
    pq.write_table(pa.table({"a": [1, 2]}), path)
    with blosc2.RemoteCTable(path, max_concurrency=2, row_buffer_bytes=1024) as remote:
        assert remote["a"][:].tolist() == [1, 2]
        assert remote.max_concurrency == 2
        assert remote.row_buffer_bytes == 1024


def test_explicit_extensionless_source_and_projection(tmp_path):
    path = tmp_path / "source"
    pq.write_table(pa.table({"a": [1, 2], "b": [3, 4]}), path)
    with blosc2.open(path, source_format="parquet", columns=["b"]) as remote:
        assert remote.col_names == ["b"]
        assert remote.to_arrow().column("b").to_pylist() == [3, 4]
    with pytest.raises(KeyError):
        blosc2.open(path, source_format="parquet", columns=["missing"])


def test_unsupported_type_rejected_at_open(tmp_path):
    path = tmp_path / "decimal.parquet"
    pq.write_table(pa.table({"amount": pa.array([Decimal("1.00")], type=pa.decimal128(6, 2))}), path)
    with pytest.raises(TypeError, match="No blosc2 spec"):
        blosc2.open(path)


def test_invalid_source_options_fail_clearly(tmp_path):
    path = tmp_path / "data.parquet"
    pq.write_table(pa.table({"a": [1]}), path)
    with pytest.raises(ValueError, match="mode='r'"):
        blosc2.open(path, mode="a")
    with pytest.raises(TypeError, match="require lazy"):
        blosc2.open(path, lazy=False, cache_dir=tmp_path / "cache")
    with pytest.raises(ValueError, match="requires cache_dir"):
        blosc2.open(path, shared_cache=True)
    with pytest.raises(TypeError, match="shared_cache requires lazy"):
        blosc2.open(path, lazy=False, shared_cache=True, cache_dir=tmp_path / "cache")
    with pytest.raises(ValueError, match="memory_map"):
        blosc2.open(path, parquet_options={"memory_map": True})
    with pytest.raises(ValueError, match="conflicts"):
        blosc2.open(path, source_format="blosc2")


def test_portable_reference_reopens_and_detects_source_change(tmp_path):
    path = tmp_path / "source.parquet"
    carrier = tmp_path / "source.b2nd"
    pq.write_table(pa.table({"id": pa.array([1, None, 3], type=pa.int8())}), path)
    with blosc2.null_policy(blosc2.NullPolicy(signed_int_strategy="max")):
        with blosc2.open(path) as remote:
            remote.save(carrier)
    with blosc2.open(carrier) as reopened:
        assert reopened.to_arrow().column("id").to_pylist() == [1, None, 3]
    with blosc2.RemoteCTable.open_reference(carrier) as reopened:
        assert reopened.to_arrow().column("id").to_pylist() == [1, None, 3]
    output = subprocess.check_output(
        [
            sys.executable,
            "-c",
            "import blosc2,sys; print(blosc2.open(sys.argv[1]).to_arrow().column('id').to_pylist())",
            str(carrier),
        ],
        text=True,
    )
    assert output.strip() == "[1, None, 3]"
    pq.write_table(pa.table({"id": pa.array([4, 5], type=pa.int8())}), path)
    with pytest.raises(RuntimeError, match="source has changed"):
        blosc2.open(carrier)


def test_reference_retains_cached_group_and_runtime_options(tmp_path):
    path = tmp_path / "source.parquet"
    carrier = tmp_path / "source.b2nd"
    pq.write_table(pa.table({"id": list(range(6))}), path, row_group_size=2)
    with blosc2.open(path, storage_options={"auto_mkdir": True}) as remote:
        assert remote["id"][5] == 5
        remote.save(carrier, include_cache=True)
    with blosc2.RemoteCTable.open_reference(carrier, storage_options={"auto_mkdir": True}) as reopened:
        before = reopened.traffic.requests
        assert reopened["id"][5] == 5
        assert reopened.traffic.requests == before


def test_reference_retains_flattened_row_map(tmp_path):
    path = tmp_path / "root.parquet"
    carrier = tmp_path / "root.b2nd"
    source = pa.Table.from_arrays(
        [pa.array([[{"id": i}] for i in range(6)], type=pa.list_(pa.struct([("id", pa.int32())])))],
        names=[""],
    )
    pq.write_table(source, path, row_group_size=2)
    with blosc2.open(path, cache_dir=tmp_path / "cache") as remote:
        assert remote["id"][5] == 5
        remote.save(carrier, include_cache=True)
    with blosc2.open(carrier) as reopened:
        assert len(reopened) == 6
        before = reopened.traffic.requests
        assert reopened["id"][5] == 5
        assert reopened.traffic.requests == before


def test_reference_rejects_nonportable_reader_option(tmp_path):
    path = tmp_path / "source.parquet"
    pq.write_table(pa.table({"id": [1]}), path)
    with blosc2.open(path) as remote:
        remote._storage._owner.reopen_kwargs["parquet_options"] = {"decryption_properties": object()}
        with pytest.raises(TypeError, match="Nonportable"):
            remote.save(tmp_path / "source.b2nd")


def test_reference_reader_options_are_frozen(tmp_path):
    path = tmp_path / "source.parquet"
    carrier = tmp_path / "source.b2nd"
    pq.write_table(pa.table({"name": ["a", "b", "a"]}), path)
    with blosc2.open(path, parquet_options={"read_dictionary": ["name"]}) as remote:
        assert remote["name"][0] == "a"
        remote.save(carrier, include_cache=True)
    with blosc2.RemoteCTable.open_reference(carrier) as reopened:
        assert reopened.to_arrow().column("name").to_pylist() == ["a", "b", "a"]
    with pytest.raises(ValueError, match="differs"):
        blosc2.RemoteCTable.open_reference(carrier, parquet_options={"read_dictionary": []})
    with pytest.raises(ValueError, match="differ"):
        blosc2.RemoteCTable.open_reference(
            carrier, parquet_options={"read_dictionary": ["name"], "coerce_int96_timestamp_unit": "ms"}
        )


def test_closed_handle_rejects_cached_column(tmp_path):
    path = tmp_path / "closed.parquet"
    pq.write_table(pa.table({"a": [1, 2]}), path)
    remote = blosc2.open(path)
    column = remote._cols["a"]
    remote.close()
    with pytest.raises(RuntimeError, match="closed"):
        column[0]


def test_refresh_invalidates_old_views(tmp_path):
    path = tmp_path / "refresh.parquet"
    pq.write_table(pa.table({"a": [1, 2]}), path)
    with blosc2.open(path) as remote:
        old_column = remote._cols["a"]
        old_view = remote.select(["a"])
        pq.write_table(pa.table({"a": [3, 4]}), path)
        remote.refresh()
        assert remote["a"][:].tolist() == [3, 4]
        with pytest.raises(RuntimeError, match=r"closed|stale"):
            old_column[0]
        with pytest.raises(RuntimeError, match=r"closed|stale"):
            old_view.to_arrow()


def test_reader_options_and_eager_path(tmp_path):
    path = tmp_path / "dictionary.parquet"
    pq.write_table(pa.table({"name": ["aa", "bb", "aa", None]}), path, row_group_size=2)
    options = {"read_dictionary": ["name"]}
    with blosc2.CTable.from_parquet(path, **options) as eager:
        with blosc2.open(path, parquet_options=options) as remote:
            assert schema_to_dict(remote._schema) == schema_to_dict(eager._schema)
            assert remote.to_arrow().equals(eager.to_arrow())
        with blosc2.open(path, lazy=False, parquet_options=options) as imported:
            assert imported.to_arrow().equals(eager.to_arrow())


def test_timestamp_reader_coercion(tmp_path):
    path = tmp_path / "int96.parquet"
    source = pa.table({"time": pa.array([0, None, 1234], type=pa.timestamp("ms"))})
    pq.write_table(source, path, row_group_size=2, use_deprecated_int96_timestamps=True)
    options = {"coerce_int96_timestamp_unit": "ms"}
    with blosc2.CTable.from_parquet(path, **options) as eager:
        with blosc2.open(path, parquet_options=options) as remote:
            assert schema_to_dict(remote._schema) == schema_to_dict(eager._schema)
            assert remote.to_arrow().equals(eager.to_arrow())


def test_fixed_string_width_error_on_access(tmp_path):
    path = tmp_path / "strings.parquet"
    pq.write_table(pa.table({"s": ["a", "long"]}), path, row_group_size=1)
    with blosc2.open(path, string_max_length=2) as remote:
        assert remote["s"][0] == "a"
        with pytest.raises(ValueError, match="longer than max_length=2"):
            remote["s"][1]


def test_fixed_string_width_mapping_matches_eager(tmp_path):
    path = tmp_path / "strings.parquet"
    pq.write_table(
        pa.table({"s": ["a", "é", None], "b": [b"\x00", b"hi", None], "v": ["wide", "text", ""]}),
        path,
        row_group_size=2,
    )
    options = {"string_max_length": {"s": 2, "b": 2}}
    with blosc2.CTable.from_parquet(path, **options) as eager, blosc2.open(path, **options) as remote:
        assert schema_to_dict(remote._schema) == schema_to_dict(eager._schema)
        assert remote.to_arrow().equals(eager.to_arrow())


def test_reordered_duplicate_and_empty_rows(tmp_path):
    path = tmp_path / "rows.parquet"
    pq.write_table(pa.table({"x": [0, 1, 2, 3, 4]}), path, row_group_size=2)
    with blosc2.open(path) as remote:
        assert remote[-1].x == 4
        assert remote[[4, 1, 4]].to_arrow().column("x").to_pylist() == [4, 1, 4]
        assert remote[2:0:-1].to_arrow().column("x").to_pylist() == [2, 1]
        assert remote["x"][np.array([True, False, True, False, False])].tolist() == [0, 2]
        assert len(remote[0:0]) == 0
        with pytest.raises(IndexError):
            remote._cols["x"][5]


def test_dictionary_codes_change_between_groups(tmp_path):
    path = tmp_path / "reordered-dictionary.parquet"
    schema = pa.schema([pa.field("category", pa.dictionary(pa.int8(), pa.string()))])
    with pq.ParquetWriter(path, schema) as writer:
        for dictionary in (["a", "b"], ["b", "a"]):
            values = pa.DictionaryArray.from_arrays(pa.array([0, 1], type=pa.int8()), pa.array(dictionary))
            writer.write_batch(pa.record_batch([values], schema=schema))
    with blosc2.CTable.from_parquet(path) as eager, blosc2.open(path) as remote:
        assert remote.to_arrow().equals(eager.to_arrow())


def test_disk_cache_reuses_group_and_root_map(tmp_path):
    path = tmp_path / "root.parquet"
    source = pa.Table.from_arrays(
        [pa.array([[{"id": i}] for i in range(6)], type=pa.list_(pa.struct([("id", pa.int32())])))],
        names=[""],
    )
    pq.write_table(source, path, row_group_size=2)
    cache_dir = tmp_path / "cache"
    with blosc2.open(path, cache_dir=cache_dir) as first:
        prepared_requests = first.traffic.requests
        assert first["id"][-1] == 5
        assert first.cache_bytes > 0
    with blosc2.open(path, cache_dir=cache_dir) as second:
        reopened_requests = second.traffic.requests
        assert reopened_requests < prepared_requests  # row map reused
        assert second["id"][-1] == 5
        assert second.traffic.requests == reopened_requests  # converted group reused
    cache_root = next(cache_dir.glob("root.parquet--*"))
    generation = next(cache_root.glob("*.b2d"))
    with blosc2.open(generation) as reopened:
        assert reopened.col_names == ["id"]
        before = reopened.traffic.requests
        assert reopened["id"][-1] == 5
        assert reopened.traffic.requests == before


def test_warm_open_uses_retained_metadata_and_lazily_opens_arrow(tmp_path, monkeypatch):
    path = tmp_path / "source.parquet"
    cache_dir = tmp_path / "cache"
    pq.write_table(pa.table({"x": [1, 2, 3, 4], "y": [5, 6, 7, 8]}), path, row_group_size=2)
    with blosc2.open(path, cache_dir=cache_dir) as cold:
        assert cold["x"][0] == 1

    real_reader = pq.ParquetFile
    readers = []

    def record_reader(*args, **kwargs):
        assert kwargs.get("metadata") is not None
        readers.append(kwargs["metadata"])
        return real_reader(*args, **kwargs)

    monkeypatch.setattr(pq, "ParquetFile", record_reader)
    real_from_arrow = CTable.from_arrow
    monkeypatch.setattr(CTable, "from_arrow", lambda *args, **kwargs: pytest.fail("schema was inferred"))
    monkeypatch.setattr(CTable, "load", lambda *args, **kwargs: pytest.fail("cached group was copied"))
    monkeypatch.setattr(remote_parquet, "_source_marker", lambda *args: pytest.fail("source was checked"))
    with blosc2.open(path, cache_dir=cache_dir) as warm:
        assert warm.nrows == 4
        assert warm.metadata_bytes > 0
        assert readers == []
        assert warm["x"][0] == 1
        assert readers == []
        monkeypatch.setattr(CTable, "from_arrow", real_from_arrow)
        assert warm["y"][3] == 8
        assert len(readers) == 1


def test_retained_metadata_is_separate_for_projections(tmp_path):
    path = tmp_path / "source.parquet"
    cache_dir = tmp_path / "cache"
    pq.write_table(pa.table({"x": [1, 2], "y": [3, 4]}), path)
    with blosc2.open(path, cache_dir=cache_dir, columns=["x"]) as x_only:
        assert x_only.col_names == ["x"]
    with blosc2.open(path, cache_dir=cache_dir, columns=["y"]) as y_only:
        assert y_only.col_names == ["y"]
    assert len(list(cache_dir.glob("source.parquet--*"))) == 2


def test_invalid_retained_metadata_is_rebuilt(tmp_path):
    path = tmp_path / "source.parquet"
    cache_dir = tmp_path / "cache"
    pq.write_table(pa.table({"x": [1, 2]}), path)
    with blosc2.open(path, cache_dir=cache_dir) as table:
        disk = table._remote_storage()._owner.disk
        manifest = disk.load()
        manifest["metadata"]["parquet_discovery"]["schema"] = {"version": 999}
        disk.publish(manifest)
    with blosc2.open(path, cache_dir=cache_dir) as rebuilt:
        assert rebuilt["x"][0] == 1
        assert rebuilt._remote_storage()._owner.disk.load()["metadata"]["parquet_discovery"]["version"] == 1


def test_existing_cache_adds_local_marker_index(tmp_path, monkeypatch):
    path = tmp_path / "source.parquet"
    cache_dir = tmp_path / "cache"
    pq.write_table(pa.table({"x": [1, 2]}), path)
    with blosc2.open(path, cache_dir=cache_dir):
        pass
    index = next(cache_dir.glob(".parquet-marker-*.json"))
    index.unlink()
    with blosc2.open(path, cache_dir=cache_dir):
        pass
    assert index.is_file()
    monkeypatch.setattr(remote_parquet, "_source_marker", lambda *args: pytest.fail("source was checked"))
    with blosc2.open(path, cache_dir=cache_dir) as warm:
        assert warm.nrows == 2


@pytest.mark.parametrize("disk", [False, True])
def test_cache_archive_reuses_warm_group_and_fetches_cold_group(tmp_path, disk, monkeypatch):
    path = tmp_path / "source.parquet"
    pq.write_table(pa.table({"x": [1, 2, 3, 4], "y": [5, 6, 7, 8]}), path, row_group_size=2)
    options = {"cache_dir": tmp_path / "cache"} if disk else {}
    artifact = tmp_path / "reference.b2z"
    cold_artifact = tmp_path / "cold-reference.b2z"
    replace = os.replace

    def checked_replace(source, destination):
        if destination in (artifact, cold_artifact):
            assert source.parent.parent == destination.parent
        return replace(source, destination)

    monkeypatch.setattr(remote_parquet.os, "replace", checked_replace)
    with blosc2.open(path, **options) as original:
        assert original["x"][0] == 1
        original.save(artifact)
        original.save(cold_artifact, include_cache=False)
    with blosc2.open(artifact) as restored:
        assert restored.col_names == ["x", "y"]
        before = restored.traffic.requests
        assert restored["x"][0] == 1
        assert restored.traffic.requests == before
        assert restored["y"][3] == 8
        assert restored.traffic.requests > before
    with blosc2.open(cold_artifact) as cold:
        before = cold.traffic.requests
        assert cold["x"][0] == 1
        assert cold.traffic.requests > before
    result = subprocess.check_output(
        [sys.executable, "-c", "import blosc2,sys; print(blosc2.open(sys.argv[1])['x'][0])", str(artifact)],
        text=True,
    )
    assert result.strip() == "1"


def test_cache_generation_reopens_with_compression_options(tmp_path):
    path = tmp_path / "source.parquet"
    pq.write_table(pa.table({"x": [1, 2, 3]}), path)
    cache_dir = tmp_path / "cache"
    with blosc2.open(path, cache_dir=cache_dir, cparams=blosc2.CParams(clevel=1)) as original:
        assert original["x"][0] == 1
    generation = next(cache_dir.glob("*/*.b2d"))
    with blosc2.open(generation) as reopened:
        assert reopened["x"][0] == 1


def test_shared_disk_cache_reuses_converted_group_across_processes(tmp_path):
    path = tmp_path / "shared.parquet"
    cache_dir = tmp_path / "cache"
    pq.write_table(pa.table({"id": list(range(6))}), path, row_group_size=2)
    with blosc2.open(path, cache_dir=cache_dir, shared_cache=True) as remote:
        assert remote["id"][5] == 5
    script = (
        "import blosc2,sys; "
        "t=blosc2.open(sys.argv[1],cache_dir=sys.argv[2],shared_cache=True); "
        "before=t.traffic.requests; print(t['id'][5],t.traffic.requests-before); t.close()"
    )
    output = subprocess.check_output([sys.executable, "-c", script, str(path), str(cache_dir)], text=True)
    assert output.strip() == "5 0"
    with blosc2.RemoteCTable.with_sparse_cache(path, cache_dir) as remote:
        before = remote.traffic.requests
        assert remote["id"][5] == 5
        assert remote.traffic.requests == before


def test_shared_disk_cache_concurrent_cold_reads(tmp_path):
    path = tmp_path / "shared-cold.parquet"
    cache_dir = tmp_path / "cache"
    pq.write_table(pa.table({"id": list(range(12))}), path, row_group_size=2)
    with blosc2.open(path, cache_dir=cache_dir, shared_cache=True) as first:
        with blosc2.open(path, cache_dir=cache_dir, shared_cache=True) as second:
            baseline = first.traffic.requests + second.traffic.requests
            with ThreadPoolExecutor(max_workers=2) as pool:
                assert list(pool.map(lambda table: table["id"][11], (first, second))) == [11, 11]
            assert first.traffic.requests + second.traffic.requests == baseline + 1


def test_disk_cache_eviction_and_corrupt_entry(tmp_path):
    path = tmp_path / "data.parquet"
    cache_dir = tmp_path / "cache"
    pq.write_table(pa.table({"id": list(range(6))}), path, row_group_size=2)
    with blosc2.open(path, cache_dir=cache_dir, max_cache_bytes=1) as remote:
        assert remote["id"][2] == 2
        assert remote["id"][5] == 5
    entries = list(cache_dir.rglob("*.b2d"))
    entries = [entry for entry in entries if entry.name.startswith(("0-", "1-", "2-"))]
    assert not entries  # The common cache budget does not retain an oversized group.
    with blosc2.open(path, cache_dir=cache_dir, max_cache_bytes=None) as remote:
        assert remote["id"][5] == 5
    entries = [entry for entry in cache_dir.rglob("*.b2d") if entry.name.startswith("2-")]
    assert len(entries) == 1
    (entries[0] / "_meta.b2f").write_bytes(b"broken")
    with blosc2.open(path, cache_dir=cache_dir) as remote:
        assert remote["id"][5] == 5


def test_refresh_disk_cache_uses_new_generation(tmp_path):
    path = tmp_path / "changing.parquet"
    cache_dir = tmp_path / "cache"
    pq.write_table(pa.table({"id": [1, 2]}), path)
    with blosc2.open(path, cache_dir=cache_dir) as remote:
        assert remote["id"][0] == 1
        pq.write_table(pa.table({"id": [3, 4, 5]}), path)
        remote.refresh()
        assert remote["id"][0] == 3


def test_same_size_source_replacement_rebuilds_metadata(tmp_path):
    path = tmp_path / "changing.parquet"
    cache_dir = tmp_path / "cache"
    pq.write_table(pa.table({"id": [1, 2]}), path)
    old = path.stat()
    with blosc2.open(path, cache_dir=cache_dir) as first:
        assert first["id"][0] == 1
    pq.write_table(pa.table({"id": [3, 4]}), path)
    assert path.stat().st_size == old.st_size
    os.utime(path, ns=(old.st_atime_ns, old.st_mtime_ns + 2_000_000_000))
    with blosc2.open(path, cache_dir=cache_dir) as replaced:
        assert replaced["id"][0] == 1  # Cached sources stay immutable until refresh.
        replaced.refresh()
        assert replaced["id"][0] == 3


def test_refresh_unchanged_disk_source_keeps_cache_owner(tmp_path):
    path = tmp_path / "unchanged.parquet"
    pq.write_table(pa.table({"id": [1, 2]}), path)
    with blosc2.open(path, cache_dir=tmp_path / "cache") as remote:
        remote.refresh()
        assert remote["id"][0] == 1


def test_http_b2_file_id_supplies_missing_cache_validator(monkeypatch):
    class Response:
        def __init__(self):
            self.headers = {"Content-Length": "49961641", "x-bz-file-id": "file-version-1"}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            pass

    class Session:
        def head(self, *args, **kwargs):
            return Response()

    class Filesystem:
        loop = fsspec.asyn.get_loop()

        def __init__(self):
            self.kwargs = {}

        def info(self, path):
            pytest.fail("HTTP source information required a second request")

        async def set_session(self):
            return Session()

        def encode_url(self, path):
            return path

        def _raise_not_found_for_status(self, response, path):
            pass

    monkeypatch.setattr(fsspec.core, "url_to_fs", lambda *args, **kwargs: (Filesystem(), args[0]))
    marker = remote_parquet._source_marker("https://example.com/table.parquet", None)
    assert marker == {"size": "49961641", "x-bz-file-id": "file-version-1"}


def test_http_range_requests_are_narrow(tmp_path):
    path = tmp_path / "served.parquet"
    pq.write_table(
        pa.table({"x": list(range(10_000)), "y": [f"value-{i:05d}" for i in range(10_000)]}),
        path,
        row_group_size=1_000,
    )
    counts = {"requests": 0, "bytes": 0, "heads": 0}

    class Ranged(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_HEAD(self):
            counts["heads"] += 1
            self.send_response(200)
            self.send_header("Content-Length", str(path.stat().st_size))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Last-Modified", formatdate(path.stat().st_mtime, usegmt=True))
            self.end_headers()

        def do_GET(self):
            body = path.read_bytes()
            span = self.headers.get("Range")
            if span:
                first, _, last = span.removeprefix("bytes=").partition("-")
                if first:
                    first = int(first)
                    last = int(last) if last else len(body) - 1
                else:
                    first, last = max(0, len(body) - int(last)), len(body) - 1
                data = body[first : last + 1]
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {first}-{last}/{len(body)}")
            else:
                data = body
                self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Last-Modified", formatdate(path.stat().st_mtime, usegmt=True))
            self.end_headers()
            counts["requests"] += 1
            counts["bytes"] += len(data)
            self.wfile.write(data)

    try:
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Ranged)
    except PermissionError:
        pytest.skip("localhost binding is blocked by the sandbox")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/served.parquet?signature=test"
        with blosc2.open(url, storage_options={"block_size": 4096, "cache_type": "none"}) as remote:
            assert counts["requests"] == 1
            assert counts["heads"] == 1
            assert "signature" not in remote.source["urlpath"]
            with pytest.raises(ValueError, match="credential-like"):
                remote.save(tmp_path / "signed.b2nd")
            before = counts.copy()
            assert remote["x"][9500] == 9500
            assert counts["requests"] > before["requests"]
            assert counts["bytes"] - before["bytes"] < path.stat().st_size
            warm = counts.copy()
            assert remote["x"][9500] == 9500
            assert counts == warm
        clean_url = url.split("?", 1)[0]
        cache_dir = tmp_path / "http-cache"
        with blosc2.open(clean_url, cache_dir=cache_dir) as remote:
            assert remote["x"][9500] == 9500
        with blosc2.open(clean_url, cache_dir=cache_dir) as remote:
            before = counts.copy()
            assert remote["x"][9500] == 9500
            assert counts["requests"] == before["requests"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.network
def test_s3_parquet_smoke():
    url = os.environ.get("BLOSC2_REMOTE_PARQUET_S3_URL")
    if not url:
        pytest.skip("set BLOSC2_REMOTE_PARQUET_S3_URL to an accessible Parquet file")
    with blosc2.open(url, source_format="parquet") as remote:
        assert remote.col_names
        if len(remote):
            remote[remote.col_names[0]][0]


def test_http_without_range_support_fails_clearly(tmp_path):
    path = tmp_path / "large.parquet"
    pq.write_table(pa.table({"x": list(range(20_000))}), path, row_group_size=1_000)

    class Unranged(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_HEAD(self):
            self.send_response(200)
            self.send_header("Content-Length", str(path.stat().st_size))
            self.end_headers()

        def do_GET(self):
            body = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    try:
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Unranged)
    except PermissionError:
        pytest.skip("localhost binding is blocked by the sandbox")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/large.parquet"
        with pytest.raises(ValueError, match="range requests"):
            blosc2.open(url, storage_options={"block_size": 4096, "cache_type": "none"})
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_shared_warm_read_holds_cache_lock(tmp_path, monkeypatch):
    path = tmp_path / "source.parquet"
    pq.write_table(pa.table({"x": [1, 2]}), path)
    with blosc2.open(path, cache_dir=tmp_path / "cache", shared_cache=True) as remote:
        assert remote["x"][0] == 1
        owner = remote._remote_storage()._owner
        guard = owner.disk.guard
        locked = False

        @contextmanager
        def tracked_guard():
            nonlocal locked
            with guard():
                locked = True
                try:
                    yield
                finally:
                    locked = False

        monkeypatch.setattr(owner.disk, "guard", tracked_guard)
        original_open = CTable.open

        def checked_open(*args, **kwargs):
            table = original_open(*args, **kwargs)
            column = table._cols["x"]

            class CheckedColumn:
                def __getitem__(self, key):
                    assert locked, "shared cache must remain locked during lazy reads"
                    return column[key]

            table._cols["x"] = CheckedColumn()
            return table

        monkeypatch.setattr(CTable, "open", checked_open)
        assert remote["x"][0] == 1
        assert not locked


def test_archive_rejects_invalid_generation(tmp_path):
    import zipfile

    path = tmp_path / "source.parquet"
    pq.write_table(pa.table({"x": [1]}), path)
    archive_path = tmp_path / "reference.b2z"
    with blosc2.open(path) as remote:
        remote.save(archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        carrier = blosc2.schunk_from_cframe(archive.read("embed.b2e"))
    manifest = carrier.vlmeta["b2remote_manifest"]
    manifest["generation"] = str(tmp_path / "escaped")
    carrier = blosc2.SChunk(meta={"b2tree": {"version": 1}, "b2remote_store": {"version": 1}})
    carrier.vlmeta["b2remote_manifest"] = manifest
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("embed.b2e", carrier.to_cframe())
    with pytest.raises(ValueError, match="generation"):
        blosc2.open(archive_path)
    assert not (tmp_path / "escaped.b2d").exists()


@pytest.mark.parametrize("whole_row", [False, True])
def test_refresh_waits_for_parquet_read(tmp_path, monkeypatch, whole_row):
    path = tmp_path / "source.parquet"
    pq.write_table(pa.table({"x": [1, 2, 3, 4], "y": [5, 6, 7, 8]}), path, row_group_size=2)
    with blosc2.open(path) as remote:
        owner = remote._remote_storage()._owner
        original_group = owner.group
        entered = threading.Event()
        resume = threading.Event()
        refresh_started = threading.Event()
        refresh_done = threading.Event()

        @contextmanager
        def paused_group(number, physical):
            with original_group(number, physical) as table:
                if number == 0 and physical == "x":
                    entered.set()
                    assert resume.wait(5)
                yield table

        monkeypatch.setattr(owner, "group", paused_group)

        def refresh():
            refresh_started.set()
            remote.refresh()
            refresh_done.set()

        with ThreadPoolExecutor(max_workers=2) as pool:
            read = pool.submit(lambda: remote[0] if whole_row else remote["x"][:])
            assert entered.wait(5)
            update = pool.submit(refresh)
            assert refresh_started.wait(5)
            assert not refresh_done.wait(0.1)
            resume.set()
            result = read.result(timeout=5)
            update.result(timeout=5)
        if whole_row:
            assert (result.x, result.y) == (1, 5)
        else:
            assert result.tolist() == [1, 2, 3, 4]


def test_oversized_memory_group_is_not_retained(tmp_path):
    path = tmp_path / "source.parquet"
    pq.write_table(pa.table({"x": [1, 2]}), path)
    with blosc2.open(path, max_cache_bytes=1) as remote:
        assert remote["x"][0] == 1
        assert remote.cache_bytes == 0
        before = remote.traffic.requests
        assert remote["x"][1] == 2
        assert remote.traffic.requests > before
