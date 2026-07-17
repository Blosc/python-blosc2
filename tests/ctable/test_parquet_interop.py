#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""Tests for CTable.to_parquet(), from_parquet(), iter_arrow_batches(),
and from_arrow()."""

import io
from dataclasses import dataclass

import numpy as np
import pytest

import blosc2
from blosc2 import CTable
from blosc2.schema import ObjectSpec, StructSpec

# Scalar Arrow strings import as utf8 on NumPy >= 2.0 (StringDType available)
# and fall back to vlstring (native-None nulls) on older NumPy.
HAVE_STRING_DTYPE = hasattr(np.dtypes, "StringDType")

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")


# ---------------------------------------------------------------------------
# Shared fixtures / dataclasses
# ---------------------------------------------------------------------------


@dataclass
class Row:
    id: int = blosc2.field(blosc2.int64(ge=0))
    score: float = blosc2.field(blosc2.float64(ge=0, le=100), default=0.0)
    active: bool = blosc2.field(blosc2.bool(), default=True)
    label: str = blosc2.field(blosc2.string(max_length=32), default="")


DATA10 = [(i, float(i * 10 % 100), i % 2 == 0, f"row{i}") for i in range(10)]


# ---------------------------------------------------------------------------
# iter_arrow_batches
# ---------------------------------------------------------------------------


class TestIterArrowBatches:
    def test_yields_record_batches(self):
        t = CTable(Row, new_data=DATA10)
        batches = list(t.iter_arrow_batches())
        assert all(isinstance(b, pa.RecordBatch) for b in batches)

    def test_total_row_count(self):
        t = CTable(Row, new_data=DATA10)
        total = sum(len(b) for b in t.iter_arrow_batches())
        assert total == 10

    def test_batching_splits_correctly(self):
        t = CTable(Row, new_data=DATA10)
        batches = list(t.iter_arrow_batches(batch_size=3))
        sizes = [len(b) for b in batches]
        assert sizes == [3, 3, 3, 1]

    def test_column_names(self):
        t = CTable(Row, new_data=DATA10)
        (batch,) = t.iter_arrow_batches()
        assert batch.schema.names == ["id", "score", "active", "label"]

    def test_int_values(self):
        t = CTable(Row, new_data=DATA10)
        (batch,) = t.iter_arrow_batches()
        np.testing.assert_array_equal(batch.column("id").to_pylist(), [r[0] for r in DATA10])

    def test_float_values(self):
        t = CTable(Row, new_data=DATA10)
        (batch,) = t.iter_arrow_batches()
        np.testing.assert_allclose(batch.column("score").to_pylist(), [r[1] for r in DATA10])

    def test_bool_values(self):
        t = CTable(Row, new_data=DATA10)
        (batch,) = t.iter_arrow_batches()
        assert batch.column("active").to_pylist() == [r[2] for r in DATA10]

    def test_string_values(self):
        t = CTable(Row, new_data=DATA10)
        (batch,) = t.iter_arrow_batches()
        assert batch.column("label").to_pylist() == [r[3] for r in DATA10]

    def test_empty_table_yields_nothing(self):
        t = CTable(Row)
        batches = list(t.iter_arrow_batches())
        assert batches == []

    def test_column_projection(self):
        t = CTable(Row, new_data=DATA10)
        (batch,) = t.iter_arrow_batches(columns=["id", "score"])
        assert batch.schema.names == ["id", "score"]
        assert len(batch) == 10

    def test_column_projection_unknown_raises(self):
        t = CTable(Row, new_data=DATA10)
        with pytest.raises(KeyError, match="nope"):
            list(t.iter_arrow_batches(columns=["nope"]))

    def test_skips_deleted_rows(self):
        t = CTable(Row, new_data=DATA10)
        t.delete([0, 1, 2])
        total = sum(len(b) for b in t.iter_arrow_batches())
        assert total == 7

    def test_include_computed_false(self):
        t = CTable(Row, new_data=DATA10)
        t.add_computed_column("double_id", "id * 2")
        (batch,) = t.iter_arrow_batches(include_computed=False)
        assert "double_id" not in batch.schema.names

    def test_computed_column_values(self):
        t = CTable(Row, new_data=DATA10)
        t.add_computed_column("double_id", "id * 2")
        (batch,) = t.iter_arrow_batches()
        assert batch.column("double_id").to_pylist() == [i * 2 for i in range(10)]

    def test_invalid_batch_size(self):
        t = CTable(Row, new_data=DATA10)
        with pytest.raises(ValueError, match="batch_size"):
            list(t.iter_arrow_batches(batch_size=0))


# ---------------------------------------------------------------------------
# to_parquet / from_parquet round-trips
# ---------------------------------------------------------------------------


class TestParquetRoundTrip:
    def test_basic_roundtrip(self, tmp_path):
        t = CTable(Row, new_data=DATA10)
        path = tmp_path / "basic.parquet"
        t.to_parquet(path)
        t2 = CTable.from_parquet(path)
        assert len(t2) == 10
        assert t2.col_names == ["id", "score", "active", "label"]
        np.testing.assert_array_equal(t2["id"][:], t["id"][:])
        np.testing.assert_allclose(t2["score"][:], t["score"][:])
        np.testing.assert_array_equal(t2["active"][:], t["active"][:])
        # label is re-imported as vlstring when no string_max_length is given
        assert list(t2["label"][:]) == t["label"][:].tolist()

    def test_roundtrip_all_numeric_types(self, tmp_path):
        at = pa.table(
            {
                "i8": pa.array([1, 2, 3], type=pa.int8()),
                "i16": pa.array([1, 2, 3], type=pa.int16()),
                "i32": pa.array([1, 2, 3], type=pa.int32()),
                "i64": pa.array([1, 2, 3], type=pa.int64()),
                "u8": pa.array([1, 2, 3], type=pa.uint8()),
                "u16": pa.array([1, 2, 3], type=pa.uint16()),
                "u32": pa.array([1, 2, 3], type=pa.uint32()),
                "u64": pa.array([1, 2, 3], type=pa.uint64()),
                "f32": pa.array([1.0, 2.0, 3.0], type=pa.float32()),
                "f64": pa.array([1.0, 2.0, 3.0], type=pa.float64()),
            }
        )
        t = CTable.from_arrow(at.schema, at.to_batches())
        path = tmp_path / "numeric.parquet"
        t.to_parquet(path)
        t2 = CTable.from_parquet(path)
        assert t2.col_names == list(at.column_names)
        assert len(t2) == 3

    def test_roundtrip_bool(self, tmp_path):
        at = pa.table({"flag": pa.array([True, False, True], type=pa.bool_())})
        t = CTable.from_arrow(at.schema, at.to_batches())
        path = tmp_path / "bool.parquet"
        t.to_parquet(path)
        t2 = CTable.from_parquet(path)
        assert t2["flag"][:].tolist() == [True, False, True]

    def test_roundtrip_strings(self, tmp_path):
        at = pa.table({"name": pa.array(["alice", "bob", "carol"], type=pa.string())})
        t = CTable.from_arrow(at.schema, at.to_batches())
        path = tmp_path / "strings.parquet"
        t.to_parquet(path)
        t2 = CTable.from_parquet(path)
        # vlstring column — [:] returns a Python list, not a numpy array
        assert list(t2["name"][:]) == ["alice", "bob", "carol"]

    def test_roundtrip_bytes(self, tmp_path):
        at = pa.table({"data": pa.array([b"hello", b"world", b"foo"], type=pa.large_binary())})
        t = CTable.from_arrow(at.schema, at.to_batches())
        path = tmp_path / "bytes.parquet"
        t.to_parquet(path)
        t2 = CTable.from_parquet(path)
        # vlbytes column — [:] returns a Python list of bytes objects
        raw = t2["data"][:]
        assert raw == [b"hello", b"world", b"foo"]

    def test_roundtrip_list_column(self, tmp_path):
        @dataclass
        class ListRow:
            vals: list[int] = blosc2.field(  # noqa: RUF009
                blosc2.list(blosc2.int64(), storage="batch", serializer="msgpack")
            )

        data = [([1, 2, 3],), ([4, 5],), ([],)]
        t = CTable(ListRow, new_data=data)
        path = tmp_path / "lists.parquet"
        t.to_parquet(path)
        t2 = CTable.from_parquet(path)
        assert len(t2) == 3
        assert t2["vals"][0] == [1, 2, 3]
        assert t2["vals"][1] == [4, 5]
        assert t2["vals"][2] == []

    def test_roundtrip_list_struct_column(self, tmp_path):
        struct_type = pa.struct([pa.field("a", pa.int32()), pa.field("b", pa.string())])
        at = pa.table(
            {
                "items": pa.array(
                    [[{"a": 1, "b": "x"}], None, [{"a": 2, "b": "yy"}]],
                    type=pa.list_(struct_type),
                )
            }
        )
        path = tmp_path / "list_struct.parquet"
        pq.write_table(at, path)
        t = CTable.from_parquet(path)
        assert t["items"][0] == [{"a": 1, "b": "x"}]
        assert t["items"][1] is None
        out = tmp_path / "list_struct_out.parquet"
        t.to_parquet(out)
        rt = pq.read_table(out)
        assert rt.schema == at.schema
        assert rt["items"].to_pylist() == at["items"].to_pylist()
        assert "arrow" in t._schema.metadata

    def test_roundtrip_top_level_struct_column(self, tmp_path):
        struct_type = pa.struct([pa.field("a", pa.int32()), pa.field("b", pa.string())])
        at = pa.table({"props": pa.array([{"a": 1, "b": "x"}, None, {"a": 2, "b": "yy"}], type=struct_type)})
        path = tmp_path / "struct.parquet"
        pq.write_table(at, path)

        t = CTable.from_parquet(path)
        assert t["props"][:] == [{"a": 1, "b": "x"}, None, {"a": 2, "b": "yy"}]
        assert isinstance(t._schema.columns_by_name["props"].spec, StructSpec)

        out = tmp_path / "struct_out.parquet"
        t.to_parquet(out)
        rt = pq.read_table(out)
        assert rt.schema == at.schema
        assert rt["props"].to_pylist() == at["props"].to_pylist()

    def test_top_level_struct_column_persistence(self, tmp_path):
        @dataclass
        class StructRow:
            props: dict = blosc2.field(  # noqa: RUF009
                blosc2.struct({"a": blosc2.int32(), "b": blosc2.vlstring()}, nullable=True)
            )

        path = tmp_path / "struct_table.b2d"
        t = CTable(StructRow, urlpath=str(path), mode="w")
        t.extend([[{"a": 1, "b": "x"}], [None], [{"a": 2, "b": "yy"}]])
        t.close()

        reopened = CTable.open(str(path), mode="r")
        assert reopened["props"][:] == [{"a": 1, "b": "x"}, None, {"a": 2, "b": "yy"}]

    def test_from_arrow_object_fallback_for_unsupported_type(self):
        map_type = pa.map_(pa.string(), pa.int32())
        batch = pa.record_batch(
            [pa.array([[("a", 1)], None, [("b", 2), ("c", 3)]], type=map_type)], names=["attrs"]
        )

        with pytest.raises(TypeError, match="object_fallback=True"):
            CTable.from_arrow(batch.schema, [batch])

        t = CTable.from_arrow(batch.schema, [batch], object_fallback=True)
        assert isinstance(t._schema.columns_by_name["attrs"].spec, ObjectSpec)
        assert t["attrs"][:] == [[("a", 1)], None, [("b", 2), ("c", 3)]]

    def test_from_parquet_does_not_use_object_fallback(self, tmp_path):
        map_type = pa.map_(pa.string(), pa.int32())
        at = pa.table({"attrs": pa.array([[("a", 1)]], type=map_type)})
        path = tmp_path / "map.parquet"
        pq.write_table(at, path)

        with pytest.raises(TypeError, match="object_fallback=True"):
            CTable.from_parquet(path)

    def test_empty_table_export_import(self, tmp_path):
        t = CTable(Row)
        path = tmp_path / "empty.parquet"
        t.to_parquet(path)
        t2 = CTable.from_parquet(path)
        assert len(t2) == 0
        assert t2.col_names == ["id", "score", "active", "label"]

    def test_column_projection_export(self, tmp_path):
        t = CTable(Row, new_data=DATA10)
        path = tmp_path / "proj.parquet"
        t.to_parquet(path, columns=["id", "score"])
        t2 = CTable.from_parquet(path)
        assert t2.col_names == ["id", "score"]
        assert len(t2) == 10

    def test_column_projection_import(self, tmp_path):
        t = CTable(Row, new_data=DATA10)
        path = tmp_path / "full.parquet"
        t.to_parquet(path)
        t2 = CTable.from_parquet(path, columns=["id", "label"])
        assert t2.col_names == ["id", "label"]
        assert len(t2) == 10

    def test_computed_column_exported_as_values(self, tmp_path):
        t = CTable(Row, new_data=DATA10)
        t.add_computed_column("double_id", "id * 2")
        path = tmp_path / "computed.parquet"
        t.to_parquet(path)
        t2 = CTable.from_parquet(path)
        assert "double_id" in t2.col_names
        # double_id is stored, not computed in t2
        assert "double_id" not in t2._computed_cols
        np.testing.assert_array_equal(
            t2["double_id"][:], np.array([i * 2 for i in range(10)], dtype=np.int64)
        )

    def test_exclude_computed_columns(self, tmp_path):
        t = CTable(Row, new_data=DATA10)
        t.add_computed_column("double_id", "id * 2")
        path = tmp_path / "no_computed.parquet"
        t.to_parquet(path, include_computed=False)
        t2 = CTable.from_parquet(path)
        assert "double_id" not in t2.col_names

    def test_only_live_rows_exported(self, tmp_path):
        t = CTable(Row, new_data=DATA10)
        t.delete([0, 1])
        path = tmp_path / "deleted.parquet"
        t.to_parquet(path)
        t2 = CTable.from_parquet(path)
        assert len(t2) == 8
        np.testing.assert_array_equal(t2["id"][:], list(range(2, 10)))

    def test_multiple_batches_written(self, tmp_path):
        t = CTable(Row, new_data=DATA10)
        path = tmp_path / "multi.parquet"
        t.to_parquet(path, batch_size=3)
        meta = pq.read_metadata(path)
        assert meta.num_row_groups == 4  # 3+3+3+1

    def test_persistent_urlpath(self, tmp_path):
        t = CTable(Row, new_data=DATA10)
        parquet_path = tmp_path / "data.parquet"
        ctable_path = str(tmp_path / "data.b2d")
        t.to_parquet(parquet_path)
        t2 = CTable.from_parquet(parquet_path, urlpath=ctable_path)
        assert len(t2) == 10
        import os

        assert os.path.exists(ctable_path)

    def test_different_compression(self, tmp_path):
        t = CTable(Row, new_data=DATA10)
        for codec in ["snappy", "lz4", None]:
            path = tmp_path / f"{codec}.parquet"
            t.to_parquet(path, compression=codec)
            t2 = CTable.from_parquet(path)
            assert len(t2) == 10

    def test_roundtrip_larger_table_batch_import(self, tmp_path):
        """Verify correct row count with batch_size < n_rows on import."""
        t = CTable(Row, new_data=DATA10)
        path = tmp_path / "large.parquet"
        t.to_parquet(path)
        t2 = CTable.from_parquet(path, batch_size=3)
        assert len(t2) == 10
        np.testing.assert_array_equal(t2["id"][:], t["id"][:])

    def test_interop_read_with_pyarrow(self, tmp_path):
        """CTable-written Parquet is readable by PyArrow."""
        t = CTable(Row, new_data=DATA10)
        path = tmp_path / "compat.parquet"
        t.to_parquet(path)
        at = pq.read_table(path)
        assert len(at) == 10
        assert at.column_names == ["id", "score", "active", "label"]

    def test_interop_write_with_pyarrow(self, tmp_path):
        """PyArrow-written Parquet is readable by CTable."""
        at = pa.table(
            {
                "x": pa.array([1, 2, 3], type=pa.int32()),
                "y": pa.array([1.1, 2.2, 3.3], type=pa.float64()),
            }
        )
        path = tmp_path / "pyarrow_written.parquet"
        pq.write_table(at, path)
        t = CTable.from_parquet(path)
        assert len(t) == 3
        assert t.col_names == ["x", "y"]

    def test_from_arrow_blosc2_batch_size_default(self):
        at = pa.table({"vals": pa.array([[1], [2, 3]], type=pa.list_(pa.int64()))})
        t = CTable.from_arrow(at.schema, at.to_batches())
        assert t._schema.columns_by_name["vals"].spec.batch_rows == 2048
        assert t["vals"][0] == [1]
        assert t["vals"][1] == [2, 3]

    def test_from_arrow_blosc2_batch_size_override_and_none(self):
        at = pa.table({"vals": pa.array([[1], [2], [3]], type=pa.list_(pa.int64()))})
        t = CTable.from_arrow(at.schema, at.to_batches(max_chunksize=1), blosc2_batch_size=2)
        assert t._schema.columns_by_name["vals"].spec.batch_rows == 2

        t2 = CTable.from_arrow(at.schema, at.to_batches(max_chunksize=1), blosc2_batch_size=None)
        assert t2._schema.columns_by_name["vals"].spec.batch_rows is None

    def test_from_arrow_invalid_blosc2_batch_size_raises(self):
        at = pa.table({"vals": pa.array([[1]], type=pa.list_(pa.int64()))})
        with pytest.raises(ValueError, match="blosc2_batch_size"):
            CTable.from_arrow(at.schema, at.to_batches(), blosc2_batch_size=0)

    def test_utf8_arrow_roundtrip_no_singleton_list(self):
        """Scalar string columns import as utf8 (not list<string>) without singleton wrapping."""
        long_str = "x" * 500
        at = pa.table({"txt": pa.array(["short", long_str, None, "end"], type=pa.string())})
        t = CTable.from_arrow(at.schema, at.to_batches())
        assert t["txt"].is_varlen_scalar
        out = t.to_arrow()
        if HAVE_STRING_DTYPE:
            assert t["txt"].is_utf8
            nv = t["txt"].null_value
            assert list(t["txt"][:]) == ["short", long_str, nv, "end"]
            # Export back to Arrow → still a scalar string column, not list<string>
            assert pa.types.is_large_string(out.schema.field("txt").type)
        else:
            assert not t["txt"].is_utf8  # vlstring fallback, native-None nulls
            assert list(t["txt"][:]) == ["short", long_str, None, "end"]
            assert pa.types.is_string(out.schema.field("txt").type)
        assert out.column("txt").to_pylist() == ["short", long_str, None, "end"]

    def test_vlbytes_arrow_roundtrip_no_singleton_list(self):
        """Scalar binary columns import as vlbytes (not list<binary>) without singleton wrapping."""
        long_bin = b"b" * 500
        at = pa.table({"bin": pa.array([b"short", long_bin, None, b"end"], type=pa.large_binary())})
        t = CTable.from_arrow(at.schema, at.to_batches())
        assert t["bin"].is_varlen_scalar
        assert list(t["bin"][:]) == [b"short", long_bin, None, b"end"]
        out = t.to_arrow()
        assert pa.types.is_large_binary(out.schema.field("bin").type)
        assert out.column("bin").to_pylist() == [b"short", long_bin, None, b"end"]

    def test_utf8_parquet_roundtrip(self, tmp_path):
        """Parquet import/export round-trips long scalar strings without singleton-list wrapping."""
        long_str = "y" * 1000
        at = pa.table(
            {
                "id": pa.array([0, 1, 2, 3], type=pa.int64()),
                "txt": pa.array(["short", long_str, None, "end"], type=pa.string()),
            }
        )
        path = tmp_path / "utf8.parquet"
        pq.write_table(at, path)

        t = CTable.from_parquet(path)
        assert t["txt"].is_varlen_scalar
        if HAVE_STRING_DTYPE:
            assert t["txt"].is_utf8
            nv = t["txt"].null_value
            assert list(t["txt"][:]) == ["short", long_str, nv, "end"]
        else:
            assert not t["txt"].is_utf8  # vlstring fallback, native-None nulls
            assert list(t["txt"][:]) == ["short", long_str, None, "end"]

        out = tmp_path / "utf8_out.parquet"
        t.to_parquet(out)
        rt = pq.read_table(out)
        if HAVE_STRING_DTYPE:
            assert pa.types.is_large_string(rt.schema.field("txt").type)
        else:
            assert pa.types.is_string(rt.schema.field("txt").type)
        assert rt.column("txt").to_pylist() == ["short", long_str, None, "end"]


# ---------------------------------------------------------------------------
# Null handling
# ---------------------------------------------------------------------------


class TestNullHandling:
    def test_nullable_list_column_roundtrip(self, tmp_path):
        @dataclass
        class NullableListRow:
            vals: list[int] = blosc2.field(  # noqa: RUF009
                blosc2.list(blosc2.int64(), nullable=True, storage="batch", serializer="msgpack")
            )

        data = [([1, 2],), (None,), ([3],)]
        t = CTable(NullableListRow, new_data=data)
        path = tmp_path / "nullable_list.parquet"
        t.to_parquet(path)
        t2 = CTable.from_parquet(path)
        assert len(t2) == 3
        assert t2["vals"][0] == [1, 2]
        assert t2["vals"][1] is None
        assert t2["vals"][2] == [3]

    def test_scalar_null_no_sentinel_raises(self, tmp_path):
        """Importing Parquet scalar nulls without a null_value sentinel fails."""
        at = pa.table({"score": pa.array([1.0, None, 3.0], type=pa.float64())})
        path = tmp_path / "nulls.parquet"
        pq.write_table(at, path)
        with pytest.raises(TypeError, match="null_value sentinel"):
            CTable.from_parquet(path, auto_null_sentinels=False)

    def test_scalar_null_exported_as_parquet_null(self, tmp_path):
        """Sentinel values become Parquet nulls on export."""

        @dataclass
        class NullRow:
            score: float = blosc2.field(blosc2.float64(null_value=float("nan")), default=float("nan"))

        t = CTable(NullRow, new_data=[(1.0,), (float("nan"),), (3.0,)])
        path = tmp_path / "null_export.parquet"
        t.to_parquet(path)
        at = pq.read_table(path)
        nulls = at["score"].is_null().to_pylist()
        assert nulls == [False, True, False]

    def test_auto_nullable_scalars_roundtrip(self, tmp_path):
        at = pa.table(
            {
                "i": pa.array([1, None, 3], type=pa.int32()),
                "f": pa.array([1.0, None, 3.0], type=pa.float64()),
                "s": pa.array(["a", None, "c"], type=pa.string()),
                "b": pa.array([b"a", None, b"c"], type=pa.large_binary()),
                "flag": pa.array([True, None, False], type=pa.bool_()),
            }
        )
        path = tmp_path / "nullable_scalars.parquet"
        pq.write_table(at, path)
        t = CTable.from_parquet(path)
        assert t["i"].null_count() == 1
        assert t["f"].null_count() == 1
        assert t["s"].null_count() == 1
        assert t["b"].null_count() == 1
        assert t["flag"].null_count() == 1
        assert t["flag"][:].tolist() == [1, 255, 0]
        out = tmp_path / "nullable_scalars_out.parquet"
        t.to_parquet(out)
        rt = pq.read_table(out)
        # "s" round-trips as utf8 -> large_string, unlike the other columns
        # (on NumPy < 2.0 it stays vlstring and exports as plain string).
        expected_schema = (
            at.schema.set(at.schema.get_field_index("s"), pa.field("s", pa.large_string()))
            if HAVE_STRING_DTYPE
            else at.schema
        )
        assert rt.schema == expected_schema
        assert rt.to_pylist() == at.to_pylist()

    def test_null_policy_controls_default_sentinels(self):
        # Null policy sentinels apply to fixed-width scalar columns (int, float, bool, string
        # with an explicit string_max_length).  vlstring / vlbytes columns represent
        # nulls natively as None and do NOT use sentinels.
        at = pa.table(
            {
                "i": pa.array([1, None, 3], type=pa.int32()),
            }
        )
        policy = blosc2.NullPolicy(signed_int_strategy="max")
        with blosc2.null_policy(policy):
            t = CTable.from_arrow(at.schema, at.to_batches())
        assert blosc2.get_null_policy() is blosc2.DEFAULT_NULL_POLICY
        assert t._schema.columns_by_name["i"].spec.null_value == np.iinfo(np.int32).max
        assert t["i"].null_count() == 1

    def test_null_policy_string_value_applies_to_fixed_width_strings(self):
        """string_value in NullPolicy applies when string_max_length is given explicitly."""
        at = pa.table(
            {
                "i": pa.array([1, None, 3], type=pa.int32()),
                "s": pa.array(["a", None, "c"], type=pa.string()),
            }
        )
        policy = blosc2.NullPolicy(signed_int_strategy="max", string_value="<NULL>")
        with blosc2.null_policy(policy):
            t = CTable.from_arrow(at.schema, at.to_batches(), string_max_length=32)
        assert t._schema.columns_by_name["i"].spec.null_value == np.iinfo(np.int32).max
        assert t._schema.columns_by_name["s"].spec.null_value == "<NULL>"
        assert t["i"].null_count() == 1
        assert t["s"].null_count() == 1

    def test_null_values_override_policy_and_auto_false(self, tmp_path):
        # column_null_values only applies to fixed-width scalar columns.
        # vlstring / vlbytes columns represent nulls as native None and do not
        # accept column_null_values overrides.
        at = pa.table(
            {
                "i": pa.array([1, None, 3], type=pa.int32()),
            }
        )
        policy = blosc2.NullPolicy(column_null_values={"i": -1})
        with blosc2.null_policy(policy):
            t = CTable.from_arrow(at.schema, at.to_batches(), auto_null_sentinels=False)
        assert t._schema.columns_by_name["i"].spec.null_value == -1
        assert t["i"].null_count() == 1

        path = tmp_path / "null_values.parquet"
        pq.write_table(at, path)
        with blosc2.null_policy(policy):
            t2 = CTable.from_parquet(path, auto_null_sentinels=False)
        assert t2._schema.columns_by_name["i"].spec.null_value == -1

    def test_null_policy_rejects_vlbytes_column_null_values(self):
        """Passing column_null_values for a vlbytes column raises TypeError."""
        at = pa.table({"b": pa.array([b"a", None, b"c"], type=pa.large_binary())})
        policy = blosc2.NullPolicy(column_null_values={"b": b"NA"})
        with blosc2.null_policy(policy), pytest.raises(TypeError, match="vlbytes"):
            CTable.from_arrow(at.schema, at.to_batches())

    def test_null_policy_column_null_values_applies_to_utf8(self):
        """Passing column_null_values for a utf8 (scalar string) column sets its sentinel.

        On NumPy < 2.0 utf8 columns are unavailable, strings import as
        vlstring, and the override is rejected like any varlen column.
        """
        at = pa.table({"s": pa.array(["a", None, "c"], type=pa.string())})
        policy = blosc2.NullPolicy(column_null_values={"s": "NA"})
        if not HAVE_STRING_DTYPE:
            with blosc2.null_policy(policy), pytest.raises(TypeError, match="native None"):
                CTable.from_arrow(at.schema, at.to_batches())
            return
        with blosc2.null_policy(policy):
            t = CTable.from_arrow(at.schema, at.to_batches())
        assert t["s"].null_value == "NA"
        assert list(t["s"][:]) == ["a", "NA", "c"]

    def test_null_policy_unknown_column_raises(self):
        at = pa.table({"i": pa.array([1, None], type=pa.int32())})
        policy = blosc2.NullPolicy(column_null_values={"missing": -1})
        with blosc2.null_policy(policy), pytest.raises(KeyError, match="unknown columns"):
            CTable.from_arrow(at.schema, at.to_batches())

    def test_null_policy_rejects_list_columns(self):
        at = pa.table({"vals": pa.array([[1], None], type=pa.list_(pa.int64()))})
        policy = blosc2.NullPolicy(column_null_values={"vals": []})
        with blosc2.null_policy(policy), pytest.raises(TypeError, match="only supports scalar columns"):
            CTable.from_arrow(at.schema, at.to_batches())

    def test_nullable_bool_filter_semantics(self, tmp_path):
        at = pa.table({"flag": pa.array([True, None, False], type=pa.bool_())})
        path = tmp_path / "nullable_bool.parquet"
        pq.write_table(at, path)
        t = CTable.from_parquet(path)
        assert t.where(t.flag).flag[:].tolist() == [1]
        assert t.where(~t.flag).flag[:].tolist() == [0]
        assert t.where(t.flag == True).flag[:].tolist() == [1]  # noqa: E712
        assert t.where(t.flag == False).flag[:].tolist() == [0]  # noqa: E712
        assert t.where(t.flag != True).flag[:].tolist() == [0]  # noqa: E712
        assert t.where(t.flag != False).flag[:].tolist() == [1]  # noqa: E712


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrors:
    def test_missing_pyarrow_to_parquet(self, tmp_path, monkeypatch):
        """to_parquet raises ImportError when pyarrow is not available."""
        import sys

        monkeypatch.setitem(sys.modules, "pyarrow.parquet", None)
        t = CTable(Row, new_data=DATA10)
        with pytest.raises(ImportError, match="pyarrow"):
            t.to_parquet(tmp_path / "x.parquet")

    def test_missing_pyarrow_from_parquet(self, tmp_path, monkeypatch):
        """from_parquet raises ImportError when pyarrow is not available."""
        import sys

        t = CTable(Row, new_data=DATA10)
        path = tmp_path / "x.parquet"
        t.to_parquet(path)
        monkeypatch.setitem(sys.modules, "pyarrow.parquet", None)
        with pytest.raises(ImportError, match="pyarrow"):
            CTable.from_parquet(path)

    def test_unsupported_arrow_type(self, tmp_path):
        at = pa.table({"duration": pa.array([1, 2, 3], type=pa.duration("s"))})
        path = tmp_path / "duration.parquet"
        pq.write_table(at, path)
        with pytest.raises(TypeError, match="No blosc2 spec"):
            CTable.from_parquet(path)

    def test_invalid_batch_size_to_parquet(self, tmp_path):
        t = CTable(Row, new_data=DATA10)
        with pytest.raises(ValueError, match="batch_size"):
            t.to_parquet(tmp_path / "x.parquet", batch_size=0)

    def test_invalid_batch_size_from_parquet(self, tmp_path):
        t = CTable(Row, new_data=DATA10)
        path = tmp_path / "x.parquet"
        t.to_parquet(path)
        with pytest.raises(ValueError, match="batch_size"):
            CTable.from_parquet(path, batch_size=0)

    def test_invalid_max_rows_from_parquet(self, tmp_path):
        t = CTable(Row, new_data=DATA10)
        path = tmp_path / "x.parquet"
        t.to_parquet(path)
        with pytest.raises(ValueError, match="max_rows"):
            CTable.from_parquet(path, max_rows=-1)

    def test_max_rows_from_parquet_limits_rows(self, tmp_path):
        t = CTable(Row, new_data=DATA10)
        path = tmp_path / "x.parquet"
        t.to_parquet(path)
        out = CTable.from_parquet(path, batch_size=4, max_rows=6)
        assert len(out) == 6
        np.testing.assert_array_equal(out["id"][:], np.arange(6))

    def test_max_rows_zero_from_parquet_imports_empty_table(self, tmp_path):
        t = CTable(Row, new_data=DATA10)
        path = tmp_path / "x.parquet"
        t.to_parquet(path)
        out = CTable.from_parquet(path, max_rows=0)
        assert len(out) == 0
        assert out.col_names == ["id", "score", "active", "label"]

    def test_string_truncation_error(self, tmp_path):
        """Importing longer strings than max_length raises ValueError."""
        at = pa.table({"name": pa.array(["a" * 300, "b"], type=pa.string())})
        path = tmp_path / "long_str.parquet"
        pq.write_table(at, path)
        # Explicit small max_length should raise on import
        with pytest.raises(ValueError, match="max_length"):
            CTable.from_parquet(path, string_max_length=10)


def test_parquet_cli_progress_is_opt_in(tmp_path, capsys):
    from blosc2.cli.parquet_to_blosc2 import main

    path = tmp_path / "progress.parquet"
    out = tmp_path / "progress.b2d"
    pq.write_table(pa.table({"x": pa.array([1, 2, 3], type=pa.int64())}), path)

    assert main(["--parquet-batch-size", "1", str(path), str(out)]) == 0
    captured = capsys.readouterr()
    assert "  batch" not in captured.out

    out_progress = tmp_path / "progress_enabled.b2d"
    assert main(["--progress", "--parquet-batch-size", "1", str(path), str(out_progress)]) == 0
    captured = capsys.readouterr()
    assert "  batch" in captured.out


def test_parquet_cli_nested_progress_skips_write_lines(tmp_path, capsys):
    from blosc2.cli.parquet_to_blosc2 import main

    buf, _ = _make_taxi_parquet_buf(n_outer_rows=3)
    path = tmp_path / "taxi.parquet"
    out = tmp_path / "taxi.b2d"
    path.write_bytes(buf.getvalue())

    assert (
        main(
            [
                "--progress",
                "--parquet-batch-size",
                "1",
                "--blosc2-batch-size",
                "1",
                str(path),
                str(out),
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert "  parquet batch" in captured.out
    assert "    write" not in captured.out


def test_parquet_cli_separate_nested_flattens_top_level_structs(tmp_path, capsys):
    from blosc2.cli.parquet_to_blosc2 import main

    trip_type = pa.struct(
        [
            pa.field("sec", pa.float32()),
            pa.field("begin", pa.struct([pa.field("lon", pa.float64()), pa.field("lat", pa.float64())])),
        ]
    )
    path = tmp_path / "struct.parquet"
    out = tmp_path / "struct.b2d"
    table = pa.table(
        {
            "trip": pa.array(
                [
                    {"sec": 10.0, "begin": {"lon": -87.6, "lat": 41.8}},
                    {"sec": 20.0, "begin": {"lon": -87.7, "lat": 41.9}},
                ],
                type=trip_type,
            ),
            "fare": pa.array([15.0, 25.0], type=pa.float32()),
        }
    )
    pq.write_table(table, path)

    assert main([str(path), str(out)]) == 0
    captured = capsys.readouterr()
    assert "Struct→columns:      1" in captured.out

    ct = CTable.open(str(out), mode="r")
    assert ct.col_names == ["trip.sec", "trip.begin.lon", "trip.begin.lat", "fare"]
    np.testing.assert_allclose(ct["trip.begin.lon"][:], [-87.6, -87.7])
    ct.close()


def test_parquet_cli_no_separate_nested_preserves_top_level_struct_as_list(tmp_path):
    from blosc2.cli.parquet_to_blosc2 import main

    trip_type = pa.struct([pa.field("sec", pa.float32())])
    path = tmp_path / "struct.parquet"
    out = tmp_path / "struct.b2d"
    pq.write_table(
        pa.table({"trip": pa.array([{"sec": 10.0}, {"sec": 20.0}], type=trip_type)}),
        path,
    )

    assert main(["--no-separate-nested-cols", str(path), str(out)]) == 0

    ct = CTable.open(str(out), mode="r")
    assert ct.col_names == ["trip"]
    assert ct["trip"][:] == [[{"sec": 10.0}], [{"sec": 20.0}]]
    ct.close()


def test_parquet_cli_timestamp_unit_auto(tmp_path):
    from blosc2.cli.parquet_to_blosc2 import main

    values = np.array(
        ["2025-01-01T00:00:00", "2025-01-01T00:00:01", "2025-01-01T00:00:02"],
        dtype="datetime64[us]",
    )
    path = tmp_path / "timestamps.parquet"
    out = tmp_path / "timestamps.b2d"
    pq.write_table(pa.table({"ts": pa.array(values, type=pa.timestamp("us"))}), path)

    assert main(["--timestamp-unit", "auto", str(path), str(out)]) == 0

    table = CTable.open(str(out), mode="r")
    assert table._schema.columns_by_name["ts"].spec.unit == "s"
    np.testing.assert_array_equal(
        table["ts"][:],
        np.array(
            ["2025-01-01T00:00:00", "2025-01-01T00:00:01", "2025-01-01T00:00:02"], dtype="datetime64[s]"
        ),
    )
    assert table._cols["ts"][:].tolist() == [1735689600, 1735689601, 1735689602]


# ---------------------------------------------------------------------------
# separate_nested_cols / unnamed-root list<struct<...>> import
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Shared schema / data helpers
# ---------------------------------------------------------------------------


def _make_taxi_schema():
    """Return a simplified taxi-like Arrow schema (inner struct fields)."""
    trip_type = pa.struct(
        [
            pa.field("sec", pa.float32()),
            pa.field(
                "begin",
                pa.struct([pa.field("lon", pa.float64()), pa.field("lat", pa.float64())]),
            ),
        ]
    )
    payment_type = pa.struct(
        [
            pa.field("fare", pa.float64()),
            pa.field("tips", pa.float64()),
        ]
    )
    return pa.struct(
        [
            pa.field("trip", trip_type),
            pa.field("payment", payment_type),
            pa.field("company", pa.string()),
        ]
    )


def _make_taxi_parquet_buf(n_outer_rows=2):
    """Create an in-memory Parquet buffer with an unnamed root list<struct<...>>.

    *n_outer_rows* controls how many Parquet rows (outer lists) to create.
    Each outer list contains 1–3 trip records.
    """
    root_struct = _make_taxi_schema()
    root_list = pa.list_(root_struct)

    all_rows = [
        [
            {
                "trip": {"sec": 10.0, "begin": {"lon": -87.6, "lat": 41.8}},
                "payment": {"fare": 15.0, "tips": 2.0},
                "company": "Taxi Corp",
            },
            {
                "trip": {"sec": 20.0, "begin": {"lon": -87.7, "lat": 41.9}},
                "payment": {"fare": 25.0, "tips": 3.0},
                "company": "Blue Cab",
            },
        ],
        [
            {
                "trip": {"sec": 5.0, "begin": {"lon": -87.5, "lat": 41.7}},
                "payment": {"fare": 10.0, "tips": 1.0},
                "company": "Taxi Corp",
            },
        ],
        [
            {
                "trip": {"sec": 30.0, "begin": {"lon": -87.3, "lat": 41.6}},
                "payment": {"fare": 5.0, "tips": 0.5},
                "company": "City Cab",
            },
            {
                "trip": {"sec": 15.0, "begin": {"lon": -87.4, "lat": 41.5}},
                "payment": {"fare": 12.0, "tips": 1.5},
                "company": "Blue Cab",
            },
            {
                "trip": {"sec": 8.0, "begin": {"lon": -87.2, "lat": 41.4}},
                "payment": {"fare": 9.0, "tips": 0.0},
                "company": "Taxi Corp",
            },
        ],
    ]
    rows = all_rows[:n_outer_rows]
    arr = pa.array(rows, type=root_list)
    buf = io.BytesIO()
    pq.write_table(pa.table({"": arr}), buf)
    buf.seek(0)
    return buf, rows


def _count_elements(rows):
    """Count the total number of list elements across outer rows."""
    return sum(len(r) for r in rows)


# ---------------------------------------------------------------------------
# Detection helper tests
# ---------------------------------------------------------------------------


class TestDetectUnnamedRootListStruct:
    def test_detects_single_unnamed_list_struct(self):
        root_struct = _make_taxi_schema()
        schema = pa.schema([pa.field("", pa.list_(root_struct))])
        assert CTable._detect_unnamed_root_list_struct(pa, schema) is True

    def test_detects_large_list_variant(self):
        root_struct = _make_taxi_schema()
        schema = pa.schema([pa.field("", pa.large_list(root_struct))])
        assert CTable._detect_unnamed_root_list_struct(pa, schema) is True

    def test_rejects_named_field(self):
        root_struct = _make_taxi_schema()
        schema = pa.schema([pa.field("events", pa.list_(root_struct))])
        assert CTable._detect_unnamed_root_list_struct(pa, schema) is False

    def test_rejects_multiple_fields(self):
        root_struct = _make_taxi_schema()
        schema = pa.schema([pa.field("", pa.list_(root_struct)), pa.field("id", pa.int64())])
        assert CTable._detect_unnamed_root_list_struct(pa, schema) is False

    def test_rejects_non_list_unnamed_field(self):
        root_struct = _make_taxi_schema()
        schema = pa.schema([pa.field("", root_struct)])
        assert CTable._detect_unnamed_root_list_struct(pa, schema) is False

    def test_rejects_list_of_scalar(self):
        schema = pa.schema([pa.field("", pa.list_(pa.int64()))])
        assert CTable._detect_unnamed_root_list_struct(pa, schema) is False


# ---------------------------------------------------------------------------
# Phase 1 acceptance tests
# ---------------------------------------------------------------------------


class TestUnnamedRootImport:
    """Acceptance tests for Phase 1: unnamed-root list<struct<...>> import."""

    def _make_ct(self, n_outer_rows=2, **kwargs):
        buf, rows = _make_taxi_parquet_buf(n_outer_rows)
        ct = CTable.from_parquet(buf, separate_nested_cols=True, **kwargs)
        return ct, rows

    # ------------------------------------------------------------------
    # Row count
    # ------------------------------------------------------------------

    def test_nrows_equals_element_count_2_outer(self):
        ct, rows = self._make_ct(n_outer_rows=2)
        assert len(ct) == _count_elements(rows)  # 3

    def test_from_parquet_separates_nested_cols_by_default(self):
        buf, rows = _make_taxi_parquet_buf(n_outer_rows=2)
        ct = CTable.from_parquet(buf)
        assert len(ct) == _count_elements(rows)
        assert "column_0" not in ct.col_names
        assert "trip.begin.lon" in ct.col_names

    def test_nrows_equals_element_count_3_outer(self):
        ct, rows = self._make_ct(n_outer_rows=3)
        assert len(ct) == _count_elements(rows)  # 6

    def test_max_rows_limits_flattened_element_rows(self):
        ct, rows = self._make_ct(n_outer_rows=3, max_rows=4, batch_size=1)
        expected = [r["payment"]["fare"] for outer in rows for r in outer][:4]
        assert len(ct) == 4
        np.testing.assert_allclose(ct["payment.fare"][:].tolist(), expected)

    def test_max_rows_zero_imports_empty_flattened_table(self):
        ct, _ = self._make_ct(n_outer_rows=3, max_rows=0)
        assert len(ct) == 0
        assert "column_0" not in ct.col_names
        assert "trip.begin.lon" in ct.col_names

    # ------------------------------------------------------------------
    # Column names — no column_0, no unnamed root in col_names
    # ------------------------------------------------------------------

    def test_col_names_no_column_0(self):
        ct, _ = self._make_ct()
        assert "column_0" not in ct.col_names
        assert "" not in ct.col_names

    def test_col_names_contains_leaf_paths(self):
        ct, _ = self._make_ct()
        expected = {
            "trip.sec",
            "trip.begin.lon",
            "trip.begin.lat",
            "payment.fare",
            "payment.tips",
            "company",
        }
        assert set(ct.col_names) == expected

    # ------------------------------------------------------------------
    # Column access and analytics
    # ------------------------------------------------------------------

    def test_payment_fare_mean(self):
        ct, rows = self._make_ct(n_outer_rows=2)
        fares = [r["payment"]["fare"] for outer in rows for r in outer]
        expected = np.mean(fares)
        np.testing.assert_allclose(ct["payment.fare"].mean(), expected)

    def test_trip_begin_lon_mean(self):
        ct, rows = self._make_ct(n_outer_rows=2)
        lons = [r["trip"]["begin"]["lon"] for outer in rows for r in outer]
        expected = np.mean(lons)
        np.testing.assert_allclose(ct["trip.begin.lon"].mean(), expected)

    def test_payment_fare_values(self):
        ct, rows = self._make_ct(n_outer_rows=2)
        expected = [r["payment"]["fare"] for outer in rows for r in outer]
        np.testing.assert_allclose(ct["payment.fare"][:].tolist(), expected)

    def test_company_column_values(self):
        ct, rows = self._make_ct(n_outer_rows=2)
        expected = [r["company"] for outer in rows for r in outer]
        assert list(ct["company"][:]) == expected

    # ------------------------------------------------------------------
    # where() filtering
    # ------------------------------------------------------------------

    def test_where_payment_fare_gt_12(self):
        ct, rows = self._make_ct(n_outer_rows=2)
        all_fares = [r["payment"]["fare"] for outer in rows for r in outer]
        expected_count = sum(1 for f in all_fares if f > 12)
        result = ct.where("payment.fare > 12")
        assert len(result) == expected_count

    def test_where_payment_fare_gt_20(self):
        ct, rows = self._make_ct(n_outer_rows=2)
        all_fares = [r["payment"]["fare"] for outer in rows for r in outer]
        expected_count = sum(1 for f in all_fares if f > 20)
        result = ct.where("payment.fare > 20")
        assert len(result) == expected_count

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Persistence: .b2d reopen
    # ------------------------------------------------------------------

    def test_b2d_reopen_nrows(self, tmp_path):
        buf, rows = _make_taxi_parquet_buf(n_outer_rows=2)
        ct = CTable.from_parquet(buf, separate_nested_cols=True, urlpath=str(tmp_path / "taxi.b2d"))
        ct.close()
        ct2 = CTable.open(str(tmp_path / "taxi.b2d"), mode="r")
        assert len(ct2) == _count_elements(rows)
        ct2.close()

    def test_b2d_reopen_col_names(self, tmp_path):
        buf, _ = _make_taxi_parquet_buf(n_outer_rows=2)
        ct = CTable.from_parquet(buf, separate_nested_cols=True, urlpath=str(tmp_path / "taxi.b2d"))
        col_names = ct.col_names
        ct.close()
        ct2 = CTable.open(str(tmp_path / "taxi.b2d"), mode="r")
        assert ct2.col_names == col_names
        ct2.close()

    def test_b2d_reopen_values(self, tmp_path):
        buf, rows = _make_taxi_parquet_buf(n_outer_rows=2)
        ct = CTable.from_parquet(buf, separate_nested_cols=True, urlpath=str(tmp_path / "taxi.b2d"))
        expected_fares = [r["payment"]["fare"] for outer in rows for r in outer]
        ct.close()
        ct2 = CTable.open(str(tmp_path / "taxi.b2d"), mode="r")
        np.testing.assert_allclose(ct2["payment.fare"][:].tolist(), expected_fares)
        ct2.close()

    def test_b2z_reopen(self, tmp_path):
        buf, rows = _make_taxi_parquet_buf(n_outer_rows=2)
        ct = CTable.from_parquet(buf, separate_nested_cols=True, urlpath=str(tmp_path / "taxi.b2z"))
        ct.close()
        ct2 = CTable.open(str(tmp_path / "taxi.b2z"), mode="r")
        assert len(ct2) == _count_elements(rows)
        assert "trip.begin.lon" in ct2.col_names
        ct2.close()

    # ------------------------------------------------------------------
    # to_arrow() emits clean logical nested table
    # ------------------------------------------------------------------

    def test_to_arrow_no_unnamed_column(self):
        ct, _ = self._make_ct()
        arrow_table = ct.to_arrow()
        assert "" not in arrow_table.schema.names
        assert "column_0" not in arrow_table.schema.names

    def test_to_arrow_has_trip_and_payment_top_level(self):
        ct, _ = self._make_ct()
        arrow_table = ct.to_arrow()
        names = arrow_table.schema.names
        assert "trip" in names
        assert "payment" in names
        assert "company" in names

    def test_to_arrow_trip_is_struct(self):
        ct, _ = self._make_ct()
        arrow_table = ct.to_arrow()
        assert pa.types.is_struct(arrow_table.schema.field("trip").type)

    def test_to_arrow_payment_fare_values(self):
        ct, rows = self._make_ct(n_outer_rows=2)
        arrow_table = ct.to_arrow()
        expected = [r["payment"]["fare"] for outer in rows for r in outer]
        payment_col = arrow_table.column("payment")
        actual = [row.as_py()["fare"] for row in payment_col]
        np.testing.assert_allclose(actual, expected)

    # ------------------------------------------------------------------
    # from_arrow with separate_nested_cols=True
    # ------------------------------------------------------------------

    def test_from_arrow_separate_nested_cols(self):
        """from_arrow accepts separate_nested_cols=True directly."""
        root_struct = _make_taxi_schema()
        root_list = pa.list_(root_struct)
        data = [
            [
                {
                    "trip": {"sec": 10.0, "begin": {"lon": -87.6, "lat": 41.8}},
                    "payment": {"fare": 15.0, "tips": 2.0},
                    "company": "Taxi",
                },
                {
                    "trip": {"sec": 5.0, "begin": {"lon": -87.5, "lat": 41.7}},
                    "payment": {"fare": 10.0, "tips": 1.0},
                    "company": "Taxi",
                },
            ]
        ]
        arr = pa.array(data, type=root_list)
        schema = pa.schema([pa.field("", root_list)])
        batch = pa.record_batch([arr], schema=schema)
        ct = CTable.from_arrow(schema, [batch], separate_nested_cols=True)
        assert len(ct) == 2
        assert "trip.begin.lon" in ct.col_names
        assert "payment.fare" in ct.col_names
        np.testing.assert_allclose(ct["payment.fare"][:].tolist(), [15.0, 10.0])

    # ------------------------------------------------------------------
    # Behaviour when separate_nested_cols=False (existing behaviour)
    # ------------------------------------------------------------------

    def test_false_flag_gives_renamed_root_column(self):
        """Without separate_nested_cols, the old renaming behaviour applies."""
        buf, _ = _make_taxi_parquet_buf(n_outer_rows=2)
        ct = CTable.from_parquet(buf, separate_nested_cols=False)
        # The unnamed "" field should be renamed to "root"
        assert "root" in ct.col_names

    def test_false_flag_nrows_equals_parquet_rows(self):
        """Without separate_nested_cols, nrows is the number of Parquet outer rows."""
        buf, rows = _make_taxi_parquet_buf(n_outer_rows=2)
        ct = CTable.from_parquet(buf, separate_nested_cols=False)
        # 2 Parquet rows, not 3 elements
        assert len(ct) == len(rows)

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_empty_outer_list(self):
        """Importing a Parquet file where all outer lists are empty gives 0 rows."""
        root_struct = _make_taxi_schema()
        root_list = pa.list_(root_struct)
        arr = pa.array([[], []], type=root_list)
        buf = io.BytesIO()
        pq.write_table(pa.table({"": arr}), buf)
        buf.seek(0)
        ct = CTable.from_parquet(buf, separate_nested_cols=True)
        assert len(ct) == 0
        assert set(ct.col_names) == {
            "trip.sec",
            "trip.begin.lon",
            "trip.begin.lat",
            "payment.fare",
            "payment.tips",
            "company",
        }

    def test_single_element(self):
        """A single-element list imports as one CTable row."""
        root_struct = _make_taxi_schema()
        root_list = pa.list_(root_struct)
        arr = pa.array(
            [
                [
                    {
                        "trip": {"sec": 7.0, "begin": {"lon": -87.0, "lat": 41.0}},
                        "payment": {"fare": 8.0, "tips": 0.5},
                        "company": "X",
                    }
                ]
            ],
            type=root_list,
        )
        buf = io.BytesIO()
        pq.write_table(pa.table({"": arr}), buf)
        buf.seek(0)
        ct = CTable.from_parquet(buf, separate_nested_cols=True)
        assert len(ct) == 1
        assert ct["payment.fare"][0] == 8.0

    def test_non_qualifying_schema_ignored_with_flag(self):
        """separate_nested_cols=True is silently ignored for a normal (non-qualifying) schema."""
        at = pa.table({"x": pa.array([1, 2, 3], type=pa.int64()), "y": pa.array([4.0, 5.0, 6.0])})
        buf = io.BytesIO()
        pq.write_table(at, buf)
        buf.seek(0)
        ct = CTable.from_parquet(buf, separate_nested_cols=True)
        assert len(ct) == 3
        assert ct.col_names == ["x", "y"]

    def test_multiple_batches(self):
        """separate_nested_cols works when Parquet is read in small batches."""
        buf, rows = _make_taxi_parquet_buf(n_outer_rows=3)
        ct = CTable.from_parquet(buf, separate_nested_cols=True, batch_size=1)
        assert len(ct) == _count_elements(rows)
        fares = [r["payment"]["fare"] for outer in rows for r in outer]
        np.testing.assert_allclose(ct["payment.fare"][:].tolist(), fares)

    def test_nested_list_inside_element_ignored_at_phase1(self):
        """A nested list inside the element struct is imported as a ListArray column (phase 1)."""
        path_type = pa.struct([pa.field("londiff", pa.float32()), pa.field("latdiff", pa.float32())])
        trip_with_path = pa.struct(
            [
                pa.field("sec", pa.float32()),
                pa.field("path", pa.list_(path_type)),
            ]
        )
        root_struct = pa.struct([pa.field("trip", trip_with_path), pa.field("fare", pa.float64())])
        root_list = pa.list_(root_struct)
        data = [
            [
                {"trip": {"sec": 10.0, "path": [{"londiff": 0.1, "latdiff": 0.2}]}, "fare": 15.0},
                {"trip": {"sec": 5.0, "path": []}, "fare": 8.0},
            ]
        ]
        arr = pa.array(data, type=root_list)
        buf = io.BytesIO()
        pq.write_table(pa.table({"": arr}), buf)
        buf.seek(0)
        ct = CTable.from_parquet(buf, separate_nested_cols=True)
        assert len(ct) == 2
        assert "fare" in ct.col_names
        assert ct["fare"][:].tolist() == [15.0, 8.0]
        # trip.path should be a ListArray column with one list per element row
        assert "trip.path" in ct.col_names
        assert ct["trip.path"].is_list
        assert ct["trip.path"][0] == [{"londiff": pytest.approx(0.1), "latdiff": pytest.approx(0.2)}]
        assert ct["trip.path"][1] == []


if __name__ == "__main__":
    pytest.main(["-v", __file__])
