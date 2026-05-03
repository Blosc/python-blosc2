#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""Tests for CTable.to_parquet(), from_parquet(), iter_arrow_batches(),
and from_arrow()."""

from dataclasses import dataclass

import numpy as np
import pytest

import blosc2
from blosc2 import CTable

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

    def test_from_arrow_list_batch_rows_default(self):
        at = pa.table({"vals": pa.array([[1], [2, 3]], type=pa.list_(pa.int64()))})
        t = CTable.from_arrow(at.schema, at.to_batches())
        assert t._schema.columns_by_name["vals"].spec.batch_rows == 2048
        assert t["vals"][0] == [1]
        assert t["vals"][1] == [2, 3]

    def test_from_arrow_list_batch_rows_override_and_none(self):
        at = pa.table({"vals": pa.array([[1], [2], [3]], type=pa.list_(pa.int64()))})
        t = CTable.from_arrow(at.schema, at.to_batches(max_chunksize=1), list_batch_rows=2)
        assert t._schema.columns_by_name["vals"].spec.batch_rows == 2

        t2 = CTable.from_arrow(at.schema, at.to_batches(max_chunksize=1), list_batch_rows=None)
        assert t2._schema.columns_by_name["vals"].spec.batch_rows is None

    def test_from_arrow_invalid_list_batch_rows_raises(self):
        at = pa.table({"vals": pa.array([[1]], type=pa.list_(pa.int64()))})
        with pytest.raises(ValueError, match="list_batch_rows"):
            CTable.from_arrow(at.schema, at.to_batches(), list_batch_rows=0)

    def test_vlstring_arrow_roundtrip_no_singleton_list(self):
        """Scalar string columns import as vlstring (not list<string>) without singleton wrapping."""
        long_str = "x" * 500
        at = pa.table({"txt": pa.array(["short", long_str, None, "end"], type=pa.string())})
        t = CTable.from_arrow(at.schema, at.to_batches())
        assert t["txt"].is_varlen_scalar
        assert list(t["txt"][:]) == ["short", long_str, None, "end"]
        # Export back to Arrow → still a scalar string column, not list<string>
        out = t.to_arrow()
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

    def test_vlstring_parquet_roundtrip(self, tmp_path):
        """Parquet import/export round-trips long scalar strings without singleton-list wrapping."""
        long_str = "y" * 1000
        at = pa.table(
            {
                "id": pa.array([0, 1, 2, 3], type=pa.int64()),
                "txt": pa.array(["short", long_str, None, "end"], type=pa.string()),
            }
        )
        path = tmp_path / "vlstring.parquet"
        pq.write_table(at, path)

        t = CTable.from_parquet(path)
        assert t["txt"].is_varlen_scalar
        assert list(t["txt"][:]) == ["short", long_str, None, "end"]

        out = tmp_path / "vlstring_out.parquet"
        t.to_parquet(out)
        rt = pq.read_table(out)
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
        assert rt.schema == at.schema
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

    def test_null_policy_rejects_vlstring_column_null_values(self):
        """Passing column_null_values for a vlstring column raises TypeError."""
        at = pa.table({"s": pa.array(["a", None, "c"], type=pa.string())})
        policy = blosc2.NullPolicy(column_null_values={"s": "NA"})
        with blosc2.null_policy(policy), pytest.raises(TypeError, match="vlstring"):
            CTable.from_arrow(at.schema, at.to_batches())

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
        at = pa.table({"ts": pa.array([1, 2, 3], type=pa.timestamp("s"))})
        path = tmp_path / "ts.parquet"
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

    def test_string_truncation_error(self, tmp_path):
        """Importing longer strings than max_length raises ValueError."""
        at = pa.table({"name": pa.array(["a" * 300, "b"], type=pa.string())})
        path = tmp_path / "long_str.parquet"
        pq.write_table(at, path)
        # Explicit small max_length should raise on import
        with pytest.raises(ValueError, match="max_length"):
            CTable.from_parquet(path, string_max_length=10)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
