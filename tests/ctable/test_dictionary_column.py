#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################
"""Tests for the CTable dictionary column type."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

import blosc2
from blosc2 import CTable, DictionarySpec
from blosc2.dictionary_column import DictionaryColumn
from blosc2.schema_compiler import compile_schema, schema_from_dict, schema_to_dict

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")


# ---------------------------------------------------------------------------
# Unit tests: DictionarySpec and schema compiler
# ---------------------------------------------------------------------------


class TestDictionarySpec:
    def test_default_construction(self):
        spec = blosc2.dictionary()
        assert spec.ordered is False
        assert spec.nullable is True
        assert spec.null_code == -1

    def test_wrong_index_type_raises(self):
        with pytest.raises(TypeError, match="int32"):
            blosc2.dictionary(index_type=blosc2.int64())

    def test_wrong_value_type_raises(self):
        with pytest.raises(TypeError, match="vlstring"):
            blosc2.dictionary(value_type=blosc2.string(max_length=32))

    def test_metadata_roundtrip(self):
        spec = blosc2.dictionary(ordered=True, nullable=False)
        d = spec.to_metadata_dict()
        assert d["kind"] == "dictionary"
        assert d["ordered"] is True
        assert d["nullable"] is False
        assert d["null_code"] == -1

    def test_schema_serialization_roundtrip(self):
        @dataclass
        class Row:
            vendor: str = blosc2.field(blosc2.dictionary())
            fare: float = blosc2.field(blosc2.float64())

        schema = compile_schema(Row)
        d = schema_to_dict(schema)
        schema2 = schema_from_dict(d)
        col = schema2.columns_by_name["vendor"]
        assert isinstance(col.spec, DictionarySpec)
        assert col.spec.ordered is False
        assert col.spec.nullable is True

    def test_dataclass_annotation_must_be_str(self):
        from blosc2.schema_compiler import validate_annotation_matches_spec

        spec = blosc2.dictionary()
        with pytest.raises(TypeError, match="str"):
            validate_annotation_matches_spec("x", int, spec)

    def test_dataclass_annotation_str_ok(self):
        from blosc2.schema_compiler import validate_annotation_matches_spec

        spec = blosc2.dictionary()
        validate_annotation_matches_spec("x", str, spec)  # should not raise


# ---------------------------------------------------------------------------
# CTable behavior tests
# ---------------------------------------------------------------------------


@dataclass
class TripRow:
    vendor: str = blosc2.field(blosc2.dictionary())
    fare: float = blosc2.field(blosc2.float64())


DATA = [
    {"vendor": "Uber", "fare": 10.5},
    {"vendor": "Lyft", "fare": 7.2},
    {"vendor": "Uber", "fare": 15.0},
    {"vendor": "Via", "fare": 5.0},
]

# Tuple form for extend()
DATA_TUPLES = [
    ("Uber", 10.5),
    ("Lyft", 7.2),
    ("Uber", 15.0),
    ("Via", 5.0),
]


def _logical_mask_values(ct, mask):
    """Materialize a physical predicate as logical/live-row values."""
    arr = mask.compute() if isinstance(mask, blosc2.LazyExpr) else mask
    arr = arr[:] if isinstance(arr, blosc2.NDArray) else arr
    return arr[ct._valid_rows[:]].tolist()


class TestCTableBehavior:
    def test_append_and_read(self):
        ct = CTable(TripRow)
        for row in DATA:
            ct.append(row)
        assert ct.nrows == 4
        assert ct["vendor"][:] == ["Uber", "Lyft", "Uber", "Via"]
        assert ct["vendor"][0] == "Uber"
        assert ct["vendor"][1] == "Lyft"

    def test_repeated_strings_reuse_codes(self):
        ct = CTable(TripRow)
        for row in DATA:
            ct.append(row)
        codes = ct._cols["vendor"].codes[:4].tolist()
        assert codes[0] == codes[2]  # "Uber" appears twice with same code
        assert len(ct._cols["vendor"].dictionary) == 3  # Uber, Lyft, Via

    def test_null_slot(self):
        ct = CTable(TripRow)
        ct.append({"vendor": None, "fare": 0.0})
        assert ct["vendor"][0] is None
        assert ct._cols["vendor"].codes[0] == -1

    def test_nullable_false_rejects_null(self):
        @dataclass
        class NNRow:
            vendor: str = blosc2.field(blosc2.dictionary(nullable=False))
            fare: float = blosc2.field(blosc2.float64())

        ct = CTable(NNRow)
        with pytest.raises((ValueError, TypeError)):
            ct.append({"vendor": None, "fare": 0.0})

    def test_invalid_type_raises(self):
        ct = CTable(TripRow)
        with pytest.raises((TypeError, ValueError)):
            ct.append({"vendor": 42, "fare": 0.0})

    def test_extend_batch(self):
        ct = CTable(TripRow)
        ct.extend(DATA_TUPLES)
        assert ct.nrows == 4
        assert ct["vendor"][:] == ["Uber", "Lyft", "Uber", "Via"]

    def test_codes_and_dictionary_properties(self):
        ct = CTable(TripRow)
        ct.extend(DATA_TUPLES)
        dc = ct._cols["vendor"]
        assert isinstance(dc, DictionaryColumn)
        assert list(dc.dictionary) == ["Uber", "Lyft", "Via"]
        codes = dc.codes[:4].tolist()
        assert codes == [0, 1, 0, 2]

    def test_equality_filter(self):
        ct = CTable(TripRow)
        ct.extend(DATA_TUPLES)
        mask = ct["vendor"] == "Uber"
        assert _logical_mask_values(ct, mask) == [True, False, True, False]

    def test_equality_absent_value_returns_false(self):
        ct = CTable(TripRow)
        ct.extend(DATA_TUPLES)
        mask = ct["vendor"] == "Waymo"
        assert _logical_mask_values(ct, mask) == [False, False, False, False]

    def test_equality_none(self):
        ct = CTable(TripRow)
        ct.extend(DATA_TUPLES)
        ct.append({"vendor": None, "fare": 0.0})
        mask = ct["vendor"] == None  # noqa: E711
        assert _logical_mask_values(ct, mask) == [False, False, False, False, True]

    def test_string_where_expression_on_dictionary_column(self):
        # A string where() referencing a dictionary column must resolve the
        # literal to its code instead of failing with "Unknown symbol".
        ct = CTable(TripRow)
        ct.extend(DATA_TUPLES)

        assert ct.where('vendor == "Uber"')["fare"][:].tolist() == [10.5, 15.0]
        assert ct.where('vendor != "Uber"')["fare"][:].tolist() == [7.2, 5.0]
        # Combined with a regular-column predicate.
        assert ct.where('vendor == "Uber" and fare > 12')["fare"][:].tolist() == [15.0]
        # Absent literal -> no match (no crash).
        assert ct.where('vendor == "Waymo"').nrows == 0

    def test_string_where_substring_in_dictionary_column(self):
        # '"needle" in dictcol' is a substring filter over the dictionary values.
        @dataclass
        class Row:
            company: str = blosc2.field(blosc2.dictionary())
            amount: float = blosc2.field(blosc2.float64())

        ct = CTable(Row)
        ct.extend([("Acme Inc", 7.0), ("Santamaria Cabs", 15.0), ("Beta", 5.0), ("Acme LLC", 9.0)])

        assert ct.where('"Acme" in company')["amount"][:].tolist() == [7.0, 9.0]
        assert ct.where('"acme" in company').nrows == 0  # case-sensitive
        assert ct.where('"zzz" in company').nrows == 0  # no match -> empty
        # Combines with other predicates and operators.
        assert ct.where('"Acme" in company and amount > 8')["amount"][:].tolist() == [9.0]
        assert ct.where('"Acme" in company or "Beta" in company').nrows == 3

    def test_string_where_dictionary_literal_with_special_chars(self):
        # Literals with commas/spaces/dashes (e.g. chicago-taxi company names).
        @dataclass
        class Row:
            company: str = blosc2.field(blosc2.dictionary())
            n: int = blosc2.field(blosc2.int64())

        name = "3721 - Santamaria Express, Alvaro Santamaria"
        ct = CTable(Row)
        ct.extend([(name, 1), ("Acme", 2), (name, 3)])

        assert ct.where(f'company == "{name}"')["n"][:].tolist() == [1, 3]
        assert ct.where(f"company == '{name}'")["n"][:].tolist() == [1, 3]  # single quotes too

    def test_dictionary_predicate_combines_with_regular_predicate_in_aggregate(self):
        ct = CTable(TripRow)
        ct.extend(DATA_TUPLES)
        assert ct["fare"].sum(where=(ct["fare"] > 6) & (ct["vendor"] == "Uber")) == pytest.approx(25.5)

    def test_isin(self):
        ct = CTable(TripRow)
        ct.extend(DATA_TUPLES)
        mask = ct["vendor"].isin(["Uber", "Via"])
        assert mask.tolist() == [True, False, True, True]

    def test_isin_absent_values(self):
        ct = CTable(TripRow)
        ct.extend(DATA_TUPLES)
        mask = ct["vendor"].isin(["Waymo"])
        assert all(not v for v in mask.tolist())

    def test_is_null(self):
        ct = CTable(TripRow)
        ct.extend(DATA_TUPLES)
        ct.append({"vendor": None, "fare": 0.0})
        assert _logical_mask_values(ct, ct["vendor"].is_null()) == [False, False, False, False, True]

    def test_null_count(self):
        ct = CTable(TripRow)
        ct.extend(DATA_TUPLES)
        ct.append({"vendor": None, "fare": 0.0})
        assert ct["vendor"].null_count() == 1

    def test_is_dictionary_property(self):
        ct = CTable(TripRow)
        ct.extend(DATA_TUPLES)
        assert ct["vendor"].is_dictionary is True
        assert ct["fare"].is_dictionary is False


# ---------------------------------------------------------------------------
# Persistence tests
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_b2d_roundtrip(self, tmp_path):
        p = str(tmp_path / "trips.b2d")
        ct = CTable(TripRow, urlpath=p, mode="w")
        ct.extend(DATA_TUPLES)
        ct.close()

        ct2 = CTable.open(p, mode="r")
        assert ct2.nrows == 4
        assert ct2["vendor"][:] == ["Uber", "Lyft", "Uber", "Via"]
        assert ct2._cols["vendor"].dictionary == ["Uber", "Lyft", "Via"]
        ct2.close()

    def test_b2z_roundtrip(self, tmp_path):
        p = str(tmp_path / "trips.b2z")
        ct = CTable(TripRow, urlpath=p, mode="w")
        ct.extend(DATA_TUPLES)
        ct.close()

        ct2 = CTable.open(p, mode="r")
        assert ct2.nrows == 4
        assert ct2["vendor"][:] == ["Uber", "Lyft", "Uber", "Via"]
        ct2.close()


# ---------------------------------------------------------------------------
# Arrow import / export tests
# ---------------------------------------------------------------------------


class TestArrowInterop:
    def _make_arrow_table(self, index_type=None, value_type=None, values=None, ordered=False):
        if index_type is None:
            index_type = pa.int32()
        if value_type is None:
            value_type = pa.string()
        if values is None:
            values = ["Uber", "Lyft", "Uber", None]
        return pa.table(
            {
                "vendor": pa.array(values, type=pa.dictionary(index_type, value_type, ordered=ordered)),
                "fare": pa.array([10.5, 7.2, 15.0, 0.0], type=pa.float64()),
            }
        )

    def test_import_dict_int32(self):
        at = self._make_arrow_table(index_type=pa.int32())
        ct = CTable.from_arrow(at.schema, at.to_batches())
        assert ct["vendor"][:] == ["Uber", "Lyft", "Uber", None]

    def test_import_dict_resizes_codes_to_arrow_capacity(self):
        values = ["Taxi"] * 4096 + ["Flash"] * 10
        at = pa.table({"vendor": pa.array(values, type=pa.dictionary(pa.int32(), pa.string()))})

        ct = CTable.from_arrow(at.schema, at.to_batches(), capacity_hint=len(values))

        assert len(ct._cols["vendor"].codes) >= len(values)
        assert ct["vendor"][-10:] == ["Flash"] * 10

    def test_import_dict_int8(self):
        at = self._make_arrow_table(index_type=pa.int8())
        ct = CTable.from_arrow(at.schema, at.to_batches())
        assert ct["vendor"][:] == ["Uber", "Lyft", "Uber", None]

    def test_import_dict_int16(self):
        at = self._make_arrow_table(index_type=pa.int16())
        ct = CTable.from_arrow(at.schema, at.to_batches())
        assert ct["vendor"][:] == ["Uber", "Lyft", "Uber", None]

    def test_import_dict_int64(self):
        at = self._make_arrow_table(index_type=pa.int64())
        ct = CTable.from_arrow(at.schema, at.to_batches())
        assert ct["vendor"][:] == ["Uber", "Lyft", "Uber", None]

    def test_import_dict_uint8(self):
        at = self._make_arrow_table(index_type=pa.uint8())
        ct = CTable.from_arrow(at.schema, at.to_batches())
        assert ct["vendor"][:] == ["Uber", "Lyft", "Uber", None]

    def test_import_dict_uint32(self):
        at = self._make_arrow_table(index_type=pa.uint32())
        ct = CTable.from_arrow(at.schema, at.to_batches())
        assert ct["vendor"][:] == ["Uber", "Lyft", "Uber", None]

    def test_import_nulls_preserved(self):
        at = self._make_arrow_table(values=["A", None, "B", None])
        ct = CTable.from_arrow(at.schema, at.to_batches())
        assert ct["vendor"][:] == ["A", None, "B", None]
        assert ct._cols["vendor"].codes[:4].tolist() == [0, -1, 1, -1]

    def test_export_produces_dict_type(self):
        at = self._make_arrow_table()
        ct = CTable.from_arrow(at.schema, at.to_batches())
        (batch,) = ct.iter_arrow_batches()
        field = batch.schema.field("vendor")
        assert pa.types.is_dictionary(field.type)
        assert field.type.index_type == pa.int32()
        assert field.type.value_type == pa.string()

    def test_export_values_match(self):
        at = self._make_arrow_table()
        ct = CTable.from_arrow(at.schema, at.to_batches())
        (batch,) = ct.iter_arrow_batches()
        assert batch.column("vendor").to_pylist() == ["Uber", "Lyft", "Uber", None]

    def test_parquet_roundtrip(self, tmp_path):
        path = tmp_path / "test.parquet"
        at = self._make_arrow_table(values=["Uber", "Lyft", "Uber", "Via"])
        pq.write_table(at, path)
        ct = CTable.from_parquet(path)
        assert isinstance(ct._schema.columns_by_name["vendor"].spec, DictionarySpec)
        assert ct["vendor"][:] == ["Uber", "Lyft", "Uber", "Via"]

        path2 = tmp_path / "roundtrip.parquet"
        ct.to_parquet(path2)
        at2 = pq.read_table(path2)
        assert pa.types.is_dictionary(at2.schema.field("vendor").type)
        assert at2.column("vendor").to_pylist() == ["Uber", "Lyft", "Uber", "Via"]

    def test_chunked_dict_unification(self):
        """Two batches with different chunk-local dictionaries → global unification."""
        batch1 = pa.record_batch(
            {"vendor": pa.array(["Uber", "Lyft"], type=pa.dictionary(pa.int32(), pa.string()))},
            schema=pa.schema([pa.field("vendor", pa.dictionary(pa.int32(), pa.string()))]),
        )
        batch2 = pa.record_batch(
            {"vendor": pa.array(["Via", "Uber"], type=pa.dictionary(pa.int32(), pa.string()))},
            schema=pa.schema([pa.field("vendor", pa.dictionary(pa.int32(), pa.string()))]),
        )
        schema = pa.schema([pa.field("vendor", pa.dictionary(pa.int32(), pa.string()))])
        ct = CTable.from_arrow(schema, [batch1, batch2])
        assert ct["vendor"][:] == ["Uber", "Lyft", "Via", "Uber"]
        codes = ct._cols["vendor"].codes[:4].tolist()
        # Uber should have the same code in both positions
        assert codes[0] == codes[3]

    def test_ordered_dict_inconsistent_order_raises(self):
        schema = pa.schema([pa.field("x", pa.dictionary(pa.int32(), pa.string(), ordered=True))])
        batch1 = pa.record_batch(
            {"x": pa.array(["A", "B"], type=pa.dictionary(pa.int32(), pa.string(), ordered=True))},
            schema=schema,
        )
        # Batch2 has different order for existing values
        batch2 = pa.record_batch(
            {"x": pa.array(["B", "A"], type=pa.dictionary(pa.int32(), pa.string(), ordered=True))},
            schema=schema,
        )
        with pytest.raises(ValueError, match="ordered"):
            CTable.from_arrow(schema, [batch1, batch2])

    def test_unsupported_dict_value_type_raises(self):
        schema = pa.schema([pa.field("x", pa.dictionary(pa.int32(), pa.int64()))])
        at = pa.table({"x": pa.array([1, 2], type=pa.dictionary(pa.int32(), pa.int64()))})
        with pytest.raises(TypeError, match="dictionary"):
            CTable.from_arrow(schema, at.to_batches())


# ---------------------------------------------------------------------------
# Index tests
# ---------------------------------------------------------------------------


class TestIndex:
    def test_create_index(self):
        ct = CTable(TripRow)
        ct.extend(DATA_TUPLES)
        idx = ct.create_index("vendor")
        assert idx is not None

    def test_index_metadata_is_logical(self):
        ct = CTable(TripRow)
        ct.extend(DATA_TUPLES)
        ct.create_index("vendor")
        catalog = ct._storage.load_index_catalog()
        assert "vendor" in catalog

    def test_equality_uses_codes(self):
        ct = CTable(TripRow)
        ct.extend(DATA_TUPLES)
        mask = ct["vendor"] == "Uber"
        assert _logical_mask_values(ct, mask) == [True, False, True, False]

    def test_isin_uses_codes(self):
        ct = CTable(TripRow)
        ct.extend(DATA_TUPLES)
        mask = ct["vendor"].isin(["Lyft", "Via"])
        assert mask.tolist() == [False, True, False, True]

    def test_append_after_index(self, tmp_path):
        p = str(tmp_path / "indexed.b2d")
        ct = CTable(TripRow, urlpath=p, mode="w")
        ct.extend(DATA_TUPLES)
        ct.create_index("vendor")
        ct.append({"vendor": "Uber", "fare": 20.0})
        assert ct.nrows == 5
        mask = ct["vendor"] == "Uber"
        assert mask.sum() == 3
        ct.close()


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


def test_cli_preserves_dict_by_default(tmp_path):
    from blosc2.cli.parquet_to_blosc2 import main

    path = tmp_path / "dict.parquet"
    out = tmp_path / "dict.b2d"
    at = pa.table(
        {"vendor": pa.array(["Uber", "Lyft", "Uber", "Via"], type=pa.dictionary(pa.int32(), pa.string()))}
    )
    pq.write_table(at, path)

    assert main([str(path), str(out)]) == 0

    ct = CTable.open(str(out), mode="r")
    assert isinstance(ct._schema.columns_by_name["vendor"].spec, DictionarySpec)
    assert ct["vendor"][:] == ["Uber", "Lyft", "Uber", "Via"]
    ct.close()


def test_cli_decode_dictionaries_flag(tmp_path):
    from blosc2.cli.parquet_to_blosc2 import main
    from blosc2.schema import VLStringSpec

    path = tmp_path / "dict.parquet"
    out = tmp_path / "dict_decoded.b2d"
    at = pa.table(
        {"vendor": pa.array(["Uber", "Lyft", "Uber"], type=pa.dictionary(pa.int32(), pa.string()))}
    )
    pq.write_table(at, path)

    assert main(["--decode-dictionaries", str(path), str(out)]) == 0

    ct = CTable.open(str(out), mode="r")
    assert isinstance(ct._schema.columns_by_name["vendor"].spec, VLStringSpec)
    assert ct["vendor"][:] == ["Uber", "Lyft", "Uber"]
    ct.close()


def test_cli_dict_export_roundtrip(tmp_path):
    from blosc2.cli.parquet_to_blosc2 import main

    path = tmp_path / "dict.parquet"
    out = tmp_path / "dict.b2d"
    exported = tmp_path / "dict_exported.parquet"

    at = pa.table(
        {
            "vendor": pa.array(["Uber", "Lyft", None, "Via"], type=pa.dictionary(pa.int32(), pa.string())),
            "score": pa.array([1, 2, 3, 4], type=pa.int32()),
        }
    )
    pq.write_table(at, path)

    assert main([str(path), str(out)]) == 0
    assert main(["--export", str(out), str(exported)]) == 0

    rt = pq.read_table(exported)
    assert rt.column("vendor").to_pylist() == ["Uber", "Lyft", None, "Via"]
    assert rt.column("score").to_pylist() == [1, 2, 3, 4]


if __name__ == "__main__":
    pytest.main(["-v", __file__])
