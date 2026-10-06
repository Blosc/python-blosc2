"""Typed pandas imports using Arrow string buffers and their fallback path."""

import builtins
from dataclasses import dataclass

import numpy as np
import pytest

import blosc2
from blosc2 import CTable
from blosc2._utf8_array import UTF8Array

pd = pytest.importorskip("pandas")
pa = pytest.importorskip("pyarrow")


def make_frame(arrow_backed, missing=False):
    df = pd.DataFrame(
        {
            "name": pd.Series(["café", "", "日本語", "a\0b"] * 4096, dtype="string[pyarrow]"),
            "count": np.arange(16_384, dtype=np.int64),
            "flag": np.arange(16_384) % 2 == 0,
            "value": np.arange(16_384, dtype=np.float64),
        }
    )
    if missing:
        df["count"] = df["count"].astype("Int64")
        df["flag"] = df["flag"].astype("boolean")
        df.loc[3, ["name", "count", "flag"]] = pd.NA
        df.loc[4, "value"] = np.nan
    if arrow_backed:
        df = df.convert_dtypes(dtype_backend="pyarrow")
    # Index labels are not imported by from_pandas.
    df.index = np.arange(len(df)) + 100
    return df


def row_schema(null_storage="mask", nullable=False):
    options = {"nullable": True, "null_storage": null_storage} if nullable else {}

    @dataclass
    class Row:
        name: str = blosc2.field(blosc2.utf8(**options))
        count: int = blosc2.field(blosc2.int32(ge=0, **options))
        flag: bool = blosc2.field(blosc2.bool(**options))
        value: float = blosc2.field(blosc2.float64(**options))

    return Row


@pytest.mark.parametrize("arrow_backed", [False, True])
@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("null_storage", ["mask", "sentinel"])
def test_typed_pandas_arrow_matches_general_path(arrow_backed, missing, null_storage, monkeypatch):
    df = make_frame(arrow_backed, missing)
    Row = row_schema(null_storage, nullable=missing)
    with monkeypatch.context() as patch:
        patch.setattr(CTable, "_import_pandas_arrow_strings", lambda *_: False)
        reference = CTable.from_pandas(df, Row)
    with reference:
        with monkeypatch.context() as patch:
            patch.setattr(UTF8Array, "extend", lambda *_: pytest.fail("Materialized Python strings"))
            table = CTable.from_pandas(df, Row)
        with table:
            assert table._row_type is Row
            assert [col.spec.to_metadata_dict() for col in table._schema.columns] == [
                col.spec.to_metadata_dict() for col in reference._schema.columns
            ]
            actual, expected = table.to_arrow(), reference.to_arrow()
            assert actual.select(["name", "count", "flag"]).equals(
                expected.select(["name", "count", "flag"]), check_metadata=False
            )
            np.testing.assert_array_equal(
                actual.column("value").to_numpy(), expected.column("value").to_numpy()
            )
            assert actual.column("value").is_valid().equals(expected.column("value").is_valid())
            assert table._cols["count"].dtype == np.dtype("int32")
            assert table._cols["count"].chunks == reference._cols["count"].chunks
            assert table["name"].null_count() == reference["name"].null_count()
            assert table["value"].null_count() == reference["value"].null_count()
            table.append(Row("tail", 42, True, 1.5))
            assert len(table) == len(df) + 1
            assert table.to_arrow().column("name")[-1].as_py() == "tail"


@pytest.mark.parametrize("invalid", ["constraint", "null"])
def test_typed_pandas_arrow_preserves_validation(invalid):
    df = make_frame(False)
    if invalid == "constraint":
        df.iloc[0, df.columns.get_loc("count")] = -1
        with pytest.raises(ValueError, match="ge=0"):
            CTable.from_pandas(df, row_schema())
    else:
        df.iloc[0, df.columns.get_loc("name")] = pd.NA
        with pytest.raises(TypeError, match="not nullable"):
            CTable.from_pandas(df, row_schema())


@pytest.mark.parametrize("fallback", ["small", "object", "no_pyarrow", "complex"])
def test_typed_pandas_arrow_fallback(fallback, monkeypatch):
    df = make_frame(False)
    Row = row_schema()
    if fallback == "small":
        df = df.iloc[:10]
    elif fallback == "object":
        df["name"] = df["name"].astype(object)
    elif fallback == "complex":

        @dataclass
        class ComplexRow(Row):
            tags: list[int] = blosc2.field(blosc2.list(blosc2.int64()))  # noqa: RUF009

        Row = ComplexRow
        df["tags"] = [[1, 2]] * len(df)
    else:
        original_import = builtins.__import__

        def without_pyarrow(name, *args, **kwargs):
            if name == "pyarrow":
                raise ImportError("PyArrow unavailable")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", without_pyarrow)
    monkeypatch.setattr(UTF8Array, "extend_arrow", lambda *_: pytest.fail("Arrow fast path used"))
    with CTable.from_pandas(df, Row) as table:
        assert len(table) == len(df)
        assert table["name"][:].tolist() == df["name"].tolist()
