#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""Tests for compile_schema(), schema_to_dict(), and schema_from_dict()."""

from dataclasses import MISSING, dataclass

import numpy as np
import pytest

import blosc2
from blosc2.schema import bool as b2_bool
from blosc2.schema import complex128, float64, int64, string
from blosc2.schema_compiler import (
    CompiledSchema,
    compile_schema,
    schema_from_dict,
    schema_to_dict,
)

# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------


@dataclass
class Simple:
    id: int = blosc2.field(blosc2.int64(ge=0))
    score: float = blosc2.field(blosc2.float64(ge=0, le=100), default=0.0)
    active: bool = blosc2.field(blosc2.bool(), default=True)


@dataclass
class WithString:
    name: str = blosc2.field(blosc2.string(max_length=16))
    value: float = blosc2.field(blosc2.float64(), default=0.0)


@dataclass
class WithComplex:
    id: int = blosc2.field(blosc2.int64())
    c_val: complex = blosc2.field(blosc2.complex128(), default=0j)


# -------------------------------------------------------------------
# compile_schema — explicit b2.field()
# -------------------------------------------------------------------


def test_compile_returns_compiled_schema():
    s = compile_schema(Simple)
    assert isinstance(s, CompiledSchema)
    assert s.row_cls is Simple


def test_compile_column_count():
    s = compile_schema(Simple)
    assert len(s.columns) == 3


def test_compile_column_names_order():
    s = compile_schema(Simple)
    assert [c.name for c in s.columns] == ["id", "score", "active"]


def test_compile_column_dtypes():
    s = compile_schema(Simple)
    assert s.columns_by_name["id"].dtype == np.dtype(np.int64)
    assert s.columns_by_name["score"].dtype == np.dtype(np.float64)
    assert s.columns_by_name["active"].dtype == np.dtype(np.bool_)


def test_compile_column_specs():
    s = compile_schema(Simple)
    assert isinstance(s.columns_by_name["id"].spec, int64)
    assert s.columns_by_name["id"].spec.ge == 0
    assert isinstance(s.columns_by_name["score"].spec, float64)
    assert s.columns_by_name["score"].spec.le == 100


def test_compile_defaults():
    s = compile_schema(Simple)
    assert s.columns_by_name["id"].default is MISSING  # required
    assert s.columns_by_name["score"].default == 0.0
    assert s.columns_by_name["active"].default is True


def test_compile_py_types():
    s = compile_schema(Simple)
    assert s.columns_by_name["id"].py_type is int
    assert s.columns_by_name["score"].py_type is float
    assert s.columns_by_name["active"].py_type is bool


def test_compile_string_column():
    s = compile_schema(WithString)
    col = s.columns_by_name["name"]
    assert isinstance(col.spec, string)
    assert col.spec.max_length == 16
    assert col.dtype == np.dtype("U16")


def test_compile_complex_column():
    s = compile_schema(WithComplex)
    col = s.columns_by_name["c_val"]
    assert isinstance(col.spec, complex128)
    assert col.dtype == np.dtype(np.complex128)
    assert col.default == 0j


# -------------------------------------------------------------------
# compile_schema — inferred shorthand (plain annotations)
# -------------------------------------------------------------------


@dataclass
class Inferred:
    count: int
    ratio: float
    flag: bool


def test_inferred_shorthand():
    s = compile_schema(Inferred)
    assert len(s.columns) == 3
    assert isinstance(s.columns_by_name["count"].spec, int64)
    assert isinstance(s.columns_by_name["ratio"].spec, float64)
    assert isinstance(s.columns_by_name["flag"].spec, b2_bool)


def test_inferred_no_constraints():
    s = compile_schema(Inferred)
    for col in s.columns:
        assert col.spec.to_pydantic_kwargs() == {}


# -------------------------------------------------------------------
# compile_schema — annotation / spec mismatch rejection
# -------------------------------------------------------------------


def test_annotation_spec_mismatch():
    @dataclass
    class Bad:
        x: str = blosc2.field(blosc2.int64())  # str annotation but int64 spec

    with pytest.raises(TypeError, match="incompatible"):
        compile_schema(Bad)


def test_non_dataclass_rejected():
    class NotADataclass:
        pass

    with pytest.raises(TypeError, match="dataclass"):
        compile_schema(NotADataclass)


# -------------------------------------------------------------------
# compile_schema — per-column cparams config
# -------------------------------------------------------------------


def test_column_cparams_stored():
    @dataclass
    class WithCparams:
        id: int = blosc2.field(blosc2.int64(), cparams={"clevel": 9})
        score: float = blosc2.field(blosc2.float64(), default=0.0)

    s = compile_schema(WithCparams)
    assert s.columns_by_name["id"].config.cparams == {"clevel": 9}
    assert s.columns_by_name["score"].config.cparams is None


# -------------------------------------------------------------------
# schema_to_dict / schema_from_dict  (Step 12)
# -------------------------------------------------------------------


def test_schema_to_dict_structure():
    d = schema_to_dict(compile_schema(Simple))
    assert d["version"] == 1
    assert d["row_cls"] == "Simple"
    assert len(d["columns"]) == 3


def test_schema_to_dict_column_fields():
    d = schema_to_dict(compile_schema(Simple))
    id_col = next(c for c in d["columns"] if c["name"] == "id")
    assert id_col["kind"] == "int64"
    assert id_col["ge"] == 0
    assert id_col["default"] is None  # MISSING → None


def test_schema_to_dict_default_values():
    d = schema_to_dict(compile_schema(Simple))
    score_col = next(c for c in d["columns"] if c["name"] == "score")
    assert score_col["default"] == 0.0

    active_col = next(c for c in d["columns"] if c["name"] == "active")
    assert active_col["default"] is True


def test_schema_to_dict_complex_default():
    d = schema_to_dict(compile_schema(WithComplex))
    c_col = next(c for c in d["columns"] if c["name"] == "c_val")
    assert c_col["default"]["__complex__"] is True
    assert c_col["default"]["real"] == 0.0
    assert c_col["default"]["imag"] == 0.0


def test_schema_roundtrip():
    """schema_from_dict(schema_to_dict(s)) reproduces the same column structure."""
    original = compile_schema(Simple)
    d = schema_to_dict(original)
    restored = schema_from_dict(d)

    assert len(restored.columns) == len(original.columns)
    for orig_col, rest_col in zip(original.columns, restored.columns, strict=True):
        assert orig_col.name == rest_col.name
        assert orig_col.dtype == rest_col.dtype
        assert type(orig_col.spec) is type(rest_col.spec)
        if orig_col.default is MISSING:
            assert rest_col.default is MISSING
        else:
            assert orig_col.default == rest_col.default


def test_schema_from_dict_no_row_cls():
    """Reconstructed schema has row_cls=None (original class not available)."""
    d = schema_to_dict(compile_schema(Simple))
    restored = schema_from_dict(d)
    assert restored.row_cls is None


def test_schema_from_dict_preserves_constraints():
    d = schema_to_dict(compile_schema(Simple))
    restored = schema_from_dict(d)
    id_col = restored.columns_by_name["id"]
    assert id_col.spec.ge == 0
    score_col = restored.columns_by_name["score"]
    assert score_col.spec.le == 100


def test_schema_from_dict_unknown_kind():
    d = {"version": 1, "row_cls": "X", "columns": [{"name": "x", "kind": "unknown", "default": None}]}
    with pytest.raises(ValueError, match="Unknown column kind"):
        schema_from_dict(d)


def test_schema_from_dict_unsupported_version():
    d = {"version": 99, "row_cls": "X", "columns": []}
    with pytest.raises(ValueError, match="Unsupported schema version"):
        schema_from_dict(d)
